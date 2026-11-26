# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed request isolation and page-table lowering on a single DP owner."""

import ctypes
from types import SimpleNamespace

import pytest
import torch


def test_packed_page_boundary_and_recycled_handle():
    from models.deepseek_v4_1_flash_a3 import config as c
    from models.deepseek_v4_1_flash_a3.host import prepare_prefill_inputs
    from models.deepseek_v4_1_flash_a3.packed import PackedSession
    from models.deepseek_v4_1_flash_a3.utils import DECODE_CACHE_NAMES

    slots, context, cap = 2, 256, 128
    template = prepare_prefill_inputs(torch.empty(0, c.FLASH.hidden_size), cap, max_seq_len=context,
                                      request_slots=slots)
    cache = {name: template[name] for name in DECODE_CACHE_NAMES.values()}
    writes = []
    reset_images = {"reset_" + name: value[0, 0].clone() for name, value in cache.items()}
    graphs = SimpleNamespace(
        host_io={"idle_inputs": (template, {}), **reset_images}, host_cache=[cache] * 16,
        ranks=[[{k: SimpleNamespace(data_ptr=v.data_ptr()) for k, v in cache.items()} for _ in range(16)]],
        runtime=SimpleNamespace(copy_to=lambda dst, src, size, **kw: writes.append((dst, src, size, kw))),
        weights=SimpleNamespace(lookup_rows=lambda ids: torch.zeros(len(ids), c.FLASH.hidden_size)),
        engram=SimpleNamespace(lookup=lambda ids, n, offset, **kw: torch.zeros(2, n, 6144), history={}),
    )
    session = PackedSession(graphs, slots, context, (cap, 16))
    a = session.admit("a", 0, 129)
    b = session.admit("b", 0, 2)
    old, live = session.history["a"], session.history["b"]
    session._inputs(0, [(old, [11] * cap)], 0, cap)
    old.committed = cap
    values, samples, count = session._inputs(0, [(old, [33]), (live, [22, 23])], 0, cap)
    assert count == 3 and samples == [(b, 1), (a, 2)]
    assert values["counts"].tolist() == [3, 3]
    assert values["query_start_loc"][0].tolist() == [0, 2, 3]
    assert values["state_block_table"].flatten().tolist() == [1, 0]
    assert values["window_slots"][0, :3].tolist() == [live.pages[0] * 128, live.pages[0] * 128 + 1, old.pages[1] * 128]
    assert set(old.pages).isdisjoint(live.pages)
    live_pages = live.pages.copy()
    session.release(a)
    expected = {
        (rank, value.data_ptr(), region.data_ptr() - value.data_ptr(), region.nbytes)
        for rank in range(c.TP_SIZE) for name, value in cache.items()
        for source in range(value.shape[0]) for page in ([a.slot] if name == "state_cache" else old.pages)
        for region in (value[source, page],)
    }
    assert len(writes) == len(expected)
    assert {(kw["worker_id"], dst, kw["dst_offset"], n) for dst, _, n, kw in writes} == expected
    for dst, src, size, kw in writes:
        assert ctypes.string_at(src, size) == ctypes.string_at(dst + kw["dst_offset"], size)
    assert old.pages == [0, 1]
    assert live_pages == [session.pages_per_slot]
    new = session.admit("a", 0, 1)
    with pytest.raises(ValueError, match="Expired"):
        session.advance_batch("decode", [(a, [1])])
    assert new is not a and live.pages == live_pages
    fresh = session.history["a"]
    session._inputs(0, [(fresh, [44])], 0, cap)
    assert fresh.pages == [old.pages[0]]

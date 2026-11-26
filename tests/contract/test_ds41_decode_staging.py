# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check that single-token staging preserves the resident Decode ABI."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from models.deepseek_v4_1_flash_a3.host import prepare_prefill_inputs, prepare_resident_inputs
from models.deepseek_v4_1_flash_a3.config import FLASH
from models.deepseek_v4_1_flash_a3.device_views import _device_view
from models.deepseek_v4_1_flash_a3.engram import ENGRAM_K
from models.deepseek_v4_1_flash_a3.utils import DECODE_CACHE_NAMES, prepare_decode_inputs
from models.deepseek_v4_1_flash_a3.replay import DecodeInputs


@pytest.mark.parametrize("capacity", [16, 80])
def test_single_token_staging_matches_all_three_tp_images(capacity):
    def embeddings(ids):
        return ids[:, None].expand(-1, FLASH.hidden_size).to(torch.bfloat16)

    def lookup(ids, rows, position, *, request_id):
        result = torch.zeros(2, rows, ENGRAM_K, dtype=torch.bfloat16)
        result[:, 0].fill_(int(ids[0]) + position + request_id)
        return result

    weights = SimpleNamespace(lookup_rows=embeddings)
    engram = SimpleNamespace(lookup=lookup)
    empty = prepare_prefill_inputs(embeddings(torch.empty(0)), capacity, max_seq_len=130)
    cache = {name: empty[name] for name in DECODE_CACHE_NAMES.values()}
    rope = empty["freqs_cos"], empty["freqs_sin"]
    idle = {key: value for key, value in prepare_decode_inputs(empty).items()
            if key not in ("freqs_cos", "freqs_sin")}
    window_pages = empty["window_cache"].shape[1] - 1
    full = DecodeInputs(idle, window_pages_per_slot=window_pages)
    single = DecodeInputs(idle, window_pages_per_slot=window_pages)
    for position, owner in ((0, 0), (127, 12), (128, 12), (129, 4)):
        token = position + 3
        active = prepare_resident_inputs(
            weights, [token], capacity, start_pos=position, max_seq_len=130,
            cache_state=cache, rope_tables=rope, engram=engram, request_id=owner,
        )
        lowered = prepare_decode_inputs(active)
        full.stage({False: idle, True: {name: lowered[name] for name in idle}}, "logit_row_indices")
        single.stage_token(weights, engram, token, position, owner)
        assert torch.equal(single.images, full.images)


def test_single_token_staging_wraps_only_the_sliding_cache():
    empty = prepare_prefill_inputs(torch.empty(0, FLASH.hidden_size), 16, max_seq_len=128)
    idle = prepare_decode_inputs(empty)
    idle["c1a_index_block_table"] = torch.arange(257, dtype=torch.int32).view(1, -1)
    idle["c2a_index_block_table"] = torch.arange(129, dtype=torch.int32).view(1, -1)
    staging = DecodeInputs(idle, window_pages_per_slot=5)
    weights = SimpleNamespace(lookup_rows=lambda ids: torch.zeros(1, FLASH.hidden_size))
    engram = SimpleNamespace(lookup=lambda *args, **kwargs: torch.zeros(2, 1, ENGRAM_K))
    for position in (639, 640, 32769):
        staging.stage_token(weights, engram, 42, position, 0)
        for fields in staging.fields[1:]:
            assert fields["window_slots"][0].item() == position % 640
            expected = torch.arange(position - FLASH.sliding_window + 1, position + 1) % 640
            assert torch.equal(fields["window_indices"][0], expected)
            assert fields["c1a_compressed_slots"][0].item() == position


def test_typed_view_retains_its_allocation_and_rejects_overruns():
    from pypto.runtime import DeviceTensor

    storage = torch.arange(32, dtype=torch.float32)
    buffer = SimpleNamespace(base=storage.data_ptr(), nbytes=storage.nbytes, tensor=storage.view)
    allocation = DeviceTensor(storage.data_ptr() + 16, (8,), storage.dtype, buffer=buffer, byte_offset=16)
    nested = _device_view(allocation, storage[6:8], base_offset=16)
    assert nested.buffer is buffer
    assert nested.data_ptr == allocation.data_ptr + 8
    assert nested.byte_offset == 24
    with pytest.raises(ValueError, match="exceeds"):
        _device_view(allocation, storage[11:14], base_offset=16)
    with pytest.raises(ValueError, match="before"):
        _device_view(allocation, storage[:1], base_offset=16)


def test_fixed_trace_drives_decode_after_a_different_prefill_argmax(tmp_path, monkeypatch):
    from models.deepseek_v4_1_flash_a3 import replay

    (tmp_path / "tokenizer_config.json").write_text('{"eos_token": "eos"}')
    weights = SimpleNamespace(tensor_layouts={}, checkpoint=SimpleNamespace(root=tmp_path))
    tokenizer = Mock()
    tokenizer.token_to_id.return_value = 1
    tokenizer.decode.return_value = ""
    request = dict(dp_rank=0, request_id="trace", input_ids=[4, 5, 6], trace_token_ids=[2, 3, 4])
    options = SimpleNamespace(capacity=16, decode_capacity=16, context_capacity=128,
                              chunk_size=2, max_new_tokens=3, ignore_eos=False)
    counts = {name: torch.zeros(16) for name in ("moe_num_tokens", "attention_num_tokens")}
    ranks = {rank: {"logits": Mock(), "sampled_ids": Mock()} for rank in range(16)}
    # Both phases return token 1 for every call, independently of the forced history.
    host_io = dict(logits=torch.tensor([[0., 1., 0., 0., 0.]] * 3), sampled_ids=torch.tensor([1]))
    monkeypatch.setattr(replay, "prepare_resident_inputs", lambda *args, **kwargs: {})
    monkeypatch.setattr(replay, "route_distribution", lambda *args, **kwargs: {})
    runtime, engram, decode_inputs = Mock(), Mock(), Mock()
    report = replay.run_request(
        runtime, ("prefill", "decode"), ({}, counts), ([], []), (ranks, ranks),
        weights, engram, tokenizer, request, [{} for _ in range(16)], None,
        None, SimpleNamespace(), {}, options=options, host_io=host_io, decode_inputs=decode_inputs,
    )
    assert [(call.args[2], call.args[3]) for call in decode_inputs.stage_token.call_args_list] == [(2, 3), (3, 4)]
    assert report["prefill_calls"] == 2 and report["decode_calls"] == 2
    assert report["decode_input_ids"] == [2, 3]
    assert report["output_sequences"][0]["token_ids"] == [1, 1, 1]

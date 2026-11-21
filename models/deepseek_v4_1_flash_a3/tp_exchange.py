# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tensor-parallel copies and BF16 attention reduction inside a resident Decode graph."""

from math import gcd

import pypto.language as pl
import pypto.language.distributed as pld


def make_tp_rank(mode, count):
    """Build the rank program for an aligned TP4 packet."""
    if mode not in ("all_gather", "all_to_all", "all_reduce"):
        raise ValueError("Unsupported Decode TP exchange")
    tp = 4
    width = count // tp if mode == "all_to_all" else count
    output_width = count * tp if mode == "all_gather" else count
    if width % 16:
        raise ValueError("Decode TP packets must be BF16 transfer aligned")
    tile = gcd(1024, width)
    tiles = (width + tile - 1) // tile
    workers = min(tiles, 16)
    is_reduce = mode == "all_reduce"
    is_exchange = mode == "all_to_all"

    @pl.jit(auto_scope=False)
    def tp_rank(
        source: pl.Tensor[[1, count], pl.BF16],
        output: pl.Out[pl.Tensor[[1, output_width], pl.BF16]],
        incoming: pld.DistributedTensor[[tp, width], pl.BF16],
        arrived: pld.DistributedTensor[[tp, 16], pl.INT32],
        consumed: pld.DistributedTensor[[tp, 16], pl.INT32],
        rank: pl.Scalar[pl.INT32], sequence: pl.Scalar[pl.INT32],
    ):
        lane = rank % tp
        base = rank - lane
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_wait_for_space") as available:
            for peer in pl.range(tp):
                pld.system.defer_wait(consumed, offsets=[peer, 0], expected=sequence - 1, cmp=pld.WaitCmp.Ge)
        with pl.spmd(tp * workers, name_hint="tp_send", deps=[available]) as written:
            block = pl.tile.get_block_idx()
            peer = block // workers
            worker = block % workers
            for index in pl.range(worker, tiles, workers):
                start = index * tile
                if is_exchange:
                    pld.tensor.put(dst=incoming, peer=base + peer, src=source, dst_offsets=[lane, start],
                                   src_offsets=[0, peer * width + start], shape=[1, tile])
                else:
                    pld.tensor.put(dst=incoming, peer=base + peer, src=source, dst_offsets=[lane, start],
                                   src_offsets=[0, start], shape=[1, tile])
        with pl.spmd(tp, name_hint="tp_publish", deps=[written]) as sent:
            peer = pl.tile.get_block_idx()
            pld.system.notify(arrived, peer=base + peer, offsets=[lane, 0], value=sequence, op=pld.NotifyOp.Set)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_wait_data", deps=[sent]) as ready:
            for peer in pl.range(tp):
                pld.system.defer_wait(arrived, offsets=[peer, 0], expected=sequence, cmp=pld.WaitCmp.Ge)
        with pl.spmd(workers, name_hint="tp_receive", deps=[ready]) as received:
            worker = pl.tile.get_block_idx()
            for index in pl.range(worker, tiles, workers):
                start = index * tile
                if is_reduce:
                    a = pl.load(incoming, [0, start], [1, tile])
                    b = pl.load(incoming, [1, start], [1, tile])
                    c = pl.load(incoming, [2, start], [1, tile])
                    d = pl.load(incoming, [3, start], [1, tile])
                    a32 = pl.cast(a, pl.FP32)
                    b32 = pl.cast(b, pl.FP32)
                    c32 = pl.cast(c, pl.FP32)
                    d32 = pl.cast(d, pl.FP32)
                    # The resident HCCL TP4 tree rounds the two pairs before the final sum.
                    ab = pl.add(a32, b32)
                    cd = pl.add(c32, d32)
                    ab16 = pl.cast(ab, pl.BF16, mode="rint")
                    cd16 = pl.cast(cd, pl.BF16, mode="rint")
                    left = pl.cast(ab16, pl.FP32)
                    right = pl.cast(cd16, pl.FP32)
                    total = pl.add(left, right)
                    result = pl.cast(total, pl.BF16, mode="rint")
                    pl.store(result, [0, start], output)
                else:
                    for peer in pl.range(tp):
                        row = pl.load(incoming, [peer, start], [1, tile])
                        pl.store(row, [0, peer * width + start], output)
        with pl.spmd(tp, name_hint="tp_release", deps=[received]):
            peer = pl.tile.get_block_idx()
            pld.system.notify(consumed, peer=base + peer, offsets=[lane, 0], value=sequence, op=pld.NotifyOp.Set)
        return output

    return tp_rank


def make_tp_exchange(mode, count):
    tp, ep = 4, 16
    width = count // tp if mode == "all_to_all" else count
    output_width = count * tp if mode == "all_gather" else count
    tp_rank = make_tp_rank(mode, count)

    @pl.jit.host
    def exchange(
        source: pl.Tensor[[ep, 1, count], pl.BF16],
        output: pl.Out[pl.Tensor[[ep, 1, output_width], pl.BF16]],
        sequence: pl.Scalar[pl.INT32],
    ):
        inbox_buf = pld.alloc_window_buffer([tp, width], dtype=pl.BF16)
        ready_buf = pld.alloc_window_buffer([tp, 16], dtype=pl.INT32)
        done_buf = pld.alloc_window_buffer([tp, 16], dtype=pl.INT32)
        for rank in pl.range(ep):
            inbox = pld.window(inbox_buf, [tp, width], dtype=pl.BF16)
            ready = pld.window(ready_buf, [tp, 16], dtype=pl.INT32)
            done = pld.window(done_buf, [tp, 16], dtype=pl.INT32)
            tp_rank(source[rank], output[rank], inbox, ready, done, rank, sequence, device=rank)

    return exchange

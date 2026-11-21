# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One-token expert dispatch over the existing blocked INT8 expert layer views."""
import pypto.language as pl
import pypto.language.distributed as pld
from .config import FLASH
from .grouped_moe import make_grouped_routed_experts

D, I = FLASH.hidden_size, FLASH.moe_intermediate_size
EP, TOPK, LOCAL, ROWS = 16, 6, 24, 16
GATE_ROWS, DOWN_ROWS = LOCAL * D * (2 * I // 64), LOCAL * I * (D // 64)
_grouped = pl.jit.inline(make_grouped_routed_experts(ROWS, vars(FLASH), LOCAL)._func)


def make_routed_rank(*, streaming=False):
    """Reserve the mailbox until the owner has consumed all returned expert rows."""
    done_rows = EP + int(streaming)

    @pl.jit(auto_scope=False)
    def routed_rank(
        token: pl.Tensor[[1, D], pl.BF16],
        ids: pl.Tensor[[1, TOPK], pl.INT32],
        gate_up: pl.Tensor[[GATE_ROWS, 64], pl.INT8],
        gate_scale: pl.Tensor[[LOCAL * 2 * I], pl.FP32],
        down: pl.Tensor[[DOWN_ROWS, 64], pl.INT8],
        down_scale: pl.Tensor[[LOCAL * D], pl.FP32],
        weights: pl.Tensor[[1, TOPK], pl.FP32],
        shared: pl.Tensor[[1, D], pl.BF16],
        combined: pl.Out[pl.Tensor[[1, D], pl.BF16]],
        token_window: pld.DistributedTensor[[1, D], pl.BF16],
        id_window: pld.DistributedTensor[[1, 8], pl.INT32],
        slot_window: pld.DistributedTensor[[TOPK, D], pl.BF16],
        ready: pld.DistributedTensor[[1, 16], pl.INT32],
        done: pld.DistributedTensor[[done_rows, 16], pl.INT32],
        owner: pl.Scalar[pl.INT32], rank: pl.Scalar[pl.INT32], sequence: pl.Scalar[pl.INT32],
    ):
        padded_ids = pl.create_tensor([1, 8], dtype=pl.INT32)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_pad_route_ids"):
            if rank == owner:
                for slot in pl.range(TOPK):
                    expert_id = pl.read(ids, [0, slot])
                    pl.write(padded_ids, [0, slot], expert_id)
                pl.write(padded_ids, [0, 6], pl.cast(-1, pl.INT32))
                pl.write(padded_ids, [0, 7], pl.cast(-1, pl.INT32))
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_expert_wait_credit") as credit:
            if streaming:
                if rank == owner:
                    pld.system.defer_wait(done, offsets=[EP, 0], expected=sequence - 1, cmp=pld.WaitCmp.Ge)
        with pl.spmd(EP, name_hint="decode_expert_broadcast", deps=[credit]) as published:
            peer = pl.tile.get_block_idx()
            if rank == owner:
                pld.tensor.put(dst=token_window, peer=peer, src=token, dst_offsets=[0, 0], src_offsets=[0, 0], shape=[1, D])
                pld.tensor.put(dst=id_window, peer=peer, src=padded_ids, dst_offsets=[0, 0], src_offsets=[0, 0], shape=[1, 8])
                pld.system.notify(ready, peer=peer, offsets=[0, 0], value=sequence, op=pld.NotifyOp.Set)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_expert_wait_input", deps=[published]) as arrived:
            pld.system.defer_wait(ready, offsets=[0, 0], expected=sequence, cmp=pld.WaitCmp.Ge)
        expert_x = pl.create_tensor([LOCAL * ROWS, D], dtype=pl.BF16)
        counts = pl.create_tensor([LOCAL], dtype=pl.INT64)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_expert_select", deps=[arrived]):
            for local in pl.range(LOCAL):
                count = pl.cast(0, pl.INT64)
                for slot in pl.range(TOPK):
                    if pl.read(id_window, [0, slot]) == rank * LOCAL + local:
                        count = pl.cast(1, pl.INT64)
                        row = pl.load(token_window, [0, 0], [1, D])
                        pl.store(row, [local * ROWS, 0], expert_x)
                pl.write(counts, [local], count)
        expert_y = pl.create_tensor([LOCAL * ROWS, D], dtype=pl.BF16)
        expert_y = _grouped(expert_x, gate_up, gate_scale, down, down_scale, counts, expert_y)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_expert_return") as sent:
            for slot in pl.range(TOPK):
                local = pl.read(id_window, [0, slot]) - rank * LOCAL
                if local >= 0 and local < LOCAL:
                    pld.tensor.put(dst=slot_window, peer=owner, src=expert_y, dst_offsets=[slot, 0], src_offsets=[local * ROWS, 0], shape=[1, D])
            pld.system.notify(done, peer=owner, offsets=[rank, 0], value=sequence, op=pld.NotifyOp.Set)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_expert_wait_output", deps=[sent]) as all_done:
            if rank == owner:
                for peer in pl.range(EP):
                    pld.system.defer_wait(done, offsets=[peer, 0], expected=sequence, cmp=pld.WaitCmp.Ge)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_expert_combine", deps=[all_done]):
            if rank == owner:
                source = pl.tile.load(weights, [0, 0], [1, 16], valid_shape=[1, TOPK])
                padded = pl.tile.fillpad(source, pad_value=pl.PadValue.zero)
                padded = pl.tile.set_validshape(padded, 1, 16)
                narrowed = pl.tile.cast(padded, pl.BF16, mode="rint")
                route_weights = pl.tile.cast(narrowed, pl.FP32)
                acc = pl.tile.full([1, D], dtype=pl.FP32, value=0.0)
                for slot in pl.range(TOPK):
                    row = pl.load(slot_window, [slot, 0], [1, D])
                    value = pl.cast(row, pl.FP32)
                    weight = pl.tile.read(route_weights, [0, slot])
                    weighted = pl.mul(value, weight)
                    acc = pl.add(acc, weighted)
                routed = pl.cast(acc, pl.BF16, mode="rint")
                resident = pl.load(shared, [0, 0], [1, D])
                shared_row = pl.cast(resident, pl.FP32)
                routed_row = pl.cast(routed, pl.FP32)
                joined = pl.add(routed_row, shared_row)
                result = pl.cast(joined, pl.BF16, mode="rint")
                pl.store(result, [0, 0], combined)
                if streaming:
                    pld.system.notify(done, peer=owner, offsets=[EP, 0], value=sequence, op=pld.NotifyOp.Set)
        return combined

    return routed_rank


routed_rank = make_routed_rank()


@pl.jit.host
def one_token_expert_graph(
    token: pl.Tensor[[EP, 1, D], pl.BF16], ids: pl.Tensor[[EP, 1, TOPK], pl.INT32],
    gate_up: pl.Tensor[[EP, GATE_ROWS, 64], pl.INT8], gate_scale: pl.Tensor[[EP, LOCAL * 2 * I], pl.FP32],
    down: pl.Tensor[[EP, DOWN_ROWS, 64], pl.INT8], down_scale: pl.Tensor[[EP, LOCAL * D], pl.FP32],
    weights: pl.Tensor[[EP, 1, TOPK], pl.FP32], shared: pl.Tensor[[EP, 1, D], pl.BF16],
    combined: pl.Out[pl.Tensor[[EP, 1, D], pl.BF16]], owner: pl.Scalar[pl.INT32], sequence: pl.Scalar[pl.INT32],
):
    x = pld.alloc_window_buffer([1, D], dtype=pl.BF16)
    i = pld.alloc_window_buffer([1, 8], dtype=pl.INT32)
    y = pld.alloc_window_buffer([TOPK, D], dtype=pl.BF16)
    r = pld.alloc_window_buffer([1, 16], dtype=pl.INT32)
    d = pld.alloc_window_buffer([EP, 16], dtype=pl.INT32)
    for rank in pl.range(EP):
        tx = pld.window(x, [1, D], dtype=pl.BF16)
        ix = pld.window(i, [1, 8], dtype=pl.INT32)
        sy = pld.window(y, [TOPK, D], dtype=pl.BF16)
        arrived = pld.window(r, [1, 16], dtype=pl.INT32)
        done = pld.window(d, [EP, 16], dtype=pl.INT32)
        routed_rank(token[rank], ids[rank], gate_up[rank], gate_scale[rank], down[rank], down_scale[rank],
                    weights[rank], shared[rank], combined[rank], tx, ix, sy, arrived, done, owner, rank, sequence, device=rank)

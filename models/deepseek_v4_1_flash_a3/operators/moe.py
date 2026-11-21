# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=16
"""DeepSeek-V4.1 prefill MoE with compact TP owner rows and mHC."""

import pypto.language as pl
import pypto.language.distributed as pld


from .config import FLASH as M
from .config import EP_SIZE, TP_SIZE
from .config import MOE_TOKENS, MOE_RECV_MAX, N_LOCAL_EXPERTS, ROUTE_T_DYN
from .config import T_DYN
from .gate import gate_normalized as npu_gate
from .expert_shared import expert_shared
from .expert_routed import (
    expert_routed_scatter,
)
from .hc import hc_pre, hc_post
from .rmsnorm import rms_norm_tiled
from .o_proj import prefill_tp_output_all_reduce


if EP_SIZE % TP_SIZE:
    raise ValueError(f"EP{EP_SIZE} must be divisible by TP{TP_SIZE}")


# model config
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
N_EXPERTS = EP_SIZE * N_LOCAL_EXPERTS
TOPK = M.num_experts_per_tok
MOE_INTER = M.moe_intermediate_size
MX_GROUP = M.weight_block_size[0]


# EP transport layout
RECV_MAX = MOE_RECV_MAX
AUX_WIDTH = 8
ROUTE_WIDTH = 8
T = MOE_TOKENS
TOKENS = MOE_TOKENS
N_RANKS = EP_SIZE
N_LOCAL = N_LOCAL_EXPERTS
K_SCALE = 1
SCALE_WIDTH = 16
MAX_PER_SRC = T
AUX_W = 0
AUX_PAD = AUX_WIDTH
IDX_PAD = ROUTE_WIDTH
SIGNAL_PAD = 128  # 512-byte padded monotonic counter lane

# tiling
SCALE_COPY_TILE = 256
SCALE_PACK_TMP_TILE = ((64 + K_SCALE + 31) // 32) * 32
RECV_TILE = 16


@pl.jit.inline(auto_scope=False)
def clear_moe_signals(
    completion_anchor: pl.Tensor,
    arrived: pld.DistributedTensor[[EP_SIZE, SIGNAL_PAD], pl.INT32],
    data_arrived: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32],
    combine_arrived: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32],
    anchor_indices: pl.constexpr,
):
    """Clear a completed entry's MoE arrival and round-release signals."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_signal_clear", allow_early_resolve=False):
        anchor = pl.read(completion_anchor, anchor_indices)
        zero = pl.cast(0, pl.INT32)
        for peer in pl.range(EP_SIZE):
            pl.write(arrived, [peer, 0], zero)
            pl.write(data_arrived, [peer, 0, 0], zero)
            pl.write(combine_arrived, [peer, 0, 0], zero)
            pl.write(combine_arrived, [peer, 1, 0], zero)
        pl.write(completion_anchor, anchor_indices, anchor)
    return completion_anchor


@pl.jit.inline
def dispatch(
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    x_norm_mx: pl.Tensor[[T, D], pl.INT8],
    x_norm_scale: pl.Tensor[[1, T * K_SCALE], pl.FP32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    recv_x_out: pl.Tensor[[N_LOCAL, RECV_MAX, D], pl.INT8],
    recv_scale_out: pl.Tensor[[1, N_LOCAL * RECV_MAX * K_SCALE], pl.FP32],
    recv_weight_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.FP32],
    recv_route_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.INT32],
    recv_count_out: pl.Tensor[[N_LOCAL, 1], pl.INT32],
    recv_meta_local: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL * RECV_MAX, SCALE_WIDTH], pl.FP32],
    recv_weights: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_routes: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, SIGNAL_PAD], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, N_LOCAL, SIGNAL_PAD], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    x_norm_mx_raw = x_norm_mx
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_reuse_wait") as _reuse_tid:
        _anchor = pl.read(indices, [0, 0])
    scale_src = pl.create_tensor([T, SCALE_WIDTH], dtype=pl.FP32)
    with pl.spmd(T // 16, name_hint="dispatch_scale_pad", deps=[_reuse_tid]) as _stage_tid:
        start = pl.tile.get_block_idx() * 16
        source_factor = pl.load(x_norm_scale, [0, start], [1, 16])
        source_factor = pl.reshape(source_factor, [16, 1])
        ones = pl.tile.full([16, SCALE_WIDTH], dtype=pl.FP32, value=1.0)
        padded = pl.row_expand_mul(ones, source_factor)
        pl.store(padded, [start, 0], scale_src)

    meta_stage = pl.create_tensor([N_RANKS, N_LOCAL], dtype=pl.INT32)
    # Publish per-source route counts.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_meta_publish", deps=[_reuse_tid], allow_early_resolve=True) as _meta_push_tid:
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)

        # Count how many routes land in each (dst, loc_e) lane (no payload move).
        cursor = pl.array.create(N_RANKS * N_LOCAL, pl.INT32)
        for d in pl.range(N_RANKS):
            for e in pl.range(N_LOCAL):
                cursor[d * N_LOCAL + e] = 0
        indices_tile = pl.tile.load(indices, [0, 0], [T, IDX_PAD], valid_shape=[T, TOPK])
        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.tile.read(indices_tile, [t, k])
                dst = eid // N_LOCAL
                loc_e = eid - dst * N_LOCAL
                cursor[dst * N_LOCAL + loc_e] = cursor[dst * N_LOCAL + loc_e] + 1

        meta_tile = pl.tile.full([1, (N_LOCAL + 7) // 8 * 8], dtype=pl.INT32, value=0)
        for dst in pl.range(N_RANKS):
            for e in pl.range(N_LOCAL):
                pl.tile.write(meta_tile, [0, e], cursor[dst * N_LOCAL + e])
            meta_valid = pl.tile.set_validshape(meta_tile, 1, N_LOCAL)
            meta_stage = pl.store(meta_valid, [dst, 0], meta_stage)
            pld.tensor.put(
                dst=recv_meta, peer=dst, src=meta_stage,
                dst_offsets=[my_rank, 0], src_offsets=[dst, 0], shape=[1, N_LOCAL],
            )
            if dst != my_rank:
                pld.system.notify(
                    target=arrived, peer=dst, offsets=[my_rank, 0],
                    value=moe_epoch, op=pld.NotifyOp.Set,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_meta_wait") as _meta_wait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(
                    signal=arrived, offsets=[src, 0],
                    expected=moe_epoch, cmp=pld.WaitCmp.Ge,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_meta_finalize", deps=[_meta_push_tid, _meta_wait_tid], allow_early_resolve=True) as _meta_tid:
        for e in pl.range(N_LOCAL):
            acc = pl.const(0, pl.INT32)
            for src in pl.range(N_RANKS):
                count = pl.read(recv_meta, [src, e])
                pl.write(recv_meta_local, [src, e], count)
                acc = acc + count
            pl.write(recv_count_out, [e, 0], acc)

    aux_stage = pl.create_tensor([N_RANKS * N_LOCAL * MAX_PER_SRC, AUX_PAD], dtype=pl.FP32)
    route_stage = pl.create_tensor([N_RANKS * N_LOCAL * MAX_PER_SRC, IDX_PAD], dtype=pl.INT32)
    # Publish one payload lane per destination and local expert.
    push_blocks = pl.cast(N_RANKS * N_LOCAL, pl.INDEX)
    if num_tokens <= 1:
        push_blocks = pl.max(1, pl.cast(num_tokens, pl.INDEX) * TOPK)
    with pl.spmd(push_blocks, name_hint="dispatch_push", deps=[_reuse_tid, _stage_tid], allow_early_resolve=False) as _push_tid:
        push_block = pl.tile.get_block_idx()
        dst = push_block // N_LOCAL
        loc_e = push_block - dst * N_LOCAL
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)

        indices_tile = pl.tile.load(indices, [0, 0], [T, IDX_PAD], valid_shape=[T, TOPK])
        if active_tokens == 1:
            destination = pl.cast(pl.tile.read(indices_tile, [0, push_block]), pl.INDEX)
            dst = destination // N_LOCAL
            loc_e = destination % N_LOCAL
            # The first slot owns this expert's lane even for repeated route IDs.
            for prior in pl.range(push_block):
                if pl.tile.read(indices_tile, [0, prior]) == destination:
                    loc_e = pl.cast(-1, pl.INDEX)
        e_lane_base = loc_e * RECV_MAX + my_rank * MAX_PER_SRC
        weights_tile = pl.tile.load(weights, [0, 0], [T, AUX_PAD], valid_shape=[T, TOPK])
        aux_lane = pl.tile.full([MAX_PER_SRC, AUX_PAD], dtype=pl.FP32, value=0.0)
        route_lane = pl.tile.full([MAX_PER_SRC, IDX_PAD], dtype=pl.INT32, value=0)
        slot_ctr = pl.array.create(1, pl.INT32)
        slot_ctr[0] = 0

        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.tile.read(indices_tile, [t, k])
                route_dst = eid // N_LOCAL
                route_local_e = eid - route_dst * N_LOCAL
                if route_local_e == loc_e:
                    if route_dst == dst:
                        slot = slot_ctr[0]
                        slot_ctr[0] = slot + 1
                        row = e_lane_base + slot
                        pld.tensor.put(
                            dst=recv_x, peer=dst, src=x_norm_mx_raw,
                            dst_offsets=[row, 0], src_offsets=[t, 0], shape=[1, D],
                        )
                        pld.tensor.put(
                            dst=recv_scale, peer=dst, src=scale_src,
                            dst_offsets=[row, 0], src_offsets=[t, 0], shape=[1, SCALE_WIDTH],
                        )
                        route_index = pl.cast(t * TOPK + k, pl.INT32)
                        route_weight = pl.tile.read(weights_tile, [t, k])
                        pl.tile.write(aux_lane, [slot, AUX_W], route_weight)
                        pl.tile.write(route_lane, [slot, 0], route_index)

        used = pl.cast(slot_ctr[0], pl.INDEX)
        if used > 0:
            stage_row = push_block * MAX_PER_SRC
            aux_stage = pl.store(aux_lane, [stage_row, 0], aux_stage)
            route_stage = pl.store(route_lane, [stage_row, 0], route_stage)
            pld.tensor.put(
                dst=recv_weights, peer=dst, src=aux_stage,
                dst_offsets=[e_lane_base, 0], src_offsets=[stage_row, 0], shape=[used, AUX_PAD], chunk_rows=RECV_TILE,
            )
            pld.tensor.put(
                dst=recv_routes, peer=dst, src=route_stage,
                dst_offsets=[e_lane_base, 0], src_offsets=[stage_row, 0], shape=[used, IDX_PAD], chunk_rows=RECV_TILE,
            )

    # Publish one epoch only after every local payload store completes.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_publish_epoch", deps=[_push_tid], allow_early_resolve=False) as _publish_tid:
        for dst in pl.range(N_RANKS):
            if dst != my_rank:
                pld.system.notify(
                    target=data_arrived, peer=dst, offsets=[my_rank, 0, 0],
                    value=moe_epoch, op=pld.NotifyOp.Set,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_wait", deps=[_publish_tid]) as _wait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(
                    signal=data_arrived, offsets=[src, 0, 0],
                    expected=moe_epoch, cmp=pld.WaitCmp.Ge,
                )

    # Gather aligned row tiles into compact expert buffers.
    recv_x_out_flat = pl.reshape(recv_x_out, [N_LOCAL * RECV_MAX, D])
    gather_tids = pl.array.create(N_LOCAL, pl.TASK_ID)
    for e in pl.parallel(N_LOCAL):
        gather_tids[e] = _wait_tid
        expert_rows = pl.cast(pl.read(recv_count_out, [e, 0]), pl.INDEX)
        expert_tiles = (expert_rows + RECV_TILE - 1) // RECV_TILE
        if expert_tiles > 0:
            with pl.spmd(expert_tiles, name_hint="dispatch_gather", deps=[_wait_tid, _push_tid, _meta_tid],
                         allow_early_resolve=False) as gather_tid:
                tile_row = pl.tile.get_block_idx() * RECV_TILE
                tile_end = pl.min(tile_row + RECV_TILE, expert_rows)
                gathered_scale = pl.tile.full([1, RECV_TILE], dtype=pl.FP32, value=0.0)
                gathered_weight = pl.tile.full([1, RECV_TILE], dtype=pl.FP32, value=0.0)
                gathered_route = pl.tile.full([1, RECV_TILE], dtype=pl.INT32, value=0)
                e_base_row = e * RECV_MAX
                source_begin = pl.cast(0, pl.INDEX)
                for src in pl.range(N_RANKS):
                    source_count = pl.read(recv_meta_local, [src, e])
                    source_rows = pl.cast(source_count, pl.INDEX)
                    source_end = source_begin + source_rows
                    src_base_row = e_base_row + src * MAX_PER_SRC
                    gather_begin = pl.max(tile_row, source_begin)
                    gather_end = pl.min(tile_end, source_end)
                    for out_col in pl.range(gather_begin, gather_end):
                        in_row = src_base_row + out_col - source_begin
                        out_row = e_base_row + out_col
                        recv_x_raw = pl.load(recv_x, [in_row, 0], [1, D])
                        recv_x_mx = recv_x_raw
                        recv_x_out_flat = pl.store(recv_x_mx, [out_row, 0], recv_x_out_flat)
                        factor = pl.read(recv_scale, [in_row, 0])
                        pl.tile.write(gathered_scale, [0, out_col - tile_row], factor)
                        recv_weight = pl.read(recv_weights, [in_row, AUX_W])
                        pl.tile.write(gathered_weight, [0, out_col - tile_row], recv_weight)
                        recv_route = pl.read(recv_routes, [in_row, 0])
                        pl.tile.write(gathered_route, [0, out_col - tile_row], recv_route)
                    source_begin = source_end
                recv_weight_out = pl.store(gathered_weight, [e, tile_row], recv_weight_out)
                recv_route_out = pl.store(gathered_route, [e, tile_row], recv_route_out)
                recv_scale_out = pl.store(gathered_scale, [0, e_base_row + tile_row], recv_scale_out)
            gather_tids[e] = gather_tid
    return pl.system.task_dummy(deps=[_wait_tid, _push_tid, _meta_tid, gather_tids])


@pl.jit.inline
def combine_scattered(
    shared_output: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    combined_counts: pl.Tensor[[T, 16], pl.INT32],
    routed_output: pld.DistributedTensor[[T * TOPK, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, N_LOCAL, SIGNAL_PAD], pl.INT32],
    output_ready: pl.Scalar[pl.TASK_ID],
    scatter_done: pl.Scalar[pl.TASK_ID],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="combine_notify", deps=[scatter_done], allow_early_resolve=False) as notify_tid:
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.notify(
                    target=combine_arrived, peer=peer, offsets=[my_rank, 0, 0],
                    value=moe_epoch, op=pld.NotifyOp.Set,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="combine_wait", deps=[notify_tid]) as _cwait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(
                    signal=combine_arrived, offsets=[src, 0, 0],
                    expected=moe_epoch, cmp=pld.WaitCmp.Ge,
                )

    # Wait for local and peer routed-output stores.
    with pl.spmd(T, name_hint="combine_reduce", deps=[_cwait_tid, output_ready]) as _reduce_tid:
        t = pl.tile.get_block_idx()
        combined = pl.cast(0, pl.INT32)
        if t < num_tokens:
            shared_row = pl.load(shared_output, [t, 0], [1, D])
            acc = pl.cast(shared_row, target_type=pl.FP32)
            for k in pl.range(TOPK):
                routed_row = pl.load(routed_output, [t * TOPK + k, 0], [1, D])
                routed_fp32 = pl.cast(routed_row, target_type=pl.FP32)
                acc = pl.add(acc, routed_fp32)
                combined = combined + pl.cast(1, pl.INT32)
            ffn_row = pl.cast(acc, target_type=pl.BF16, mode="rint")
            ffn_out = pl.store(ffn_row, [t, 0], ffn_out)
        pl.write(combined_counts, [t, 0], combined)
    return _reduce_tid


@pl.jit.inline
def _local_token_count(num_tokens: pl.Scalar[pl.INT32], tp_rank: pl.Scalar[pl.INT32]):
    """Map the replicated TP token prefix to this rank's compact local prefix."""
    # Native INDEX scalar for local row counts and addresses.
    return (num_tokens + TP_SIZE - 1 - tp_rank) // TP_SIZE


@pl.jit.inline
def _compact_local_tokens(
    source: pl.Tensor[[T_DYN, D], pl.BF16],
    compact: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Pack this TP rank's interleaved owner rows into the local-token ABI prefix."""
    local_tokens = _local_token_count(num_tokens, tp_rank)
    for row in pl.spmd(pl.tensor.dim(compact, 0), name_hint="prefill_moe_compact"):
        value = pl.tile.full([1, D], dtype=pl.BF16, value=0.0)
        if row < local_tokens:
            source_row = row * TP_SIZE + tp_rank
            value = pl.tile.load(source, [source_row, 0], [1, D])
        pl.tile.store(value, [row, 0], compact, shapes=[1, D])
    return compact


@pl.jit.inline
def _scatter_local_tokens(
    compact: pl.Tensor[[T_DYN, D], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    compact_ready: pl.Scalar[pl.TASK_ID],
):
    """Scatter compact local MoE rows back into the original TP token positions."""
    tokens = pl.tensor.dim(output, 0)
    local_tokens = _local_token_count(num_tokens, tp_rank)
    with pl.spmd(tokens, name_hint="prefill_moe_scatter_zero") as zero_tid:
        row = pl.tile.get_block_idx()
        zero = pl.tile.full([1, D], dtype=pl.BF16, value=0.0)
        pl.tile.store(zero, [row, 0], output, shapes=[1, D])
    with pl.spmd(tokens, name_hint="prefill_moe_scatter", deps=[zero_tid, compact_ready]):
        row = pl.tile.get_block_idx()
        if row < local_tokens:
            dst_row = row * TP_SIZE + tp_rank
            value = pl.tile.load(compact, [row, 0], [1, D])
            pl.tile.store(value, [dst_row, 0], output, shapes=[1, D])
    return output


@pl.jit.inline(auto_scope=False)
def prefill_moe_sublayer(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * (1), MOE_INTER], pl.FP32, pl.ND],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * (1), D], pl.FP32, pl.ND],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * (1), MOE_INTER], pl.FP32, pl.ND],

    shared_w1: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w1_scale: pl.Tensor[[1, MOE_INTER], pl.FP32, pl.ND],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w2_scale: pl.Tensor[[1, D], pl.FP32, pl.ND],
    shared_w3: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w3_scale: pl.Tensor[[1, MOE_INTER], pl.FP32, pl.ND],
    recv_meta: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, SCALE_WIDTH], pl.FP32],
    recv_weights: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], pl.FP32],
    recv_routes: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32],
    arrived: pld.DistributedTensor[[EP_SIZE, SIGNAL_PAD], pl.INT32],
    data_arrived: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32],
    routed_output: pld.DistributedTensor[[ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32],
    ffn_input: pl.Tensor[[T_DYN, D], pl.BF16],
    ffn_owned: pl.Tensor[[T_DYN, D], pl.BF16],
    next_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    local_ffn_normed: pl.Tensor[[TOKENS, D], pl.BF16],
    indices: pl.Tensor[[TOKENS, TOPK], pl.INT32],
    weights: pl.Tensor[[TOKENS, TOPK], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    ep_rank: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Collapse mHC streams, normalize compact TP owner rows, and run the EP MoE."""
    hc_pre(
        x_hc, pre_mix, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        ffn_input, next_pre_mix, post_mix, residual_mix,
    )
    # Compact interleaved TP owner rows for the EP local-token ABI.
    local_ffn_input = pl.create_tensor([TOKENS, D], dtype=pl.BF16)
    _compact_local_tokens(ffn_input, local_ffn_input, num_tokens, tp_rank)
    norm_tokens = pl.tensor.dim(local_ffn_input, 0)
    for norm_block in pl.spmd((norm_tokens + 7) // 8, name_hint="rms_norm"):
        rms_norm_tiled(local_ffn_input, ffn_norm_weight, local_ffn_normed, norm_block, norm_tokens, output_fp32=False)

    local_num_tokens = _local_token_count(num_tokens, tp_rank)
    local_ffn_owned = pl.create_tensor([TOKENS, D], dtype=pl.BF16)
    round_counts = pl.create_tensor([3], dtype=pl.INT32)
    with pl.scope():
        with pl.spmd(TOKENS, name_hint="moe_output_zero") as _output_zero_tid:
            zero_t = pl.tile.get_block_idx()
            zero_row = pl.tile.full([1, D], dtype=pl.BF16, value=0.0)
            local_ffn_owned = pl.store(zero_row, [zero_t, 0], local_ffn_owned)

        x_norm_mx = pl.create_tensor([TOKENS, D], dtype=pl.INT8)
        x_norm_scale = pl.create_tensor([1, TOKENS * 1], dtype=pl.FP32)
        npu_gate(
            local_ffn_normed, gate_weight, correction_bias, local_num_tokens,
            x_norm_mx, x_norm_scale, indices, weights,
        )
        dispatch_indices = indices

        shared_output = pl.create_tensor([TOKENS, D], dtype=pl.BF16)
        expert_shared(
            x_norm_mx, x_norm_scale,
            shared_w1, shared_w1_scale,
            shared_w3, shared_w3_scale,
            shared_w2, shared_w2_scale,
            shared_output, local_num_tokens,
        )

        recv_x_local = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX, D], dtype=pl.INT8)
        # Contiguous per-token FP32 scales for the routed experts.
        recv_scale_local_backing = pl.create_tensor([1, N_LOCAL_EXPERTS * RECV_MAX * 1], dtype=pl.FP32)
        recv_weight_local = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX], dtype=pl.FP32)
        recv_route_local = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX], dtype=pl.INT32)
        recv_count_local = pl.create_tensor([N_LOCAL_EXPERTS, 1], dtype=pl.INT32)
        recv_meta_local = pl.create_tensor([EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
        input_ready = dispatch(
            dispatch_indices, x_norm_mx, x_norm_scale, weights,
            recv_x_local, recv_scale_local_backing,
            recv_weight_local, recv_route_local, recv_count_local, recv_meta_local,
            recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
            arrived, data_arrived, local_num_tokens, ep_rank, moe_epoch,
        )

        scatter_done = expert_routed_scatter(
            recv_x_local, recv_scale_local_backing,
            recv_weight_local, recv_count_local,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            recv_route_local, recv_meta_local,
            routed_output, input_ready,
        )
        combined_counts = pl.create_tensor([TOKENS, 16], dtype=pl.INT32)
        output_ready = combine_scattered(
            shared_output, local_ffn_owned, combined_counts,
            routed_output, combine_arrived,
            _output_zero_tid, scatter_done,
            local_num_tokens, ep_rank, moe_epoch,
        )
        # Complete all peer readers before reusing the windows.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_round_release", deps=[output_ready]) as released:
            for peer in pl.range(EP_SIZE):
                if peer != ep_rank:
                    pld.system.notify(
                        target=combine_arrived, peer=peer, offsets=[ep_rank, 1, 0],
                        value=moe_epoch, op=pld.NotifyOp.Set,
                    )
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_round_released", deps=[released]) as round_ready:
            for peer in pl.range(EP_SIZE):
                if peer != ep_rank:
                    pld.system.defer_wait(
                        signal=combine_arrived, offsets=[peer, 1, 0],
                        expected=moe_epoch, cmp=pld.WaitCmp.Ge,
                    )
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_round_counts", deps=[round_ready]):
            received = pl.cast(0, pl.INT32)
            for expert in pl.range(N_LOCAL_EXPERTS):
                received = received + pl.read(recv_count_local, [expert, 0])
            sent = pl.cast(local_num_tokens * TOPK, pl.INT32)
            combined = pl.cast(0, pl.INT32)
            for token in pl.range(TOKENS):
                combined = combined + pl.read(combined_counts, [token, 0])
            pl.write(round_counts, [0], sent)
            pl.write(round_counts, [1], received)
            pl.write(round_counts, [2], combined)
        _scatter_local_tokens(local_ffn_owned, ffn_owned, num_tokens, tp_rank, round_ready)
    return ffn_owned, round_counts


@pl.jit.inline(auto_scope=False)
def prefill_ffn_restore(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    ffn_owned: pl.Tensor[[T_DYN, D], pl.BF16],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    ffn_window: pld.DistributedTensor,
    ffn_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    ffn_output: pl.Tensor[[T_DYN, D], pl.BF16],
    x_hc_out: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    ffn_epoch: pl.Scalar[pl.INT32],
):
    """Sum TP routed partials and expand the residual stream with mHC."""
    prefill_tp_output_all_reduce(
        ffn_owned, ffn_window, ffn_arrived, ffn_output, group_base, tp_rank, num_tokens, ffn_epoch,
    )
    hc_post_deps = pl.array.create(2, pl.TASK_ID)
    hc_post(ffn_output, x_hc, post_mix, residual_mix, x_hc_out, deps=hc_post_deps)
    return x_hc_out

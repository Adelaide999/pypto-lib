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
from .config import MOE_TOKENS, N_LOCAL_EXPERTS, ROUTE_T_DYN
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
AUX_WIDTH = 8
ROUTE_WIDTH = 8
T = MOE_TOKENS
TOKENS = MOE_TOKENS
N_RANKS = EP_SIZE
N_LOCAL = N_LOCAL_EXPERTS
ROUTED_RECV_ROWS = N_RANKS * T * TOPK
LOCAL_RECV_ROWS = ROUTED_RECV_ROWS + N_LOCAL * 16
SIGNAL_PAD = 128  # 512-byte padded monotonic counter lane

# tiling
RECV_TILE = 16
# Local owner rows are interleaved across TP ranks.  Panel copies amortize the
# wide hidden-vector transfer while keeping the temporary below the A3 vector
# buffer limit.
LOCAL_ROW_TILE = 8
COMBINE_TOKEN_TILE = 2


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
    x_norm_scale: pl.Tensor[[1, T], pl.FP32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    recv_x_out: pl.Tensor[[LOCAL_RECV_ROWS, D], pl.INT8],
    recv_scale_out: pl.Tensor[[1, LOCAL_RECV_ROWS], pl.FP32],
    recv_weight_out: pl.Tensor[[1, LOCAL_RECV_ROWS], pl.FP32],
    recv_route_out: pl.Tensor[[1, LOCAL_RECV_ROWS], pl.INT32],
    recv_count_out: pl.Tensor[[N_LOCAL, 2], pl.INT32],
    recv_meta_local: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[ROUTED_RECV_ROWS, D], pl.INT8],
    recv_weights: pld.DistributedTensor[[ROUTED_RECV_ROWS, AUX_WIDTH], pl.FP32],
    arrived: pld.DistributedTensor[[N_RANKS, SIGNAL_PAD], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, N_LOCAL, SIGNAL_PAD], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    reuse_deps: pl.Array[1, pl.TASK_ID],
):
    x_norm_mx_raw = x_norm_mx
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_reuse_wait", deps=[reuse_deps]) as _reuse_tid:
        _anchor = pl.read(indices, [0, 0])
    send_starts = pl.create_tensor([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_starts = pl.create_tensor([N_RANKS, N_LOCAL], dtype=pl.INT32)
    route_slots = pl.create_tensor([T, ROUTE_WIDTH], dtype=pl.INT32)
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
        indices_tile = pl.tile.load(indices, [0, 0], [T, ROUTE_WIDTH], valid_shape=[T, TOPK])
        slot_tile = pl.tile.full([T, ROUTE_WIDTH], dtype=pl.INT32, value=0)
        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.tile.read(indices_tile, [t, k])
                slot = cursor[eid]
                pl.tile.write(slot_tile, [t, k], slot)
                cursor[eid] = slot + 1
        pl.store(slot_tile, [0, 0], route_slots)

        meta_tile = pl.tile.full([1, (N_LOCAL + 7) // 8 * 8], dtype=pl.INT32, value=0)
        for dst in pl.range(N_RANKS):
            prefix = pl.const(0, pl.INT32)
            for e in pl.range(N_LOCAL):
                count = cursor[dst * N_LOCAL + e]
                pl.write(send_starts, [dst, e], prefix)
                pl.tile.write(meta_tile, [0, e], count)
                prefix = prefix + count
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

    # Metadata is consumed by the gather stage, so finalize it in the same
    # core-group task that waits for peer counts.  This removes one graph task
    # while keeping the metadata dependency ahead of every gather worker.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dispatch_meta_wait_finalize",
        deps=[_meta_push_tid],
    ) as _meta_wait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(
                    signal=arrived, offsets=[src, 0],
                    expected=moe_epoch, cmp=pld.WaitCmp.Ge,
                )
        source_prefix = pl.array.create(N_RANKS, pl.INT32)
        for src in pl.range(N_RANKS):
            source_prefix[src] = 0
        expert_prefix = pl.const(0, pl.INT32)
        for e in pl.range(N_LOCAL):
            acc = pl.const(0, pl.INT32)
            for src in pl.range(N_RANKS):
                count = pl.read(recv_meta, [src, e])
                pl.write(recv_meta_local, [src, e], count)
                pl.write(recv_starts, [src, e], source_prefix[src])
                source_prefix[src] = source_prefix[src] + count
                acc = acc + count
            pl.write(recv_count_out, [e, 0], acc)
            pl.write(recv_count_out, [e, 1], expert_prefix)
            expert_prefix = expert_prefix + pl.cast((acc + 15) // 16 * 16, pl.INT32)

    aux_stage = pl.create_tensor([T * TOPK, AUX_WIDTH], dtype=pl.FP32)
    routes = pl.max(0, pl.min(num_tokens, T)) * TOPK
    workers = pl.max(1, pl.min(routes, 48))
    with pl.spmd(workers, name_hint="dispatch_push", deps=[_meta_push_tid], allow_early_resolve=False) as _push_tid:
        worker = pl.tile.get_block_idx()
        for route in pl.range(worker, routes, workers):
            token = route // TOPK
            choice = route % TOPK
            eid = pl.read(indices, [token, choice])
            dst = eid // N_LOCAL
            slot = pl.read(route_slots, [token, choice])
            start = pl.read(send_starts, [dst, eid % N_LOCAL])
            row = my_rank * (T * TOPK) + start + slot
            weight = pl.read(weights, [token, choice])
            factor = pl.read(x_norm_scale, [0, token])
            metadata = pl.tile.full([1, AUX_WIDTH], dtype=pl.FP32, value=0.0)
            pl.tile.write(metadata, [0, 0], weight)
            pl.tile.write(metadata, [0, 1], factor)
            pl.tile.write(metadata, [0, 2], pl.cast(pl.cast(route, pl.INT32), pl.FP32))
            pl.store(metadata, [route, 0], aux_stage)
            pld.tensor.put(
                dst=recv_x, peer=dst, src=x_norm_mx_raw,
                dst_offsets=[row, 0], src_offsets=[token, 0], shape=[1, D],
            )
            pld.tensor.put(
                dst=recv_weights, peer=dst, src=aux_stage,
                dst_offsets=[row, 0], src_offsets=[route, 0], shape=[1, AUX_WIDTH],
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
    gather_tids = pl.array.create(N_LOCAL, pl.TASK_ID)
    for e in pl.parallel(N_LOCAL):
        gather_tids[e] = _wait_tid
        expert_rows = pl.cast(pl.read(recv_count_out, [e, 0]), pl.INDEX)
        expert_tiles = (expert_rows + RECV_TILE - 1) // RECV_TILE
        if expert_tiles > 0:
            with pl.spmd(expert_tiles, name_hint="dispatch_gather", deps=[_wait_tid, _push_tid, _meta_wait_tid],
                         allow_early_resolve=False) as gather_tid:
                tile_row = pl.tile.get_block_idx() * RECV_TILE
                tile_end = pl.min(tile_row + RECV_TILE, expert_rows)
                gathered_scale = pl.tile.full([1, RECV_TILE], dtype=pl.FP32, value=0.0)
                gathered_weight = pl.tile.full([1, RECV_TILE], dtype=pl.FP32, value=0.0)
                gathered_route = pl.tile.full([1, RECV_TILE], dtype=pl.INT32, value=0)
                e_base_row = pl.cast(pl.read(recv_count_out, [e, 1]), pl.INDEX)
                source_begin = pl.cast(0, pl.INDEX)
                for src in pl.range(N_RANKS):
                    source_count = pl.read(recv_meta_local, [src, e])
                    source_rows = pl.cast(source_count, pl.INDEX)
                    source_end = source_begin + source_rows
                    src_base_row = src * (T * TOPK) + pl.cast(pl.read(recv_starts, [src, e]), pl.INDEX)
                    gather_begin = pl.max(tile_row, source_begin)
                    gather_end = pl.min(tile_end, source_end)
                    # Rows for one source/expert lane are contiguous in the
                    # transport window. Load their hidden payload once per
                    # panel and keep the route metadata scalar path below.
                    gather_rows = gather_end - gather_begin
                    if gather_rows > 0:
                        in_row = src_base_row + gather_begin - source_begin
                        out_row = e_base_row + gather_begin
                        recv_x_raw = pl.load(
                            recv_x,
                            [in_row, 0],
                            [RECV_TILE, D],
                            valid_shape=[gather_rows, D],
                        )
                        recv_x_out = pl.store(
                            recv_x_raw,
                            [out_row, 0],
                            recv_x_out,
                        )
                    for out_col in pl.range(gather_begin, gather_end):
                        in_row = src_base_row + out_col - source_begin
                        out_row = e_base_row + out_col
                        factor = pl.read(recv_weights, [in_row, 1])
                        pl.tile.write(gathered_scale, [0, out_col - tile_row], factor)
                        recv_weight = pl.read(recv_weights, [in_row, 0])
                        pl.tile.write(gathered_weight, [0, out_col - tile_row], recv_weight)
                        recv_route = pl.cast(pl.read(recv_weights, [in_row, 2]), pl.INT32)
                        return_address = pl.cast(src * (T * TOPK), pl.INT32) + recv_route
                        pl.tile.write(gathered_route, [0, out_col - tile_row], return_address)
                    source_begin = source_end
                recv_weight_out = pl.store(gathered_weight, [0, e_base_row + tile_row], recv_weight_out)
                recv_route_out = pl.store(gathered_route, [0, e_base_row + tile_row], recv_route_out)
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

    # Wait for local and peer routed-output stores. A two-row panel amortizes
    # the 5120-wide hidden loads while retaining the compact worker schedule.
    combine_blocks = (T + COMBINE_TOKEN_TILE - 1) // COMBINE_TOKEN_TILE
    with pl.spmd(combine_blocks, name_hint="combine_reduce", deps=[_cwait_tid]) as _reduce_tid:
        block = pl.tile.get_block_idx()
        t0 = block * COMBINE_TOKEN_TILE
        valid_rows = pl.min(COMBINE_TOKEN_TILE, pl.max(num_tokens - t0, 0))
        if valid_rows > 0:
            shared_rows = pl.load(shared_output, [t0, 0], [COMBINE_TOKEN_TILE, D])
            acc_rows = pl.cast(shared_rows, target_type=pl.FP32)
            for k in pl.range(TOPK):
                routed_rows = pl.load(
                    routed_output,
                    [t0 * TOPK + k, 0],
                    [COMBINE_TOKEN_TILE, D],
                )
                acc_rows = pl.add(acc_rows, pl.cast(routed_rows, target_type=pl.FP32))
            ffn_rows = pl.cast(acc_rows, target_type=pl.BF16, mode="rint")
            if valid_rows == COMBINE_TOKEN_TILE:
                ffn_out = pl.store(ffn_rows, [t0, 0], ffn_out)
            else:
                for lane in pl.range(COMBINE_TOKEN_TILE):
                    if lane < valid_rows:
                        ffn_row = pl.tile.slice(ffn_rows, [1, D], [lane, 0])
                        ffn_out = pl.store(ffn_row, [t0 + lane, 0], ffn_out)
    # Every active row contributes TOPK routed outputs. Keep the count metadata
    # on one block so the 64-byte rows are written without dynamic-index races.
    with pl.spmd(1, name_hint="combine_counts", deps=[_reduce_tid]) as _counts_tid:
        worker = pl.tile.get_block_idx()
        for t in pl.range(worker, T, 1):
            combined = pl.cast(0, pl.INT32)
            if t < num_tokens:
                combined = pl.cast(TOPK, pl.INT32)
            pl.write(combined_counts, [t, 0], combined)
    return _counts_tid


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
    compact_rows = pl.tensor.dim(compact, 0)
    compact_blocks = (compact_rows + LOCAL_ROW_TILE - 1) // LOCAL_ROW_TILE
    with pl.spmd(compact_blocks, name_hint="prefill_moe_compact"):
        block = pl.tile.get_block_idx()
        row0 = block * LOCAL_ROW_TILE
        rows = pl.min(LOCAL_ROW_TILE, pl.max(local_tokens - row0, 0))
        value = pl.tile.full([LOCAL_ROW_TILE, D], dtype=pl.BF16, value=0.0)
        for lane in pl.range(LOCAL_ROW_TILE):
            row = row0 + lane
            if row < local_tokens:
                source_row = row * TP_SIZE + tp_rank
                value[lane : lane + 1, :] = pl.tile.load(source, [source_row, 0], [1, D])
        pl.tile.store(
            pl.set_validshape(value, rows, D),
            [row0, 0],
            compact,
            shapes=[LOCAL_ROW_TILE, D],
        )
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
    local_tokens = _local_token_count(num_tokens, tp_rank)
    blocks = (local_tokens + LOCAL_ROW_TILE - 1) // LOCAL_ROW_TILE
    workers = pl.max(1, pl.min(blocks, 2 * pl.system.available_cluster_count()))
    with pl.spmd(workers, name_hint="prefill_moe_scatter", deps=[compact_ready]):
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, blocks, workers):
            row0 = block * LOCAL_ROW_TILE
            rows = pl.min(LOCAL_ROW_TILE, local_tokens - row0)
            value = pl.tile.load(compact, [row0, 0], [LOCAL_ROW_TILE, D])
            for lane in pl.range(LOCAL_ROW_TILE):
                if lane < rows:
                    row = row0 + lane
                    pl.tile.store(
                        pl.tile.slice(value, [1, D], [lane, 0]),
                        [row * TP_SIZE + tp_rank, 0],
                        output,
                        shapes=[1, D],
                    )
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
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * (1), MOE_INTER], pl.FP32, pl.ND],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * (1), D], pl.FP32, pl.ND],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * (1), MOE_INTER], pl.FP32, pl.ND],

    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[1, MOE_INTER], pl.FP32, pl.ND],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[1, D], pl.FP32, pl.ND],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[1, MOE_INTER], pl.FP32, pl.ND],
    recv_meta: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[ROUTED_RECV_ROWS, D], pl.INT8],
    recv_weights: pld.DistributedTensor[[ROUTED_RECV_ROWS, AUX_WIDTH], pl.FP32],
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
    reuse_deps: pl.Array[1, pl.TASK_ID],
    prefill: pl.constexpr,
):
    """Collapse mHC streams, normalize compact TP owner rows, and run the EP MoE."""
    hc_pre(
        x_hc, pre_mix, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        ffn_input, next_pre_mix, post_mix, residual_mix, prefill=prefill,
    )
    local_num_tokens = _local_token_count(num_tokens, tp_rank)
    local_ffn_owned = pl.create_tensor([TOKENS, D], dtype=pl.BF16)
    round_counts = pl.create_tensor([3], dtype=pl.INT32)
    with pl.scope():
        x_norm_mx = pl.create_tensor([TOKENS, D], dtype=pl.INT8)
        x_norm_scale = pl.create_tensor([1, TOKENS * 1], dtype=pl.FP32)
        shared_output = pl.create_tensor([TOKENS, D], dtype=pl.BF16)
        if local_num_tokens > 0:
            local_ffn_input = pl.create_tensor([TOKENS, D], dtype=pl.BF16)
            _compact_local_tokens(ffn_input, local_ffn_input, num_tokens, tp_rank)
            norm_tokens = pl.tensor.dim(local_ffn_input, 0)
            for norm_block in pl.spmd((norm_tokens + 7) // 8, name_hint="rms_norm"):
                rms_norm_tiled(local_ffn_input, ffn_norm_weight, local_ffn_normed, norm_block, norm_tokens, output_fp32=False)
            npu_gate(
                local_ffn_normed, gate_weight, correction_bias, local_num_tokens,
                x_norm_mx, x_norm_scale, indices, weights,
            )

            expert_shared(
                x_norm_mx, x_norm_scale,
                shared_w1, shared_w1_scale,
                shared_w3, shared_w3_scale,
                shared_w2, shared_w2_scale,
                shared_output, local_num_tokens, True, prefill,
            )
        else:
            workers = pl.min(48, TOKENS // 4)
            with pl.spmd(workers, name_hint="moe_empty_owner", deps=[reuse_deps]) as empty_owner:
                worker = pl.tile.get_block_idx()
                for group in pl.range(worker, TOKENS // 4, workers):
                    first = group * 4
                    zeros = pl.tile.full([4, D], dtype=pl.BF16, value=0.0)
                    pl.store(zeros, [first, 0], local_ffn_normed)
                    empty_ids = pl.tile.full([4, 8], dtype=pl.INT32, value=0)
                    pl.store(pl.set_validshape(empty_ids, 4, TOPK), [first, 0], indices)
                    empty_weights = pl.tile.full([4, 8], dtype=pl.FP32, value=0.0)
                    pl.store(pl.set_validshape(empty_weights, 4, TOPK), [first, 0], weights)
        recv_x_local = pl.create_tensor([LOCAL_RECV_ROWS, D], dtype=pl.INT8)
        # Contiguous per-token FP32 scales for the routed experts.
        recv_scale_local_backing = pl.create_tensor([1, LOCAL_RECV_ROWS], dtype=pl.FP32)
        recv_weight_local = pl.create_tensor([1, LOCAL_RECV_ROWS], dtype=pl.FP32)
        recv_route_local = pl.create_tensor([1, LOCAL_RECV_ROWS], dtype=pl.INT32)
        recv_count_local = pl.create_tensor([N_LOCAL_EXPERTS, 2], dtype=pl.INT32)
        recv_meta_local = pl.create_tensor([EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
        input_ready = dispatch(
            indices, x_norm_mx, x_norm_scale, weights,
            recv_x_local, recv_scale_local_backing,
            recv_weight_local, recv_route_local, recv_count_local, recv_meta_local,
            recv_meta, recv_x, recv_weights,
            arrived, data_arrived, local_num_tokens, ep_rank, moe_epoch, reuse_deps,
        )

        scatter_done = expert_routed_scatter(
            recv_x_local, recv_scale_local_backing,
            recv_weight_local, recv_count_local,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            recv_route_local,
            routed_output, input_ready, prefill,
        )
        combined_counts = pl.create_tensor([TOKENS, 16], dtype=pl.INT32)
        output_ready = combine_scattered(
            shared_output, local_ffn_owned, combined_counts,
            routed_output, combine_arrived,
            scatter_done,
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
    prefill: pl.constexpr = False,
):
    """Sum TP routed partials and expand the residual stream with mHC."""
    prefill_tp_output_all_reduce(
        ffn_owned, ffn_window, ffn_arrived, ffn_output, group_base, tp_rank, num_tokens, ffn_epoch, owner_rows=True,
    )
    hc_post_deps = pl.array.create(2, pl.TASK_ID)
    hc_post(
        ffn_output, x_hc, post_mix, residual_mix, x_hc_out,
        deps=hc_post_deps, prefill=prefill,
    )
    return x_hc_out

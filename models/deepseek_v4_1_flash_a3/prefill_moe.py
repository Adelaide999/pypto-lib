# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A3 W8A8 Prefill routing, dispatch, experts and Torch validation."""

import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"
import pypto.language as pl
import torch

from .gemm import (
    VEC_M_TILE as SHARED_VEC_M_TILE,
    make_dot as _ecotech_shared_make_dot,
    make_dot as _shared_make_dot,
    make_fp32_dequantize as _moe_make_fp32_dequantize,
    make_int8_gemm as _moe_make_int8_gemm,
    prepare_ecotech_dynamic_weight as _ecotech_shared_prepare_ecotech_dynamic_weight,
    prepare_ecotech_tail_weight as _ecotech_shared_prepare_ecotech_tail_weight,
)
from .quant import (
    make_cast_tile as _router_make_cast_tile,
    make_dequantize as _ecotech_shared_make_dequantize,
    make_dequantize as _shared_make_dequantize,
    make_dynamic_int8_quantize as _moe_make_dynamic_int8_quantize,
    make_prefill_projection_quantize,
    make_quantize as _ecotech_shared_make_quantize,
    make_quantize as _shared_make_quantize,
    make_quantize_row as _shared_make_quantize_row,
    w8a8_linear as _moe_w8a8_linear,
)
from .utils import pack_blocked_weight, unpack_blocked_weight


# model config
DISPATCH_EXPERTS = 384
DISPATCH_TOPK = 6
DISPATCH_D = 5120
COMBINE_TOPK = 6
COMBINE_D = 5120

# tiling
MOE_VEC_M_TILE = 8
MOE_ACT_TILE = 256
ROUTER_D_TILE = 256
ROUTER_EXPERT_TILE = 32
ROUTER_TOKEN_TILE = 16
ROUTER_GATE_WORKERS = 64
ROUTER_TOPK_TILE = 8
ROUTER_SELECT_TOKEN_TILE = 16
ROUTER_SELECT_WORKERS = 48
DISPATCH_RECV_MAX = 64
DISPATCH_D_TILE = 512
DISPATCH_MAP_PAD = 8
DISPATCH_SOURCE_MAP_PAD = 16


def w8a8_expert_linear(
    value: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run one W8A8 expert projection using the shared linear golden."""

    return _moe_w8a8_linear(value, weight, scale, out_dtype=out_dtype)


def make_w8a8_expert_projection(capacity: int, input_width: int, output_width: int):
    """Build the shared leaf used by routed and shared W8A8 projections.

    Dispatch/combine metadata stays outside this kernel.  Each routed expert
    receives a contiguous token slice and uses the same A2/A3 W8A8 boundary as
    the shared expert: dynamic INT8 activation, INT32 dot, FP32 scales, BF16
    output.
    """

    quantize = _moe_make_dynamic_int8_quantize(capacity, input_width)
    gemm = _moe_make_int8_gemm(capacity, input_width, output_width)
    dequantize = _moe_make_fp32_dequantize(capacity, output_width)

    @pl.jit.inline
    def projection(
        value: pl.Tensor[[capacity, input_width], pl.BF16],
        packed_weight: pl.Tensor[[input_width * (output_width // 64), 64], pl.INT8],
        channel_scale: pl.Tensor[[output_width], pl.FP32],
        payload: pl.Tensor[[capacity, input_width], pl.INT8],
        token_scale: pl.Tensor[[capacity, 1], pl.FP32],
        accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
        output: pl.Tensor[[capacity, output_width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        quantize(value, payload, token_scale, active)
        gemm(payload, packed_weight, accumulator, active)
        dequantize(accumulator, token_scale, channel_scale, output, active)
        return output

    return projection


def make_shared_expert_projection(capacity: int, input_width: int, output_width: int):
    """Alias the W8A8 projection leaf for the shared expert path."""

    return make_w8a8_expert_projection(capacity, input_width, output_width)


def make_routed_expert_projection(capacity: int, input_width: int, output_width: int):
    """Alias the W8A8 projection leaf for one grouped routed expert."""

    return make_w8a8_expert_projection(capacity, input_width, output_width)


def make_shared_expert_w8a8(
    capacity: int,
    hidden: int,
    intermediate: int,
    clamp_limit: float = 10.0,
):
    """Build the complete W8A8 shared expert for one dispatched token slab.

    The activation is evaluated in FP32, narrowed to BF16 at the same point
    as vLLM-Ascend, then dynamically re-quantized before the down projection.
    A routed expert uses this exact entry point after host dispatch has made a
    contiguous ``[tokens, hidden]`` slice for one expert.
    """

    if capacity % 16 or hidden % 256 or intermediate % 256:
        raise ValueError("A2/A3 expert GEMM requires capacity multiple of 16 and K widths multiple of 256")
    gate_up_width = intermediate * 2
    quant_x = _moe_make_dynamic_int8_quantize(capacity, hidden)
    quant_hidden = _moe_make_dynamic_int8_quantize(capacity, intermediate)
    dot_gate_up = _moe_make_int8_gemm(capacity, hidden, gate_up_width)
    dot_down = _moe_make_int8_gemm(capacity, intermediate, hidden)
    dequant_gate_up = _moe_make_fp32_dequantize(capacity, gate_up_width)
    dequant_down = _moe_make_fp32_dequantize(capacity, hidden)

    @pl.jit
    def shared_expert(
        value: pl.Tensor[[capacity, hidden], pl.BF16],
        gate_up_weight: pl.Tensor[[hidden * (gate_up_width // 64), 64], pl.INT8],
        gate_up_scale: pl.Tensor[[gate_up_width], pl.FP32],
        down_weight: pl.Tensor[[intermediate * (hidden // 64), 64], pl.INT8],
        down_scale: pl.Tensor[[hidden], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        x_payload = pl.create_tensor([capacity, hidden], dtype=pl.INT8)
        x_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quant_x(value, x_payload, x_scale, active)

        gate_up_acc = pl.create_tensor([capacity, gate_up_width], dtype=pl.INT32)
        dot_gate_up(x_payload, gate_up_weight, gate_up_acc, active)
        gate_up = pl.create_tensor([capacity, gate_up_width], dtype=pl.BF16)
        dequant_gate_up(gate_up_acc, x_scale, gate_up_scale, gate_up, active)

        activation = pl.create_tensor([capacity, intermediate], dtype=pl.BF16)
        for block in pl.spmd(
            (active + MOE_VEC_M_TILE - 1) // MOE_VEC_M_TILE * (intermediate // MOE_ACT_TILE),
            name_hint="dsv41_a2a3_swiglu_v2",
        ):
            token = block // (intermediate // MOE_ACT_TILE) * MOE_VEC_M_TILE
            column = block % (intermediate // MOE_ACT_TILE) * MOE_ACT_TILE
            rows = pl.min(MOE_VEC_M_TILE, active - token)
            # vLLM's A2/A3 ``SiluAndMulWithClamp`` keeps the two products in
            # BF16.  Use FP32 only for the exponential, then multiply the
            # BF16 sigmoid, gate, and up values without an FP32 promotion.
            gate = pl.cast(pl.slice(gate_up, [MOE_VEC_M_TILE, MOE_ACT_TILE], [token, column]), pl.FP32)
            up = pl.cast(
                pl.slice(gate_up, [MOE_VEC_M_TILE, MOE_ACT_TILE], [token, intermediate + column]), pl.FP32
            )
            gate = pl.minimum(gate, clamp_limit)
            up = pl.maximum(pl.minimum(up, clamp_limit), -clamp_limit)
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate)), 1.0))
            sigmoid_bf16 = pl.cast(sigmoid, pl.BF16, mode="rint")
            silu = pl.mul(gate, pl.cast(sigmoid_bf16, pl.FP32))
            silu_bf16 = pl.cast(silu, pl.BF16, mode="rint")
            activated = pl.mul(pl.cast(silu_bf16, pl.FP32), up)
            activation[token : token + MOE_VEC_M_TILE, column : column + MOE_ACT_TILE] = pl.set_validshape(
                pl.cast(activated, pl.BF16, mode="rint"), rows, MOE_ACT_TILE
            )

        hidden_payload = pl.create_tensor([capacity, intermediate], dtype=pl.INT8)
        hidden_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quant_hidden(activation, hidden_payload, hidden_scale, active)
        down_acc = pl.create_tensor([capacity, hidden], dtype=pl.INT32)
        dot_down(hidden_payload, down_weight, down_acc, active)
        dequant_down(down_acc, hidden_scale, down_scale, output, active)
        return output

    return shared_expert


def make_routed_expert_w8a8(capacity: int, hidden: int, intermediate: int, clamp_limit: float = 10.0):
    """Alias the full expert kernel for one post-dispatch routed expert."""

    return make_shared_expert_w8a8(capacity, hidden, intermediate, clamp_limit)


def _make_router_gate(capacity, hidden, experts, input_dtype):
    """Reuse each FP32 weight tile across a block of source tokens."""
    token_tile = math.gcd(capacity, ROUTER_TOKEN_TILE)
    expert_blocks = experts // ROUTER_EXPERT_TILE
    needs_widen = input_dtype != pl.FP32
    widen = _router_make_cast_tile(1, ROUTER_D_TILE, input_dtype, pl.FP32)

    @pl.jit.inline
    def project(
        x: pl.Tensor[[capacity, hidden], input_dtype],
        gate_weight: pl.Tensor[[experts, hidden], pl.FP32],
        logits: pl.Tensor[[capacity, experts], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        block_count = ((active + token_tile - 1) // token_tile) * expert_blocks
        for worker in pl.spmd(pl.min(block_count, ROUTER_GATE_WORKERS), name_hint="a2_moe_gate_fp32"):
            for block in pl.range(worker, block_count, ROUTER_GATE_WORKERS):
                token = (block // expert_blocks) * token_tile
                expert = (block % expert_blocks) * ROUTER_EXPERT_TILE
                dot_sum = pl.full([token_tile, ROUTER_EXPERT_TILE], dtype=pl.FP32, value=0.0)
                for inner in pl.range(0, hidden, ROUTER_D_TILE):
                    weight = gate_weight[expert : expert + ROUTER_EXPERT_TILE, inner : inner + ROUTER_D_TILE]
                    for row in pl.unroll(token_tile):
                        source = x[token + row : token + row + 1, inner : inner + ROUTER_D_TILE]
                        if needs_widen:
                            source_storage = pl.create_tensor([1, ROUTER_D_TILE], dtype=input_dtype)
                            source_storage[:, :] = source
                            source_fp32 = widen(source_storage)
                        else:
                            source_fp32 = source
                        product = pl.col_expand_mul(weight, source_fp32)
                        partial = pl.reshape(pl.row_sum(product), [1, ROUTER_EXPERT_TILE])
                        dot_sum[row : row + 1, :] = pl.add(dot_sum[row : row + 1, :], partial)
                rows = pl.min(token_tile, active - token)
                logits[token : token + token_tile, expert : expert + ROUTER_EXPERT_TILE] = pl.set_validshape(
                    dot_sum, rows, ROUTER_EXPERT_TILE
                )
        return logits

    return project


def make_moe_router_inline(capacity, config, input_dtype=pl.BF16):
    """Build full-precision gate projection and 384-expert top-six routing."""
    hidden = config["hidden_size"]
    experts = config["n_routed_experts"]
    topk = config["num_experts_per_tok"]
    route_scale = config["routed_scaling_factor"]
    sentinel_base = config["image_sentinel_base_id"]
    sentinel_end = sentinel_base + 5
    controlled = bool(config.get("controlled_routing", False))
    if experts != 384 or topk != 6:
        raise ValueError("The A2 router requires 384 experts and six selected experts")
    if hidden % ROUTER_D_TILE:
        raise ValueError("The gate hidden size must be divisible by D_TILE")
    project = _make_router_gate(capacity, hidden, experts, input_dtype)

    @pl.jit.inline
    def moe_router_inline(
        x: pl.Tensor[[capacity, hidden], input_dtype],
        gate_weight: pl.Tensor[[experts, hidden], pl.FP32],
        text_bias: pl.Tensor[[experts], pl.FP32],
        vision_bias: pl.Tensor[[experts], pl.FP32],
        input_ids: pl.Tensor[[capacity], pl.INT64],
        logits: pl.Tensor[[capacity, experts], pl.FP32],
        route_ids: pl.Tensor[[capacity, topk], pl.INT32],
        route_weights: pl.Tensor[[capacity, topk], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        logits = project(x, gate_weight, logits, active)

        # The six-slot output is contiguous across each token block.
        flat_ids = pl.reshape(route_ids, [1, capacity * topk])
        flat_weights = pl.reshape(route_weights, [1, capacity * topk])
        num_blocks = (active + ROUTER_SELECT_TOKEN_TILE - 1) // ROUTER_SELECT_TOKEN_TILE
        workers = pl.min(num_blocks, ROUTER_SELECT_WORKERS)
        for worker in pl.spmd(workers, name_hint="a2_moe_route_select"):
            for block in pl.range(worker, num_blocks, workers):
                begin = block * ROUTER_SELECT_TOKEN_TILE
                rows = pl.min(ROUTER_SELECT_TOKEN_TILE, active - begin)
                gate_logits = pl.slice(logits, [ROUTER_SELECT_TOKEN_TILE, experts], [begin, 0], valid_shape=[rows, experts])
                gate_logits = pl.fillpad(gate_logits, pad_value=pl.PadValue.zero)
                positive = pl.maximum(gate_logits, 0.0)
                negative_abs = pl.neg(pl.abs(gate_logits))
                exponential = pl.exp(negative_abs)
                log_tail = pl.log(pl.add(exponential, 1.0))
                softplus_log = pl.add(positive, log_tail)
                negative_shift = pl.sub(pl.neg(gate_logits), 10.0)
                negative_mask = pl.minimum(pl.maximum(negative_shift, 0.0), 1.0)
                # The negative floor reuses exp(-abs(logit)) for logit < -10.
                negative_floor = pl.mul(negative_mask, exponential)
                softplus = pl.maximum(softplus_log, negative_floor)
                scores = pl.sqrt(softplus)
                index_range = pl.arange(0, [1, experts], dtype=pl.UINT32)
                if not controlled:
                    text_bias_row = pl.reshape(text_bias[:], [1, experts])
                    text_scores = pl.col_expand_add(scores, text_bias_row)
                ids_out = pl.create_tensor([1, ROUTER_SELECT_TOKEN_TILE * topk], dtype=pl.INT32)
                weights_out = pl.create_tensor([1, ROUTER_SELECT_TOKEN_TILE * topk], dtype=pl.FP32)
                for row in pl.range(rows):
                    token = begin + row
                    if not controlled:
                        biased_scores = text_scores[row : row + 1, :]
                        token_id = pl.read(input_ids, [token])
                        if token_id >= sentinel_base:
                            if token_id < sentinel_end:
                                score_row = scores[row : row + 1, :]
                                bias = pl.reshape(vision_bias[:], [1, experts])
                                biased_scores = pl.add(score_row, bias)
                        sorted_32 = pl.sort32(biased_scores, index_range)
                        sorted_64 = pl.mrgsort(sorted_32, block_len=64)
                        prefix = 2 * ROUTER_TOPK_TILE
                        first = sorted_64[:, :prefix]
                        second = sorted_64[:, 256 : 256 + prefix]
                        third = sorted_64[:, 512 : 512 + prefix]
                        sorted_pairs = pl.mrgsort(first, second, third)
                        top_pairs = sorted_pairs[:, : 2 * ROUTER_TOPK_TILE]
                        top_ids = pl.gather(top_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                    selected_ids = pl.array.create(topk, pl.INT32)
                    selected_scores = pl.create_tensor([1, ROUTER_TOPK_TILE], dtype=pl.FP32)
                    score_sum = pl.cast(0.0, pl.FP32)
                    for rank in pl.unroll(topk):
                        if controlled:
                            expert = pl.read(route_ids, [token, rank])
                        else:
                            expert = pl.read(top_ids, [0, rank])
                        score = pl.read(scores, [row, pl.cast(expert, pl.INDEX)])
                        selected_ids[rank] = expert
                        pl.write(selected_scores, [0, rank], score)
                        score_sum = score_sum + score
                    for rank in pl.unroll(topk):
                        score = pl.read(selected_scores, [0, rank])
                        normalized = (score / score_sum) * route_scale
                        pl.write(ids_out, [0, row * topk + rank], selected_ids[rank])
                        pl.write(weights_out, [0, row * topk + rank], normalized)
                offset = begin * topk
                flat_ids[:, offset : offset + ROUTER_SELECT_TOKEN_TILE * topk] = pl.set_validshape(ids_out, 1, rows * topk)
                flat_weights[:, offset : offset + ROUTER_SELECT_TOKEN_TILE * topk] = pl.set_validshape(
                    weights_out, 1, rows * topk
                )
        return logits, route_ids, route_weights

    return moe_router_inline


def make_moe_router(capacity, config, input_dtype=pl.BF16):
    """Build the standalone MoE router entry with untouched inactive rows."""
    hidden = config["hidden_size"]
    experts = config["n_routed_experts"]
    topk = config["num_experts_per_tok"]
    compute = make_moe_router_inline(capacity, config, input_dtype)

    @pl.jit
    def moe_router(
        x: pl.InOut[pl.Tensor[[capacity, hidden], input_dtype]],
        gate_weight: pl.Tensor[[experts, hidden], pl.FP32],
        text_bias: pl.Tensor[[experts], pl.FP32],
        vision_bias: pl.Tensor[[experts], pl.FP32],
        input_ids: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        logits: pl.InOut[pl.Tensor[[capacity, experts], pl.FP32]],
        route_ids: pl.InOut[pl.Tensor[[capacity, topk], pl.INT32]],
        route_weights: pl.InOut[pl.Tensor[[capacity, topk], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        compute(x, gate_weight, text_bias, vision_bias, input_ids, logits, route_ids, route_weights, active)
        return x, input_ids, logits, route_ids, route_weights

    return moe_router


def make_ep_dispatch_pack(capacity, hidden, experts, topk, world):
    """Pack source token/top-k rows into contiguous destination buckets."""
    routes = capacity * topk
    local_experts = experts // world
    packet_rows = world * routes

    @pl.jit
    def ep_dispatch_pack(
        x: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        route_ids: pl.Tensor[[capacity, topk], pl.INT32],
        sent_routes: pl.InOut[pl.Tensor[[packet_rows, 4], pl.INT64]],
        sent_hidden: pl.InOut[pl.Tensor[[packet_rows, hidden], pl.BF16]],
        sent_counts: pl.InOut[pl.Tensor[[world, DISPATCH_MAP_PAD], pl.INT32]],
        source_map: pl.InOut[pl.Tensor[[routes, DISPATCH_SOURCE_MAP_PAD], pl.INT32]],
        active: pl.Scalar[pl.INT32],
        source_rank: pl.Scalar[pl.INT32],
    ):
        # Each route map row and each pair of count rows occupy whole scalar-store cache lines.
        for peer_pair in pl.spmd((world + 1) // 2, name_hint="moe_ep_pack_metadata"):
            first_peer = peer_pair * 2
            bucket_counts = pl.array.create(2, pl.INT32)
            bucket_counts[0] = pl.cast(0, pl.INT32)
            bucket_counts[1] = pl.cast(0, pl.INT32)
            for token in pl.range(active):
                for slot in pl.range(topk):
                    expert = pl.read(route_ids, [token, slot])
                    peer = pl.cast(expert // local_experts, pl.INDEX)
                    lane = peer - first_peer
                    if lane >= 0 and lane < 2 and peer < world:
                        count = bucket_counts[lane]
                        packet_index = peer * routes + pl.cast(count, pl.INDEX)
                        pl.write(sent_routes, [packet_index, 0], pl.cast(source_rank, pl.INT64))
                        pl.write(sent_routes, [packet_index, 1], pl.cast(token, pl.INT64))
                        pl.write(sent_routes, [packet_index, 2], pl.cast(slot, pl.INT64))
                        pl.write(sent_routes, [packet_index, 3], pl.cast(expert, pl.INT64))
                        original_route = token * topk + slot
                        pl.write(source_map, [original_route, 0], pl.cast(packet_index, pl.INT32))
                        bucket_counts[lane] = count + 1
            pl.write(sent_counts, [first_peer, 0], bucket_counts[0])
            if first_peer + 1 < world:
                pl.write(sent_counts, [first_peer + 1, 0], bucket_counts[1])

        for token in pl.spmd(active, name_hint="moe_ep_pack_hidden"):
            source_row = x[token : token + 1, :]
            first_route = token * topk
            for route in pl.range(first_route, first_route + topk):
                target = pl.cast(pl.read(source_map, [route, 0]), pl.INDEX)
                sent_hidden[target : target + 1, :] = source_row
        return x, sent_routes, sent_hidden, sent_counts, source_map

    return ep_dispatch_pack


def make_ep_dispatch_unpack(capacity, hidden, experts, topk, world, expert_capacity=None):
    """Unpack received buckets into owned expert slabs and compact audit payloads."""
    routes = capacity * topk
    local_experts = experts // world
    packet_rows = world * routes
    if expert_capacity is None:
        expert_capacity = ((capacity * world + 15) // 16) * 16
    slab_rows = local_experts * expert_capacity
    copy_workers = min(routes, 8)

    @pl.jit
    def ep_dispatch_unpack(
        packet_routes: pl.Tensor[[packet_rows, 4], pl.INT64],
        packet_hidden: pl.InOut[pl.Tensor[[packet_rows, hidden], pl.BF16]],
        packet_counts: pl.Tensor[[world, DISPATCH_MAP_PAD], pl.INT32],
        recv_routes: pl.InOut[pl.Tensor[[packet_rows, 4], pl.INT64]],
        recv_hidden: pl.InOut[pl.Tensor[[packet_rows, hidden], pl.BF16]],
        expert_x: pl.InOut[pl.Tensor[[slab_rows, hidden], pl.BF16]],
        expert_counts: pl.InOut[pl.Tensor[[local_experts], pl.INT64]],
        packet_map: pl.InOut[pl.Tensor[[packet_rows, DISPATCH_MAP_PAD], pl.INT32]],
        received_total: pl.InOut[pl.Tensor[[DISPATCH_MAP_PAD], pl.INT32]],
        owner_rank: pl.Scalar[pl.INT32],
    ):
        for _block in pl.spmd(1, name_hint="moe_ep_unpack_metadata"):
            expert_begin = owner_rank * local_experts
            counts = pl.array.create(local_experts, pl.INT32)
            starts = pl.array.create(local_experts, pl.INT32)
            for local in pl.range(local_experts):
                counts[local] = pl.cast(0, pl.INT32)
            for peer in pl.range(world):
                peer_count = pl.cast(pl.read(packet_counts, [peer, 0]), pl.INDEX)
                for row in pl.range(peer_count):
                    packet = peer * routes + row
                    expert = pl.read(packet_routes, [packet, 3])
                    if expert >= expert_begin and expert < expert_begin + local_experts:
                        local = pl.cast(expert - expert_begin, pl.INDEX)
                        counts[local] = counts[local] + 1

            compact_begin = pl.cast(0, pl.INDEX)
            for local in pl.range(local_experts):
                starts[local] = pl.cast(compact_begin, pl.INT32)
                count = counts[local]
                pl.write(expert_counts, [local], pl.cast(count, pl.INT64))
                compact_begin = compact_begin + pl.cast(count, pl.INDEX)
                counts[local] = pl.cast(0, pl.INT32)
            pl.write(received_total, [0], pl.cast(compact_begin, pl.INT32))

            for peer in pl.range(world):
                peer_count = pl.cast(pl.read(packet_counts, [peer, 0]), pl.INDEX)
                for row in pl.range(peer_count):
                    packet = peer * routes + row
                    expert = pl.read(packet_routes, [packet, 3])
                    if expert >= expert_begin and expert < expert_begin + local_experts:
                        local = pl.cast(expert - expert_begin, pl.INDEX)
                        index = pl.cast(counts[local], pl.INDEX)
                        compact = pl.cast(starts[local], pl.INDEX) + index
                        target = local * expert_capacity + index
                        pl.write(packet_map, [packet, 0], pl.cast(target, pl.INT32))
                        pl.write(packet_map, [packet, 1], pl.cast(compact, pl.INT32))
                        counts[local] = counts[local] + 1

        for block in pl.spmd(world * copy_workers, name_hint="moe_ep_unpack_hidden"):
            peer = block // copy_workers
            lane = block % copy_workers
            bucket_rows = pl.cast(pl.read(packet_counts, [peer, 0]), pl.INDEX)
            for row in pl.range(lane, bucket_rows, copy_workers):
                packet = peer * routes + row
                target = pl.cast(pl.read(packet_map, [packet, 0]), pl.INDEX)
                compact = pl.cast(pl.read(packet_map, [packet, 1]), pl.INDEX)
                metadata = packet_routes[packet : packet + 1, :]
                recv_routes[compact : compact + 1, :] = metadata
                value = packet_hidden[packet : packet + 1, :]
                expert_x[target : target + 1, :] = value
                recv_hidden[compact : compact + 1, :] = value
        return packet_hidden, recv_routes, recv_hidden, expert_x, expert_counts, packet_map, received_total

    return ep_dispatch_unpack


def make_ep_return_pack(capacity, hidden, experts, topk, world, expert_capacity=None):
    """Gather owned expert results into each original source's return bucket."""
    routes = capacity * topk
    packet_rows = world * routes
    if expert_capacity is None:
        expert_capacity = ((capacity * world + 15) // 16) * 16
    slab_rows = (experts // world) * expert_capacity
    copy_workers = min(routes, 8)

    @pl.jit
    def ep_return_pack(
        expert_y: pl.Tensor[[slab_rows, hidden], pl.BF16],
        packet_counts: pl.Tensor[[world, DISPATCH_MAP_PAD], pl.INT32],
        packet_map: pl.Tensor[[packet_rows, DISPATCH_MAP_PAD], pl.INT32],
        sent_hidden: pl.InOut[pl.Tensor[[packet_rows, hidden], pl.BF16]],
    ):
        for block in pl.spmd(world * copy_workers, name_hint="moe_ep_return_pack"):
            peer = block // copy_workers
            lane = block % copy_workers
            bucket_rows = pl.cast(pl.read(packet_counts, [peer, 0]), pl.INDEX)
            for row in pl.range(lane, bucket_rows, copy_workers):
                packet = peer * routes + row
                source = pl.cast(pl.read(packet_map, [packet, 0]), pl.INDEX)
                sent_hidden[packet : packet + 1, :] = expert_y[source : source + 1, :]
        return sent_hidden

    return ep_return_pack


def make_dispatch_layout(tokens):
    """Map native count-mode dispatch into the full384 expert ABI."""
    routes = tokens * DISPATCH_TOPK

    @pl.jit
    def dispatch_layout(
        payload: pl.Tensor[[routes, DISPATCH_D], pl.INT8],
        scales: pl.Tensor[[routes], pl.FP32],
        counts: pl.Tensor[[DISPATCH_EXPERTS], pl.INT64],
        expanded: pl.Tensor[[routes], pl.INT32],
        weights: pl.Tensor[[tokens, DISPATCH_TOPK], pl.FP32],
        recv_x: pl.InOut[pl.Tensor[[DISPATCH_EXPERTS, DISPATCH_RECV_MAX, DISPATCH_D], pl.INT8]],
        recv_scale: pl.InOut[pl.Tensor[[DISPATCH_EXPERTS, DISPATCH_RECV_MAX], pl.FP32]],
        recv_weights: pl.InOut[pl.Tensor[[DISPATCH_EXPERTS, DISPATCH_RECV_MAX], pl.FP32]],
        recv_count: pl.InOut[pl.Tensor[[DISPATCH_EXPERTS, 1], pl.INT32]],
        route_map: pl.InOut[pl.Tensor[[routes, DISPATCH_MAP_PAD], pl.INT32]],
    ):
        # Serialize narrow scalar stores that can share a cache line.
        for _block in pl.spmd(1, name_hint="a2_moe_dispatch_metadata"):
            begin = pl.cast(0, pl.INT32)
            for expert in pl.range(DISPATCH_EXPERTS):
                count = pl.cast(pl.read(counts, [expert]), pl.INT32)
                pl.write(recv_count, [expert, 0], count)
                for original in pl.range(routes):
                    dispatched = pl.read(expanded, [original])
                    if dispatched >= begin and dispatched < begin + count:
                        row = dispatched - begin
                        route_weight = pl.read(weights, [original // DISPATCH_TOPK, original % DISPATCH_TOPK])
                        token_scale = pl.read(scales, [dispatched])
                        pl.write(recv_scale, [expert, row], token_scale)
                        pl.write(recv_weights, [expert, row], route_weight)
                        target = pl.cast(expert * DISPATCH_RECV_MAX, pl.INT32) + row
                        pl.write(route_map, [original, 0], target)
                begin = begin + count

        recv_flat = pl.reshape(recv_x, [DISPATCH_EXPERTS * DISPATCH_RECV_MAX, DISPATCH_D])
        for block in pl.spmd(routes * (DISPATCH_D // DISPATCH_D_TILE), name_hint="a2_moe_dispatch_payload"):
            original = block // (DISPATCH_D // DISPATCH_D_TILE)
            column = block % (DISPATCH_D // DISPATCH_D_TILE) * DISPATCH_D_TILE
            dispatched = pl.read(expanded, [original])
            target = pl.read(route_map, [original, 0])
            value = payload[dispatched : dispatched + 1, column : column + DISPATCH_D_TILE]
            recv_flat[target : target + 1, column : column + DISPATCH_D_TILE] = value
        return recv_x, recv_scale, recv_weights, recv_count, route_map

    return dispatch_layout


def make_route_restore(tokens):
    """Gather every actual weighted expert result in original token/top6 order."""
    routes = tokens * DISPATCH_TOPK

    @pl.jit
    def route_restore(
        recv_y: pl.Tensor[[DISPATCH_EXPERTS, DISPATCH_RECV_MAX, DISPATCH_D], pl.FP32],
        route_map: pl.Tensor[[routes, DISPATCH_MAP_PAD], pl.INT32],
        dispatch_y: pl.Out[pl.Tensor[[routes, DISPATCH_D], pl.FP32]],
    ):
        recv_flat = pl.reshape(recv_y, [DISPATCH_EXPERTS * DISPATCH_RECV_MAX, DISPATCH_D])
        for block in pl.spmd(routes * (DISPATCH_D // DISPATCH_D_TILE), name_hint="a2_moe_route_restore"):
            original = block // (DISPATCH_D // DISPATCH_D_TILE)
            column = block % (DISPATCH_D // DISPATCH_D_TILE) * DISPATCH_D_TILE
            source = pl.read(route_map, [original, 0])
            value = recv_flat[source : source + 1, column : column + DISPATCH_D_TILE]
            dispatch_y[original : original + 1, column : column + DISPATCH_D_TILE] = value
        return dispatch_y

    return route_restore


def make_moe_combine(capacity, hidden=COMBINE_D, topk=COMBINE_TOPK, world=2, *, weight_dtype=pl.FP32):
    """Restore returned token/top-k rows, weight experts, and add the shared result."""
    routes = capacity * topk
    packet_rows = world * routes
    restore_token_tile = min(8, capacity)
    restore_route_tile = restore_token_tile * topk
    if weight_dtype not in (pl.FP32, pl.BF16):
        raise ValueError("Route weights require FP32 or BF16 multiplication boundaries")
    round_weights = weight_dtype == pl.BF16

    @pl.jit
    def moe_combine(
        packet_routes: pl.Tensor[[packet_rows, 4], pl.INT64],
        packet_hidden: pl.Tensor[[packet_rows, hidden], pl.BF16],
        packet_counts: pl.Tensor[[world, 8], pl.INT32],
        route_weights: pl.Tensor[[capacity, topk], pl.FP32],
        shared: pl.Tensor[[capacity, hidden], pl.BF16],
        routed: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        returned_routes: pl.InOut[pl.Tensor[[routes, 4], pl.INT64]],
        active: pl.Scalar[pl.INT32],
    ):
        rounded_weights = pl.create_tensor([capacity, 16], dtype=pl.FP32)
        for block in pl.spmd((active + 7) // 8, name_hint="moe_route_weight_cast"):
            token_begin = block * 8
            rows = pl.min(8, active - token_begin)
            source = pl.slice(route_weights, [8, 16], [token_begin, 0], valid_shape=[rows, topk])
            padded = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 8, 16)
            widened = padded
            if round_weights:
                rounded = pl.cast(padded, pl.BF16, mode="rint")
                widened = pl.cast(rounded, pl.FP32)
            rounded_weights[token_begin : token_begin + 8, :] = pl.set_validshape(widened, rows, 16)
        route_map = pl.create_tensor([routes, 8], dtype=pl.INT32)
        for block in pl.spmd((capacity + restore_token_tile - 1) // restore_token_tile, name_hint="moe_ep_restore_metadata"):
            route_begin = block * restore_route_tile
            route_count = pl.min(restore_route_tile, routes - route_begin)
            mapping = pl.tile.full([restore_route_tile, 8], dtype=pl.INT32, value=-1)
            restored = pl.tile.load(returned_routes, [route_begin, 0], [restore_route_tile, 4],
                                    valid_shape=[route_count, 4])
            for peer in pl.range(world):
                count = pl.cast(pl.read(packet_counts, [peer, 0]), pl.INDEX)
                if count > 0:
                    peer_begin = peer * routes
                    metadata = pl.tile.load(packet_routes, [peer_begin, 0], [routes, 4], valid_shape=[count, 4])
                    for row in pl.range(count):
                        source_token = pl.cast(pl.tile.read(metadata, [row, 1]), pl.INDEX)
                        token_route = source_token * topk - route_begin
                        if token_route >= 0 and token_route < route_count:
                            source_slot = pl.cast(pl.tile.read(metadata, [row, 2]), pl.INDEX)
                            original = token_route + source_slot
                            pl.tile.write(mapping, [original, 0], pl.cast(peer_begin + row, pl.INT32))
                            for field in pl.range(4):
                                field_value = pl.tile.read(metadata, [row, field])
                                pl.tile.write(restored, [original, field], field_value)
            valid_mapping = pl.set_validshape(mapping, route_count, 8)
            pl.tile.store(valid_mapping, [route_begin, 0], route_map)
            pl.tile.store(restored, [route_begin, 0], returned_routes)

        for token in pl.spmd(active, name_hint="moe_ep_source_combine"):
            acc = pl.full([1, hidden], dtype=pl.FP32, value=0.0)
            for slot in pl.range(topk):
                route_row = token * topk + slot
                packet = pl.cast(pl.read(route_map, [route_row, 0]), pl.INDEX)
                if packet >= 0:
                    value = pl.cast(packet_hidden[packet : packet + 1, :], pl.FP32)
                    weight = pl.read(rounded_weights, [token, slot])
                    weighted = pl.mul(value, weight)
                    acc = pl.add(acc, weighted)
            narrowed = pl.cast(acc, pl.BF16, mode="rint")
            routed[token : token + 1, :] = narrowed
            shared_row = pl.cast(shared[token : token + 1, :], pl.FP32)
            total = pl.add(pl.cast(narrowed, pl.FP32), shared_row)
            output[token : token + 1, :] = pl.cast(total, pl.BF16, mode="rint")
        return routed, output, returned_routes

    return moe_combine


def make_moe_combine_from_dispatch(capacity, hidden, topk, world):
    """Combine returned buckets that retain their source dispatch order."""
    routes = capacity * topk
    packet_rows = world * routes
    weight_tile = ((topk + 15) // 16) * 16

    @pl.jit
    def moe_combine_from_dispatch(
        packet_routes: pl.Tensor[[packet_rows, 4], pl.INT64],
        packet_hidden: pl.Tensor[[packet_rows, hidden], pl.BF16],
        source_map: pl.Tensor[[routes, DISPATCH_SOURCE_MAP_PAD], pl.INT32],
        route_weights: pl.Tensor[[capacity, topk], pl.FP32],
        shared: pl.Tensor[[capacity, hidden], pl.BF16],
        routed: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        returned_routes: pl.InOut[pl.Tensor[[routes, 4], pl.INT64]],
        active: pl.Scalar[pl.INT32],
        valid: pl.Scalar[pl.INT32],
    ):
        for token in pl.spmd(active, name_hint="moe_ep_source_combine_indexed"):
            acc = pl.full([1, hidden], dtype=pl.FP32, value=0.0)
            if token < valid:
                source = pl.tile.load(route_weights, [token, 0], [1, weight_tile], valid_shape=[1, topk])
                filled = pl.tile.fillpad(source, pad_value=pl.PadValue.zero)
                padded = pl.tile.set_validshape(filled, 1, weight_tile)
                rounded = pl.tile.cast(padded, pl.BF16, mode="rint")
                weights = pl.tile.cast(rounded, pl.FP32)
                for slot in pl.range(topk):
                    original = token * topk + slot
                    packet = pl.cast(pl.read(source_map, [original, 0]), pl.INDEX)
                    metadata = packet_routes[packet : packet + 1, :]
                    returned_routes[original : original + 1, :] = metadata
                    value = pl.cast(packet_hidden[packet : packet + 1, :], pl.FP32)
                    weight = pl.tile.read(weights, [0, slot])
                    weighted = pl.mul(value, weight)
                    acc = pl.add(acc, weighted)
            narrowed = pl.cast(acc, pl.BF16, mode="rint")
            routed[token : token + 1, :] = narrowed
            shared_row = pl.cast(shared[token : token + 1, :], pl.FP32)
            total = pl.add(pl.cast(narrowed, pl.FP32), shared_row)
            output[token : token + 1, :] = pl.cast(total, pl.BF16, mode="rint")
        return routed, output, returned_routes

    return moe_combine_from_dispatch


def _make_shared_activation(
    capacity, intermediate, limit, fused_activation, fused_channel_dtype=pl.FP16, token_first=False,
    *, quantize_output=False, fused_requantize=False,
):
    """Select fused channel precision or standalone BF16 activation rounding."""
    width = intermediate * 2
    tile = intermediate if quantize_output else 256
    row_tile = 1 if quantize_output else SHARED_VEC_M_TILE
    dequantize = _shared_make_dequantize(capacity, width)

    if not fused_activation:

        @pl.jit.inline
        def bf16_activation(
            accumulator: pl.Tensor[[capacity, width], pl.INT32],
            channels: pl.Tensor[[width], pl.FP32],
            scales: pl.Tensor[[capacity, 8], pl.FP32],
            output: pl.Tensor[[capacity, intermediate], pl.BF16],
            active: pl.Scalar[pl.INT32],
        ):
            gate_up = pl.create_tensor([capacity, width], dtype=pl.BF16)
            dequantize(accumulator, channels, scales, gate_up, active)
            row_blocks = (active + SHARED_VEC_M_TILE - 1) // SHARED_VEC_M_TILE
            for block in pl.spmd(row_blocks * (intermediate // tile), name_hint="a2_moe_swiglu"):
                token = block // (intermediate // tile) * SHARED_VEC_M_TILE
                column = block % (intermediate // tile) * tile
                rows = pl.min(SHARED_VEC_M_TILE, active - token)
                up_column = intermediate + column
                gate_bf16 = pl.slice(
                    gate_up, [SHARED_VEC_M_TILE, tile], [token, column], valid_shape=[rows, tile]
                )
                up_bf16 = pl.slice(
                    gate_up, [SHARED_VEC_M_TILE, tile], [token, up_column], valid_shape=[rows, tile]
                )
                gate_bf16 = pl.set_validshape(
                    pl.fillpad(gate_bf16, pad_value=pl.PadValue.zero), SHARED_VEC_M_TILE, tile
                )
                up_bf16 = pl.set_validshape(
                    pl.fillpad(up_bf16, pad_value=pl.PadValue.zero), SHARED_VEC_M_TILE, tile
                )
                gate = pl.minimum(pl.cast(gate_bf16, pl.FP32), limit)
                up = pl.maximum(pl.minimum(pl.cast(up_bf16, pl.FP32), limit), -limit)
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate)), 1.0))
                sigmoid_bf16 = pl.cast(sigmoid, pl.BF16, mode="rint")
                silu_bf16 = pl.cast(pl.mul(gate, pl.cast(sigmoid_bf16, pl.FP32)), pl.BF16, mode="rint")
                activated = pl.mul(pl.cast(silu_bf16, pl.FP32), up)
                narrowed = pl.cast(activated, pl.BF16, mode="rint")
                output[token : token + SHARED_VEC_M_TILE, column : column + tile] = pl.set_validshape(
                    narrowed, rows, tile
                )
            return output

        return bf16_activation

    fp16_channel = fused_channel_dtype == pl.FP16
    output_dtype = pl.INT8 if quantize_output else pl.FP32
    quantize_row = _shared_make_quantize_row(intermediate, fused_requantize=fused_requantize)

    @pl.jit.inline
    def fp32_activation(
        accumulator: pl.Tensor[[capacity, width], pl.INT32],
        channels: pl.Tensor[[width], pl.FP32],
        scales: pl.Tensor[[capacity, 8], pl.FP32],
        output: pl.Tensor[[capacity, intermediate], output_dtype],
        output_scale: pl.Tensor[[capacity, 8], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        row_blocks = (active + row_tile - 1) // row_tile
        blocks = row_blocks * (intermediate // tile)
        workers = pl.min(blocks, 48)
        for worker in pl.spmd(workers, name_hint="a2_moe_swiglu_fused"):
            for block in pl.range(worker, blocks, workers):
                token = block // (intermediate // tile) * row_tile
                column = block % (intermediate // tile) * tile
                rows = pl.min(row_tile, active - token)
                up_column = intermediate + column
                gate_i32 = pl.slice(
                    accumulator, [row_tile, tile], [token, column], valid_shape=[rows, tile]
                )
                up_i32 = pl.slice(
                    accumulator, [row_tile, tile], [token, up_column], valid_shape=[rows, tile]
                )
                gate_channels = pl.reshape(channels[column : column + tile], [1, tile])
                up_channels = pl.reshape(channels[up_column : up_column + tile], [1, tile])
                if token_first:
                    if quantize_output:
                        scalar_scale = pl.read(scales, [token, 0])
                        gate_token = pl.mul(pl.cast(gate_i32, pl.FP32), scalar_scale)
                        up_token = pl.mul(pl.cast(up_i32, pl.FP32), scalar_scale)
                    else:
                        row_scale = pl.row_min(scales[token : token + row_tile, :])
                        gate_token = pl.row_expand_mul(pl.cast(gate_i32, pl.FP32), row_scale)
                        up_token = pl.row_expand_mul(pl.cast(up_i32, pl.FP32), row_scale)
                    gate = pl.col_expand_mul(gate_token, gate_channels)
                    up = pl.col_expand_mul(up_token, up_channels)
                else:
                    gate_channel = pl.col_expand_mul(pl.cast(gate_i32, pl.FP32), gate_channels)
                    up_channel = pl.col_expand_mul(pl.cast(up_i32, pl.FP32), up_channels)
                    if fp16_channel:
                        gate_half = pl.cast(gate_channel, pl.FP16, mode="rint")
                        up_half = pl.cast(up_channel, pl.FP16, mode="rint")
                        gate_channel = pl.cast(gate_half, pl.FP32)
                        up_channel = pl.cast(up_half, pl.FP32)
                    if quantize_output:
                        scalar_scale = pl.read(scales, [token, 0])
                        gate = pl.mul(gate_channel, scalar_scale)
                        up = pl.mul(up_channel, scalar_scale)
                    else:
                        row_scale = pl.row_min(scales[token : token + row_tile, :])
                        gate = pl.row_expand_mul(gate_channel, row_scale)
                        up = pl.row_expand_mul(up_channel, row_scale)
                gate = pl.minimum(gate, limit)
                up = pl.maximum(pl.minimum(up, limit), -limit)
                denominator = pl.add(pl.exp(pl.neg(gate)), 1.0)
                silu = pl.div(gate, denominator)
                activated = pl.mul(silu, up)
                if quantize_output:
                    quant_value = pl.create_tensor([1, intermediate], dtype=pl.FP32)
                    quant_value[:, :] = activated
                    codes, activation_scale = quantize_row(quant_value)
                    output[token : token + 1, :] = codes
                    output_scale[token : token + 1, :] = activation_scale
                else:
                    output[token : token + row_tile, column : column + tile] = pl.set_validshape(activated, rows, tile)
        return output

    return fp32_activation


def _make_shared_moe_partition(
    capacity, config, *, fused_activation=False, fused_channel_dtype=pl.FP16,
    token_first=False, fused_requantize=False,
):
    """Build gate/up, clamped SwiGLU and down with selected fused channel precision."""
    if fused_channel_dtype not in (pl.FP16, pl.FP32):
        raise ValueError("Fused channel scaling requires FP16 or FP32")
    hidden = config["hidden_size"]
    intermediate = config["moe_intermediate_size"]
    gate_up_width = intermediate * 2
    limit = config["swiglu_limit"]
    activation_dtype = pl.FP32 if fused_activation else pl.BF16
    activate = _make_shared_activation(
        capacity, intermediate, limit, fused_activation, fused_channel_dtype, token_first,
        quantize_output=fused_activation, fused_requantize=fused_requantize,
    )

    quant_hidden = _shared_make_quantize(
        capacity, intermediate, activation_dtype, fused_requantize=fused_requantize, whole_row=True,
    )
    dot_gate_up = _shared_make_dot(capacity, hidden, gate_up_width, full_k_reuse=True, column_group=3, row_tile=32)
    down_row_tile = 256 if intermediate <= 768 else 16
    down_stages = 1 if down_row_tile > 16 else 2
    dot_down = _shared_make_dot(
        capacity, intermediate, hidden, full_k_reuse=True, column_group=4,
        row_tile=down_row_tile, column_stages=down_stages,
    )
    channel_dtype = fused_channel_dtype if fused_activation else pl.FP32
    dequant_down = _shared_make_dequantize(capacity, hidden, channel_dtype=channel_dtype)

    @pl.jit.inline
    def shared_moe_partition(
        x_payload: pl.Tensor[[capacity, hidden], pl.INT8],
        x_scale: pl.Tensor[[capacity, 8], pl.FP32],
        gate_up_weight: pl.Tensor[[hidden * (gate_up_width // 64), 64], pl.INT8],
        gate_up_scale: pl.Tensor[[gate_up_width], pl.FP32],
        down_weight: pl.Tensor[[intermediate * (hidden // 64), 64], pl.INT8],
        down_scale: pl.Tensor[[hidden], pl.FP32],
        output: pl.Tensor[[capacity, hidden], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        gate_up_acc = pl.create_tensor([capacity, gate_up_width], dtype=pl.INT32)
        dot_gate_up(x_payload, gate_up_weight, gate_up_acc, active)
        hidden_payload = pl.create_tensor([capacity, intermediate], dtype=pl.INT8)
        hidden_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        if fused_activation:
            activate(gate_up_acc, gate_up_scale, x_scale, hidden_payload, hidden_scale, active)
        else:
            activation = pl.create_tensor([capacity, intermediate], dtype=activation_dtype)
            activate(gate_up_acc, gate_up_scale, x_scale, activation, active)
            quant_hidden(activation, hidden_payload, hidden_scale, active)
        down_acc = pl.create_tensor([capacity, hidden], dtype=pl.INT32)
        dot_down(hidden_payload, down_weight, down_acc, active)
        dequant_down(down_acc, down_scale, hidden_scale, output, active)
        return output

    return shared_moe_partition


def make_shared_moe_w8a8_inline(capacity, config, *, fused_activation=False, fused_channel_dtype=pl.FP16,
                                token_first=False, fused_requantize=False):
    """Build a single shared expert with unpartitioned weights."""
    hidden = config["hidden_size"]
    intermediate = config["moe_intermediate_size"]
    quantize = make_prefill_projection_quantize(capacity, hidden)
    partition = _make_shared_moe_partition(capacity, config, fused_activation=fused_activation,
                                           fused_channel_dtype=fused_channel_dtype, token_first=token_first,
                                           fused_requantize=fused_requantize)

    @pl.jit.inline
    def shared_moe_w8a8_inline(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        gate_up_weight: pl.Tensor[[hidden * (2 * intermediate // 64), 64], pl.INT8],
        gate_up_scale: pl.Tensor[[2 * intermediate], pl.FP32],
        down_weight: pl.Tensor[[intermediate * (hidden // 64), 64], pl.INT8],
        down_scale: pl.Tensor[[hidden], pl.FP32],
        output: pl.Tensor[[capacity, hidden], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        payload = pl.create_tensor([capacity, hidden], dtype=pl.INT8)
        scales = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize(x, payload, scales, active)
        partition(payload, scales, gate_up_weight, gate_up_scale, down_weight, down_scale, output, active)
        return output

    return shared_moe_w8a8_inline


def make_shared_moe_w8a8(
    capacity, config, *, fused_activation=False, fused_channel_dtype=pl.FP16,
    token_first=False, fused_requantize=False, partitions=1,
):
    """Build a standalone JIT kernel for the shared expert."""
    hidden = config["hidden_size"]
    if partitions not in (1, 2, 4) or config["moe_intermediate_size"] % partitions:
        raise ValueError("Shared expert partitions must divide the intermediate width")
    local_width = config["moe_intermediate_size"] // partitions
    intermediate = (local_width + 255) // 256 * 256
    local_config = dict(config, moe_intermediate_size=intermediate)
    gate_up_width = intermediate * 2
    quantize = make_prefill_projection_quantize(capacity, hidden)
    compute = _make_shared_moe_partition(
        capacity,
        local_config,
        fused_activation=fused_activation,
        fused_channel_dtype=fused_channel_dtype,
        token_first=token_first,
        fused_requantize=fused_requantize,
    )

    @pl.jit
    def shared_moe_w8a8(
        x: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        gate_up_weight: pl.Tensor[[partitions * hidden * (gate_up_width // 64), 64], pl.INT8],
        gate_up_scale: pl.Tensor[[partitions * gate_up_width], pl.FP32],
        down_weight: pl.Tensor[[partitions * intermediate * (hidden // 64), 64], pl.INT8],
        down_scale: pl.Tensor[[hidden], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        payload = pl.create_tensor([capacity, hidden], dtype=pl.INT8)
        input_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize(x, payload, input_scale, active)
        if partitions == 1:
            compute(payload, input_scale, gate_up_weight, gate_up_scale, down_weight, down_scale, output, active)
        else:
            partials = pl.create_tensor([partitions * capacity, hidden], dtype=pl.BF16)
            for peer in pl.unroll(partitions):
                gate_begin = peer * hidden * (gate_up_width // 64)
                gate_end = gate_begin + hidden * (gate_up_width // 64)
                down_begin = peer * intermediate * (hidden // 64)
                down_end = down_begin + intermediate * (hidden // 64)
                local_gate = gate_up_weight[gate_begin : gate_end, :]
                local_scale = gate_up_scale[peer * gate_up_width : (peer + 1) * gate_up_width]
                local_down = down_weight[down_begin : down_end, :]
                local_output = partials[peer * capacity : (peer + 1) * capacity, :]
                compute(payload, input_scale, local_gate, local_scale, local_down, down_scale, local_output, active)
            for block in pl.spmd(((active + 7) // 8) * (hidden // 256), name_hint="shared_tp_reduce"):
                token = block // (hidden // 256) * 8
                column = block % (hidden // 256) * 256
                rows = pl.min(8, active - token)
                a = pl.cast(partials[token : token + 8, column : column + 256], pl.FP32)
                b = pl.cast(partials[capacity + token : capacity + token + 8, column : column + 256], pl.FP32)
                pair = pl.cast(pl.add(a, b), pl.BF16, mode="rint")
                if partitions == 4:
                    third = 2 * capacity + token
                    fourth = 3 * capacity + token
                    c = pl.cast(partials[third : third + 8, column : column + 256], pl.FP32)
                    d = pl.cast(partials[fourth : fourth + 8, column : column + 256], pl.FP32)
                    other = pl.cast(pl.add(c, d), pl.BF16, mode="rint")
                    total = pl.add(pl.cast(pair, pl.FP32), pl.cast(other, pl.FP32))
                    pair = pl.cast(total, pl.BF16, mode="rint")
                output[token : token + 8, column : column + 256] = pl.set_validshape(pair, rows, 256)
        return x, output

    return shared_moe_w8a8


def prepare_ecotech_shared_weights(gate, up, down):
    """Prepare the original CPU NK weight/F32 scale/offset triples."""
    if any(len(values) != 3 for values in (gate, up, down)):
        raise ValueError("Require weight, scale and offset for every projection")
    if gate[0].shape != up[0].shape or gate[0].ndim != 2:
        raise ValueError("Gate and up require the same local NK shape")
    intermediate, hidden = gate[0].shape
    if down[0].shape != (hidden, intermediate):
        raise ValueError("Down must consume the complete local intermediate")
    merged = tuple(torch.cat([left, right], dim=0) for left, right in zip(gate, up))
    gate_weight, gate_scale = _ecotech_shared_prepare_ecotech_dynamic_weight(*merged)
    down_weight, down_scale = _ecotech_shared_prepare_ecotech_tail_weight(*down)
    return gate_weight, gate_scale, down_weight, down_scale


def ecotech_shared_make_shared_activation(capacity, intermediate, padded_intermediate, limit):
    width = intermediate * 2

    @pl.jit.inline
    def activate(
        accumulator: pl.Tensor[[capacity, width], pl.INT32],
        channels: pl.Tensor[[width], pl.FP32],
        scales: pl.Tensor[[capacity, 8], pl.FP32],
        output: pl.Tensor[[capacity, padded_intermediate], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(((active + 7) // 8) * (intermediate // 32), name_hint="ecotech_shared_swiglu"):
            token = block // (intermediate // 32) * 8
            column = block % (intermediate // 32) * 32
            up_column = intermediate + column
            rows = pl.min(8, active - token)
            gate_i32 = pl.slice(accumulator, [8, 32], [token, column], valid_shape=[rows, 32])
            up_i32 = pl.slice(accumulator, [8, 32], [token, up_column], valid_shape=[rows, 32])
            gate_channels = pl.reshape(channels[column : column + 32], [1, 32])
            up_channels = pl.reshape(channels[up_column : up_column + 32], [1, 32])
            row_scale = pl.row_min(scales[token : token + 8, :])
            gate_fp32 = pl.cast(gate_i32, pl.FP32, mode="rint")
            up_fp32 = pl.cast(up_i32, pl.FP32, mode="rint")
            gate_channel_scaled = pl.col_expand_mul(gate_fp32, gate_channels)
            up_channel_scaled = pl.col_expand_mul(up_fp32, up_channels)
            gate_scaled = pl.row_expand_mul(gate_channel_scaled, row_scale)
            up_scaled = pl.row_expand_mul(up_channel_scaled, row_scale)
            gate_filled = pl.fillpad(gate_scaled, pad_value=pl.PadValue.zero)
            up_filled = pl.fillpad(up_scaled, pad_value=pl.PadValue.zero)
            gate_full = pl.set_validshape(gate_filled, 8, 32)
            up_full = pl.set_validshape(up_filled, 8, 32)
            gate = pl.minimum(gate_full, limit)
            up_upper = pl.minimum(up_full, limit)
            up = pl.maximum(up_upper, -limit)
            negative_gate = pl.neg(gate)
            exponent = pl.exp(negative_gate)
            denominator = pl.add(exponent, 1.0)
            silu = pl.div(gate, denominator)
            activation = pl.mul(silu, up)
            output[token : token + 8, column : column + 32] = activation
        if intermediate < padded_intermediate:
            for block in pl.spmd((active + 7) // 8, name_hint="ecotech_shared_activation_pad"):
                token = block * 8
                for column in pl.range(intermediate, padded_intermediate, 32):
                    output[token : token + 8, column : column + 32] = pl.full(
                        [8, 32], dtype=pl.FP32, value=0.0
                    )
        return output

    return activate


def make_ecotech_shared_inline(capacity, hidden, intermediate, *, limit=10.0):
    """Compose local shared computation with an explicit produced INT8 handoff."""
    if capacity <= 0 or capacity % 16 or hidden <= 0 or hidden % 256:
        raise ValueError("Require a positive capacity/16 and hidden/256")
    if intermediate <= 0 or intermediate % 32 or limit != 10.0:
        raise ValueError("Require local intermediate/32 and the source clamp10")
    width = intermediate * 2
    padded_intermediate = ((intermediate + 255) // 256) * 256
    quantize_input = _ecotech_shared_make_quantize(capacity, hidden)
    dot_gate = _ecotech_shared_make_dot(capacity, hidden, width)
    activate = ecotech_shared_make_shared_activation(capacity, intermediate, padded_intermediate, limit)
    quantize_down = _ecotech_shared_make_quantize(capacity, padded_intermediate, pl.FP32)
    dot_down = _ecotech_shared_make_dot(capacity, padded_intermediate, hidden)
    dequantize_down = _ecotech_shared_make_dequantize(capacity, hidden)

    @pl.jit.inline
    def shared(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        gate_weight: pl.Tensor[[hidden * (width // 64), 64], pl.INT8],
        gate_scale: pl.Tensor[[width], pl.FP32],
        down_weight: pl.Tensor[[padded_intermediate * (hidden // 64), 64], pl.INT8],
        down_scale: pl.Tensor[[hidden], pl.FP32],
        output: pl.Tensor[[capacity, hidden], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        input_payload = pl.create_tensor([capacity, hidden], dtype=pl.INT8)
        input_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize_input(x, input_payload, input_scale, active)
        gate_accumulator = pl.create_tensor([capacity, width], dtype=pl.INT32)
        dot_gate(input_payload, gate_weight, gate_accumulator, active)
        activation = pl.create_tensor([capacity, padded_intermediate], dtype=pl.FP32)
        activate(gate_accumulator, gate_scale, input_scale, activation, active)
        down_payload = pl.create_tensor([capacity, padded_intermediate], dtype=pl.INT8)
        down_token_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize_down(activation, down_payload, down_token_scale, active)
        down_accumulator = pl.create_tensor([capacity, hidden], dtype=pl.INT32)
        dot_down(down_payload, down_weight, down_accumulator, active)
        dequantize_down(down_accumulator, down_scale, down_token_scale, output, active)
        return (
            output,
            input_payload,
            input_scale,
            gate_accumulator,
            activation,
            down_payload,
            down_token_scale,
        )

    return shared


def make_ecotech_shared(capacity, hidden, intermediate):
    """Build the local shared-expert entry without outer TP reduction."""
    width = intermediate * 2
    padded_intermediate = ((intermediate + 255) // 256) * 256
    compute = make_ecotech_shared_inline(capacity, hidden, intermediate)

    @pl.jit
    def shared(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        gate_weight: pl.Tensor[[hidden * (width // 64), 64], pl.INT8],
        gate_scale: pl.Tensor[[width], pl.FP32],
        down_weight: pl.Tensor[[padded_intermediate * (hidden // 64), 64], pl.INT8],
        down_scale: pl.Tensor[[hidden], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute(x, gate_weight, gate_scale, down_weight, down_scale, output, active)
        return output

    return shared


def shared_expert_reference(
    value: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
    clamp_limit: float = 10.0,
) -> torch.Tensor:
    """Reference a shared W8A8 expert using the loader's blocked operands.

    The packed down weight includes any zero-padded intermediate columns.
    """

    intermediate = gate_up_scale.numel() // 2
    padded_intermediate = ((intermediate + 255) // 256) * 256
    gate_up_weight = unpack_blocked_weight(gate_up_weight, value.shape[-1], intermediate * 2).T.contiguous()
    down_kn = unpack_blocked_weight(down_weight, padded_intermediate, down_scale.numel())
    down_weight = down_kn[:intermediate].T.contiguous()
    gate_up = w8a8_expert_linear(value, gate_up_weight, gate_up_scale, torch.bfloat16).float()
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.clamp(max=clamp_limit)
    up = up.clamp(-clamp_limit, clamp_limit)
    # vLLM-Ascend's A2/A3 clipped SwiGLU performs both products in BF16.
    sigmoid_bf16 = torch.sigmoid(gate).to(torch.bfloat16)
    silu_bf16 = gate.to(torch.bfloat16) * sigmoid_bf16
    hidden = silu_bf16 * up.to(torch.bfloat16)
    return w8a8_expert_linear(hidden, down_weight, down_scale)


def routed_expert_reference(
    value: torch.Tensor,
    expert_ids: torch.Tensor,
    router_weights: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w3: torch.Tensor,
    w3_scale: torch.Tensor,
    clamp_limit: float = 10.0,
) -> torch.Tensor:
    """Reference for top-k routed W8A8 SwiGLU experts.

    ``w1`` and ``w3`` are gate/up weights with shape ``[E, I, D]`` and ``w2``
    is the down weight with shape ``[E, D, I]``.  Routing is kept explicit so
    each expert boundary can be compared with the vLLM A2/A3 dump.
    """

    result = torch.zeros_like(value, dtype=torch.float32)
    for token in range(value.shape[0]):
        for route in range(expert_ids.shape[1]):
            expert = int(expert_ids[token, route])
            gate = w8a8_expert_linear(value[token : token + 1], w1[expert], w1_scale[expert]).float()
            up = w8a8_expert_linear(value[token : token + 1], w3[expert], w3_scale[expert]).float()
            gate = gate.clamp(max=clamp_limit)
            up = up.clamp(-clamp_limit, clamp_limit)
            sigmoid_bf16 = torch.sigmoid(gate).to(torch.bfloat16)
            silu_bf16 = gate.to(torch.bfloat16) * sigmoid_bf16
            hidden = silu_bf16 * up.to(torch.bfloat16)
            down = w8a8_expert_linear(hidden, w2[expert], w2_scale[expert]).float()
            result[token] += down[0] * router_weights[token, route].float()
    return result.to(torch.bfloat16)


def golden_moe_router(tensors, config):
    """Evaluate the gate and normalized natural or controlled top-k routes for validation."""
    import torch
    import torch.nn.functional as functional

    active = int(tensors["active"])
    if not active:
        return
    logits = tensors["x"][:active].float() @ tensors["gate_weight"].float().T
    scores = functional.softplus(logits).sqrt()
    if config.get("controlled_routing", False):
        indices = tensors["route_ids"][:active].long()
    else:
        ids = tensors["input_ids"][:active]
        base = config["image_sentinel_base_id"]
        vision = ((ids >= base) & (ids < base + 5)).reshape(-1, 1)
        bias = torch.where(vision, tensors["vision_bias"], tensors["text_bias"])
        indices = torch.topk(scores + bias, config["num_experts_per_tok"], dim=-1).indices
    selected = scores.gather(1, indices)
    weights = selected / selected.sum(-1, keepdim=True) * config["routed_scaling_factor"]
    tensors["logits"][:active] = logits
    tensors["route_ids"][:active] = indices.to(torch.int32)
    tensors["route_weights"][:active] = weights


def golden_ep_dispatch_pack(tensors):
    """Pack destination buckets for standalone Golden Harness validation."""
    world = tensors["sent_counts"].shape[0]
    capacity, topk = tensors["route_ids"].shape
    local_experts = 384 // world
    routes = capacity * topk
    source = int(tensors["source_rank"])
    for peer in range(world):
        count = 0
        for token in range(int(tensors["active"])):
            for slot in range(topk):
                expert = int(tensors["route_ids"][token, slot])
                if expert // local_experts == peer:
                    target = peer * routes + count
                    for field, value in enumerate((source, token, slot, expert)):
                        tensors["sent_routes"][target, field] = value
                    tensors["sent_hidden"][target] = tensors["x"][token]
                    tensors["source_map"][token * topk + slot, 0] = target
                    count += 1
        tensors["sent_counts"][peer, 0] = count


def golden_ep_dispatch_unpack(tensors):
    """Unpack received model rows for standalone Golden Harness validation."""
    world = tensors["packet_counts"].shape[0]
    packet_rows = tensors["packet_routes"].shape[0]
    routes = packet_rows // world
    local_experts = tensors["expert_counts"].shape[0]
    expert_capacity = tensors["expert_x"].shape[0] // local_experts
    begin = 0
    for local in range(local_experts):
        expert = int(tensors["owner_rank"]) * local_experts + local
        count = 0
        for peer in range(world):
            for row in range(int(tensors["packet_counts"][peer, 0])):
                packet = peer * routes + row
                if int(tensors["packet_routes"][packet, 3]) == expert:
                    target, compact = local * expert_capacity + count, begin + count
                    tensors["packet_map"][packet, 0] = target
                    tensors["packet_map"][packet, 1] = compact
                    tensors["recv_routes"][compact] = tensors["packet_routes"][packet]
                    tensors["recv_hidden"][compact] = tensors["packet_hidden"][packet]
                    tensors["expert_x"][target] = tensors["packet_hidden"][packet]
                    count += 1
        tensors["expert_counts"][local] = count
        begin += count
    tensors["received_total"][0] = begin


def golden_ep_return_pack(tensors):
    """Gather expert slabs for standalone Golden Harness validation."""
    world = tensors["packet_counts"].shape[0]
    routes = tensors["sent_hidden"].shape[0] // world
    for peer in range(world):
        for row in range(int(tensors["packet_counts"][peer, 0])):
            packet = peer * routes + row
            source = int(tensors["packet_map"][packet, 0])
            tensors["sent_hidden"][packet] = tensors["expert_y"][source]


def golden_moe_combine(tensors, *, weight_dtype=torch.float32):
    """Evaluate the same ordered FP32 route reduction for standalone validation."""
    import torch

    active = int(tensors["active"])
    topk = tensors["route_weights"].shape[1]
    routes = tensors["route_weights"].shape[0] * topk
    mapping = {}
    for peer in range(tensors["packet_counts"].shape[0]):
        for row in range(int(tensors["packet_counts"][peer, 0])):
            packet = peer * routes + row
            token, slot = (int(tensors["packet_routes"][packet, field]) for field in (1, 2))
            original = token * topk + slot
            mapping[original] = packet
            tensors["returned_routes"][original] = tensors["packet_routes"][packet]
    for token in range(active):
        acc = torch.zeros_like(tensors["shared"][token], dtype=torch.float32)
        for slot in range(topk):
            row = token * topk + slot
            if row in mapping:
                value = tensors["packet_hidden"][mapping[row]].float()
                weight = tensors["route_weights"][token, slot].to(weight_dtype).float()
                acc += value * weight
        routed = acc.to(torch.bfloat16)
        tensors["routed"][token] = routed
        tensors["output"][token] = (routed.float() + tensors["shared"][token].float()).to(torch.bfloat16)


def golden_shared_moe_w8a8(
    tensors, *, fused_activation=False, fused_channel_dtype=pl.FP16,
    token_first=False, fused_requantize=False, partitions=1,
):
    """Evaluate channel scaling, activation rounding and requantization for validation."""
    import torch

    active = int(tensors["active"])
    if not active:
        return

    def quantize(value, fused=False):
        value = value.float()
        maximum = value.abs().amax(-1, keepdim=True).clamp_min(1e-30)
        if fused:
            scale = maximum / 127.0
            scaled = (value * scale.reciprocal()).to(torch.float16).float()
        else:
            scale = maximum * (1.0 / 127.0)
            scaled = value * (127.0 / maximum)
        quantized = scaled.round().clamp(-127, 127).to(torch.int32)
        return quantized, scale

    def dot(payload, weight, channels, scale, *, token_first=False):
        logical = unpack_blocked_weight(weight, payload.shape[-1], channels.numel())
        accumulator = payload @ logical.to(torch.int32)
        if token_first:
            return (accumulator.float() * scale) * channels.float()
        channel = accumulator.float() * channels.float()
        if fused_activation and fused_channel_dtype == pl.FP16:
            channel = channel.to(torch.float16).float()
        return channel * scale

    parts = []
    gates = tensors["gate_up_weight"].chunk(partitions, dim=0)
    channels = tensors["gate_up_scale"].chunk(partitions)
    downs = tensors["down_weight"].chunk(partitions, dim=0)
    for gate_weight, gate_scale, down_weight in zip(gates, channels, downs):
        payload, scale = quantize(tensors["x"][:active])
        gate_up = dot(payload, gate_weight, gate_scale, scale, token_first=token_first)
        if not fused_activation:
            gate_up = gate_up.to(torch.bfloat16).float()
        gate, up = gate_up.chunk(2, dim=-1)
        gate, up = gate.clamp(max=10.0), up.clamp(-10.0, 10.0)
        if fused_activation:
            activation = (gate / (1.0 + torch.exp(-gate))) * up
        else:
            sigmoid = torch.sigmoid(gate).to(torch.bfloat16)
            silu = (gate * sigmoid.float()).to(torch.bfloat16)
            activation = (silu.float() * up).to(torch.bfloat16)
        payload, scale = quantize(activation, fused_requantize)
        down = dot(payload, down_weight, tensors["down_scale"], scale)
        parts.append(down.to(torch.bfloat16))
    while len(parts) > 1:
        parts = [(a.float() + b.float()).to(torch.bfloat16) for a, b in zip(parts[::2], parts[1::2])]
    tensors["output"][:active] = parts[0]


def moe_router_golden_cases(*, build=True):
    """Cover natural and controlled Top6 routing with 384 experts and inactive rows."""
    from functools import partial

    import torch
    from golden import ScalarSpec, TensorSpec

    torch.manual_seed(41)
    capacity, hidden, experts, topk = 16, 256, 384, 6
    config = {
        "hidden_size": hidden,
        "n_routed_experts": experts,
        "num_experts_per_tok": topk,
        "routed_scaling_factor": 1.5,
        "image_sentinel_base_id": 129264,
    }
    for controlled in (False, True):
        settings = dict(config, controlled_routing=controlled)
        for active in (0, 1, 7, 16):
            ids = torch.arange(capacity, dtype=torch.int64) + 129260
            routes = torch.tensor([2, 195, 1, 194, 0, 193], dtype=torch.int32).expand(capacity, -1).clone()
            specs = [
                TensorSpec("x", [capacity, hidden], torch.bfloat16, init_value=torch.randn),
                TensorSpec("gate_weight", [experts, hidden], torch.float32, init_value=torch.randn),
                TensorSpec("text_bias", [experts], torch.float32, init_value=torch.randn),
                TensorSpec("vision_bias", [experts], torch.float32, init_value=torch.randn),
                TensorSpec("input_ids", [capacity], torch.int64, init_value=ids),
                TensorSpec("logits", [capacity, experts], torch.float32, init_value=-12.5),
                TensorSpec("route_ids", [capacity, topk], torch.int32, init_value=routes),
                TensorSpec("route_weights", [capacity, topk], torch.float32, init_value=-12.5),
                ScalarSpec("active", torch.int32, active),
            ]
            case = dict(
                name=f"router-controlled{int(controlled)}-active{active}",
                specs=specs,
                golden_fn=partial(golden_moe_router, config=settings),
                rtol=0.01,
                atol=0.001,
            )
            if build:
                case["fn"] = make_moe_router(capacity, settings)
            yield case


def validate_moe_router(*, device=0, compile_only=False, platform="a2a3", sim=False):
    """Validate natural and controlled routing using the product router definition."""
    from golden import run

    for case in moe_router_golden_cases():
        name = case.pop("name")
        result = run(
            **case,
            compile_only=compile_only,
            config={"backend_type": platform, "execution_mode": sim, "device_id": device},
        )
        if not result.passed:
            raise RuntimeError(f"Router validation failed: {name}: {result.error}")


def validate_received_routes(packet, rank, layout, routes, valid_by_rank=None, *, expert_capacity=None):
    """Check transported source, expert ownership and bounded bucket metadata."""
    counts = [int(packet["counts"][peer, 0]) for peer in range(layout.ep)]
    if any(not 0 <= count <= routes for count in counts):
        raise ValueError("HCCL received bucket count exceeds its fixed capacity")
    expert_counts = {}
    for peer, count in enumerate(counts):
        for row in range(count):
            source, token, slot, expert = packet["routes"][peer * routes + row].tolist()
            if source != peer or layout.owner(expert) != rank or token < 0 or slot < 0:
                raise RuntimeError("HCCL delivered inconsistent route metadata")
            if valid_by_rank is not None and token >= valid_by_rank[peer]:
                raise RuntimeError("HCCL dispatched a masked source row")
            expert_counts[expert] = expert_counts.get(expert, 0) + 1
    if expert_capacity is not None and any(count > expert_capacity for count in expert_counts.values()):
        raise RuntimeError("Received routes exceed the proven per-expert slab bound")
    return sum(counts)


def validate_returned_routes(packet, rank, routes, route_ids, valid):
    """Require one returned result for every valid source token and top-k slot."""
    seen = set()
    topk = route_ids.shape[1]
    for peer in range(packet["counts"].shape[0]):
        count = int(packet["counts"][peer, 0])
        if not 0 <= count <= routes:
            raise ValueError("Returned bucket count exceeds capacity")
        for row in range(count):
            source, token, slot, expert = packet["routes"][peer * routes + row].tolist()
            if source != rank or not 0 <= token < valid or not 0 <= slot < topk:
                raise RuntimeError("Returned source token or top-k slot is invalid")
            if (token, slot) in seen or expert != int(route_ids[token, slot]):
                raise RuntimeError("Duplicate or inconsistent returned expert route")
            seen.add((token, slot))
    if len(seen) != valid * topk:
        raise RuntimeError("HCCL did not return every valid top-k route")


def validate_ep_dispatch(*, device=0, compile_only=False, platform="a2a3", sim=False):
    """Validate packing, received expert layout, empty buckets, and return gathering."""
    import torch
    from golden import ScalarSpec, TensorSpec, run

    experts, topk = 384, 6
    cases = [(world, 16, 512, (0, 1, 7, 16)) for world in (2, 16)]
    # A wide EP packet exercises full-row copies with nonempty buckets and untouched padding.
    cases.append((16, 64, 3072, (5,)))
    for world, capacity, hidden, active_rows in cases:
        routes, local_experts = capacity * topk, experts // world
        packet_rows = world * routes
        expert_capacity = ((capacity * world + 15) // 16) * 16
        slab_rows = local_experts * expert_capacity
        for active in active_rows:
            ids = torch.tensor([0, 1, 2, experts - 3, experts - 2, experts - 1], dtype=torch.int32)
            ids = ids.expand(capacity, -1).clone()
            pack_specs = [
                TensorSpec("x", [capacity, hidden], torch.bfloat16, init_value=torch.randn),
                TensorSpec("route_ids", [capacity, topk], torch.int32, init_value=ids),
                TensorSpec("sent_routes", [packet_rows, 4], torch.int64, init_value=-1),
                TensorSpec("sent_hidden", [packet_rows, hidden], torch.bfloat16, init_value=0.0),
                TensorSpec("sent_counts", [world, DISPATCH_MAP_PAD], torch.int32, init_value=0),
                TensorSpec("source_map", [routes, DISPATCH_SOURCE_MAP_PAD], torch.int32, init_value=-1),
                ScalarSpec("active", torch.int32, active),
                ScalarSpec("source_rank", torch.int32, 0),
            ]
            result = run(
                compile_only=compile_only,
                fn=make_ep_dispatch_pack(capacity, hidden, experts, topk, world),
                specs=pack_specs,
                golden_fn=golden_ep_dispatch_pack,
                config={"backend_type": platform, "execution_mode": sim, "device_id": device},
                rtol=0.0,
                atol=0.0,
            )
            if not result.passed:
                raise RuntimeError(f"EP pack validation failed: EP{world}, active={active}: {result.error}")
            metadata = torch.full((packet_rows, 4), -1, dtype=torch.int64)
            counts = torch.zeros((world, DISPATCH_MAP_PAD), dtype=torch.int32)
            for peer in range(world):
                counts[peer, 0] = active
                for token in range(active):
                    metadata[peer * routes + token] = torch.tensor([peer, token, 0, token % local_experts])
            payload = torch.randn(packet_rows, hidden, dtype=torch.bfloat16)
            unpack_specs = [
                TensorSpec("packet_routes", [packet_rows, 4], torch.int64, init_value=metadata),
                TensorSpec("packet_hidden", [packet_rows, hidden], torch.bfloat16, init_value=payload),
                TensorSpec("packet_counts", [world, DISPATCH_MAP_PAD], torch.int32, init_value=counts),
                TensorSpec("recv_routes", [packet_rows, 4], torch.int64, init_value=-1),
                TensorSpec("recv_hidden", [packet_rows, hidden], torch.bfloat16, init_value=0.0),
                TensorSpec("expert_x", [slab_rows, hidden], torch.bfloat16, init_value=0.0),
                TensorSpec("expert_counts", [local_experts], torch.int64, init_value=0),
                TensorSpec("packet_map", [packet_rows, DISPATCH_MAP_PAD], torch.int32, init_value=-1),
                TensorSpec("received_total", [DISPATCH_MAP_PAD], torch.int32, init_value=0),
                ScalarSpec("owner_rank", torch.int32, 0),
            ]
            result = run(
                compile_only=compile_only,
                fn=make_ep_dispatch_unpack(capacity, hidden, experts, topk, world),
                specs=unpack_specs,
                golden_fn=golden_ep_dispatch_unpack,
                config={"backend_type": platform, "execution_mode": sim, "device_id": device},
                rtol=0.0,
                atol=0.0,
            )
            if not result.passed:
                raise RuntimeError(f"EP unpack validation failed: EP{world}, active={active}: {result.error}")
            mapping = torch.full((packet_rows, DISPATCH_MAP_PAD), -1, dtype=torch.int32)
            for peer in range(world):
                for token in range(active):
                    target = token % local_experts * expert_capacity + peer
                    mapping[peer * routes + token, 0] = target
            return_specs = [
                TensorSpec("expert_y", [slab_rows, hidden], torch.bfloat16, init_value=torch.randn),
                TensorSpec("packet_counts", [world, DISPATCH_MAP_PAD], torch.int32, init_value=counts),
                TensorSpec("packet_map", [packet_rows, DISPATCH_MAP_PAD], torch.int32, init_value=mapping),
                TensorSpec("sent_hidden", [packet_rows, hidden], torch.bfloat16, init_value=0.0),
            ]
            result = run(
                compile_only=compile_only,
                fn=make_ep_return_pack(capacity, hidden, experts, topk, world),
                specs=return_specs,
                golden_fn=golden_ep_return_pack,
                config={"backend_type": platform, "execution_mode": sim, "device_id": device},
                rtol=0.0,
                atol=0.0,
            )
            if not result.passed:
                raise RuntimeError(f"EP return validation failed: EP{world}, active={active}: {result.error}")


def validate_moe_combine(*, device=0, compile_only=False, platform="a2a3", sim=False):
    """Validate ordered route weighting and inactive sentinel preservation."""
    import torch
    from golden import ScalarSpec, TensorSpec, run

    capacity, hidden, topk, world = 16, 256, 6, 2
    for active, valid in ((0, 0), (1, 0), (1, 1), (7, 1), (7, 7), (16, 7), (16, 16)):
        routes = capacity * topk
        metadata = torch.full((world * routes, 4), -1, dtype=torch.int64)
        counts = torch.zeros((world, 8), dtype=torch.int32)
        source_map = torch.full((routes, DISPATCH_SOURCE_MAP_PAD), -1, dtype=torch.int32)
        for token in range(valid):
            for slot in range(topk):
                peer = slot % world
                row = int(counts[peer, 0])
                metadata[peer * routes + row] = torch.tensor([0, token, slot, peer * 192 + slot])
                source_map[token * topk + slot, 0] = peer * routes + row
                counts[peer, 0] += 1
        specs = [
            TensorSpec("packet_routes", [world * routes, 4], torch.int64, init_value=metadata),
            TensorSpec("packet_hidden", [world * routes, hidden], torch.bfloat16, init_value=torch.randn),
            TensorSpec("packet_counts", [world, 8], torch.int32, init_value=counts),
            TensorSpec("route_weights", [capacity, topk], torch.float32, init_value=torch.rand),
            TensorSpec("shared", [capacity, hidden], torch.bfloat16, init_value=torch.randn),
            TensorSpec("routed", [capacity, hidden], torch.bfloat16, init_value=-12.5),
            TensorSpec("output", [capacity, hidden], torch.bfloat16, init_value=-12.5),
            TensorSpec("returned_routes", [routes, 4], torch.int64, init_value=-1),
            ScalarSpec("active", torch.int32, active),
        ]
        result = run(
            compile_only=compile_only,
            fn=make_moe_combine(capacity, hidden, topk),
            specs=specs,
            golden_fn=golden_moe_combine,
            config={"backend_type": platform, "execution_mode": sim, "device_id": device},
            rtol=0.05,
            atol=0.05,
        )
        if not result.passed:
            raise RuntimeError(f"Combine validation failed: active={active}, valid={valid}: {result.error}")
        indexed_specs = list(specs)
        indexed_specs[2] = TensorSpec("source_map", list(source_map.shape), torch.int32, init_value=source_map)
        indexed_specs.append(ScalarSpec("valid", torch.int32, valid))

        def indexed_golden(tensors):
            tensors["packet_counts"] = counts
            golden_moe_combine(tensors, weight_dtype=torch.bfloat16)

        result = run(
            compile_only=compile_only,
            fn=make_moe_combine_from_dispatch(capacity, hidden, topk, world),
            specs=indexed_specs,
            golden_fn=indexed_golden,
            config={"backend_type": platform, "execution_mode": sim, "device_id": device},
            rtol=0.05,
            atol=0.05,
        )
        if not result.passed:
            raise RuntimeError(f"Indexed combine validation failed: active={active}, valid={valid}: {result.error}")


def shared_moe_golden_cases(*, build=True):
    """Cover W8A8 experts with BF16, FP16/FP32 and partitioned activation boundaries."""
    from functools import partial

    import torch
    from golden import ScalarSpec, TensorSpec

    torch.manual_seed(41)
    capacity, hidden, intermediate = 16, 256, 256
    modes = [
        dict(fused_activation=False, fused_channel_dtype=pl.FP32),
        dict(fused_activation=True, fused_channel_dtype=pl.FP16),
        dict(fused_activation=True, fused_channel_dtype=pl.FP32),
        dict(fused_activation=True, fused_channel_dtype=pl.FP32, token_first=True, fused_requantize=True),
        dict(fused_activation=True, fused_channel_dtype=pl.FP32, partitions=4),
    ]
    for mode_index, mode in enumerate(modes):
        partitions = mode.get("partitions", 1)
        config = dict(hidden_size=hidden, moe_intermediate_size=intermediate * partitions, swiglu_limit=10.0)
        for active in (0, 1, 7, 16):
            x = torch.randn(capacity, hidden, dtype=torch.bfloat16)
            gates = [torch.randint(-32, 33, (hidden, 2 * intermediate), dtype=torch.int8) for _ in range(partitions)]
            downs = [torch.randint(-32, 33, (intermediate, hidden), dtype=torch.int8) for _ in range(partitions)]
            gate_up = torch.cat([pack_blocked_weight(value) for value in gates])
            down = torch.cat([pack_blocked_weight(value) for value in downs])
            specs = [
                TensorSpec("x", list(x.shape), x.dtype, init_value=x),
                TensorSpec("gate_up_weight", list(gate_up.shape), gate_up.dtype, init_value=gate_up),
                TensorSpec("gate_up_scale", [2 * intermediate * partitions], torch.float32, init_value=0.002),
                TensorSpec("down_weight", list(down.shape), down.dtype, init_value=down),
                TensorSpec("down_scale", [hidden], torch.float32, init_value=0.002),
                TensorSpec("output", [capacity, hidden], torch.bfloat16, init_value=-12.5),
                ScalarSpec("active", torch.int32, active),
            ]
            case = dict(
                name=f"shared-mode{mode_index}-active{active}",
                specs=specs,
                golden_fn=partial(golden_shared_moe_w8a8, **mode),
                rtol=0.05,
                atol=0.05,
            )
            if build:
                case["fn"] = make_shared_moe_w8a8(capacity, config, **mode)
            yield case


def validate_shared_moe(*, device=0, compile_only=False, platform="a2a3", sim=False):
    """Validate standalone BF16 and fused FP16/FP32 channel boundaries."""
    from golden import run

    for case in shared_moe_golden_cases():
        name = case.pop("name")
        result = run(
            **case,
            compile_only=compile_only,
            config={"backend_type": platform, "execution_mode": sim, "device_id": device},
        )
        if not result.passed:
            raise RuntimeError(f"Expert validation failed: {name}: {result.error}")


def main():
    """Validate Prefill MoE operations using their original goldens."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    args = vars(parser.parse_args())
    for validate in (validate_moe_router, validate_ep_dispatch, validate_moe_combine, validate_shared_moe):
        validate(**args)


if __name__ == "__main__":
    main()

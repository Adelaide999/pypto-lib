# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared MoE kernels, distinct phase boundaries, Torch goldens and validation."""
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"


import torch
import pypto.language as pl


from .decode_network_ecotech_linear import prepare_ecotech_dynamic_weight as _ecotech_shared_prepare_ecotech_dynamic_weight
from .decode_network_ecotech_linear import prepare_ecotech_tail_weight as _ecotech_shared_prepare_ecotech_tail_weight
from .decode_network_linear_primitives import make_dequantize as _ecotech_shared_make_dequantize, make_dot as _ecotech_shared_make_dot, make_pack_weight as _ecotech_shared_make_pack_weight, make_quantize as _ecotech_shared_make_quantize

from .decode_network_gemm import make_fp32_dequantize as _decode_moe_make_fp32_dequantize, make_int8_gemm as _decode_moe_make_int8_gemm, make_pack_weight as _decode_moe_make_pack_weight
from .decode_network_quant import make_dynamic_int8_quantize as _decode_moe_make_dynamic_int8_quantize
from .decode_network_quant import w8a8_linear as _decode_moe_w8a8_linear

from .decode_network_decode_primitives import N_TILE as DECODE_SHARED_N_TILE, VEC_M_TILE as DECODE_SHARED_VEC_M_TILE, make_dequantize as _decode_shared_make_dequantize, make_dot as _decode_shared_make_dot, make_pack_weight as _decode_shared_make_pack_weight, make_quantize as _decode_shared_make_quantize


MOE_VEC_M_TILE = 8
MOE_ACT_TILE = 256


# tiling
ROUTER_D_TILE = 256
ROUTER_EXPERT_TILE = 8
ROUTER_TOPK_TILE = 8


# model config
DISPATCH_EXPERTS = 384
DISPATCH_TOPK = 6
DISPATCH_D = 5120
DISPATCH_RECV_MAX = 64

# tiling
DISPATCH_D_TILE = 512
DISPATCH_MAP_PAD = 8

COMBINE_TOPK = 6
COMBINE_D = 5120


DECODE_MOE_VEC_M_TILE = 8
DECODE_MOE_ACT_TILE = 256


# tiling
DECODE_ROUTER_D_TILE = 256
DECODE_ROUTER_EXPERT_TILE = 8
DECODE_ROUTER_TOPK_TILE = 8


# model config
DECODE_DISPATCH_EXPERTS = 384
DECODE_DISPATCH_TOPK = 6
DECODE_DISPATCH_D = 5120
DECODE_DISPATCH_RECV_MAX = 64

# tiling
DECODE_DISPATCH_D_TILE = 512
DECODE_DISPATCH_MAP_PAD = 8


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
        active: pl.Scalar[pl.INT32],
        source_rank: pl.Scalar[pl.INT32],
    ):
        route_map = pl.create_tensor([routes, DISPATCH_MAP_PAD], dtype=pl.INT32)
        for _block in pl.spmd(1, name_hint="moe_ep_pack_metadata"):
            for peer in pl.range(world):
                bucket_count = pl.cast(0, pl.INDEX)
                for token in pl.range(active):
                    for slot in pl.range(topk):
                        expert = pl.read(route_ids, [token, slot])
                        if expert // local_experts == peer:
                            packet_index = peer * routes + bucket_count
                            pl.write(sent_routes, [packet_index, 0], pl.cast(source_rank, pl.INT64))
                            pl.write(sent_routes, [packet_index, 1], pl.cast(token, pl.INT64))
                            pl.write(sent_routes, [packet_index, 2], pl.cast(slot, pl.INT64))
                            pl.write(sent_routes, [packet_index, 3], pl.cast(expert, pl.INT64))
                            original_route = token * topk + slot
                            pl.write(route_map, [original_route, 0], pl.cast(packet_index, pl.INT32))
                            bucket_count = bucket_count + 1
                pl.write(sent_counts, [peer, 0], pl.cast(bucket_count, pl.INT32))

        for block in pl.spmd(active * topk * (hidden // DISPATCH_D_TILE), name_hint="moe_ep_pack_hidden"):
            route_index = block // (hidden // DISPATCH_D_TILE)
            column = block % (hidden // DISPATCH_D_TILE) * DISPATCH_D_TILE
            source_token = route_index // topk
            target_index = pl.cast(pl.read(route_map, [route_index, 0]), pl.INDEX)
            source_row = x[source_token:source_token + 1, column:column + DISPATCH_D_TILE]
            sent_hidden[target_index:target_index + 1, column:column + DISPATCH_D_TILE] = source_row
        return x, sent_routes, sent_hidden, sent_counts

    return ep_dispatch_pack


def make_ep_dispatch_unpack(capacity, hidden, experts, topk, world, expert_capacity=None):
    """Unpack received buckets into owned expert slabs and compact audit payloads."""
    routes = capacity * topk
    local_experts = experts // world
    packet_rows = world * routes
    if expert_capacity is None:
        expert_capacity = ((capacity * world + 15) // 16) * 16
    slab_rows = local_experts * expert_capacity

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
            compact_begin = pl.cast(0, pl.INDEX)
            for local in pl.range(local_experts):
                expert = owner_rank * local_experts + local
                owned_count = pl.cast(0, pl.INDEX)
                for peer in pl.range(world):
                    peer_count = pl.cast(pl.read(packet_counts, [peer, 0]), pl.INDEX)
                    for row in pl.range(peer_count):
                        received_packet = peer * routes + row
                        global_expert = pl.read(packet_routes, [received_packet, 3])
                        if global_expert == expert:
                            compact_row = compact_begin + owned_count
                            slab_row = local * expert_capacity + owned_count
                            pl.write(packet_map, [received_packet, 0], pl.cast(slab_row, pl.INT32))
                            pl.write(packet_map, [received_packet, 1], pl.cast(compact_row, pl.INT32))
                            for field in pl.range(4):
                                field_value = pl.read(packet_routes, [received_packet, field])
                                pl.write(recv_routes, [compact_row, field], field_value)
                            owned_count = owned_count + 1
                pl.write(expert_counts, [local], pl.cast(owned_count, pl.INT64))
                compact_begin = compact_begin + owned_count
            pl.write(received_total, [0], pl.cast(compact_begin, pl.INT32))

        for block in pl.spmd(packet_rows * (hidden // DISPATCH_D_TILE), name_hint="moe_ep_unpack_hidden"):
            packet = block // (hidden // DISPATCH_D_TILE)
            column = block % (hidden // DISPATCH_D_TILE) * DISPATCH_D_TILE
            source_peer = packet // routes
            bucket_rows = pl.read(packet_counts, [source_peer, 0])
            if packet % routes < bucket_rows:
                target = pl.cast(pl.read(packet_map, [packet, 0]), pl.INDEX)
                compact = pl.cast(pl.read(packet_map, [packet, 1]), pl.INDEX)
                value = packet_hidden[packet:packet + 1, column:column + DISPATCH_D_TILE]
                expert_x[target:target + 1, column:column + DISPATCH_D_TILE] = value
                recv_hidden[compact:compact + 1, column:column + DISPATCH_D_TILE] = value
        return packet_hidden, recv_routes, recv_hidden, expert_x, expert_counts, packet_map, received_total

    return ep_dispatch_unpack


def make_ep_return_pack(capacity, hidden, experts, topk, world, expert_capacity=None):
    """Gather owned expert results into each original source's return bucket."""
    routes = capacity * topk
    packet_rows = world * routes
    if expert_capacity is None:
        expert_capacity = ((capacity * world + 15) // 16) * 16
    slab_rows = (experts // world) * expert_capacity

    @pl.jit
    def ep_return_pack(
        expert_y: pl.Tensor[[slab_rows, hidden], pl.BF16],
        packet_counts: pl.Tensor[[world, DISPATCH_MAP_PAD], pl.INT32],
        packet_map: pl.Tensor[[packet_rows, DISPATCH_MAP_PAD], pl.INT32],
        sent_hidden: pl.InOut[pl.Tensor[[packet_rows, hidden], pl.BF16]],
    ):
        for block in pl.spmd(packet_rows * (hidden // DISPATCH_D_TILE), name_hint="moe_ep_return_pack"):
            packet = block // (hidden // DISPATCH_D_TILE)
            column = block % (hidden // DISPATCH_D_TILE) * DISPATCH_D_TILE
            count = pl.read(packet_counts, [packet // routes, 0])
            if packet % routes < count:
                source = pl.cast(pl.read(packet_map, [packet, 0]), pl.INDEX)
                expert_row = expert_y[source:source + 1, column:column + DISPATCH_D_TILE]
                sent_hidden[packet:packet + 1, column:column + DISPATCH_D_TILE] = expert_row
        return sent_hidden

    return ep_return_pack


def make_moe_combine(capacity, hidden=COMBINE_D, topk=COMBINE_TOPK, world=2):
    """Restore returned token/top-k rows, weight experts, and add the shared result."""
    routes = capacity * topk
    packet_rows = world * routes

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
        route_map = pl.create_tensor([routes, 8], dtype=pl.INT32)
        for _block in pl.spmd(1, name_hint="moe_ep_restore_metadata"):
            for original in pl.range(routes):
                pl.write(route_map, [original, 0], pl.cast(-1, pl.INT32))
            for peer in pl.range(world):
                count = pl.cast(pl.read(packet_counts, [peer, 0]), pl.INDEX)
                for row in pl.range(count):
                    metadata_packet = peer * routes + row
                    source_token = pl.cast(pl.read(packet_routes, [metadata_packet, 1]), pl.INDEX)
                    source_slot = pl.cast(pl.read(packet_routes, [metadata_packet, 2]), pl.INDEX)
                    original = source_token * topk + source_slot
                    pl.write(route_map, [original, 0], pl.cast(metadata_packet, pl.INT32))
                    for field in pl.range(4):
                        field_value = pl.read(packet_routes, [metadata_packet, field])
                        pl.write(returned_routes, [original, field], field_value)

        for token in pl.spmd(active, name_hint="moe_ep_source_combine"):
            acc = pl.full([1, hidden], dtype=pl.FP32, value=0.0)
            for slot in pl.range(topk):
                route_row = token * topk + slot
                packet = pl.cast(pl.read(route_map, [route_row, 0]), pl.INDEX)
                if packet >= 0:
                    value = pl.cast(packet_hidden[packet:packet + 1, :], pl.FP32)
                    weight = pl.read(route_weights, [token, slot])
                    acc = pl.add(acc, pl.mul(value, weight))
            narrowed = pl.cast(acc, pl.BF16, mode="rint")
            routed[token:token + 1, :] = narrowed
            shared_row = pl.cast(shared[token:token + 1, :], pl.FP32)
            total = pl.add(pl.cast(narrowed, pl.FP32), shared_row)
            output[token:token + 1, :] = pl.cast(total, pl.BF16, mode="rint")
        return routed, output, returned_routes

    return moe_combine


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
            gate_channels = pl.reshape(channels[column:column + 32], [1, 32])
            up_channels = pl.reshape(channels[up_column:up_column + 32], [1, 32])
            row_scale = pl.row_min(scales[token:token + 8, :])
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
            output[token:token + 8, column:column + 32] = activation
        if intermediate < padded_intermediate:
            for block in pl.spmd((active + 7) // 8, name_hint="ecotech_shared_activation_pad"):
                token = block * 8
                for column in pl.range(intermediate, padded_intermediate, 32):
                    output[token:token + 8, column:column + 32] = pl.full([8, 32], dtype=pl.FP32, value=0.0)
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
    gate_packed_rows = hidden * (width // 64)
    down_packed_rows = padded_intermediate * (hidden // 64)
    quantize_input = _ecotech_shared_make_quantize(capacity, hidden)
    pack_gate = _ecotech_shared_make_pack_weight(hidden, width)
    dot_gate = _ecotech_shared_make_dot(capacity, hidden, width)
    activate = ecotech_shared_make_shared_activation(capacity, intermediate, padded_intermediate, limit)
    quantize_down = _ecotech_shared_make_quantize(capacity, padded_intermediate, pl.FP32)
    pack_down = _ecotech_shared_make_pack_weight(padded_intermediate, hidden)
    dot_down = _ecotech_shared_make_dot(capacity, padded_intermediate, hidden)
    dequantize_down = _ecotech_shared_make_dequantize(capacity, hidden)

    @pl.jit.inline
    def shared(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        gate_weight: pl.Tensor[[hidden, width], pl.INT8],
        gate_scale: pl.Tensor[[width], pl.FP32],
        down_weight: pl.Tensor[[padded_intermediate, hidden], pl.INT8],
        down_scale: pl.Tensor[[hidden], pl.FP32],
        output: pl.Tensor[[capacity, hidden], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        input_payload = pl.create_tensor([capacity, hidden], dtype=pl.INT8)
        input_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize_input(x, input_payload, input_scale, active)
        gate_packed = pl.create_tensor([gate_packed_rows, 64], dtype=pl.INT8)
        pack_gate(gate_weight, gate_packed)
        gate_accumulator = pl.create_tensor([capacity, width], dtype=pl.INT32)
        dot_gate(input_payload, gate_packed, gate_accumulator, active)
        activation = pl.create_tensor([capacity, padded_intermediate], dtype=pl.FP32)
        activate(gate_accumulator, gate_scale, input_scale, activation, active)
        down_payload = pl.create_tensor([capacity, padded_intermediate], dtype=pl.INT8)
        down_token_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize_down(activation, down_payload, down_token_scale, active)
        down_packed = pl.create_tensor([down_packed_rows, 64], dtype=pl.INT8)
        pack_down(down_weight, down_packed)
        down_accumulator = pl.create_tensor([capacity, hidden], dtype=pl.INT32)
        dot_down(down_payload, down_packed, down_accumulator, active)
        dequantize_down(down_accumulator, down_scale, down_token_scale, output, active)
        return output, input_payload, input_scale, gate_accumulator, activation, down_payload, down_token_scale

    return shared


def make_ecotech_shared(capacity, hidden, intermediate):
    """Build the local shared-expert entry without outer TP reduction."""
    width = intermediate * 2
    padded_intermediate = ((intermediate + 255) // 256) * 256
    compute = make_ecotech_shared_inline(capacity, hidden, intermediate)

    @pl.jit
    def shared(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        gate_weight: pl.Tensor[[hidden, width], pl.INT8],
        gate_scale: pl.Tensor[[width], pl.FP32],
        down_weight: pl.Tensor[[padded_intermediate, hidden], pl.INT8],
        down_scale: pl.Tensor[[hidden], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute(x, gate_weight, gate_scale, down_weight, down_scale, output, active)
        return output

    return shared


def decode_moe_w8a8_expert_linear(
    value: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run one W8A8 expert projection using the shared linear golden."""

    return _decode_moe_w8a8_linear(value, weight, scale, out_dtype=out_dtype)


def make_decode_w8a8_expert_projection(capacity: int, input_width: int, output_width: int):
    """Build the shared leaf used by routed and shared W8A8 projections.

    Dispatch/combine metadata stays outside this kernel.  Each routed expert
    receives a contiguous token slice and uses the same A2/A3 W8A8 boundary as
    the shared expert: dynamic INT8 activation, INT32 dot, FP32 scales, BF16
    output.
    """

    quantize = _decode_moe_make_dynamic_int8_quantize(capacity, input_width)
    gemm = _decode_moe_make_int8_gemm(capacity, input_width, output_width)
    dequantize = _decode_moe_make_fp32_dequantize(capacity, output_width)

    @pl.jit.inline
    def projection(
        value: pl.Tensor[[capacity, input_width], pl.BF16],
        packed_weight: pl.Tensor[[input_width, output_width], pl.INT8],
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


def make_decode_shared_expert_projection(capacity: int, input_width: int, output_width: int):
    """Alias the W8A8 projection leaf for the shared expert path."""

    return make_decode_w8a8_expert_projection(capacity, input_width, output_width)


def make_decode_routed_expert_projection(capacity: int, input_width: int, output_width: int):
    """Alias the W8A8 projection leaf for one grouped routed expert."""

    return make_decode_w8a8_expert_projection(capacity, input_width, output_width)


def make_decode_shared_expert_w8a8(
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
    quant_x = _decode_moe_make_dynamic_int8_quantize(capacity, hidden)
    quant_hidden = _decode_moe_make_dynamic_int8_quantize(capacity, intermediate)
    pack_gate_up = _decode_moe_make_pack_weight(hidden, gate_up_width)
    pack_down = _decode_moe_make_pack_weight(intermediate, hidden)
    dot_gate_up = _decode_moe_make_int8_gemm(capacity, hidden, gate_up_width)
    dot_down = _decode_moe_make_int8_gemm(capacity, intermediate, hidden)
    dequant_gate_up = _decode_moe_make_fp32_dequantize(capacity, gate_up_width)
    dequant_down = _decode_moe_make_fp32_dequantize(capacity, hidden)

    @pl.jit
    def shared_expert(
        value: pl.Tensor[[capacity, hidden], pl.BF16],
        gate_up_weight: pl.Tensor[[hidden, gate_up_width], pl.INT8],
        gate_up_scale: pl.Tensor[[gate_up_width], pl.FP32],
        down_weight: pl.Tensor[[intermediate, hidden], pl.INT8],
        down_scale: pl.Tensor[[hidden], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        x_payload = pl.create_tensor([capacity, hidden], dtype=pl.INT8)
        x_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quant_x(value, x_payload, x_scale, active)

        gate_up_packed = pl.create_tensor([hidden * (gate_up_width // 64), 64], dtype=pl.INT8)
        pack_gate_up(gate_up_weight, gate_up_packed)
        gate_up_acc = pl.create_tensor([capacity, gate_up_width], dtype=pl.INT32)
        dot_gate_up(x_payload, gate_up_packed, gate_up_acc, active)
        gate_up = pl.create_tensor([capacity, gate_up_width], dtype=pl.BF16)
        dequant_gate_up(gate_up_acc, x_scale, gate_up_scale, gate_up, active)

        activation = pl.create_tensor([capacity, intermediate], dtype=pl.BF16)
        for block in pl.spmd((active + DECODE_MOE_VEC_M_TILE - 1) // DECODE_MOE_VEC_M_TILE * (intermediate // DECODE_MOE_ACT_TILE), name_hint="dsv41_a2a3_swiglu_v2"):
            token = block // (intermediate // DECODE_MOE_ACT_TILE) * DECODE_MOE_VEC_M_TILE
            column = block % (intermediate // DECODE_MOE_ACT_TILE) * DECODE_MOE_ACT_TILE
            rows = pl.min(DECODE_MOE_VEC_M_TILE, active - token)
            # vLLM's A2/A3 ``SiluAndMulWithClamp`` keeps the two products in
            # BF16.  Use FP32 only for the exponential, then multiply the
            # BF16 sigmoid, gate, and up values without an FP32 promotion.
            gate = pl.cast(pl.slice(gate_up, [DECODE_MOE_VEC_M_TILE, DECODE_MOE_ACT_TILE], [token, column]), pl.FP32)
            up = pl.cast(pl.slice(gate_up, [DECODE_MOE_VEC_M_TILE, DECODE_MOE_ACT_TILE], [token, intermediate + column]), pl.FP32)
            gate = pl.minimum(gate, clamp_limit)
            up = pl.maximum(pl.minimum(up, clamp_limit), -clamp_limit)
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate)), 1.0))
            sigmoid_bf16 = pl.cast(sigmoid, pl.BF16, mode="rint")
            silu = pl.mul(gate, pl.cast(sigmoid_bf16, pl.FP32))
            silu_bf16 = pl.cast(silu, pl.BF16, mode="rint")
            activated = pl.mul(pl.cast(silu_bf16, pl.FP32), up)
            activation[token:token + DECODE_MOE_VEC_M_TILE, column:column + DECODE_MOE_ACT_TILE] = pl.set_validshape(
                pl.cast(activated, pl.BF16, mode="rint"), rows, DECODE_MOE_ACT_TILE
            )

        hidden_payload = pl.create_tensor([capacity, intermediate], dtype=pl.INT8)
        hidden_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quant_hidden(activation, hidden_payload, hidden_scale, active)
        down_packed = pl.create_tensor([intermediate * (hidden // 64), 64], dtype=pl.INT8)
        pack_down(down_weight, down_packed)
        down_acc = pl.create_tensor([capacity, hidden], dtype=pl.INT32)
        dot_down(hidden_payload, down_packed, down_acc, active)
        dequant_down(down_acc, hidden_scale, down_scale, output, active)
        return output

    return shared_expert


def make_decode_routed_expert_w8a8(capacity: int, hidden: int, intermediate: int, clamp_limit: float = 10.0):
    """Alias the full expert kernel for one post-dispatch routed expert."""

    return make_decode_shared_expert_w8a8(capacity, hidden, intermediate, clamp_limit)


def make_decode_moe_router_inline(capacity, config):
    """Build full-precision gate projection and 384-expert top-six routing."""
    hidden = config["hidden_size"]
    experts = config["n_routed_experts"]
    topk = config["num_experts_per_tok"]
    route_scale = config["routed_scaling_factor"]
    sentinel_base = config["image_sentinel_base_id"]
    sentinel_end = sentinel_base + 5
    if experts != 384 or topk != 6:
        raise ValueError("The A2 router requires 384 experts and six selected experts")
    if hidden % DECODE_ROUTER_D_TILE:
        raise ValueError("The gate hidden size must be divisible by D_TILE")

    @pl.jit.inline
    def moe_router_inline(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        router_x: pl.Tensor[[capacity, hidden], pl.FP32],
        gate_weight: pl.Tensor[[experts, hidden], pl.FP32],
        text_bias: pl.Tensor[[experts], pl.FP32],
        vision_bias: pl.Tensor[[experts], pl.FP32],
        input_ids: pl.Tensor[[capacity], pl.INT64],
        logits: pl.Tensor[[capacity, experts], pl.FP32],
        route_ids: pl.Tensor[[capacity, topk], pl.INT32],
        route_weights: pl.Tensor[[capacity, topk], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(active * (experts // DECODE_ROUTER_EXPERT_TILE), name_hint="a2_moe_gate_fp32"):
            token = block // (experts // DECODE_ROUTER_EXPERT_TILE)
            expert = block % (experts // DECODE_ROUTER_EXPERT_TILE) * DECODE_ROUTER_EXPERT_TILE
            dot_sum = pl.full([1, DECODE_ROUTER_EXPERT_TILE], dtype=pl.FP32, value=0.0)
            for inner in pl.range(0, hidden, DECODE_ROUTER_D_TILE):
                source = router_x[token:token + 1, inner:inner + DECODE_ROUTER_D_TILE]
                weight = gate_weight[expert:expert + DECODE_ROUTER_EXPERT_TILE, inner:inner + DECODE_ROUTER_D_TILE]
                product = pl.col_expand_mul(weight, source)
                partial = pl.reshape(pl.row_sum(product), [1, DECODE_ROUTER_EXPERT_TILE])
                dot_sum = pl.add(dot_sum, partial)
            logits[token:token + 1, expert:expert + DECODE_ROUTER_EXPERT_TILE] = dot_sum

        raw_scores = pl.create_tensor([capacity, experts], dtype=pl.FP32)
        selected_ids = pl.create_tensor([capacity, DECODE_ROUTER_TOPK_TILE], dtype=pl.INT32)
        for token in pl.spmd(active, name_hint="a2_moe_route_sort"):
            gate_logits = logits[token:token + 1, :]
            positive = pl.maximum(gate_logits, 0.0)
            negative_abs = pl.neg(pl.abs(gate_logits))
            exponential = pl.exp(negative_abs)
            log_tail = pl.log(pl.add(exponential, 1.0))
            softplus_log = pl.add(positive, log_tail)
            negative_shift = pl.sub(pl.neg(gate_logits), 10.0)
            negative_mask = pl.minimum(pl.maximum(negative_shift, 0.0), 1.0)
            negative_exp = pl.exp(pl.minimum(gate_logits, 0.0))
            negative_floor = pl.mul(negative_mask, negative_exp)
            softplus = pl.maximum(softplus_log, negative_floor)
            scores = pl.sqrt(softplus)
            raw_scores[token:token + 1, :] = scores

            bias = pl.reshape(text_bias[:], [1, experts])
            token_id = pl.read(input_ids, [token])
            if token_id >= sentinel_base:
                if token_id < sentinel_end:
                    bias = pl.reshape(vision_bias[:], [1, experts])
            biased_scores = pl.add(scores, bias)
            index_range = pl.arange(0, [1, experts], dtype=pl.UINT32)
            sorted_32 = pl.sort32(biased_scores, index_range)
            sorted_64 = pl.mrgsort(sorted_32, block_len=64)
            first = sorted_64[:, 0:256]
            second = sorted_64[:, 256:512]
            third = sorted_64[:, 512:768]
            sorted_pairs = pl.mrgsort(first, second, third)
            top_pairs = sorted_pairs[:, 0:16]
            top_ids = pl.gather(top_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            selected_ids[token:token + 1, :] = top_ids

        # Each row is only 6 INT32/FP32 values (24 bytes), so parallel scalar
        # stores would share a 64-byte cache line and lose neighboring rows.
        # Serialize this small write-back until the output is padded or tile
        # stores can provide an atomic full-line update.
        for _block in pl.spmd(1, name_hint="a2_moe_route_normalize"):
            for token in pl.range(active):
                score_sum = pl.cast(0.0, pl.FP32)
                for rank in pl.range(topk):
                    selected_expert_idx = pl.cast(pl.read(selected_ids, [token, rank]), pl.INDEX)
                    score = pl.read(raw_scores, [token, selected_expert_idx])
                    score_sum = score_sum + score
                for rank in pl.range(topk):
                    selected_expert_idx = pl.cast(pl.read(selected_ids, [token, rank]), pl.INDEX)
                    score = pl.read(raw_scores, [token, selected_expert_idx])
                    normalized = (score / score_sum) * route_scale
                    pl.write(route_ids, [token, rank], pl.cast(selected_expert_idx, pl.INT32))
                    pl.write(route_weights, [token, rank], normalized)
        return logits, route_ids, route_weights

    return moe_router_inline


def make_decode_moe_router(capacity, config):
    """Build the standalone MoE router entry with untouched inactive rows."""
    hidden = config["hidden_size"]
    experts = config["n_routed_experts"]
    topk = config["num_experts_per_tok"]
    compute = make_decode_moe_router_inline(capacity, config)

    @pl.jit
    def moe_router(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        router_x: pl.Tensor[[capacity, hidden], pl.FP32],
        gate_weight: pl.Tensor[[experts, hidden], pl.FP32],
        text_bias: pl.Tensor[[experts], pl.FP32],
        vision_bias: pl.Tensor[[experts], pl.FP32],
        input_ids: pl.Tensor[[capacity], pl.INT64],
        logits: pl.InOut[pl.Tensor[[capacity, experts], pl.FP32]],
        route_ids: pl.InOut[pl.Tensor[[capacity, topk], pl.INT32]],
        route_weights: pl.InOut[pl.Tensor[[capacity, topk], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        compute(x, router_x, gate_weight, text_bias, vision_bias, input_ids, logits, route_ids, route_weights, active)
        return logits, route_ids, route_weights

    return moe_router


def make_decode_dispatch_layout(tokens):
    """Map native count-mode dispatch into the full384 expert ABI."""
    routes = tokens * DECODE_DISPATCH_TOPK

    @pl.jit
    def dispatch_layout(
        payload: pl.Tensor[[routes, DECODE_DISPATCH_D], pl.INT8],
        scales: pl.Tensor[[routes], pl.FP32],
        counts: pl.Tensor[[DECODE_DISPATCH_EXPERTS], pl.INT64],
        expanded: pl.Tensor[[routes], pl.INT32],
        weights: pl.Tensor[[tokens, DECODE_DISPATCH_TOPK], pl.FP32],
        recv_x: pl.InOut[pl.Tensor[[DECODE_DISPATCH_EXPERTS, DECODE_DISPATCH_RECV_MAX, DECODE_DISPATCH_D], pl.INT8]],
        recv_scale: pl.InOut[pl.Tensor[[DECODE_DISPATCH_EXPERTS, DECODE_DISPATCH_RECV_MAX], pl.FP32]],
        recv_weights: pl.InOut[pl.Tensor[[DECODE_DISPATCH_EXPERTS, DECODE_DISPATCH_RECV_MAX], pl.FP32]],
        recv_count: pl.InOut[pl.Tensor[[DECODE_DISPATCH_EXPERTS, 1], pl.INT32]],
        route_map: pl.InOut[pl.Tensor[[routes, DECODE_DISPATCH_MAP_PAD], pl.INT32]],
    ):
        # Serialize narrow scalar stores that can share a cache line.
        for _block in pl.spmd(1, name_hint="a2_moe_dispatch_metadata"):
            begin = pl.cast(0, pl.INT32)
            for expert in pl.range(DECODE_DISPATCH_EXPERTS):
                count = pl.cast(pl.read(counts, [expert]), pl.INT32)
                pl.write(recv_count, [expert, 0], count)
                for original in pl.range(routes):
                    dispatched = pl.read(expanded, [original])
                    if dispatched >= begin and dispatched < begin + count:
                        row = dispatched - begin
                        route_weight = pl.read(weights, [original // DECODE_DISPATCH_TOPK, original % DECODE_DISPATCH_TOPK])
                        token_scale = pl.read(scales, [dispatched])
                        pl.write(recv_scale, [expert, row], token_scale)
                        pl.write(recv_weights, [expert, row], route_weight)
                        target = pl.cast(expert * DECODE_DISPATCH_RECV_MAX, pl.INT32) + row
                        pl.write(route_map, [original, 0], target)
                begin = begin + count

        recv_flat = pl.reshape(recv_x, [DECODE_DISPATCH_EXPERTS * DECODE_DISPATCH_RECV_MAX, DECODE_DISPATCH_D])
        for block in pl.spmd(routes * (DECODE_DISPATCH_D // DECODE_DISPATCH_D_TILE), name_hint="a2_moe_dispatch_payload"):
            original = block // (DECODE_DISPATCH_D // DECODE_DISPATCH_D_TILE)
            column = block % (DECODE_DISPATCH_D // DECODE_DISPATCH_D_TILE) * DECODE_DISPATCH_D_TILE
            dispatched = pl.read(expanded, [original])
            target = pl.read(route_map, [original, 0])
            value = payload[dispatched:dispatched + 1, column:column + DECODE_DISPATCH_D_TILE]
            recv_flat[target:target + 1, column:column + DECODE_DISPATCH_D_TILE] = value
        return recv_x, recv_scale, recv_weights, recv_count, route_map

    return dispatch_layout


def make_decode_route_restore(tokens):
    """Gather every actual weighted expert result in original token/top6 order."""
    routes = tokens * DECODE_DISPATCH_TOPK

    @pl.jit
    def route_restore(
        recv_y: pl.Tensor[[DECODE_DISPATCH_EXPERTS, DECODE_DISPATCH_RECV_MAX, DECODE_DISPATCH_D], pl.FP32],
        route_map: pl.Tensor[[routes, DECODE_DISPATCH_MAP_PAD], pl.INT32],
        dispatch_y: pl.Out[pl.Tensor[[routes, DECODE_DISPATCH_D], pl.FP32]],
    ):
        recv_flat = pl.reshape(recv_y, [DECODE_DISPATCH_EXPERTS * DECODE_DISPATCH_RECV_MAX, DECODE_DISPATCH_D])
        for block in pl.spmd(routes * (DECODE_DISPATCH_D // DECODE_DISPATCH_D_TILE), name_hint="a2_moe_route_restore"):
            original = block // (DECODE_DISPATCH_D // DECODE_DISPATCH_D_TILE)
            column = block % (DECODE_DISPATCH_D // DECODE_DISPATCH_D_TILE) * DECODE_DISPATCH_D_TILE
            source = pl.read(route_map, [original, 0])
            value = recv_flat[source:source + 1, column:column + DECODE_DISPATCH_D_TILE]
            dispatch_y[original:original + 1, column:column + DECODE_DISPATCH_D_TILE] = value
        return dispatch_y

    return route_restore


def decode_shared_make_shared_activation(capacity, intermediate, limit, fused_activation):
    """Select native fused FP32 or the preserved standalone BF16 rounding."""
    width = intermediate * 2
    tile = 256
    dequantize = _decode_shared_make_dequantize(capacity, width)

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
            row_blocks = (active + DECODE_SHARED_VEC_M_TILE - 1) // DECODE_SHARED_VEC_M_TILE
            for block in pl.spmd(row_blocks * (intermediate // tile), name_hint="a2_moe_swiglu"):
                token = block // (intermediate // tile) * DECODE_SHARED_VEC_M_TILE
                column = block % (intermediate // tile) * tile
                rows = pl.min(DECODE_SHARED_VEC_M_TILE, active - token)
                up_column = intermediate + column
                gate_bf16 = pl.slice(gate_up, [DECODE_SHARED_VEC_M_TILE, tile], [token, column], valid_shape=[rows, tile])
                up_bf16 = pl.slice(gate_up, [DECODE_SHARED_VEC_M_TILE, tile], [token, up_column], valid_shape=[rows, tile])
                gate_bf16 = pl.set_validshape(pl.fillpad(gate_bf16, pad_value=pl.PadValue.zero), DECODE_SHARED_VEC_M_TILE, tile)
                up_bf16 = pl.set_validshape(pl.fillpad(up_bf16, pad_value=pl.PadValue.zero), DECODE_SHARED_VEC_M_TILE, tile)
                gate = pl.minimum(pl.cast(gate_bf16, pl.FP32), limit)
                up = pl.maximum(pl.minimum(pl.cast(up_bf16, pl.FP32), limit), -limit)
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate)), 1.0))
                sigmoid_bf16 = pl.cast(sigmoid, pl.BF16, mode="rint")
                silu_bf16 = pl.cast(pl.mul(gate, pl.cast(sigmoid_bf16, pl.FP32)), pl.BF16, mode="rint")
                activated = pl.mul(pl.cast(silu_bf16, pl.FP32), up)
                narrowed = pl.cast(activated, pl.BF16, mode="rint")
                output[token:token + DECODE_SHARED_VEC_M_TILE, column:column + tile] = pl.set_validshape(narrowed, rows, tile)
            return output

        return bf16_activation

    @pl.jit.inline
    def fp32_activation(
        accumulator: pl.Tensor[[capacity, width], pl.INT32],
        channels: pl.Tensor[[width], pl.FP32],
        scales: pl.Tensor[[capacity, 8], pl.FP32],
        output: pl.Tensor[[capacity, intermediate], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        row_blocks = (active + DECODE_SHARED_VEC_M_TILE - 1) // DECODE_SHARED_VEC_M_TILE
        for block in pl.spmd(row_blocks * (intermediate // tile), name_hint="a2_moe_swiglu_fused"):
            token = block // (intermediate // tile) * DECODE_SHARED_VEC_M_TILE
            column = block % (intermediate // tile) * tile
            rows = pl.min(DECODE_SHARED_VEC_M_TILE, active - token)
            up_column = intermediate + column
            gate_i32 = pl.slice(accumulator, [DECODE_SHARED_VEC_M_TILE, tile], [token, column], valid_shape=[rows, tile])
            up_i32 = pl.slice(accumulator, [DECODE_SHARED_VEC_M_TILE, tile], [token, up_column], valid_shape=[rows, tile])
            gate_channels = pl.reshape(channels[column:column + tile], [1, tile])
            up_channels = pl.reshape(channels[up_column:up_column + tile], [1, tile])
            row_scale = pl.row_min(scales[token:token + DECODE_SHARED_VEC_M_TILE, :])
            gate = pl.row_expand_mul(pl.col_expand_mul(pl.cast(gate_i32, pl.FP32), gate_channels), row_scale)
            up = pl.row_expand_mul(pl.col_expand_mul(pl.cast(up_i32, pl.FP32), up_channels), row_scale)
            gate = pl.set_validshape(pl.fillpad(gate, pad_value=pl.PadValue.zero), DECODE_SHARED_VEC_M_TILE, tile)
            up = pl.set_validshape(pl.fillpad(up, pad_value=pl.PadValue.zero), DECODE_SHARED_VEC_M_TILE, tile)
            gate = pl.minimum(gate, limit)
            up = pl.maximum(pl.minimum(up, limit), -limit)
            denominator = pl.add(pl.exp(pl.neg(gate)), 1.0)
            silu = pl.div(gate, denominator)
            activated = pl.mul(silu, up)
            output[token:token + DECODE_SHARED_VEC_M_TILE, column:column + tile] = pl.set_validshape(activated, rows, tile)
        return output

    return fp32_activation


def make_decode_shared_moe_w8a8_inline(capacity, config, *, fused_activation=False):
    """Build shared gate/up, clamped SwiGLU, and down projection."""
    hidden = config["hidden_size"]
    intermediate = config["moe_intermediate_size"]
    gate_up_width = intermediate * 2
    limit = config["swiglu_limit"]
    gate_up_packed_rows = hidden * (gate_up_width // DECODE_SHARED_N_TILE)
    down_packed_rows = intermediate * (hidden // DECODE_SHARED_N_TILE)
    activation_dtype = pl.FP32 if fused_activation else pl.BF16
    activate = decode_shared_make_shared_activation(capacity, intermediate, limit, fused_activation)

    quant_x = _decode_shared_make_quantize(capacity, hidden)
    quant_hidden = _decode_shared_make_quantize(capacity, intermediate, activation_dtype)
    pack_gate_up = _decode_shared_make_pack_weight(hidden, gate_up_width)
    pack_down = _decode_shared_make_pack_weight(intermediate, hidden)
    dot_gate_up = _decode_shared_make_dot(capacity, hidden, gate_up_width)
    dot_down = _decode_shared_make_dot(capacity, intermediate, hidden)
    # vLLM's quantized down projection emits BF16 before the shared/routed
    # combine. Keep the fused activation's FP32 intermediates, but preserve
    # that output boundary.
    dequant_down = _decode_shared_make_dequantize(capacity, hidden)

    if fused_activation:

        @pl.jit.inline
        def shared_moe_w8a8_inline_fused(
            x: pl.Tensor[[capacity, hidden], pl.BF16],
            gate_up_weight: pl.Tensor[[hidden, gate_up_width], pl.INT8],
            gate_up_scale: pl.Tensor[[gate_up_width], pl.FP32],
            down_weight: pl.Tensor[[intermediate, hidden], pl.INT8],
            down_scale: pl.Tensor[[hidden], pl.FP32],
            output: pl.Tensor[[capacity, hidden], pl.BF16],
            active: pl.Scalar[pl.INT32],
        ):
            x_payload = pl.create_tensor([capacity, hidden], dtype=pl.INT8)
            x_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
            quant_x(x, x_payload, x_scale, active)

            gate_up_packed = pl.create_tensor([gate_up_packed_rows, DECODE_SHARED_N_TILE], dtype=pl.INT8)
            pack_gate_up(gate_up_weight, gate_up_packed)
            gate_up_acc = pl.create_tensor([capacity, gate_up_width], dtype=pl.INT32)
            dot_gate_up(x_payload, gate_up_packed, gate_up_acc, active)
            activation = pl.create_tensor([capacity, intermediate], dtype=activation_dtype)
            activate(gate_up_acc, gate_up_scale, x_scale, activation, active)

            hidden_payload = pl.create_tensor([capacity, intermediate], dtype=pl.INT8)
            hidden_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
            quant_hidden(activation, hidden_payload, hidden_scale, active)
            down_packed = pl.create_tensor([down_packed_rows, DECODE_SHARED_N_TILE], dtype=pl.INT8)
            pack_down(down_weight, down_packed)
            down_acc = pl.create_tensor([capacity, hidden], dtype=pl.INT32)
            dot_down(hidden_payload, down_packed, down_acc, active)
            dequant_down(down_acc, down_scale, hidden_scale, output, active)
            return output

        return shared_moe_w8a8_inline_fused

    @pl.jit.inline
    def shared_moe_w8a8_inline(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        gate_up_weight: pl.Tensor[[hidden, gate_up_width], pl.INT8],
        gate_up_scale: pl.Tensor[[gate_up_width], pl.FP32],
        down_weight: pl.Tensor[[intermediate, hidden], pl.INT8],
        down_scale: pl.Tensor[[hidden], pl.FP32],
        output: pl.Tensor[[capacity, hidden], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        x_payload = pl.create_tensor([capacity, hidden], dtype=pl.INT8)
        x_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quant_x(x, x_payload, x_scale, active)

        gate_up_packed = pl.create_tensor([gate_up_packed_rows, DECODE_SHARED_N_TILE], dtype=pl.INT8)
        pack_gate_up(gate_up_weight, gate_up_packed)
        gate_up_acc = pl.create_tensor([capacity, gate_up_width], dtype=pl.INT32)
        dot_gate_up(x_payload, gate_up_packed, gate_up_acc, active)
        activation = pl.create_tensor([capacity, intermediate], dtype=activation_dtype)
        activate(gate_up_acc, gate_up_scale, x_scale, activation, active)

        hidden_payload = pl.create_tensor([capacity, intermediate], dtype=pl.INT8)
        hidden_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quant_hidden(activation, hidden_payload, hidden_scale, active)
        down_packed = pl.create_tensor([down_packed_rows, DECODE_SHARED_N_TILE], dtype=pl.INT8)
        pack_down(down_weight, down_packed)
        down_acc = pl.create_tensor([capacity, hidden], dtype=pl.INT32)
        dot_down(hidden_payload, down_packed, down_acc, active)
        dequant_down(down_acc, down_scale, hidden_scale, output, active)
        return output

    return shared_moe_w8a8_inline


def make_decode_shared_moe_w8a8(capacity, config, *, fused_activation=False):
    """Build a standalone JIT kernel for the shared expert."""
    hidden = config["hidden_size"]
    intermediate = config["moe_intermediate_size"]
    gate_up_width = intermediate * 2
    compute = make_decode_shared_moe_w8a8_inline(capacity, config, fused_activation=fused_activation)

    @pl.jit
    def shared_moe_w8a8(
        x: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        gate_up_weight: pl.Tensor[[hidden, gate_up_width], pl.INT8],
        gate_up_scale: pl.Tensor[[gate_up_width], pl.FP32],
        down_weight: pl.Tensor[[intermediate, hidden], pl.INT8],
        down_scale: pl.Tensor[[hidden], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute(x, gate_up_weight, gate_up_scale, down_weight, down_scale, output, active)
        return x, output

    return shared_moe_w8a8


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


def golden_moe_combine(tensors):
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
                acc += value * tensors["route_weights"][token, slot]
        routed = acc.to(torch.bfloat16)
        tensors["routed"][token] = routed
        tensors["output"][token] = (routed.float() + tensors["shared"][token].float()).to(torch.bfloat16)


def golden_shared_moe_w8a8(tensors, *, fused_activation=False, fused_channel_dtype=pl.FP16):
    """Evaluate channel scaling, activation rounding and requantization for validation."""
    import torch

    active = int(tensors["active"])
    if not active:
        return

    def quantize(value):
        value = value.float()
        maximum = value.abs().amax(-1, keepdim=True).clamp_min(1e-30)
        scale = maximum * (1.0 / 127.0)
        quantized = (value * (127.0 / maximum)).round().clamp(-127, 127).to(torch.int32)
        return quantized, scale

    def dot(payload, weight, channels, scale):
        accumulator = payload @ weight.to(torch.int32)
        channel = accumulator.float() * channels.float()
        if fused_activation and fused_channel_dtype == pl.FP16:
            channel = channel.to(torch.float16).float()
        return channel * scale

    payload, scale = quantize(tensors["x"][:active])
    gate_up = dot(payload, tensors["gate_up_weight"], tensors["gate_up_scale"], scale)
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
    payload, scale = quantize(activation)
    down = dot(payload, tensors["down_weight"], tensors["down_scale"], scale)
    tensors["output"][:active] = down.to(torch.bfloat16)


def decode_moe_shared_expert_reference(
    value: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
    clamp_limit: float = 10.0,
) -> torch.Tensor:
    """Reference for a shared W8A8 SwiGLU expert."""

    gate_up = decode_moe_w8a8_expert_linear(value, gate_up_weight, gate_up_scale, torch.bfloat16).float()
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.clamp(max=clamp_limit)
    up = up.clamp(-clamp_limit, clamp_limit)
    # vLLM-Ascend's A2/A3 clipped SwiGLU performs both products in BF16.
    sigmoid_bf16 = torch.sigmoid(gate).to(torch.bfloat16)
    silu_bf16 = gate.to(torch.bfloat16) * sigmoid_bf16
    hidden = silu_bf16 * up.to(torch.bfloat16)
    return decode_moe_w8a8_expert_linear(hidden, down_weight, down_scale)


def decode_moe_routed_expert_reference(
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
            gate = decode_moe_w8a8_expert_linear(value[token:token + 1], w1[expert], w1_scale[expert]).float()
            up = decode_moe_w8a8_expert_linear(value[token:token + 1], w3[expert], w3_scale[expert]).float()
            gate = gate.clamp(max=clamp_limit)
            up = up.clamp(-clamp_limit, clamp_limit)
            sigmoid_bf16 = torch.sigmoid(gate).to(torch.bfloat16)
            silu_bf16 = gate.to(torch.bfloat16) * sigmoid_bf16
            hidden = silu_bf16 * up.to(torch.bfloat16)
            down = decode_moe_w8a8_expert_linear(hidden, w2[expert], w2_scale[expert]).float()
            result[token] += down[0] * router_weights[token, route].float()
    return result.to(torch.bfloat16)


def validate_moe_router(*, device=0, compile_only=False, platform="a2a3", sim=False):
    """Validate natural and controlled routing using the product router definition."""
    from functools import partial

    import torch
    from golden import ScalarSpec, TensorSpec, run

    torch.manual_seed(41)
    capacity, hidden, experts, topk = 16, 256, 384, 6
    config = {"hidden_size": hidden, "n_routed_experts": experts, "num_experts_per_tok": topk,
              "routed_scaling_factor": 1.5, "image_sentinel_base_id": 129264}
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
            result = run(
                         compile_only=compile_only,
                fn=make_moe_router(capacity, settings),
                specs=specs,
                golden_fn=partial(golden_moe_router, config=settings),
                config={"backend_type": platform, "execution_mode": sim, "device_id": device},
                rtol=0.01,
                atol=0.001,
            )
            if not result.passed:
                raise RuntimeError(f"Router validation failed: controlled={controlled}, active={active}: {result.error}")


def validate_received_routes(packet, rank, layout, routes, valid_by_rank=None):
    """Check transported source, expert ownership and bounded bucket metadata."""
    counts = [int(packet["counts"][peer, 0]) for peer in range(layout.ep)]
    if any(not 0 <= count <= routes for count in counts):
        raise ValueError("HCCL received bucket count exceeds its fixed capacity")
    for peer, count in enumerate(counts):
        for row in range(count):
            source, token, slot, expert = packet["routes"][peer * routes + row].tolist()
            if source != peer or layout.owner(expert) != rank or token < 0 or slot < 0:
                raise RuntimeError("HCCL delivered inconsistent route metadata")
            if valid_by_rank is not None and token >= valid_by_rank[peer]:
                raise RuntimeError("HCCL dispatched a masked source row")
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

    capacity, hidden, experts, topk = 16, 512, 384, 6
    for world in (2, 16):
        routes, local_experts = capacity * topk, experts // world
        packet_rows = world * routes
        expert_capacity = ((capacity * world + 15) // 16) * 16
        slab_rows = local_experts * expert_capacity
        for active in (0, 1, 7, 16):
            ids = torch.tensor([0, 1, 2, experts - 3, experts - 2, experts - 1], dtype=torch.int32)
            ids = ids.expand(capacity, -1).clone()
            pack_specs = [
                TensorSpec("x", [capacity, hidden], torch.bfloat16, init_value=torch.randn),
                TensorSpec("route_ids", [capacity, topk], torch.int32, init_value=ids),
                TensorSpec("sent_routes", [packet_rows, 4], torch.int64, init_value=-1),
                TensorSpec("sent_hidden", [packet_rows, hidden], torch.bfloat16, init_value=0.0),
                TensorSpec("sent_counts", [world, DISPATCH_MAP_PAD], torch.int32, init_value=0),
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
        for token in range(valid):
            for slot in range(topk):
                peer = slot % world
                row = int(counts[peer, 0])
                metadata[peer * routes + row] = torch.tensor([0, token, slot, peer * 192 + slot])
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


def validate_shared_moe(*, device=0, compile_only=False, platform="a2a3", sim=False):
    """Validate standalone BF16 and fused FP16/FP32 channel boundaries."""
    from functools import partial

    import torch
    from golden import ScalarSpec, TensorSpec, run

    torch.manual_seed(41)
    capacity, hidden, intermediate = 16, 256, 256
    config = {"hidden_size": hidden, "moe_intermediate_size": intermediate, "swiglu_limit": 10.0}
    for fused, channel_dtype in ((False, pl.FP32), (True, pl.FP16), (True, pl.FP32)):
        for active in (0, 1, 7, 16):
            x = torch.randn(capacity, hidden, dtype=torch.bfloat16)
            gate_up = torch.randint(-32, 33, (hidden, 2 * intermediate), dtype=torch.int8)
            down = torch.randint(-32, 33, (intermediate, hidden), dtype=torch.int8)
            specs = [
                TensorSpec("x", list(x.shape), x.dtype, init_value=x),
                TensorSpec("gate_up_weight", list(gate_up.shape), gate_up.dtype, init_value=gate_up),
                TensorSpec("gate_up_scale", [2 * intermediate], torch.float32, init_value=0.002),
                TensorSpec("down_weight", list(down.shape), down.dtype, init_value=down),
                TensorSpec("down_scale", [hidden], torch.float32, init_value=0.002),
                TensorSpec("output", [capacity, hidden], torch.bfloat16, init_value=-12.5),
                ScalarSpec("active", torch.int32, active),
            ]
            result = run(
                compile_only=compile_only,
                fn=make_shared_moe_w8a8(
                    capacity, config, fused_activation=fused, fused_channel_dtype=channel_dtype,
                ),
                specs=specs,
                golden_fn=partial(
                    golden_shared_moe_w8a8, fused_activation=fused, fused_channel_dtype=channel_dtype,
                ),
                config={"backend_type": platform, "execution_mode": sim, "device_id": device},
                rtol=0.05,
                atol=0.05,
            )
            if not result.passed:
                mode = f"fused={fused}, channel={channel_dtype}, active={active}"
                raise RuntimeError(f"Expert validation failed: {mode}: {result.error}")


# The EP transport below consumes the Decode-specialized MoE builders above.
# Keep the aliases at the composition boundary so this module remains the
# single Decode MoE owner after the operator and transport implementations are
# combined.
make_moe_router_inline = make_decode_moe_router_inline
make_moe_router = make_decode_moe_router
make_shared_moe_w8a8_inline = make_decode_shared_moe_w8a8_inline
make_shared_moe_w8a8 = make_decode_shared_moe_w8a8
make_dispatch_layout = make_decode_dispatch_layout
make_route_restore = make_decode_route_restore
make_routed_expert_projection = make_decode_routed_expert_projection
make_routed_expert_w8a8 = make_decode_routed_expert_w8a8
make_shared_expert_projection = make_decode_shared_expert_projection
make_shared_expert_w8a8 = make_decode_shared_expert_w8a8
make_w8a8_expert_projection = make_decode_w8a8_expert_projection
w8a8_expert_linear = decode_moe_w8a8_expert_linear
routed_expert_reference = decode_moe_routed_expert_reference
shared_expert_reference = decode_moe_shared_expert_reference


# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4.1 A2/A3 W8A8 Decode MoE for EP2 or EP16 with TP1.

This entry owns the product-side expert-parallel transport. It partitions all
384 routed experts across the selected rank group and moves PyPTO tensors
across rank windows. The vLLM reference process is outside this module.
"""

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import pypto.language.distributed as pld

from .config import FLASH
from .decode_network_decode_config import _parse_parallel_size
from .decode_network_quant import make_dynamic_int8_quantize


# Expert-parallel topology and Decode capacity.
EP_SIZE = _parse_parallel_size("ep", 16)
if EP_SIZE not in (2, 16):
    raise ValueError("Decode transport supports EP2 or EP16 with TP1")
N_LOCAL_EXPERTS = 384 // EP_SIZE
N_EXPERTS = 384
TOPK = 6
T = 16
D = 5120
MOE_INTER = 2304
MOE_INTER_PAD = 2560
SWIGLU_LIMIT = 10.0
N_ROUTES = T * TOPK
LANE_CAP = N_ROUTES
LANE_ROWS = EP_SIZE * LANE_CAP
RECV_MAX = max(256, ((EP_SIZE * T + 63) // 64) * 64)
AUX_SCALE, AUX_WEIGHT, AUX_ROUTE, AUX_EXPERT = 0, 1, 2, 3
AUX_PAD = 16
COUNT_PAD = 16
# Each Set signal has one source writer and a separate 64-byte cache line.
SIGNAL_PAD = 16

# The routed leaf reads its complete local shard dimensions during import.
os.environ["A2_MOE_LOCAL_EXPERTS"] = str(N_LOCAL_EXPERTS)
os.environ["A2_MOE_RECV_MAX"] = str(RECV_MAX)


_ROUTER_CONFIG = asdict(FLASH)
_ROUTER_CONFIG["image_sentinel_base_id"] = 129257
ROUTER = make_moe_router_inline(T, _ROUTER_CONFIG)
# CANN's fused EP2 dispatch quantizes normalized rows with FP16 truncation
# followed by a rounded INT8 store.  The legacy mode remains available for a
# controlled comparison while other model paths keep their existing contract.
QUANTIZE = make_dynamic_int8_quantize(
    T,
    D,
    cann_dispatch=os.environ.get("PYPTO_EP2_INPUT_QUANT", "legacy") == "cann",
)
SHARED = make_shared_moe_w8a8_inline(T, _ROUTER_CONFIG, fused_activation=True)
# Temporary NPU diagnosis switch.  It is consumed only by the local EP2
# runner to isolate the two additive MoE branches; the formal path remains
# the default ``sum`` route.
_DEBUG_COMPONENT = {"routed_only": 1, "shared_only": 2}.get(
    os.environ.get("PYPTO_EP2_DEBUG_COMPONENT", "sum"), 0
)


@pl.jit.inline
def decode_ep2_pack(
    route_ids: pl.Tensor[[T, TOPK], pl.INT32],
    route_weights: pl.Tensor[[T, TOPK], pl.FP32],
    x_int8: pl.Tensor[[T, D], pl.INT8],
    x_scale: pl.Tensor[[T, 1], pl.FP32],
    lane_x: pld.DistributedTensor[[LANE_ROWS, D], pl.INT8],
    lane_aux: pld.DistributedTensor[[LANE_ROWS, AUX_PAD], pl.FP32],
    lane_count: pld.DistributedTensor[[EP_SIZE, COUNT_PAD], pl.INT32],
    lane_ready: pld.DistributedTensor[[EP_SIZE, SIGNAL_PAD], pl.INT32],
    active: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Send each route to its owning rank and publish empty buckets too."""

    active_rows = pl.min(pl.max(pl.cast(active, pl.INDEX), 0), T)
    destination_counts = pl.create_tensor([EP_SIZE], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dsv41_ep2_pack_count") as count_tid:
        for destination in pl.range(EP_SIZE):
            pl.write(destination_counts, [destination], pl.cast(0, pl.INT32))
        for token in pl.range(active_rows):
            for route in pl.range(TOPK):
                expert = pl.cast(pl.read(route_ids, [token, route]), pl.INDEX)
                destination = expert // N_LOCAL_EXPERTS
                old = pl.read(destination_counts, [destination])
                pl.write(destination_counts, [destination], pl.cast(old + 1, pl.INT32))

    with pl.spmd(EP_SIZE, name_hint="dsv41_ep2_pack_payload", deps=[count_tid]) as push_tid:
        destination = pl.tile.get_block_idx()
        lane_slot = pl.cast(0, pl.INDEX)
        for token in pl.range(active_rows):
            for route in pl.range(TOPK):
                expert = pl.cast(pl.read(route_ids, [token, route]), pl.INDEX)
                target = expert // N_LOCAL_EXPERTS
                if target == destination:
                    lane_row = pl.cast(rank, pl.INDEX) * LANE_CAP + lane_slot
                    pld.tensor.put(
                        dst=lane_x,
                        peer=destination,
                        src=x_int8,
                        dst_offsets=[lane_row, 0],
                        src_offsets=[token, 0],
                        shape=[1, D],
                    )
                    metadata = pl.tile.full([1, AUX_PAD], dtype=pl.FP32, value=0.0)
                    pl.tile.write(metadata, [0, AUX_SCALE], pl.read(x_scale, [token, 0]))
                    pl.tile.write(metadata, [0, AUX_WEIGHT], pl.read(route_weights, [token, route]))
                    route_slot = pl.cast(token * TOPK + route, pl.INT32)
                    pl.tile.write(metadata, [0, AUX_ROUTE], pl.cast(route_slot, pl.FP32))
                    local_expert = expert - pl.cast(target * N_LOCAL_EXPERTS, pl.INDEX)
                    local_expert_i32 = pl.cast(local_expert, pl.INT32)
                    pl.tile.write(metadata, [0, AUX_EXPERT], pl.cast(local_expert_i32, pl.FP32))
                    pld.tile.remote_store(metadata, target=lane_aux, peer=destination, offsets=[lane_row, 0])
                    lane_slot = lane_slot + 1

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dsv41_ep2_pack_publish", deps=[push_tid]) as publish_tid:
        for destination in pl.range(EP_SIZE):
            count_tile = pl.tile.full([1, COUNT_PAD], dtype=pl.INT32, value=0)
            pl.tile.write(count_tile, [0, 0], pl.read(destination_counts, [destination]))
            pld.tile.remote_store(count_tile, target=lane_count, peer=destination, offsets=[rank, 0])
            if destination != rank:
                pld.system.notify(
                    target=lane_ready,
                    peer=destination,
                    offsets=[rank, 0],
                    value=epoch,
                    op=pld.NotifyOp.Set,
                )
    return publish_tid


@pl.jit.inline
def decode_ep2_arrive(
    lane_x: pld.DistributedTensor[[LANE_ROWS, D], pl.INT8],
    lane_aux: pld.DistributedTensor[[LANE_ROWS, AUX_PAD], pl.FP32],
    lane_count: pld.DistributedTensor[[EP_SIZE, COUNT_PAD], pl.INT32],
    lane_ready: pld.DistributedTensor[[EP_SIZE, SIGNAL_PAD], pl.INT32],
    route_x: pl.Tensor[[LANE_ROWS, D], pl.INT8],
    route_scale: pl.Tensor[[LANE_ROWS, 1], pl.FP32],
    route_weight: pl.Tensor[[LANE_ROWS, 1], pl.FP32],
    route_id: pl.Tensor[[LANE_ROWS, 1], pl.INT32],
    route_expert: pl.Tensor[[LANE_ROWS, 1], pl.INT32],
    route_source: pl.Tensor[[LANE_ROWS, 1], pl.INT32],
    route_count: pl.Tensor[[1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    pack_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Wait for both sources and flatten source-major receive lanes."""

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dsv41_ep2_arrive_wait", deps=[pack_tid]) as wait_tid:
        for source in pl.range(EP_SIZE):
            if source != rank:
                pld.system.wait(signal=lane_ready, offsets=[source, 0], expected=epoch, cmp=pld.WaitCmp.Ge)

    source_offsets = pl.create_tensor([EP_SIZE], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dsv41_ep2_arrive_prefix", deps=[wait_tid]) as prefix_tid:
        running = pl.cast(0, pl.INDEX)
        for source in pl.range(EP_SIZE):
            pl.write(source_offsets, [source], pl.cast(running, pl.INT32))
            rows = pl.cast(pl.read(lane_count, [source, 0]), pl.INDEX)
            for row in pl.range(rows):
                flat = running + row
                lane_row = source * LANE_CAP + row
                pl.write(route_scale, [flat, 0], pl.read(lane_aux, [lane_row, AUX_SCALE]))
                pl.write(route_weight, [flat, 0], pl.read(lane_aux, [lane_row, AUX_WEIGHT]))
                pl.write(route_id, [flat, 0], pl.cast(pl.read(lane_aux, [lane_row, AUX_ROUTE]), pl.INT32))
                pl.write(route_expert, [flat, 0], pl.cast(pl.read(lane_aux, [lane_row, AUX_EXPERT]), pl.INT32))
                pl.write(route_source, [flat, 0], pl.cast(source, pl.INT32))
            running = running + rows
        pl.write(route_count, [0], pl.cast(running, pl.INT32))

    with pl.spmd(EP_SIZE, name_hint="dsv41_ep2_arrive_copy", deps=[prefix_tid]):
        source = pl.tile.get_block_idx()
        base = pl.cast(pl.read(source_offsets, [source]), pl.INDEX)
        rows = pl.cast(pl.read(lane_count, [source, 0]), pl.INDEX)
        for row in pl.range(rows):
            flat = base + row
            lane_row = source * LANE_CAP + row
            route_x[flat:flat + 1, :] = lane_x[lane_row:lane_row + 1, :]
    return prefix_tid


@pl.jit.inline
def decode_ep2_regroup(
    route_expert: pl.Tensor[[LANE_ROWS, 1], pl.INT32],
    route_source: pl.Tensor[[LANE_ROWS, 1], pl.INT32],
    route_id: pl.Tensor[[LANE_ROWS, 1], pl.INT32],
    route_x: pl.Tensor[[LANE_ROWS, D], pl.INT8],
    route_scale: pl.Tensor[[LANE_ROWS, 1], pl.FP32],
    route_weight: pl.Tensor[[LANE_ROWS, 1], pl.FP32],
    route_count: pl.Tensor[[1], pl.INT32],
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_weight: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_source: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32],
    recv_route: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32],
    recv_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    expert_offsets: pl.Tensor[[N_LOCAL_EXPERTS + 1], pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Stable expert-major regroup with explicit counts and untouched tails."""

    total = pl.min(pl.max(pl.cast(pl.read(route_count, [0]), pl.INDEX), 0), LANE_ROWS)
    bucket = pl.create_tensor([LANE_ROWS], dtype=pl.INT32)
    expert_counts = pl.create_tensor([N_LOCAL_EXPERTS], dtype=pl.INT32)
    cursors = pl.create_tensor([N_LOCAL_EXPERTS], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dsv41_ep2_regroup_meta") as meta_tid:
        for expert in pl.range(N_LOCAL_EXPERTS):
            pl.write(expert_counts, [expert], pl.cast(0, pl.INT32))
        for row in pl.range(total):
            expert = pl.cast(pl.read(route_expert, [row, 0]), pl.INDEX)
            pl.write(expert_counts, [expert], pl.cast(pl.read(expert_counts, [expert]) + 1, pl.INT32))
        running = pl.cast(0, pl.INT32)
        for expert in pl.range(N_LOCAL_EXPERTS):
            pl.write(expert_offsets, [expert], running)
            pl.write(cursors, [expert], pl.cast(0, pl.INT32))
            pl.write(recv_count, [expert, 0], pl.read(expert_counts, [expert]))
            running = running + pl.read(expert_counts, [expert])
        pl.write(expert_offsets, [N_LOCAL_EXPERTS], running)
        for row in pl.range(total):
            expert = pl.cast(pl.read(route_expert, [row, 0]), pl.INDEX)
            local_slot = pl.read(cursors, [expert])
            pl.write(cursors, [expert], pl.cast(local_slot + 1, pl.INT32))
            bucket_slot = pl.read(expert_offsets, [expert]) + local_slot
            pl.write(bucket, [bucket_slot], pl.cast(row, pl.INT32))
            pl.write(recv_scale, [expert, local_slot], pl.read(route_scale, [row, 0]))
            pl.write(recv_weight, [expert, local_slot], pl.read(route_weight, [row, 0]))
            pl.write(recv_source, [expert, local_slot], pl.read(route_source, [row, 0]))
            pl.write(recv_route, [expert, local_slot], pl.read(route_id, [row, 0]))

    recv_x_flat = pl.reshape(recv_x, [N_LOCAL_EXPERTS * RECV_MAX, D])
    with pl.spmd(N_LOCAL_EXPERTS, name_hint="dsv41_ep2_regroup_payload", deps=[meta_tid]):
        expert = pl.tile.get_block_idx()
        base = pl.cast(pl.read(expert_offsets, [expert]), pl.INDEX)
        rows = pl.cast(pl.read(expert_counts, [expert]), pl.INDEX)
        for row in pl.range(rows):
            source_row = pl.cast(pl.read(bucket, [base + row]), pl.INDEX)
            target_row = expert * RECV_MAX + row
            recv_x_flat[target_row:target_row + 1, :] = route_x[source_row:source_row + 1, :]
    return meta_tid


@pl.jit.inline
def decode_ep2_return(
    expert_output: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.FP32],
    recv_source: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32],
    recv_route: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32],
    recv_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    slot_out: pld.DistributedTensor[[N_ROUTES, D], pl.FP32],
) -> pl.Scalar[pl.TASK_ID]:
    """Return weighted expert rows to the source rank's route slots."""

    flat = pl.reshape(expert_output, [N_LOCAL_EXPERTS * RECV_MAX, D])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dsv41_ep2_return", allow_early_resolve=True) as return_tid:
        for expert in pl.range(N_LOCAL_EXPERTS):
            rows = pl.cast(pl.read(recv_count, [expert, 0]), pl.INDEX)
            for row in pl.range(rows):
                source = pl.read(recv_source, [expert, row])
                route = pl.cast(pl.read(recv_route, [expert, row]), pl.INDEX)
                source_row = expert * RECV_MAX + row
                pld.tensor.put(
                    dst=slot_out,
                    peer=source,
                    src=flat,
                    dst_offsets=[route, 0],
                    src_offsets=[source_row, 0],
                    shape=[1, D],
                )
    return return_tid


@pl.jit.inline
def decode_ep2_combine(
    slot_out: pld.DistributedTensor[[N_ROUTES, D], pl.FP32],
    return_ready: pld.DistributedTensor[[EP_SIZE, SIGNAL_PAD], pl.INT32],
    shared_output: pl.Tensor[[T, D], pl.BF16],
    output: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    active: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    return_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Wait for every rank and add routed output to the shared expert."""

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dsv41_ep2_return_notify", deps=[return_tid]) as notify_tid:
        for peer in pl.range(EP_SIZE):
            if peer != rank:
                pld.system.notify(
                    target=return_ready,
                    peer=peer,
                    offsets=[rank, 0],
                    value=epoch,
                    op=pld.NotifyOp.Set,
                )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dsv41_ep2_return_wait", deps=[notify_tid]) as wait_tid:
        for source in pl.range(EP_SIZE):
            if source != rank:
                pld.system.wait(signal=return_ready, offsets=[source, 0], expected=epoch, cmp=pld.WaitCmp.Ge)

    active_rows = pl.min(pl.max(pl.cast(active, pl.INDEX), 0), T)
    with pl.spmd(T, name_hint="dsv41_ep2_combine", deps=[wait_tid]):
        token = pl.tile.get_block_idx()
        if token < active_rows:
            # The fused reference stores each routed expert result as BF16 and
            # widens that value at combine.  Keep this per-route rounding
            # boundary before the FP32 reduction; narrowing the aggregate
            # after summation changes the observable MoE output.
            routed = pl.full([1, D], dtype=pl.FP32, value=0.0)
            for route in pl.range(TOPK):
                slot = token * TOPK + route
                routed_row = pl.cast(slot_out[slot:slot + 1, :], pl.BF16, mode="rint")
                routed = pl.add(routed, pl.cast(routed_row, pl.FP32))
            shared = pl.cast(shared_output[token:token + 1, :], pl.FP32)
            if _DEBUG_COMPONENT == 1:
                combined = routed
            elif _DEBUG_COMPONENT == 2:
                combined = shared
            else:
                combined = pl.add(shared, routed)
            output[token:token + 1, :] = pl.cast(combined, pl.BF16, mode="rint")
        else:
            output[token:token + 1, :] = pl.full([1, D], dtype=pl.BF16, value=0.0)
    return wait_tid


@pl.jit(auto_scope=False)
def decode_ep2_rank(
    value: pl.Tensor[[T, D], pl.BF16],
    router_value: pl.Tensor[[T, D], pl.FP32],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    text_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    vision_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, 2560], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    shared_gate_up: pl.Tensor[[D, MOE_INTER * 2], pl.INT8],
    shared_gate_up_scale: pl.Tensor[[MOE_INTER * 2], pl.FP32],
    shared_down: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_down_scale: pl.Tensor[[D], pl.FP32],
    lane_x: pld.DistributedTensor[[LANE_ROWS, D], pl.INT8],
    lane_aux: pld.DistributedTensor[[LANE_ROWS, AUX_PAD], pl.FP32],
    lane_count: pld.DistributedTensor[[EP_SIZE, COUNT_PAD], pl.INT32],
    lane_ready: pld.DistributedTensor[[EP_SIZE, SIGNAL_PAD], pl.INT32],
    slot_out: pld.DistributedTensor[[N_ROUTES, D], pl.FP32],
    return_ready: pld.DistributedTensor[[EP_SIZE, SIGNAL_PAD], pl.INT32],
    logits: pl.InOut[pl.Tensor[[T, N_EXPERTS], pl.FP32]],
    route_ids: pl.InOut[pl.Tensor[[T, TOPK], pl.INT32]],
    route_weights: pl.InOut[pl.Tensor[[T, TOPK], pl.FP32]],
    output: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    active: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """Run router, distributed EP2, local routed experts, and shared expert."""

    x_int8 = pl.create_tensor([T, D], dtype=pl.INT8)
    x_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    QUANTIZE(value, x_int8, x_scale, active)
    pl.dump_tag(x_int8)
    pl.dump_tag(x_scale)
    ROUTER(value, router_value, gate_weight, text_bias, vision_bias, input_ids, logits, route_ids, route_weights, active)

    pack_tid = decode_ep2_pack(
        route_ids, route_weights, x_int8, x_scale, lane_x, lane_aux, lane_count, lane_ready, active, rank, epoch,
    )
    route_x = pl.create_tensor([LANE_ROWS, D], dtype=pl.INT8)
    route_scale = pl.create_tensor([LANE_ROWS, 1], dtype=pl.FP32)
    route_weight = pl.create_tensor([LANE_ROWS, 1], dtype=pl.FP32)
    route_id = pl.create_tensor([LANE_ROWS, 1], dtype=pl.INT32)
    route_expert = pl.create_tensor([LANE_ROWS, 1], dtype=pl.INT32)
    route_source = pl.create_tensor([LANE_ROWS, 1], dtype=pl.INT32)
    route_count = pl.create_tensor([1], dtype=pl.INT32)
    decode_ep2_arrive(
        lane_x, lane_aux, lane_count, lane_ready, route_x, route_scale, route_weight,
        route_id, route_expert, route_source, route_count, rank, epoch, pack_tid,
    )

    recv_x = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX, D], dtype=pl.INT8)
    recv_scale = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX], dtype=pl.FP32)
    recv_weight = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX], dtype=pl.FP32)
    recv_source = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX], dtype=pl.INT32)
    recv_route = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX], dtype=pl.INT32)
    recv_count = pl.create_tensor([N_LOCAL_EXPERTS, 1], dtype=pl.INT32)
    expert_offsets = pl.create_tensor([N_LOCAL_EXPERTS + 1], dtype=pl.INT32)
    decode_ep2_regroup(
        route_expert, route_source, route_id, route_x, route_scale, route_weight, route_count,
        recv_x, recv_scale, recv_weight, recv_source, recv_route, recv_count, expert_offsets,
    )

    routed_output = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX, D], dtype=pl.FP32)
    routed_output_flat = pl.reshape(routed_output, [N_LOCAL_EXPERTS * RECV_MAX, D])
    d_tiles = D // 256
    with pl.spmd(N_LOCAL_EXPERTS * (RECV_MAX // 64) * d_tiles, name_hint="dsv41_ep2_expert_output_zero") as inputs_ready:
        block = pl.tile.get_block_idx()
        zero_row = (block // d_tiles) * 64
        zero_col = (block % d_tiles) * 256
        routed_output_flat[zero_row : zero_row + 64, zero_col : zero_col + 256] = pl.full(
            [64, 256], dtype=pl.FP32, value=0.0
        )
    for local_e in pl.parallel(N_LOCAL_EXPERTS):
        n_rows = pl.cast(pl.read(recv_count, [local_e, 0]), pl.INDEX)
        n_tiles = (n_rows + 64 - 1) // 64
        for tile in pl.parallel(n_tiles):
            tile_row = tile * 64
            valid_rows = pl.min(64, n_rows - tile_row)
            flat_tile_row = local_e * RECV_MAX + tile_row
            recv_y_tile = pl.create_tensor([64, D], dtype=pl.FP32)
            expert_routed_tile(
                recv_x, recv_scale, recv_weight,
                routed_w1, routed_w1_scale,
                routed_w3, routed_w3_scale,
                routed_w2, routed_w2_scale,
                recv_y_tile,
                local_e, tile_row, valid_rows, inputs_ready,
            )
            routed_output_flat[flat_tile_row : flat_tile_row + 64, :] = recv_y_tile
    return_tid = decode_ep2_return(routed_output, recv_source, recv_route, recv_count, slot_out)

    shared_output = pl.create_tensor([T, D], dtype=pl.BF16)
    SHARED(value, shared_gate_up, shared_gate_up_scale, shared_down, shared_down_scale, shared_output, active)
    decode_ep2_combine(slot_out, return_ready, shared_output, output, active, rank, epoch, return_tid)
    return output


@pl.jit.host
def decode_ep2(
    value: pl.Tensor[[EP_SIZE, T, D], pl.BF16],
    router_value: pl.Tensor[[EP_SIZE, T, D], pl.FP32],
    gate_weight: pl.Tensor[[EP_SIZE, N_EXPERTS, D], pl.FP32],
    text_bias: pl.Tensor[[EP_SIZE, N_EXPERTS], pl.FP32],
    vision_bias: pl.Tensor[[EP_SIZE, N_EXPERTS], pl.FP32],
    input_ids: pl.Tensor[[EP_SIZE, T], pl.INT64],
    routed_w1: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, D, 2560], pl.INT8],
    routed_w2_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, D], pl.FP32],
    shared_gate_up: pl.Tensor[[EP_SIZE, D, MOE_INTER * 2], pl.INT8],
    shared_gate_up_scale: pl.Tensor[[EP_SIZE, MOE_INTER * 2], pl.FP32],
    shared_down: pl.Tensor[[EP_SIZE, MOE_INTER, D], pl.INT8],
    shared_down_scale: pl.Tensor[[EP_SIZE, D], pl.FP32],
    logits: pl.Out[pl.Tensor[[EP_SIZE, T, N_EXPERTS], pl.FP32]],
    route_ids: pl.Out[pl.Tensor[[EP_SIZE, T, TOPK], pl.INT32]],
    route_weights: pl.Out[pl.Tensor[[EP_SIZE, T, TOPK], pl.FP32]],
    output: pl.Out[pl.Tensor[[EP_SIZE, T, D], pl.BF16]],
    active: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """Launch one complete EP2 Decode MoE layer per logical rank."""

    lane_x_buf = pld.alloc_window_buffer([LANE_ROWS, D], dtype=pl.INT8)
    lane_aux_buf = pld.alloc_window_buffer([LANE_ROWS, AUX_PAD], dtype=pl.FP32)
    lane_count_buf = pld.alloc_window_buffer([EP_SIZE, COUNT_PAD], dtype=pl.INT32)
    lane_ready_buf = pld.alloc_window_buffer([EP_SIZE, SIGNAL_PAD], dtype=pl.INT32)
    slot_out_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.FP32)
    return_ready_buf = pld.alloc_window_buffer([EP_SIZE, SIGNAL_PAD], dtype=pl.INT32)
    for rank in pl.range(EP_SIZE):
        lane_x = pld.window(lane_x_buf, [LANE_ROWS, D], dtype=pl.INT8)
        lane_aux = pld.window(lane_aux_buf, [LANE_ROWS, AUX_PAD], dtype=pl.FP32)
        lane_count = pld.window(lane_count_buf, [EP_SIZE, COUNT_PAD], dtype=pl.INT32)
        lane_ready = pld.window(lane_ready_buf, [EP_SIZE, SIGNAL_PAD], dtype=pl.INT32)
        slot_out = pld.window(slot_out_buf, [N_ROUTES, D], dtype=pl.FP32)
        return_ready = pld.window(return_ready_buf, [EP_SIZE, SIGNAL_PAD], dtype=pl.INT32)
        decode_ep2_rank(
            value[rank], router_value[rank], gate_weight[rank], text_bias[rank], vision_bias[rank], input_ids[rank],
            routed_w1[rank], routed_w1_scale[rank], routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank], shared_gate_up[rank], shared_gate_up_scale[rank],
            shared_down[rank], shared_down_scale[rank], lane_x, lane_aux, lane_count, lane_ready,
            slot_out, return_ready, logits[rank], route_ids[rank], route_weights[rank], output[rank],
            active, rank, epoch,
            device=rank,
        )


def product_contract() -> dict[str, object]:
    """Return the selected expert-parallel product ABI without claiming a run."""

    return {
        "status": "implementation_ready_not_run",
        "topology": {"dp": EP_SIZE, "tp": 1, "ep": EP_SIZE, "pp": 1, "pcp": 1, "dcp": 1},
        "routed_experts": N_EXPERTS,
        "local_experts_per_rank": N_LOCAL_EXPERTS,
        "top_k": TOPK,
        "capacity_tokens_per_rank": T,
        "route_lane_capacity": LANE_CAP,
        "receive_capacity_per_local_expert": RECV_MAX,
        "communication": ["remote_put", "notify", "wait", "return_put", "combine"],
        "product_compute": "PyPTO only",
        "reference_process": "independent vLLM-Ascend process",
        "epoch": "shared monotonic decode-step signal",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", action="store_true", help="print the EP2 product ABI")
    parser.add_argument("--compile-only", action="store_true", help="reserved for the queued PyPTO compile harness")
    args = parser.parse_args(argv)
    if args.contract or args.compile_only:
        import json

        print(json.dumps(product_contract(), indent=2, sort_keys=True))
        return 0
    parser.error("EP2 Decode requires the frozen checkpoint/input/cache contract; use --contract for the ABI")
    return 2

B = 1
S = 1
M = FLASH
RECV_TILE = 64
K_TILE = 512
INTER_K = 512
MM_INTER_TILE = 256
MM_GATE_INNER = 3
ACT_INTER_TILE = 64
ACT_GATE_INNER = 4
D_OUT_TILE = 256
QUANT_TILE = 256
QUANT_ROW_TILE = 16
QUANT_SCALE_PAD = 8
D_OUT_TILE_ACT = 256
W2_INNER = 1
W2_ACT_INNER = 10
TILES_PER_EXPERT = RECV_MAX // RECV_TILE

INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-30

if RECV_MAX % RECV_TILE:
    raise ValueError("RECV_MAX must be a whole number of RECV_TILE row-tiles")


@pl.jit.inline(auto_scope=False)
def expert_routed_tile(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER_PAD], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    recv_y_tile: pl.Tensor[[RECV_TILE, D], pl.FP32],
    local_e: pl.Scalar[pl.INDEX],
    tile_row: pl.Scalar[pl.INDEX],
    valid_rows: pl.Scalar[pl.INDEX],
    inputs_ready: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    recv_x_flat = pl.reshape(recv_x, [N_LOCAL_EXPERTS * RECV_MAX, D])
    flat_tile_row = local_e * RECV_MAX + tile_row
    w1_base = local_e * MOE_INTER
    w2_base = local_e * D
    h_tile_i8 = pl.create_tensor([RECV_TILE, MOE_INTER_PAD], dtype=pl.INT8)
    h_tile_scale_dq = pl.create_tensor([RECV_TILE, QUANT_SCALE_PAD], dtype=pl.FP32, manual_dep=True)
    pl.dump_tag(h_tile_i8)
    pl.dump_tag(h_tile_scale_dq)
    quant_tids = pl.array.create(1, pl.TASK_ID)

    with pl.scope():
        gate_tile_i32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT32)
        up_tile_i32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT32)
        pl.dump_tag(gate_tile_i32)
        pl.dump_tag(up_tile_i32)

        with pl.spmd(MOE_INTER // (MM_GATE_INNER * MM_INTER_TILE), name_hint="exp_gate_mm", deps=[inputs_ready]):
            block = pl.tile.get_block_idx()
            n_base = block * (MM_GATE_INNER * MM_INTER_TILE)
            for inner in pl.range(MM_GATE_INNER):
                n0 = n_base + inner * MM_INTER_TILE
                gate_acc = pl.create_tensor([RECV_TILE, MM_INTER_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                    x_chunk = recv_x_flat[flat_tile_row : flat_tile_row + RECV_TILE, k0 : k0 + K_TILE]
                    w1_chunk = pl.reshape(
                        routed_w1[local_e : local_e + 1, n0 : n0 + MM_INTER_TILE, k0 : k0 + K_TILE],
                        [MM_INTER_TILE, K_TILE],
                    )
                    gate_acc = pl.matmul_acc(gate_acc, x_chunk, w1_chunk, b_trans=True, init_cond=(k0 == 0))
                gate_tile_i32[:, n0 : n0 + MM_INTER_TILE] = gate_acc

        with pl.spmd(MOE_INTER // (MM_GATE_INNER * MM_INTER_TILE), name_hint="exp_up_mm", deps=[inputs_ready]):
            block = pl.tile.get_block_idx()
            n_base = block * (MM_GATE_INNER * MM_INTER_TILE)
            for inner in pl.range(MM_GATE_INNER):
                n0 = n_base + inner * MM_INTER_TILE
                up_acc = pl.create_tensor([RECV_TILE, MM_INTER_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                    x_chunk = recv_x_flat[flat_tile_row : flat_tile_row + RECV_TILE, k0 : k0 + K_TILE]
                    w3_chunk = pl.reshape(
                        routed_w3[local_e : local_e + 1, n0 : n0 + MM_INTER_TILE, k0 : k0 + K_TILE],
                        [MM_INTER_TILE, K_TILE],
                    )
                    up_acc = pl.matmul_acc(up_acc, x_chunk, w3_chunk, b_trans=True, init_cond=(k0 == 0))
                up_tile_i32[:, n0 : n0 + MM_INTER_TILE] = up_acc

        h_tile_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32)
        pl.dump_tag(h_tile_fp32)
        with pl.spmd(MOE_INTER // (ACT_GATE_INNER * ACT_INTER_TILE), name_hint="exp_gate_up_act"):
            block = pl.tile.get_block_idx()
            inter_base = block * (ACT_GATE_INNER * ACT_INTER_TILE)
            for inner in pl.pipeline(ACT_GATE_INNER, stage=2):
                inter0 = inter_base + inner * ACT_INTER_TILE
                gate_i32 = gate_tile_i32[:, inter0 : inter0 + ACT_INTER_TILE]
                up_i32 = up_tile_i32[:, inter0 : inter0 + ACT_INTER_TILE]
                x_scale = pl.reshape(
                    recv_scale_dq[
                        local_e : local_e + 1,
                        tile_row : tile_row + RECV_TILE,
                    ],
                    [RECV_TILE, 1],
                )
                gate_scale = pl.reshape(
                    pl.reshape(
                        routed_w1_scale[local_e : local_e + 1, inter0 : inter0 + ACT_INTER_TILE],
                        [ACT_INTER_TILE],
                    ),
                    [1, ACT_INTER_TILE],
                )
                up_scale = pl.reshape(
                    pl.reshape(
                        routed_w3_scale[local_e : local_e + 1, inter0 : inter0 + ACT_INTER_TILE],
                        [ACT_INTER_TILE],
                    ),
                    [1, ACT_INTER_TILE],
                )
                gate_fp32 = pl.col_expand_mul(
                    pl.row_expand_mul(
                        pl.cast(gate_i32, target_type=pl.FP32, mode="none"),
                        x_scale,
                    ),
                    gate_scale,
                )
                up_fp32 = pl.col_expand_mul(
                    pl.row_expand_mul(
                        pl.cast(up_i32, target_type=pl.FP32, mode="none"),
                        x_scale,
                    ),
                    up_scale,
                )
                if SWIGLU_LIMIT > 0.0:
                    gate_fp32 = pl.minimum(gate_fp32, SWIGLU_LIMIT)
                    up_fp32 = pl.maximum(pl.minimum(up_fp32, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                denominator = pl.add(pl.exp(pl.neg(gate_fp32)), 1.0)
                silu = pl.div(gate_fp32, denominator)
                activated = pl.mul(silu, up_fp32)
                # vLLM-Ascend applies the normalized route probability before
                # the W2 activation quantizer. Preserve the BF16 boundary used
                # by the fused expert path before staging back to FP32.
                route_weight = pl.reshape(
                    recv_weights[local_e : local_e + 1, tile_row : tile_row + RECV_TILE],
                    [RECV_TILE, 1],
                )
                route_weight_bf16 = pl.cast(route_weight, target_type=pl.BF16, mode="rint")
                activated = pl.row_expand_mul(
                    pl.cast(pl.cast(activated, target_type=pl.BF16, mode="rint"), target_type=pl.FP32),
                    pl.cast(route_weight_bf16, target_type=pl.FP32, mode="none"),
                )
                activated = pl.set_validshape(activated, valid_rows, ACT_INTER_TILE)
                h_tile_fp32[:, inter0 : inter0 + ACT_INTER_TILE] = pl.fillpad(activated, pad_value=pl.PadValue.zero)

        with pl.spmd(RECV_TILE // QUANT_ROW_TILE, name_hint="exp_h_q") as quant_tid:
            quant_block = pl.tile.get_block_idx()
            quant_row = quant_block * QUANT_ROW_TILE
            row_amax = pl.full([1, QUANT_ROW_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for k0 in pl.pipeline(0, MOE_INTER, QUANT_TILE, stage=2):
                h_amax_chunk = h_tile_fp32[quant_row : quant_row + QUANT_ROW_TILE, k0 : k0 + QUANT_TILE]
                h_abs = pl.maximum(h_amax_chunk, pl.neg(h_amax_chunk))
                row_amax = pl.maximum(row_amax, pl.reshape(pl.row_max(h_abs), [1, QUANT_ROW_TILE]))
            dequant_scale = pl.full([1, QUANT_ROW_TILE], dtype=pl.FP32, value=1.0)
            quant_scale = pl.full([1, QUANT_ROW_TILE], dtype=pl.FP32, value=1.0)
            for scale_row in pl.range(QUANT_ROW_TILE):
                maximum = pl.read(row_amax, [0, scale_row])
                scale_value = maximum / INT8_SCALE_MAX
                inverse_value = 1.0 / scale_value
                pl.write(dequant_scale, [0, scale_row], scale_value)
                pl.write(quant_scale, [0, scale_row], inverse_value)
            dequant_scale_col = pl.reshape(dequant_scale, [QUANT_ROW_TILE, 1])
            dequant_scale_zeros = pl.full([QUANT_ROW_TILE, QUANT_SCALE_PAD], dtype=pl.FP32, value=0.0)
            h_tile_scale_dq[
                quant_row : quant_row + QUANT_ROW_TILE,
                :,
            ] = pl.row_expand(dequant_scale_zeros, dequant_scale_col)
            quant_scale_col = pl.reshape(quant_scale, [QUANT_ROW_TILE, 1])
            for k0 in pl.pipeline(0, MOE_INTER, QUANT_TILE, stage=2):
                h_quant_chunk = h_tile_fp32[quant_row : quant_row + QUANT_ROW_TILE, k0 : k0 + QUANT_TILE]
                h_scaled = pl.row_expand_mul(h_quant_chunk, quant_scale_col)
                # CANN's A2 dequant_swiglu_quant rounds the normalized value
                # to INT32 before the INT8 store, with an FP16 staging cast.
                h_quant_i32 = pl.cast(h_scaled, target_type=pl.INT32, mode="rint")
                h_tile_i8[
                    quant_row : quant_row + QUANT_ROW_TILE,
                    k0 : k0 + QUANT_TILE,
                ] = pl.cast(
                    pl.cast(h_quant_i32, target_type=pl.FP16, mode="rint"),
                    target_type=pl.INT8,
                    mode="rint",
                )
            # The final W2 K tile also reads the aligned activation tail.
            h_padding_fp32 = pl.full([QUANT_ROW_TILE, MOE_INTER_PAD - MOE_INTER], dtype=pl.FP32, value=0.0)
            h_padding_i32 = pl.cast(h_padding_fp32, target_type=pl.INT32, mode="rint")
            h_padding = pl.cast(
                pl.cast(h_padding_i32, target_type=pl.FP16, mode="rint"),
                target_type=pl.INT8,
                mode="rint",
            )
            h_tile_i8[quant_row : quant_row + QUANT_ROW_TILE, MOE_INTER : MOE_INTER_PAD] = h_padding
        quant_tids[0] = quant_tid

    y_i32 = pl.create_tensor([RECV_TILE, D], dtype=pl.INT32)
    pl.dump_tag(y_i32)
    with pl.spmd(
        D // (W2_INNER * D_OUT_TILE),
        name_hint="exp_w2_mm",
        deps=[quant_tids[0]],
        allow_early_resolve=True,
    ) as w2_tid:
        block = pl.tile.get_block_idx()
        d_base = block * (W2_INNER * D_OUT_TILE)
        for inner in pl.range(W2_INNER):
            d0 = d_base + inner * D_OUT_TILE
            y_acc = pl.create_tensor([RECV_TILE, D_OUT_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, MOE_INTER, INTER_K, stage=2):
                h_w2_chunk = h_tile_i8[:, k0 : k0 + INTER_K]
                w2_chunk = pl.reshape(
                    routed_w2[local_e : local_e + 1, d0 : d0 + D_OUT_TILE, k0 : k0 + INTER_K],
                    [D_OUT_TILE, INTER_K],
                )
                y_acc = pl.matmul_acc(y_acc, h_w2_chunk, w2_chunk, b_trans=True, init_cond=(k0 == 0))
            y_i32[:, d0 : d0 + D_OUT_TILE] = y_acc

    with pl.spmd(
        D // (W2_ACT_INNER * D_OUT_TILE_ACT),
        name_hint="exp_w2_act",
        deps=[w2_tid],
        allow_early_resolve=True,
    ) as w2_act_tid:
        block = pl.tile.get_block_idx()
        d_base = block * (W2_ACT_INNER * D_OUT_TILE_ACT)
        row_scale = pl.row_max(h_tile_scale_dq)
        for inner in pl.pipeline(W2_ACT_INNER, stage=2):
            d0 = d_base + inner * D_OUT_TILE_ACT
            y_fp32 = pl.cast(y_i32[:, d0 : d0 + D_OUT_TILE_ACT], target_type=pl.FP32, mode="none")
            w2_scale_tile = pl.reshape(
                pl.reshape(
                    routed_w2_scale[local_e : local_e + 1, d0 : d0 + D_OUT_TILE_ACT],
                    [D_OUT_TILE_ACT],
                ),
                [1, D_OUT_TILE_ACT],
            )
            y_channels = pl.col_expand_mul(y_fp32, w2_scale_tile)
            y_fp32 = pl.row_expand_mul(y_channels, row_scale)
            y_bf16 = pl.cast(y_fp32, target_type=pl.BF16, mode="rint")
            y_rounded = pl.cast(y_bf16, target_type=pl.FP32, mode="none")
            recv_y_tile[:, d0 : d0 + D_OUT_TILE_ACT] = y_rounded
    return w2_act_tid


@pl.jit.inline(auto_scope=False)
def expert_routed_inline(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER_PAD], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    recv_y: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.FP32]],
):
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL_EXPERTS * RECV_MAX, D])
    d_tiles = D // D_OUT_TILE_ACT
    # Rows past an expert's receive count are never written by a tile; zero them
    # so the whole recv_y slab matches the golden.
    with pl.spmd(N_LOCAL_EXPERTS * TILES_PER_EXPERT * d_tiles, name_hint="expert_recv_y_zero") as inputs_ready:
        block = pl.tile.get_block_idx()
        zero_row = (block // d_tiles) * RECV_TILE
        zero_col = (block % d_tiles) * D_OUT_TILE_ACT
        recv_y_flat[
            zero_row : zero_row + RECV_TILE,
            zero_col : zero_col + D_OUT_TILE_ACT,
        ] = pl.full([RECV_TILE, D_OUT_TILE_ACT], dtype=pl.FP32, value=0.0)
    for local_e in pl.parallel(N_LOCAL_EXPERTS):
        n_rows = pl.cast(pl.read(recv_expert_count, [local_e, 0]), pl.INDEX)
        n_tiles = (n_rows + RECV_TILE - 1) // RECV_TILE
        for tile in pl.parallel(n_tiles):
            tile_row = tile * RECV_TILE
            valid_rows = pl.min(RECV_TILE, n_rows - tile_row)
            flat_tile_row = local_e * RECV_MAX + tile_row
            with pl.scope():
                recv_y_tile = pl.create_tensor([RECV_TILE, D], dtype=pl.FP32)
                expert_routed_tile(
                    recv_x, recv_scale_dq, recv_weights,
                    routed_w1, routed_w1_scale,
                    routed_w3, routed_w3_scale,
                    routed_w2, routed_w2_scale,
                    recv_y_tile,
                    local_e, tile_row, valid_rows, inputs_ready,
                )
                recv_y_flat[flat_tile_row : flat_tile_row + RECV_TILE, :] = recv_y_tile
    return recv_y


@pl.jit(auto_scope=False)
def expert_routed_test(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER_PAD], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    recv_y: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.FP32]],
):
    """Compile the routed expert as a standalone operator."""
    return expert_routed_inline(
        recv_x,
        recv_scale_dq,
        recv_weights,
        recv_expert_count,
        routed_w1,
        routed_w1_scale,
        routed_w3,
        routed_w3_scale,
        routed_w2,
        routed_w2_scale,
        recv_y,
    )


def golden_expert_routed(tensors):
    """Torch reference for the routed expert. recv_y is the per-row routing-
    weight-scaled SwiGLU output, ready for combine reduce to simply sum.

    Per-expert layout: recv_x[e, 0:cnt[e], :] is the valid INT8 receive
    payload; recv_y[e, cnt[e]:, :] stays at zero. The routed weights arrive in
    FRACTAL_NZ order, as the kernel declares them."""
    from .utils import int8_quant_per_row, unpack_nz
    import torch
    import torch.nn.functional as F

    def dequant_w(w_i8, w_scale):
        return unpack_nz(w_i8).to(torch.float32) * w_scale.unsqueeze(-1)

    recv_x_i8 = tensors["recv_x"]  # INT8, pre-quantized in dispatch
    recv_scale_dq = tensors["recv_scale_dq"].float()  # [E, RECV_MAX]
    recv_weights = tensors["recv_weights"].float()  # [E, RECV_MAX]
    recv_expert_count = tensors["recv_expert_count"]  # [E, 1] int32
    w1 = dequant_w(tensors["routed_w1"], tensors["routed_w1_scale"].float())
    w3 = dequant_w(tensors["routed_w3"], tensors["routed_w3_scale"].float())
    w2 = dequant_w(tensors["routed_w2"], tensors["routed_w2_scale"].float())

    recv_y = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D)
    for e in range(N_LOCAL_EXPERTS):
        n_rows = int(recv_expert_count[e, 0].item())
        if n_rows == 0:
            continue
        x_sub_i8 = recv_x_i8[e, :n_rows, :]
        x_sub_sd = recv_scale_dq[e, :n_rows].reshape(-1, 1)
        x_sub_q = x_sub_i8.float() * x_sub_sd
        w_per_row = recv_weights[e, :n_rows].reshape(-1, 1)

        gate = x_sub_q @ w1[e].T
        up = x_sub_q @ w3[e].T
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h = F.silu(gate) * up
        # A8 requant before w2 matmul.
        h_i8, h_sd = int8_quant_per_row(h)
        h = h_i8.float() * h_sd
        y_bf16 = (h @ w2[e].T).to(torch.bfloat16)
        recv_y[e, :n_rows, :] = y_bf16.float() * w_per_row.to(torch.bfloat16).float()

    tensors["recv_y"][:] = recv_y


def gen_routed_weight(shape, dequant_std):
    """Synthesize a routed-expert per-channel-symmetric INT8 weight + FP32 scale by
    simulating the real DeepSeek-V4-Flash MXFP4 routed-expert quant grid (e2m1, per-32-group
    E8M0 scale), then re-quantizing per-output-channel. A plain ``randn`` INT8 is wrong:
    routed collapses onto ~37 discrete levels with an ~11.6% zero spike (the FP4 grid) and a
    per-channel scale CV ~0.09 (the fine group scale flattens it). Per-output-channel INT8 is
    scale-invariant, so the level structure / zero spike emerge from the grid alone and
    ``dequant_std`` only sets the absolute scale magnitude. (shared experts use a different
    grid -- see expert_shared.gen_shared_weight.)

    ``shape`` last dim = reduction (in) dim; leading dims map to the per-output-channel
    scale shape ([E, out, in] -> scale [E, out]). The returned INT8 weight comes back in
    FRACTAL_NZ order, matching what the kernel declares; the golden reads it back through
    ``unpack_nz``.
    """
    import torch

    from .utils import pack_nz

    FP4_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    FP4_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])  # nearest-grid bounds
    FP4_MAX, TINY = 6.0, 1e-20
    GROUP = 32
    CHUNK_ELEMS = 1 << 25    # elements per pass; unchunked this walks GiB-sized temporaries

    *lead, out, inn = shape
    n_lead = 1
    for dim in lead:
        n_lead *= dim

    W = torch.randn(*shape).reshape(n_lead, out, inn)
    w_i8 = torch.empty(n_lead, out, inn, dtype=torch.int8)
    scale = torch.empty(n_lead, out, 1, dtype=torch.float32)

    # e2m1 + per-32-group E8M0 (round-up) scale on the in dim, then per-output-channel
    # INT8. Chunked over the leading dim: every reduction here is confined to one
    # (out, in) row, so the chunk boundary cannot change a result.
    step = max(1, CHUNK_ELEMS // (out * inn))
    for i0 in range(0, n_lead, step):
        w = W[i0:i0 + step]
        wg = w.reshape(-1, out, inn // GROUP, GROUP)
        absw = wg.abs()
        grp_scale = torch.exp2(torch.ceil(torch.log2((absw.amax(-1, keepdim=True) / FP4_MAX).clamp_min(TINY))))
        idx = torch.bucketize(absw.div_(grp_scale), FP4_MID).clamp_max_(7)
        wq = (torch.sign(wg) * FP4_MAG[idx]).mul_(grp_scale).reshape(w.shape)
        amax = wq.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
        chan_scale = amax / INT8_SCALE_MAX
        w_i8[i0:i0 + step] = torch.round(wq.div_(chan_scale)).clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
        scale[i0:i0 + step] = chan_scale
    del W

    scale = (scale * (dequant_std / (w_i8.float() * scale).std())).squeeze(-1).float()
    return pack_nz(w_i8.reshape(*shape)), scale.reshape(*lead, out)


def build_tensor_specs():
    from .utils import int8_quant_per_row
    import torch
    from golden import TensorSpec

    # Across-layer-mean dequant std of the real DeepSeek-V4-Flash MXFP4 routed experts.
    ROUTED_DEQUANT_STD = {"w1": 2.47e-2, "w2": 2.44e-2, "w3": 2.46e-2}

    # Distribute B*S*TOPK token-expert pairs uniformly across local experts.
    total = B * S * M.num_experts_per_tok
    counts = torch.bincount(torch.randint(0, N_LOCAL_EXPERTS, (total,)), minlength=N_LOCAL_EXPERTS).to(torch.int32)
    counts_2d = counts.reshape(N_LOCAL_EXPERTS, 1)

    # INT8 recv_x + per-row dequant scale. Tail rows are INT8 0 with scale 0,
    # so dequant produces 0.
    x_bf16 = torch.randn(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.bfloat16)
    valid_mask_3d = (torch.arange(RECV_MAX).reshape(1, RECV_MAX, 1) < counts.reshape(N_LOCAL_EXPERTS, 1, 1))
    recv_x_i8_pre, recv_scale_dq_pre = int8_quant_per_row(x_bf16)
    recv_x_i8_pre = torch.where(valid_mask_3d, recv_x_i8_pre, torch.zeros_like(recv_x_i8_pre))
    valid_mask_2d = valid_mask_3d.squeeze(-1)
    recv_scale_dq_pre = torch.where(
        valid_mask_2d,
        recv_scale_dq_pre.squeeze(-1),
        torch.zeros_like(recv_scale_dq_pre.squeeze(-1)),
    )

    def init_recv_x():
        return recv_x_i8_pre

    def init_recv_scale_dq():
        return recv_scale_dq_pre.float()

    def init_recv_expert_count():
        return counts_2d

    # Per-row routing weight in [0, 1); tail rows (slot >= count) stay 0.
    recv_weights_pre = torch.rand(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.float32)
    recv_weights_pre = torch.where(valid_mask_2d, recv_weights_pre, torch.zeros_like(recv_weights_pre))

    def init_recv_weights():
        return recv_weights_pre

    # (int8, per-channel scale) on the real MXFP4 routed-expert quant grid.
    w1_i8, w1_s = gen_routed_weight((N_LOCAL_EXPERTS, MOE_INTER, D), ROUTED_DEQUANT_STD["w1"])
    w3_i8, w3_s = gen_routed_weight((N_LOCAL_EXPERTS, MOE_INTER, D), ROUTED_DEQUANT_STD["w3"])
    w2_i8, w2_s = gen_routed_weight((N_LOCAL_EXPERTS, D, MOE_INTER), ROUTED_DEQUANT_STD["w2"])

    return [
        TensorSpec("recv_x", [N_LOCAL_EXPERTS, RECV_MAX, D], torch.int8, init_value=init_recv_x),
        TensorSpec("recv_scale_dq", [N_LOCAL_EXPERTS, RECV_MAX], torch.float32, init_value=init_recv_scale_dq),
        TensorSpec("recv_weights", [N_LOCAL_EXPERTS, RECV_MAX], torch.float32, init_value=init_recv_weights),
        TensorSpec("recv_expert_count", [N_LOCAL_EXPERTS, 1], torch.int32, init_value=init_recv_expert_count),
        TensorSpec("routed_w1", [N_LOCAL_EXPERTS, MOE_INTER, D], torch.int8, init_value=lambda: w1_i8),
        TensorSpec("routed_w1_scale", [N_LOCAL_EXPERTS, MOE_INTER], torch.float32, init_value=lambda: w1_s),
        TensorSpec("routed_w3", [N_LOCAL_EXPERTS, MOE_INTER, D], torch.int8, init_value=lambda: w3_i8),
        TensorSpec("routed_w3_scale", [N_LOCAL_EXPERTS, MOE_INTER], torch.float32, init_value=lambda: w3_s),
        TensorSpec("routed_w2", [N_LOCAL_EXPERTS, D, MOE_INTER], torch.int8, init_value=lambda: w2_i8),
        TensorSpec("routed_w2_scale", [N_LOCAL_EXPERTS, D], torch.float32, init_value=lambda: w2_s),
        TensorSpec("recv_y", [N_LOCAL_EXPERTS, RECV_MAX, D], torch.float32),
    ]

if __name__ == "__main__":

    raise SystemExit(main())


__all__ = [
    "D",
    "EP_SIZE",
    "MOE_INTER",
    "N_EXPERTS",
    "N_LOCAL_EXPERTS",
    "T",
    "TOPK",
    "decode_ep2",
    "decode_ep2_arrive",
    "decode_ep2_combine",
    "decode_ep2_pack",
    "decode_ep2_rank",
    "decode_ep2_regroup",
    "decode_ep2_return",
    "expert_routed_inline",
    "expert_routed_tile",
    "make_dispatch_layout",
    "make_moe_router",
    "make_moe_router_inline",
    "make_routed_expert_projection",
    "make_routed_expert_w8a8",
    "make_route_restore",
    "make_shared_expert_projection",
    "make_shared_expert_w8a8",
    "make_shared_moe_w8a8",
    "make_shared_moe_w8a8_inline",
    "make_w8a8_expert_projection",
    "product_contract",
    "routed_expert_reference",
    "shared_expert_reference",
    "w8a8_expert_linear",
]

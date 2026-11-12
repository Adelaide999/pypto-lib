# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Output projection computations and phase-specific numerical boundaries."""

import pypto.language as pl
import torch
from .decode_projection import make_bf16_linear as decode_make_bf16_linear
from .gemm import K_TILE, M_TILE, N_TILE, make_bf16_linear, make_blocked_weight_pack as make_pack_weight, make_dot, make_fp32_dequantize, make_int8_gemm, make_pack_weight as output_proj_make_pack_weight
from .quant import int8_matmul, make_decode_dynamic_int8_quantize as decode_make_dynamic_int8_quantize, make_dequantize, make_dynamic_int8_quantize, make_quantize, quantize_per_token_int8
from .rope import decode_rope_reference, make_adjacent_rope as make_rope, rope_reference


def make_o_proj_inline(capacity, config):
    """Build the complete grouped attention output projection."""
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    rope_dim = config["qk_rope_head_dim"]
    pairs = rope_dim // 2
    width = heads * head_dim
    groups = config["o_groups"]
    group_input = width // groups
    rank = config["o_lora_rank"]
    group_width = groups * rank
    packed_rows = group_width * (hidden // N_TILE)
    rotate = make_rope(capacity, heads, head_dim, rope_dim)
    quantize = make_quantize(capacity, group_width)
    pack = make_pack_weight(group_width, hidden)
    dot = make_dot(capacity, group_width, hidden)
    dequantize = make_dequantize(capacity, hidden, combine_scales=True)

    @pl.jit.inline
    def project(
        attention: pl.Tensor[[capacity, width], pl.BF16],
        wo_a: pl.Tensor[[groups, group_input, rank], pl.BF16],
        wo_b: pl.Tensor[[group_width, hidden], pl.INT8],
        wo_b_scale: pl.Tensor[[hidden], pl.FP32],
        rope_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        output: pl.Tensor[[capacity, hidden], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        inverse_sine = pl.create_tensor([capacity, pairs], dtype=pl.FP32)
        for token in pl.spmd(active, name_hint="a2_inverse_sine"):
            inverse_sine[token : token + 1, :] = pl.mul(rope_sin[token : token + 1, :], -1.0)
        rotated = pl.create_tensor([capacity, width], dtype=pl.BF16)
        rotate(attention, rope_cos, inverse_sine, rotated, active)

        grouped = pl.create_tensor([capacity, group_width], dtype=pl.BF16)
        for block in pl.spmd((active + M_TILE - 1) // M_TILE * (group_width // N_TILE), name_hint="a2_wo_a"):
            token = block // (group_width // N_TILE) * M_TILE
            column = block % (group_width // N_TILE) * N_TILE
            group = column // rank
            group_column = column % rank
            rows = pl.min(M_TILE, active - token)
            accumulated = pl.create_tensor([M_TILE, N_TILE], dtype=pl.FP32)
            for part in pl.pipeline(group_input // K_TILE, stage=2):
                inner = part * K_TILE
                offset = group * group_input + inner
                x = pl.slice(rotated, [M_TILE, K_TILE], [token, offset], valid_shape=[rows, K_TILE])
                x = pl.fillpad(x, pad_value=pl.PadValue.zero)
                w = wo_a[group : group + 1, inner : inner + K_TILE, group_column : group_column + N_TILE]
                w = pl.reshape(w, [K_TILE, N_TILE])
                accumulated = pl.matmul_acc(accumulated, x, w, init_cond=(part == 0))
            narrowed = pl.cast(accumulated, pl.BF16, mode="rint")
            grouped[token : token + M_TILE, column : column + N_TILE] = pl.set_validshape(
                narrowed, rows, N_TILE
            )

        payload = pl.create_tensor([capacity, group_width], dtype=pl.INT8)
        token_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize(grouped, payload, token_scale, active)
        packed = pl.create_tensor([packed_rows, N_TILE], dtype=pl.INT8)
        pack(wo_b, packed)
        output_int32 = pl.create_tensor([capacity, hidden], dtype=pl.INT32)
        dot(payload, packed, output_int32, active)
        dequantize(output_int32, wo_b_scale, token_scale, output, active)
        return output

    return project


def make_o_proj_w8a8(capacity, config):
    """Build inverse RoPE and all grouped W8A8 Wo-A/Wo-B projections.

    Wo-A input quantization is independent for each native group. The package
    stores [group_input, groups*rank] with per-output-channel FP32 scales.
    """
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    rope_dim = config["qk_rope_head_dim"]
    pairs = rope_dim // 2
    width = heads * head_dim
    groups = config["o_groups"]
    group_input = width // groups
    rank = config["o_lora_rank"]
    group_width = groups * rank
    rotate = make_rope(capacity, heads, head_dim, rope_dim)
    quantize_a = make_quantize(capacity, group_input)
    pack_a = make_pack_weight(group_input, rank)
    dot_a = make_dot(capacity, group_input, rank)
    dequantize_a = make_dequantize(capacity, rank)
    quantize_b = make_quantize(capacity, group_width)
    pack_b = make_pack_weight(group_width, hidden)
    dot_b = make_dot(capacity, group_width, hidden)
    dequantize_b = make_dequantize(capacity, hidden, combine_scales=True)
    packed_a_rows = group_input * (rank // N_TILE)
    packed_b_rows = group_width * (hidden // N_TILE)

    @pl.jit
    def project(
        attention: pl.Tensor[[capacity, width], pl.FP32],
        wo_a: pl.Tensor[[group_input, group_width], pl.INT8],
        wo_a_scale: pl.Tensor[[group_width], pl.FP32],
        wo_b: pl.Tensor[[group_width, hidden], pl.INT8],
        wo_b_scale: pl.Tensor[[hidden], pl.FP32],
        rope_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        rotated: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        grouped: pl.InOut[pl.Tensor[[capacity, group_width], pl.BF16]],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        narrowed = pl.create_tensor([capacity, width], dtype=pl.BF16)
        for block in pl.spmd(active * (width // N_TILE), name_hint="a2_attention_bf16_boundary"):
            token = block // (width // N_TILE)
            col = block % (width // N_TILE) * N_TILE
            source = attention[token : token + 1, col : col + N_TILE]
            narrowed[token : token + 1, col : col + N_TILE] = pl.cast(source, pl.BF16, mode="rint")
        inverse_sine = pl.create_tensor([capacity, pairs], dtype=pl.FP32)
        for token in pl.spmd(active, name_hint="a2_w8a8_inverse_sine"):
            inverse_sine[token : token + 1, :] = pl.mul(rope_sin[token : token + 1, :], -1.0)
        rotate(narrowed, rope_cos, inverse_sine, rotated, active)

        for group in pl.range(groups):
            group_source = pl.create_tensor([capacity, group_input], dtype=pl.BF16)
            for block in pl.spmd(active * (group_input // N_TILE), name_hint="a2_wo_a_group_input"):
                token = block // (group_input // N_TILE)
                column = block % (group_input // N_TILE) * N_TILE
                offset = group * group_input + column
                group_source[token : token + 1, column : column + N_TILE] = rotated[
                    token : token + 1, offset : offset + N_TILE
                ]
            payload_a = pl.create_tensor([capacity, group_input], dtype=pl.INT8)
            scale_a = pl.create_tensor([capacity, 8], dtype=pl.FP32)
            quantize_a(group_source, payload_a, scale_a, active)
            group_weight = pl.create_tensor([group_input, rank], dtype=pl.INT8)
            for block in pl.spmd(
                (group_input // K_TILE) * (rank // N_TILE), name_hint="a2_wo_a_group_weight"
            ):
                inner = block // (rank // N_TILE) * K_TILE
                column = block % (rank // N_TILE) * N_TILE
                offset = group * rank + column
                group_weight[inner : inner + K_TILE, column : column + N_TILE] = wo_a[
                    inner : inner + K_TILE, offset : offset + N_TILE
                ]
            packed_a = pl.create_tensor([packed_a_rows, N_TILE], dtype=pl.INT8)
            pack_a(group_weight, packed_a)
            accumulator_a = pl.create_tensor([capacity, rank], dtype=pl.INT32)
            dot_a(payload_a, packed_a, accumulator_a, active)
            channel_scale = wo_a_scale[group * rank : (group + 1) * rank]
            group_output = pl.create_tensor([capacity, rank], dtype=pl.BF16)
            dequantize_a(accumulator_a, channel_scale, scale_a, group_output, active)
            for block in pl.spmd(active * (rank // N_TILE), name_hint="a2_wo_a_group_output"):
                token = block // (rank // N_TILE)
                column = block % (rank // N_TILE) * N_TILE
                offset = group * rank + column
                grouped[token : token + 1, offset : offset + N_TILE] = group_output[
                    token : token + 1, column : column + N_TILE
                ]

        payload_b = pl.create_tensor([capacity, group_width], dtype=pl.INT8)
        scale_b = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize_b(grouped, payload_b, scale_b, active)
        packed_b = pl.create_tensor([packed_b_rows, N_TILE], dtype=pl.INT8)
        pack_b(wo_b, packed_b)
        accumulator_b = pl.create_tensor([capacity, hidden], dtype=pl.INT32)
        dot_b(payload_b, packed_b, accumulator_b, active)
        dequantize_b(accumulator_b, wo_b_scale, scale_b, output, active)
        return rotated, grouped, output

    return project


def make_output_projection_bf16(
    capacity: int,
    heads: int,
    head_dim: int,
    rope_dim: int,
    groups: int,
    rank: int,
    hidden: int = 5120,
):
    """Build the grouped BF16 Wo-A/Wo-B projection used by target layers."""

    width = heads * head_dim
    group_input = width // groups
    if width % groups or group_input % 256 or rank % 64 or hidden % 256:
        raise ValueError("BF16 grouped output projection dimensions are not tile aligned")
    rotate = _make_inverse_rope(capacity, heads, head_dim, rope_dim)
    project_b_entry = make_bf16_linear(capacity, groups * rank, hidden)
    project_b = pl.jit.inline(project_b_entry._func)

    @pl.jit
    def project(
        attention: pl.Tensor[[capacity, width], pl.BF16],
        wo_a: pl.Tensor[[groups, group_input, rank], pl.BF16],
        wo_b: pl.Tensor[[groups * rank, hidden], pl.BF16],
        cosine: pl.Tensor[[capacity, rope_dim // 2], pl.FP32],
        sine: pl.Tensor[[capacity, rope_dim // 2], pl.FP32],
        rotated: pl.Out[pl.Tensor[[capacity, width], pl.BF16]],
        grouped: pl.Out[pl.Tensor[[capacity, groups * rank], pl.BF16]],
        output: pl.Out[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(capacity, name_hint="dsv41_a2a3_bf16_output_init"):
            token = pl.tile.get_block_idx()
            if token >= active:
                rotated_zero = pl.tile.full([1, width], dtype=pl.BF16, value=0.0)
                grouped_zero = pl.tile.full([1, groups * rank], dtype=pl.BF16, value=0.0)
                output_zero = pl.tile.full([1, hidden], dtype=pl.BF16, value=0.0)
                pl.store(rotated_zero, [token, 0], rotated)
                pl.store(grouped_zero, [token, 0], grouped)
                pl.store(output_zero, [token, 0], output)
        inverse_sine = pl.create_tensor([capacity, rope_dim // 2], dtype=pl.FP32)
        for block in pl.spmd(active, name_hint="dsv41_a2a3_bf16_inverse_sine"):
            inverse_sine[block : block + 1, :] = pl.mul(sine[block : block + 1, :], -1.0)
        rotate(attention, cosine, inverse_sine, rotated, active)
        n_tile = 128
        k_tile = 256
        blocks = (groups * rank) // n_tile
        for block in pl.spmd(((active + 15) // 16) * blocks, name_hint="dsv41_a2a3_bf16_wo_a"):
            token0 = (block // blocks) * 16
            column = (block % blocks) * n_tile
            group = column // rank
            group_column = column % rank
            rows = pl.min(16, active - token0)
            acc = pl.create_tensor([16, n_tile], dtype=pl.FP32)
            for k0 in pl.pipeline(0, group_input, k_tile, stage=2):
                source_tile = pl.slice(
                    rotated,
                    [16, k_tile],
                    [token0, group * group_input + k0],
                    valid_shape=[rows, k_tile],
                )
                source_tile = pl.set_validshape(
                    pl.fillpad(source_tile, pad_value=pl.PadValue.zero), 16, k_tile
                )
                weight_tile = pl.reshape(
                    wo_a[group : group + 1, k0 : k0 + k_tile, group_column : group_column + n_tile],
                    [k_tile, n_tile],
                )
                acc = pl.matmul_acc(acc, source_tile, weight_tile, init_cond=(k0 == 0))
            grouped[token0 : token0 + 16, column : column + n_tile] = pl.set_validshape(
                pl.cast(acc, pl.BF16, mode="rint"), rows, n_tile
            )
        project_b(grouped, wo_b, output, active)
        return rotated, grouped, output

    return project


def make_output_projection(capacity: int, heads: int, head_dim: int, rope_dim: int, groups: int, rank: int):
    """Build the standalone A2/A3 output projection."""

    width = heads * head_dim
    group_input = width // groups
    if width % groups or group_input % 256 or rank % 64:
        raise ValueError("grouped output projection requires aligned group and rank widths")
    if rope_dim % 2:
        raise ValueError("rope dimension must be even")
    pairs = rope_dim // 2
    rotate = _make_inverse_rope(capacity, heads, head_dim, rope_dim)
    quantize = make_dynamic_int8_quantize(capacity, groups * rank)
    hidden = 5120
    pack = output_proj_make_pack_weight(groups * rank, hidden)
    dot = make_int8_gemm(capacity, groups * rank, hidden)
    dequant = make_fp32_dequantize(capacity, hidden)

    @pl.jit.inline
    def project(
        attention: pl.Tensor[[capacity, width], pl.BF16],
        wo_a: pl.Tensor[[groups, group_input, rank], pl.BF16],
        wo_b: pl.Tensor[[groups * rank, hidden], pl.INT8],
        wo_b_scale: pl.Tensor[[hidden], pl.FP32],
        cosine: pl.Tensor[[capacity, pairs], pl.FP32],
        sine: pl.Tensor[[capacity, pairs], pl.FP32],
        output: pl.Tensor[[capacity, hidden], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        rotated = pl.create_tensor([capacity, width], dtype=pl.BF16)
        inverse_sine = pl.create_tensor([capacity, pairs], dtype=pl.FP32)
        for block in pl.spmd(active, name_hint="dsv41_a2a3_inverse_sine"):
            inverse_sine[block : block + 1, :] = pl.mul(sine[block : block + 1, :], -1.0)
        rotate(attention, cosine, inverse_sine, rotated, active)
        grouped = pl.create_tensor([capacity, groups * rank], dtype=pl.BF16)
        for block in pl.spmd(active * groups, name_hint="dsv41_a2a3_wo_a"):
            token = block // groups
            group = block % groups
            source = rotated[token : token + 1, group * group_input : (group + 1) * group_input]
            weights = wo_a[group : group + 1, :, :]
            projected = pl.matmul(source, pl.reshape(weights, [group_input, rank]), out_dtype=pl.FP32)
            grouped[token : token + 1, group * rank : (group + 1) * rank] = pl.cast(
                projected, pl.BF16, mode="rint"
            )
        payload = pl.create_tensor([capacity, groups * rank], dtype=pl.INT8)
        token_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quantize(grouped, payload, token_scale, active)
        packed = pl.create_tensor([groups * rank * (hidden // 64), 64], dtype=pl.INT8)
        pack(wo_b, packed)
        accumulator = pl.create_tensor([capacity, hidden], dtype=pl.INT32)
        dot(payload, packed, accumulator, active)
        dequant(accumulator, token_scale, wo_b_scale, output, active)
        return rotated, grouped, output

    return project


def _make_inverse_rope(capacity: int, heads: int, head_dim: int, rope_dim: int):
    """Use the common RoPE kernel with a caller-supplied negated sine table."""

    from .rope import make_rope

    return make_rope(capacity, heads, head_dim, rope_dim)


def make_grouped_wo_a(capacity: int, heads: int, head_dim: int, groups: int, rank: int):
    """Apply grouped BF16 Wo-A while retaining only one GEMM tile on core."""

    width = heads * head_dim
    if width % groups or (width // groups) % 256 or rank % 64:
        raise ValueError("Wo-A dimensions are not A2/A3 aligned")
    group_input = width // groups
    group_blocks = rank // 64
    input_blocks = group_input // 256

    @pl.jit
    def project(
        rotated: pl.Tensor[[capacity, width], pl.BF16],
        wo_a: pl.Tensor[[groups, group_input, rank], pl.BF16],
        grouped: pl.InOut[pl.Tensor[[capacity, groups * rank], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(
            ((active + 15) // 16) * groups * group_blocks, name_hint="dsv41_a2a3_grouped_wo_a"
        ):
            token = (block // (groups * group_blocks)) * 16
            group = (block % (groups * group_blocks)) // group_blocks
            n0 = (block % group_blocks) * 64
            rows = pl.min(16, active - token)
            acc = pl.create_tensor([16, 64], dtype=pl.FP32)
            for k0 in pl.pipeline(0, group_input, 256, stage=2):
                source = pl.slice(
                    rotated,
                    [16, 256],
                    [token, group * group_input + k0],
                    valid_shape=[rows, 256],
                )
                source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 16, 256)
                weight = wo_a[group : group + 1, k0 : k0 + 256, n0 : n0 + 64]
                weight = pl.reshape(weight, [256, 64])
                acc = pl.matmul_acc(acc, source, weight, init_cond=(k0 == 0))
            grouped[token : token + 16, group * rank + n0 : group * rank + n0 + 64] = pl.set_validshape(
                pl.cast(acc, pl.BF16, mode="rint"), rows, 64
            )
        return grouped

    return project


def make_decode_output_projection_bf16(
    capacity: int,
    heads: int,
    head_dim: int,
    rope_dim: int,
    groups: int,
    rank: int,
    hidden: int = 5120,
):
    """Build the grouped BF16 Wo-A/Wo-B projection used by target layers."""

    width = heads * head_dim
    group_input = width // groups
    if width % groups or group_input % 256 or rank % 64 or hidden % 256:
        raise ValueError("BF16 grouped output projection dimensions are not tile aligned")
    rotate = _decode_make_inverse_rope(capacity, heads, head_dim, rope_dim)
    project_b_entry = decode_make_bf16_linear(capacity, groups * rank, hidden)
    project_b = pl.jit.inline(project_b_entry._func)

    @pl.jit.inline
    def project(
        attention: pl.Tensor[[capacity, width], pl.BF16],
        wo_a: pl.Tensor[[groups, group_input, rank], pl.BF16],
        wo_b: pl.Tensor[[groups * rank, hidden], pl.BF16],
        cosine: pl.Tensor[[capacity, rope_dim // 2], pl.FP32],
        sine: pl.Tensor[[capacity, rope_dim // 2], pl.FP32],
        rotated: pl.Out[pl.Tensor[[capacity, width], pl.BF16]],
        grouped: pl.Out[pl.Tensor[[capacity, groups * rank], pl.BF16]],
        output: pl.Out[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(capacity, name_hint="dsv41_a2a3_bf16_output_init"):
            token = pl.tile.get_block_idx()
            if token >= active:
                rotated_zero = pl.tile.full([1, width], dtype=pl.BF16, value=0.0)
                grouped_zero = pl.tile.full([1, groups * rank], dtype=pl.BF16, value=0.0)
                output_zero = pl.tile.full([1, hidden], dtype=pl.BF16, value=0.0)
                pl.store(rotated_zero, [token, 0], rotated)
                pl.store(grouped_zero, [token, 0], grouped)
                pl.store(output_zero, [token, 0], output)
        inverse_sine = pl.create_tensor([capacity, rope_dim // 2], dtype=pl.FP32)
        for block in pl.spmd(active, name_hint="dsv41_a2a3_bf16_inverse_sine"):
            inverse_sine[block : block + 1, :] = pl.mul(sine[block : block + 1, :], -1.0)
        rotate(attention, cosine, inverse_sine, rotated, active)
        n_tile = 128
        k_tile = 256
        blocks = (groups * rank) // n_tile
        for block in pl.spmd(((active + 15) // 16) * blocks, name_hint="dsv41_a2a3_bf16_wo_a"):
            token0 = (block // blocks) * 16
            column = (block % blocks) * n_tile
            group = column // rank
            group_column = column % rank
            rows = pl.min(16, active - token0)
            acc = pl.create_tensor([16, n_tile], dtype=pl.FP32)
            for k0 in pl.pipeline(0, group_input, k_tile, stage=2):
                source_tile = pl.slice(
                    rotated,
                    [16, k_tile],
                    [token0, group * group_input + k0],
                    valid_shape=[rows, k_tile],
                )
                source_tile = pl.set_validshape(
                    pl.fillpad(source_tile, pad_value=pl.PadValue.zero), 16, k_tile
                )
                weight_tile = pl.reshape(
                    wo_a[group : group + 1, k0 : k0 + k_tile, group_column : group_column + n_tile],
                    [k_tile, n_tile],
                )
                acc = pl.matmul_acc(acc, source_tile, weight_tile, init_cond=(k0 == 0))
            grouped[token0 : token0 + 16, column : column + n_tile] = pl.set_validshape(
                pl.cast(acc, pl.BF16, mode="rint"), rows, n_tile
            )
        project_b(grouped, wo_b, output, active)
        return rotated, grouped, output

    return project


def make_decode_output_projection(
    capacity: int, heads: int, head_dim: int, rope_dim: int, groups: int, rank: int
):
    """Build the standalone A2/A3 output projection."""

    width = heads * head_dim
    group_input = width // groups
    if width % groups or group_input % 256 or rank % 64:
        raise ValueError("grouped output projection requires aligned group and rank widths")
    if rope_dim % 2:
        raise ValueError("rope dimension must be even")
    pairs = rope_dim // 2
    rotate = _decode_make_inverse_rope(capacity, heads, head_dim, rope_dim)
    quantize = decode_make_dynamic_int8_quantize(capacity, groups * rank)
    hidden = 5120
    pack = output_proj_make_pack_weight(groups * rank, hidden)
    dot = make_int8_gemm(capacity, groups * rank, hidden)
    dequant = make_fp32_dequantize(capacity, hidden)

    @pl.jit.inline
    def project(
        attention: pl.Tensor[[capacity, width], pl.BF16],
        wo_a: pl.Tensor[[groups, group_input, rank], pl.BF16],
        wo_b: pl.Tensor[[groups * rank, hidden], pl.INT8],
        wo_b_scale: pl.Tensor[[hidden], pl.FP32],
        cosine: pl.Tensor[[capacity, pairs], pl.FP32],
        sine: pl.Tensor[[capacity, pairs], pl.FP32],
        output: pl.Tensor[[capacity, hidden], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        rotated = pl.create_tensor([capacity, width], dtype=pl.BF16)
        inverse_sine = pl.create_tensor([capacity, pairs], dtype=pl.FP32)
        for block in pl.spmd(active, name_hint="dsv41_a2a3_inverse_sine"):
            inverse_sine[block : block + 1, :] = pl.mul(sine[block : block + 1, :], -1.0)
        rotate(attention, cosine, inverse_sine, rotated, active)
        grouped = pl.create_tensor([capacity, groups * rank], dtype=pl.BF16)
        for block in pl.spmd(active * groups, name_hint="dsv41_a2a3_wo_a"):
            token = block // groups
            group = block % groups
            source = rotated[token : token + 1, group * group_input : (group + 1) * group_input]
            weights = wo_a[group : group + 1, :, :]
            projected = pl.matmul(source, pl.reshape(weights, [group_input, rank]), out_dtype=pl.FP32)
            grouped[token : token + 1, group * rank : (group + 1) * rank] = pl.cast(
                projected, pl.BF16, mode="rint"
            )
        payload = pl.create_tensor([capacity, groups * rank], dtype=pl.INT8)
        token_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quantize(grouped, payload, token_scale, active)
        packed = pl.create_tensor([groups * rank * (hidden // 64), 64], dtype=pl.INT8)
        pack(wo_b, packed)
        accumulator = pl.create_tensor([capacity, hidden], dtype=pl.INT32)
        dot(payload, packed, accumulator, active)
        dequant(accumulator, token_scale, wo_b_scale, output, active)
        return rotated, grouped, output

    return project


def _decode_make_inverse_rope(capacity: int, heads: int, head_dim: int, rope_dim: int):
    """Use the common RoPE kernel with a caller-supplied negated sine table."""

    from .rope import make_decode_rope as make_rope

    return make_rope(capacity, heads, head_dim, rope_dim)


def output_projection_reference(
    attention: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    heads: int,
    head_dim: int,
    rope_dim: int,
) -> dict[str, torch.Tensor]:
    """CPU golden for inverse RoPE, grouped Wo-A, and Wo-B W8A8."""

    groups, group_input, rank = wo_a.shape
    rotated = rope_reference(attention, cosine, -sine, heads, head_dim, rope_dim)
    chunks = rotated.reshape(rotated.shape[0], groups, group_input)
    grouped = (
        torch.einsum("tgi,gir->tgr", chunks.float(), wo_a.float())
        .reshape(rotated.shape[0], groups * rank)
        .to(torch.bfloat16)
    )
    payload, token_scale = quantize_per_token_int8(grouped)
    # Prepared A2/A3 assets use canonical ``[K, N]`` layout; the reference
    # helper consumes output-major ``[N, K]`` weights.
    accumulator = int8_matmul(payload, wo_b.transpose(0, 1).contiguous())
    output = (accumulator.float() * token_scale.float() * wo_b_scale.float().reshape(1, -1)).to(
        torch.bfloat16
    )
    return {"rotated": rotated, "grouped": grouped, "output": output}


def output_projection_bf16_reference(
    attention: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    heads: int,
    head_dim: int,
    rope_dim: int,
) -> dict[str, torch.Tensor]:
    """CPU golden for the BF16 Wo-A/Wo-B checkpoint path."""

    groups, group_input, rank = wo_a.shape
    flat = attention.reshape(attention.shape[0], heads * head_dim)
    rotated = rope_reference(flat, cosine, -sine, heads, head_dim, rope_dim)
    chunks = rotated.reshape(rotated.shape[0], groups, group_input)
    grouped = torch.einsum("tgi,gir->tgr", chunks.float(), wo_a.float()).to(torch.bfloat16)
    output = torch.matmul(grouped.float().reshape(grouped.shape[0], groups * rank), wo_b.float())
    return {
        "rotated": rotated,
        "grouped": grouped.reshape(grouped.shape[0], groups * rank),
        "output": output.to(torch.bfloat16),
    }


def decode_output_projection_reference(
    attention: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    heads: int,
    head_dim: int,
    rope_dim: int,
) -> dict[str, torch.Tensor]:
    """CPU golden for inverse RoPE, grouped Wo-A, and Wo-B W8A8."""

    groups, group_input, rank = wo_a.shape
    rotated = decode_rope_reference(attention, cosine, -sine, heads, head_dim, rope_dim)
    chunks = rotated.reshape(rotated.shape[0], groups, group_input)
    grouped = (
        torch.einsum("tgi,gir->tgr", chunks.float(), wo_a.float())
        .reshape(rotated.shape[0], groups * rank)
        .to(torch.bfloat16)
    )
    payload, token_scale = quantize_per_token_int8(grouped)
    # Prepared A2/A3 assets use canonical ``[K, N]`` layout; the reference
    # helper consumes output-major ``[N, K]`` weights.
    accumulator = int8_matmul(payload, wo_b.transpose(0, 1).contiguous())
    output = (accumulator.float() * token_scale.float() * wo_b_scale.float().reshape(1, -1)).to(
        torch.bfloat16
    )
    return {"rotated": rotated, "grouped": grouped, "output": output}


def decode_output_projection_bf16_reference(
    attention: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    heads: int,
    head_dim: int,
    rope_dim: int,
) -> dict[str, torch.Tensor]:
    """CPU golden for the BF16 Wo-A/Wo-B checkpoint path."""

    groups, group_input, rank = wo_a.shape
    flat = attention.reshape(attention.shape[0], heads * head_dim)
    rotated = decode_rope_reference(flat, cosine, -sine, heads, head_dim, rope_dim)
    chunks = rotated.reshape(rotated.shape[0], groups, group_input)
    grouped = torch.einsum("tgi,gir->tgr", chunks.float(), wo_a.float()).to(torch.bfloat16)
    output = torch.matmul(grouped.float().reshape(grouped.shape[0], groups * rank), wo_b.float())
    return {
        "rotated": rotated,
        "grouped": grouped.reshape(grouped.shape[0], groups * rank),
        "output": output.to(torch.bfloat16),
    }

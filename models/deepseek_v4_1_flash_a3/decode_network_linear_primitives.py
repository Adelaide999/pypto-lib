# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Blocked INT8 quantization, GEMM and FP32 dequantization primitives."""

import pypto.language as pl

M_TILE = 16
VEC_M_TILE = 8
K_TILE = 256
N_TILE = 64


def make_cast_tile(rows, width, input_dtype, output_dtype):
    """Select a typed identity or cross-dtype rounding conversion."""
    if input_dtype == output_dtype:
        @pl.jit.inline
        def identity(value: pl.Tensor[[rows, width], input_dtype]):
            return value

        return identity

    @pl.jit.inline
    def cast_tile(value: pl.Tensor[[rows, width], input_dtype]):
        return pl.cast(value, output_dtype, mode="rint")

    return cast_tile


def make_quantize(capacity, width, input_dtype=pl.BF16, *, rows_tile=VEC_M_TILE):
    """Specialize symmetric per-token INT8 quantization."""
    width_tile = min(K_TILE, width)
    widen = make_cast_tile(rows_tile, width_tile, input_dtype, pl.FP32)

    @pl.jit.inline
    def quantize(
        x: pl.Tensor[[capacity, width], input_dtype],
        payload: pl.Tensor[[capacity, width], pl.INT8],
        scales: pl.Tensor[[capacity, 8], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd((active + rows_tile - 1) // rows_tile, name_hint="a2_token_quantize"):
            t0 = block * rows_tile
            rows = pl.min(rows_tile, active - t0)
            maximum_row = pl.full([1, rows_tile], dtype=pl.FP32, value=0.0)
            for k0 in pl.range(0, width, width_tile):
                source = pl.slice(x, [rows_tile, width_tile], [t0, k0], valid_shape=[rows, width_tile])
                widened = widen(source)
                value = pl.set_validshape(pl.fillpad(widened, pad_value=pl.PadValue.zero), rows_tile, width_tile)
                chunk_maximum = pl.reshape(pl.row_max(pl.abs(value)), [1, rows_tile])
                maximum_row = pl.maximum(maximum_row, chunk_maximum)
            maximum = pl.reshape(maximum_row, [rows_tile, 1])
            scale = pl.mul(maximum, 1.0 / 127.0)
            numerator = pl.full([1, rows_tile], dtype=pl.FP32, value=127.0)
            safe_maximum_row = pl.maximum(maximum_row, 1e-30)
            inverse_row = pl.div(numerator, safe_maximum_row)
            inverse = pl.reshape(inverse_row, [rows_tile, 1])
            scale_rows = pl.full([rows_tile, 8], dtype=pl.FP32, value=1.0)
            expanded_scale = pl.row_expand_mul(scale_rows, scale)
            scales[t0:t0 + rows_tile, :] = pl.set_validshape(expanded_scale, rows, 8)
            for k0 in pl.range(0, width, width_tile):
                source = pl.slice(x, [rows_tile, width_tile], [t0, k0], valid_shape=[rows, width_tile])
                widened = widen(source)
                value = pl.set_validshape(pl.fillpad(widened, pad_value=pl.PadValue.zero), rows_tile, width_tile)
                normalized = pl.row_expand_mul(value, inverse)
                bounded = pl.minimum(pl.maximum(normalized, -127.0), 127.0)
                rounded = pl.cast(bounded, pl.INT32, mode="rint")
                exact_half = pl.cast(rounded, pl.FP16, mode="rint")
                codes = pl.cast(exact_half, pl.INT8, mode="trunc")
                payload[t0:t0 + rows_tile, k0:k0 + width_tile] = pl.set_validshape(codes, rows, width_tile)
        return payload, scales

    return quantize


def make_pack_weight(width, output_width):
    """Reorder canonical [K,N] INT8 bytes to [N/64,K,64]."""
    packed_rows = width * (output_width // N_TILE)

    @pl.jit.inline
    def pack_weight(
        weight: pl.Tensor[[width, output_width], pl.INT8],
        packed: pl.Tensor[[packed_rows, N_TILE], pl.INT8],
    ):
        for nb in pl.spmd(output_width // N_TILE, name_hint="a2_pack_weight"):
            n0 = nb * N_TILE
            for k0 in pl.range(0, width, K_TILE):
                block = weight[k0:k0 + K_TILE, n0:n0 + N_TILE]
                packed_k0 = nb * width + k0
                packed[packed_k0:packed_k0 + K_TILE, :] = block
        return packed

    return pack_weight


def make_dot(capacity, width, output_width):
    """Accumulate complete blocked INT8 products in INT32."""
    packed_rows = width * (output_width // N_TILE)

    @pl.jit.inline
    def dot(
        payload: pl.Tensor[[capacity, width], pl.INT8],
        packed: pl.Tensor[[packed_rows, N_TILE], pl.INT8],
        accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        row_blocks = (active + M_TILE - 1) // M_TILE
        for block in pl.spmd(row_blocks * (output_width // N_TILE), name_hint="a2_int8_dot"):
            t0 = (block // (output_width // N_TILE)) * M_TILE
            nb = block % (output_width // N_TILE)
            n0 = nb * N_TILE
            rows = pl.min(M_TILE, active - t0)
            acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, width, K_TILE, stage=2):
                a = pl.slice(payload, [M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                a = pl.set_validshape(pl.fillpad(a, pad_value=pl.PadValue.zero), M_TILE, K_TILE)
                packed_k0 = nb * width + k0
                b = packed[packed_k0:packed_k0 + K_TILE, :]
                acc = pl.matmul_acc(acc, a, b, init_cond=(k0 == 0))
            accumulator[t0:t0 + M_TILE, n0:n0 + N_TILE] = acc
        return accumulator

    return dot


def make_dequantize(capacity, output_width, combine_scales=False, channel_dtype=pl.FP32):
    """Apply channel-first or combined scales with the selected channel rounding."""
    narrow_channel = make_cast_tile(VEC_M_TILE, N_TILE, pl.FP32, channel_dtype)
    widen_channel = make_cast_tile(VEC_M_TILE, N_TILE, channel_dtype, pl.FP32)

    @pl.jit.inline
    def dequantize(
        accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
        channel_scale: pl.Tensor[[output_width], pl.FP32],
        token_scale: pl.Tensor[[capacity, 8], pl.FP32],
        output: pl.Tensor[[capacity, output_width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        row_blocks = (active + VEC_M_TILE - 1) // VEC_M_TILE
        for block in pl.spmd(row_blocks * (output_width // N_TILE), name_hint="a2_dequantize"):
            t0 = (block // (output_width // N_TILE)) * VEC_M_TILE
            n0 = (block % (output_width // N_TILE)) * N_TILE
            rows = pl.min(VEC_M_TILE, active - t0)
            integer = accumulator[t0:t0 + VEC_M_TILE, n0:n0 + N_TILE]
            value = pl.cast(integer, pl.FP32, mode="rint")
            channels = pl.reshape(channel_scale[n0:n0 + N_TILE], [1, N_TILE])
            scale = pl.row_min(token_scale[t0:t0 + VEC_M_TILE, :])
            if combine_scales:
                ones = pl.full([VEC_M_TILE, N_TILE], dtype=pl.FP32, value=1.0)
                channel_rows = pl.col_expand_mul(ones, channels)
                combined = pl.row_expand_mul(channel_rows, scale)
                scaled = pl.mul(value, combined)
            else:
                scaled_channels = pl.create_tensor([VEC_M_TILE, N_TILE], dtype=pl.FP32)
                scaled_channels[:, :] = pl.col_expand_mul(value, channels)
                rounded_channels = pl.create_tensor([VEC_M_TILE, N_TILE], dtype=channel_dtype)
                rounded_channels[:, :] = narrow_channel(scaled_channels)
                channel_fp32 = widen_channel(rounded_channels)
                scaled = pl.row_expand_mul(channel_fp32, scale)
            narrowed = pl.cast(scaled, pl.BF16, mode="rint")
            output[t0:t0 + VEC_M_TILE, n0:n0 + N_TILE] = pl.set_validshape(narrowed, rows, N_TILE)
        return output

    return dequantize

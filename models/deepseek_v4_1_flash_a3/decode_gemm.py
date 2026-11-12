# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode direct W8A8 linear with phase-specific activation quantization."""

from __future__ import annotations

import pypto.language as pl

M_TILE = 16
N_TILE = 64
K_TILE = 256


def make_int8_linear_direct(capacity: int, width: int, output_width: int):
    """Build a W8A8 linear that consumes canonical ``[K, N]`` weights.

    The packed-weight helper is useful for the leaf ABI, but materializing a
    full packed matrix is too large for the A2/A3 Mat buffer at V4.1 widths.
    This variant loads one ``[K_TILE, N_TILE]`` canonical tile per GEMM
    reduction and keeps only the accumulator tile on core.
    """

    if capacity % M_TILE or width % K_TILE or output_width % N_TILE:
        raise ValueError("direct INT8 GEMM requires aligned capacity, K and N")

    from .quant import make_decode_dynamic_int8_quantize as make_dynamic_int8_quantize

    quantize = make_dynamic_int8_quantize(capacity, width)
    blocks = output_width // N_TILE

    @pl.jit
    def linear(
        value: pl.Tensor[[capacity, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.INT8],
        channel_scale: pl.Tensor[[output_width], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, output_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        payload = pl.create_tensor([capacity, width], dtype=pl.INT8)
        token_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quantize(value, payload, token_scale, active)
        for block in pl.spmd(
            ((active + M_TILE - 1) // M_TILE) * blocks, name_hint="dsv41_a2a3_direct_int8_linear"
        ):
            t0 = (block // blocks) * M_TILE
            n0 = (block % blocks) * N_TILE
            rows = pl.min(M_TILE, active - t0)
            acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, width, K_TILE, stage=2):
                lhs = pl.slice(payload, [M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                lhs = pl.set_validshape(pl.fillpad(lhs, pad_value=pl.PadValue.zero), M_TILE, K_TILE)
                rhs = weight[k0 : k0 + K_TILE, n0 : n0 + N_TILE]
                acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(k0 == 0))
            value_fp32 = pl.cast(acc, pl.FP32)
            channels = pl.reshape(channel_scale[n0 : n0 + N_TILE], [1, N_TILE])
            value_fp32 = pl.col_expand_mul(value_fp32, channels)
            value_fp32 = pl.row_expand_mul(value_fp32, token_scale[t0 : t0 + M_TILE, :])
            output[t0 : t0 + M_TILE, n0 : n0 + N_TILE] = pl.set_validshape(
                pl.cast(value_fp32, pl.BF16, mode="rint"), rows, N_TILE
            )
        return output

    return linear


__all__ = [
    "make_int8_linear_direct",
]

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tiled RMS normalization with explicit FP32 accumulation and BF16 outputs."""

import pypto.language as pl
import torch
from .config import FLASH as M
from .linear_primitives import make_cast_tile, VEC_M_TILE


# model config
D = M.hidden_size
EPS = M.rms_norm_eps

# tiling
NORM_T_TILE = 8
NORM_D_TILE = 512


def make_rms_norm_tiled(output_dtype=pl.BF16):
    """Specialize the tiled normalization output dtype before compilation."""
    convert = make_cast_tile(NORM_T_TILE, NORM_D_TILE, pl.FP32, output_dtype)

    @pl.jit.inline
    def norm_tiled(
        x: pl.Tensor,
        norm_w: pl.Tensor[[D], pl.BF16],
        x_normed: pl.Tensor,
        block: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        """Normalize one eight-row block using 512-column accumulation."""
        t0 = block * NORM_T_TILE
        valid_rows = pl.min(NORM_T_TILE, num_tokens - t0)
        sq_sum = pl.full([1, NORM_T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(D // NORM_D_TILE, stage=2):
            k0 = kb * NORM_D_TILE
            source = pl.full([NORM_T_TILE, NORM_D_TILE], dtype=pl.BF16, value=0.0)
            for row in pl.range(valid_rows):
                source_row = t0 + row
                source[row:row + 1, :] = x[source_row:source_row + 1, k0:k0 + NORM_D_TILE]
            value = pl.cast(source, target_type=pl.FP32)
            square = pl.mul(value, value)
            partial = pl.reshape(pl.row_sum(square), [1, NORM_T_TILE])
            sq_sum = pl.add(sq_sum, partial)
        mean = pl.mul(sq_sum, 1.0 / D)
        inverse = pl.rsqrt(pl.add(mean, EPS), high_precision=True)
        inv_rms = pl.reshape(inverse, [NORM_T_TILE, 1])
        for kb in pl.pipeline(D // NORM_D_TILE, stage=2):
            k0 = kb * NORM_D_TILE
            source = pl.full([NORM_T_TILE, NORM_D_TILE], dtype=pl.BF16, value=0.0)
            for row in pl.range(valid_rows):
                source_row = t0 + row
                source[row:row + 1, :] = x[source_row:source_row + 1, k0:k0 + NORM_D_TILE]
            value = pl.cast(source, target_type=pl.FP32)
            weight = pl.cast(norm_w[k0:k0 + NORM_D_TILE], target_type=pl.FP32)
            gamma = pl.reshape(weight, [1, NORM_D_TILE])
            scaled = pl.row_expand_mul(value, inv_rms)
            normalized = pl.col_expand_mul(scaled, gamma)
            normalized_storage = pl.create_tensor([NORM_T_TILE, NORM_D_TILE], dtype=pl.FP32)
            normalized_storage[:, :] = normalized
            rounded_storage = pl.create_tensor([NORM_T_TILE, NORM_D_TILE], dtype=output_dtype)
            rounded_storage[:, :] = convert(normalized_storage)
            result = pl.set_validshape(rounded_storage, valid_rows, NORM_D_TILE)
            x_normed[t0:t0 + NORM_T_TILE, k0:k0 + NORM_D_TILE] = result
        return x_normed

    return norm_tiled


rms_norm_tiled = make_rms_norm_tiled()


@pl.jit.inline
def rms_norm(
    x: pl.Tensor,
    weight: pl.Tensor,
    output: pl.Tensor,
    block: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.constexpr,
):
    """Normalize one eight-row block using whole-row reduction."""
    t = block * 8
    rows = pl.min(8, num_tokens - t)
    source = pl.slice(x, [8, width], [t, 0], valid_shape=[rows, width])
    source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 8, width)
    value = pl.cast(source, pl.FP32)
    row_inverse = pl.rsqrt(
        pl.add(pl.mul(pl.row_sum(pl.mul(value, value)), 1.0 / width), EPS),
        high_precision=True,
    )
    gamma = pl.reshape(pl.cast(weight[:], pl.FP32), [1, width])
    normalized = pl.col_expand_mul(pl.row_expand_mul(value, row_inverse), gamma)
    output[t : t + 8, :] = pl.set_validshape(
        pl.cast(normalized, pl.BF16, mode="rint"), rows, width
    )
    return output


def golden_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = M.rms_norm_eps) -> torch.Tensor:
    """Apply RMSNorm in FP32 and restore the activation dtype."""
    dtype = x.dtype
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    return (value * weight.float()).to(dtype)


def make_blocked_rmsnorm(capacity, width, eps):
    """Specialize weighted FP32 RMSNorm with a BF16 output."""

    @pl.jit.inline
    def norm(
        x: pl.Tensor[[capacity, width], pl.BF16],
        weight: pl.Tensor[[width], pl.BF16],
        output: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd((active + VEC_M_TILE - 1) // VEC_M_TILE, name_hint="a2_rms_norm"):
            t0 = block * VEC_M_TILE
            rows = pl.min(VEC_M_TILE, active - t0)
            source = pl.slice(x, [VEC_M_TILE, width], [t0, 0], valid_shape=[rows, width])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), VEC_M_TILE, width)
            value = pl.cast(source, pl.FP32)
            square = pl.mul(value, value)
            mean = pl.mul(pl.row_sum(square), 1.0 / width)
            inverse = pl.rsqrt(pl.add(mean, eps), high_precision=True)
            normalized = pl.row_expand_mul(value, inverse)
            gamma = pl.reshape(pl.cast(weight[:], pl.FP32), [1, width])
            weighted = pl.col_expand_mul(normalized, gamma)
            narrowed = pl.cast(weighted, pl.BF16, mode="rint")
            output[t0:t0 + VEC_M_TILE, :] = pl.set_validshape(narrowed, rows, width)
        return output

    return norm

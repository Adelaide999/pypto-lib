# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Index cache, query projection, candidate pooling and selection."""

from __future__ import annotations


import torch
import pypto.language as pl

from .operator_config import CONTRACT

from .quant import make_dynamic_int8_quantize
from .norm_rope import make_rope, rope_reference
from .quant import w8a8_linear

from .quant import make_decode_dynamic_int8_quantize as decode_make_dynamic_int8_quantize
from .decode_norm_rope import make_rope as decode_make_rope, rope_reference as decode_rope_reference


# ``npu_dynamic_quant`` is nearest rounding for ordinary values, but an exact
# half-way value is rounded toward zero on the A2/A3 kernel.  Keep the small
# bias in FP32 so the published FP16 scale remains unchanged.
_ROUND_TIE_EPS = 2.0**-20


def quantize_index_cache(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[..., 128]`` index vectors using FP16 dequant scales."""

    if value.shape[-1] != CONTRACT.index_head_dim:
        raise ValueError(f"index cache width must be {CONTRACT.index_head_dim}")
    source = value.float()
    amax = source.abs().amax(dim=-1, keepdim=True)
    # vLLM-Ascend computes the INT8 payload with the unrounded FP32 scale and
    # only converts the published row scale to FP16.  Reusing the rounded
    # FP16 scale for the payload changes exact half-way cases (for example
    # -63.5) and causes a one-code mismatch with ``npu_dynamic_quant``.
    raw_scale = torch.maximum(
        amax / CONTRACT.int8_scale_max,
        torch.full_like(amax, CONTRACT.int8_amax_epsilon),
    )
    scale = torch.maximum(
        raw_scale.to(torch.float16), torch.full_like(raw_scale, 2.0**-24, dtype=torch.float16)
    )
    normalized = source / raw_scale
    epsilon = torch.maximum(normalized.abs(), torch.ones_like(normalized)) * _ROUND_TIE_EPS
    # The A2/A3 kernel's observed tie behavior is asymmetric: positive
    # half-way values round down while negative values retain the normal rint
    # rule.  This only affects exact half-way cases.
    rounded = torch.where(
        normalized >= 0,
        torch.round(normalized - epsilon),
        torch.round(normalized),
    )
    payload = rounded.clamp(-127, 127).to(torch.int8)
    return payload, scale.squeeze(-1)


def dequantize_index_cache(payload: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return payload.float() * scale.float().unsqueeze(-1)


def make_index_cache_quantize(capacity: int, heads: int, head_dim: int = 128):
    if head_dim != CONTRACT.index_head_dim:
        raise ValueError("A2/A3 index cache uses 128-wide heads")

    @pl.jit.inline
    def quantize(
        value: pl.Tensor[[capacity, heads, head_dim], pl.BF16],
        payload: pl.Tensor[[capacity, heads, head_dim], pl.INT8],
        scales: pl.Tensor[[capacity, heads], pl.FP16],
        active: pl.Scalar[pl.INT32],
    ):
        source = pl.reshape(value, [capacity * heads, head_dim])
        output = pl.reshape(payload, [capacity * heads, head_dim])
        rows = (active * heads + 15) // 16
        with pl.spmd(rows, name_hint="dsv41_a2a3_index_cache_quant"):
            block = pl.tile.get_block_idx()
            row = block * 16
            valid_rows = pl.min(16, active * heads - row)
            tile = pl.slice(source, [16, head_dim], [row, 0], valid_shape=[valid_rows, head_dim])
            tile = pl.set_validshape(pl.fillpad(tile, pad_value=pl.PadValue.zero), 16, head_dim)
            value_fp32 = pl.cast(tile, pl.FP32)
            amax = pl.row_max(pl.abs(value_fp32))
            raw_scale = pl.maximum(pl.div(amax, 127.0), 1.0e-30)
            rounded_scale = pl.cast(pl.cast(raw_scale, pl.FP16, mode="rint"), pl.FP32)
            scale = pl.maximum(rounded_scale, 2.0**-24)
            normalized = pl.row_expand_div(value_fp32, raw_scale)
            # Match the A2/A3 dynamic-quant tie rule: nearest for ordinary
            # values, half-way values toward zero.  Splitting signs keeps the
            # conversion in truncation mode, which is legal for INT32 output.
            positive = pl.maximum(normalized, 0.0)
            negative = pl.minimum(normalized, 0.0)
            epsilon = pl.mul(pl.maximum(pl.abs(normalized), 1.0), _ROUND_TIE_EPS)
            positive = pl.cast(pl.sub(positive, epsilon), pl.INT32, mode="rint")
            negative = pl.cast(negative, pl.INT32, mode="rint")
            bounded = pl.minimum(pl.maximum(pl.add(positive, negative), -127), 127)
            rounded = pl.cast(bounded, pl.INT8, mode="trunc")
            output[row : row + 16, :] = pl.set_validshape(rounded, valid_rows, head_dim)
            head = row % heads
            token = row // heads
            scale_row = pl.reshape(pl.cast(scale, pl.FP16), [1, 16])
            scales[token : token + 1, head : head + 16] = scale_row
        return payload, scales

    return quantize


def make_index_quantize(capacity: int, heads: int, head_dim: int = 128):
    quantize = make_index_cache_quantize(capacity, heads, head_dim)

    @pl.jit
    def kernel(
        value: pl.Tensor[[capacity, heads, head_dim], pl.BF16],
        payload: pl.InOut[pl.Tensor[[capacity, heads, head_dim], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, heads], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        quantize(value, payload, scales, active)
        return payload, scales

    return kernel


# ``npu_dynamic_quant`` is nearest rounding for ordinary values, but an exact
# half-way value is rounded toward zero on the A2/A3 kernel.  Keep the small
# bias in FP32 so the published FP16 scale remains unchanged.
DECODE__ROUND_TIE_EPS = 2.0**-20


def decode_quantize_index_cache(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[..., 128]`` index vectors using FP16 dequant scales."""

    if value.shape[-1] != CONTRACT.index_head_dim:
        raise ValueError(f"index cache width must be {CONTRACT.index_head_dim}")
    source = value.float()
    amax = source.abs().amax(dim=-1, keepdim=True)
    # vLLM-Ascend computes the INT8 payload with the unrounded FP32 scale and
    # only converts the published row scale to FP16.  Reusing the rounded
    # FP16 scale for the payload changes exact half-way cases (for example
    # -63.5) and causes a one-code mismatch with ``npu_dynamic_quant``.
    raw_scale = torch.maximum(
        amax / CONTRACT.int8_scale_max,
        torch.full_like(amax, CONTRACT.int8_amax_epsilon),
    )
    scale = torch.maximum(
        raw_scale.to(torch.float16), torch.full_like(raw_scale, 2.0**-24, dtype=torch.float16)
    )
    normalized = source / raw_scale
    epsilon = torch.maximum(normalized.abs(), torch.ones_like(normalized)) * DECODE__ROUND_TIE_EPS
    # The A2/A3 kernel's observed tie behavior is asymmetric: positive
    # half-way values round down while negative values retain the normal rint
    # rule.  This only affects exact half-way cases.
    rounded = torch.where(
        normalized >= 0,
        torch.round(normalized - epsilon),
        torch.round(normalized),
    )
    payload = rounded.clamp(-127, 127).to(torch.int8)
    return payload, scale.squeeze(-1)


def make_decode_index_cache_quantize(capacity: int, heads: int, head_dim: int = 128):
    if head_dim != CONTRACT.index_head_dim:
        raise ValueError("A2/A3 index cache uses 128-wide heads")

    @pl.jit.inline
    def quantize(
        value: pl.Tensor[[capacity, heads, head_dim], pl.BF16],
        payload: pl.Tensor[[capacity, heads, head_dim], pl.INT8],
        scales: pl.Tensor[[capacity, heads], pl.FP16],
        active: pl.Scalar[pl.INT32],
    ):
        source = pl.reshape(value, [capacity * heads, head_dim])
        output = pl.reshape(payload, [capacity * heads, head_dim])
        rows = (active * heads + 15) // 16
        with pl.spmd(rows, name_hint="dsv41_a2a3_index_cache_quant"):
            block = pl.tile.get_block_idx()
            row = block * 16
            valid_rows = pl.min(16, active * heads - row)
            tile = pl.slice(source, [16, head_dim], [row, 0], valid_shape=[valid_rows, head_dim])
            tile = pl.set_validshape(pl.fillpad(tile, pad_value=pl.PadValue.zero), 16, head_dim)
            value_fp32 = pl.cast(tile, pl.FP32)
            amax = pl.row_max(pl.abs(value_fp32))
            raw_scale = pl.maximum(pl.div(amax, 127.0), 1.0e-30)
            rounded_scale = pl.cast(pl.cast(raw_scale, pl.FP16, mode="rint"), pl.FP32)
            scale = pl.maximum(rounded_scale, 2.0**-24)
            normalized = pl.row_expand_div(value_fp32, raw_scale)
            # Match the A2/A3 dynamic-quant tie rule: nearest for ordinary
            # values, half-way values toward zero.  Splitting signs keeps the
            # conversion in truncation mode, which is legal for INT32 output.
            positive = pl.maximum(normalized, 0.0)
            negative = pl.minimum(normalized, 0.0)
            epsilon = pl.mul(pl.maximum(pl.abs(normalized), 1.0), DECODE__ROUND_TIE_EPS)
            positive = pl.cast(pl.sub(positive, epsilon), pl.INT32, mode="rint")
            negative = pl.cast(negative, pl.INT32, mode="rint")
            bounded = pl.minimum(pl.maximum(pl.add(positive, negative), -127), 127)
            rounded = pl.cast(bounded, pl.INT8, mode="trunc")
            output[row : row + 16, :] = pl.set_validshape(rounded, valid_rows, head_dim)
            head = row % heads
            token = row // heads
            scale_row = pl.reshape(pl.cast(scale, pl.FP16), [1, 16])
            scales[token : token + 1, head : head + 16] = scale_row
        return payload, scales

    return quantize


INDEXER_TOPK = 512
SORT_LEAF = 512
NEGATIVE_INF = -3.0e38


def make_indexer_prepare(
    capacity: int,
    topk: int = INDEXER_TOPK,
    *,
    compress_ratio: int = 2,
):
    """Build a fixed-bucket A2/A3 index preparation kernel."""

    if capacity <= 0 or capacity % 16:
        raise ValueError("capacity must be a positive multiple of 16")
    if topk <= 0 or topk % 32 or topk > SORT_LEAF:
        raise ValueError("topk must be a positive multiple of 32 and <= 512")
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")

    @pl.jit
    def kernel(
        raw_selected: pl.Tensor[[capacity, topk], pl.INT32],
        positions: pl.Tensor[[capacity], pl.INT32],
        indices: pl.Out[pl.Tensor[[capacity, topk], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(active, name_hint="dsv41_a2a3_prepare_indexer_indices"):
            token = pl.tile.get_block_idx()
            position = pl.max(pl.read(positions, [token]), 0)
            visible = pl.min((position + 1) // compress_ratio, topk)
            # sort32 is descending.  Negated logical IDs therefore produce
            # ascending chronological IDs, while the very negative sentinel
            # remains at the tail.
            scores = pl.tile.full([1, topk], dtype=pl.FP32, value=NEGATIVE_INF)
            ids = pl.tile.full([1, topk], dtype=pl.INT32, value=-1)
            for lane in pl.range(topk):
                index = pl.read(raw_selected, [token, lane])
                if index >= 0:
                    if index < visible:
                        pl.tile.write(scores, [0, lane], -pl.cast(index, pl.FP32))
                        pl.tile.write(ids, [0, lane], index)
            pairs = pl.tile.sort32(scores, pl.reinterpret_view(ids, pl.UINT32))
            if topk >= 128:
                pairs = pl.tile.mrgsort(pairs, block_len=64)
                pairs = pl.tile.mrgsort(pairs, block_len=256)
            ordered = pl.tile.slice(pairs, [1, 2 * topk], [0, 0])
            output = pl.tile.gather_mask(
                ordered, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
            )
            pl.store(output, [token, 0], indices)
        return indices

    return kernel


INDEX_HEAD_DIM = 128
INDEX_HEADS = 32
INDEX_ROPE_DIM = 64
INDEX_SCALE_MIN = 2.0**-24


def _make_indexer_query_quantize_inline(
    capacity: int,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
):
    """Build the PyPTO per-head INT8 query quantizer used by QLI."""

    if capacity % 16 or heads <= 0 or head_dim != INDEX_HEAD_DIM:
        raise ValueError("indexer quantization requires capacity divisible by 16 and head_dim=128")
    total_rows = capacity * heads

    @pl.jit.inline
    def quantize(
        value: pl.Tensor[[capacity, heads, head_dim], pl.BF16],
        payload: pl.InOut[pl.Tensor[[capacity, heads, head_dim], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, heads], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        source = pl.reshape(value, [total_rows, head_dim])
        output = pl.reshape(payload, [total_rows, head_dim])
        scale_output = pl.reshape(scales, [total_rows, 1])
        blocks = (active * heads + 15) // 16
        for block in pl.spmd(blocks, name_hint="dsv41_a2a3_indexer_query_quant"):
            row = block * 16
            valid_rows = pl.min(16, active * heads - row)
            tile = pl.slice(source, [16, head_dim], [row, 0], valid_shape=[valid_rows, head_dim])
            tile = pl.set_validshape(pl.fillpad(tile, pad_value=pl.PadValue.zero), 16, head_dim)
            value_fp32 = pl.cast(tile, pl.FP32)
            amax = pl.row_max(pl.abs(value_fp32))
            raw_scale = pl.div(amax, 127.0)
            rounded_scale = pl.cast(pl.cast(raw_scale, pl.FP16, mode="rint"), pl.FP32)
            rounded_scale = pl.maximum(rounded_scale, INDEX_SCALE_MIN)
            normalized = pl.row_expand_div(value_fp32, rounded_scale)
            rounded = pl.cast(
                pl.minimum(pl.maximum(normalized, -127.0), 127.0),
                pl.INT32,
                mode="rint",
            )
            output[row : row + 16, :] = pl.set_validshape(
                pl.cast(rounded, pl.INT8, mode="trunc"), valid_rows, head_dim
            )
            scale_output[row : row + 16, 0:1] = pl.set_validshape(
                pl.reshape(pl.cast(rounded_scale, pl.FP16, mode="rint"), [16, 1]),
                valid_rows,
                1,
            )
        return payload, scales

    return quantize


def make_indexer_query_quantize(
    capacity: int,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
):
    """Build a standalone PyPTO indexer quantization entry point."""

    quantize = _make_indexer_query_quantize_inline(capacity, heads, head_dim)

    @pl.jit
    def kernel(
        value: pl.Tensor[[capacity, heads, head_dim], pl.BF16],
        payload: pl.InOut[pl.Tensor[[capacity, heads, head_dim], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, heads], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        quantize(value, payload, scales, active)
        return payload, scales

    return kernel


def make_l3_indexer_query(
    capacity: int,
    q_lora_rank: int = 1280,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
    rope_dim: int = INDEX_ROPE_DIM,
):
    """Project ``q_lora`` and publish the QLI INT8 query boundary.

    ``weight`` is canonical A2/A3 ``[K, N]`` INT8 and ``weight_scale`` is
    FP32 per output channel.  ``query`` remains BF16 for diagnostics while
    ``payload``/``scales`` are the exact QLI ABI.
    """

    if heads * head_dim != 4096 or rope_dim != INDEX_ROPE_DIM:
        raise ValueError("V4.1 A2/A3 indexer uses 32 heads x 128 with a 64-wide RoPE tail")
    rotate = make_rope(capacity, heads, head_dim, rope_dim)
    quantize = _make_indexer_query_quantize_inline(capacity, heads, head_dim)
    quantize_input = make_dynamic_int8_quantize(capacity, q_lora_rank)
    pairs = rope_dim // 2
    width = heads * head_dim
    blocks = width // 64

    @pl.jit.inline
    def project(
        value: pl.Tensor[[capacity, q_lora_rank], pl.BF16],
        weight: pl.Tensor[[q_lora_rank, width], pl.INT8],
        weight_scale: pl.Tensor[[width], pl.FP32],
        output: pl.Tensor[[capacity, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        payload = pl.create_tensor([capacity, q_lora_rank], dtype=pl.INT8)
        token_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quantize_input(value, payload, token_scale, active)
        for block in pl.spmd(((active + 15) // 16) * blocks, name_hint="dsv41_a2a3_l3_indexer_gemm"):
            t0 = (block // blocks) * 16
            n0 = (block % blocks) * 64
            rows = pl.min(16, active - t0)
            acc = pl.create_tensor([16, 64], dtype=pl.INT32)
            for k0 in pl.pipeline(0, q_lora_rank, 256, stage=2):
                lhs = pl.slice(payload, [16, 256], [t0, k0], valid_shape=[rows, 256])
                lhs = pl.set_validshape(pl.fillpad(lhs, pad_value=pl.PadValue.zero), 16, 256)
                rhs = weight[k0 : k0 + 256, n0 : n0 + 64]
                acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(k0 == 0))
            value_fp32 = pl.cast(acc, pl.FP32)
            value_fp32 = pl.col_expand_mul(value_fp32, pl.reshape(weight_scale[n0 : n0 + 64], [1, 64]))
            value_fp32 = pl.row_expand_mul(value_fp32, token_scale[t0 : t0 + 16, :])
            output[t0 : t0 + 16, n0 : n0 + 64] = pl.set_validshape(
                pl.cast(value_fp32, pl.BF16, mode="rint"), rows, 64
            )
        return output

    @pl.jit
    def kernel(
        q_lora: pl.Tensor[[capacity, q_lora_rank], pl.BF16],
        weight: pl.Tensor[[q_lora_rank, width], pl.INT8],
        weight_scale: pl.Tensor[[width], pl.FP32],
        cosine: pl.Tensor[[capacity, pairs], pl.FP32],
        sine: pl.Tensor[[capacity, pairs], pl.FP32],
        query: pl.InOut[pl.Tensor[[capacity, heads, head_dim], pl.BF16]],
        payload: pl.InOut[pl.Tensor[[capacity, heads, head_dim], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, heads], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        projected = pl.create_tensor([capacity, width], dtype=pl.BF16)
        project(q_lora, weight, weight_scale, projected, active)
        query_flat = pl.reshape(query, [capacity, width])
        rotate(projected, cosine, sine, query_flat, active)
        quantize(query, payload, scales, active)
        return query, payload, scales

    return kernel


DECODE_INDEX_SCALE_MIN = 2.0**-24


def _decode_make_indexer_query_quantize_inline(
    capacity: int,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
):
    """Build the PyPTO per-head INT8 query quantizer used by QLI."""

    if capacity % 16 or heads <= 0 or head_dim != INDEX_HEAD_DIM:
        raise ValueError("indexer quantization requires capacity divisible by 16 and head_dim=128")
    total_rows = capacity * heads

    @pl.jit.inline
    def quantize(
        value: pl.Tensor[[capacity, heads, head_dim], pl.BF16],
        payload: pl.InOut[pl.Tensor[[capacity, heads, head_dim], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, heads], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        source = pl.reshape(value, [total_rows, head_dim])
        output = pl.reshape(payload, [total_rows, head_dim])
        scale_output = pl.reshape(scales, [total_rows, 1])
        blocks = (active * heads + 15) // 16
        for block in pl.spmd(blocks, name_hint="dsv41_a2a3_indexer_query_quant"):
            row = block * 16
            valid_rows = pl.min(16, active * heads - row)
            tile = pl.slice(source, [16, head_dim], [row, 0], valid_shape=[valid_rows, head_dim])
            tile = pl.set_validshape(pl.fillpad(tile, pad_value=pl.PadValue.zero), 16, head_dim)
            value_fp32 = pl.cast(tile, pl.FP32)
            amax = pl.row_max(pl.abs(value_fp32))
            raw_scale = pl.div(amax, 127.0)
            rounded_scale = pl.cast(pl.cast(raw_scale, pl.FP16, mode="rint"), pl.FP32)
            rounded_scale = pl.maximum(rounded_scale, DECODE_INDEX_SCALE_MIN)
            normalized = pl.row_expand_div(value_fp32, rounded_scale)
            rounded = pl.cast(
                pl.minimum(pl.maximum(normalized, -127.0), 127.0),
                pl.INT32,
                mode="rint",
            )
            output[row : row + 16, :] = pl.set_validshape(
                pl.cast(rounded, pl.INT8, mode="trunc"), valid_rows, head_dim
            )
            scale_output[row : row + 16, 0:1] = pl.set_validshape(
                pl.reshape(pl.cast(rounded_scale, pl.FP16, mode="rint"), [16, 1]),
                valid_rows,
                1,
            )
        return payload, scales

    return quantize


def make_decode_indexer_query_quantize(
    capacity: int,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
):
    """Build a standalone PyPTO indexer quantization entry point."""

    quantize = _decode_make_indexer_query_quantize_inline(capacity, heads, head_dim)

    @pl.jit
    def kernel(
        value: pl.Tensor[[capacity, heads, head_dim], pl.BF16],
        payload: pl.InOut[pl.Tensor[[capacity, heads, head_dim], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, heads], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        quantize(value, payload, scales, active)
        return payload, scales

    return kernel


def make_decode_l3_indexer_query(
    capacity: int,
    q_lora_rank: int = 1280,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
    rope_dim: int = INDEX_ROPE_DIM,
):
    """Project ``q_lora`` and publish the QLI INT8 query boundary.

    ``weight`` is canonical A2/A3 ``[K, N]`` INT8 and ``weight_scale`` is
    FP32 per output channel.  ``query`` remains BF16 for diagnostics while
    ``payload``/``scales`` are the exact QLI ABI.
    """

    if heads * head_dim != 4096 or rope_dim != INDEX_ROPE_DIM:
        raise ValueError("V4.1 A2/A3 indexer uses 32 heads x 128 with a 64-wide RoPE tail")
    rotate = decode_make_rope(capacity, heads, head_dim, rope_dim)
    quantize = _decode_make_indexer_query_quantize_inline(capacity, heads, head_dim)
    quantize_input = decode_make_dynamic_int8_quantize(capacity, q_lora_rank)
    pairs = rope_dim // 2
    width = heads * head_dim
    blocks = width // 64

    @pl.jit.inline
    def project(
        value: pl.Tensor[[capacity, q_lora_rank], pl.BF16],
        weight: pl.Tensor[[q_lora_rank, width], pl.INT8],
        weight_scale: pl.Tensor[[width], pl.FP32],
        output: pl.Tensor[[capacity, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        payload = pl.create_tensor([capacity, q_lora_rank], dtype=pl.INT8)
        token_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quantize_input(value, payload, token_scale, active)
        for block in pl.spmd(((active + 15) // 16) * blocks, name_hint="dsv41_a2a3_l3_indexer_gemm"):
            t0 = (block // blocks) * 16
            n0 = (block % blocks) * 64
            rows = pl.min(16, active - t0)
            acc = pl.create_tensor([16, 64], dtype=pl.INT32)
            for k0 in pl.pipeline(0, q_lora_rank, 256, stage=2):
                lhs = pl.slice(payload, [16, 256], [t0, k0], valid_shape=[rows, 256])
                lhs = pl.set_validshape(pl.fillpad(lhs, pad_value=pl.PadValue.zero), 16, 256)
                rhs = weight[k0 : k0 + 256, n0 : n0 + 64]
                acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(k0 == 0))
            value_fp32 = pl.cast(acc, pl.FP32)
            value_fp32 = pl.col_expand_mul(value_fp32, pl.reshape(weight_scale[n0 : n0 + 64], [1, 64]))
            value_fp32 = pl.row_expand_mul(value_fp32, token_scale[t0 : t0 + 16, :])
            output[t0 : t0 + 16, n0 : n0 + 64] = pl.set_validshape(
                pl.cast(value_fp32, pl.BF16, mode="rint"), rows, 64
            )
        return output

    @pl.jit
    def kernel(
        q_lora: pl.Tensor[[capacity, q_lora_rank], pl.BF16],
        weight: pl.Tensor[[q_lora_rank, width], pl.INT8],
        weight_scale: pl.Tensor[[width], pl.FP32],
        cosine: pl.Tensor[[capacity, pairs], pl.FP32],
        sine: pl.Tensor[[capacity, pairs], pl.FP32],
        query: pl.InOut[pl.Tensor[[capacity, heads, head_dim], pl.BF16]],
        payload: pl.InOut[pl.Tensor[[capacity, heads, head_dim], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, heads], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        projected = pl.create_tensor([capacity, width], dtype=pl.BF16)
        project(q_lora, weight, weight_scale, projected, active)
        query_flat = pl.reshape(query, [capacity, width])
        rotate(projected, cosine, sine, query_flat, active)
        quantize(query, payload, scales, active)
        return query, payload, scales

    return kernel


# model config
CANDIDATE_BLOCK_SIZE = 8
CANDIDATE_TOPK_BLOCKS = 2048
SCORE_WIDTH = 2048
POOL_CAPACITY = SCORE_WIDTH // CANDIDATE_BLOCK_SIZE


def make_candidate_block_pool(
    capacity: int,
    *,
    score_width: int = SCORE_WIDTH,
    topk_blocks: int = CANDIDATE_TOPK_BLOCKS,
    block_size: int = CANDIDATE_BLOCK_SIZE,
):
    """Build the fixed-bucket candidate publisher used by Prefill."""

    if capacity <= 0 or capacity % 16:
        raise ValueError("candidate pool capacity must be a positive multiple of 16")
    if score_width != SCORE_WIDTH or topk_blocks != CANDIDATE_TOPK_BLOCKS:
        raise ValueError("A2/A3 candidate pool uses a fixed 2048-row score bucket")
    if block_size != CANDIDATE_BLOCK_SIZE:
        raise ValueError("A2/A3 candidate blocks contain eight compressed rows")

    @pl.jit
    def kernel(
        scores: pl.Tensor[[capacity, score_width], pl.FP32],
        compressed_lengths: pl.Tensor[[capacity], pl.INT32],
        pool_block_ids: pl.Out[pl.Tensor[[capacity, POOL_CAPACITY], pl.INT32]],
        pool_block_counts: pl.Out[pl.Tensor[[capacity], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(capacity, name_hint="dsv41_a2a3_candidate_pool"):
            token = pl.tile.get_block_idx()
            if token < active:
                raw = pl.load(scores, [token, 0], [1, score_width])
                grouped = pl.reshape(raw, [POOL_CAPACITY, CANDIDATE_BLOCK_SIZE])
                reduce_tmp = pl.tile.create([POOL_CAPACITY, CANDIDATE_BLOCK_SIZE], dtype=pl.FP32)
                maxima = pl.reshape(pl.row_max(grouped, tmp_tile=reduce_tmp), [1, POOL_CAPACITY])
                length = pl.max(pl.read(compressed_lengths, [token]), 0)
                visible_blocks = pl.min(
                    (length + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE,
                    POOL_CAPACITY,
                )
                lanes = pl.tile.arange(0, [1, POOL_CAPACITY], dtype=pl.INT32)
                # Build masks and IDs with vector operations. Scalar tile writes
                # can race the following sort on A3.
                last_block = pl.cast(visible_blocks - 1, pl.INT32)
                invalid = pl.minimum(pl.maximum(pl.sub(lanes, last_block), 0), 1)
                one = pl.tile.full([1, POOL_CAPACITY], dtype=pl.INT32, value=1)
                valid = pl.sub(one, invalid)
                maxima = pl.add(
                    pl.mul(maxima, pl.cast(valid, pl.FP32)), pl.mul(pl.cast(invalid, pl.FP32), -1.0e30)
                )
                delta = pl.sub(lanes, last_block)
                distance = pl.maximum(delta, pl.mul(delta, -1))
                pinned = pl.sub(one, pl.minimum(distance, 1))
                maxima = pl.add(
                    pl.mul(maxima, pl.cast(pl.sub(one, pinned), pl.FP32)),
                    pl.mul(pl.cast(pinned, pl.FP32), 1.0e30),
                )
                logical_ids = pl.sub(lanes, pl.mul(pl.add(lanes, 1), invalid))
                sort_scores = pl.tile.full([1, POOL_CAPACITY], dtype=pl.FP32, value=-1.0e30)
                sort_scores = pl.tile.assemble(sort_scores, maxima, [0, 0])
                sort_ids = pl.tile.full([1, POOL_CAPACITY], dtype=pl.INT32, value=-1)
                sort_ids = pl.tile.assemble(sort_ids, logical_ids, [0, 0])
                sort_tmp = pl.tile.create([1, 2 * POOL_CAPACITY], dtype=pl.FP32)
                unsigned_ids = pl.reinterpret_view(sort_ids, pl.UINT32)
                pairs = pl.tile.sort32(sort_scores, unsigned_ids, tmp=sort_tmp)
                pairs = pl.tile.mrgsort(pairs, block_len=64)
                pairs = pl.tile.mrgsort(pairs, block_len=256)
                leading_pairs = pl.tile.slice(pairs, [1, 2 * POOL_CAPACITY], [0, 0])
                sorted_ids = pl.tile.gather_mask(
                    leading_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                )
                # Native A3 preserves score order, with the newest block first.
                pl.store(sorted_ids, [token, 0], pool_block_ids)
            else:
                empty = pl.tile.full([1, POOL_CAPACITY], dtype=pl.INT32, value=-1)
                pl.store(empty, [token, 0], pool_block_ids)
        # Scalar INT32 stores from one worker per token can share a cache line
        # on A2/A3.  Publish the compact count vector from a single worker.
        with pl.spmd(1, name_hint="dsv41_a2a3_candidate_counts"):
            worker = pl.tile.get_block_idx()
            if worker == 0:
                for token in pl.range(capacity):
                    length = pl.max(pl.read(compressed_lengths, [token]), 0)
                    visible_blocks = pl.min(
                        (length + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE,
                        POOL_CAPACITY,
                    )
                    count = pl.min(visible_blocks, POOL_CAPACITY)
                    if token >= active:
                        count = 0
                    pl.write(pool_block_counts, [token], pl.cast(count, pl.INT32))
        return pool_block_ids, pool_block_counts

    return kernel


def make_candidate_pool_publish(capacity: int, shared_rows: int = 8192):
    """Publish compact candidate blocks into active persistent-state rows."""

    @pl.jit
    def candidate_pool_publish(
        blocks: pl.Tensor[[capacity, POOL_CAPACITY], pl.INT32],
        counts: pl.Tensor[[capacity], pl.INT32],
        shared_blocks: pl.InOut[pl.Tensor[[shared_rows, 1, CANDIDATE_TOPK_BLOCKS], pl.INT32]],
        shared_counts: pl.InOut[pl.Tensor[[shared_rows, 1], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(shared_blocks, [shared_rows, CANDIDATE_TOPK_BLOCKS])
        for section in pl.spmd(
            capacity * (CANDIDATE_TOPK_BLOCKS // POOL_CAPACITY), name_hint="prefill_candidate_blocks_publish"
        ):
            token = section // (CANDIDATE_TOPK_BLOCKS // POOL_CAPACITY)
            column = section % (CANDIDATE_TOPK_BLOCKS // POOL_CAPACITY) * POOL_CAPACITY
            if token < active:
                if column == 0:
                    ids = pl.load(blocks, [token, 0], [1, POOL_CAPACITY])
                    pl.store(ids, [token, 0], flat)
                else:
                    empty = pl.tile.full([1, POOL_CAPACITY], dtype=pl.INT32, value=-1)
                    pl.store(empty, [token, column], flat)
        # The pinned A3 native QLI returns candidates but leaves this persistent
        # length buffer unchanged on its nonempty source path.
        return shared_blocks, shared_counts

    return candidate_pool_publish


DECODE_SCORE_WIDTH = 512
DECODE_POOL_CAPACITY = DECODE_SCORE_WIDTH // CANDIDATE_BLOCK_SIZE


def make_decode_candidate_block_pool(
    capacity: int,
    *,
    score_width: int = DECODE_SCORE_WIDTH,
    topk_blocks: int = CANDIDATE_TOPK_BLOCKS,
    block_size: int = CANDIDATE_BLOCK_SIZE,
):
    """Build the fixed-bucket candidate publisher used by the decoder stages."""

    if capacity <= 0 or capacity % 16:
        raise ValueError("candidate pool capacity must be a positive multiple of 16")
    if score_width != DECODE_SCORE_WIDTH or topk_blocks != CANDIDATE_TOPK_BLOCKS:
        raise ValueError("A2/A3 candidate pool uses a fixed 512-row score bucket")
    if block_size != CANDIDATE_BLOCK_SIZE:
        raise ValueError("A2/A3 candidate blocks contain eight compressed rows")

    @pl.jit
    def kernel(
        scores: pl.Tensor[[capacity, score_width], pl.FP32],
        compressed_lengths: pl.Tensor[[capacity], pl.INT32],
        pool_block_ids: pl.Out[pl.Tensor[[capacity, DECODE_POOL_CAPACITY], pl.INT32]],
        pool_block_counts: pl.Out[pl.Tensor[[capacity], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(capacity, name_hint="dsv41_a2a3_candidate_pool"):
            token = pl.tile.get_block_idx()
            if token < active:
                length = pl.max(pl.read(compressed_lengths, [token]), 0)
                visible_blocks = pl.min(
                    (length + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE,
                    DECODE_POOL_CAPACITY,
                )
                # Candidate publication retains every visible causal block.  A
                # score sort is unnecessary here because the pool capacity is
                # the full visible prefix; sorting the mixed FP32/UINT32 pair
                # on A2/A3 also reinterprets the score sentinel as a block ID.
                ordered_ids = pl.tile.full([1, DECODE_POOL_CAPACITY], dtype=pl.INT32, value=-1)
                for lane in pl.range(DECODE_POOL_CAPACITY):
                    if lane < visible_blocks:
                        pl.tile.write(ordered_ids, [0, lane], pl.cast(lane, pl.INT32))
                pl.store(ordered_ids, [token, 0], pool_block_ids)
            else:
                empty = pl.tile.full([1, DECODE_POOL_CAPACITY], dtype=pl.INT32, value=-1)
                pl.store(empty, [token, 0], pool_block_ids)
        # Scalar INT32 stores from one worker per token can share a cache line
        # on A2/A3.  Publish the compact count vector from a single worker.
        with pl.spmd(1, name_hint="dsv41_a2a3_candidate_counts"):
            worker = pl.tile.get_block_idx()
            if worker == 0:
                for token in pl.range(capacity):
                    length = pl.max(pl.read(compressed_lengths, [token]), 0)
                    visible_blocks = pl.min(
                        (length + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE,
                        DECODE_POOL_CAPACITY,
                    )
                    count = pl.min(visible_blocks, DECODE_POOL_CAPACITY)
                    if token >= active:
                        count = 0
                    pl.write(pool_block_counts, [token], pl.cast(count, pl.INT32))
        return pool_block_ids, pool_block_counts

    return kernel


INDEX_TOPK = 512
KEY_CAPACITY = 512
KEY_TILE = 32
HEAD_TILE = 16


def make_indexer_select(
    capacity: int,
    key_capacity: int,
    *,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
    topk: int = INDEX_TOPK,
    compress_ratio: int = 2,
):
    """Build a fixed-bucket PyPTO QLI score/Top-K kernel."""

    if capacity <= 0 or capacity % HEAD_TILE:
        raise ValueError("capacity must be a positive multiple of 16")
    if heads != INDEX_HEADS or head_dim != INDEX_HEAD_DIM:
        raise ValueError("A2/A3 QLI uses 32 heads x 128")
    # The native A2/A3 QLI handoff allocates one 512-entry candidate row.
    # Smaller visible prefixes are represented by score padding and are
    # filtered by ``prepare_indexer_indices`` in the next boundary.  The
    # fixed signature is deliberate: larger or variable rows are not stable
    # on the current A2/A3 sort32/mrgsort backend.
    if key_capacity != KEY_CAPACITY:
        raise ValueError(f"A2/A3 QLI uses fixed key_capacity={KEY_CAPACITY}")
    if topk != INDEX_TOPK:
        raise ValueError(f"A2/A3 QLI uses fixed topk={INDEX_TOPK}")
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")

    @pl.jit.incore
    def select_one(
        query: pl.Tensor[[capacity, heads, head_dim], pl.INT8],
        query_scale: pl.Tensor[[capacity, heads], pl.FP16],
        key: pl.Tensor[[key_capacity, head_dim], pl.INT8],
        key_scale: pl.Tensor[[key_capacity], pl.FP16],
        weights: pl.Tensor[[capacity, heads], pl.FP16],
        positions: pl.Tensor[[capacity], pl.INT32],
        running_pairs: pl.Tensor[[capacity, 2 * topk], pl.FP32],
        scores: pl.Tensor[[capacity, key_capacity], pl.FP32],
        indices: pl.Tensor[[capacity, topk], pl.INT32],
    ) -> None:
        token = pl.tile.get_block_idx()
        # The A2/A3 QLI cache ABI is one fixed 512-entry candidate row.  Sort
        # the complete row here; ``prepare_indexer_indices`` applies the
        # request-specific visible-prefix filter before cache gather.  Keeping
        # this boundary fixed avoids dynamic tile tail writes on A2/A3 while
        # preserving the final selected set.
        visible = key_capacity
        # QLI writes a score tile for every visible key.  Invalid rows never
        # enter the sort, and score padding stays at the hardware minimum.
        # Materialize that floor explicitly: dynamic ``valid_shape`` on a
        # sort32 input is not reliable on the A2/A3 backend for partial rows.
        pl.store(
            pl.tile.full([1, key_capacity], dtype=pl.FP32, value=NEGATIVE_INF),
            [token, 0],
            scores,
        )
        for k0 in pl.range(0, visible, KEY_TILE):
            valid = pl.min(KEY_TILE, visible - k0)
            key_tile = key[k0 : k0 + KEY_TILE, :]
            scale_tile = pl.reshape(pl.cast(key_scale[k0 : k0 + KEY_TILE], pl.FP32), [1, KEY_TILE])
            total = pl.full([1, KEY_TILE], dtype=pl.FP32, value=0.0)
            for h0 in pl.range(0, heads, HEAD_TILE):
                q_tile = pl.reshape(query[token : token + 1, h0 : h0 + HEAD_TILE, :], [HEAD_TILE, head_dim])
                dots = pl.matmul(q_tile, key_tile, b_trans=True, out_dtype=pl.INT32)
                dots = pl.cast(dots, pl.FP32, mode="none")
                q_scale_tile = pl.reshape(
                    pl.cast(query_scale[token, h0 : h0 + HEAD_TILE], pl.FP32), [HEAD_TILE, 1]
                )
                weight_tile = pl.reshape(
                    pl.cast(weights[token, h0 : h0 + HEAD_TILE], pl.FP32), [HEAD_TILE, 1]
                )
                dots = pl.row_expand_mul(dots, q_scale_tile)
                dots = pl.col_expand_mul(dots, scale_tile)
                dots = pl.maximum(dots, 0.0)
                dots = pl.row_expand_mul(dots, weight_tile)
                total = pl.add(total, pl.reshape(pl.col_sum(dots), [1, KEY_TILE]))
            scores[token : token + 1, k0 : k0 + KEY_TILE] = pl.fillpad(
                pl.set_validshape(total, 1, valid), pad_value=pl.PadValue.min
            )

        # Match QLI's score-order result with a bounded merge sort.  The
        # payload carries absolute key positions, so no host-side offset is
        # needed for the fixed flattened cache used by the A2/A3 path.
        # The A2/A3 ABI uses one 512-entry QLI candidate row.  Always execute
        # the fixed sort once and represent a short visible prefix with the
        # explicit score floor above; a dynamic ``pl.range(leaf_count)`` leaves
        # the pair arena uninitialized for partial prefixes on A2/A3.
        leaf0 = 0
        row_raw = pl.load(scores, [token, leaf0], [1, SORT_LEAF])
        row = pl.tile.fillpad(row_raw, pad_value=pl.PadValue.min)
        row = pl.maximum(row, pl.tile.full([1, SORT_LEAF], dtype=pl.FP32, value=NEGATIVE_INF))
        ramp = pl.tile.arange(0, [1, SORT_LEAF], dtype=pl.INT32)
        leaf_indices = pl.tile.add(ramp, pl.cast(leaf0, pl.INT32))
        sort_tmp = pl.tile.create([1, 2 * SORT_LEAF], dtype=pl.FP32)
        leaf_pairs = pl.tile.sort32(row, pl.reinterpret_view(leaf_indices, pl.UINT32), tmp=sort_tmp)
        leaf_pairs = pl.tile.mrgsort(leaf_pairs, block_len=64)
        leaf_pairs = pl.tile.mrgsort(leaf_pairs, block_len=256)
        leaf_pairs = pl.tile.slice(leaf_pairs, [1, 2 * topk], [0, 0])
        pl.store(leaf_pairs, [token, 0], running_pairs)

        # Materialize the complete padding row in one UB tile.  Explicitly
        # initialize every lane because the backend may preserve stale pair
        # lanes when only a dynamic prefix is written.
        selected = pl.tile.full([1, topk], dtype=pl.INT32, value=-1)
        if visible > 0:
            pairs = pl.load(running_pairs, [token, 0], [1, 2 * topk])
            picked = pl.tile.gather_mask(pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            selected = picked
        pl.store(selected, [token, 0], indices)

    @pl.jit
    def kernel(
        query: pl.Tensor[[capacity, heads, head_dim], pl.INT8],
        query_scale: pl.Tensor[[capacity, heads], pl.FP16],
        key: pl.Tensor[[key_capacity, head_dim], pl.INT8],
        key_scale: pl.Tensor[[key_capacity], pl.FP16],
        weights: pl.Tensor[[capacity, heads], pl.FP16],
        positions: pl.Tensor[[capacity], pl.INT32],
        scores: pl.Out[pl.Tensor[[capacity, key_capacity], pl.FP32]],
        indices: pl.Out[pl.Tensor[[capacity, topk], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        running_pairs = pl.create_tensor([capacity, 2 * topk], dtype=pl.FP32)
        with pl.spmd(active, name_hint="dsv41_a2a3_qli_select"):
            select_one(
                query,
                query_scale,
                key,
                key_scale,
                weights,
                positions,
                running_pairs,
                scores,
                indices,
            )
        return scores, indices

    return kernel


def make_indexer_select_restricted(
    capacity: int,
    key_capacity: int,
    *,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
    topk: int = INDEX_TOPK,
    compress_ratio: int = 1,
    candidate_block_size: int = 8,
    candidate_block_capacity: int = 64,
):
    """Select C1A QLI rows inside the source candidate block pool."""

    if capacity <= 0 or capacity % HEAD_TILE:
        raise ValueError("capacity must be a positive multiple of 16")
    if heads != INDEX_HEADS or head_dim != INDEX_HEAD_DIM:
        raise ValueError("A2/A3 QLI uses 32 heads x 128")
    if key_capacity != KEY_CAPACITY or topk != INDEX_TOPK:
        raise ValueError("A2/A3 restricted QLI uses fixed 512 rows and topk")
    if compress_ratio != 1 or candidate_block_size != 8:
        raise ValueError("C1A reindex uses ratio-1 eight-row candidate blocks")
    if candidate_block_capacity <= 0:
        raise ValueError("candidate block capacity must be positive")

    @pl.jit.incore
    def select_one(
        query: pl.Tensor[[capacity, heads, head_dim], pl.INT8],
        query_scale: pl.Tensor[[capacity, heads], pl.FP16],
        key: pl.Tensor[[key_capacity, head_dim], pl.INT8],
        key_scale: pl.Tensor[[key_capacity], pl.FP16],
        weights: pl.Tensor[[capacity, heads], pl.FP16],
        positions: pl.Tensor[[capacity], pl.INT32],
        candidate_blocks: pl.Tensor[[capacity, candidate_block_capacity], pl.INT32],
        candidate_counts: pl.Tensor[[capacity], pl.INT32],
        running_pairs: pl.Tensor[[capacity, 2 * topk], pl.FP32],
        scores: pl.Tensor[[capacity, key_capacity], pl.FP32],
        indices: pl.Tensor[[capacity, topk], pl.INT32],
    ) -> None:
        token = pl.tile.get_block_idx()
        pl.store(
            pl.tile.full([1, key_capacity], dtype=pl.FP32, value=NEGATIVE_INF),
            [token, 0],
            scores,
        )
        for k0 in pl.range(0, key_capacity, KEY_TILE):
            key_tile = key[k0 : k0 + KEY_TILE, :]
            scale_tile = pl.reshape(pl.cast(key_scale[k0 : k0 + KEY_TILE], pl.FP32), [1, KEY_TILE])
            total = pl.full([1, KEY_TILE], dtype=pl.FP32, value=0.0)
            for h0 in pl.range(0, heads, HEAD_TILE):
                q_tile = pl.reshape(query[token : token + 1, h0 : h0 + HEAD_TILE, :], [HEAD_TILE, head_dim])
                dots = pl.matmul(q_tile, key_tile, b_trans=True, out_dtype=pl.INT32)
                dots = pl.cast(dots, pl.FP32, mode="none")
                q_scale_tile = pl.reshape(
                    pl.cast(query_scale[token, h0 : h0 + HEAD_TILE], pl.FP32), [HEAD_TILE, 1]
                )
                weight_tile = pl.reshape(
                    pl.cast(weights[token, h0 : h0 + HEAD_TILE], pl.FP32), [HEAD_TILE, 1]
                )
                dots = pl.row_expand_mul(dots, q_scale_tile)
                dots = pl.col_expand_mul(dots, scale_tile)
                dots = pl.maximum(dots, 0.0)
                dots = pl.row_expand_mul(dots, weight_tile)
                total = pl.add(total, pl.reshape(pl.col_sum(dots), [1, KEY_TILE]))
            visible = pl.min(pl.read(positions, [token]) + 1, key_capacity)
            for lane in pl.range(KEY_TILE):
                key_index = k0 + lane
                block_id = key_index // candidate_block_size
                allowed = False
                for candidate in pl.range(candidate_block_capacity):
                    if candidate < pl.read(candidate_counts, [token]):
                        if pl.read(candidate_blocks, [token, candidate]) == block_id:
                            allowed = True
                if key_index >= visible or not allowed:
                    pl.write(total, [0, lane], NEGATIVE_INF)
            scores[token : token + 1, k0 : k0 + KEY_TILE] = total

        row = pl.load(scores, [token, 0], [1, SORT_LEAF])
        row = pl.tile.fillpad(row, pad_value=pl.PadValue.min)
        ramp = pl.tile.arange(0, [1, SORT_LEAF], dtype=pl.INT32)
        leaf_pairs = pl.tile.sort32(
            row,
            pl.reinterpret_view(ramp, pl.UINT32),
            tmp=pl.tile.create([1, 2 * SORT_LEAF], dtype=pl.FP32),
        )
        leaf_pairs = pl.tile.mrgsort(leaf_pairs, block_len=64)
        leaf_pairs = pl.tile.mrgsort(leaf_pairs, block_len=256)
        leaf_pairs = pl.tile.slice(leaf_pairs, [1, 2 * topk], [0, 0])
        pl.store(leaf_pairs, [token, 0], running_pairs)
        pairs = pl.load(running_pairs, [token, 0], [1, 2 * topk])
        picked = pl.tile.gather_mask(pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        for lane in pl.range(topk):
            score = pl.read(running_pairs, [token, 2 * lane])
            if score <= NEGATIVE_INF / 2:
                pl.tile.write(picked, [0, lane], pl.cast(-1, pl.INT32))
        pl.store(picked, [token, 0], indices)

    @pl.jit
    def kernel(
        query: pl.Tensor[[capacity, heads, head_dim], pl.INT8],
        query_scale: pl.Tensor[[capacity, heads], pl.FP16],
        key: pl.Tensor[[key_capacity, head_dim], pl.INT8],
        key_scale: pl.Tensor[[key_capacity], pl.FP16],
        weights: pl.Tensor[[capacity, heads], pl.FP16],
        positions: pl.Tensor[[capacity], pl.INT32],
        candidate_blocks: pl.Tensor[[capacity, candidate_block_capacity], pl.INT32],
        candidate_counts: pl.Tensor[[capacity], pl.INT32],
        scores: pl.Out[pl.Tensor[[capacity, key_capacity], pl.FP32]],
        indices: pl.Out[pl.Tensor[[capacity, topk], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        running_pairs = pl.create_tensor([capacity, 2 * topk], dtype=pl.FP32)
        with pl.spmd(active, name_hint="dsv41_a2a3_qli_select_restricted"):
            select_one(
                query,
                query_scale,
                key,
                key_scale,
                weights,
                positions,
                candidate_blocks,
                candidate_counts,
                running_pairs,
                scores,
                indices,
            )
        return scores, indices

    return kernel


def prepare_indexer_indices_reference(
    selected: torch.Tensor,
    positions: torch.Tensor,
    *,
    compress_ratio: int = 2,
) -> dict[str, torch.Tensor]:
    """CPU reference for vLLM's visible-prefix filter and chronological sort."""

    if selected.ndim != 2 or selected.dtype != torch.int32:
        raise ValueError("selected must be a 2-D INT32 tensor")
    if positions.shape != (selected.shape[0],) or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must be a 1-D INT32/INT64 tensor")
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")
    tokens, topk = selected.shape
    visible = ((positions.to(torch.int64) + 1) // compress_ratio).clamp(min=0, max=topk)
    valid = selected.ge(0) & selected.lt(visible[:, None])
    sentinel = torch.iinfo(torch.int32).max
    sortable = torch.where(valid, selected, torch.full_like(selected, sentinel))
    ordered = sortable.sort(dim=-1, stable=True).values
    output = torch.where(ordered.eq(sentinel), torch.full_like(ordered, -1), ordered)
    lengths = valid.sum(dim=-1, keepdim=True, dtype=torch.int32)
    return {"indices": output, "lengths": lengths}


def quantize_indexer_query_reference(
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU oracle for vLLM's A2/A3 ``quantize_indexer_query``.

    The scale is rounded to FP16 before it is used to form the INT8 payload;
    this differs from the index-cache writer, whose payload uses the raw FP32
    scale and publishes an FP16 copy.
    """

    if query.ndim != 3 or query.shape[-1] != INDEX_HEAD_DIM:
        raise ValueError("indexer query must have shape [tokens, heads, 128]")
    source = query.detach().float()
    scale = (source.abs().amax(dim=-1) / 127.0).to(torch.float16)
    scale = torch.maximum(scale, torch.full_like(scale, INDEX_SCALE_MIN, dtype=torch.float16))
    payload = torch.round(source / scale.float().unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return payload, scale


def indexer_query_reference(
    q_lora: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    *,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
    rope_dim: int = INDEX_ROPE_DIM,
) -> dict[str, torch.Tensor]:
    """CPU reference for the complete L3 indexer query boundary."""

    # The PyPTO ABI stores canonical [K, N], while the Torch helper consumes
    # output-major [N, K] weights.
    projected = w8a8_linear(q_lora, weight.transpose(0, 1).contiguous(), weight_scale)
    query = rope_reference(projected, cosine, sine, heads, head_dim, rope_dim)
    query = query.reshape(query.shape[0], heads, head_dim)
    payload, scales = quantize_indexer_query_reference(query)
    return {"query": query, "payload": payload, "scales": scales}


def decode_quantize_indexer_query_reference(
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU oracle for vLLM's A2/A3 ``quantize_indexer_query``.

    The scale is rounded to FP16 before it is used to form the INT8 payload;
    this differs from the index-cache writer, whose payload uses the raw FP32
    scale and publishes an FP16 copy.
    """

    if query.ndim != 3 or query.shape[-1] != INDEX_HEAD_DIM:
        raise ValueError("indexer query must have shape [tokens, heads, 128]")
    source = query.detach().float()
    scale = (source.abs().amax(dim=-1) / 127.0).to(torch.float16)
    scale = torch.maximum(scale, torch.full_like(scale, DECODE_INDEX_SCALE_MIN, dtype=torch.float16))
    payload = torch.round(source / scale.float().unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return payload, scale


def decode_indexer_query_reference(
    q_lora: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    *,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
    rope_dim: int = INDEX_ROPE_DIM,
) -> dict[str, torch.Tensor]:
    """CPU reference for the complete L3 indexer query boundary."""

    # The PyPTO ABI stores canonical [K, N], while the Torch helper consumes
    # output-major [N, K] weights.
    projected = w8a8_linear(q_lora, weight.transpose(0, 1).contiguous(), weight_scale)
    query = decode_rope_reference(projected, cosine, sine, heads, head_dim, rope_dim)
    query = query.reshape(query.shape[0], heads, head_dim)
    payload, scales = decode_quantize_indexer_query_reference(query)
    return {"query": query, "payload": payload, "scales": scales}


def golden_candidate_block_pool(args):
    """Order visible logical blocks by maximum score, with the newest block first."""
    import torch

    args["pool_block_ids"].fill_(-1)
    args["pool_block_counts"].zero_()
    maxima = args["scores"].reshape(-1, POOL_CAPACITY, CANDIDATE_BLOCK_SIZE).amax(dim=-1)
    for token in range(int(args["active"])):
        count = min(POOL_CAPACITY, max(0, (int(args["compressed_lengths"][token]) + 7) // 8))
        if count:
            ordered = torch.argsort(maxima[token, : count - 1], descending=True)
            ids = torch.cat((torch.tensor([count - 1]), ordered)).int()
            args["pool_block_ids"][token, :count] = ids
            args["pool_block_counts"][token] = count


def golden_candidate_pool_publish(args):
    """Replace active candidate rows and preserve persistent count bytes and inactive rows."""
    active = int(args["active"])
    args["shared_blocks"][:active].fill_(-1)
    args["shared_blocks"][:active, 0, :POOL_CAPACITY] = args["blocks"][:active]


def indexer_select_reference(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    *,
    topk: int = INDEX_TOPK,
    compress_ratio: int = 2,
) -> dict[str, torch.Tensor]:
    """CPU oracle for the A2/A3 QLI score and score-order Top-K contract.

    ``query`` is ``[T, H, 128]`` INT8, ``key`` is ``[K, 128]`` INT8, and the
    two scale tensors are FP16.  ``positions`` is the raw token position used
    by vLLM to derive the visible compressed-key prefix.
    """

    if query.ndim != 3 or tuple(query.shape[1:]) != (INDEX_HEADS, INDEX_HEAD_DIM):
        raise ValueError("query must have shape [T, 32, 128]")
    if key.ndim != 2 or key.shape[-1] != INDEX_HEAD_DIM:
        raise ValueError("key must have shape [K, 128]")
    tokens, heads, _ = query.shape
    key_rows = key.shape[0]
    if query_scale.shape != (tokens, heads) or weights.shape != (tokens, heads):
        raise ValueError("query_scale and weights must have shape [T, 32]")
    if key_scale.numel() != key_rows or positions.shape != (tokens,):
        raise ValueError("key_scale must be [K] and positions must be [T]")
    if not 1 <= topk <= key_rows:
        raise ValueError("topk must be in [1, key.shape[0]]")
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")

    dots = torch.einsum("thd,kd->thk", query.to(torch.int32), key.to(torch.int32)).float()
    dots.mul_(query_scale.float().unsqueeze(-1))
    dots.mul_(key_scale.float().reshape(1, 1, key_rows))
    scores = torch.relu(dots)
    scores.mul_(weights.float().unsqueeze(-1))
    scores = scores.sum(dim=1)

    indices = torch.full((tokens, topk), -1, dtype=torch.int32)
    visible = ((positions.to(torch.int64) + 1) // compress_ratio).clamp(min=0, max=key_rows)
    for token in range(tokens):
        count = min(int(visible[token]), topk)
        if count:
            # QLI returns score order.  ``sorted=False`` matches the native
            # operator's contract; ties are not expected for random fixtures.
            picked = torch.topk(scores[token, : int(visible[token])], count, sorted=False).indices
            indices[token, :count] = picked.to(torch.int32)
    return {"scores": scores, "indices": indices}


def validate(*, device=0, sim=False, compile_only=False, cpu_only=False):
    """Check empty rows, partial blocks, 512/1024 boundaries and the 2048-row limit."""
    import torch
    from golden import ScalarSpec, TensorSpec, run

    if cpu_only and compile_only:
        raise ValueError("CPU golden checks and kernel compilation are separate modes")
    capacity, shared_rows = 16, 32
    lengths = torch.tensor([0, 1, 7, 8, 9, 511, 512, 513, 1023, 1024, 1025, 2040, 2041, 2047, 2048, -1]).int()
    block_scores = (torch.arange(POOL_CAPACITY) * 73) % POOL_CAPACITY
    scores = (block_scores[:, None] + torch.arange(8)[None, :] / 16).flatten().repeat(capacity, 1)
    records = []
    for active in (0, 1, 7, capacity):
        pool_specs = [
            TensorSpec("scores", [capacity, SCORE_WIDTH], torch.float32, init_value=scores),
            TensorSpec("compressed_lengths", [capacity], torch.int32, init_value=lengths),
            TensorSpec("pool_block_ids", [capacity, POOL_CAPACITY], torch.int32, init_value=-53),
            TensorSpec("pool_block_counts", [capacity], torch.int32, init_value=-59),
            ScalarSpec("active", torch.int32, active, compile_runtime=True),
        ]
        pool = {
            spec.name: spec.create_tensor() if hasattr(spec, "create_tensor") else spec.to_python()
            for spec in pool_specs
        }
        golden_candidate_block_pool(pool)
        for token in range(capacity):
            count = int(pool["pool_block_counts"][token])
            expected = min(POOL_CAPACITY, max(0, (int(lengths[token]) + 7) // 8)) if token < active else 0
            ids = pool["pool_block_ids"][token]
            if (
                count != expected
                or set(ids[:count].tolist()) != set(range(count))
                or (ids[count:] != -1).any()
            ):
                raise RuntimeError("Candidate pool lost a visible block or published an invalid block")
            if count and int(ids[0]) != count - 1:
                raise RuntimeError("Candidate pool did not pin the newest block")
        publish_specs = [
            TensorSpec("blocks", [capacity, POOL_CAPACITY], torch.int32, init_value=pool["pool_block_ids"]),
            TensorSpec("counts", [capacity], torch.int32, init_value=pool["pool_block_counts"]),
            TensorSpec("shared_blocks", [shared_rows, 1, CANDIDATE_TOPK_BLOCKS], torch.int32, init_value=-67),
            TensorSpec("shared_counts", [shared_rows, 1], torch.int32, init_value=-71),
            ScalarSpec("active", torch.int32, active, compile_runtime=True),
        ]
        cases = [
            ("pool", pool_specs, golden_candidate_block_pool, make_candidate_block_pool),
            ("publish", publish_specs, golden_candidate_pool_publish, make_candidate_pool_publish),
        ]
        for name, specs, golden_fn, builder in cases:
            if cpu_only:
                values = {
                    spec.name: spec.create_tensor() if hasattr(spec, "create_tensor") else spec.to_python()
                    for spec in specs
                }
                golden_fn(values)
                if name == "publish":
                    if (
                        (values["shared_blocks"][active:] != -67).any()
                        or (values["shared_counts"] != -71).any()
                        or (values["shared_blocks"][:active, :, POOL_CAPACITY:] != -1).any()
                    ):
                        raise RuntimeError(
                            "Candidate publication changed readonly bytes or retained stale tail IDs"
                        )
            else:
                fn = builder(capacity) if name == "pool" else builder(capacity, shared_rows)
                result = run(
                    fn=fn,
                    specs=specs,
                    golden_fn=golden_fn,
                    config=dict(backend_type="a2a3", execution_mode=bool(sim), device_id=device),
                    compile_only=compile_only,
                    rtol=0,
                    atol=0,
                )
                if not result.passed:
                    raise RuntimeError(f"candidate-{name}-active{active}: {result.error}")
            records.append(
                dict(
                    case=f"{name}-active{active}",
                    cpu_golden="passed" if cpu_only else "not_run",
                    kernel="not_run" if cpu_only or compile_only else "passed",
                    compiled=not cpu_only,
                )
            )
    return records


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3"], default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    result = validate(
        device=args.device, sim=args.sim, compile_only=args.compile_only, cpu_only=args.cpu_only
    )
    print(json.dumps(result))

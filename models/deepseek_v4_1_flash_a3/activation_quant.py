# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Per-token W8A8 quantization for A3 projections and experts."""

import inspect

import pypto.language as pl


K_TILE = 256
VEC_M_TILE = 8
PREFILL_QUANT_WORKERS = 48
QUANT_REDUCE_TILE = 1024
CHECKPOINT_ROWS = pl.dynamic("CHECKPOINT_QUANT_ROWS")
_TILE_RSQRT_HAS_HIGH_PRECISION = "high_precision" in inspect.signature(pl.tile.rsqrt).parameters

if _TILE_RSQRT_HAS_HIGH_PRECISION:
    @pl.jit.inline(auto_scope=False)
    def _tile_rsqrt_for_norm(
        value: pl.Tile[[VEC_M_TILE, 1], pl.FP32],
        scratch: pl.Tile[[VEC_M_TILE, 1], pl.FP32],
    ):
        return pl.tile.rsqrt(value, tmp=scratch, high_precision=True)
else:
    @pl.jit.inline(auto_scope=False)
    def _tile_rsqrt_for_norm(
        value: pl.Tile[[VEC_M_TILE, 1], pl.FP32],
        scratch: pl.Tile[[VEC_M_TILE, 1], pl.FP32],
    ):
        return pl.tile.rsqrt(value, tmp=scratch)


@pl.jit.inline(auto_scope=False)
def _checkpoint_codes(
    value: pl.Tile[[CHECKPOINT_ROWS, K_TILE], pl.FP32],
    denominator: pl.Tile[[CHECKPOINT_ROWS, 1], pl.FP32],
    inverse: pl.Tile[[CHECKPOINT_ROWS, 1], pl.FP32],
):
    """Resolve signed half-way cases against the original BF16 row peak."""
    magnitude = pl.abs(value)
    # Scale both sides of the midpoint comparison by a power of two to keep finite BF16 inputs in range.
    scaled_magnitude = pl.mul(magnitude, 127.0 / 128.0)
    estimate = pl.row_expand_mul(scaled_magnitude, inverse)
    floor_int = pl.cast(pl.minimum(estimate, 127.0), pl.INT32, mode="floor")
    lower = pl.cast(floor_int, pl.FP32)
    midpoint = pl.row_expand_mul(pl.add(lower, 0.5), denominator)
    upper = pl.add(lower, 1.0)
    nearest = pl.tile.select(pl.tile.cmp(scaled_magnitude, midpoint, cmp_type=4), upper, lower)
    odd = pl.cast(pl.ands(pl.cast(lower, pl.INT16), 1), pl.FP32)
    nearest = pl.tile.select(pl.tile.cmp(scaled_magnitude, midpoint, cmp_type=0), pl.add(lower, odd), nearest)
    signed = pl.tile.select(pl.tile.cmps(value, 0.0, cmp_type=2), pl.neg(nearest), nearest)
    encoded = pl.cast(signed, pl.INT8, mode="rint")
    return encoded


@pl.jit.inline(auto_scope=False)
def _encode_checkpoint_panel(
    source: pl.Tensor, payload: pl.Tensor, row: pl.Scalar[pl.INDEX],
    k0: pl.Scalar[pl.INDEX], rows: pl.Scalar[pl.INDEX],
    denominator: pl.Tile[[8, 1], pl.FP32], inverse: pl.Tile[[8, 1], pl.FP32],
):
    raw = pl.load(source, [row, k0], [8, 256], valid_shape=[rows, 256])
    padded = pl.set_validshape(pl.fillpad(raw, pad_value=pl.PadValue.zero), 8, 256)
    value = pl.cast(padded, pl.FP32)
    encoded = _checkpoint_codes(value, denominator, inverse)
    pl.store(encoded, [row, k0], payload)


@pl.jit.inline(auto_scope=False)
def _quantize_cached_rows(
    source: pl.Tensor, payload: pl.Tensor, scales: pl.Tensor, active: pl.Scalar[pl.INT32],
    capacity: pl.Scalar[pl.INDEX], width: pl.constexpr, checkpoint: pl.constexpr,
):
    """Keep complete input rows in UB for the peak and midpoint passes."""
    blocks = capacity // VEC_M_TILE
    if not checkpoint:
        blocks = (active + VEC_M_TILE - 1) // VEC_M_TILE
    workers = pl.min(blocks, PREFILL_QUANT_WORKERS)
    for worker in pl.spmd(workers, name_hint="prefill_token_quantize_whole_row"):
        start = worker * blocks // workers
        end = (worker + 1) * blocks // workers
        for block in pl.pipeline(start, end, stage=2):
            row = block * VEC_M_TILE
            rows = pl.max(0, pl.min(VEC_M_TILE, active - row))
            raw = pl.load(source, [row, 0], [VEC_M_TILE, width], valid_shape=[rows, width])
            cached = pl.set_validshape(pl.fillpad(raw, pad_value=pl.PadValue.zero), VEC_M_TILE, width)
            fp = pl.cast(cached, pl.FP32)
            scratch = pl.create_tile([VEC_M_TILE, width], dtype=pl.FP32)
            maximum = pl.row_max(pl.abs(fp), tmp_tile=scratch)
            if checkpoint:
                factor = pl.maximum(pl.mul(maximum, 1.0 / 127.0), 1e-30)
                pl.store(factor, [row, 0], scales)
                denominator = pl.mul(pl.maximum(maximum, 127e-30), 1.0 / 128.0)
                inverse = pl.recip(denominator)
                for k0 in pl.range(0, width, 256):
                    part = pl.tile.slice(fp, [VEC_M_TILE, 256], [0, k0])
                    encoded = _checkpoint_codes(part, denominator, inverse)
                    pl.store(encoded, [row, k0], payload)
            else:
                row_peak = pl.reshape(maximum, [1, VEC_M_TILE])
                numerator = pl.tile.full([1, VEC_M_TILE], dtype=pl.FP32, value=127.0)
                inverse = pl.reshape(pl.div(numerator, pl.maximum(row_peak, 1e-30)), [VEC_M_TILE, 1])
                scale = pl.mul(maximum, 1.0 / 127.0)
                pl.store(pl.set_validshape(scale, rows, 1), [row, 0], scales)
                rounded = pl.cast(pl.row_expand_mul(fp, inverse), pl.INT32, mode="rint")
                codes = pl.cast(rounded, pl.INT8, mode="trunc")
                pl.store(pl.set_validshape(codes, rows, width), [row, 0], payload)
    return payload, scales


@pl.jit.inline(auto_scope=False)
def quantize_projection_activation(
    source: pl.Tensor, num_tokens: pl.Scalar[pl.INT32], width: pl.constexpr, dense_query: pl.constexpr,
):
    """Quantize the resident A3 checkpoint's padded rows with ties-to-even rounding."""
    capacity = pl.tensor.dim(source, 0)
    padded_rows = (capacity + 15) // 16 * 16
    payload = pl.create_tensor([padded_rows, width], dtype=pl.INT8)
    scale = pl.create_tensor([padded_rows, 1], dtype=pl.FP32)
    if padded_rows <= 32:
        panels = (width + QUANT_REDUCE_TILE - 1) // QUANT_REDUCE_TILE
        partials = pl.create_tensor([padded_rows // 8 * panels, 8], dtype=pl.FP32)
        with pl.spmd(padded_rows // 8 * panels, name_hint="quant_absmax") as partials_ready:
            shard = pl.tile.get_block_idx()
            shard_row = shard // panels * 8
            shard_col = shard % panels * QUANT_REDUCE_TILE
            shard_rows = pl.max(0, pl.min(8, num_tokens - shard_row))
            shard_cols = pl.min(QUANT_REDUCE_TILE, width - shard_col)
            raw = pl.tile.load(source, [shard_row, shard_col], [8, QUANT_REDUCE_TILE], valid_shape=[shard_rows, shard_cols])
            padded = pl.set_validshape(pl.fillpad(raw, pad_value=pl.PadValue.zero), 8, QUANT_REDUCE_TILE)
            absolute = pl.abs(pl.cast(padded, pl.FP32))
            scratch = pl.create_tile([8, QUANT_REDUCE_TILE], dtype=pl.FP32)
            shard_max = pl.reshape(pl.row_max(absolute, tmp_tile=scratch), [1, 8])
            pl.tile.store(shard_max, [shard, 0], partials)
        tiles = padded_rows // 8 * (width // 256)
        workers = pl.min(tiles, 2 * pl.system.available_cluster_count())
        with pl.spmd(workers, name_hint="quant_encode", deps=[partials_ready]):
            worker = pl.tile.get_block_idx()
            for block in pl.range(worker, tiles, workers):
                group = block // (width // 256)
                row = group * 8
                k0 = block % (width // 256) * 256
                rows = pl.max(0, pl.min(8, num_tokens - row))
                maximum = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
                for panel in pl.range(panels):
                    local = pl.load(partials, [group * panels + panel, 0], [1, 8])
                    maximum = pl.maximum(maximum, local)
                if k0 == 0:
                    factor = pl.reshape(pl.maximum(pl.mul(maximum, 1.0 / 127.0), 1e-30), [8, 1])
                    pl.store(factor, [row, 0], scale)
                denominator = pl.reshape(pl.mul(pl.maximum(maximum, 127e-30), 1.0 / 128.0), [8, 1])
                inverse = pl.recip(denominator)
                _encode_checkpoint_panel(source, payload, row, k0, rows, denominator, inverse)
    elif dense_query and width == 1280:
        _quantize_cached_rows(source, payload, scale, num_tokens, padded_rows, width, True)
    else:
        workers = pl.max(1, pl.min(2 * pl.system.available_cluster_count(), padded_rows // 8))
        for worker in pl.spmd(workers, name_hint="dynamic_int8_quant"):
            for row in pl.range(worker * 8, padded_rows, workers * 8):
                rows = pl.max(0, pl.min(8, num_tokens - row))
                maximum = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
                for k0 in pl.range(0, width, QUANT_REDUCE_TILE):
                    cols = pl.min(QUANT_REDUCE_TILE, width - k0)
                    reduce_raw = pl.load(source, [row, k0], [8, QUANT_REDUCE_TILE], valid_shape=[rows, cols])
                    reduce_padded = pl.set_validshape(pl.fillpad(reduce_raw, pad_value=pl.PadValue.zero), 8, QUANT_REDUCE_TILE)
                    reduce_value = pl.cast(reduce_padded, pl.FP32)
                    tmp = pl.create_tile([8, QUANT_REDUCE_TILE], dtype=pl.FP32)
                    local_maximum = pl.reshape(pl.row_max(pl.abs(reduce_value), tmp_tile=tmp), [1, 8])
                    maximum = pl.maximum(maximum, local_maximum)
                factor = pl.reshape(pl.maximum(pl.mul(maximum, 1.0 / 127.0), 1e-30), [8, 1])
                scale = pl.store(factor, [row, 0], scale)
                denominator = pl.reshape(pl.mul(pl.maximum(maximum, 127e-30), 1.0 / 128.0), [8, 1])
                inverse = pl.recip(denominator)
                for k0 in pl.range(0, width, 256):
                    _encode_checkpoint_panel(source, payload, row, k0, rows, denominator, inverse)
    return payload, scale


@pl.jit.inline(auto_scope=False)
def normalize_quantize_projection_activation(
    source: pl.Tensor,
    norm_weight: pl.Tensor,
    normalized: pl.Tensor,
    payload: pl.Tensor,
    scale: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.constexpr,
    eps: pl.constexpr,
):
    """Apply BF16 RMSNorm and checkpoint INT8 quantization to projection rows."""
    capacity = pl.tensor.dim(source, 0)
    blocks = ((capacity + 15) // 16 * 16) // VEC_M_TILE
    workers = pl.min(blocks, pl.system.available_cluster_count())
    weight_row = pl.tensor.view(norm_weight, [1, width])
    for worker in pl.spmd(workers, name_hint="decode_norm_quant"):
        for block in pl.range(worker, blocks, workers):
            row = block * VEC_M_TILE
            rows = pl.max(0, pl.min(VEC_M_TILE, num_tokens - row))
            raw = pl.load(source, [row, 0], [VEC_M_TILE, width], valid_shape=[rows, width])
            padded = pl.set_validshape(pl.fillpad(raw, pad_value=pl.PadValue.zero), VEC_M_TILE, width)
            value = pl.cast(padded, pl.FP32)
            square = pl.mul(value, value)
            norm_scratch = pl.create_tile([VEC_M_TILE, width], dtype=pl.FP32)
            row_sum = pl.row_sum(square, tmp_tile=norm_scratch)
            mean = pl.mul(row_sum, 1.0 / width)
            rms_argument = pl.add(mean, eps)
            rsqrt_scratch = pl.create_tile([VEC_M_TILE, 1], dtype=pl.FP32)
            inverse_rms = _tile_rsqrt_for_norm(rms_argument, rsqrt_scratch)
            normalized_fp32 = pl.row_expand_mul(value, inverse_rms)
            gamma = pl.cast(pl.load(weight_row, [0, 0], [1, width]), pl.FP32)
            weighted = pl.col_expand_mul(normalized_fp32, gamma)
            normalized_bf16 = pl.cast(weighted, pl.BF16, mode="rint")
            normalized = pl.store(
                pl.set_validshape(normalized_bf16, rows, width),
                [row, 0],
                normalized,
            )
            quantized_fp32 = pl.cast(normalized_bf16, pl.FP32)
            scratch = pl.create_tile([VEC_M_TILE, width], dtype=pl.FP32)
            maximum = pl.row_max(pl.abs(quantized_fp32), tmp_tile=scratch)
            factor = pl.reshape(pl.maximum(pl.mul(maximum, 1.0 / 127.0), 1e-30), [VEC_M_TILE, 1])
            pl.store(factor, [row, 0], scale)
            denominator = pl.reshape(pl.mul(pl.maximum(maximum, 127e-30), 1.0 / 128.0), [VEC_M_TILE, 1])
            inverse = pl.recip(denominator)
            for k0 in pl.range(0, width, K_TILE):
                part = pl.tile.slice(quantized_fp32, [VEC_M_TILE, K_TILE], [0, k0])
                encoded = _checkpoint_codes(part, denominator, inverse)
                pl.store(encoded, [row, k0], payload)
    return normalized, payload, scale


@pl.jit.inline(auto_scope=False)
def quantize_activation(source: pl.Tensor, num_tokens: pl.Scalar[pl.INT32], width: pl.constexpr):
    """Quantize A3 checkpoint activations using the matching width and row schedule."""
    return quantize_projection_activation(source, num_tokens, width, True)

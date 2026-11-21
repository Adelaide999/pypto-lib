# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared W8A8 goldens and phase-specific device quantization."""

import pypto.language as pl
import torch
from .operator_config import CONTRACT

K_TILE = 256
N_TILE = 64
VEC_M_TILE = 8
WHOLE_ROW_TILE = 5120
PREFILL_QUANT_WORKERS = 48
QUANT_REDUCE_TILE = 1024

INT8_AMAX_EPS = CONTRACT.int8_amax_epsilon


DECODE_INT8_AMAX_EPS = CONTRACT.int8_amax_epsilon


CHECKPOINT_ROWS = pl.dynamic("CHECKPOINT_QUANT_ROWS")


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
                ones = pl.tile.full([VEC_M_TILE, 8], dtype=pl.FP32, value=1.0)
                scale_rows = pl.row_expand_mul(ones, scale)
                pl.store(pl.set_validshape(scale_rows, rows, 8), [row, 0], scales)
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
def quantize_activation(source: pl.Tensor, num_tokens: pl.Scalar[pl.INT32], width: pl.constexpr):
    """Quantize A3 checkpoint activations using the matching width and row schedule."""
    return quantize_projection_activation(source, num_tokens, width, True)


def make_dynamic_int8_quantize(capacity: int, width: int):
    """Build a PyPTO kernel for BF16 to INT8 per-token quantization.

    The scale uses the vLLM/A2/A3 contract directly: one FP32 dequantization
    scale per token, stored as ``[T, 1]``.
    """

    if capacity % 8 or width % K_TILE:
        raise ValueError("capacity must be divisible by 8 and width by 256")

    @pl.jit.inline
    def quantize(
        value: pl.Tensor[[capacity, width], pl.BF16],
        payload: pl.Tensor[[capacity, width], pl.INT8],
        scales: pl.Tensor[[capacity, 1], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(capacity // 8, name_hint="dsv41_a2a3_dynamic_quant"):
            t0 = block * 8
            maximum = pl.full([1, 8], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for k0 in pl.range(0, width, K_TILE):
                source = pl.slice(value, [8, K_TILE], [t0, k0])
                maximum = pl.maximum(
                    maximum, pl.reshape(pl.row_max(pl.abs(pl.cast(source, pl.FP32))), [1, 8])
                )
            inverse = pl.reshape(pl.div(pl.full([1, 8], dtype=pl.FP32, value=127.0), maximum), [8, 1])
            scale = pl.reshape(pl.div(maximum, 127.0), [8, 1])
            scales[t0 : t0 + 8, 0:1] = scale
            for k0 in pl.range(0, width, K_TILE):
                source_fp32 = pl.cast(pl.slice(value, [8, K_TILE], [t0, k0]), pl.FP32)
                normalized = pl.row_expand_mul(source_fp32, inverse)
                bounded = pl.minimum(pl.maximum(normalized, -127.0), 127.0)
                rounded = pl.cast(bounded, pl.INT32, mode="rint")
                payload[t0 : t0 + 8, k0 : k0 + K_TILE] = pl.cast(rounded, pl.INT8, mode="trunc")
        return payload, scales

    return quantize


def make_decode_dynamic_int8_quantize(capacity: int, width: int, *, cann_dispatch: bool = False):
    """Build a PyPTO kernel for BF16 to INT8 per-token quantization.

    The scale uses the vLLM/A2/A3 contract directly: one FP32 dequantization
    scale per token, stored as ``[T, 1]``.  CANN's fused dispatch path
    uses FP16 truncation before its rounded INT8 store; other paths retain
    the existing INT32 staging sequence.
    """

    if capacity % 8 or width % K_TILE:
        raise ValueError("capacity must be divisible by 8 and width by 256")

    @pl.jit.inline
    def quantize(
        value: pl.Tensor[[capacity, width], pl.BF16],
        payload: pl.Tensor[[capacity, width], pl.INT8],
        scales: pl.Tensor[[capacity, 1], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(capacity // 8, name_hint="dsv41_a2a3_dynamic_quant"):
            t0 = block * 8
            maximum = pl.full([1, 8], dtype=pl.FP32, value=DECODE_INT8_AMAX_EPS)
            for k0 in pl.range(0, width, K_TILE):
                source = pl.slice(value, [8, K_TILE], [t0, k0])
                maximum = pl.maximum(
                    maximum, pl.reshape(pl.row_max(pl.abs(pl.cast(source, pl.FP32))), [1, 8])
                )
            inverse = pl.reshape(pl.div(pl.full([1, 8], dtype=pl.FP32, value=127.0), maximum), [8, 1])
            scale = pl.reshape(pl.div(maximum, 127.0), [8, 1])
            scales[t0 : t0 + 8, 0:1] = scale
            for k0 in pl.range(0, width, K_TILE):
                source_fp32 = pl.cast(pl.slice(value, [8, K_TILE], [t0, k0]), pl.FP32)
                normalized = pl.row_expand_mul(source_fp32, inverse)
                bounded = pl.minimum(pl.maximum(normalized, -127.0), 127.0)
                if cann_dispatch:
                    payload[t0 : t0 + 8, k0 : k0 + K_TILE] = pl.cast(
                        pl.cast(bounded, target_type=pl.FP16, mode="trunc"),
                        target_type=pl.INT8,
                        mode="round",
                    )
                else:
                    rounded = pl.cast(bounded, pl.INT32, mode="rint")
                    payload[t0 : t0 + 8, k0 : k0 + K_TILE] = pl.cast(rounded, pl.INT8, mode="trunc")
        return payload, scales

    return quantize


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


def make_decode_cast_tile(rows, width, input_dtype, output_dtype):
    """Choose identity or a rounding cast before the DSL parses either body."""
    # PyPTO needs a concrete Tensor annotation while parsing an inline body.
    # A closure over ``input_dtype`` is a Python value, so the generic
    # annotation used here previously lost the tensor metadata during a full
    # layer compile.  Keep the small set of dtypes used by the model explicit.
    if input_dtype == pl.BF16 and output_dtype == pl.BF16:

        @pl.jit.inline
        def identity_bf16(value: pl.Tensor[[rows, width], pl.BF16]):
            return value

        return identity_bf16
    if input_dtype == pl.BF16 and output_dtype == pl.FP32:

        @pl.jit.inline
        def cast_bf16_fp32(value: pl.Tensor[[rows, width], pl.BF16]):
            return pl.cast(value, pl.FP32, mode="rint")

        return cast_bf16_fp32
    if input_dtype == pl.FP32 and output_dtype == pl.FP32:

        @pl.jit.inline
        def identity_fp32(value: pl.Tensor[[rows, width], pl.FP32]):
            return value

        return identity_fp32
    if input_dtype == pl.FP32 and output_dtype == pl.BF16:

        @pl.jit.inline
        def cast_fp32_bf16(value: pl.Tensor[[rows, width], pl.FP32]):
            return pl.cast(value, pl.BF16, mode="rint")

        return cast_fp32_bf16
    raise ValueError(f"unsupported cast tile dtypes: {input_dtype} -> {output_dtype}")


def make_quantize_row(width, *, fused_requantize=False):
    """Quantize one FP32 row with eight-segment maximum reduction."""
    @pl.jit.inline
    def quantize_row(value: pl.Tensor[[1, width], pl.FP32]):
        segments = pl.reshape(pl.abs(value), [8, width // 8])
        partial = pl.row_max(segments)
        ones = pl.full([8, 8], dtype=pl.FP32, value=1.0)
        repeated = pl.row_expand_mul(ones, partial)
        maximum = pl.col_max(repeated)
        if fused_requantize:
            scale = pl.div(maximum, 127.0)
            safe_scale = pl.maximum(scale, 1e-30)
            inverse_row = pl.recip(safe_scale)
        else:
            scale = pl.mul(maximum, 1.0 / 127.0)
            safe_maximum = pl.maximum(maximum, 1e-30)
            numerator = pl.full([1, 8], dtype=pl.FP32, value=127.0)
            inverse_row = pl.div(numerator, safe_maximum)
        inverse = pl.read(inverse_row, [0, 0])
        normalized = pl.mul(value, inverse)
        if fused_requantize:
            normalized_half = pl.cast(normalized, pl.FP16, mode="rint")
            normalized = pl.cast(normalized_half, pl.FP32)
        lower_bounded = pl.maximum(normalized, -127.0)
        bounded = pl.minimum(lower_bounded, 127.0)
        rounded = pl.cast(bounded, pl.INT32, mode="rint")
        exact_half = pl.cast(rounded, pl.FP16, mode="rint")
        codes = pl.cast(exact_half, pl.INT8, mode="trunc")
        return codes, scale

    return quantize_row


def make_quantize(
    capacity, width, input_dtype=pl.BF16, *, rows_tile=VEC_M_TILE, fused_requantize=False, whole_row=False,
):
    """Specialize symmetric per-token INT8 quantization."""
    if whole_row and width <= WHOLE_ROW_TILE:
        if width % 8:
            raise ValueError("Whole-row quantization requires eight equal reduction segments")
        widen_row = make_cast_tile(1, width, input_dtype, pl.FP32)
        quantize_row = make_quantize_row(width, fused_requantize=fused_requantize)

        @pl.jit.inline
        def quantize_whole_row(
            x: pl.Tensor[[capacity, width], input_dtype],
            payload: pl.Tensor[[capacity, width], pl.INT8],
            scales: pl.Tensor[[capacity, 8], pl.FP32],
            active: pl.Scalar[pl.INT32],
        ):
            for token in pl.spmd(active, name_hint="prefill_token_quantize_whole_row"):
                source = pl.slice(x, [1, width], [token, 0], valid_shape=[1, width])
                value = pl.create_tensor([1, width], dtype=pl.FP32)
                value[:, :] = widen_row(source)
                codes, scale = quantize_row(value)
                scales[token : token + 1, :] = scale
                payload[token : token + 1, :] = codes
            return payload, scales

        return quantize_whole_row

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
                value = pl.set_validshape(
                    pl.fillpad(widened, pad_value=pl.PadValue.zero), rows_tile, width_tile
                )
                chunk_maximum = pl.reshape(pl.row_max(pl.abs(value)), [1, rows_tile])
                maximum_row = pl.maximum(maximum_row, chunk_maximum)
            maximum = pl.reshape(maximum_row, [rows_tile, 1])
            if fused_requantize:
                scale = pl.div(maximum, 127.0)
            else:
                scale = pl.mul(maximum, 1.0 / 127.0)
            numerator = pl.full([1, rows_tile], dtype=pl.FP32, value=127.0)
            safe_maximum_row = pl.maximum(maximum_row, 1e-30)
            if fused_requantize:
                safe_scale_row = pl.maximum(pl.reshape(scale, [1, rows_tile]), 1e-30)
                inverse_row = pl.recip(safe_scale_row)
            else:
                inverse_row = pl.div(numerator, safe_maximum_row)
            inverse = pl.reshape(inverse_row, [rows_tile, 1])
            scale_rows = pl.full([rows_tile, 8], dtype=pl.FP32, value=1.0)
            expanded_scale = pl.row_expand_mul(scale_rows, scale)
            scales[t0 : t0 + rows_tile, :] = pl.set_validshape(expanded_scale, rows, 8)
            for k0 in pl.range(0, width, width_tile):
                source = pl.slice(x, [rows_tile, width_tile], [t0, k0], valid_shape=[rows, width_tile])
                widened = widen(source)
                value = pl.set_validshape(
                    pl.fillpad(widened, pad_value=pl.PadValue.zero), rows_tile, width_tile
                )
                normalized = pl.row_expand_mul(value, inverse)
                if fused_requantize:
                    normalized_half = pl.cast(normalized, pl.FP16, mode="rint")
                    normalized = pl.cast(normalized_half, pl.FP32)
                bounded = pl.minimum(pl.maximum(normalized, -127.0), 127.0)
                rounded = pl.cast(bounded, pl.INT32, mode="rint")
                exact_half = pl.cast(rounded, pl.FP16, mode="rint")
                codes = pl.cast(exact_half, pl.INT8, mode="trunc")
                payload[t0 : t0 + rows_tile, k0 : k0 + width_tile] = pl.set_validshape(
                    codes, rows, width_tile
                )
        return payload, scales

    return quantize


def make_prefill_projection_quantize(capacity, width):
    """Quantize contiguous groups of BF16 projection inputs to INT8."""
    if capacity < 1024 or capacity % VEC_M_TILE or width not in (1280, 5120):
        return make_quantize(capacity, width, whole_row=True)
    if width == 1280:

        @pl.jit.inline
        def quantize_query(
            x: pl.Tensor[[capacity, width], pl.BF16],
            payload: pl.Tensor[[capacity, width], pl.INT8],
            scales: pl.Tensor[[capacity, 8], pl.FP32],
            active: pl.Scalar[pl.INT32],
        ):
            return _quantize_cached_rows(x, payload, scales, active, capacity, width, False)

        return quantize_query

    column_tile = 1280

    @pl.jit.inline
    def quantize(
        x: pl.Tensor[[capacity, width], pl.BF16],
        payload: pl.Tensor[[capacity, width], pl.INT8],
        scales: pl.Tensor[[capacity, 8], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        tiles = (active + VEC_M_TILE - 1) // VEC_M_TILE
        workers = pl.min(tiles, PREFILL_QUANT_WORKERS)
        for worker in pl.spmd(workers, name_hint="prefill_token_quantize_whole_row"):
            start = worker * tiles // workers
            end = (worker + 1) * tiles // workers
            for block in pl.pipeline(start, end, stage=1):
                token = block * VEC_M_TILE
                rows = pl.min(VEC_M_TILE, active - token)
                source = x[token : token + VEC_M_TILE, :]
                peak = pl.full([1, VEC_M_TILE], dtype=pl.FP32, value=0.0)
                for col in pl.range(0, width, column_tile):
                    part = pl.cast(source[:, col : col + column_tile], pl.FP32)
                    peak = pl.maximum(peak, pl.reshape(pl.row_max(pl.abs(part)), [1, VEC_M_TILE]))
                numerator = pl.full([1, VEC_M_TILE], dtype=pl.FP32, value=127.0)
                inverse = pl.reshape(pl.div(numerator, pl.maximum(peak, 1e-30)), [VEC_M_TILE, 1])
                scale = pl.reshape(pl.mul(peak, 1.0 / 127.0), [VEC_M_TILE, 1])
                ones = pl.full([VEC_M_TILE, 8], dtype=pl.FP32, value=1.0)
                scales[token : token + VEC_M_TILE, :] = pl.set_validshape(pl.row_expand_mul(ones, scale), rows, 8)
                for col in pl.range(0, width, column_tile):
                    part = pl.cast(source[:, col : col + column_tile], pl.FP32)
                    rounded = pl.cast(pl.row_expand_mul(part, inverse), pl.INT32, mode="rint")
                    codes = pl.cast(rounded, pl.INT8, mode="trunc")
                    payload[token : token + VEC_M_TILE, col : col + column_tile] = pl.set_validshape(codes, rows, column_tile)
        return payload, scales

    return quantize


def make_prefill_projection_quantize_entry(capacity, width):
    """Compile the shared Prefill projection quantizer as a standalone leaf."""
    quantize_projection = make_prefill_projection_quantize(capacity, width)

    @pl.jit
    def quant(
        x: pl.Tensor[[capacity, width], pl.BF16],
        payload: pl.InOut[pl.Tensor[[capacity, width], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, 8], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        quantize_projection(x, payload, scales, active)
        return payload, scales

    return quant


def make_decode_quantize(capacity, width, input_dtype=pl.BF16):
    """Specialize symmetric per-token INT8 quantization."""
    if input_dtype == pl.FP32:

        @pl.jit.inline
        def quantize(
            x: pl.Tensor[[capacity, width], pl.FP32],
            payload: pl.Tensor[[capacity, width], pl.INT8],
            scales: pl.Tensor[[capacity, 8], pl.FP32],
            active: pl.Scalar[pl.INT32],
        ):
            for block in pl.spmd((active + VEC_M_TILE - 1) // VEC_M_TILE, name_hint="a2_token_quantize"):
                t0 = block * VEC_M_TILE
                rows = pl.min(VEC_M_TILE, active - t0)
                maximum_row = pl.full([1, VEC_M_TILE], dtype=pl.FP32, value=1e-30)
                for k0 in pl.range(0, width, K_TILE):
                    source = pl.slice(x, [VEC_M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                    source = pl.set_validshape(
                        pl.fillpad(source, pad_value=pl.PadValue.zero), VEC_M_TILE, K_TILE
                    )
                    chunk_maximum = pl.reshape(pl.row_max(pl.abs(source)), [1, VEC_M_TILE])
                    maximum_row = pl.maximum(maximum_row, chunk_maximum)
                maximum = pl.reshape(maximum_row, [VEC_M_TILE, 1])
                scale = pl.mul(maximum, 1.0 / 127.0)
                numerator = pl.full([1, VEC_M_TILE], dtype=pl.FP32, value=127.0)
                inverse_row = pl.div(numerator, maximum_row)
                inverse = pl.reshape(inverse_row, [VEC_M_TILE, 1])
                scale_rows = pl.full([VEC_M_TILE, 8], dtype=pl.FP32, value=1.0)
                scales[t0 : t0 + VEC_M_TILE, :] = pl.row_expand_mul(scale_rows, scale)
                for k0 in pl.range(0, width, K_TILE):
                    source = pl.slice(x, [VEC_M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                    source = pl.set_validshape(
                        pl.fillpad(source, pad_value=pl.PadValue.zero), VEC_M_TILE, K_TILE
                    )
                    normalized = pl.row_expand_mul(source, inverse)
                    bounded = pl.minimum(pl.maximum(normalized, -127.0), 127.0)
                    rounded = pl.cast(bounded, pl.INT32, mode="rint")
                    exact_half = pl.cast(rounded, pl.FP16, mode="rint")
                    payload[t0 : t0 + VEC_M_TILE, k0 : k0 + K_TILE] = pl.cast(
                        exact_half, pl.INT8, mode="trunc"
                    )
            return payload, scales

        return quantize

    @pl.jit.inline
    def quantize(
        x: pl.Tensor[[capacity, width], input_dtype],
        payload: pl.Tensor[[capacity, width], pl.INT8],
        scales: pl.Tensor[[capacity, 8], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd((active + VEC_M_TILE - 1) // VEC_M_TILE, name_hint="a2_token_quantize"):
            t0 = block * VEC_M_TILE
            rows = pl.min(VEC_M_TILE, active - t0)
            maximum_row = pl.full([1, VEC_M_TILE], dtype=pl.FP32, value=1e-30)
            for k0 in pl.range(0, width, K_TILE):
                source = pl.slice(x, [VEC_M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), VEC_M_TILE, K_TILE)
                value = pl.cast(source, pl.FP32, mode="rint")
                chunk_maximum = pl.reshape(pl.row_max(pl.abs(value)), [1, VEC_M_TILE])
                maximum_row = pl.maximum(maximum_row, chunk_maximum)
            maximum = pl.reshape(maximum_row, [VEC_M_TILE, 1])
            scale = pl.mul(maximum, 1.0 / 127.0)
            numerator = pl.full([1, VEC_M_TILE], dtype=pl.FP32, value=127.0)
            inverse_row = pl.div(numerator, maximum_row)
            inverse = pl.reshape(inverse_row, [VEC_M_TILE, 1])
            scale_rows = pl.full([VEC_M_TILE, 8], dtype=pl.FP32, value=1.0)
            scales[t0 : t0 + VEC_M_TILE, :] = pl.row_expand_mul(scale_rows, scale)
            for k0 in pl.range(0, width, K_TILE):
                source = pl.slice(x, [VEC_M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), VEC_M_TILE, K_TILE)
                value = pl.cast(source, pl.FP32, mode="rint")
                normalized = pl.row_expand_mul(value, inverse)
                bounded = pl.minimum(pl.maximum(normalized, -127.0), 127.0)
                rounded = pl.cast(bounded, pl.INT32, mode="rint")
                exact_half = pl.cast(rounded, pl.FP16, mode="rint")
                payload[t0 : t0 + VEC_M_TILE, k0 : k0 + K_TILE] = pl.cast(exact_half, pl.INT8, mode="trunc")
        return payload, scales

    return quantize


def make_dequantize(capacity, output_width, combine_scales=False, channel_dtype=pl.FP32, *, reuse_workers=False):
    """Apply channel-first or combined scales with the selected channel rounding."""
    if reuse_workers and not combine_scales and channel_dtype == pl.FP32 and output_width == 5120:
        column_tile = 1280

        @pl.jit.inline
        def dequantize_shared(
            accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
            channel_scale: pl.Tensor[[output_width], pl.FP32],
            token_scale: pl.Tensor[[capacity, 8], pl.FP32],
            output: pl.Tensor[[capacity, output_width], pl.BF16],
            active: pl.Scalar[pl.INT32],
        ):
            tiles = (active + VEC_M_TILE - 1) // VEC_M_TILE
            groups = pl.min(48 // (output_width // column_tile), tiles)
            for worker in pl.spmd(groups * (output_width // column_tile), name_hint="shared_down_dequant_stream"):
                column = worker % (output_width // column_tile) * column_tile
                group = worker // (output_width // column_tile)
                channels = pl.reshape(channel_scale[column : column + column_tile], [1, column_tile])
                begin = group * tiles // groups
                end = (group + 1) * tiles // groups
                for block in pl.range(begin, end):
                    token = block * VEC_M_TILE
                    rows = pl.min(VEC_M_TILE, active - token)
                    integer = accumulator[token : token + VEC_M_TILE, column : column + column_tile]
                    value = pl.col_expand_mul(pl.cast(integer, pl.FP32, mode="rint"), channels)
                    scale = pl.row_min(token_scale[token : token + VEC_M_TILE, :])
                    result = pl.cast(pl.row_expand_mul(value, scale), pl.BF16, mode="rint")
                    narrowed = pl.set_validshape(result, rows, column_tile)
                    output[token : token + VEC_M_TILE, column : column + column_tile] = narrowed
            return output

        return dequantize_shared

    column_tile = 1280 if output_width == 5120 else 1024
    while column_tile > N_TILE and output_width % column_tile:
        column_tile //= 2
    narrow_channel = make_cast_tile(VEC_M_TILE, column_tile, pl.FP32, channel_dtype)
    widen_channel = make_cast_tile(VEC_M_TILE, column_tile, channel_dtype, pl.FP32)

    @pl.jit.inline
    def dequantize(
        accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
        channel_scale: pl.Tensor[[output_width], pl.FP32],
        token_scale: pl.Tensor[[capacity, 8], pl.FP32],
        output: pl.Tensor[[capacity, output_width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        row_blocks = (active + VEC_M_TILE - 1) // VEC_M_TILE
        for block in pl.spmd(row_blocks * (output_width // column_tile), name_hint="a2_dequantize"):
            t0 = (block // (output_width // column_tile)) * VEC_M_TILE
            n0 = (block % (output_width // column_tile)) * column_tile
            rows = pl.min(VEC_M_TILE, active - t0)
            integer = accumulator[t0 : t0 + VEC_M_TILE, n0 : n0 + column_tile]
            value = pl.cast(integer, pl.FP32, mode="rint")
            channels = pl.reshape(channel_scale[n0 : n0 + column_tile], [1, column_tile])
            scale = pl.row_min(token_scale[t0 : t0 + VEC_M_TILE, :])
            if combine_scales:
                ones = pl.full([VEC_M_TILE, column_tile], dtype=pl.FP32, value=1.0)
                channel_rows = pl.col_expand_mul(ones, channels)
                combined = pl.row_expand_mul(channel_rows, scale)
                scaled = pl.mul(value, combined)
            else:
                scaled_channels = pl.create_tensor([VEC_M_TILE, column_tile], dtype=pl.FP32)
                scaled_channels[:, :] = pl.col_expand_mul(value, channels)
                rounded_channels = pl.create_tensor([VEC_M_TILE, column_tile], dtype=channel_dtype)
                rounded_channels[:, :] = narrow_channel(scaled_channels)
                channel_fp32 = widen_channel(rounded_channels)
                scaled = pl.row_expand_mul(channel_fp32, scale)
            narrowed = pl.cast(scaled, pl.BF16, mode="rint")
            output[t0 : t0 + VEC_M_TILE, n0 : n0 + column_tile] = pl.set_validshape(narrowed, rows, column_tile)
        return output

    return dequantize


def make_decode_dequantize(capacity, output_width, combine_scales=False, output_dtype=pl.BF16):
    """Apply the selected native FP32 scale boundary to INT32 products.

    The native per-token optimized kernel combines channel/token scales before
    multiplying the accumulator. Other native kernels scale channel-first.
    Callers select their source boundary explicitly; QKV keeps its existing ABI.
    """
    if output_dtype == pl.FP32:

        @pl.jit.inline
        def dequantize(
            accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
            channel_scale: pl.Tensor[[output_width], pl.FP32],
            token_scale: pl.Tensor[[capacity, 8], pl.FP32],
            output: pl.Tensor[[capacity, output_width], pl.FP32],
            active: pl.Scalar[pl.INT32],
        ):
            row_blocks = (active + VEC_M_TILE - 1) // VEC_M_TILE
            for block in pl.spmd(row_blocks * (output_width // N_TILE), name_hint="a2_dequantize_fp32"):
                t0 = (block // (output_width // N_TILE)) * VEC_M_TILE
                n0 = (block % (output_width // N_TILE)) * N_TILE
                rows = pl.min(VEC_M_TILE, active - t0)
                integer = accumulator[t0 : t0 + VEC_M_TILE, n0 : n0 + N_TILE]
                value = pl.cast(integer, pl.FP32, mode="rint")
                channels = pl.reshape(channel_scale[n0 : n0 + N_TILE], [1, N_TILE])
                scale = pl.row_min(token_scale[t0 : t0 + VEC_M_TILE, :])
                if combine_scales:
                    ones = pl.full([VEC_M_TILE, N_TILE], dtype=pl.FP32, value=1.0)
                    channel_rows = pl.col_expand_mul(ones, channels)
                    combined = pl.row_expand_mul(channel_rows, scale)
                    scaled = pl.mul(value, combined)
                else:
                    scaled_channels = pl.col_expand_mul(value, channels)
                    scaled = pl.row_expand_mul(scaled_channels, scale)
                output[t0 : t0 + VEC_M_TILE, n0 : n0 + N_TILE] = pl.set_validshape(scaled, rows, N_TILE)
            return output

        return dequantize

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
            integer = accumulator[t0 : t0 + VEC_M_TILE, n0 : n0 + N_TILE]
            value = pl.cast(integer, pl.FP32, mode="rint")
            channels = pl.reshape(channel_scale[n0 : n0 + N_TILE], [1, N_TILE])
            scale = pl.row_min(token_scale[t0 : t0 + VEC_M_TILE, :])
            if combine_scales:
                ones = pl.full([VEC_M_TILE, N_TILE], dtype=pl.FP32, value=1.0)
                channel_rows = pl.col_expand_mul(ones, channels)
                combined = pl.row_expand_mul(channel_rows, scale)
                scaled = pl.mul(value, combined)
            else:
                scaled_channels = pl.col_expand_mul(value, channels)
                scaled = pl.row_expand_mul(scaled_channels, scale)
            narrowed = pl.cast(scaled, pl.BF16, mode="rint")
            output[t0 : t0 + VEC_M_TILE, n0 : n0 + N_TILE] = pl.set_validshape(narrowed, rows, N_TILE)
        return output

    return dequantize


def _round_symmetric_int8(value: torch.Tensor) -> torch.Tensor:
    rounded = torch.round(value).clamp(-CONTRACT.int8_scale_max, CONTRACT.int8_scale_max)
    return rounded.to(torch.int32).to(torch.float16).to(torch.int8)


def quantize_per_token_int8(
    value: torch.Tensor,
    smooth_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each row to INT8 and return an FP32 dequant scale.

    This mirrors ``torch_npu.npu_dynamic_quant`` used by
    ``AscendW8A8DynamicLinearMethod``.  The returned scale has a trailing
    singleton dimension and is suitable for broadcasting over the row.
    """

    source = value.float()
    if smooth_scale is not None:
        source = source * smooth_scale.float()
    amax = source.abs().amax(dim=-1, keepdim=True).clamp_min(CONTRACT.int8_amax_epsilon)
    dequant_scale = amax / CONTRACT.int8_scale_max
    payload = _round_symmetric_int8(source / dequant_scale)
    return payload, dequant_scale


def quantize_per_channel_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an output-major ``[N, K]`` weight with FP32 channel scales."""

    source = weight.float()
    amax = source.abs().amax(dim=-1, keepdim=True).clamp_min(CONTRACT.int8_amax_epsilon)
    dequant_scale = amax / CONTRACT.int8_scale_max
    payload = _round_symmetric_int8(source / dequant_scale)
    return payload, dequant_scale


def decode_quantize_per_token_int8(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror the FP32 division and rounding order of ``make_decode_quantize``.

    Both BF16 and FP32 inputs use ``x * (127 / amax)`` before nearest-even
    conversion. Dividing by the dequant scale can cross an INT8 half-integer
    boundary because its FP32 intermediates differ from the device path.
    """
    source = value.float()
    maximum = source.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    dequant_scale = maximum * (1.0 / CONTRACT.int8_scale_max)
    inverse = torch.div(torch.full_like(maximum, CONTRACT.int8_scale_max), maximum)
    payload = _round_symmetric_int8(source * inverse)
    return payload, dequant_scale


def int8_matmul(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``[..., K] @ [N, K].T`` with an INT32 accumulator."""

    if value.shape[-1] != weight.shape[-1]:
        raise ValueError(f"K mismatch: activation={value.shape[-1]}, weight={weight.shape[-1]}")
    if value.shape[-1] % CONTRACT.int8_k_alignment:
        raise ValueError(
            f"A2/A3 INT8 GEMM requires K aligned to {CONTRACT.int8_k_alignment}, got {value.shape[-1]}"
        )
    return torch.matmul(value.to(torch.int32), weight.to(torch.int32).transpose(-2, -1))


def dequantize_int32(
    accumulator: torch.Tensor,
    token_scale: torch.Tensor,
    channel_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Apply FP32 token/channel scales after INT32 accumulation."""

    result = accumulator.float() * token_scale.float()
    result = result * channel_scale.float().reshape(*([1] * (result.ndim - 1)), -1)
    return result.to(out_dtype)


def w8a8_linear(
    value: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    smooth_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reference for dynamic W8A8 linear on A2/A3."""

    payload, token_scale = quantize_per_token_int8(value, smooth_scale)
    accumulator = int8_matmul(payload, weight)
    result = dequantize_int32(accumulator, token_scale, weight_scale, out_dtype)
    if bias is not None:
        result = (result.float() + bias.float()).to(out_dtype)
    return result

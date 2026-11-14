# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode indexer, candidate-pool, cache, and C1/C2 cache-publish operators."""

from __future__ import annotations

import torch
import pypto.language as pl

from .operator_config import CONTRACT
from .decode_network_decode_primitives import (
    N_TILE,
    make_dequantize,
    make_dot,
    make_pack_weight,
    make_quantize,
    make_rope,
)
from .decode_network_decode_qkv_proj_rope import make_bf16_projection_staged


# ---- decode_network_decode_c1c2.py ----

def make_compressed_cache_publish(capacity, blocks, width):
    """Publish BF16 compressed KV rows into PA_BBND pages.

    ``slots`` contains the flattened page row (``page * 128 + offset``), as
    produced by the vLLM A2 scheduler.  A negative slot is padding and leaves
    the cache byte-for-byte untouched.
    """
    rows = blocks * 128

    @pl.jit.inline
    def publish(
        value: pl.Tensor[[capacity, width], pl.BF16],
        slots: pl.Tensor[[capacity], pl.INT64],
        cache: pl.Tensor[[blocks, 128, 1, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(cache, [rows, width])
        with pl.spmd(active, name_hint="a2_c2_bf16_cache_publish"):
            token = pl.tile.get_block_idx()
            slot_i64 = pl.read(slots, [token])
            if slot_i64 >= 0:
                slot = pl.cast(slot_i64, pl.INDEX)
                flat[slot : slot + 1, :] = value[token : token + 1, :]
        return cache

    return publish


def make_indexer_cache_publish(capacity, blocks, width):
    """Quantize index keys to INT8/FP16 and publish both cache tensors.

    Ascend's non-A5 indexer path uses one dynamic scale for each key row.  The
    A2 PyPTO representation stores that scale as ``[blocks,128,1,1]`` FP16,
    matching ``indexer_quant_scatter`` in vLLM-Ascend.  Quantization is the
    same symmetric per-token INT8 operation used by the A2 W8A8 projections.
    """
    rows = blocks * 128
    quantize = make_quantize(capacity, width)

    @pl.jit.inline
    def publish(
        value: pl.Tensor[[capacity, width], pl.BF16],
        slots: pl.Tensor[[capacity], pl.INT64],
        cache: pl.Tensor[[blocks, 128, 1, width], pl.INT8],
        scales: pl.Tensor[[blocks, 128, 1, 1], pl.FP16],
        active: pl.Scalar[pl.INT32],
    ):
        payload = pl.create_tensor([capacity, width], dtype=pl.INT8)
        row_scales = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize(value, payload, row_scales, active)
        cache_flat = pl.reshape(cache, [rows, width])
        scale_flat = pl.reshape(scales, [rows, 1])
        with pl.spmd(active, name_hint="a2_c1_int8_cache_publish"):
            token = pl.tile.get_block_idx()
            slot_i64 = pl.read(slots, [token])
            if slot_i64 >= 0:
                slot = pl.cast(slot_i64, pl.INDEX)
                cache_flat[slot : slot + 1, :] = payload[token : token + 1, :]
        # A scalar FP16 row is only two bytes, while Ascend writes a full
        # cache line.  Serialize these writes so adjacent rows cannot race.
        with pl.spmd(1, name_hint="a2_c1_int8_scale_publish"):
            _worker = pl.tile.get_block_idx()
            for token in pl.range(active):
                slot_i64 = pl.read(slots, [token])
                if slot_i64 >= 0:
                    slot = pl.cast(slot_i64, pl.INDEX)
                    scale = pl.cast(pl.read(row_scales, [token, 0]), pl.FP16)
                    pl.write(scale_flat, [slot, 0], scale)
        return cache, scales

    return publish

# ---- decode_network_decode_candidate_pool.py ----

CANDIDATE_BLOCK_SIZE = 8
CANDIDATE_TOPK_BLOCKS = 2048
SCORE_WIDTH = 512
POOL_CAPACITY = SCORE_WIDTH // CANDIDATE_BLOCK_SIZE


def make_candidate_block_pool(
    capacity: int,
    *,
    score_width: int = SCORE_WIDTH,
    topk_blocks: int = CANDIDATE_TOPK_BLOCKS,
    block_size: int = CANDIDATE_BLOCK_SIZE,
):
    """Build the fixed-bucket candidate publisher used by the decoder stages."""

    if capacity <= 0 or capacity % 16:
        raise ValueError("candidate pool capacity must be a positive multiple of 16")
    if score_width != SCORE_WIDTH or topk_blocks != CANDIDATE_TOPK_BLOCKS:
        raise ValueError("A2/A3 candidate pool uses a fixed 512-row score bucket")
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
                length = pl.max(pl.read(compressed_lengths, [token]), 0)
                visible_blocks = pl.min(
                    (length + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE,
                    POOL_CAPACITY,
                )
                # Candidate publication retains every visible causal block.  A
                # score sort is unnecessary here because the pool capacity is
                # the full visible prefix; sorting the mixed FP32/UINT32 pair
                # on A2/A3 also reinterprets the score sentinel as a block ID.
                ordered_ids = pl.tile.full([1, POOL_CAPACITY], dtype=pl.INT32, value=-1)
                for lane in pl.range(POOL_CAPACITY):
                    if lane < visible_blocks:
                        pl.tile.write(ordered_ids, [0, lane], pl.cast(lane, pl.INT32))
                pl.store(ordered_ids, [token, 0], pool_block_ids)
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

# ---- decode_network_indexer_prepare.py ----

INDEXER_TOPK = 512
SORT_LEAF = 512
NEGATIVE_INF = -3.0e38


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

# ---- decode_network_decode_index_cache.py ----

# ``npu_dynamic_quant`` is nearest rounding for ordinary values, but an exact
# half-way value is rounded toward zero on the A2/A3 kernel.  Keep the small
# bias in FP32 so the published FP16 scale remains unchanged.
_ROUND_TIE_EPS = 2.0 ** -20


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
    scale = torch.maximum(raw_scale.to(torch.float16), torch.full_like(raw_scale, 2.0 ** -24, dtype=torch.float16))
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
            scale = pl.maximum(rounded_scale, 2.0 ** -24)
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
            output[row:row + 16, :] = pl.set_validshape(rounded, valid_rows, head_dim)
            head = row % heads
            token = row // heads
            scale_row = pl.reshape(pl.cast(scale, pl.FP16), [1, 16])
            scales[token:token + 1, head:head + 16] = scale_row
        return payload, scales

    return quantize

# ---- decode_network_decode_indexer_projection.py ----

# Model configuration.
INDEX_HEADS = 32
INDEX_WIDTH = 128
INDEX_TOPK = 512
WINDOW_SIZE = 128

# Tiling.
QUANT_ROWS_TILE = 16
WEIGHT_ROWS_TILE = 8


def make_indexer_projection(capacity, config):
    """Project and rotate index queries and emit scaled FP16 head weights."""
    hidden = config["hidden_size"]
    latent = config["q_lora_rank"]
    rope_width = config["qk_rope_head_dim"]
    pairs = rope_width // 2
    index_width = INDEX_HEADS * INDEX_WIDTH
    packed_rows = latent * (index_width // N_TILE)
    weights_scale = INDEX_WIDTH ** -0.5 * INDEX_HEADS ** -0.5
    quantize = make_quantize(capacity, latent)
    pack = make_pack_weight(latent, index_width)
    dot = make_dot(capacity, latent, index_width)
    dequantize = make_dequantize(capacity, index_width)
    rotate = make_rope(capacity, INDEX_HEADS, INDEX_WIDTH, rope_width)
    project_weights = make_bf16_projection_staged(hidden, INDEX_HEADS)

    @pl.jit
    def indexer_projection(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        query_latent: pl.Tensor[[capacity, latent], pl.BF16],
        index_wq_b: pl.Tensor[[latent, index_width], pl.INT8],
        index_wq_b_scale: pl.Tensor[[index_width], pl.FP32],
        weights_proj: pl.Tensor[[hidden, INDEX_HEADS], pl.BF16],
        cosine: pl.Tensor[[capacity, pairs], pl.FP32],
        sine: pl.Tensor[[capacity, pairs], pl.FP32],
        query: pl.InOut[pl.Tensor[[capacity, INDEX_HEADS, INDEX_WIDTH], pl.BF16]],
        weights: pl.InOut[pl.Tensor[[capacity, INDEX_HEADS], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        latent_i8 = pl.create_tensor([capacity, latent], dtype=pl.INT8)
        latent_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize(query_latent, latent_i8, latent_scale, active)
        packed = pl.create_tensor([packed_rows, N_TILE], dtype=pl.INT8)
        pack(index_wq_b, packed)
        accumulator = pl.create_tensor([capacity, index_width], dtype=pl.INT32)
        dot(latent_i8, packed, accumulator, active)
        projected = pl.create_tensor([capacity, index_width], dtype=pl.BF16)
        dequantize(accumulator, index_wq_b_scale, latent_scale, projected, active)
        query_flat = pl.reshape(query, [capacity, index_width])
        rotate(projected, cosine, sine, query_flat, active)

        projected_weights = pl.create_tensor([capacity, INDEX_HEADS], dtype=pl.BF16)
        project_weights(x, weights_proj, projected_weights, active)
        with pl.spmd((active + WEIGHT_ROWS_TILE - 1) // WEIGHT_ROWS_TILE, name_hint="a2_index_weights"):
            block = pl.tile.get_block_idx()
            row = block * WEIGHT_ROWS_TILE
            rows = pl.min(WEIGHT_ROWS_TILE, active - row)
            value = pl.slice(projected_weights, [WEIGHT_ROWS_TILE, INDEX_HEADS], [row, 0], valid_shape=[rows, INDEX_HEADS])
            scaled = pl.mul(pl.cast(value, pl.FP32), weights_scale)
            narrowed = pl.cast(scaled, pl.FP16, mode="rint")
            weights[row:row + WEIGHT_ROWS_TILE, :] = pl.set_validshape(narrowed, rows, INDEX_HEADS)
        return query, weights

    return indexer_projection


def make_indexer_query_quantize(capacity):
    """Quantize each index head using its rounded FP16 scale."""

    @pl.jit
    def indexer_query_quantize(
        query: pl.Tensor[[capacity, INDEX_HEADS, INDEX_WIDTH], pl.BF16],
        payload: pl.InOut[pl.Tensor[[capacity, INDEX_HEADS, INDEX_WIDTH], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, INDEX_HEADS], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        source = pl.reshape(query, [capacity * INDEX_HEADS, INDEX_WIDTH])
        output = pl.reshape(payload, [capacity * INDEX_HEADS, INDEX_WIDTH])
        blocks = (active * INDEX_HEADS + QUANT_ROWS_TILE - 1) // QUANT_ROWS_TILE
        with pl.spmd(blocks, name_hint="a2_index_query_quant"):
            block = pl.tile.get_block_idx()
            row = block * QUANT_ROWS_TILE
            rows = pl.min(QUANT_ROWS_TILE, active * INDEX_HEADS - row)
            q = pl.slice(source, [QUANT_ROWS_TILE, INDEX_WIDTH], [row, 0], valid_shape=[rows, INDEX_WIDTH])
            q = pl.set_validshape(pl.fillpad(q, pad_value=pl.PadValue.zero), QUANT_ROWS_TILE, INDEX_WIDTH)
            value = pl.cast(q, pl.FP32)
            maximum = pl.row_max(pl.abs(value))
            raw_scale = pl.div(maximum, 127.0)
            rounded_scale = pl.cast(pl.cast(raw_scale, pl.FP16, mode="rint"), pl.FP32)
            scale = pl.maximum(rounded_scale, 2.0 ** -24)
            normalized = pl.row_expand_div(value, scale)
            rounded = pl.cast(normalized, pl.INT32, mode="rint")
            bounded = pl.minimum(pl.maximum(rounded, -127), 127)
            quantized = pl.cast(bounded, pl.INT8, mode="trunc")
            output[row:row + QUANT_ROWS_TILE, :] = pl.set_validshape(quantized, rows, INDEX_WIDTH)
            scale_row = pl.reshape(pl.cast(scale, pl.FP16, mode="rint"), [1, QUANT_ROWS_TILE])
            token = row // INDEX_HEADS
            head = row % INDEX_HEADS
            scales[token:token + 1, head:head + QUANT_ROWS_TILE] = scale_row
        return payload, scales

    return indexer_query_quantize


def make_empty_selection(capacity):
    """Publish the source empty-cache selection sentinel."""

    @pl.jit
    def empty_selection(
        raw_selected: pl.InOut[pl.Tensor[[capacity, INDEX_TOPK], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(active, name_hint="a2_empty_sparse_selection"):
            token = pl.tile.get_block_idx()
            raw_selected[token:token + 1, :] = pl.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
        return raw_selected

    return empty_selection


def make_sparse_controls(capacity, requests, long_pages, window_pages, long_page_rows=64, *, compress_ratio=2):
    """Filter and sort QLI logical IDs, then map both caches to physical rows."""
    if long_page_rows not in (64, 128):
        raise ValueError("A2 compressed page rows must match the actual 64/128-row cache ABI")
    if compress_ratio not in (1, 2):
        raise ValueError("Sparse controls require the physical source compression ratio 1 or 2")

    @pl.jit
    def sparse_controls(
        raw_selected: pl.Tensor[[capacity, INDEX_TOPK], pl.INT32],
        positions: pl.Tensor[[capacity], pl.INT64],
        request_ids: pl.Tensor[[capacity], pl.INT32],
        long_block_table: pl.Tensor[[requests, long_pages], pl.INT32],
        window_block_table: pl.Tensor[[requests, window_pages], pl.INT32],
        selected: pl.InOut[pl.Tensor[[capacity, INDEX_TOPK], pl.INT32]],
        selected_lengths: pl.InOut[pl.Tensor[[capacity, 1], pl.INT32]],
        compressed_indices: pl.InOut[pl.Tensor[[capacity, INDEX_TOPK], pl.INT32]],
        window_indices: pl.InOut[pl.Tensor[[capacity, WINDOW_SIZE], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(active, name_hint="a2_prepare_sparse_indices") as controls_tid:
            token = pl.tile.get_block_idx()
            position = pl.cast(pl.read(positions, [token]), pl.INT32)
            visible = (position + 1) // compress_ratio
            request = pl.read(request_ids, [token])
            sort_scores = pl.tile.full([1, INDEX_TOPK], dtype=pl.FP32, value=-1e30)
            sort_ids = pl.tile.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
            for lane in pl.range(INDEX_TOPK):
                index = pl.read(raw_selected, [token, lane])
                if index >= 0:
                    if index < visible:
                        pl.tile.write(sort_scores, [0, lane], -pl.cast(index, pl.FP32))
                        pl.tile.write(sort_ids, [0, lane], index)
            sorted_pairs = pl.tile.sort32(sort_scores, pl.reinterpret_view(sort_ids, pl.UINT32))
            sorted_pairs = pl.tile.mrgsort(sorted_pairs, block_len=64)
            sorted_pairs = pl.tile.mrgsort(sorted_pairs, block_len=256)
            logical_ids = pl.tile.gather_mask(sorted_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            pl.store(logical_ids, [token, 0], selected)
            physical_ids = pl.tile.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
            for lane in pl.range(INDEX_TOPK):
                logical = pl.tile.read(logical_ids, [0, lane])
                if logical >= 0:
                    page = logical // long_page_rows
                    offset = logical % long_page_rows
                    block = pl.read(long_block_table, [request, page])
                    physical = pl.cast(block * long_page_rows + offset, pl.INT32)
                    pl.tile.write(physical_ids, [0, lane], physical)
            pl.store(physical_ids, [token, 0], compressed_indices)

            left = pl.max(0, position + 1 - WINDOW_SIZE)
            window_count = position + 1 - left
            window_ids = pl.tile.full([1, WINDOW_SIZE], dtype=pl.INT32, value=-1)
            for lane in pl.range(WINDOW_SIZE):
                if lane < window_count:
                    window_logical = left + lane
                    window_page = window_logical // WINDOW_SIZE
                    window_offset = window_logical % WINDOW_SIZE
                    window_block = pl.read(window_block_table, [request, window_page])
                    window_physical = pl.cast(window_block * WINDOW_SIZE + window_offset, pl.INT32)
                    pl.tile.write(window_ids, [0, lane], window_physical)
            pl.store(window_ids, [token, 0], window_indices)
        with pl.spmd(1, name_hint="a2_sparse_lengths_publish", deps=[controls_tid]):
            _worker = pl.tile.get_block_idx()
            for token in pl.range(active):
                count = 0
                for lane in pl.range(INDEX_TOPK):
                    if pl.read(selected, [token, lane]) >= 0:
                        count = count + 1
                pl.write(selected_lengths, [token, 0], pl.cast(count, pl.INT32))
        return selected, selected_lengths, compressed_indices, window_indices

    return sparse_controls

# ---- decode_network_decode_indexer_select.py ----

INDEX_HEADS = 32
INDEX_HEAD_DIM = 128
INDEX_TOPK = 512
KEY_CAPACITY = 512
KEY_TILE = 32
HEAD_TILE = 16
# A2/A3 sort32 is most reliable at this width.  The public kernel signature
# currently fixes the candidate row to this size.
SORT_LEAF = 512
NEGATIVE_INF = -3.0e38


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
            key_tile = key[k0:k0 + KEY_TILE, :]
            scale_tile = pl.reshape(pl.cast(key_scale[k0:k0 + KEY_TILE], pl.FP32), [1, KEY_TILE])
            total = pl.full([1, KEY_TILE], dtype=pl.FP32, value=0.0)
            for h0 in pl.range(0, heads, HEAD_TILE):
                q_tile = pl.reshape(query[token:token + 1, h0:h0 + HEAD_TILE, :], [HEAD_TILE, head_dim])
                dots = pl.matmul(q_tile, key_tile, b_trans=True, out_dtype=pl.INT32)
                dots = pl.cast(dots, pl.FP32, mode="none")
                q_scale_tile = pl.reshape(
                    pl.cast(query_scale[token, h0:h0 + HEAD_TILE], pl.FP32), [HEAD_TILE, 1]
                )
                weight_tile = pl.reshape(
                    pl.cast(weights[token, h0:h0 + HEAD_TILE], pl.FP32), [HEAD_TILE, 1]
                )
                dots = pl.row_expand_mul(dots, q_scale_tile)
                dots = pl.col_expand_mul(dots, scale_tile)
                dots = pl.maximum(dots, 0.0)
                dots = pl.row_expand_mul(dots, weight_tile)
                total = pl.add(total, pl.reshape(pl.col_sum(dots), [1, KEY_TILE]))
            scores[token:token + 1, k0:k0 + KEY_TILE] = pl.fillpad(
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
        leaf_pairs = pl.tile.sort32(
            row, pl.reinterpret_view(leaf_indices, pl.UINT32), tmp=sort_tmp
        )
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
                query, query_scale, key, key_scale, weights, positions,
                running_pairs, scores, indices,
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
            key_tile = key[k0:k0 + KEY_TILE, :]
            scale_tile = pl.reshape(pl.cast(key_scale[k0:k0 + KEY_TILE], pl.FP32), [1, KEY_TILE])
            total = pl.full([1, KEY_TILE], dtype=pl.FP32, value=0.0)
            for h0 in pl.range(0, heads, HEAD_TILE):
                q_tile = pl.reshape(query[token:token + 1, h0:h0 + HEAD_TILE, :], [HEAD_TILE, head_dim])
                dots = pl.matmul(q_tile, key_tile, b_trans=True, out_dtype=pl.INT32)
                dots = pl.cast(dots, pl.FP32, mode="none")
                q_scale_tile = pl.reshape(
                    pl.cast(query_scale[token, h0:h0 + HEAD_TILE], pl.FP32), [HEAD_TILE, 1]
                )
                weight_tile = pl.reshape(
                    pl.cast(weights[token, h0:h0 + HEAD_TILE], pl.FP32), [HEAD_TILE, 1]
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
            scores[token:token + 1, k0:k0 + KEY_TILE] = total

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


__all__ = [
    "CANDIDATE_BLOCK_SIZE",
    "CANDIDATE_TOPK_BLOCKS",
    "INDEX_HEADS",
    "INDEX_HEAD_DIM",
    "INDEX_TOPK",
    "INDEXER_TOPK",
    "POOL_CAPACITY",
    "SCORE_WIDTH",
    "dequantize_index_cache",
    "indexer_select_reference",
    "make_candidate_block_pool",
    "make_compressed_cache_publish",
    "make_index_cache_quantize",
    "make_indexer_prepare",
    "make_indexer_projection",
    "make_indexer_query_quantize",
    "make_indexer_select",
    "make_indexer_select_restricted",
    "make_sparse_controls",
    "prepare_indexer_indices_reference",
    "quantize_index_cache",
]

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill C1A and C2A index-key preparation, paged scoring, and selection."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

from .gemm import gemm_bf16_to_bf16, gemm_dynamic_int8

import torch
import torch.nn.functional as F


import pypto.language as pl
INDEX_POOL_SCORE_DYN = pl.dynamic("A3_INDEX_POOL_SCORE_DYN")
from .quant import decode_fp4, decode_scale

from .config import INDEX_CACHE_GROUP, FLASH as M, INDEX_BLOCKS_DYN, INDEX_POOL_BLOCKS_DYN, T_DYN
from .rmsnorm import EPS, rms_norm, golden_rms_norm
from .quant import dequantize_mxfp4_cache, quantize_mxfp4_cache
from .gemm import golden_dynamic_int8
from .qkv_proj_rope import _rotate_rope, golden_rope_tail, golden_rope_interleave
from .rope import make_pipelined_rope


# model config
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
INDEX_H = M.index_n_heads
INDEX_DIM = M.index_head_dim
INDEX_TOPK = M.index_topk
INDEX_QUERY_WIDTH = M.index_n_heads * INDEX_DIM
INDEX_QUERY_SCALE_WIDTH = INDEX_QUERY_WIDTH // 32
CANDIDATE_BLOCK_SIZE = M.candidate_block_size
CANDIDATE_TOPK_BLOCKS = M.candidate_topk_blocks
CANDIDATE_PAIR_WIDTH = 2 * CANDIDATE_TOPK_BLOCKS
TOPK_CACHED_PAGES = (M.max_position_embeddings + 127) // 128


rotate_index_query = make_pipelined_rope(
    T_DYN, INDEX_H, INDEX_DIM, inverse=False, name_hint="prefill_index_rope", allow_early_resolve=False, inplace=True,
)

INDEX_SCORE_SCALE = INDEX_DIM ** -0.5 * INDEX_H ** -0.5
INDEX_MAX_LOGITS_BYTES = 64 * 1024 * 1024
TOPK_LEAF = 8192
TOPK_PAIR_WIDTH = 2 * INDEX_TOPK
TOPK_GROUP_LEAVES = 2
TOPK_ORDER_BITS = 14
TOPK_ORDER_WIDTH = 1 << TOPK_ORDER_BITS
IDX_SCALES = INDEX_DIM // 32
IDX_PACKED = INDEX_DIM // 2
LEAF = 2048
SORT_FLOOR = -3.0e38

# tiling
M_TILE = 16
INDEX_PROJ_N_TILE = 128
INDEX_PROJ_K_TILE = 256
INDEX_POOL_QUERY_TILE = 128
INDEX_SCORE_WORKERS = 24
INDEX_SCORE_QUERY_TILE = 2


CANDIDATE_SCORE_TILE = 1024
CANDIDATE_LEAF_TILE = 8192
CANDIDATE_SHORT_LEAF_TILE = 2048
CANDIDATE_MAX_LEAVES = (M.max_position_embeddings // CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_LEAF_TILE + 1
INDEX_SCORE_POSITIONS_DYN = pl.dynamic("V41_INDEX_SCORE_POSITIONS_DYN")


@pl.jit.inline
def publish_index_cache(
    value: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    scales: pl.Tensor[
        [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP],
        pl.UINT8,
    ],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    """Publish nearest-even group-32 E2M1 keys with upward E8M0 scales."""
    cache_rows = pl.tensor.dim(cache, 0) * 128
    cache_flat = pl.reshape(cache, [cache_rows, INDEX_DIM // 2])
    scale_flat = pl.reshape(scales, [cache_rows, INDEX_DIM // INDEX_CACHE_GROUP])
    with pl.spmd(num_tokens, name_hint="prefill_index_publish", deps=[cache_ready]) as publish_tid:
        token = pl.tile.get_block_idx()
        slot_i64 = pl.read(slots, [token])
        if slot_i64 >= 0:
            slot = pl.cast(slot_i64, pl.INDEX)
            source = pl.cast(pl.load(value, [token, 0], [1, INDEX_DIM]), pl.FP32)
            source_groups = pl.reshape(source, [INDEX_DIM // INDEX_CACHE_GROUP, INDEX_CACHE_GROUP])
            grouped = pl.tile.full([8, INDEX_CACHE_GROUP], dtype=pl.FP32, value=0.0)
            grouped[:INDEX_DIM // INDEX_CACHE_GROUP, :] = source_groups
            maximum_tmp = pl.create_tile([8, 128], dtype=pl.FP32)
            maximum = pl.row_max(pl.abs(grouped), tmp_tile=maximum_tmp)
            raw_scale = pl.maximum(pl.mul(maximum, 1.0 / 6.0), 2.0**-126)
            bits = pl.reinterpret_view(raw_scale, pl.INT32)
            exponent = pl.shrs(pl.add(bits, 8388607), 23)
            scale = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
            normalized = pl.reshape(pl.row_expand_div(grouped, scale), [1, 8 * INDEX_CACHE_GROUP])
            padded = pl.tile.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0)
            padded[:, :8 * INDEX_CACHE_GROUP] = normalized
            magnitude = pl.minimum(pl.abs(padded), 6.0)
            delta0 = pl.reinterpret_view(pl.sub(magnitude, 0.25), pl.INT32)
            delta1 = pl.reinterpret_view(pl.sub(magnitude, 0.75), pl.INT32)
            delta2 = pl.reinterpret_view(pl.sub(magnitude, 1.25), pl.INT32)
            delta3 = pl.reinterpret_view(pl.sub(magnitude, 1.75), pl.INT32)
            delta4 = pl.reinterpret_view(pl.sub(magnitude, 2.5), pl.INT32)
            delta5 = pl.reinterpret_view(pl.sub(magnitude, 3.5), pl.INT32)
            delta6 = pl.reinterpret_view(pl.sub(magnitude, 5.0), pl.INT32)
            step0 = pl.shrs(pl.sub(delta0, 1), 31)
            step1 = pl.shrs(delta1, 31)
            step2 = pl.shrs(pl.sub(delta2, 1), 31)
            step3 = pl.shrs(delta3, 31)
            step4 = pl.shrs(pl.sub(delta4, 1), 31)
            step5 = pl.shrs(delta5, 31)
            step6 = pl.shrs(pl.sub(delta6, 1), 31)
            lower = pl.add(pl.add(step0, step1), pl.add(step2, step3))
            upper = pl.add(pl.add(step4, step5), pl.add(step6, 7))
            unsigned = pl.add(lower, upper)
            sign = pl.mul(pl.shrs(pl.reinterpret_view(padded, pl.INT32), 31), -8)
            payload_codes = pl.add(unsigned, sign)
            pair_ids = pl.tile.arange(0, [1, HEAD_DIM // 2], dtype=pl.INT32)
            low_indices = pl.mul(pair_ids, 2)
            high_indices = pl.add(low_indices, 1)
            low_tmp = pl.create_tile([1, HEAD_DIM // 2], dtype=pl.INT32)
            high_tmp = pl.create_tile([1, HEAD_DIM // 2], dtype=pl.INT32)
            low = pl.tile.gather(payload_codes, low_indices, low_tmp)
            high = pl.tile.gather(payload_codes, high_indices, high_tmp)
            packed = pl.reshape(
                pl.cast(pl.add(low, pl.shls(high, 4)), pl.UINT8),
                [1, HEAD_DIM // 2],
            )
            payload_bytes = pl.tile.slice(packed, [1, INDEX_DIM // 2], [0, 0])
            pl.store(payload_bytes, [slot, 0], cache_flat)
            exponent_row = pl.reshape(exponent, [1, 8])
            exponent_padded = pl.tile.full([1, 32], dtype=pl.INT32, value=0)
            exponent_padded[:, :8] = exponent_row
            signed_exponent = pl.sub(
                exponent_padded,
                pl.mul(pl.shrs(exponent_padded, 7), 256),
            )
            codes = pl.reinterpret_view(pl.cast(signed_exponent, pl.INT8), pl.UINT8)
            encoded = codes
            encoded = pl.tile.set_validshape(
                encoded,
                1,
                INDEX_DIM // INDEX_CACHE_GROUP,
            )
            pl.store(
                encoded,
                [slot, 0],
                scale_flat,
            )
    return publish_tid


@pl.jit.inline(auto_scope=False)
def quantize_index_query(
    source: pl.Tensor[[T_DYN, INDEX_QUERY_WIDTH], pl.BF16],
    output: pl.Tensor[[T_DYN, INDEX_QUERY_WIDTH], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Round group-32 queries to E2M1/E8M0 and materialize them in BF16."""
    # Materialize the producer dependency before entering the scoped task.
    entry_deps = pl.array.create(1, pl.TASK_ID)
    entry_deps[0] = ready
    workers = pl.min(48, num_tokens)
    with pl.spmd(workers, name_hint="prefill_index_query_fp4", deps=[entry_deps[0]]) as quant_tid:
        worker = pl.tile.get_block_idx()
        for t in pl.range(worker, num_tokens, workers):
            raw = pl.load(source, [t, 0], [1, INDEX_QUERY_WIDTH])
            values = pl.reshape(pl.cast(raw, pl.FP32), [INDEX_QUERY_SCALE_WIDTH, 32])
            reduce_tmp = pl.tile.create([INDEX_QUERY_SCALE_WIDTH, 32], dtype=pl.FP32)
            raw_maximum = pl.row_max(pl.abs(values), tmp_tile=reduce_tmp)
            maximum = pl.maximum(raw_maximum, 7.052966104933725e-38)
            bits = pl.reinterpret_view(pl.mul(maximum, 1.0 / 6.0), pl.INT32)
            exponent = pl.shrs(pl.add(bits, 8388607), 23)
            scale = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
            inverse_exponent = pl.add(pl.mul(exponent, -1), 254)
            reciprocal = pl.reinterpret_view(pl.shls(inverse_exponent, 23), pl.FP32)
            normalized = pl.row_expand_mul(values, reciprocal)
            magnitude = pl.minimum(pl.abs(normalized), 6.0)
            # FP32 ULPs at 2**22, 2**23 and 2**24 are the three E2M1 spacings.
            small_bias = pl.tile.full([INDEX_QUERY_SCALE_WIDTH, 32], dtype=pl.FP32, value=4194304.0)
            middle = pl.tile.cmps(magnitude, 2.0, cmp_type=4)
            bias = pl.tile.select(middle, 8388608.0, small_bias)
            large = pl.tile.cmps(magnitude, 4.0, cmp_type=4)
            bias = pl.tile.select(large, 16777216.0, bias)
            biased = pl.add(magnitude, bias)
            rounded = pl.sub(biased, bias)
            sign = pl.shrs(pl.reinterpret_view(values, pl.INT32), 31)
            signed_bits = pl.add(pl.reinterpret_view(rounded, pl.INT32), pl.shls(sign, 31))
            exact = pl.reinterpret_view(signed_bits, pl.FP32)
            dequantized = pl.row_expand_mul(exact, scale)
            stored_values = pl.reshape(dequantized, [1, INDEX_QUERY_WIDTH])
            bf16_result = pl.cast(stored_values, pl.BF16, mode="rint")
            pl.store(bf16_result, [t, 0], output)
    return quant_tid


@pl.jit.inline
def materialize_index_topk(
    scores: pl.Tensor,
    block_table: pl.Tensor,
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    pool_ids: pl.Tensor,
    output: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    token: pl.Scalar[pl.INDEX],
    width: pl.Scalar[pl.INDEX],
    cutoff: pl.Scalar[pl.FP32],
    tie_quota: pl.Scalar[pl.INDEX],
    pool: pl.constexpr,
):
    """Compact winners in logical order and gather physical pages from UB."""
    request = pl.cast(pl.read(request_ids, [token]), pl.INDEX)
    table_width = pl.tensor.dim(block_table, 1)
    valid_pages = pl.min(table_width, TOPK_CACHED_PAGES)
    raw_pages = pl.load(block_table, [request, 0], [1, TOPK_CACHED_PAGES], valid_shape=[1, valid_pages])
    pages = pl.tile.fillpad(raw_pages, pad_value=pl.PadValue.zero)
    empty_blocks = pl.tile.full([1, CANDIDATE_TOPK_BLOCKS], dtype=pl.INT32, value=0)
    pool_blocks = pl.tile.fillpad(empty_blocks, pad_value=pl.PadValue.zero)
    if pool:
        pool_width = pl.tensor.dim(pool_ids, 1)
        raw_blocks = pl.load(pool_ids, [token, 0], [1, CANDIDATE_TOPK_BLOCKS], valid_shape=[1, pool_width])
        pool_blocks = pl.tile.fillpad(raw_blocks, pad_value=pl.PadValue.zero)
    pl.store(pl.tile.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1), [token, 0], output)
    ones = pl.tile.full([1, 256], dtype=pl.FP32, value=1.0)
    lanes = pl.tile.arange(0, [1, 256], dtype=pl.INT32)
    lane_origin = pl.tile.full([1, 256], dtype=pl.INT32, value=256)
    # Unique integer lane keys: selected=256-lane, discarded=0.
    priority_fp32 = pl.cast(pl.sub(lane_origin, lanes), pl.FP32)
    tmp = pl.tile.create([1, 256], dtype=pl.INT32)
    count = pl.cast(0, pl.INDEX)
    remaining = tie_quota
    for start in pl.range(0, width, 256):
        if count < INDEX_TOPK:
            valid = pl.min(256, width - start)
            raw = pl.load(scores, [token, start], [1, 256], valid_shape=[1, valid])
            values = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
            strict = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=4), ones, 0.0)
            ties = pl.tile.full([1, 256], dtype=pl.FP32, value=0.0)
            if remaining > 0:
                eq = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=0), ones, 0.0)
                eq_valid = pl.tile.cmps(pl.cast(lanes, pl.FP32), pl.cast(pl.cast(valid, pl.INT32), pl.FP32), cmp_type=2)
                eq = pl.tile.select(eq_valid, eq, 0.0)
                eq_sums64 = pl.col_sum(pl.reshape(eq, [4, 64]))
                eq_sums8 = pl.col_sum(pl.reshape(eq_sums64, [8, 8]))
                available = pl.cast(0, pl.INDEX)
                for eq_lane in pl.unroll(8):
                    eq_partial = pl.cast(pl.tile.read(eq_sums8, [0, eq_lane]), pl.INT32)
                    available = available + pl.cast(eq_partial, pl.INDEX)
                take = pl.min(available, remaining)
                if take > 0:
                    if take == available:
                        ties = eq
                    else:
                        # Only the chunk crossing the tie quota needs this boundary.
                        tie_keys = pl.mul(eq, priority_fp32)
                        tie_pairs = pl.tile.sort32(tie_keys, pl.reinterpret_view(lanes, pl.UINT32))
                        tie_pairs = pl.tile.mrgsort(tie_pairs, block_len=64)
                        tie_left = pl.tile.slice(tie_pairs, [1, 256], [0, 0])
                        tie_right = pl.tile.slice(tie_pairs, [1, 256], [0, 256])
                        tie_tmp = pl.tile.create([1, 512], dtype=pl.FP32)
                        tie_merged = pl.tile.mrgsort(tie_left, tie_right, tmp=tie_tmp)
                        equal_keys = pl.tile.gather_mask(tie_merged, mask_pattern=pl.tile.MaskPattern.P0101)
                        last_key = pl.cast(pl.tile.read(equal_keys, [0, take - 1]), pl.INT32)
                        boundary = pl.cast(256, pl.INT32) - last_key
                        ties = pl.tile.select(pl.tile.cmps(pl.cast(lanes, pl.FP32), pl.cast(pl.cast(boundary, pl.INT32), pl.FP32), cmp_type=3), eq, 0.0)
                    remaining = remaining - take
            keep = pl.maximum(strict, ties)
            keep_valid = pl.tile.set_validshape(keep, 1, valid)
            keep_padded = pl.tile.fillpad(keep_valid, pad_value=pl.PadValue.zero)
            keep_sums64 = pl.col_sum(pl.reshape(keep_padded, [4, 64]))
            keep_sums8 = pl.col_sum(pl.reshape(keep_sums64, [8, 8]))
            keep_count = pl.cast(0, pl.INDEX)
            for keep_lane in pl.unroll(8):
                keep_partial = pl.cast(pl.tile.read(keep_sums8, [0, keep_lane]), pl.INT32)
                keep_count = keep_count + pl.cast(keep_partial, pl.INDEX)
            selected_count = pl.min(keep_count, INDEX_TOPK - count)
            if selected_count > 0:
                order_keys = pl.mul(keep_padded, priority_fp32)
                order_pairs = pl.tile.sort32(order_keys, pl.reinterpret_view(lanes, pl.UINT32))
                order_pairs = pl.tile.mrgsort(order_pairs, block_len=64)
                order_left = pl.tile.slice(order_pairs, [1, 256], [0, 0])
                order_right = pl.tile.slice(order_pairs, [1, 256], [0, 256])
                order_tmp = pl.tile.create([1, 512], dtype=pl.FP32)
                order_merged = pl.tile.mrgsort(order_left, order_right, tmp=order_tmp)
                order_keys = pl.tile.gather_mask(order_merged, mask_pattern=pl.tile.MaskPattern.P0101)
                offsets_i32 = pl.sub(lane_origin, pl.cast(order_keys, pl.INT32))
                compact_valid = pl.tile.set_validshape(offsets_i32, 1, selected_count)
                compact_safe = pl.tile.fillpad(compact_valid, pad_value=pl.PadValue.zero)
                logical = pl.add(compact_safe, pl.cast(start, pl.INT32))
                if pool:
                    block_indices = pl.tile.shrs(logical, 3)
                    blocks = pl.tile.gather(pool_blocks, block_indices, tmp)
                    logical = pl.add(pl.mul(blocks, CANDIDATE_BLOCK_SIZE), pl.sub(logical, pl.shls(pl.shrs(logical, 3), 3)))
                page_indices = pl.tile.shrs(logical, 7)
                if start < TOPK_CACHED_PAGES * 128:
                    page_ids = pl.tile.gather(pages, page_indices, tmp)
                else:
                    # A scan chunk spans at most two pages; keep gathers within this bank.
                    first_page = start // 128
                    bank_valid = pl.min(8, table_width - first_page)
                    bank_raw = pl.load(block_table, [request, first_page], [1, 8], valid_shape=[1, bank_valid])
                    bank = pl.tile.fillpad(bank_raw, pad_value=pl.PadValue.zero)
                    local_pages = pl.sub(page_indices, pl.cast(first_page, pl.INT32))
                    page_ids = pl.tile.gather(bank, local_pages, tmp)
                offsets = pl.sub(logical, pl.mul(page_indices, 128))
                physical = pl.add(pl.mul(page_ids, 128), offsets)
                pl.store(pl.tile.set_validshape(physical, 1, selected_count), [token, count], output)
                count = count + selected_count
    return output


@pl.jit.inline
def _merge_candidate_pairs(
    arena: pl.Tensor,
    left: pl.Scalar[pl.INDEX],
    right: pl.Scalar[pl.INDEX],
    output: pl.Scalar[pl.INDEX],
):
    left_pairs = pl.load(arena, [left, 0], [1, CANDIDATE_PAIR_WIDTH])
    right_pairs = pl.load(arena, [right, 0], [1, CANDIDATE_PAIR_WIDTH])
    temporary = pl.tile.create([1, 2 * CANDIDATE_PAIR_WIDTH], dtype=pl.FP32)
    merged_all = pl.tile.mrgsort(left_pairs, right_pairs, tmp=temporary)
    merged = pl.tile.slice(merged_all, [1, CANDIDATE_PAIR_WIDTH], [0, 0])
    pl.store(merged, [output, 0], arena)


@pl.jit.inline
def _merge_candidate_level(
    arena: pl.Tensor,
    base: pl.Scalar[pl.INDEX],
    count: pl.Scalar[pl.INDEX],
):
    output_count = (count + 1) // 2
    for output in pl.range(output_count):
        left = base + 2 * output
        right = left + 1
        if right < base + count:
            _merge_candidate_pairs(arena, left, right, base + output)
        else:
            forwarded = pl.load(arena, [left, 0], [1, CANDIDATE_PAIR_WIDTH])
            pl.store(forwarded, [base + output, 0], arena)
    return output_count


@pl.jit.inline
def _sort_candidate_leaf(
    scores: pl.Tensor,
    arena: pl.Tensor,
    token: pl.Scalar[pl.INDEX],
    block_begin: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
):
    raw = pl.load(scores, [token, block_begin], [1, CANDIDATE_LEAF_TILE], valid_shape=[1, valid_count])
    padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
    floor = pl.tile.full([1, CANDIDATE_LEAF_TILE], dtype=pl.FP32, value=-1e30)
    values = pl.maximum(padded, floor)
    block_begin_i32 = pl.cast(block_begin, pl.INT32)
    indices = pl.add(pl.tile.arange(0, [1, CANDIDATE_LEAF_TILE], dtype=pl.INT32), block_begin_i32)
    pairs = pl.tile.sort32(values, pl.reinterpret_view(indices, pl.UINT32))
    pairs = pl.tile.mrgsort(pairs, block_len=64)
    pairs = pl.tile.mrgsort(pairs, block_len=256)
    pairs = pl.tile.mrgsort(pairs, block_len=1024)
    pairs = pl.tile.mrgsort(pairs, block_len=4096)
    top_pairs = pl.tile.slice(pairs, [1, CANDIDATE_PAIR_WIDTH], [0, 0])
    pl.store(top_pairs, [output_slot, 0], arena)


@pl.jit.inline
def _sort_candidate_short_leaf(
    scores: pl.Tensor,
    arena: pl.Tensor,
    token: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
):
    raw = pl.load(scores, [token, 0], [1, CANDIDATE_SHORT_LEAF_TILE], valid_shape=[1, valid_count])
    values = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
    values = pl.maximum(values, -1e30)
    indices = pl.tile.arange(0, [1, CANDIDATE_SHORT_LEAF_TILE], dtype=pl.INT32)
    pairs = pl.tile.sort32(values, pl.reinterpret_view(indices, pl.UINT32))
    pairs = pl.tile.mrgsort(pairs, block_len=64)
    pairs = pl.tile.mrgsort(pairs, block_len=256)
    pairs = pl.tile.mrgsort(pairs, block_len=1024)
    top_pairs = pl.tile.slice(pairs, [1, CANDIDATE_PAIR_WIDTH], [0, 0])
    pl.store(top_pairs, [output_slot, 0], arena)


@pl.jit.inline(auto_scope=False)
def _publish_all_candidate_blocks(lengths: pl.Tensor, block_ids: pl.Tensor, counts: pl.Tensor,
                                  num_tokens: pl.Scalar[pl.INT32], positions: pl.Scalar[pl.INDEX],
                                  ready: pl.Scalar[pl.TASK_ID]):
    """Publish a contiguous logical pool when every visible block fits."""
    tokens = pl.tensor.dim(lengths, 0)
    capacity = pl.tensor.dim(block_ids, 1)
    workers = pl.min(32, tokens)
    with pl.spmd(workers, name_hint="c1a_pool_all_blocks", deps=[ready]) as pool_ready:
        worker = pl.tile.get_block_idx()
        lanes = pl.tile.arange(0, [1, CANDIDATE_TOPK_BLOCKS], dtype=pl.INT32)
        for token in pl.range(worker, tokens, workers):
            count = pl.cast(0, pl.INT32)
            if token < num_tokens:
                visible = pl.min(pl.max(pl.read(lengths, [token]), 0), positions)
                count = pl.cast((visible + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE, pl.INT32)
            keep = pl.tile.cmps(pl.cast(lanes, pl.FP32), pl.cast(count, pl.FP32), cmp_type=2)
            selected = pl.tile.select(keep, lanes, -1)
            pl.store(pl.set_validshape(selected, 1, capacity), [token, 0], block_ids)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c1a_pool_counts", deps=[pool_ready]) as done:
        for token in pl.range(tokens):
            count = pl.cast(0, pl.INT32)
            if token < num_tokens:
                visible = pl.min(pl.max(pl.read(lengths, [token]), 0), positions)
                count = pl.cast((visible + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE, pl.INT32)
            pl.write(counts, [token], count)
    return done


@pl.jit.inline(auto_scope=False)
def select_candidate_block_pool(
    index_scores: pl.Tensor[[T_DYN, INDEX_SCORE_POSITIONS_DYN], pl.FP32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    scores_ready: pl.Scalar[pl.TASK_ID],
):
    """Publish sorted logical candidate block IDs and per-query counts."""
    tokens = pl.tensor.dim(index_scores, 0)
    positions = pl.tensor.dim(index_scores, 1)
    capacity = pl.tensor.dim(pool_block_ids, 1)
    block_count = (positions + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
    padded_block_count = (block_count + CANDIDATE_LEAF_TILE - 1) // CANDIDATE_LEAF_TILE * CANDIDATE_LEAF_TILE
    score_tiles = (block_count + CANDIDATE_SCORE_TILE - 1) // CANDIDATE_SCORE_TILE
    block_scores = pl.create_tensor([tokens, padded_block_count], dtype=pl.FP32)
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > tokens:
        active_tokens = pl.cast(tokens, pl.INDEX)
    # Dispatch one guarded work item for an empty batch.
    score_task_tokens = pl.max(active_tokens, pl.cast(1, pl.INDEX))
    score_task_tiles = pl.max(score_tiles, pl.cast(1, pl.INDEX))

    with pl.spmd(
        score_task_tokens * score_task_tiles,
        name_hint="c1a_pool_block_scores",
        deps=[scores_ready],
    ) as score_tid:
        work = pl.tile.get_block_idx()
        token = work // score_task_tiles
        score_tile = work % score_task_tiles
        if token < active_tokens and score_tile < score_tiles:
            block_begin = score_tile * CANDIDATE_SCORE_TILE
            position_begin = block_begin * CANDIDATE_BLOCK_SIZE
            valid_blocks = pl.min(CANDIDATE_SCORE_TILE, block_count - block_begin)
            valid_positions = pl.min(CANDIDATE_SCORE_TILE * CANDIDATE_BLOCK_SIZE, positions - position_begin)
            raw = pl.load(
                index_scores,
                [token, position_begin],
                [1, CANDIDATE_SCORE_TILE * CANDIDATE_BLOCK_SIZE],
                valid_shape=[1, valid_positions],
            )
            padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
            grouped = pl.reshape(padded, [CANDIDATE_SCORE_TILE, CANDIDATE_BLOCK_SIZE])
            reduce_tmp = pl.create_tile([CANDIDATE_SCORE_TILE, CANDIDATE_BLOCK_SIZE], dtype=pl.FP32)
            maxima = pl.reshape(pl.row_max(grouped, tmp_tile=reduce_tmp), [1, CANDIDATE_SCORE_TILE])
            visible = pl.min(pl.max(pl.read(compressed_lens, [token]), 0), positions)
            if visible > 0:
                last_block = (visible - 1) // CANDIDATE_BLOCK_SIZE
                if last_block >= block_begin and last_block < block_begin + valid_blocks:
                    pl.tile.write(maxima, [0, last_block - block_begin], 1e30)
            maxima = pl.tile.set_validshape(maxima, 1, valid_blocks)
            pl.store(maxima, [token, block_begin], block_scores)

    arena = pl.create_tensor([tokens * CANDIDATE_MAX_LEAVES, CANDIDATE_PAIR_WIDTH], dtype=pl.FP32)
    with pl.spmd(tokens, name_hint="c1a_pool_block_select", deps=[score_tid]) as pool_tid:
        token = pl.tile.get_block_idx()
        sorted_ids = pl.tile.full([1, CANDIDATE_TOPK_BLOCKS], dtype=pl.INT32, value=-1)
        if token < active_tokens:
            visible = pl.min(pl.max(pl.read(compressed_lens, [token]), 0), positions)
            visible_blocks = (visible + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
            arena_base = token * CANDIDATE_MAX_LEAVES
            count = 0
            if visible_blocks > 0 and visible_blocks <= CANDIDATE_SHORT_LEAF_TILE:
                _sort_candidate_short_leaf(block_scores, arena, token, visible_blocks, arena_base)
                count = 1
            elif visible_blocks > 0:
                leaf_count = (visible_blocks + CANDIDATE_LEAF_TILE - 1) // CANDIDATE_LEAF_TILE
                for leaf in pl.range(leaf_count):
                    block_begin = leaf * CANDIDATE_LEAF_TILE
                    valid_count = pl.min(CANDIDATE_LEAF_TILE, visible_blocks - block_begin)
                    _sort_candidate_leaf(block_scores, arena, token, block_begin, valid_count, arena_base + leaf)
                count = leaf_count
            if count > 1:
                count = _merge_candidate_level(arena, arena_base, count)
            if count > 1:
                count = _merge_candidate_level(arena, arena_base, count)
            if count > 1:
                count = _merge_candidate_level(arena, arena_base, count)
            if count > 1:
                count = _merge_candidate_level(arena, arena_base, count)

            selected_count = pl.min(pl.min(visible_blocks, CANDIDATE_TOPK_BLOCKS), capacity)
            ordered_scores = pl.tile.full([1, CANDIDATE_TOPK_BLOCKS], dtype=pl.FP32, value=-1e30)
            ordered_indices = pl.tile.full([1, CANDIDATE_TOPK_BLOCKS], dtype=pl.INT32, value=-1)
            if selected_count > 0:
                root = pl.load(arena, [arena_base, 0], [1, CANDIDATE_PAIR_WIDTH])
                selected = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                for lane in pl.range(selected_count):
                    block_id = pl.tile.read(selected, [0, lane])
                    pl.tile.write(ordered_scores, [0, lane], -pl.cast(pl.cast(block_id, pl.INT32), pl.FP32))
                    pl.tile.write(ordered_indices, [0, lane], block_id)
            ordered = pl.tile.sort32(ordered_scores, pl.reinterpret_view(ordered_indices, pl.UINT32))
            ordered = pl.tile.mrgsort(ordered, block_len=64)
            ordered = pl.tile.mrgsort(ordered, block_len=256)
            ordered = pl.tile.mrgsort(ordered, block_len=1024)
            sorted_ids = pl.tile.gather_mask(ordered, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        # Publish the sorted prefix and -1 padding as one aligned pool row.
        pl.store(pl.tile.set_validshape(sorted_ids, 1, capacity), [token, 0], pool_block_ids)
    # Publish cache-line-sharing counts from one worker after the ID rows.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c1a_pool_count_publish", deps=[pool_tid]) as counts_tid:
        for publish_token in pl.range(tokens):
            published_count = pl.cast(0, pl.INT32)
            if publish_token < active_tokens:
                publish_visible = pl.min(pl.max(pl.read(compressed_lens, [publish_token]), 0), positions)
                publish_blocks = (publish_visible + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
                published_count = pl.cast(pl.min(pl.min(publish_blocks, CANDIDATE_TOPK_BLOCKS), capacity), pl.INT32)
            pl.write(pool_block_counts, [publish_token], published_count)
    return counts_tid


@pl.jit.inline
def _index_rope(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    heads: pl.constexpr,
    max_workers: pl.constexpr,
):
    """Rotate adjacent pairs in trailing lanes of INDEX_DIM-wide heads."""
    blocks = num_tokens * heads
    workers = pl.min(blocks, max_workers)
    for worker in pl.spmd(workers, name_hint="prefill_index_rope"):
        for block in pl.range(worker, blocks, workers):
            token = block // heads
            head = block % heads
            base = head * INDEX_DIM
            rope_start = base + INDEX_DIM - ROPE_DIM
            nope = pl.slice(x, [1, INDEX_DIM - ROPE_DIM], [token, base])
            output[token:token + 1, base:base + INDEX_DIM - ROPE_DIM] = nope
            tail = pl.cast(pl.slice(x, [1, ROPE_DIM], [token, rope_start]), pl.FP32)
            even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            c = cos[token:token + 1, :]
            sine = pl.mul(sin[token:token + 1, :], 1.0)
            re = pl.sub(pl.mul(even, c), pl.mul(odd, sine))
            im = pl.add(pl.mul(even, sine), pl.mul(odd, c))
            rotated = pl.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
            rotated = pl.tensor.scatter(re, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
            rotated = pl.tensor.scatter(im, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
            output[token:token + 1, rope_start:rope_start + ROPE_DIM] = pl.cast(rotated, pl.BF16, mode="rint")
    return output


@pl.jit.inline
def project_index_weights(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D, INDEX_H], pl.BF16, pl.ND],
    output: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Project BF16 per-head weights with BF16 rounding before and after scaling."""
    tokens = pl.tensor.dim(x, 0)
    projected_weights = pl.create_tensor([tokens, INDEX_H], dtype=pl.BF16)
    ready = pl.system.task_dummy(deps=[])
    projected_result, projected_tid = gemm_bf16_to_bf16(
        x, weight, projected_weights, num_tokens, ready,
        width=D, output_width=INDEX_H, n_tile=INDEX_H, k_tile=INDEX_PROJ_K_TILE,
        workers=2147483647, name_hint="prefill_index_weights",
        source_col=0, output_col=0, ordered=False, row_tile=16, stages=1,
    )
    with pl.spmd((num_tokens + M_TILE - 1) // M_TILE, name_hint="c1a_index_weights_scale") as weights_tid:
        block = pl.tile.get_block_idx()
        token = block * M_TILE
        rows = pl.min(M_TILE, num_tokens - token)
        projected = pl.slice(projected_weights, [M_TILE, INDEX_H], [token, 0], valid_shape=[rows, INDEX_H])
        scaled = pl.mul(pl.cast(projected, pl.FP32), INDEX_SCORE_SCALE)
        rounded = pl.cast(scaled, pl.BF16, mode="rint")
        output[token:token + M_TILE, :] = pl.set_validshape(rounded, rows, INDEX_H)
    return weights_tid


@pl.jit.inline(auto_scope=False)
def prepare_prefill_index_query(
    x: pl.Tensor,
    latent: pl.Tensor,
    query_weight: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.INT8],
    query_weight_scale: pl.Tensor,
    weight_projection: pl.Tensor[[D, INDEX_H], pl.BF16, pl.ND],
    cos: pl.Tensor,
    sin: pl.Tensor,
    query: pl.Tensor,
    weights: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Project, rotate, and quantize query rows with group-32 scales."""
    tokens = pl.tensor.dim(x, 0)
    projected = pl.create_tensor([tokens, INDEX_H * INDEX_DIM], dtype=pl.BF16)
    gemm_dynamic_int8(
        latent, query_weight, query_weight_scale, projected, num_tokens,
        Q_LORA, INDEX_H * INDEX_DIM, "c1a_mx_projection", False, True,
    )
    if num_tokens >= 64:
        rotated = rotate_index_query(projected, cos, sin, projected, num_tokens)
    else:
        rotated = _rotate_rope(projected, projected, cos, sin, 0, num_tokens, INDEX_H, INDEX_DIM, "prefill_index_rope", False, False)
    query_tid = quantize_index_query(rotated, query, num_tokens, ready)
    weights_tid = project_index_weights(x, weight_projection, weights, num_tokens)
    return query_tid, weights_tid


@pl.jit.inline(auto_scope=False)
def _decode_index_pages(cache: pl.Tensor, cache_scale: pl.Tensor, ready: pl.Scalar[pl.TASK_ID]):
    """Decode each physical key once before scoring all queries in this layer."""
    pages = pl.tensor.dim(cache, 0)
    rows = pages * 128
    packed = pl.reshape(cache, [rows, INDEX_DIM // 2])
    scales = pl.reshape(cache_scale, [1, rows * IDX_SCALES])
    decoded = pl.create_tensor([rows, INDEX_DIM], dtype=pl.BF16)
    workers = pl.min(32, rows // 32)
    entry_deps = pl.array.create(1, pl.TASK_ID)
    entry_deps[0] = ready
    with pl.spmd(workers, name_hint="prefill_index_decode_pages", deps=[entry_deps[0]]) as decoded_tid:
        worker = pl.tile.get_block_idx()
        for panel in pl.range(worker, rows // 32, workers):
            row = panel * 32
            codes = pl.load(packed, [row, 0], [32, INDEX_DIM // 2])
            raw_scales = pl.load(scales, [0, row * IDX_SCALES], [1, 32 * IDX_SCALES])
            magnitudes = decode_fp4(codes, 32, INDEX_DIM)
            groups = pl.reshape(pl.cast(magnitudes, pl.FP32), [32 * IDX_SCALES, 32])
            factors = pl.reshape(decode_scale(raw_scales, 1, 32 * IDX_SCALES), [32 * IDX_SCALES, 1])
            scaled = pl.row_expand_mul(groups, factors)
            values = pl.reshape(pl.cast(scaled, pl.BF16, mode="rint"), [32, INDEX_DIM])
            pl.store(values, [row, 0], decoded)
    return decoded, decoded_tid


@pl.jit.inline(auto_scope=False)
def _order_index_pages(
    decoded: pl.Tensor, table: pl.Tensor, ordered: pl.Tensor, ready: pl.Scalar[pl.TASK_ID],
):
    """Gather one request's key pages into a contiguous two-page layout."""
    pages = pl.tensor.dim(table, 1)
    physical_pages = pl.tensor.dim(decoded, 0) // 128
    entries = pl.array.create(1, pl.TASK_ID)
    entries[0] = ready
    with pl.spmd(32, name_hint="prefill_index_order_pages", deps=[entries[0]]) as ordered_tid:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, (pages + 1) // 2 * 8, 32):
            page = block // 4
            physical = pl.cast(0, pl.INT32)
            if page < pages:
                physical = pl.read(table, [0, page])
            physical = pl.min(physical_pages - 1, pl.max(physical, 0))
            row = (block % 4) * 32
            keys = pl.load(decoded, [physical * 128 + row, 0], [32, INDEX_DIM])
            pl.store(keys, [block * 32, 0], ordered)
    return ordered_tid


@pl.jit.inline(auto_scope=False)
def _compact_pool_cache(
    decoded_keys: pl.Tensor,
    table: pl.Tensor,
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    lengths: pl.Tensor[[T_DYN], pl.INT32],
    pool_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_counts: pl.Tensor[[T_DYN], pl.INT32],
    compact: pl.Tensor,
    compact_table: pl.Tensor,
    compact_requests: pl.Tensor[[T_DYN], pl.INT32],
    compact_lengths: pl.Tensor[[T_DYN], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Gather eight-key candidate blocks into per-query contiguous GM pages."""
    capacity = pl.tensor.dim(pool_ids, 1)
    pages = capacity // 16
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_pool_plan", deps=[ready]) as planned:
        for token in pl.range(num_tokens):
            count = pl.min(capacity, pl.max(0, pl.read(pool_counts, [token])))
            visible = pl.max(0, pl.read(lengths, [token]))
            retained = pl.cast(pl.max(0, count * 8 - (8 - visible % 8) % 8), pl.INT32)
            pl.write(compact_lengths, [token], retained)
            pl.write(compact_requests, [token], pl.cast(token, pl.INT32))
            for page in pl.range(pages):
                physical = pl.cast(token * pages + page, pl.INT32)
                pl.write(compact_table, [token, page], physical)
    cache_pages = pl.tensor.dim(decoded_keys, 0) // 128
    with pl.spmd(32, name_hint="prefill_pool_gather", deps=[planned]) as gathered:
        worker = pl.tile.get_block_idx()
        for work in pl.range(worker, num_tokens * capacity, 32):
            token = work // capacity
            slot = work % capacity
            values = pl.tile.full([8, INDEX_DIM], dtype=pl.BF16, value=0.0)
            count = pl.min(capacity, pl.max(0, pl.read(pool_counts, [token])))
            if slot < count:
                logical = pl.read(pool_ids, [token, slot]) * 8
                request = pl.read(request_ids, [token])
                if logical >= 0 and logical < pl.read(lengths, [token]):
                    physical_page = pl.read(table, [request, logical // 128])
                    if physical_page >= 0 and physical_page < cache_pages:
                        physical_row = physical_page * 128 + logical % 128
                        values = pl.load(decoded_keys, [physical_row, 0], [8, INDEX_DIM])
            pl.store(values, [work * 8, 0], compact)
    return gathered


@pl.jit.inline(auto_scope=False)
def _prefill_index_scores(
    query: pl.Tensor,
    weights: pl.Tensor,
    decoded_keys: pl.Tensor,
    block_table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
    tile_n: pl.constexpr,
    query_tile: pl.constexpr,
    skip_short: pl.constexpr,
    head_rows: pl.constexpr,
    materialize_padding: pl.constexpr,
    pipe_slots: pl.constexpr,
    ordered_keys: pl.constexpr,
    physical_pages: pl.Scalar[pl.INDEX],
):
    """Score contiguous paged MX index keys."""
    groups = pl.cast((num_tokens + query_tile - 1) // query_tile, pl.INDEX)
    width = pl.tensor.dim(output, 1)
    prefix_rows = groups + 1
    prefix = pl.create_tensor([prefix_rows], dtype=pl.INT32)
    table_width = pl.tensor.dim(block_table, 1)
    cache_pages = pl.tensor.dim(decoded_keys, 0) // 128
    if ordered_keys:
        cache_pages = physical_pages
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_index_plan", deps=[ready]) as plan_tid:
        total = pl.cast(0, pl.INDEX)
        pl.write(prefix, [0], pl.cast(total, pl.INT32))
        for group in pl.range(groups):
            visible = pl.cast(0, pl.INDEX)
            for row in pl.unroll(query_tile):
                token = group * query_tile + row
                if token < num_tokens:
                    length = pl.cast(pl.max(0, pl.read(lengths, [token])), pl.INDEX)
                    if skip_short:
                        if length <= INDEX_TOPK:
                            length = pl.cast(0, pl.INDEX)
                    visible = pl.max(visible, pl.min(length, width))
            total = total + (visible + tile_n - 1) // tile_n
            pl.write(prefix, [group + 1], pl.cast(total, pl.INT32))

    clear_deps = pl.array.create(1, pl.TASK_ID)
    clear_deps[0] = plan_tid
    if materialize_padding and not skip_short:
        # Scores beyond the last scheduled panel are observable in the dense score ABI.
        with pl.spmd(32, name_hint="prefill_index_score_clear", deps=[plan_tid]) as clear_tid:
            worker = pl.tile.get_block_idx()
            floor_bits = pl.tile.full([1, 128], dtype=pl.INT32, value=-8388608)
            floor = pl.reinterpret_view(floor_bits, pl.FP32)
            for token in pl.range(worker, num_tokens, 32):
                group = token // query_tile
                panels = pl.read(prefix, [group + 1]) - pl.read(prefix, [group])
                for begin in pl.range(panels * tile_n, width, 128):
                    pl.store(pl.set_validshape(floor, 1, pl.min(128, width - begin)), [token, begin], output)
        clear_deps[0] = clear_tid

    query_rows = pl.tensor.dim(query, 0) * INDEX_H
    query_bf16 = pl.reshape(query, [query_rows, INDEX_DIM])
    requests = pl.tensor.dim(block_table, 0)
    lane_cols = tile_n // 2
    with pl.spmd(INDEX_SCORE_WORKERS, name_hint="prefill_index_mx_scores", deps=[clear_deps[0]],
                 optimizations=[pl.cross_core_slot(slot_num=pipe_slots)]) as score_tid:
        worker = pl.tile.get_block_idx()
        active_groups = pl.max(groups, 1)
        first_group = worker % active_groups
        panel_lane = worker // active_groups
        panel_stride = (INDEX_SCORE_WORKERS + active_groups - 1 - first_group) // active_groups
        for _ in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
            lanes = pl.cast(pl.tile.arange(0, [1, tile_n // 2], dtype=pl.INT32), pl.FP32)
            invalid_bits = pl.tile.full([1, tile_n // 2], dtype=pl.INT32, value=-8388608)
            invalid = pl.reinterpret_view(invalid_bits, pl.FP32)
        for group in pl.range(first_group, groups, INDEX_SCORE_WORKERS):
            panels = pl.read(prefix, [group + 1]) - pl.read(prefix, [group])
            if panel_lane < panels:
                token = group * query_tile
                request = pl.cast(0, pl.INT32)
                if query_tile == 1:
                    request = pl.read(request_ids, [token])
                safe_request = pl.cast(pl.min(requests - 1, pl.max(request, 0)), pl.INDEX)
                live = pl.min(query_tile, num_tokens - token)
                query_panel = pl.load(query_bf16, [token * INDEX_H, 0], [head_rows, INDEX_DIM],
                                      target_memory=pl.MemorySpace.Mat)
                for _ in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
                    weight_rows = pl.load(weights, [token, 0], [query_tile, INDEX_H], valid_shape=[live, INDEX_H])
                    weight_rows = pl.tile.fillpad(weight_rows, pad_value=pl.PadValue.zero)
                    head_weights = pl.reshape(pl.cast(weight_rows, pl.FP32), [head_rows, 1])
                for macro in pl.range(panel_lane, panels, panel_stride):
                    begin = macro * tile_n
                    first_page = begin // 128
                    if ordered_keys:
                        key_panel = pl.load(decoded_keys, [begin, 0], [tile_n, INDEX_DIM],
                                            target_memory=pl.MemorySpace.Mat)
                    elif tile_n == 128:
                        page = pl.read(block_table, [safe_request, pl.min(table_width - 1, first_page)])
                        physical = pl.min(cache_pages - 1, pl.max(page, 0))
                        key_panel = pl.load(decoded_keys, [physical * 128, 0], [tile_n, INDEX_DIM],
                                            target_memory=pl.MemorySpace.Mat)
                    else:
                        key_panel = pl.create_tile([tile_n, INDEX_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Mat)
                        for part in pl.unroll(tile_n // 128):
                            safe_page = pl.min(table_width - 1, first_page + part)
                            part_page = pl.read(block_table, [safe_request, safe_page])
                            part_row = pl.min(cache_pages - 1, pl.max(part_page, 0)) * 128
                            key_panel = pl.gather_row(
                                key_panel, decoded_keys, [part * 128, 0], [part_row, 0], [128, INDEX_DIM],
                            )
                    product = pl.matmul(query_panel, pl.tile.transpose_view(key_panel), out_dtype=pl.FP32)
                    for lane in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
                        dot = pl.aiv_shard(product)
                        lane_begin = begin + lane * lane_cols
                        logical_page = lane_begin // 128
                        safe_page_slot = pl.min(table_width - 1, logical_page)
                        output_page = pl.read(block_table, [safe_request, safe_page_slot])
                        rounded = pl.cast(dot, pl.BF16, mode="rint")
                        relu = pl.maximum(pl.cast(rounded, pl.FP32), 0.0)
                        weighted = pl.row_expand_mul(relu, head_weights)
                        weighted = pl.cast(pl.cast(weighted, pl.BF16, mode="rint"), pl.FP32)
                        for row in pl.unroll(query_tile):
                            if token + row < num_tokens and lane_begin < width:
                                heads = pl.tile.slice(weighted, [INDEX_H, tile_n // 2], [row * INDEX_H, 0])
                                sums = pl.cast(pl.col_sum(heads), pl.BF16, mode="rint")
                                reduced_scores = pl.cast(sums, pl.FP32)
                                masked = invalid
                                if request >= 0 and request < requests and logical_page < table_width:
                                    if output_page >= 0 and output_page < cache_pages:
                                        if ordered_keys:
                                            masked = reduced_scores
                                        else:
                                            row_length = pl.read(lengths, [token + row])
                                            if query_tile > 1 and pl.read(request_ids, [token + row]) != 0:
                                                row_length = pl.cast(0, pl.INT32)
                                            if skip_short and row_length <= INDEX_TOPK:
                                                row_length = pl.cast(0, pl.INT32)
                                            edge = pl.cast(pl.cast(row_length - lane_begin, pl.INT32), pl.FP32)
                                            visible_mask = pl.tile.cmps(lanes, edge, cmp_type=2)
                                            masked = pl.tile.select(visible_mask, reduced_scores, invalid)
                                cols = pl.min(lane_cols, width - lane_begin)
                                pl.store(pl.set_validshape(masked, 1, cols), [token + row, lane_begin], output)
    return score_tid


@pl.jit.inline
def _merge_topk_pairs(
    arena: pl.Tensor,
    left: pl.Scalar[pl.INDEX],
    right: pl.Scalar[pl.INDEX],
    output: pl.Scalar[pl.INDEX],
):
    left_pairs = pl.load(arena, [left, 0], [1, TOPK_PAIR_WIDTH])
    right_pairs = pl.load(arena, [right, 0], [1, TOPK_PAIR_WIDTH])
    temporary = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
    merged_all = pl.tile.mrgsort(left_pairs, right_pairs, tmp=temporary)
    merged = pl.tile.slice(merged_all, [1, TOPK_PAIR_WIDTH], [0, 0])
    pl.store(merged, [output, 0], arena)


@pl.jit.inline
def _merge_topk_level(
    arena: pl.Tensor,
    base: pl.Scalar[pl.INDEX],
    count: pl.Scalar[pl.INDEX],
):
    output_count = (count + 1) // 2
    for output in pl.range(output_count):
        left = base + 2 * output
        right = left + 1
        if right < base + count:
            _merge_topk_pairs(arena, left, right, base + output)
        else:
            forwarded = pl.load(arena, [left, 0], [1, TOPK_PAIR_WIDTH])
            pl.store(forwarded, [base + output, 0], arena)
    return output_count


@pl.jit.inline
def _sort_topk_leaf(
    scores: pl.Tensor,
    arena: pl.Tensor,
    token: pl.Scalar[pl.INDEX],
    logical_begin: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
):
    raw = pl.load(
        scores,
        [token, logical_begin],
        [1, TOPK_LEAF],
        valid_shape=[1, valid_count],
    )
    padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
    floor = pl.tile.full([1, TOPK_LEAF], dtype=pl.FP32, value=-1e30)
    values = pl.maximum(padded, floor)
    logical_begin_i32 = pl.cast(logical_begin, pl.INT32)
    indices = pl.add(pl.tile.arange(0, [1, TOPK_LEAF], dtype=pl.INT32), logical_begin_i32)
    pairs = pl.tile.sort32(values, pl.reinterpret_view(indices, pl.UINT32))
    pairs = pl.tile.mrgsort(pairs, block_len=64)
    pairs = pl.tile.mrgsort(pairs, block_len=256)
    pairs = pl.tile.mrgsort(pairs, block_len=1024)
    pairs = pl.tile.mrgsort(pairs, block_len=4096)
    top_pairs = pl.tile.slice(pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
    pl.store(top_pairs, [output_slot, 0], arena)


@pl.jit.inline(auto_scope=False)
def _select_c1a_topk(
    scores: pl.Tensor,
    lengths: pl.Tensor,
    requests: pl.Tensor,
    table: pl.Tensor,
    pool_ids: pl.Tensor,
    pool_counts: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
    pool: pl.constexpr,
):
    """Select stable C1A Top-K physical rows."""
    tokens = pl.tensor.dim(scores, 0)
    width = pl.tensor.dim(scores, 1)
    groups_per_token = (width + TOPK_GROUP_LEAVES * TOPK_LEAF - 1) // (TOPK_GROUP_LEAVES * TOPK_LEAF)
    roots = tokens * groups_per_token
    arena_rows = roots + tokens * TOPK_GROUP_LEAVES
    arena = pl.create_tensor([arena_rows, TOPK_PAIR_WIDTH], dtype=pl.FP32)
    with pl.spmd(num_tokens, name_hint="prefill_index_topk_groups", deps=[ready]) as group_tid:
        token = pl.tile.get_block_idx()
        visible = pl.min(pl.max(pl.read(lengths, [token]), 0), width)
        if pool:
            visible = pl.min(pl.max(pl.read(pool_counts, [token]), 0) * 8, width)
        work_visible = pl.max(visible, 1)
        leaves = (work_visible + TOPK_LEAF - 1) // TOPK_LEAF
        group_count = (leaves + TOPK_GROUP_LEAVES - 1) // TOPK_GROUP_LEAVES
        scratch = roots + token * TOPK_GROUP_LEAVES
        for group in pl.range(group_count):
            leaf_begin = group * TOPK_GROUP_LEAVES
            count = pl.min(TOPK_GROUP_LEAVES, leaves - leaf_begin)
            root = token * groups_per_token + group
            if count == 1:
                begin = leaf_begin * TOPK_LEAF
                valid = pl.min(TOPK_LEAF, work_visible - begin)
                _sort_topk_leaf(scores, arena, token, begin, valid, root)
            else:
                for leaf in pl.unroll(TOPK_GROUP_LEAVES):
                    begin = (leaf_begin + leaf) * TOPK_LEAF
                    valid = pl.min(TOPK_LEAF, work_visible - begin)
                    _sort_topk_leaf(scores, arena, token, begin, valid, scratch + leaf)
                _merge_topk_pairs(arena, scratch, scratch + 1, root)
    with pl.spmd(num_tokens, name_hint="prefill_index_topk_merge", deps=[group_tid]) as selected:
        token = pl.tile.get_block_idx()
        visible = pl.min(pl.max(pl.read(lengths, [token]), 0), width)
        if pool:
            visible = pl.min(pl.max(pl.read(pool_counts, [token]), 0) * 8, width)
        count = (pl.max(visible, 1) + TOPK_GROUP_LEAVES * TOPK_LEAF - 1) // (TOPK_GROUP_LEAVES * TOPK_LEAF)
        base = token * groups_per_token
        for _ in pl.unroll(6):
            if count > 1:
                count = _merge_topk_level(arena, base, count)
        pairs = pl.load(arena, [base, 0], [1, TOPK_PAIR_WIDTH])
        values = pl.tile.gather_mask(pairs, mask_pattern=pl.tile.MaskPattern.P0101)
        finite_values = pl.maximum(values, -1e29)
        cutoff = pl.tile.read(finite_values, [0, INDEX_TOPK - 1])
        ones = pl.tile.full([1, INDEX_TOPK], dtype=pl.FP32, value=1.0)
        flags = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=4), ones, 0.0)
        sums64 = pl.col_sum(pl.reshape(flags, [8, 64]))
        sums8 = pl.col_sum(pl.reshape(sums64, [8, 8]))
        above = pl.cast(0, pl.INDEX)
        for lane in pl.unroll(8):
            partial = pl.cast(pl.tile.read(sums8, [0, lane]), pl.INT32)
            above = above + pl.cast(partial, pl.INDEX)
        materialize_index_topk(
            scores, table, requests, pool_ids, output, token, visible, cutoff, INDEX_TOPK - above, pool=pool,
        )
    return selected


@pl.jit.inline(auto_scope=False)
def _select_c2a_topk(
    scores: pl.Tensor,
    lengths: pl.Tensor,
    requests: pl.Tensor,
    table: pl.Tensor,
    cache_pages: pl.Scalar[pl.INDEX],
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Fold 2048-key leaves into the original running physical-row Top-K."""
    tokens = pl.tensor.dim(scores, 0)
    width = pl.tensor.dim(scores, 1)
    leaf_rows = pl.create_tensor([tokens, LEAF], dtype=pl.INT32)
    with pl.spmd(num_tokens, name_hint="prefill_c2a_topk", deps=[ready]) as selected:
        token = pl.tile.get_block_idx()
        request = pl.read(requests, [token])
        length = pl.min(pl.max(pl.read(lengths, [token]), 0), width)
        empty = pl.tile.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
        pl.store(empty, [token, 0], output)
        if length <= INDEX_TOPK:
            written = pl.cast(0, pl.INDEX)
            for part in pl.range(INDEX_TOPK // 128):
                begin = part * 128
                if begin < length and request >= 0 and request < pl.tensor.dim(table, 0):
                    page = pl.read(table, [request, part])
                    if page >= 0 and page < cache_pages:
                        count = pl.min(128, length - begin)
                        short_rows = pl.add(pl.tile.arange(0, [1, 128], dtype=pl.INT32), pl.cast(page * 128, pl.INT32))
                        pl.store(pl.set_validshape(short_rows, 1, count), [token, written], output)
                        written = written + count
        else:
            running = pl.tile.full([1, TOPK_PAIR_WIDTH], dtype=pl.FP32, value=SORT_FLOOR)
            for leaf, (previous,) in pl.range((length + LEAF - 1) // LEAF, init_values=(running,)):
                begin = leaf * LEAF
                for part in pl.range(LEAF // 128):
                    logical = begin + part * 128
                    physical_rows = pl.tile.full([1, 128], dtype=pl.INT32, value=-1)
                    if logical < length and request >= 0 and request < pl.tensor.dim(table, 0):
                        page = pl.read(table, [request, logical // 128])
                        if page >= 0 and page < cache_pages:
                            row_offsets = pl.tile.arange(0, [1, 128], dtype=pl.INT32)
                            physical_rows = pl.add(row_offsets, pl.cast(page * 128, pl.INT32))
                    pl.store(physical_rows, [token, part * 128], leaf_rows)
                raw = pl.load(scores, [token, begin], [1, LEAF], valid_shape=[1, pl.min(LEAF, length - begin)])
                padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
                values = pl.maximum(padded, SORT_FLOOR)
                leaf_row_ids = pl.load(leaf_rows, [token, 0], [1, LEAF])
                pairs = pl.tile.sort32(values, pl.reinterpret_view(leaf_row_ids, pl.UINT32))
                pairs = pl.tile.mrgsort(pairs, block_len=64)
                pairs = pl.tile.mrgsort(pairs, block_len=256)
                pairs = pl.tile.mrgsort(pairs, block_len=1024)
                merge_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
                merged = pl.tile.mrgsort(previous, pl.tile.slice(pairs, [1, TOPK_PAIR_WIDTH], [0, 0]), tmp=merge_tmp)
                running = pl.yield_(pl.tile.slice(merged, [1, TOPK_PAIR_WIDTH], [0, 0]))
            chosen_values = pl.tile.gather_mask(running, mask_pattern=pl.tile.MaskPattern.P0101)
            chosen = pl.tile.gather_mask(running, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            valid = pl.tile.cmps(chosen_values, -1e29, cmp_type=4)
            order_values = pl.tile.select(valid, pl.mul(pl.cast(chosen, pl.FP32), -1.0), SORT_FLOOR)
            order_rows = pl.tile.select(valid, chosen, -1)
            ordered = pl.tile.sort32(order_values, pl.reinterpret_view(order_rows, pl.UINT32))
            ordered = pl.tile.mrgsort(ordered, block_len=64)
            ordered = pl.tile.mrgsort(ordered, block_len=256)
            final = pl.tile.gather_mask(ordered, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            pl.store(final, [token, 0], output)
    return selected


@pl.jit.inline(auto_scope=False)
def _select_paged_topk(
    scores: pl.Tensor,
    lengths: pl.Tensor,
    requests: pl.Tensor,
    table: pl.Tensor,
    pool_ids: pl.Tensor,
    pool_counts: pl.Tensor,
    cache_pages: pl.Scalar[pl.INDEX],
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
    pool: pl.constexpr,
    c2a: pl.constexpr,
):
    """Select paged rows with stable BF16 score and logical-position keys."""
    width = pl.tensor.dim(scores, 1)
    table_width = pl.tensor.dim(table, 1)
    request_count = pl.tensor.dim(table, 0)
    workers = pl.min(48, num_tokens)
    with pl.spmd(workers, name_hint="prefill_index_paged_topk", deps=[ready]) as selected:
        worker = pl.tile.get_block_idx()
        leaf_ids = pl.tile.arange(0, [1, LEAF], dtype=pl.INT32)
        output_ids = pl.tile.arange(0, [1, INDEX_TOPK], dtype=pl.INT32)
        for token in pl.range(worker, num_tokens, workers):
            request = pl.read(requests, [token])
            length = pl.min(pl.max(pl.read(lengths, [token]), 0), width)
            if pool:
                length = pl.min(pl.max(pl.read(pool_counts, [token]), 0) * CANDIDATE_BLOCK_SIZE, width)
            if request < 0 or request >= request_count:
                length = pl.cast(0, pl.INDEX)
            if length == 0:
                empty = pl.tile.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
                pl.store(empty, [token, 0], output)
            else:
                chosen = output_ids
                limit = pl.cast(pl.cast(length, pl.INT32), pl.FP32)
                valid = pl.tile.cmps(pl.cast(output_ids, pl.FP32), limit, cmp_type=2)
                use_scores = True
                if c2a:
                    use_scores = length > INDEX_TOPK
                if use_scores:
                    initial = pl.tile.full([1, TOPK_PAIR_WIDTH], dtype=pl.FP32, value=0.0)
                    for begin, (previous,) in pl.range(0, length, LEAF, init_values=(initial,)):
                        cols = pl.min(LEAF, length - begin)
                        raw = pl.load(scores, [token, begin], [1, LEAF], valid_shape=[1, cols])
                        values = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
                        indices = pl.add(leaf_ids, pl.cast(begin, pl.INT32))
                        bf16_bits = pl.shrs(pl.reinterpret_view(values, pl.INT32), 16)
                        sign = pl.shrs(bf16_bits, 15)
                        positive_rank = pl.add(bf16_bits, 32768)
                        negative_rank = pl.mul(bf16_bits, -1)
                        rank = pl.add(positive_rank, pl.mul(sign, pl.sub(positive_rank, negative_rank)))
                        priority = pl.add(pl.mul(indices, -1), TOPK_ORDER_WIDTH - 1)
                        # Bias the combined bits into the positive normal FP32 range.
                        key_bits = pl.add(pl.shls(rank, TOPK_ORDER_BITS), pl.add(priority, 8388608))
                        finite = pl.tile.cmps(values, -1e29, cmp_type=4)
                        keys = pl.tile.select(finite, pl.reinterpret_view(key_bits, pl.FP32), 0.0)
                        pairs = pl.tile.sort32(keys, pl.reinterpret_view(indices, pl.UINT32))
                        pairs = pl.tile.mrgsort(pairs, block_len=64)
                        pairs = pl.tile.mrgsort(pairs, block_len=256)
                        pairs = pl.tile.mrgsort(pairs, block_len=1024)
                        best = pl.tile.slice(pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
                        merge_tmp = pl.create_tile([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
                        merged = pl.tile.mrgsort(previous, best, tmp=merge_tmp)
                        running = pl.yield_(pl.tile.slice(merged, [1, TOPK_PAIR_WIDTH], [0, 0]))
                    top_keys = pl.tile.gather_mask(running, mask_pattern=pl.tile.MaskPattern.P0101)
                    chosen = pl.tile.gather_mask(running, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                    valid = pl.tile.cmps(top_keys, 0.0, cmp_type=4)

                safe_ids = pl.tile.select(valid, chosen, 0)
                logical = safe_ids
                gather_tmp = pl.create_tile([1, INDEX_TOPK], dtype=pl.INT32)
                if pool:
                    block_count = pl.tensor.dim(pool_ids, 1)
                    raw_blocks = pl.load(pool_ids, [token, 0], [1, CANDIDATE_TOPK_BLOCKS], valid_shape=[1, block_count])
                    blocks = pl.tile.fillpad(raw_blocks, pad_value=pl.PadValue.zero)
                    block_lanes = pl.shrs(safe_ids, 3)
                    block_ids = pl.tile.gather(blocks, block_lanes, gather_tmp)
                    block_offsets = pl.sub(safe_ids, pl.shls(block_lanes, 3))
                    logical = pl.add(pl.shls(block_ids, 3), block_offsets)
                page_slots = pl.shrs(logical, 7)
                max_page_slot = pl.cast(table_width - 1, pl.INT32)
                safe_slots = pl.minimum(pl.maximum(page_slots, 0), max_page_slot)
                raw_pages = pl.load(table, [request, 0], [1, TOPK_ORDER_WIDTH // 128], valid_shape=[1, table_width])
                pages = pl.tile.fillpad(raw_pages, pad_value=pl.PadValue.zero)
                page_ids = pl.tile.gather(pages, safe_slots, gather_tmp)
                cache_limit = pl.cast(cache_pages - 1, pl.INT32)
                from_end = pl.add(pl.mul(page_ids, -1), cache_limit)
                in_cache = pl.minimum(page_ids, from_end)
                page_valid = pl.tile.cmps(pl.cast(in_cache, pl.FP32), 0.0, cmp_type=5)
                offsets = pl.sub(logical, pl.shls(page_slots, 7))
                physical = pl.add(pl.shls(page_ids, 7), offsets)
                visible_rows = pl.tile.select(valid, physical, -1)
                physical_rows = pl.tile.select(page_valid, visible_rows, -1)
                order_ids = pl.tile.fillpad(safe_ids, pad_value=pl.PadValue.zero)
                if c2a:
                    if use_scores:
                        order_ids = pl.tile.fillpad(physical, pad_value=pl.PadValue.zero)
                order = pl.mul(pl.cast(order_ids, pl.FP32), -1.0)
                order = pl.tile.select(valid, order, SORT_FLOOR)
                order = pl.tile.select(page_valid, order, SORT_FLOOR)
                sorted_pairs = pl.tile.sort32(order, pl.reinterpret_view(physical_rows, pl.UINT32))
                sorted_pairs = pl.tile.mrgsort(sorted_pairs, block_len=64)
                sorted_pairs = pl.tile.mrgsort(sorted_pairs, block_len=256)
                rows = pl.tile.gather_mask(sorted_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                pl.store(rows, [token, 0], output)
    return selected


@pl.jit.inline
def prefill_index_key_project(
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    weight: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.ND],
    norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    output: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    pool_ready: pl.Scalar[pl.TASK_ID],
):
    """Project C2A keys, then normalize after the compressor pool_ready dependency."""
    tokens = pl.tensor.dim(latent, 0)
    projected = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    projection_ready = pl.system.task_dummy(deps=[])
    projected_result, projected_tid = gemm_bf16_to_bf16(
        latent, weight, projected, num_tokens, projection_ready,
        width=HEAD_DIM, output_width=INDEX_DIM, n_tile=INDEX_PROJ_N_TILE, k_tile=INDEX_PROJ_K_TILE,
        workers=2147483647, name_hint="prefill_index_key",
        source_col=0, output_col=0, ordered=False, row_tile=16, stages=1,
    )
    norm_blocks = (num_tokens + M_TILE - 1) // M_TILE
    with pl.spmd(norm_blocks, name_hint="prefill_c2a_index_norm", deps=[pool_ready]) as norm_tid:
        t0 = pl.tile.get_block_idx() * M_TILE
        rows = pl.min(M_TILE, num_tokens - t0)
        value = pl.cast(pl.slice(projected, [M_TILE, INDEX_DIM], [t0, 0], valid_shape=[rows, INDEX_DIM]), pl.FP32)
        square = pl.mul(pl.row_sum(pl.mul(value, value)), 1.0 / INDEX_DIM)
        inv = pl.rsqrt(pl.add(square, EPS), high_precision=True)
        gamma = pl.reshape(pl.cast(norm_weight[:], pl.FP32), [1, INDEX_DIM])
        normalized = pl.col_expand_mul(pl.row_expand_mul(value, inv), gamma)
        output[t0 : t0 + M_TILE, :] = pl.set_validshape(pl.cast(normalized, pl.BF16, mode="rint"), rows, INDEX_DIM)
    return norm_tid


@pl.jit.inline(auto_scope=False)
def publish_c1a_index_key(
    compressed_latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.ND],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.UINT8],
    num_tokens: pl.Scalar[pl.INT32],
):
    tokens = pl.tensor.dim(compressed_latent, 0)
    index_projection = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    projection_ready = pl.system.task_dummy(deps=[])
    projected_result, projected_tid = gemm_bf16_to_bf16(
        compressed_latent, index_wk, index_projection, num_tokens, projection_ready,
        width=HEAD_DIM, output_width=INDEX_DIM, n_tile=INDEX_PROJ_N_TILE, k_tile=INDEX_PROJ_K_TILE,
        workers=2147483647, name_hint="prefill_index_key",
        source_col=0, output_col=0, ordered=False, row_tile=16, stages=1,
    )
    index_normalized = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    norm_blocks = (num_tokens + 7) // 8
    for norm_worker in pl.spmd(pl.max(1, norm_blocks), name_hint="attention_rmsnorm"):
        if norm_worker < norm_blocks:
            rms_norm(index_projection, index_norm_weight, index_normalized, norm_worker, num_tokens, width=INDEX_DIM)
    index_rotated = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    _index_rope(
        index_normalized, compressed_rope_cos, compressed_rope_sin, index_rotated, num_tokens,
        heads=1, max_workers=2147483647,
    )
    cache_ready = pl.system.task_dummy(deps=[])
    index_ready = publish_index_cache(
        index_rotated, compressed_slots, index_cache, index_cache_scale, num_tokens, cache_ready,
    )
    return index_ready


@pl.jit.inline(auto_scope=False)
def publish_c2a_index_key(
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.ND],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_PACKED], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_SCALES], pl.UINT8],
    num_tokens: pl.Scalar[pl.INT32],
    pool_tid: pl.Scalar[pl.TASK_ID],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    tokens = pl.tensor.dim(latent, 0)
    index_key = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    prefill_index_key_project(latent, index_wk, index_norm_weight, index_key, num_tokens, pool_tid)
    rotated_key = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    _index_rope(
        index_key, compressed_rope_cos, compressed_rope_sin, rotated_key, num_tokens,
        heads=1, max_workers=2147483647,
    )
    index_publish_tid = publish_index_cache(
        rotated_key, compressed_slots, index_cache, index_cache_scale, num_tokens, cache_ready
    )
    return index_publish_tid


@pl.jit.inline(auto_scope=False)
def _clear_inactive_pool(
    ids: pl.Tensor,
    counts: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    rows = pl.tensor.dim(ids, 0)
    width = pl.tensor.dim(ids, 1)
    first = pl.cast(pl.max(num_tokens, 0), pl.INDEX)
    tiles = (width + 127) // 128
    with pl.spmd(32, name_hint="prefill_inactive_pool_ids", deps=[ready]) as ids_tid:
        worker = pl.tile.get_block_idx()
        for work in pl.range(worker, (rows - first) * tiles, 32):
            row = first + work // tiles
            column = work % tiles * 128
            invalid = pl.tile.full([1, 128], dtype=pl.INT32, value=-1)
            pl.store(pl.set_validshape(invalid, 1, pl.min(128, width - column)), [row, column], ids)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_inactive_pool_counts", deps=[ids_tid]) as counts_tid:
        for row in pl.range(first, rows):
            pl.write(counts, [row], pl.cast(0, pl.INT32))
    return counts_tid


@pl.jit.inline(auto_scope=False)
def prefill_indexer(
    x: pl.Tensor,
    latent: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    cache: pl.Tensor,
    cache_scale: pl.Tensor,
    table: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    query_weight: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.INT8],
    query_weight_scale: pl.Tensor,
    weight_projection: pl.Tensor[[D, INDEX_H], pl.BF16, pl.ND],
    pool_ids: pl.Tensor,
    pool_counts: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
    pool: pl.constexpr,
    c2a: pl.constexpr,
):
    """Prepare queries, score bounded chunks, and select sparse cache rows."""
    width = pl.tensor.dim(table, 1) * 128
    if pool:
        width = pl.tensor.dim(pool_ids, 1) * 8
    width = (width + 127) // 128 * 128
    completion = pl.array.create(1, pl.TASK_ID)
    completion[0] = ready
    if not pool and not c2a:
        if num_tokens < pl.tensor.dim(x, 0):
            cleared = _clear_inactive_pool(pool_ids, pool_counts, num_tokens, ready)
            completion[0] = cleared
    if num_tokens > 0:
        decoded_keys, decoded_tid = _decode_index_pages(cache, cache_scale, completion[0])
        linear_available = pl.tensor.dim(table, 0) == 1
        if pool:
            linear_available = False
        linear_rows = pl.cast(1, pl.INDEX)
        if linear_available:
            linear_rows = (pl.tensor.dim(table, 1) + 1) // 2 * 256
        linear_keys = pl.create_tensor([linear_rows, INDEX_DIM], dtype=pl.BF16)
        if linear_available:
            decoded_tid = _order_index_pages(decoded_keys, table, linear_keys, decoded_tid)
        completion[0] = decoded_tid
        active = pl.cast(num_tokens, pl.INDEX)
        chunk = pl.min(active, pl.max(1, INDEX_MAX_LOGITS_BYTES // (width * 4)))
        if pool:
            chunk = pl.min(chunk, INDEX_POOL_QUERY_TILE)
        scores = pl.create_tensor([chunk, width], dtype=pl.FP32)
        for begin in pl.range(0, active, chunk):
            with pl.scope():
                rows = pl.min(chunk, active - begin)
                count = pl.cast(rows, pl.INT32)
                chunk_x = pl.slice(x, [rows, D], [begin, 0])
                chunk_latent = pl.slice(latent, [rows, Q_LORA], [begin, 0])
                chunk_requests = pl.slice(request_ids, [rows], [begin])
                chunk_lengths = pl.slice(lengths, [rows], [begin])
                chunk_cos = pl.slice(cos, [rows, ROPE_DIM // 2], [begin, 0])
                chunk_sin = pl.slice(sin, [rows, ROPE_DIM // 2], [begin, 0])
                capacity = pl.tensor.dim(pool_ids, 1)
                chunk_pool = pl.slice(pool_ids, [rows, capacity], [begin, 0])
                chunk_counts = pl.slice(pool_counts, [rows], [begin])
                chunk_output = pl.slice(output, [rows, INDEX_TOPK], [begin, 0])
                chunk_scores = pl.slice(scores, [rows, width], [0, 0])
                query = pl.create_tensor([rows, INDEX_QUERY_WIDTH], dtype=pl.BF16)
                weights = pl.create_tensor([rows, INDEX_H], dtype=pl.BF16)
                maximum_visible = pl.cast(0, pl.INDEX)
                if not pool:
                    for token in pl.range(rows):
                        if c2a:
                            maximum_visible = pl.max(maximum_visible, pl.read(chunk_lengths, [token]))
                chunk_ready = completion[0]
                scoring = pl.array.create(1, pl.TASK_ID)
                scoring[0] = chunk_ready
                if not c2a or maximum_visible > INDEX_TOPK:
                    query_tid, weights_tid = prepare_prefill_index_query(
                        chunk_x, chunk_latent, query_weight, query_weight_scale, weight_projection,
                        chunk_cos, chunk_sin, query, weights, count, ready,
                    )
                    prepared = pl.system.task_dummy(deps=[query_tid, weights_tid, chunk_ready])
                    if pool:
                        compact_pages = capacity // 16
                        compact = pl.create_tensor([rows * compact_pages * 128, INDEX_DIM], dtype=pl.BF16)
                        compact_table = pl.create_tensor([rows, compact_pages], dtype=pl.INT32)
                        compact_requests = pl.create_tensor([rows], dtype=pl.INT32)
                        compact_lengths = pl.create_tensor([rows], dtype=pl.INT32)
                        compact_ready = _compact_pool_cache(
                            decoded_keys, table, chunk_requests, chunk_lengths, chunk_pool, chunk_counts,
                            compact, compact_table, compact_requests, compact_lengths, count, prepared,
                        )
                        general_tid = _prefill_index_scores(
                            query, weights, compact, compact_table,
                            compact_requests, compact_lengths, chunk_scores,
                            count, compact_ready,
                            tile_n=128, query_tile=1, skip_short=False, head_rows=INDEX_H,
                            materialize_padding=False, pipe_slots=1,
                            ordered_keys=False, physical_pages=0,
                        )
                        scoring[0] = general_tid
                    else:
                        if linear_available and rows % INDEX_SCORE_QUERY_TILE == 0:
                            paired = _prefill_index_scores(
                                query, weights, linear_keys, table, chunk_requests, chunk_lengths,
                                chunk_scores, count, prepared,
                                tile_n=256, query_tile=INDEX_SCORE_QUERY_TILE, skip_short=c2a,
                                head_rows=INDEX_SCORE_QUERY_TILE * INDEX_H, materialize_padding=False, pipe_slots=1,
                                ordered_keys=True, physical_pages=pl.tensor.dim(cache, 0),
                            )
                            scoring[0] = paired
                        else:
                            general_tid = _prefill_index_scores(
                                query, weights, decoded_keys, table, chunk_requests, chunk_lengths,
                                chunk_scores, count, prepared,
                                tile_n=128, query_tile=1, skip_short=c2a, head_rows=INDEX_H,
                                materialize_padding=False, pipe_slots=1,
                                ordered_keys=False, physical_pages=0,
                            )
                            scoring[0] = general_tid
                score_ready = scoring[0]
                selection = pl.array.create(1, pl.TASK_ID)
                selection[0] = score_ready
                if width <= TOPK_ORDER_WIDTH and pl.tensor.dim(table, 1) <= TOPK_ORDER_WIDTH // 128:
                    selected = _select_paged_topk(
                        chunk_scores, chunk_lengths, chunk_requests, table, chunk_pool, chunk_counts,
                        pl.tensor.dim(cache, 0), chunk_output, count, score_ready, pool=pool, c2a=c2a,
                    )
                    selection[0] = selected
                elif c2a:
                    selected = _select_c2a_topk(chunk_scores, chunk_lengths, chunk_requests,
                                              table, pl.tensor.dim(cache, 0), chunk_output, count, score_ready)
                    selection[0] = selected
                else:
                    selected = _select_c1a_topk(
                        chunk_scores, chunk_lengths, chunk_requests, table,
                        chunk_pool, chunk_counts, chunk_output, count, score_ready, pool=pool,
                    )
                    selection[0] = selected
                topk_ready = selection[0]
                completion[0] = topk_ready
                if not pool and not c2a:
                    if width <= capacity * CANDIDATE_BLOCK_SIZE and width <= CANDIDATE_TOPK_BLOCKS * CANDIDATE_BLOCK_SIZE:
                        pool_ready = _publish_all_candidate_blocks(chunk_lengths, chunk_pool, chunk_counts,
                                                                   count, width, score_ready)
                        completion[0] = pl.system.task_dummy(deps=[topk_ready, pool_ready])
                    else:
                        sorted_pool = select_candidate_block_pool(chunk_scores, chunk_lengths, chunk_pool,
                                                                  chunk_counts, count, score_ready)
                        completion[0] = pl.system.task_dummy(deps=[topk_ready, sorted_pool])
    return completion[0]


def golden_index_query_fp4(value: torch.Tensor) -> torch.Tensor:
    """Round group-32 index queries to E2M1 with upward E8M0 scales."""
    payload, scales = quantize_mxfp4_cache(value, 32, "e8m0", round_to_even=True)
    return dequantize_mxfp4_cache(payload, scales, 32, "e8m0").to(value.dtype)


def golden_index_key(
    latent: torch.Tensor,
    wk: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Project an unrotated compressor latent into a rotated index key."""
    projected = torch.matmul(latent.float(), wk.float()).to(latent.dtype)
    key = golden_rms_norm(projected, norm_weight)
    rd = cos.shape[-1] * 2
    return torch.cat((key[..., :-rd], golden_rope_interleave(key[..., -rd:], cos, sin)), dim=-1)


def golden_select_candidate_block_pool(
    logits: torch.Tensor,
    compressed_lens: torch.Tensor | int,
    topk_blocks: int = M.candidate_topk_blocks,
    block_size: int = M.candidate_block_size,
    *,
    num_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sorted logical block IDs and counts with a causal tail and fixed selected prefix."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be rank-2, got shape {tuple(logits.shape)}")
    if block_size != CANDIDATE_BLOCK_SIZE:
        raise ValueError(f"block_size must be {CANDIDATE_BLOCK_SIZE}, got {block_size}")
    if topk_blocks != CANDIDATE_TOPK_BLOCKS:
        raise ValueError(f"topk_blocks must be {CANDIDATE_TOPK_BLOCKS}, got {topk_blocks}")

    tokens, width = logits.shape
    # Inactive tensor rows publish canonical empty pool state.
    active_tokens = tokens if num_tokens is None else max(0, min(tokens, int(num_tokens)))
    capacity = index_pool_block_capacity(width)
    ids = torch.full((tokens, capacity), -1, dtype=torch.int32, device=logits.device)
    counts = torch.zeros((tokens,), dtype=torch.int32, device=logits.device)
    if width == 0:
        return ids, counts

    lengths = torch.as_tensor(compressed_lens, device=logits.device, dtype=torch.int64)
    if lengths.ndim == 0:
        lengths = lengths.expand(tokens)
    if lengths.ndim != 1 or lengths.numel() != tokens:
        raise ValueError("compressed_lens must provide one causal length per score row")

    padded = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    block_scores = padded.unflatten(-1, (-1, block_size)).amax(dim=-1)
    for token in range(active_tokens):
        visible = max(0, min(int(lengths[token]), width))
        visible_blocks = (visible + block_size - 1) // block_size
        if visible_blocks == 0:
            continue
        selected_count = min(visible_blocks, topk_blocks, capacity)
        # Rank invalid block maxima at the device sort floor.
        scores = block_scores[token, :visible_blocks].clone().clamp_min(-1e30)
        # Include the newest causal block independently of its score.
        scores[visible_blocks - 1] = 1e30
        # Resolve equal block maxima by stable logical order.
        selected = torch.argsort(scores, descending=True, stable=True)[:selected_count].to(torch.int32)
        ids[token, :selected_count] = selected.sort().values
        counts[token] = selected_count
    return ids, counts


def golden_prefill_indexer(
    x, qr, request_ids, index_cache, block_table, compressed_lens,
    wq_b, wq_b_scale, weights_proj, cos, sin, *,
    pool_block_ids=None, pool_block_counts=None, c2a=False, topk=INDEX_TOPK,
):
    """Reference logical FP4 queries, BF16 rounding, and sparse physical-row selection."""
    tokens = x.shape[0]
    width = int(compressed_lens.max()) if tokens else 0
    scores = torch.full((tokens, width), -torch.inf, dtype=torch.float32)
    selected = torch.full((tokens, topk), -1, dtype=torch.int32)
    query = golden_dynamic_int8(qr, wq_b, wq_b_scale).reshape(tokens, INDEX_H, INDEX_DIM)
    query = golden_rope_tail(query, cos, sin)
    query = golden_index_query_fp4(query)
    weights = (x.float() @ weights_proj.float()).bfloat16()
    weights = (weights.float() * INDEX_SCORE_SCALE).bfloat16()
    keys = index_cache.flatten(0, 1).squeeze(-2)
    for token in range(tokens):
        length = max(0, int(compressed_lens[token]))
        positions = torch.arange(length)
        if pool_block_ids is not None:
            count = int(pool_block_counts[token])
            positions = (pool_block_ids[token, :count, None].long() * 8 + torch.arange(8)).flatten()
            positions = positions[(positions >= 0) & (positions < length)]
        if not positions.numel():
            continue
        pages = block_table[int(request_ids[token]), positions // 128]
        present = (pages >= 0) & (pages < index_cache.shape[0])
        positions, pages = positions[present], pages[present]
        physical = pages.long() * 128 + positions % 128
        if c2a and length <= topk:
            selected[token, :physical.numel()] = physical.int()
            continue
        dots = (query[token].float() @ keys[physical].T.float()).bfloat16().float().relu()
        weighted = (dots * weights[token].float()[:, None]).bfloat16().float()
        values = weighted.sum(0).bfloat16().float()
        scores[token, positions] = values
        count = min(topk, positions.numel())
        if count:
            chosen = torch.argsort(values, descending=True, stable=True)[:count]
            order = physical[chosen].argsort() if c2a else positions[chosen].argsort()
            selected[token, :count] = physical[chosen[order]].int()
    return scores, selected


def golden_prefill_index_scores(mode, rank, cache_outputs, inputs):
    """Use the same Prefill profile for the existing cutoff comparator."""
    from .qkv_proj_rope import golden_qkv_proj_rope

    payload = cache_outputs["index_cache"] if mode == "full" else inputs["index_cache"]
    scales = cache_outputs["index_cache_scale"] if mode == "full" else inputs["index_cache_scale"]
    keys = dequantize_mxfp4_cache(payload[rank], scales[rank], 32, "e8m0").bfloat16()
    _, _, latent = golden_qkv_proj_rope(
        inputs["x"][rank], inputs["wq_a"][rank], inputs["wq_a_scale"][rank], inputs["q_norm_weight"][rank],
        inputs["wq_b"][rank], inputs["wq_b_scale"][rank], inputs["wkv"][rank], inputs["wkv_scale"][rank],
        inputs["kv_norm_weight"][rank], inputs["rope_cos"][rank], inputs["rope_sin"][rank],
    )
    pool = {} if mode == "full" else {name: inputs[name][rank] for name in ("pool_block_ids", "pool_block_counts")}
    scores, _ = golden_prefill_indexer(
        inputs["x"][rank], latent, inputs["request_ids"][rank], keys, inputs["index_block_table"][rank],
        inputs["compressed_lens"][rank], inputs["index_wq_b"][rank], inputs["index_wq_b_scale"][rank],
        inputs["index_weights_proj"][rank], inputs["rope_cos"][rank], inputs["rope_sin"][rank], **pool,
    )
    return scores


def index_pool_block_capacity(score_width: int) -> int:
    """Return the aligned compact-pool capacity for one score row."""
    if score_width < 0:
        raise ValueError(f"score_width must be nonnegative, got {score_width}")
    block_size = M.candidate_block_size
    blocks = max(1, (score_width + block_size - 1) // block_size)
    aligned = (blocks + 31) // 32 * 32
    return min(M.candidate_topk_blocks, aligned)


@pl.jit(auto_scope=False)
def l2_prefill_indexer_validation(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    latent: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    query_weight: pl.Tensor[[Q_LORA, INDEX_QUERY_WIDTH], pl.INT8],
    query_weight_scale: pl.Tensor[[1, INDEX_QUERY_WIDTH], pl.FP32],
    weight_projection: pl.Tensor[[D, INDEX_H], pl.BF16, pl.ND],
    cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, 4], pl.UINT8],
    table: pl.Tensor,
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    lengths: pl.Tensor[[T_DYN], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    query: pl.InOut[pl.Tensor[[T_DYN, INDEX_QUERY_WIDTH], pl.BF16]],
    weights: pl.InOut[pl.Tensor[[T_DYN, INDEX_H], pl.BF16]],
    scores: pl.InOut[pl.Tensor[[T_DYN, INDEX_SCORE_POSITIONS_DYN], pl.FP32]],
    c1a: pl.InOut[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    c2a: pl.InOut[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    pool_ids: pl.InOut[pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    pool_counts: pl.InOut[pl.Tensor[[T_DYN], pl.INT32]],
    pool_scores: pl.InOut[pl.Tensor[[T_DYN, INDEX_POOL_SCORE_DYN], pl.FP32]],
    pool_topk: pl.InOut[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
):
    entry = pl.system.task_dummy(deps=[])
    query_tid, weight_tid = prepare_prefill_index_query(
        x, latent, query_weight, query_weight_scale, weight_projection,
        cos, sin, query, weights, num_tokens, entry,
    )
    prepared = pl.system.task_dummy(deps=[query_tid, weight_tid])
    decoded_keys, decode_tid = _decode_index_pages(cache, cache_scale, prepared)
    scored = _prefill_index_scores(
        query, weights, decoded_keys, table,
        request_ids, lengths, scores, num_tokens, decode_tid,
        tile_n=128, query_tile=1, skip_short=False, head_rows=INDEX_H, materialize_padding=True, pipe_slots=1,
        ordered_keys=False, physical_pages=0,
    )
    _select_c1a_topk(
        scores, lengths, request_ids, table, pool_ids, pool_counts, c1a, num_tokens, scored, pool=False,
    )
    _select_c2a_topk(scores, lengths, request_ids, table, pl.tensor.dim(cache, 0), c2a, num_tokens, scored)
    pool_ready = select_candidate_block_pool(scores, lengths, pool_ids, pool_counts, num_tokens, scored)
    pool_tokens = pl.tensor.dim(pool_ids, 0)
    compact_pages = pl.tensor.dim(pool_ids, 1) // 16
    compact = pl.create_tensor([pool_tokens * compact_pages * 128, INDEX_DIM], dtype=pl.BF16)
    compact_table = pl.create_tensor([pool_tokens, compact_pages], dtype=pl.INT32)
    compact_requests = pl.create_tensor([pool_tokens], dtype=pl.INT32)
    compact_lengths = pl.create_tensor([pool_tokens], dtype=pl.INT32)
    compact_ready = _compact_pool_cache(
        decoded_keys, table, request_ids, lengths, pool_ids, pool_counts,
        compact, compact_table, compact_requests, compact_lengths, num_tokens, pool_ready,
    )
    rescored = _prefill_index_scores(
        query, weights, compact, compact_table,
        compact_requests, compact_lengths, pool_scores, num_tokens, compact_ready,
        tile_n=128, query_tile=1, skip_short=False, head_rows=INDEX_H, materialize_padding=True, pipe_slots=1,
        ordered_keys=False, physical_pages=0,
    )
    _select_c1a_topk(
        pool_scores, lengths, request_ids, table, pool_ids, pool_counts, pool_topk,
        num_tokens, rescored, pool=True,
    )
    return query, weights, scores, c1a, c2a, pool_ids, pool_counts, pool_scores, pool_topk


@pl.jit
def l2_index_cache_validation(
    source: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    scales: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_SCALES], pl.UINT8]],
    num_tokens: pl.Scalar[pl.INT32],
):
    ready = pl.system.task_dummy(deps=[])
    publish_index_cache(source, slots, cache, scales, num_tokens, ready)
    return cache, scales


def golden_index_cache_validation(tensors):
    """Check exact FP4 midpoint codes, small scales, page boundaries and preserved rows."""
    count = int(tensors["num_tokens"])
    source = tensors["source"][:count]
    payload, scales = quantize_mxfp4_cache(source, 32, "e8m0", round_to_even=True)
    slots = tensors["slots"][:count]
    valid = slots >= 0
    for name, value in (("cache", payload), ("scales", scales)):
        target = tensors[name].reshape(-1, value.shape[-1])
        target[slots[valid].long()] = value[valid]


def golden_prefill_indexer_validation(tensors):
    """Check rounded queries, scores, physical TopK and candidate-pool membership."""
    from .indexer_reference import indexer_reference

    reference = indexer_reference(tensors)
    active = int(tensors["num_tokens"])
    tensors["query"][:active] = reference["query"][:active]
    tensors["weights"][:active] = reference["weights"][:active]
    tensors["scores"][:active] = reference["scores"][:active]
    tensors["c1a"][:active] = reference["c1a"][:active]
    tensors["c2a"][:active] = reference["c2a"][:active]
    ids, counts = golden_select_candidate_block_pool(reference["scores"], tensors["lengths"], num_tokens=active)
    tensors["pool_ids"][:active] = ids[:active]
    tensors["pool_counts"][:active] = counts[:active]
    tensors["pool_scores"][:active] = -torch.inf
    for token in range(active):
        blocks = ids[token, :int(counts[token])].long()
        positions = (blocks[:, None] * 8 + torch.arange(8)).flatten()
        present = positions < int(tensors["lengths"][token])
        compact = torch.arange(positions.numel())[present]
        tensors["pool_scores"][token, compact] = reference["scores"][token, positions[present]]
    tensors["pool_topk"][:active] = reference["c1a"][:active]


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    from golden import ScalarSpec, TensorSpec, ratio_allclose, run
    from .indexer_reference import compare_index_scores, compare_pool_scores, make_index_case
    from .weights import Checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3"], default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--history", type=int, default=8320)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    checkpoint = Checkpoint(args.checkpoint)
    records = []
    midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
    signed = torch.cat((midpoints, -midpoints)).repeat(INDEX_DIM // 16)
    source = (signed[None, :] * torch.exp2(torch.arange(16) - 8)[:, None]).bfloat16()
    source[0].zero_()
    source[1] = (signed * 2.0**-120).bfloat16()
    slots = torch.tensor([0, 1, 127, 128, 129, 255, -1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64)
    cache_specs = [TensorSpec("source", [16, INDEX_DIM], torch.bfloat16, init_value=source),
                   TensorSpec("slots", [16], torch.int64, init_value=slots),
                   TensorSpec("cache", [2, 128, 1, IDX_PACKED], torch.uint8,
                              init_value=torch.full((2, 128, 1, IDX_PACKED), 199, dtype=torch.uint8)),
                   TensorSpec("scales", [2, 128, 1, IDX_SCALES], torch.uint8,
                              init_value=torch.full((2, 128, 1, IDX_SCALES), 17, dtype=torch.uint8)),
                   ScalarSpec("num_tokens", torch.int32, 13)]
    exact = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
    cache_result = run(
        l2_index_cache_validation, cache_specs, golden_fn=golden_index_cache_validation,
        config=dict(backend_type=args.platform, execution_mode=False, device_id=args.device),
        compare_fn={"cache": exact, "scales": exact}, compile_only=args.compile_only,
    )
    records.append(dict(stage="cache_publication", passed=cache_result.passed, work_dir=str(cache_result.work_dir)))
    if not cache_result.passed:
        raise RuntimeError(cache_result.error)
    for ties in (False, True):
        values = make_index_case(checkpoint, args.layer, args.history, ties=ties)
        width = values["table"].shape[1] * 128
        capacity = index_pool_block_capacity(width)
        values.update(query=torch.full((16, INDEX_QUERY_WIDTH), 99, dtype=torch.bfloat16),
                      weights=torch.full((16, INDEX_H), 7.25, dtype=torch.bfloat16),
                      scores=torch.full((16, width), 23.5),
                      c1a=torch.full((16, INDEX_TOPK), -1, dtype=torch.int32),
                      c2a=torch.full((16, INDEX_TOPK), -1, dtype=torch.int32),
                      pool_ids=torch.full((16, capacity), -1, dtype=torch.int32),
                      pool_counts=torch.zeros(16, dtype=torch.int32),
                      pool_scores=torch.full((16, capacity * 8), 23.5),
                      pool_topk=torch.full((16, INDEX_TOPK), -1, dtype=torch.int32))
        specs = [ScalarSpec(name, torch.int32, value) if name == "num_tokens"
                 else TensorSpec(name, list(value.shape), value.dtype, init_value=value)
                 for name, value in values.items()]
        exact = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
        comparisons = {name: exact for name in ("query", "c1a", "c2a", "pool_ids", "pool_counts", "pool_topk")}
        comparisons.update(weights=ratio_allclose(atol=0.003, rtol=1.0 / 128.0, max_error_ratio=0.0),
                           scores=compare_index_scores, pool_scores=compare_pool_scores)
        result = run(
            l2_prefill_indexer_validation,
            specs,
            golden_fn=golden_prefill_indexer_validation,
            config=dict(backend_type=args.platform, execution_mode=False, device_id=args.device),
            compare_fn=comparisons,
            compile_only=args.compile_only,
            save_data=True,
        )
        records.append(dict(ties=ties, history=args.history, passed=result.passed, work_dir=str(result.work_dir)))
        if not result.passed:
            raise RuntimeError(result.error)
    if args.report:
        args.report.write_text(json.dumps(dict(checkpoint=checkpoint.identity, cases=records), indent=2) + "\n")

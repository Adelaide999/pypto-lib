# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compression projections, rotary embedding, cache publication, and ratio-2 ring state."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

import pypto.language as pl
import torch

from .config import (
    FLASH as M, STATE_CAPACITY, B_DYN, Q_START_DYN, STATE_BLOCKS_DYN,
    T_DYN, COMPRESSED_CACHE_GROUP, CMP_BLOCKS_DYN,
)
from .rmsnorm import golden_rms_norm, rms_norm
from .quant import encode_fp4
from .rope import make_pipelined_token_rope


# model config
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
EPS = M.rms_norm_eps
STATE_WIDTH = 2 * M.head_dim

# tiling
C1A_PROJ_M_TILE = 48
C1A_PROJ_N_TILE = 64
C1A_PROJ_K_TILE = 512
C2A_PROJ_M_TILE = 80
C2A_PROJ_N_TILE = 64
C2A_PROJ_K_TILE = 256


rotate_prefill_compressed = make_pipelined_token_rope(T_DYN, HEAD_DIM, name_hint="rope_cmp", inplace=False)


@pl.jit.inline
def _compressor_rope_block(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    block: pl.Scalar[pl.INT32],
):
    t = block
    row = pl.tile.load(x, [t, 0], [1, HEAD_DIM])
    tail_bf16 = pl.tile.slice(row, [1, ROPE_DIM], [0, HEAD_DIM - ROPE_DIM])
    tail = pl.cast(tail_bf16, pl.FP32)
    even = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P0101)
    odd = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P1010)
    c = pl.tile.load(cos, [t, 0], [1, ROPE_DIM // 2])
    sine = pl.tile.load(sin, [t, 0], [1, ROPE_DIM // 2])
    s = pl.mul(sine, 1.0)
    even_cos = pl.mul(even, c)
    odd_sin = pl.mul(odd, s)
    re = pl.sub(even_cos, odd_sin)
    even_sin = pl.mul(even, s)
    odd_cos = pl.mul(odd, c)
    im = pl.add(even_sin, odd_cos)
    real_lanes = pl.tile.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
    real_lanes = pl.tile.scatter_mask(real_lanes, re, mask_pattern=pl.tile.MaskPattern.P0101)
    imag_lanes = pl.tile.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
    imag_lanes = pl.tile.scatter_mask(imag_lanes, im, mask_pattern=pl.tile.MaskPattern.P1010)
    real_bits = pl.reinterpret_view(real_lanes, pl.INT16)
    imag_bits = pl.reinterpret_view(imag_lanes, pl.INT16)
    rotated_bits = pl.or_(real_bits, imag_bits)
    rotated = pl.reinterpret_view(rotated_bits, pl.FP32)
    rounded = pl.cast(rotated, pl.BF16, mode="rint")
    row = pl.tile.assemble(row, rounded, [0, HEAD_DIM - ROPE_DIM])
    pl.tile.store(row, [block, 0], output)
    return output


@pl.jit.inline(auto_scope=False)
def compressor_rope(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    if num_tokens >= 64:
        output = rotate_prefill_compressed(x, cos, sin, output, num_tokens)
    else:
        workers = pl.min(num_tokens, 12)
        with pl.spmd(workers, name_hint="rope_cmp"):
            worker = pl.tile.get_block_idx()
            for block in pl.range(worker, num_tokens, workers):
                _compressor_rope_block(x, cos, sin, output, block)
    return output


@pl.jit.inline(auto_scope=False)
def c1a_compressor_project(
    source: pl.Tensor,
    weight: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.ND],
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Project BF16 ND weights with full-height Cube tiles and store active rows."""
    blocks = (num_tokens + C1A_PROJ_M_TILE - 1) // C1A_PROJ_M_TILE * (HEAD_DIM // C1A_PROJ_N_TILE)
    workers = pl.max(blocks, 1)
    with pl.spmd(workers, name_hint="proj_cmp", deps=[ready]) as completed:
        with pl.at(level=pl.Level.CORE_GROUP):
            block = pl.tile.get_block_idx()
            row = block // (HEAD_DIM // C1A_PROJ_N_TILE) * C1A_PROJ_M_TILE
            col = block % (HEAD_DIM // C1A_PROJ_N_TILE) * C1A_PROJ_N_TILE
            rows = pl.max(pl.min(C1A_PROJ_M_TILE, num_tokens - row), 0)
            a = pl.load(source, [row, 0], [C1A_PROJ_M_TILE, C1A_PROJ_K_TILE], valid_shape=[rows, C1A_PROJ_K_TILE], target_memory=pl.Mem.Mat)
            b = pl.load(weight, [0, col], [C1A_PROJ_K_TILE, C1A_PROJ_N_TILE], target_memory=pl.Mem.Mat)
            full_a = pl.set_validshape(a, C1A_PROJ_M_TILE, C1A_PROJ_K_TILE)
            acc = pl.matmul(full_a, b)
            for k in pl.pipeline(C1A_PROJ_K_TILE, D, C1A_PROJ_K_TILE, stage=2):
                a = pl.load(source, [row, k], [C1A_PROJ_M_TILE, C1A_PROJ_K_TILE], valid_shape=[rows, C1A_PROJ_K_TILE], target_memory=pl.Mem.Mat)
                b = pl.load(weight, [k, col], [C1A_PROJ_K_TILE, C1A_PROJ_N_TILE], target_memory=pl.Mem.Mat)
                full_a = pl.set_validshape(a, C1A_PROJ_M_TILE, C1A_PROJ_K_TILE)
                acc = pl.matmul_acc(acc, full_a, b)
            value = pl.set_validshape(acc, rows, C1A_PROJ_N_TILE)
            output = pl.store(value, [row, col], output)
    return output, completed


@pl.jit.inline(auto_scope=False)
def c1a_compressor_publish(
    source: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    scales: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    """Publish group-16 MXFP4 with BF16 scales and nearest-even rounding."""
    cache_rows = pl.tensor.dim(cache, 0) * 128
    flat = pl.reshape(cache, [cache_rows, HEAD_DIM // 2])
    flat_scale = pl.reshape(scales, [cache_rows, HEAD_DIM // COMPRESSED_CACHE_GROUP])
    # Each physical cache page has one publishing worker.
    workers = pl.min(num_tokens, 8)
    with pl.spmd(workers, name_hint="pub_cmp", deps=[cache_ready]) as published:
        worker = pl.tile.get_block_idx()
        for t in pl.range(num_tokens):
            slot = pl.read(slots, [t])
            if slot >= 0 and (slot // 128) % workers == worker:
                physical = pl.cast(slot, pl.INDEX)
                row = pl.tile.load(source, [t, 0], [1, HEAD_DIM])
                values = pl.reshape(pl.cast(row, pl.FP32), [HEAD_DIM // COMPRESSED_CACHE_GROUP, COMPRESSED_CACHE_GROUP])
                temporary = pl.tile.create([HEAD_DIM // COMPRESSED_CACHE_GROUP, COMPRESSED_CACHE_GROUP], dtype=pl.FP32)
                amax = pl.row_max(pl.abs(values), tmp_tile=temporary)
                factors_raw = pl.maximum(pl.mul(amax, 1.0 / 6.0), 0.001953125)
                stored_scale = pl.cast(factors_raw, pl.BF16, mode="rint")
                factors = pl.cast(stored_scale, pl.FP32)
                payload_bytes = encode_fp4(values, factors, HEAD_DIM // COMPRESSED_CACHE_GROUP, COMPRESSED_CACHE_GROUP)
                payload = pl.set_validshape(payload_bytes, 1, HEAD_DIM // 2)
                scale_row = pl.reshape(stored_scale, [1, HEAD_DIM // COMPRESSED_CACHE_GROUP])
                encoded = pl.set_validshape(scale_row, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP)
                pl.tile.store(payload, [physical, 0], flat)
                pl.tile.store(encoded, [physical, 0], flat_scale)
    return published


@pl.jit.inline
def _c2a_compressor_project_tile(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.ND],
    output: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    row: pl.Scalar[pl.INDEX],
    col: pl.Scalar[pl.INDEX],
    rows: pl.Scalar[pl.INDEX],
):
    a = pl.load(x, [row, 0], [C2A_PROJ_M_TILE, C2A_PROJ_K_TILE], valid_shape=[rows, C2A_PROJ_K_TILE], target_memory=pl.Mem.Mat)
    b = pl.load(weight, [0, col], [C2A_PROJ_K_TILE, C2A_PROJ_N_TILE], target_memory=pl.Mem.Mat)
    a_full = pl.set_validshape(a, C2A_PROJ_M_TILE, C2A_PROJ_K_TILE)
    acc = pl.matmul(a_full, b)
    for k in pl.pipeline(C2A_PROJ_K_TILE, D, C2A_PROJ_K_TILE, stage=2):
        a = pl.load(x, [row, k], [C2A_PROJ_M_TILE, C2A_PROJ_K_TILE], valid_shape=[rows, C2A_PROJ_K_TILE], target_memory=pl.Mem.Mat)
        b = pl.load(weight, [k, col], [C2A_PROJ_K_TILE, C2A_PROJ_N_TILE], target_memory=pl.Mem.Mat)
        a_full = pl.set_validshape(a, C2A_PROJ_M_TILE, C2A_PROJ_K_TILE)
        acc = pl.matmul_acc(acc, a_full, b)
    value = pl.set_validshape(acc, rows, C2A_PROJ_N_TILE)
    output = pl.store(value, [row, col], output)
    return output


@pl.jit.inline
def c2a_compressor_project(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.ND],
    wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.ND],
    kv_out: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    score_out: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Project KV and gate with ND BF16 weights and FP32 Cube output."""
    blocks = (num_tokens + C2A_PROJ_M_TILE - 1) // C2A_PROJ_M_TILE * (2 * HEAD_DIM // C2A_PROJ_N_TILE)
    with pl.spmd(pl.max(blocks, 1), name_hint="proj_cmp"):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_cmp"):
            block = pl.tile.get_block_idx()
            row = block // (2 * HEAD_DIM // C2A_PROJ_N_TILE) * C2A_PROJ_M_TILE
            col = block % (HEAD_DIM // C2A_PROJ_N_TILE) * C2A_PROJ_N_TILE
            rows = pl.max(pl.min(C2A_PROJ_M_TILE, num_tokens - row), 0)
            if block % (2 * HEAD_DIM // C2A_PROJ_N_TILE) < HEAD_DIM // C2A_PROJ_N_TILE:
                kv_out = _c2a_compressor_project_tile(x, wkv, kv_out, row, col, rows)
            else:
                score_out = _c2a_compressor_project_tile(x, wgate, score_out, row, col, rows)
    return kv_out, score_out


@pl.jit.inline
def c2a_compressor_pool(
    kv_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    score_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    num_state_blocks: pl.Scalar[pl.INT32],
    pair_ready: pl.Scalar[pl.TASK_ID],
):
    """Assemble partners and pool eight rows before updating ring state."""
    state_rows = num_state_blocks * STATE_CAPACITY
    flat = pl.reshape(state_cache, [state_rows, STATE_WIDTH])
    with pl.spmd(pl.max((num_tokens + 7) // 8, 1), name_hint="pool_cmp", deps=[pair_ready]) as pool_tid:
        group = pl.tile.get_block_idx()
        t = group * 8
        if t < num_tokens:
            rows = pl.min(8, num_tokens - t)
            current_kv = pl.slice(kv_proj, [8, HEAD_DIM], [t, 0], valid_shape=[rows, HEAD_DIM])
            current_kv = pl.set_validshape(pl.fillpad(current_kv, pad_value=pl.PadValue.zero), 8, HEAD_DIM)
            current_score = pl.slice(score_proj, [8, HEAD_DIM], [t, 0], valid_shape=[rows, HEAD_DIM])
            current_score = pl.set_validshape(pl.fillpad(current_score, pad_value=pl.PadValue.zero), 8, HEAD_DIM)
            previous_kv = pl.full([8, HEAD_DIM], dtype=pl.FP32, value=0.0)
            previous_score = pl.full([8, HEAD_DIM], dtype=pl.FP32, value=0.0)
            for lane in pl.unroll(8):
                token = t + lane
                if token < num_tokens:
                    request = pl.cast(pl.read(token_to_req_indices, [token]), pl.INDEX)
                    if request >= 0 and request < pl.tensor.dim(state_block_table, 0):
                        block = pl.cast(pl.read(state_block_table, [request, 0]), pl.INDEX)
                        if block >= 0 and block < num_state_blocks:
                            start = pl.read(query_start_loc, [request])
                            end = pl.read(query_start_loc, [request + 1])
                            position = pl.read(position_ids, [token])
                            if token >= start and token < end and position >= 0 and position % 2 == 1:
                                if token > start:
                                    partner_kv = kv_proj[token - 1 : token, :]
                                    partner_score = score_proj[token - 1 : token, :]
                                else:
                                    slot = pl.cast(block * STATE_CAPACITY + (position - 1) % STATE_CAPACITY, pl.INDEX)
                                    partner_kv = flat[slot : slot + 1, :HEAD_DIM]
                                    partner_score = flat[slot : slot + 1, HEAD_DIM:]
                                previous_kv[lane : lane + 1, :] = partner_kv
                                previous_score[lane : lane + 1, :] = partner_score
            highest = pl.maximum(previous_score, current_score)
            previous_weight = pl.exp(pl.sub(previous_score, highest))
            current_weight = pl.exp(pl.sub(current_score, highest))
            previous_weighted = pl.mul(previous_kv, previous_weight)
            current_weighted = pl.mul(current_kv, current_weight)
            weighted = pl.add(previous_weighted, current_weighted)
            pooled = pl.div(weighted, pl.add(previous_weight, current_weight))
            # Round the pooled row to BF16 before normalizing.
            value = pl.cast(pl.cast(pooled, pl.BF16, mode="rint"), pl.FP32)
            square = pl.mul(pl.row_sum(pl.mul(value, value)), 1.0 / HEAD_DIM)
            inv = pl.rsqrt(pl.add(square, EPS), high_precision=True)
            gamma = pl.reshape(pl.cast(norm_weight[:], pl.FP32), [1, HEAD_DIM])
            normalized = pl.col_expand_mul(pl.row_expand_mul(value, inv), gamma)
            parity = pl.full([8, 8], dtype=pl.FP32, value=0.0)
            for lane in pl.unroll(8):
                token = t + lane
                if token < num_tokens:
                    request = pl.cast(pl.read(token_to_req_indices, [token]), pl.INDEX)
                    if request >= 0 and request < pl.tensor.dim(state_block_table, 0):
                        block = pl.cast(pl.read(state_block_table, [request, 0]), pl.INDEX)
                        if block >= 0 and block < num_state_blocks:
                            start = pl.read(query_start_loc, [request])
                            end = pl.read(query_start_loc, [request + 1])
                            position = pl.read(position_ids, [token])
                            if token >= start and token < end and position >= 0 and position % 2 == 1:
                                pl.write(parity, [lane, 0], 1.0)
            parity = pl.row_sum(parity)
            published = pl.row_expand_mul(normalized, parity)
            published_bf16 = pl.cast(published, pl.BF16, mode="rint")
            latent[t : t + 8, :] = pl.set_validshape(published_bf16, rows, HEAD_DIM)
    return pool_tid


@pl.jit.inline
def c2a_compressor_state_write(
    kv_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    score_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    pair_ready: pl.Scalar[pl.TASK_ID],
):
    """Save only the final capacity rows of each chunk after all historical reads."""
    blocks = pl.tensor.dim(state_cache, 0)
    state_rows = blocks * STATE_CAPACITY
    flat = pl.reshape(state_cache, [state_rows, STATE_WIDTH])
    workers = pl.max(pl.min(num_tokens, 16), 1)
    with pl.spmd(workers, name_hint="state_cmp", deps=[pair_ready]) as state_tid:
        worker = pl.tile.get_block_idx()
        for t in pl.range(worker, num_tokens, workers):
            request = pl.cast(pl.read(token_to_req_indices, [t]), pl.INDEX)
            if request >= 0 and request < pl.tensor.dim(state_block_table, 0):
                block = pl.cast(pl.read(state_block_table, [request, 0]), pl.INDEX)
                if block >= 0 and block < blocks:
                    start = pl.read(query_start_loc, [request])
                    end = pl.min(pl.read(query_start_loc, [request + 1]), num_tokens)
                    position = pl.read(position_ids, [t])
                    if t >= start and t < end and t >= end - STATE_CAPACITY and position >= 0:
                        slot = pl.cast(block * STATE_CAPACITY + position % STATE_CAPACITY, pl.INDEX)
                        flat[slot : slot + 1, :HEAD_DIM] = kv_proj[t : t + 1, :]
                        flat[slot : slot + 1, HEAD_DIM:] = score_proj[t : t + 1, :]
    return state_tid


@pl.jit.inline
def c2a_compressor_pool_state(
    kv_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    score_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    state_ready: pl.Scalar[pl.TASK_ID],
):
    """Pool completed pairs and persist request state using a shared prefill/decode ABI."""
    blocks = pl.cast(pl.tensor.dim(state_cache, 0), pl.INT32)
    pool_tid = c2a_compressor_pool(
        kv_proj, score_proj, state_cache, position_ids, norm_weight, latent,
        num_tokens, query_start_loc, token_to_req_indices, state_block_table, blocks, state_ready,
    )
    state_tid = c2a_compressor_state_write(
        kv_proj, score_proj, query_start_loc, position_ids, token_to_req_indices,
        state_block_table, state_cache, num_tokens, pool_tid,
    )
    return pool_tid, state_tid


@pl.jit.inline(auto_scope=False)
def c2a_compressor_publish(
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    cos: pl.Tensor,
    sin: pl.Tensor,
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, (HEAD_DIM // 2)], pl.UINT8],
    scales: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, (HEAD_DIM // 16)], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    source_ready: pl.Scalar[pl.TASK_ID],
    max_workers: pl.constexpr,
):
    """Rotate and publish compressed rows with one worker per physical page."""
    blocks = pl.tensor.dim(cache, 0)
    cache_rows = blocks * 128
    flat = pl.reshape(cache, [cache_rows, (HEAD_DIM // 2)])
    scale_flat = pl.reshape(scales, [cache_rows, (HEAD_DIM // 16)])
    workers = pl.min(num_tokens, max_workers)
    with pl.spmd(workers, name_hint="rope_pub_cmp", deps=[cache_ready, source_ready]) as publish_tid:
        worker = pl.tile.get_block_idx()
        for t in pl.range(num_tokens):
            slot_i64 = pl.read(slots, [t])
            if slot_i64 >= 0 and (slot_i64 // 128) % workers == worker:
                slot = pl.cast(slot_i64, pl.INDEX)
                row = pl.tile.load(latent, [t, 0], [1, HEAD_DIM])
                tail_bf16 = pl.tile.slice(row, [1, ROPE_DIM], [0, HEAD_DIM - ROPE_DIM])
                tail = pl.cast(tail_bf16, pl.FP32)
                even = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P0101)
                odd = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P1010)
                c = pl.tile.load(cos, [t, 0], [1, ROPE_DIM // 2])
                sine = pl.tile.load(sin, [t, 0], [1, ROPE_DIM // 2])
                s = pl.mul(sine, 1.0)
                even_cos = pl.mul(even, c)
                odd_sin = pl.mul(odd, s)
                re = pl.sub(even_cos, odd_sin)
                even_sin = pl.mul(even, s)
                odd_cos = pl.mul(odd, c)
                im = pl.add(even_sin, odd_cos)
                real_lanes = pl.tile.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
                real_lanes = pl.tile.scatter_mask(real_lanes, re, mask_pattern=pl.tile.MaskPattern.P0101)
                imag_lanes = pl.tile.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
                imag_lanes = pl.tile.scatter_mask(imag_lanes, im, mask_pattern=pl.tile.MaskPattern.P1010)
                real_bits = pl.reinterpret_view(real_lanes, pl.INT16)
                imag_bits = pl.reinterpret_view(imag_lanes, pl.INT16)
                rotated_bits = pl.or_(real_bits, imag_bits)
                rotated = pl.reinterpret_view(rotated_bits, pl.FP32)
                rounded = pl.cast(rotated, pl.BF16, mode="rint")
                row = pl.tile.assemble(row, rounded, [0, HEAD_DIM - ROPE_DIM])
                source = pl.tile.full([1, HEAD_DIM * 2], dtype=pl.BF16, value=0.0)
                source = pl.tile.assemble(source, row, [0, 0])
                groups = pl.reshape(pl.cast(source, pl.FP32), [HEAD_DIM * 2 // 16, 16])
                scratch = pl.tile.create([HEAD_DIM * 2 // 16, 16], dtype=pl.FP32)
                amax = pl.tile.row_max(pl.abs(groups), tmp_tile=scratch)
                stored = pl.cast(pl.maximum(pl.mul(amax, (1.0 / 6.0)), (2.0**-9)), pl.BF16, mode="rint")
                scale = pl.cast(stored, pl.FP32)
                payload_bytes = encode_fp4(groups, scale, HEAD_DIM * 2 // 16, 16)
                pl.tile.store(pl.set_validshape(payload_bytes, 1, HEAD_DIM // 2), [slot, 0], flat)
                scale_row = pl.reshape(stored, [1, HEAD_DIM * 2 // 16])
                pl.tile.store(pl.set_validshape(scale_row, 1, HEAD_DIM // 16), [slot, 0], scale_flat)
    return publish_tid


def golden_c1a_compressor(
    x: torch.Tensor,
    wkv: torch.Tensor,
    norm_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute C1A ND-weight projection followed by RMSNorm."""
    projected = torch.matmul(x.float(), wkv.float()).to(x.dtype)
    return golden_rms_norm(projected, norm_weight)


def golden_c2a_compressor(
    x: torch.Tensor,
    query_start_loc: torch.Tensor,
    position_ids: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    state_block_table: torch.Tensor,
    state_cache: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute C2A compression with request-owned FP32 ring state.

    Query boundaries count valid packed tokens; any tensor tail is padding.
    Negative/out-of-range request or block indices suppress state access and
    publication. Valid chunks contain consecutive absolute positions.
    """
    kv = torch.matmul(x.float(), wkv.float())
    score = torch.matmul(x.float(), wgate.float())
    latent = torch.zeros_like(kv, dtype=x.dtype)
    publish = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    blocks, capacity, width = state_cache.shape
    if capacity < 1 or width != 2 * kv.shape[-1] or state_cache.dtype != torch.float32:
        raise ValueError("state_cache must be FP32 [blocks, positive capacity, 2 * head_dim]")
    head_dim = kv.shape[-1]
    for request in range(state_block_table.shape[0]):
        start = int(query_start_loc[request])
        end = min(int(query_start_loc[request + 1]), x.shape[0])
        block = int(state_block_table[request, 0])
        if start >= end or not 0 <= block < blocks:
            continue
        for token in range(start, end):
            position = int(position_ids[token])
            if int(token_to_req_indices[token]) != request or position < 0:
                continue
            if position % 2:
                if token > start:
                    previous_kv, previous_score = kv[token - 1], score[token - 1]
                else:
                    previous = state_cache[block, (position - 1) % capacity]
                    previous_kv, previous_score = previous[:head_dim], previous[head_dim:]
                pair_kv = torch.stack((previous_kv, kv[token]))
                pair_score = torch.stack((previous_score, score[token]))
                pooled = (pair_kv * pair_score.softmax(dim=0)).sum(dim=0)
                latent[token] = golden_rms_norm(pooled.to(x.dtype), norm_weight)
                publish[token] = True
        for token in range(max(start, end - capacity), end):
            position = int(position_ids[token])
            if int(token_to_req_indices[token]) == request and position >= 0:
                state_cache[block, position % capacity, :head_dim] = kv[token]
                state_cache[block, position % capacity, head_dim:] = score[token]
    return latent, publish


@pl.jit(auto_scope=False)
def l2_compressor(
    source: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    positions: pl.Tensor[[T_DYN], pl.INT32],
    requests: pl.Tensor[[T_DYN], pl.INT32],
    starts: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    ratio: pl.Scalar[pl.INT32],
    state: pl.InOut[pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    latent: pl.InOut[pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16]],
    cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    scales: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]],
):
    entry = pl.system.task_dummy(deps=[])
    tokens = pl.tensor.dim(source, 0)
    if ratio == 1:
        projected = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        projected, projected_tid = c1a_compressor_project(source, wkv, projected, num_tokens, entry)
        with pl.spmd((num_tokens + 7) // 8, name_hint="norm_cmp", deps=[projected_tid]) as normalized:
            worker = pl.tile.get_block_idx()
            rms_norm(projected, norm_weight, latent, worker, num_tokens, width=HEAD_DIM)
        rotated = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        compressor_rope(latent, cos, sin, rotated, num_tokens)
        c1a_compressor_publish(rotated, slots, cache, scales, num_tokens, normalized)
    else:
        kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.FP32)
        score = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.FP32)
        c2a_compressor_project(source, wkv, wgate, kv, score, num_tokens)
        pool_tid, state_tid = c2a_compressor_pool_state(
            kv, score, starts, positions, requests, state_table, state, norm_weight, latent, num_tokens, entry,
        )
        c2a_compressor_publish(latent, cos, sin, slots, cache, scales, num_tokens, entry, pool_tid, max_workers=4)
    return state, latent, cache, scales


def golden_compressor(tensors):
    """Validate compression, ring-state writes, rotary positions and group-16 cache publication."""
    from .qkv_proj_rope import golden_rope_tail
    from .quant import quantize_mxfp4_cache

    t = tensors
    active = int(t["num_tokens"])
    if int(t["ratio"]) == 1:
        latent = golden_c1a_compressor(t["source"][:active], t["wkv"], t["norm_weight"])
    else:
        latent, _ = golden_c2a_compressor(
            t["source"][:active], t["starts"], t["positions"][:active], t["requests"][:active],
            t["state_table"], t["state"], t["wkv"], t["wgate"], t["norm_weight"],
        )
    t["latent"][:active] = latent
    rotated = golden_rope_tail(latent, t["cos"][:active], t["sin"][:active])
    codes, factors = quantize_mxfp4_cache(rotated, 16, "bf16", round_to_even=True)
    slots = t["slots"][:active].long()
    valid = slots >= 0
    t["cache"].reshape(-1, HEAD_DIM // 2)[slots[valid]] = codes[valid]
    t["scales"].reshape(-1, HEAD_DIM // 16)[slots[valid]] = factors[valid]


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    from golden import ScalarSpec, TensorSpec, mapped_pool_ratio_allclose, ratio_allclose, run
    from .quant import mapped_fp4_cache_compare
    from .weights import Checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3"], default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ratio", type=int, choices=[1, 2])
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    checkpoint = Checkpoint(args.checkpoint)
    torch.set_num_threads(8)
    records = []
    ratios = (args.ratio,) if args.ratio else (1, 2)
    for ratio in ratios:
        layer = 20 if ratio == 1 else 2
        prefix = f"layers.{layer}.attn.compressor."
        wkv = checkpoint.tensor(prefix + "wkv.weight").t().contiguous()
        wgate = checkpoint.tensor(prefix + "wgate.weight").t().contiguous() if ratio == 2 else torch.zeros_like(wkv)
        norm = checkpoint.tensor(prefix + "norm.weight")
        for start, active in ((0, 129), (127, 3), (511, 3), (8191, 3), (8192, 2)):
            rows = (active + 15) // 16 * 16
            generator = torch.Generator().manual_seed(410710 + start)
            positions = torch.full((rows,), -1, dtype=torch.int32)
            positions[:active] = torch.arange(start, start + active, dtype=torch.int32)
            slots = positions.long().clone()
            if ratio == 2:
                slots = torch.where((positions >= 0) & (positions % 2 == 1), positions.long() // 2, -1)
            rope_positions = positions.clamp_min(0)
            if ratio == 2:
                rope_positions = rope_positions // 2 * 2
            angles = rope_positions.float().unsqueeze(-1) * 160000.0 ** (-torch.arange(0, 64, 2).float() / 64)
            requests = torch.full((rows,), -1, dtype=torch.int32)
            requests[:active] = 0
            values = dict(source=torch.randn(rows, D, generator=generator).bfloat16(), wkv=wkv, wgate=wgate,
                          norm_weight=norm, positions=positions, requests=requests,
                          starts=torch.tensor([0, active], dtype=torch.int32),
                          state_table=torch.zeros((1, 1), dtype=torch.int32), slots=slots,
                          cos=angles.cos(), sin=angles.sin(), num_tokens=active, ratio=ratio,
                          state=torch.randn(2, STATE_CAPACITY, STATE_WIDTH, generator=generator),
                          latent=torch.full((rows, HEAD_DIM), 7.25, dtype=torch.bfloat16),
                          cache=torch.full((66, 128, 1, HEAD_DIM // 2), 199, dtype=torch.uint8),
                          scales=torch.full((66, 128, 1, HEAD_DIM // 16), 2.0, dtype=torch.bfloat16))
            specs = [ScalarSpec(name, torch.int32, value) if name in ("num_tokens", "ratio")
                     else TensorSpec(name, list(value.shape), value.dtype, init_value=value)
                     for name, value in values.items()]
            compare = {"state": ratio_allclose(atol=0.003, rtol=0.003, max_error_ratio=0.005),
                       "latent": ratio_allclose(atol=0.003, rtol=0.03, max_error_ratio=0.005),
                       "cache": mapped_fp4_cache_compare("slots", mapping_shape=(rows,), block_size=128,
                                                        pool_name="cache", max_error_ratio=0.005),
                       "scales": mapped_pool_ratio_allclose("slots", mapping_shape=(rows,), block_size=128,
                                                            pool_name="scales", atol=0.003, rtol=0.03)}
            result = run(
                l2_compressor,
                specs,
                golden_fn=golden_compressor,
                compare_fn=compare,
                config=dict(backend_type=args.platform, execution_mode=False, device_id=args.device),
                compile_only=args.compile_only,
                save_data=True,
            )
            records.append(dict(ratio=ratio, start=start, active=active, passed=result.passed,
                                work_dir=str(result.work_dir)))
            if not result.passed:
                raise RuntimeError(result.error)
    if args.report:
        args.report.write_text(json.dumps(dict(checkpoint=checkpoint.identity, cases=records), indent=2) + "\n")

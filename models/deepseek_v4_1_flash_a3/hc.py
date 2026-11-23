# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 HC coefficient generation, stream collapse, residual expansion, and final collapse."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

import math

import pypto.language as pl
import torch
import torch.nn.functional as F

from .config import FLASH as M, SP_T_DYN, T_DYN


# model config
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
NORM_EPS = M.rms_norm_eps

# tiling
MIX_PAD = 32
HC_PAD = 8
T_TILE = 8
LINEAR_T_TILE = 16
SINKHORN_LOAD_T_TILE = 8
SINKHORN_T_TILE = 2
SINKHORN_WORKER_TILE = 24
RMS_K_TILE = 1024
LINEAR_K_TILE = 256
LINEAR_OK = 4
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK
D_TILE = 2560
PRE_T_TILE = 2
HEAD_D_TILE = 5120
PRE_NORM_T_TILE = 8
PRE_NORM_D_TILE = 256
POST_WORKER_TILE = 64
PREFILL_RMS_D_TILE = 256
PREFILL_RMS_ROWS = HC_DIM // PREFILL_RMS_D_TILE
PREFILL_RMS_T_TILE = 8
PREFILL_PROJ_T_TILE = 32
PREFILL_PROJ_N_TILE = 32
PREFILL_PROJ_K_TILE = 1024
PREFILL_SINKHORN_T_TILE = 16
PREFILL_WORKER_TILE = 48
PREFILL_MERGE_T_TILE = 16
PREFILL_POST_D_TILE = 2560
PREFILL_POST_WORKERS = 48


@pl.jit.inline
def hc_sinkhorn(
    inv_rms: pl.Tensor,
    partials: pl.Tensor,
    scale: pl.Tensor,
    base: pl.Tensor,
    pre_mix: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
):
    """Normalize independent HC matrices in two-row tiles."""
    tokens = pl.tensor.dim(pre_mix, 0)
    linear_tokens = pl.tensor.dim(inv_rms, 0)
    base_row = pl.reshape(base, [1, MIX_HC])
    residual_flat = pl.reshape(residual_mix, [tokens, HC_MULT * HC_MULT])
    pre_flat = pl.reshape(pre_mix, [1, tokens * HC_MULT])
    post_flat = pl.reshape(post_mix, [1, tokens * HC_MULT])
    scale0 = pl.read(scale, [0])
    scale1 = pl.read(scale, [1])
    scale2 = pl.read(scale, [2])
    blocks = (tokens + SINKHORN_T_TILE - 1) // SINKHORN_T_TILE
    workers = pl.min(blocks, SINKHORN_WORKER_TILE)
    with pl.spmd(workers, name_hint="mhc_sinkhorn") as ready:
        worker = pl.tile.get_block_idx()
        lanes = pl.tile.arange(0, [1, HC_PAD * HC_PAD], dtype=pl.INT32)
        lane_values = pl.cast(lanes, pl.FP32)
        token = pl.cast(pl.shrs(lanes, 5), pl.FP32)
        wide_row = pl.cast(pl.shrs(lanes, 3), pl.FP32)
        row = pl.sub(wide_row, pl.mul(token, 4.0))
        groups = pl.cast(pl.shrs(lanes, 2), pl.FP32)
        column = pl.sub(lane_values, pl.mul(groups, 4.0))
        token_offset = pl.mul(token, 32.0)
        row_offset = pl.mul(row, 4.0)
        indices = pl.add(token_offset, row_offset)
        indices = pl.add(indices, column)
        indices = pl.add(indices, 8.0)
        gather_indices = pl.cast(indices, pl.INT32, mode="rint")
        gather_tmp = pl.create_tile([1, HC_PAD * HC_PAD], dtype=pl.INT32)
        column_token = pl.sub(groups, pl.mul(wide_row, 2.0))
        different_token = pl.abs(pl.sub(token, column_token))
        diagonal = pl.add(pl.mul(different_token, -1.0), 1.0)
        diagonal = pl.reshape(diagonal, [HC_PAD, HC_PAD])
        output_lanes = pl.tile.arange(0, [1, 2 * HC_MULT * HC_MULT], dtype=pl.INT32)
        output_lane_values = pl.cast(output_lanes, pl.FP32)
        output_token = pl.cast(pl.shrs(output_lanes, 4), pl.FP32)
        output_groups = pl.cast(pl.shrs(output_lanes, 2), pl.FP32)
        output_row = pl.sub(output_groups, pl.mul(output_token, 4.0))
        output_column = pl.sub(output_lane_values, pl.mul(output_groups, 4.0))
        output_token_offset = pl.mul(output_token, 36.0)
        output_row_offset = pl.mul(output_row, 8.0)
        output_indices = pl.add(output_token_offset, output_row_offset)
        output_indices = pl.add(output_indices, output_column)
        output_gather_indices = pl.cast(output_indices, pl.INT32, mode="rint")
        output_tmp = pl.create_tile([1, 2 * HC_MULT * HC_MULT], dtype=pl.INT32)
        coefficient_lanes = pl.tile.arange(0, [1, SINKHORN_T_TILE * HC_MULT], dtype=pl.INT32)
        coefficient_token = pl.shrs(coefficient_lanes, 2)
        coefficient_token = pl.cast(coefficient_token, pl.FP32)
        coefficient_values = pl.cast(coefficient_lanes, pl.FP32)
        coefficient_offsets = pl.mul(coefficient_token, 1.0 * (MIX_PAD - HC_MULT))
        pre_indices = pl.add(coefficient_values, coefficient_offsets)
        post_indices = pl.add(pre_indices, 1.0 * HC_MULT)
        pre_indices = pl.cast(pre_indices, pl.INT32, mode="rint")
        post_indices = pl.cast(post_indices, pl.INT32, mode="rint")
        coefficient_tmp = pl.create_tile([1, SINKHORN_T_TILE * HC_MULT], dtype=pl.INT32)
        for block in pl.range(worker, blocks, workers):
            t0 = block * SINKHORN_T_TILE
            valid_rows = pl.min(SINKHORN_T_TILE, tokens - t0)
            load_t0 = t0 // SINKHORN_LOAD_T_TILE * SINKHORN_LOAD_T_TILE
            local_t0 = t0 - load_t0
            total = pl.tile.full([SINKHORN_T_TILE, MIX_PAD], dtype=pl.FP32, value=0.0)
            for split in pl.unroll(LINEAR_OK):
                offset = split * linear_tokens + t0
                part = pl.load(partials, [offset, 0], [SINKHORN_T_TILE, MIX_PAD])
                total = pl.add(total, part)
            inverse_group = pl.load(inv_rms, [load_t0, 0], [SINKHORN_LOAD_T_TILE, 1])
            inverse0 = pl.tile.read(inverse_group, [local_t0, 0])
            inverse1 = pl.tile.read(inverse_group, [local_t0 + 1, 0])
            row0 = pl.tile.slice(total, [1, MIX_PAD], [0, 0])
            row1 = pl.tile.slice(total, [1, MIX_PAD], [1, 0])
            normalized0 = pl.mul(row0, inverse0)
            normalized1 = pl.mul(row1, inverse1)
            normalized = pl.create_tile([SINKHORN_T_TILE, MIX_PAD], dtype=pl.FP32)
            normalized = pl.tile.assemble(normalized, normalized0, [0, 0])
            normalized = pl.tile.assemble(normalized, normalized1, [1, 0])
            bias = pl.load(base_row, [0, 0], [1, MIX_PAD], valid_shape=[1, MIX_HC])
            bias = pl.fillpad(bias, pad_value=pl.PadValue.zero)
            pre_scaled = pl.mul(normalized, scale0)
            pre_logits = pl.col_expand_add(pre_scaled, bias)
            pre_logits_flat = pl.reshape(pre_logits, [1, SINKHORN_T_TILE * MIX_PAD])
            pre_logits = pl.tile.gather(pre_logits_flat, pre_indices, coefficient_tmp)
            pre_negated = pl.neg(pre_logits)
            pre_exp = pl.exp(pre_negated)
            pre_denominator = pl.add(pre_exp, 1.0)
            pre = pl.recip(pre_denominator)
            pre = pl.add(pre, HC_EPS)
            pre = pl.set_validshape(pre, 1, valid_rows * HC_MULT)
            pl.store(pre, [0, t0 * HC_MULT], pre_flat)
            post_scaled = pl.mul(normalized, scale1)
            post_logits = pl.col_expand_add(post_scaled, bias)
            post_logits_flat = pl.reshape(post_logits, [1, SINKHORN_T_TILE * MIX_PAD])
            post_logits = pl.tile.gather(post_logits_flat, post_indices, coefficient_tmp)
            post_negated = pl.neg(post_logits)
            post_exp = pl.exp(post_negated)
            post_denominator = pl.add(post_exp, 1.0)
            post = pl.recip(post_denominator)
            post = pl.mul(post, 2.0)
            post = pl.set_validshape(post, 1, valid_rows * HC_MULT)
            pl.store(post, [0, t0 * HC_MULT], post_flat)
            comb_scaled = pl.mul(normalized, scale2)
            comb_logits = pl.col_expand_add(comb_scaled, bias)
            comb_logits = pl.slice(comb_logits, [SINKHORN_T_TILE, MIX_PAD], [0, 0])
            comb_logits = pl.mul(comb_logits, 1.0)
            comb_logits = pl.set_validshape(comb_logits, valid_rows, MIX_HC)
            comb_logits = pl.fillpad(comb_logits, pad_value=pl.PadValue.zero)
            comb_logits = pl.set_validshape(comb_logits, SINKHORN_T_TILE, MIX_PAD)
            flat_logits = pl.reshape(comb_logits, [1, SINKHORN_T_TILE * MIX_PAD])
            gathered = pl.tile.gather(flat_logits, gather_indices, gather_tmp)
            rows = pl.reshape(gathered, [HC_PAD, HC_PAD])
            row_tmp = pl.create_tile([HC_PAD, HC_PAD], dtype=pl.FP32)
            maximum = pl.row_max(rows, row_tmp)
            rows = pl.row_expand_sub(rows, maximum)
            rows = pl.exp(rows)
            rows = pl.mul(rows, diagonal)
            row_sum = pl.row_sum(rows, row_tmp)
            rows = pl.row_expand_div(rows, row_sum)
            rows = pl.add(rows, HC_EPS)
            rows = pl.mul(rows, diagonal)
            column_sum = pl.col_sum(rows)
            column_sum = pl.add(column_sum, HC_EPS)
            rows = pl.col_expand_div(rows, column_sum)
            for _ in pl.unroll(HC_SINKHORN_ITER - 1):
                row_sum = pl.row_sum(rows, row_tmp)
                row_sum = pl.add(row_sum, HC_EPS)
                rows = pl.row_expand_div(rows, row_sum)
                column_sum = pl.col_sum(rows)
                column_sum = pl.add(column_sum, HC_EPS)
                rows = pl.col_expand_div(rows, column_sum)
            flat_rows = pl.reshape(rows, [1, HC_PAD * HC_PAD])
            packed = pl.tile.gather(flat_rows, output_gather_indices, output_tmp)
            output = pl.reshape(packed, [SINKHORN_T_TILE, HC_MULT * HC_MULT])
            output = pl.set_validshape(output, valid_rows, HC_MULT * HC_MULT)
            pl.store(output, [t0, 0], residual_flat)
    return ready


@pl.jit.inline
def hc_project(x_hc: pl.Tensor, function: pl.Tensor):
    """Compute the inverse RMS and split-K coefficient projection."""
    t_dim = pl.tensor.dim(x_hc, 0)
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])

    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    for block in pl.spmd((t_dim + T_TILE - 1) // T_TILE, name_hint="mhc_rms"):
        t0 = block * T_TILE
        valid_rows = pl.min(T_TILE, t_dim - t0)
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HC_DIM // RMS_K_TILE, stage=4):
            k0 = kb * RMS_K_TILE
            rms_x_tile = pl.slice(
                x_flat,
                [T_TILE, RMS_K_TILE],
                [t0, k0],
                valid_shape=[valid_rows, RMS_K_TILE],
            )
            sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(rms_x_tile, rms_x_tile)), [1, T_TILE]))
        rms_arg = pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS)
        inv_rms[t0 : t0 + T_TILE, 0:1] = pl.reshape(
            pl.rsqrt(rms_arg, high_precision=True), [T_TILE, 1]
        )

    mixes_partials = pl.create_tensor([LINEAR_OK * t_linear, MIX_PAD], dtype=pl.FP32)
    for task in pl.spmd((t_linear // LINEAR_T_TILE) * LINEAR_OK, name_hint="mhc_linear"):
        t0 = (task // LINEAR_OK) * LINEAR_T_TILE
        split = task % LINEAR_OK
        k_base = split * LINEAR_K_PER_SPLIT
        valid_rows = pl.min(LINEAR_T_TILE, t_dim - t0)
        acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
        for kb in pl.pipeline(LINEAR_K_PER_SPLIT // LINEAR_K_TILE, stage=2):
            k0 = k_base + kb * LINEAR_K_TILE
            linear_x_tile = pl.slice(
                x_flat,
                [LINEAR_T_TILE, LINEAR_K_TILE],
                [t0, k0],
                valid_shape=[valid_rows, LINEAR_K_TILE],
            )
            w_tile = pl.slice(
                function,
                [MIX_PAD, LINEAR_K_TILE],
                [0, k0],
                valid_shape=[MIX_HC, LINEAR_K_TILE],
            )
            acc = pl.matmul_acc(acc, linear_x_tile, w_tile, b_trans=True, init_cond=(kb == 0))
        partial = split * t_linear + t0
        mixes_partials[partial : partial + LINEAR_T_TILE, 0:MIX_PAD] = acc

    return inv_rms, mixes_partials


@pl.jit.inline
def hc_project_prefill(
    x_hc: pl.Tensor,
    function: pl.Tensor,
    incoming_pre_mix: pl.Tensor,
    output: pl.Tensor,
):
    """Batch Prefill RMS, projection and incoming-stream collapse."""
    tokens = pl.tensor.dim(x_hc, 0)
    flat = pl.reshape(x_hc, [tokens, HC_DIM])
    flat_chunks = pl.reshape(flat, [tokens * PREFILL_RMS_ROWS, PREFILL_RMS_D_TILE])
    inverse = pl.create_tensor([tokens, PREFILL_RMS_T_TILE], dtype=pl.FP32)
    collapsed_chunks = pl.reshape(
        output, [tokens * (D // PREFILL_RMS_D_TILE), PREFILL_RMS_D_TILE]
    )
    workers = pl.min(tokens, PREFILL_WORKER_TILE)
    for worker in pl.spmd(workers, name_hint="mhc_rms"):
        for token in pl.range(worker, tokens, workers):
            chunk_start = token * PREFILL_RMS_ROWS
            values = pl.load(
                flat_chunks,
                [chunk_start, 0],
                [PREFILL_RMS_ROWS, PREFILL_RMS_D_TILE],
            )
            values_fp32 = values

            stream0 = pl.slice(values_fp32, [D // PREFILL_RMS_D_TILE, PREFILL_RMS_D_TILE], [0, 0])
            stream1 = pl.slice(
                values_fp32,
                [D // PREFILL_RMS_D_TILE, PREFILL_RMS_D_TILE],
                [D // PREFILL_RMS_D_TILE, 0],
            )
            stream2 = pl.slice(
                values_fp32,
                [D // PREFILL_RMS_D_TILE, PREFILL_RMS_D_TILE],
                [2 * D // PREFILL_RMS_D_TILE, 0],
            )
            stream3 = pl.slice(
                values_fp32,
                [D // PREFILL_RMS_D_TILE, PREFILL_RMS_D_TILE],
                [3 * D // PREFILL_RMS_D_TILE, 0],
            )
            weight0 = pl.read(incoming_pre_mix, [token, 0])
            weight1 = pl.read(incoming_pre_mix, [token, 1])
            weight2 = pl.read(incoming_pre_mix, [token, 2])
            weight3 = pl.read(incoming_pre_mix, [token, 3])
            first_pair = pl.add(pl.mul(stream0, weight0), pl.mul(stream1, weight1))
            second_pair = pl.add(pl.mul(stream2, weight2), pl.mul(stream3, weight3))
            collapsed = pl.add(first_pair, second_pair)
            collapsed = pl.cast(collapsed, pl.BF16, mode="rint")
            pl.store(
                collapsed,
                [token * (D // PREFILL_RMS_D_TILE), 0],
                collapsed_chunks,
            )

            square = pl.mul(values_fp32, values_fp32)
            sum_tmp = pl.create_tile([PREFILL_RMS_ROWS, PREFILL_RMS_D_TILE], dtype=pl.FP32)
            partial_sums = pl.row_sum(square, sum_tmp)
            square_sum = 0.0
            for chunk in pl.range(PREFILL_RMS_ROWS):
                square_sum = square_sum + pl.tile.read(partial_sums, [chunk, 0])
            zero = pl.tile.full(
                [1, PREFILL_RMS_T_TILE], dtype=pl.FP32, value=0.0
            )
            square_sum_tile = pl.add(zero, square_sum)
            rms_arg = pl.add(pl.mul(square_sum_tile, HC_DIM_INV), NORM_EPS)
            rsqrt_tmp = pl.create_tile([1, PREFILL_RMS_T_TILE], dtype=pl.FP32)
            inverse_tile = pl.tile.rsqrt(rms_arg, rsqrt_tmp, high_precision=True)
            pl.store(inverse_tile, [token, 0], inverse)

    accumulated = pl.create_tensor([tokens, MIX_PAD], dtype=pl.FP32)
    channel_blocks = (MIX_HC + PREFILL_PROJ_N_TILE - 1) // PREFILL_PROJ_N_TILE
    projection_blocks = (tokens + PREFILL_PROJ_T_TILE - 1) // PREFILL_PROJ_T_TILE
    for block in pl.spmd(projection_blocks * channel_blocks, name_hint="mhc_linear"):
        token = (block // channel_blocks) * PREFILL_PROJ_T_TILE
        channel = (block % channel_blocks) * PREFILL_PROJ_N_TILE
        rows = pl.min(PREFILL_PROJ_T_TILE, tokens - token)
        columns = pl.min(PREFILL_PROJ_N_TILE, MIX_HC - channel)
        acc = pl.create_tile(
            [PREFILL_PROJ_T_TILE, PREFILL_PROJ_N_TILE],
            dtype=pl.FP32,
            target_memory=pl.Mem.Acc,
            compact=True,
        )
        for inner in pl.pipeline(0, HC_DIM, PREFILL_PROJ_K_TILE, stage=2):
            source = pl.load(
                flat,
                [token, inner],
                [PREFILL_PROJ_T_TILE, PREFILL_PROJ_K_TILE],
                valid_shape=[rows, PREFILL_PROJ_K_TILE],
                target_memory=pl.Mem.Mat,
            )
            weight = pl.load(
                function,
                [channel, inner],
                [PREFILL_PROJ_N_TILE, PREFILL_PROJ_K_TILE],
                valid_shape=[columns, PREFILL_PROJ_K_TILE],
                target_memory=pl.Mem.Mat,
            )
            acc = pl.matmul_acc(
                acc,
                source,
                pl.tile.transpose_view(weight),
                init_cond=(inner == 0),
            )
        projection = pl.set_validshape(acc, rows, columns)
        pl.store(projection, [token, channel], accumulated)
    return inverse, accumulated


@pl.jit.inline
def hc_sinkhorn_prefill(
    inv_rms: pl.Tensor,
    projected: pl.Tensor,
    scale: pl.Tensor,
    base: pl.Tensor,
    pre_mix: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
):
    """Batch Prefill sigmoid and Sinkhorn coefficient generation."""
    tokens = pl.tensor.dim(pre_mix, 0)
    residual_flat = pl.reshape(residual_mix, [tokens, HC_MULT * HC_MULT])
    bias = pl.reshape(base, [1, MIX_HC])
    scale0 = pl.read(scale, [0])
    scale1 = pl.read(scale, [1])
    scale2 = pl.read(scale, [2])
    blocks = (tokens + PREFILL_SINKHORN_T_TILE - 1) // PREFILL_SINKHORN_T_TILE
    workers = pl.min(blocks, PREFILL_WORKER_TILE)
    with pl.spmd(workers, name_hint="mhc_sinkhorn") as ready:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, blocks, workers):
            token = block * PREFILL_SINKHORN_T_TILE
            rows = pl.min(PREFILL_SINKHORN_T_TILE, tokens - token)
            inverse_group = pl.load(
                inv_rms,
                [token, 0],
                [PREFILL_SINKHORN_T_TILE, PREFILL_RMS_T_TILE],
                valid_shape=[rows, PREFILL_RMS_T_TILE],
            )
            inverse_tmp = pl.create_tile(
                [PREFILL_SINKHORN_T_TILE, PREFILL_RMS_T_TILE], dtype=pl.FP32
            )
            inverse = pl.row_min(inverse_group, tmp_tile=inverse_tmp)

            raw_pre = pl.load(
                projected,
                [token, 0],
                [PREFILL_SINKHORN_T_TILE, MIX_PAD],
                valid_shape=[rows, HC_MULT],
            )
            raw_post = pl.load(
                projected,
                [token, HC_MULT],
                [PREFILL_SINKHORN_T_TILE, MIX_PAD],
                valid_shape=[rows, HC_MULT],
            )
            pre_bias = pl.load(bias, [0, 0], [1, MIX_PAD], valid_shape=[1, HC_MULT])
            post_bias = pl.load(bias, [0, HC_MULT], [1, MIX_PAD], valid_shape=[1, HC_MULT])
            pre_normalized = pl.row_expand_mul(raw_pre, inverse)
            post_normalized = pl.row_expand_mul(raw_post, inverse)
            pre_logits = pl.col_expand_add(pl.mul(pre_normalized, scale0), pre_bias)
            post_logits = pl.col_expand_add(pl.mul(post_normalized, scale1), post_bias)
            pre = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0)), HC_EPS)
            post = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0)), 2.0)
            pre = pl.set_validshape(pre, rows, HC_MULT)
            post = pl.set_validshape(post, rows, HC_MULT)
            pl.store(pre, [token, 0], pre_mix)
            pl.store(post, [token, 0], post_mix)

            residual0 = pl.load(
                projected,
                [token, 2 * HC_MULT],
                [PREFILL_SINKHORN_T_TILE, MIX_PAD],
                valid_shape=[rows, HC_MULT],
            )
            residual1 = pl.load(
                projected,
                [token, 3 * HC_MULT],
                [PREFILL_SINKHORN_T_TILE, MIX_PAD],
                valid_shape=[rows, HC_MULT],
            )
            residual2 = pl.load(
                projected,
                [token, 4 * HC_MULT],
                [PREFILL_SINKHORN_T_TILE, MIX_PAD],
                valid_shape=[rows, HC_MULT],
            )
            residual3 = pl.load(
                projected,
                [token, 5 * HC_MULT],
                [PREFILL_SINKHORN_T_TILE, MIX_PAD],
                valid_shape=[rows, HC_MULT],
            )
            residual0 = pl.row_expand_mul(residual0, inverse)
            residual1 = pl.row_expand_mul(residual1, inverse)
            residual2 = pl.row_expand_mul(residual2, inverse)
            residual3 = pl.row_expand_mul(residual3, inverse)
            residual_bias0 = pl.load(bias, [0, 2 * HC_MULT], [1, MIX_PAD], valid_shape=[1, HC_MULT])
            residual_bias1 = pl.load(bias, [0, 3 * HC_MULT], [1, MIX_PAD], valid_shape=[1, HC_MULT])
            residual_bias2 = pl.load(bias, [0, 4 * HC_MULT], [1, MIX_PAD], valid_shape=[1, HC_MULT])
            residual_bias3 = pl.load(bias, [0, 5 * HC_MULT], [1, MIX_PAD], valid_shape=[1, HC_MULT])
            residual0 = pl.col_expand_add(pl.mul(residual0, scale2), residual_bias0)
            residual1 = pl.col_expand_add(pl.mul(residual1, scale2), residual_bias1)
            residual2 = pl.col_expand_add(pl.mul(residual2, scale2), residual_bias2)
            residual3 = pl.col_expand_add(pl.mul(residual3, scale2), residual_bias3)
            residual0 = pl.fillpad(residual0, pad_value=pl.PadValue.min)
            residual1 = pl.fillpad(residual1, pad_value=pl.PadValue.min)
            residual2 = pl.fillpad(residual2, pad_value=pl.PadValue.min)
            residual3 = pl.fillpad(residual3, pad_value=pl.PadValue.min)
            row_tmp = pl.create_tile(
                [PREFILL_SINKHORN_T_TILE, MIX_PAD], dtype=pl.FP32
            )
            exp0 = pl.exp(pl.row_expand_sub(residual0, pl.row_max(residual0, row_tmp)))
            exp1 = pl.exp(pl.row_expand_sub(residual1, pl.row_max(residual1, row_tmp)))
            exp2 = pl.exp(pl.row_expand_sub(residual2, pl.row_max(residual2, row_tmp)))
            exp3 = pl.exp(pl.row_expand_sub(residual3, pl.row_max(residual3, row_tmp)))
            prob0 = pl.add(pl.row_expand_div(exp0, pl.row_sum(exp0, row_tmp)), HC_EPS)
            prob1 = pl.add(pl.row_expand_div(exp1, pl.row_sum(exp1, row_tmp)), HC_EPS)
            prob2 = pl.add(pl.row_expand_div(exp2, pl.row_sum(exp2, row_tmp)), HC_EPS)
            prob3 = pl.add(pl.row_expand_div(exp3, pl.row_sum(exp3, row_tmp)), HC_EPS)
            prob0 = pl.fillpad(pl.set_validshape(prob0, rows, HC_MULT), pad_value=pl.PadValue.zero)
            prob1 = pl.fillpad(pl.set_validshape(prob1, rows, HC_MULT), pad_value=pl.PadValue.zero)
            prob2 = pl.fillpad(pl.set_validshape(prob2, rows, HC_MULT), pad_value=pl.PadValue.zero)
            prob3 = pl.fillpad(pl.set_validshape(prob3, rows, HC_MULT), pad_value=pl.PadValue.zero)
            column_sum = pl.add(pl.add(prob0, prob1), pl.add(prob2, prob3))
            column_sum = pl.add(column_sum, HC_EPS)
            prob0 = pl.div(prob0, column_sum)
            prob1 = pl.div(prob1, column_sum)
            prob2 = pl.div(prob2, column_sum)
            prob3 = pl.div(prob3, column_sum)
            for _ in pl.unroll(HC_SINKHORN_ITER - 1):
                norm0 = pl.row_expand_div(prob0, pl.add(pl.row_sum(prob0, row_tmp), HC_EPS))
                norm1 = pl.row_expand_div(prob1, pl.add(pl.row_sum(prob1, row_tmp), HC_EPS))
                norm2 = pl.row_expand_div(prob2, pl.add(pl.row_sum(prob2, row_tmp), HC_EPS))
                norm3 = pl.row_expand_div(prob3, pl.add(pl.row_sum(prob3, row_tmp), HC_EPS))
                column_sum = pl.add(pl.add(norm0, norm1), pl.add(norm2, norm3))
                column_sum = pl.add(column_sum, HC_EPS)
                prob0 = pl.div(norm0, column_sum)
                prob1 = pl.div(norm1, column_sum)
                prob2 = pl.div(norm2, column_sum)
                prob3 = pl.div(norm3, column_sum)
            pl.store(pl.set_validshape(prob0, rows, HC_MULT), [token, 0], residual_flat)
            pl.store(pl.set_validshape(prob1, rows, HC_MULT), [token, HC_MULT], residual_flat)
            pl.store(pl.set_validshape(prob2, rows, HC_MULT), [token, 2 * HC_MULT], residual_flat)
            pl.store(pl.set_validshape(prob3, rows, HC_MULT), [token, 3 * HC_MULT], residual_flat)
    return ready


@pl.jit.inline
def hc_head(
    x_hc: pl.Tensor,
    pre_mix: pl.Tensor,
    output: pl.Tensor,
):
    """Collapse HC streams with the delayed pre-mix and BF16 rounding."""
    t_dim = pl.tensor.dim(x_hc, 0)
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    for block in pl.spmd((t_dim + PRE_T_TILE - 1) // PRE_T_TILE, name_hint="hc_collapse", allow_early_resolve=True):
        for t in pl.range(block * PRE_T_TILE, pl.min((block + 1) * PRE_T_TILE, t_dim)):
            pre_tile = pl.load(pre_mix, [t, 0], [1, HC_PAD], valid_shape=[1, HC_MULT])
            pre0 = pl.tile.read(pre_tile, [0, 0])
            pre1 = pl.tile.read(pre_tile, [0, 1])
            pre2 = pl.tile.read(pre_tile, [0, 2])
            pre3 = pl.tile.read(pre_tile, [0, 3])
            for d0 in pl.pipeline(0, D, HEAD_D_TILE, stage=2):
                x0 = pl.load(x_flat, [t, d0], [1, HEAD_D_TILE])
                x1 = pl.load(x_flat, [t, D + d0], [1, HEAD_D_TILE])
                x2 = pl.load(x_flat, [t, 2 * D + d0], [1, HEAD_D_TILE])
                x3 = pl.load(x_flat, [t, 3 * D + d0], [1, HEAD_D_TILE])
                y0 = pl.mul(x0, pre0)
                y1 = pl.mul(x1, pre1)
                y2 = pl.mul(x2, pre2)
                y3 = pl.mul(x3, pre3)
                y01 = pl.add(y0, y1)
                y23 = pl.add(y2, y3)
                y_tile = pl.add(y01, y23)
                y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
                pl.store(y_bf16, [t, d0], output)
    return output


@pl.jit.inline
def hc_head_norm(
    x_hc: pl.Tensor,
    pre_mix: pl.Tensor,
    norm_weight: pl.Tensor[[D], pl.BF16],
    output: pl.Tensor,
):
    """Collapse HC rows and normalize their rounded BF16 values."""
    tokens = pl.tensor.dim(x_hc, 0)
    x_flat = pl.reshape(x_hc, [tokens, HC_DIM])
    with pl.spmd((tokens + PRE_NORM_T_TILE - 1) // PRE_NORM_T_TILE, name_hint="hc_collapse_norm") as ready:
        block = pl.tile.get_block_idx()
        t0 = block * PRE_NORM_T_TILE
        valid_rows = pl.min(PRE_NORM_T_TILE, tokens - t0)
        pre_tile = pl.slice(pre_mix, [PRE_NORM_T_TILE, HC_PAD], [t0, 0], valid_shape=[valid_rows, HC_MULT])
        pre_tile = pl.set_validshape(pre_tile, PRE_NORM_T_TILE, HC_PAD)
        pre_rows = pl.transpose(pre_tile, 0, 1)
        pre0 = pl.reshape(pre_rows[0:1, 0:PRE_NORM_T_TILE], [PRE_NORM_T_TILE, 1])
        pre1 = pl.reshape(pre_rows[1:2, 0:PRE_NORM_T_TILE], [PRE_NORM_T_TILE, 1])
        pre2 = pl.reshape(pre_rows[2:3, 0:PRE_NORM_T_TILE], [PRE_NORM_T_TILE, 1])
        pre3 = pl.reshape(pre_rows[3:4, 0:PRE_NORM_T_TILE], [PRE_NORM_T_TILE, 1])
        pre0 = pl.set_validshape(pre0, valid_rows, 1)
        pre1 = pl.set_validshape(pre1, valid_rows, 1)
        pre2 = pl.set_validshape(pre2, valid_rows, 1)
        pre3 = pl.set_validshape(pre3, valid_rows, 1)
        collapsed = pl.create_tensor([D // PRE_NORM_D_TILE * PRE_NORM_T_TILE, PRE_NORM_D_TILE], dtype=pl.BF16)
        sq_sum = pl.full([1, PRE_NORM_T_TILE], dtype=pl.FP32, value=0.0)
        for d0 in pl.pipeline(0, D, PRE_NORM_D_TILE, stage=2):
            x0 = pl.slice(x_flat, [PRE_NORM_T_TILE, PRE_NORM_D_TILE], [t0, d0], valid_shape=[valid_rows, PRE_NORM_D_TILE])
            x1 = pl.slice(x_flat, [PRE_NORM_T_TILE, PRE_NORM_D_TILE], [t0, D + d0], valid_shape=[valid_rows, PRE_NORM_D_TILE])
            x2 = pl.slice(x_flat, [PRE_NORM_T_TILE, PRE_NORM_D_TILE], [t0, 2 * D + d0], valid_shape=[valid_rows, PRE_NORM_D_TILE])
            x3 = pl.slice(x_flat, [PRE_NORM_T_TILE, PRE_NORM_D_TILE], [t0, 3 * D + d0], valid_shape=[valid_rows, PRE_NORM_D_TILE])
            y0 = pl.row_expand_mul(x0, pre0)
            y1 = pl.row_expand_mul(x1, pre1)
            y2 = pl.row_expand_mul(x2, pre2)
            y3 = pl.row_expand_mul(x3, pre3)
            y01 = pl.add(y0, y1)
            y23 = pl.add(y2, y3)
            mixed = pl.add(y01, y23)
            rounded = pl.cast(mixed, target_type=pl.BF16, mode="rint")
            cache_row = d0 // PRE_NORM_D_TILE * PRE_NORM_T_TILE
            collapsed[cache_row : cache_row + PRE_NORM_T_TILE, 0:PRE_NORM_D_TILE] = rounded
            padded = pl.fillpad(rounded, pad_value=pl.PadValue.zero)
            padded = pl.set_validshape(padded, PRE_NORM_T_TILE, PRE_NORM_D_TILE)
            value = pl.cast(padded, target_type=pl.FP32)
            squared = pl.mul(value, value)
            partial = pl.reshape(pl.row_sum(squared), [1, PRE_NORM_T_TILE])
            sq_sum[0:1, 0:PRE_NORM_T_TILE] = pl.add(sq_sum, partial)
        rms_arg = pl.add(pl.mul(sq_sum, 1.0 / D), NORM_EPS)
        inverse = pl.rsqrt(rms_arg, high_precision=True)
        inverse = pl.reshape(inverse, [PRE_NORM_T_TILE, 1])
        for d0 in pl.pipeline(0, D, PRE_NORM_D_TILE, stage=2):
            cache_row = d0 // PRE_NORM_D_TILE * PRE_NORM_T_TILE
            value = pl.cast(collapsed[cache_row : cache_row + PRE_NORM_T_TILE, 0:PRE_NORM_D_TILE], target_type=pl.FP32)
            gamma = pl.cast(norm_weight[d0 : d0 + PRE_NORM_D_TILE], target_type=pl.FP32)
            gamma = pl.reshape(gamma, [1, PRE_NORM_D_TILE])
            scaled = pl.row_expand_mul(value, inverse)
            normalized = pl.col_expand_mul(scaled, gamma)
            result = pl.cast(normalized, target_type=pl.BF16, mode="rint")
            result = pl.set_validshape(result, valid_rows, PRE_NORM_D_TILE)
            output[t0 : t0 + PRE_NORM_T_TILE, d0 : d0 + PRE_NORM_D_TILE] = result
    return ready


@pl.jit.inline
def hc_pre(
    x_hc: pl.Tensor,
    incoming_pre_mix: pl.Tensor,
    function: pl.Tensor,
    scale: pl.Tensor,
    base: pl.Tensor,
    output: pl.Tensor,
    next_pre_mix: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    prefill: pl.constexpr = False,
):
    """Generate delayed HC coefficients and collapse the incoming streams."""
    if prefill:
        inv_rms, projected = hc_project_prefill(x_hc, function, incoming_pre_mix, output)
        mixes_ready = hc_sinkhorn_prefill(
            inv_rms, projected, scale, base, next_pre_mix, post_mix, residual_mix,
        )
    else:
        inv_rms, partials = hc_project(x_hc, function)
        mixes_ready = hc_sinkhorn(
            inv_rms, partials, scale, base, next_pre_mix, post_mix, residual_mix,
        )
        hc_head(x_hc, incoming_pre_mix, output)
    return mixes_ready


@pl.jit.inline
def zero_bf16_padding(x: pl.Tensor, valid_rows: pl.Scalar[pl.INT32]):
    """Clear physical rows that belong to an empty or partial TP slab.

    A TP rank with no logical token still publishes one fixed-capacity slab so
    every rank executes the same distributed window protocol.  Clearing the
    unused rows makes that padding deterministic and prevents an empty owner
    from publishing uninitialized data.
    """
    tokens = pl.tensor.dim(x, 0)
    if tokens > 0 and valid_rows < tokens:
        blocks = (tokens - valid_rows + 3) // 4
        workers = pl.min(blocks, 48)
        for worker in pl.spmd(workers, name_hint="decode_sp_zero_padding"):
            for block in pl.range(worker, blocks, workers):
                first = valid_rows + block * 4
                rows = pl.min(4, tokens - first)
                empty = pl.tile.full([4, D], dtype=pl.BF16, value=0.0)
                pl.store(pl.set_validshape(empty, rows, D), [first, 0], x)


@pl.jit.inline(auto_scope=False)
def hc_post_block(
    sublayer: pl.Tensor,
    residual_flat: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix_flat: pl.Tensor,
    output_flat: pl.Tensor,
    block: pl.Scalar[pl.INDEX],
    stream_tile: pl.constexpr,
):
    """Mix one hidden-width tile for one or four output streams."""
    pair_block = block // (D // D_TILE)
    t = pair_block // (HC_MULT // stream_tile)
    out_base = (pair_block % (HC_MULT // stream_tile)) * stream_tile
    d0 = (block % (D // D_TILE)) * D_TILE
    x_tile = pl.cast(sublayer[t : t + 1, d0 : d0 + D_TILE], pl.FP32)
    residual0 = residual_flat[t : t + 1, d0 : d0 + D_TILE]
    residual1 = residual_flat[t : t + 1, D + d0 : D + d0 + D_TILE]
    residual2 = residual_flat[t : t + 1, 2 * D + d0 : 2 * D + d0 + D_TILE]
    residual3 = residual_flat[t : t + 1, 3 * D + d0 : 3 * D + d0 + D_TILE]
    for out_offset in pl.unroll(stream_tile):
        out_h = out_base + out_offset
        update = pl.mul(x_tile, pl.read(post_mix, [t, out_h]))
        term0 = pl.mul(residual0, pl.read(residual_mix_flat, [t, out_h]))
        term1 = pl.mul(residual1, pl.read(residual_mix_flat, [t, HC_MULT + out_h]))
        value = pl.add(term0, term1)
        term2 = pl.mul(residual2, pl.read(residual_mix_flat, [t, 2 * HC_MULT + out_h]))
        value = pl.add(value, term2)
        term3 = pl.mul(residual3, pl.read(residual_mix_flat, [t, 3 * HC_MULT + out_h]))
        value = pl.add(value, term3)
        value = pl.add(value, update)
        output_flat[t : t + 1, out_h * D + d0 : out_h * D + d0 + D_TILE] = value
    return output_flat


@pl.jit.inline
def hc_post(
    sublayer: pl.Tensor,
    residual: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    output: pl.Tensor,
    deps: pl.Array[2, pl.TASK_ID],
    prefill: pl.constexpr = False,
):
    """Mix all four residual streams after the supplied producer dependencies."""
    t_dim = pl.tensor.dim(sublayer, 0)
    residual_flat = pl.reshape(residual, [t_dim, HC_DIM])
    residual_mix_flat = pl.reshape(residual_mix, [t_dim, HC_MULT * HC_MULT])
    output_flat = pl.reshape(output, [t_dim, HC_DIM])
    if prefill:
        tiles = t_dim * HC_MULT
        workers = pl.min(tiles, PREFILL_POST_WORKERS)
        with pl.spmd(workers, name_hint="hc_post_prefill", deps=[deps]):
            worker = pl.tile.get_block_idx()
            for block in pl.pipeline(worker, tiles, PREFILL_POST_WORKERS, stage=2):
                token = block // HC_MULT
                out_h = block % HC_MULT
                for d0 in pl.pipeline(0, D, PREFILL_POST_D_TILE, stage=2):
                    value = pl.cast(
                        sublayer[token : token + 1, d0 : d0 + PREFILL_POST_D_TILE],
                        pl.FP32,
                    )
                    value = pl.mul(value, pl.read(post_mix, [token, out_h]))
                    for source_stream in pl.unroll(HC_MULT):
                        source = residual_flat[
                            token : token + 1,
                            source_stream * D + d0 : source_stream * D + d0 + PREFILL_POST_D_TILE,
                        ]
                        coefficient = pl.read(
                            residual_mix_flat, [token, source_stream * HC_MULT + out_h]
                        )
                        value = pl.add(value, pl.mul(source, coefficient))
                    output_flat[
                        token : token + 1,
                        out_h * D + d0 : out_h * D + d0 + PREFILL_POST_D_TILE,
                    ] = value
        return output
    total_work = t_dim * (D // D_TILE)
    full_work = total_work // POST_WORKER_TILE * POST_WORKER_TILE
    tail_work = (total_work - full_work) * HC_MULT
    workers = pl.min(total_work * HC_MULT, POST_WORKER_TILE)
    with pl.spmd(workers, name_hint="hc_post", deps=[deps]):
        worker = pl.tile.get_block_idx()
        for block in pl.pipeline(worker, full_work, POST_WORKER_TILE, stage=1):
            hc_post_block(sublayer, residual_flat, post_mix, residual_mix_flat, output_flat, block, HC_MULT)
        for tail in pl.range(worker, tail_work, POST_WORKER_TILE):
            tail_chunk = full_work + tail // HC_MULT
            tail_stream = (tail_chunk // (D // D_TILE)) * HC_MULT + tail % HC_MULT
            tail_block = tail_stream * (D // D_TILE) + tail_chunk % (D // D_TILE)
            hc_post_block(sublayer, residual_flat, post_mix, residual_mix_flat, output_flat, tail_block, 1)
    return output


@pl.jit.inline
def prefill_sp_post(
    sublayer: pl.Tensor[[SP_T_DYN, D], pl.BF16],
    residual: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    post_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[SP_T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    next_pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Run local mHC-post, writing active values and zero padding in one producer.

    A later pure-output clear is not an ordering edge for the post task. Each
    residual element is instead written exactly once, with hc_post's FP32
    accumulation order and FP32 output.
    """
    tokens = pl.tensor.dim(residual, 0)
    residual_flat = pl.reshape(residual, [tokens, HC_MULT * D])
    mix_flat = pl.reshape(residual_mix, [tokens, HC_MULT * HC_MULT])
    output_flat = pl.reshape(output, [tokens, HC_MULT * D])
    tiles = tokens * HC_MULT
    workers = pl.min(tiles, PREFILL_POST_WORKERS)
    for worker in pl.spmd(workers, name_hint="sp_mhc_post"):
        for block in pl.pipeline(worker, tiles, PREFILL_POST_WORKERS, stage=2):
            row = block // HC_MULT
            out_h = block % HC_MULT
            for col in pl.pipeline(0, D, PREFILL_POST_D_TILE, stage=2):
                value = pl.full([1, PREFILL_POST_D_TILE], dtype=pl.FP32, value=0.0)
                if tp_rank * tokens + row < num_tokens:
                    hidden = pl.cast(
                        sublayer[row:row + 1, col:col + PREFILL_POST_D_TILE], pl.FP32
                    )
                    update = pl.mul(hidden, pl.read(post_mix, [row, out_h]))
                    lane0 = residual_flat[row:row + 1, col:col + PREFILL_POST_D_TILE]
                    value = pl.mul(lane0, pl.read(mix_flat, [row, out_h]))
                    for index in pl.unroll(HC_MULT - 1):
                        in_h = index + 1
                        lane = residual_flat[
                            row:row + 1,
                            in_h * D + col:in_h * D + col + PREFILL_POST_D_TILE,
                        ]
                        value = pl.add(
                            value,
                            pl.mul(lane, pl.read(mix_flat, [row, in_h * HC_MULT + out_h])),
                        )
                    value = pl.add(value, update)
                output_flat[
                    row:row + 1,
                    out_h * D + col:out_h * D + col + PREFILL_POST_D_TILE,
                ] = value
    for row in pl.spmd(tokens, name_hint="sp_mix_zero_padding"):
        if tp_rank * tokens + row >= num_tokens:
            zero_mix = pl.full([1, 8], dtype=pl.FP32, value=0.0)
            next_pre_mix[row:row + 1, :] = pl.set_validshape(zero_mix, 1, HC_MULT)
    return output, next_pre_mix


def golden_mhc_mixes(
    x_hc: torch.Tensor,
    function: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    iterations: int = M.hc_sinkhorn_iters,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project one HC stream into pre, post, and doubly-stochastic residual mixes."""
    flat = x_hc.flatten(-2).float()
    inverse_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + M.rms_norm_eps)
    mixes = F.linear(flat, function.float()) * inverse_rms
    hc = M.hc_mult
    pre_raw, post_raw, residual_raw = mixes.split((hc, hc, hc * hc), dim=-1)
    pre = torch.sigmoid(pre_raw * scale[0] + base[:hc]) + M.hc_eps
    post = 2 * torch.sigmoid(post_raw * scale[1] + base[hc : 2 * hc])
    residual_base = base[2 * hc :].unflatten(-1, (hc, hc))
    residual = (residual_raw.unflatten(-1, (hc, hc)) * scale[2] + residual_base).softmax(dim=-1)
    residual = residual + M.hc_eps
    residual = residual / (residual.sum(dim=-2, keepdim=True) + M.hc_eps)
    for _ in range(iterations - 1):
        residual = residual / (residual.sum(dim=-1, keepdim=True) + M.hc_eps)
        residual = residual / (residual.sum(dim=-2, keepdim=True) + M.hc_eps)
    return pre, post, residual


def golden_hc_collapse(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Collapse four HC streams with the kernel's pairwise FP32 reduction."""
    products = x.float() * pre_mix.float().unsqueeze(-1)
    first_pair = products[..., 0, :] + products[..., 1, :]
    second_pair = products[..., 2, :] + products[..., 3, :]
    return (first_pair + second_pair).to(x.dtype)


def golden_hc_post(
    sublayer: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    residual_mix: torch.Tensor,
) -> torch.Tensor:
    """Expand a sublayer result and mix the four residual streams in FP32."""
    update = post_mix.unsqueeze(-1) * sublayer.unsqueeze(-2)
    skip = (residual_mix.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=-3)
    return update.float() + skip.float()


def golden_l2_hc_pre(tensors):
    """Compute the four outputs with the incoming mix independent of the next mix."""
    pre_mix, post_mix, residual_mix = golden_mhc_mixes(
        tensors["x_hc"], tensors["function"], tensors["scale"], tensors["base"],
    )
    tensors["pre_mix"][:] = pre_mix
    tensors["post_mix"][:] = post_mix
    tensors["residual_mix"][:] = residual_mix
    collapsed = golden_hc_collapse(tensors["x_hc"], tensors["incoming_pre_mix"])
    tensors["output"][:] = collapsed.to(torch.bfloat16)


def golden_l2_hc_post(tensors):
    """Fill the expanded residual stream with FP32 accumulation and storage."""
    tensors["output"][:] = golden_hc_post(
        tensors["sublayer"], tensors["residual"], tensors["post_mix"], tensors["residual_mix"],
    )


def golden_l2_hc_head(tensors):
    """Fill the expected collapsed stream output."""
    tensors["output"][:] = golden_hc_collapse(tensors["x_hc"], tensors["pre_mix"]).to(torch.bfloat16)


HC_INPUT_NAMES = ("x_hc", "pre_mix", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight")


def make_hc_inputs(tokens, seed, case):
    """mHC inputs with distinct per-token magnitudes so both RMS statistics stay observable."""
    gen = torch.Generator().manual_seed(seed)
    magnitudes = (0.5 + torch.arange(tokens) % 4).reshape(-1, 1, 1)
    x_hc = (torch.randn(tokens, HC_MULT, D, generator=gen) * magnitudes).bfloat16().float()
    values = {
        "x_hc": x_hc,
        "pre_mix": torch.sigmoid(torch.randn(tokens, HC_MULT, generator=gen)) + M.hc_eps,
        "hc_attn_fn": torch.randn(MIX_HC, HC_DIM, generator=gen) / math.sqrt(HC_DIM),
        "hc_attn_scale": torch.randn(3, generator=gen),
        "hc_attn_base": torch.randn(MIX_HC, generator=gen),
        "attn_norm_weight": (torch.randn(D, generator=gen) * 0.1 + 1).bfloat16(),
    }
    if case == "zero":
        values["x_hc"] = torch.zeros_like(x_hc)
    return values


@pl.jit
def l2_hc_pre(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    function: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    scale: pl.Tensor[[3], pl.FP32],
    base: pl.Tensor[[MIX_HC], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    post_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    residual_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32]],
):
    """Validate the shared HcPre computation."""
    x_hc.bind_dynamic(0, T_DYN)
    incoming_pre_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    pre_mix.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    hc_pre(
        x_hc, incoming_pre_mix, function, scale, base, output,
        pre_mix, post_mix, residual_mix, prefill=True,
    )
    return output


def build_l2_hc_pre_specs(batch: int = 33, sequence: int = 1):
    """Build coefficient and independent incoming-mix fixtures."""
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(3)
    incoming_generator = torch.Generator().manual_seed(5)

    def init_x_hc():
        return torch.randn(tokens, HC_MULT, D, generator=generator)

    def init_incoming_pre_mix():
        return torch.sigmoid(torch.randn(tokens, HC_MULT, generator=incoming_generator)) + HC_EPS

    def init_function():
        return torch.randn(MIX_HC, HC_DIM, generator=generator) / math.sqrt(HC_DIM)

    def init_scale():
        return torch.randn(3, generator=generator)

    def init_base():
        return torch.randn(MIX_HC, generator=generator)

    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.float32, init_value=init_x_hc),
        TensorSpec("incoming_pre_mix", [tokens, HC_MULT], torch.float32, init_value=init_incoming_pre_mix),
        TensorSpec("function", [MIX_HC, HC_DIM], torch.float32, init_value=init_function),
        TensorSpec("scale", [3], torch.float32, init_value=init_scale),
        TensorSpec("base", [MIX_HC], torch.float32, init_value=init_base),
        TensorSpec("output", [tokens, D], torch.bfloat16),
        TensorSpec("pre_mix", [tokens, HC_MULT], torch.float32),
        TensorSpec("post_mix", [tokens, HC_MULT], torch.float32),
        TensorSpec("residual_mix", [tokens, HC_MULT, HC_MULT], torch.float32),
    ]


@pl.jit
def l2_hc_post(
    sublayer: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    """Validate HC post mapping and residual mixing."""
    sublayer.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    hc_post_deps = pl.array.create(2, pl.TASK_ID)
    hc_post(sublayer, residual, post_mix, residual_mix, output, deps=hc_post_deps, prefill=True)
    return output


def build_l2_hc_post_specs(batch: int = 33, sequence: int = 1):
    """Build BF16-valued residual streams and nonsymmetric mixing coefficients."""
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(7)

    def init_sublayer():
        return torch.randn(tokens, D, generator=generator).bfloat16()

    def init_residual():
        return torch.randn(tokens, HC_MULT, D, generator=generator)

    def init_post_mix():
        return 2 * torch.sigmoid(torch.randn(tokens, HC_MULT, generator=generator))

    def init_residual_mix():
        mix = torch.rand(tokens, HC_MULT, HC_MULT, generator=generator) + HC_EPS
        for _ in range(HC_SINKHORN_ITER):
            mix = mix / (mix.sum(dim=-1, keepdim=True) + HC_EPS)
            mix = mix / (mix.sum(dim=-2, keepdim=True) + HC_EPS)
        return mix

    return [
        TensorSpec("sublayer", [tokens, D], torch.bfloat16, init_value=init_sublayer),
        TensorSpec("residual", [tokens, HC_MULT, D], torch.float32, init_value=init_residual),
        TensorSpec("post_mix", [tokens, HC_MULT], torch.float32, init_value=init_post_mix),
        TensorSpec("residual_mix", [tokens, HC_MULT, HC_MULT], torch.float32, init_value=init_residual_mix),
        TensorSpec("output", [tokens, HC_MULT, D], torch.float32),
    ]


@pl.jit
def l2_hc_head(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    """Validate the final HC stream collapse."""
    x_hc.bind_dynamic(0, T_DYN)
    pre_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    hc_head(x_hc, pre_mix, output)
    return output


def build_l2_hc_head_specs(batch: int = 33, sequence: int = 1):
    """Build deterministic inputs and output for stream-collapse validation."""
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(3)

    def init_x_hc():
        return torch.randn(tokens, HC_MULT, D, generator=generator)

    def init_pre_mix():
        return torch.rand(tokens, HC_MULT, generator=generator)

    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.float32, init_value=init_x_hc),
        TensorSpec("pre_mix", [tokens, HC_MULT], torch.float32, init_value=init_pre_mix),
        TensorSpec("output", [tokens, D], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 HC pre, post, and head validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=33)
    parser.add_argument("--sequence", type=int, default=1)
    parser.add_argument("--op", choices=["pre", "post", "head", "all"], default="all")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data")
    parser.add_argument("--runtime-dir")
    parser.add_argument("--enable-chip-swimlane", type=int, choices=[1, 2, 3, 4], default=0)
    args = parser.parse_args()
    if args.op == "all" and (args.golden_data or args.runtime_dir):
        parser.error("replay requires selecting one --op")
    cases = [
        ("pre", l2_hc_pre, build_l2_hc_pre_specs, golden_l2_hc_pre),
        ("post", l2_hc_post, build_l2_hc_post_specs, golden_l2_hc_post),
        ("head", l2_hc_head, build_l2_hc_head_specs, golden_l2_hc_head),
    ]
    failures = []
    for name, fn, build_specs, golden_fn in cases:
        if args.op != "all" and args.op != name:
            continue
        print(f"[HC] {name}", flush=True)
        if name == "pre":
            compare_fn = {
                "pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
                "post_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
                "residual_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            }
            compare_fn["output"] = ratio_allclose(atol=1e-4, rtol=1.0 / 128)
        elif name == "post":
            compare_fn = {"output": ratio_allclose(atol=1e-6, rtol=1e-5)}
        else:
            compare_fn = {"output": ratio_allclose(atol=1e-4, rtol=1.0 / 128)}
        config = {"backend_type": args.backend_type, "execution_mode": args.execution_mode}
        config.update(device_id=args.device, enable_chip_swimlane=args.enable_chip_swimlane)
        result = run(
            fn=fn,
            specs=build_specs(args.batch, args.sequence),
            golden_fn=golden_fn,
            config=config,
            rtol=1e-3,
            atol=1e-3,
            compare_fn=compare_fn,
            compile_only=args.compile_only,
            save_data=args.save_data,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
        )
        if not result.passed:
            failures.append(f"{name}: {result.error or 'validation failed'}")
    if failures:
        raise SystemExit("; ".join(failures))

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hyper-connection kernels, Torch goldens and validation."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"
import pypto.language as pl

from .operator_config import SMALL_TOKEN_CAPACITY


# tiling
D_TILE = 256
TOKEN_TILE = 8
CHANNEL_TILE = 8
MIX_TILE = 8
PROJ_T_TILE = 32
PROJ_N_TILE = 32
PROJ_K_TILE = 64
SHARED_PROJ_K_TILE = 1024
SHARED_PREP_K_TILE = 4096
SHARED_PREP_N_TILE = 8
PREFILL_POST_D_TILE = 2560
PREFILL_VECTOR_WORKERS = 48
SHARED_MERGE_D_TILE = 1024
PREFILL_MERGE_T_TILE = 8
SINKHORN_T_TILE = 16


def make_hc_collapse_inline(capacity, config):
    """Collapse four BF16 residual streams with FP32 premix weights."""
    hidden = config["hidden_size"]
    hc = config["hc_mult"]
    merge_tile = min(SHARED_MERGE_D_TILE, hidden)
    bulk = capacity >= 2048 and capacity % 2 == 0 and hc == 4 and hidden == 5120

    @pl.jit.inline
    def collapse(
        x: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        incoming: pl.Tensor[[capacity, hc], pl.FP32],
        collapsed: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        if bulk:
            streams = pl.reshape(x, [capacity * hc, hidden])
            weights_by_pair = pl.reshape(incoming, [capacity // 2, 2 * hc])
            for worker in pl.spmd((active + PREFILL_MERGE_T_TILE - 1) // PREFILL_MERGE_T_TILE, name_hint="a2_hc_collapse"):
                first_pair = worker * (PREFILL_MERGE_T_TILE // 2)
                last_pair = pl.min(first_pair + PREFILL_MERGE_T_TILE // 2, (active + 1) // 2)
                for pair in pl.range(first_pair, last_pair):
                    weights = pl.load(weights_by_pair, [pair, 0], [1, 2 * hc])
                    for row in pl.range(0, pl.min(2, active - pair * 2)):
                        token = pair * 2 + row
                        values = pl.load(streams, [token * hc, 0], [hc, hidden])
                        accumulated = pl.tile.full([1, hidden], dtype=pl.FP32, value=0.0)
                        for stream in pl.unroll(hc):
                            stream_bf16 = pl.slice(values, [1, hidden], [stream, 0])
                            stream_fp32 = pl.cast(stream_bf16, pl.FP32)
                            weight = pl.tile.read(weights, [0, row * hc + stream])
                            weighted_stream = pl.mul(stream_fp32, weight)
                            accumulated = pl.add(accumulated, weighted_stream)
                        value = pl.cast(accumulated, pl.BF16, mode="rint")
                        pl.store(value, [token, 0], collapsed)
        else:
            flat = pl.reshape(x, [capacity, hc * hidden])
            for block in pl.spmd(active * (hidden // merge_tile), name_hint="a2_hc_collapse"):
                token = block // (hidden // merge_tile)
                offset = block % (hidden // merge_tile) * merge_tile
                collapsed_tile = pl.full([1, merge_tile], dtype=pl.FP32, value=0.0)
                for stream in pl.unroll(hc):
                    source_offset = stream * hidden + offset
                    source_bf16 = flat[token : token + 1, source_offset : source_offset + merge_tile]
                    source_fp32 = pl.cast(source_bf16, pl.FP32)
                    coefficient = pl.read(incoming, [token, stream])
                    term = pl.mul(source_fp32, coefficient)
                    collapsed_tile = pl.add(collapsed_tile, term)
                collapsed[token : token + 1, offset : offset + merge_tile] = pl.cast(
                    collapsed_tile, pl.BF16, mode="rint"
                )
        return collapsed

    return collapse


def make_hc_pre_inline(capacity, config):
    """Build four-stream HC coefficient generation and incoming-premix collapse."""
    hidden = config["hidden_size"]
    hc = config["hc_mult"]
    flat_width = hidden * hc
    channels = (hc + 2) * hc
    iterations = config["hc_sinkhorn_iters"]
    eps = config["hc_eps"]
    norm_eps = config["rms_norm_eps"]
    rms_rows = max(TOKEN_TILE, flat_width // D_TILE)
    collapse_rows = hidden // D_TILE
    prep_elements = min(SHARED_PREP_K_TILE * SHARED_PREP_N_TILE, channels * flat_width)
    small_capacity = capacity <= SMALL_TOKEN_CAPACITY
    proj_rows = 16 if small_capacity else PROJ_T_TILE
    proj_cols = 16 if small_capacity else PROJ_N_TILE
    proj_k = 512 if small_capacity else SHARED_PROJ_K_TILE
    sinkhorn_rows = 8 if small_capacity else SINKHORN_T_TILE

    @pl.jit.inline
    def pre(
        x: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        function: pl.Tensor[[channels, flat_width], pl.FP32],
        scale: pl.Tensor[[3], pl.FP32],
        base: pl.Tensor[[channels], pl.FP32],
        incoming: pl.Tensor[[capacity, hc], pl.FP32],
        next_pre: pl.InOut[pl.Tensor[[capacity, hc], pl.FP32]],
        post: pl.InOut[pl.Tensor[[capacity, hc], pl.FP32]],
        comb: pl.InOut[pl.Tensor[[capacity, hc, hc], pl.FP32]],
        collapsed: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(x, [capacity * (flat_width // D_TILE), D_TILE])
        prepared = pl.create_tensor([capacity, flat_width], dtype=pl.FP32)
        prepared_chunks = pl.reshape(prepared, [capacity * (flat_width // D_TILE), D_TILE])
        inverse = pl.create_tensor([capacity, TOKEN_TILE], dtype=pl.FP32)
        collapsed_rows = pl.reshape(collapsed, [capacity * collapse_rows, D_TILE])
        for worker in pl.spmd(pl.min(active, PREFILL_VECTOR_WORKERS), name_hint="a2_hc_rms"):
            for token in pl.range(worker, active, PREFILL_VECTOR_WORKERS):
                chunk_start = token * (flat_width // D_TILE)
                values = pl.load(flat, [chunk_start, 0], [rms_rows, D_TILE], valid_shape=[flat_width // D_TILE, D_TILE])
                values = pl.fillpad(values, pad_value=pl.PadValue.zero)
                values_fp32 = pl.cast(values, pl.FP32)
                prepared_valid = pl.set_validshape(values_fp32, flat_width // D_TILE, D_TILE)
                pl.store(prepared_valid, [chunk_start, 0], prepared_chunks)
                collapsed_acc = pl.tile.full([collapse_rows, D_TILE], dtype=pl.FP32, value=0.0)
                for lane in pl.unroll(hc):
                    stream_values = pl.slice(values_fp32, [collapse_rows, D_TILE], [lane * collapse_rows, 0])
                    mix = pl.read(incoming, [token, lane])
                    collapsed_acc = pl.add(collapsed_acc, pl.mul(stream_values, mix))
                collapsed_tile = pl.cast(collapsed_acc, pl.BF16, mode="rint")
                pl.store(collapsed_tile, [token * collapse_rows, 0], collapsed_rows)
                square = pl.mul(values_fp32, values_fp32)
                sum_tmp = pl.create_tile([rms_rows, D_TILE], dtype=pl.FP32)
                partial_sums = pl.row_sum(square, sum_tmp)
                square_sum = 0.0
                for chunk in pl.range(flat_width // D_TILE):
                    partial = pl.tile.read(partial_sums, [chunk, 0])
                    square_sum = square_sum + partial
                argument_zero = pl.tile.full([1, TOKEN_TILE], dtype=pl.FP32, value=0.0)
                square_sum_tile = pl.add(argument_zero, square_sum)
                mean = pl.mul(square_sum_tile, 1.0 / flat_width)
                argument = pl.add(mean, norm_eps)
                rsqrt_tmp = pl.create_tile([1, TOKEN_TILE], dtype=pl.FP32)
                inverse_tile = pl.tile.rsqrt(argument, rsqrt_tmp, high_precision=True)
                pl.store(inverse_tile, [token, 0], inverse)

        rounded_function = pl.create_tensor([channels, flat_width], dtype=pl.FP32)
        function_flat = pl.reshape(function, [1, channels * flat_width])
        rounded_flat = pl.reshape(rounded_function, [1, channels * flat_width])
        for block in pl.spmd(
            (active + capacity - 1) // capacity * (channels * flat_width // prep_elements),
            name_hint="a2_hc_weight_prepare"
        ):
            prep_start = block * prep_elements
            prep_weight = function_flat[0:1, prep_start : prep_start + prep_elements]
            prep_bits = pl.reinterpret_view(prep_weight, pl.INT32)
            prep_units = pl.shrs(pl.add(prep_bits, 2048), 12)
            prep_rounded_bits = pl.shls(prep_units, 12)
            prep_result = pl.reinterpret_view(prep_rounded_bits, pl.FP32)
            rounded_flat[0:1, prep_start : prep_start + prep_elements] = prep_result

        accumulated = pl.create_tensor([capacity, channels], dtype=pl.FP32)
        channel_blocks = (channels + proj_cols - 1) // proj_cols
        for block in pl.spmd(
            (active + proj_rows - 1) // proj_rows * channel_blocks, name_hint="a2_hc_projection_cube"
        ):
            token = block // channel_blocks * proj_rows
            channel = block % channel_blocks * proj_cols
            rows = pl.min(proj_rows, active - token)
            columns = pl.min(proj_cols, channels - channel)
            acc = pl.create_tile([proj_rows, proj_cols], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            for inner in pl.pipeline(0, flat_width, proj_k, stage=2):
                source = pl.load(
                    prepared, [token, inner], [proj_rows, proj_k],
                    valid_shape=[rows, proj_k], target_memory=pl.Mem.Mat,
                )
                weight = pl.load(
                    rounded_function, [channel, inner], [proj_cols, proj_k],
                    valid_shape=[columns, proj_k], target_memory=pl.Mem.Mat,
                )
                acc = pl.matmul_acc(acc, source, pl.tile.transpose_view(weight), init_cond=(inner == 0))
            projection_tile = pl.set_validshape(acc, rows, columns)
            pl.store(projection_tile, [token, channel], accumulated)

        comb_flat = pl.reshape(comb, [capacity, hc * hc])
        bias = pl.reshape(base, [1, channels])
        for block in pl.spmd((active + sinkhorn_rows - 1) // sinkhorn_rows, name_hint="a2_hc_sinkhorn"):
            token = block * sinkhorn_rows
            rows = pl.min(sinkhorn_rows, active - token)
            inv_repeated = pl.load(inverse, [token, 0], [sinkhorn_rows, TOKEN_TILE], valid_shape=[rows, TOKEN_TILE])
            inv_scratch = pl.create_tile([sinkhorn_rows, TOKEN_TILE], dtype=pl.FP32)
            inv = pl.row_min(inv_repeated, tmp_tile=inv_scratch)
            scale0 = pl.read(scale, [0])
            scale1 = pl.read(scale, [1])
            scale2 = pl.read(scale, [2])
            raw_p0 = pl.load(accumulated, [token, 0], [sinkhorn_rows, MIX_TILE], valid_shape=[rows, hc])
            raw_p1 = pl.load(accumulated, [token, hc], [sinkhorn_rows, MIX_TILE], valid_shape=[rows, hc])
            p0 = pl.row_expand_mul(raw_p0, inv)
            p1 = pl.row_expand_mul(raw_p1, inv)
            b0 = pl.load(bias, [0, 0], [1, MIX_TILE], valid_shape=[1, hc])
            b1 = pl.load(bias, [0, hc], [1, MIX_TILE], valid_shape=[1, hc])
            pre_logits = pl.col_expand_add(pl.mul(p0, scale0), b0)
            post_logits = pl.col_expand_add(pl.mul(p1, scale1), b1)
            pre_result = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0)), eps)
            post_result = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0)), 2.0)
            pre_valid = pl.set_validshape(pre_result, rows, hc)
            post_valid = pl.set_validshape(post_result, rows, hc)
            pl.store(pre_valid, [token, 0], next_pre)
            pl.store(post_valid, [token, 0], post)
            raw_v0 = pl.load(accumulated, [token, 2 * hc], [sinkhorn_rows, MIX_TILE], valid_shape=[rows, hc])
            raw_v1 = pl.load(accumulated, [token, 3 * hc], [sinkhorn_rows, MIX_TILE], valid_shape=[rows, hc])
            raw_v2 = pl.load(accumulated, [token, 4 * hc], [sinkhorn_rows, MIX_TILE], valid_shape=[rows, hc])
            raw_v3 = pl.load(accumulated, [token, 5 * hc], [sinkhorn_rows, MIX_TILE], valid_shape=[rows, hc])
            v0 = pl.row_expand_mul(raw_v0, inv)
            v1 = pl.row_expand_mul(raw_v1, inv)
            v2 = pl.row_expand_mul(raw_v2, inv)
            v3 = pl.row_expand_mul(raw_v3, inv)
            c0 = pl.load(bias, [0, 2 * hc], [1, MIX_TILE], valid_shape=[1, hc])
            c1 = pl.load(bias, [0, 3 * hc], [1, MIX_TILE], valid_shape=[1, hc])
            c2 = pl.load(bias, [0, 4 * hc], [1, MIX_TILE], valid_shape=[1, hc])
            c3 = pl.load(bias, [0, 5 * hc], [1, MIX_TILE], valid_shape=[1, hc])
            logits0 = pl.fillpad(pl.col_expand_add(pl.mul(v0, scale2), c0), pad_value=pl.PadValue.min)
            logits1 = pl.fillpad(pl.col_expand_add(pl.mul(v1, scale2), c1), pad_value=pl.PadValue.min)
            logits2 = pl.fillpad(pl.col_expand_add(pl.mul(v2, scale2), c2), pad_value=pl.PadValue.min)
            logits3 = pl.fillpad(pl.col_expand_add(pl.mul(v3, scale2), c3), pad_value=pl.PadValue.min)
            row_tmp = pl.create_tile([sinkhorn_rows, MIX_TILE], dtype=pl.FP32)
            exp0 = pl.exp(pl.row_expand_sub(logits0, pl.row_max(logits0, row_tmp)))
            exp1 = pl.exp(pl.row_expand_sub(logits1, pl.row_max(logits1, row_tmp)))
            exp2 = pl.exp(pl.row_expand_sub(logits2, pl.row_max(logits2, row_tmp)))
            exp3 = pl.exp(pl.row_expand_sub(logits3, pl.row_max(logits3, row_tmp)))
            soft0 = pl.add(pl.row_expand_div(exp0, pl.row_sum(exp0, row_tmp)), eps)
            soft1 = pl.add(pl.row_expand_div(exp1, pl.row_sum(exp1, row_tmp)), eps)
            soft2 = pl.add(pl.row_expand_div(exp2, pl.row_sum(exp2, row_tmp)), eps)
            soft3 = pl.add(pl.row_expand_div(exp3, pl.row_sum(exp3, row_tmp)), eps)
            g0 = pl.fillpad(pl.set_validshape(soft0, rows, hc), pad_value=pl.PadValue.zero)
            g1 = pl.fillpad(pl.set_validshape(soft1, rows, hc), pad_value=pl.PadValue.zero)
            g2 = pl.fillpad(pl.set_validshape(soft2, rows, hc), pad_value=pl.PadValue.zero)
            g3 = pl.fillpad(pl.set_validshape(soft3, rows, hc), pad_value=pl.PadValue.zero)
            column_sum = pl.add(pl.add(g0, g1), pl.add(g2, g3))
            column_sum = pl.add(column_sum, eps)
            prob0 = pl.div(g0, column_sum)
            prob1 = pl.div(g1, column_sum)
            prob2 = pl.div(g2, column_sum)
            prob3 = pl.div(g3, column_sum)
            for _ in pl.range(iterations - 1):
                norm0 = pl.row_expand_div(prob0, pl.add(pl.row_sum(prob0, row_tmp), eps))
                norm1 = pl.row_expand_div(prob1, pl.add(pl.row_sum(prob1, row_tmp), eps))
                norm2 = pl.row_expand_div(prob2, pl.add(pl.row_sum(prob2, row_tmp), eps))
                norm3 = pl.row_expand_div(prob3, pl.add(pl.row_sum(prob3, row_tmp), eps))
                column_sum = pl.add(pl.add(norm0, norm1), pl.add(norm2, norm3))
                column_sum = pl.add(column_sum, eps)
                prob0 = pl.div(norm0, column_sum)
                prob1 = pl.div(norm1, column_sum)
                prob2 = pl.div(norm2, column_sum)
                prob3 = pl.div(norm3, column_sum)
            pl.store(pl.set_validshape(prob0, rows, hc), [token, 0], comb_flat)
            pl.store(pl.set_validshape(prob1, rows, hc), [token, hc], comb_flat)
            pl.store(pl.set_validshape(prob2, rows, hc), [token, 2 * hc], comb_flat)
            pl.store(pl.set_validshape(prob3, rows, hc), [token, 3 * hc], comb_flat)

        return next_pre, post, comb, collapsed

    return pre


def make_hc_post_inline(capacity, config):
    """Build BF16 residual expansion with FP32 combination arithmetic."""
    hidden = config["hidden_size"]
    hc = config["hc_mult"]
    pipelined = capacity > SMALL_TOKEN_CAPACITY and hidden == 5120 and hc == 4
    merge_tile = PREFILL_POST_D_TILE if pipelined else min(SHARED_MERGE_D_TILE, hidden)
    stages = 2 if pipelined else 1

    @pl.jit.inline
    def post(
        sublayer: pl.Tensor[[capacity, hidden], pl.BF16],
        residual: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        post_mix: pl.Tensor[[capacity, hc], pl.FP32],
        comb: pl.Tensor[[capacity, hc, hc], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, hc, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(residual, [capacity * hc, hidden])
        output_flat = pl.reshape(output, [capacity * hc, hidden])
        comb_flat = pl.reshape(comb, [capacity, hc * hc])
        tiles = active * (hidden // merge_tile)
        workers = pl.min(tiles, PREFILL_VECTOR_WORKERS)
        for worker in pl.spmd(workers, name_hint="a2_hc_post"):
            if pipelined:
                start = worker
                end = tiles
                step = PREFILL_VECTOR_WORKERS
            else:
                start = worker * tiles // workers
                end = (worker + 1) * tiles // workers
                step = 1
            for block in pl.pipeline(start, end, step, stage=stages):
                token = block // (hidden // merge_tile)
                offset = block % (hidden // merge_tile) * merge_tile
                source_row = token * hc
                value_bf16 = sublayer[token : token + 1, offset : offset + merge_tile]
                sources = flat[source_row : source_row + hc, offset : offset + merge_tile]
                value = pl.cast(value_bf16, pl.FP32)
                sources_fp32 = pl.cast(sources, pl.FP32)
                for stream in pl.unroll(hc):
                    coefficient = pl.read(post_mix, [token, stream])
                    merged = pl.mul(value, coefficient)
                    for source_stream in pl.unroll(hc):
                        source = sources_fp32[source_stream : source_stream + 1, :]
                        coefficient = pl.read(comb_flat, [token, source_stream * hc + stream])
                        term = pl.mul(source, coefficient)
                        merged = pl.add(merged, term)
                    destination = token * hc + stream
                    merged_bf16 = pl.cast(merged, pl.BF16, mode="rint")
                    output_flat[destination : destination + 1, offset : offset + merge_tile] = merged_bf16
        return output

    return post


def make_hc(capacity, config):
    """Build the standalone HC operator diagnostic entry."""
    hidden = config["hidden_size"]
    hc = config["hc_mult"]
    channels = (hc + 2) * hc
    pre = make_hc_pre_inline(capacity, config)
    post_fn = make_hc_post_inline(capacity, config)

    @pl.jit
    def hc_entry(
        x: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        function: pl.Tensor[[channels, hc * hidden], pl.FP32],
        scale: pl.Tensor[[3], pl.FP32],
        base: pl.Tensor[[channels], pl.FP32],
        incoming: pl.Tensor[[capacity, hc], pl.FP32],
        sublayer: pl.Tensor[[capacity, hidden], pl.BF16],
        next_pre: pl.InOut[pl.Tensor[[capacity, hc], pl.FP32]],
        post: pl.InOut[pl.Tensor[[capacity, hc], pl.FP32]],
        comb: pl.InOut[pl.Tensor[[capacity, hc, hc], pl.FP32]],
        collapsed: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        merged: pl.InOut[pl.Tensor[[capacity, hc, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        pre(x, function, scale, base, incoming, next_pre, post, comb, collapsed, active)
        post_fn(sublayer, x, post, comb, merged, active)
        return next_pre, post, comb, collapsed, merged

    return hc_entry


def make_decode_hc_pre_inline(capacity, config):
    """Build four-stream HC coefficient generation and incoming-premix collapse."""
    hidden = config["hidden_size"]
    hc = config["hc_mult"]
    flat_width = hidden * hc
    channels = (hc + 2) * hc
    iterations = config["hc_sinkhorn_iters"]
    eps = config["hc_eps"]
    norm_eps = config["rms_norm_eps"]

    @pl.jit.inline
    def pre(
        x: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        function: pl.Tensor[[channels, flat_width], pl.FP32],
        scale: pl.Tensor[[3], pl.FP32],
        base: pl.Tensor[[channels], pl.FP32],
        incoming: pl.Tensor[[capacity, hc], pl.FP32],
        next_pre: pl.Tensor[[capacity, hc], pl.FP32],
        post: pl.Tensor[[capacity, hc], pl.FP32],
        comb: pl.Tensor[[capacity, hc, hc], pl.FP32],
        collapsed: pl.Tensor[[capacity, hidden], pl.BF16],
        collapsed_fp32: pl.Tensor[[capacity, hidden], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(x, [capacity, flat_width])
        inverse = pl.create_tensor([capacity, TOKEN_TILE], dtype=pl.FP32)
        for block in pl.spmd((active + TOKEN_TILE - 1) // TOKEN_TILE, name_hint="a2_hc_rms"):
            token = block * TOKEN_TILE
            rows = pl.min(TOKEN_TILE, active - token)
            square_sum = pl.full([1, TOKEN_TILE], dtype=pl.FP32, value=0.0)
            for inner in pl.range(0, flat_width, D_TILE):
                values = pl.slice(flat, [TOKEN_TILE, D_TILE], [token, inner], valid_shape=[rows, D_TILE])
                values = pl.fillpad(values, pad_value=pl.PadValue.zero)
                values_fp32 = pl.cast(values, pl.FP32)
                square = pl.mul(values_fp32, values_fp32)
                square_sum = pl.add(square_sum, pl.reshape(pl.row_sum(square), [1, TOKEN_TILE]))
            mean = pl.mul(square_sum, 1.0 / flat_width)
            argument = pl.add(mean, norm_eps)
            inverse_column = pl.reshape(pl.rsqrt(argument, high_precision=True), [TOKEN_TILE, 1])
            ones = pl.full([TOKEN_TILE, TOKEN_TILE], dtype=pl.FP32, value=1.0)
            inverse_tile = pl.row_expand_mul(ones, inverse_column)
            inverse[token : token + TOKEN_TILE, :] = pl.set_validshape(inverse_tile, rows, TOKEN_TILE)

        projected = pl.create_tensor([capacity, channels], dtype=pl.FP32)
        for block in pl.spmd(active * (channels // CHANNEL_TILE), name_hint="a2_hc_projection"):
            token = block // (channels // CHANNEL_TILE)
            channel = block % (channels // CHANNEL_TILE) * CHANNEL_TILE
            dot_sum = pl.full([1, CHANNEL_TILE], dtype=pl.FP32, value=0.0)
            for inner in pl.range(0, flat_width, D_TILE):
                source = pl.cast(flat[token : token + 1, inner : inner + D_TILE], pl.FP32)
                weight = function[channel : channel + CHANNEL_TILE, inner : inner + D_TILE]
                product = pl.col_expand_mul(weight, source)
                dot_sum = pl.add(dot_sum, pl.reshape(pl.row_sum(product), [1, CHANNEL_TILE]))
            inverse_scalar = pl.read(inverse, [token, 0])
            normalized = pl.mul(dot_sum, inverse_scalar)
            projected[token : token + 1, channel : channel + CHANNEL_TILE] = normalized

        comb_flat = pl.reshape(comb, [capacity, hc * hc])
        for token in pl.spmd(active, name_hint="a2_hc_sinkhorn"):
            scale0 = pl.read(scale, [0])
            scale1 = pl.read(scale, [1])
            scale2 = pl.read(scale, [2])
            pre_logits = pl.tile.full([1, MIX_TILE], dtype=pl.FP32, value=0.0)
            post_logits = pl.tile.full([1, MIX_TILE], dtype=pl.FP32, value=0.0)
            for column in pl.unroll(MIX_TILE):
                logical = column % hc
                pre_value = pl.read(projected, [token, logical]) * scale0 + pl.read(base, [logical])
                post_value = pl.read(projected, [token, hc + logical]) * scale1 + pl.read(
                    base, [hc + logical]
                )
                pl.tile.write(pre_logits, [0, column], pre_value)
                pl.tile.write(post_logits, [0, column], post_value)
            pre_neg = pl.neg(pre_logits)
            pre_exp = pl.exp(pre_neg)
            pre_denominator = pl.add(pre_exp, 1.0)
            pre_sigmoid = pl.recip(pre_denominator)
            pre_result = pl.add(pre_sigmoid, eps)
            post_neg = pl.neg(post_logits)
            post_exp = pl.exp(post_neg)
            post_denominator = pl.add(post_exp, 1.0)
            post_sigmoid = pl.recip(post_denominator)
            post_result = pl.mul(post_sigmoid, 2.0)
            pre_valid = pl.set_validshape(pre_result, 1, hc)
            post_valid = pl.set_validshape(post_result, 1, hc)
            pl.store(pre_valid, [token, 0], next_pre)
            pl.store(post_valid, [token, 0], post)
            logits = pl.tile.full([MIX_TILE, MIX_TILE], dtype=pl.FP32, value=0.0)
            for row in pl.unroll(MIX_TILE):
                for column in pl.unroll(MIX_TILE):
                    logical = 2 * hc + (row % hc) * hc + column % hc
                    value = pl.read(projected, [token, logical]) * scale2 + pl.read(base, [logical])
                    pl.tile.write(logits, [row, column], value)
            logits_valid = pl.set_validshape(logits, hc, hc)
            logits_padded = pl.fillpad(logits_valid, pad_value=pl.PadValue.min)
            row_tmp = pl.create_tile([MIX_TILE, MIX_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            row_max = pl.row_max(logits_padded, row_tmp)
            shifted = pl.row_expand_sub(logits_padded, row_max)
            exponential = pl.exp(shifted)
            row_sum = pl.row_sum(exponential, row_tmp)
            probabilities_softmax = pl.add(pl.row_expand_div(exponential, row_sum), eps)
            probabilities_valid = pl.set_validshape(probabilities_softmax, hc, hc)
            probabilities_padded = pl.fillpad(probabilities_valid, pad_value=pl.PadValue.zero)
            column_sum = pl.col_sum(probabilities_padded)
            column_sum = pl.add(column_sum, eps)
            probabilities = pl.col_expand_div(probabilities_padded, column_sum)
            for _ in pl.range(iterations - 1):
                iter_valid = pl.set_validshape(probabilities, hc, hc)
                iter_padded = pl.fillpad(iter_valid, pad_value=pl.PadValue.zero)
                row_sum = pl.row_sum(iter_padded, row_tmp)
                row_sum = pl.add(row_sum, eps)
                row_normalized = pl.row_expand_div(iter_padded, row_sum)
                row_normalized_valid = pl.set_validshape(row_normalized, hc, hc)
                row_normalized_padded = pl.fillpad(row_normalized_valid, pad_value=pl.PadValue.zero)
                column_sum = pl.col_sum(row_normalized_padded)
                column_sum = pl.add(column_sum, eps)
                probabilities = pl.col_expand_div(row_normalized_padded, column_sum)
            for row in pl.unroll(hc):
                comb_value = pl.slice(probabilities, [1, MIX_TILE], [row, 0], valid_shape=[1, hc])
                pl.store(comb_value, [token, row * hc], comb_flat)

        for block in pl.spmd(active * (hidden // D_TILE), name_hint="a2_hc_collapse"):
            token = block // (hidden // D_TILE)
            offset = block % (hidden // D_TILE) * D_TILE
            collapsed_tile = pl.full([1, D_TILE], dtype=pl.FP32, value=0.0)
            for stream in pl.unroll(hc):
                source_offset = stream * hidden + offset
                source_bf16 = flat[token : token + 1, source_offset : source_offset + D_TILE]
                source_fp32 = pl.cast(source_bf16, pl.FP32)
                coefficient = pl.read(incoming, [token, stream])
                term = pl.mul(source_fp32, coefficient)
                collapsed_tile = pl.add(collapsed_tile, term)
            collapsed_fp32[token : token + 1, offset : offset + D_TILE] = collapsed_tile
            collapsed[token : token + 1, offset : offset + D_TILE] = pl.cast(
                collapsed_tile, pl.BF16, mode="rint"
            )
        return next_pre, post, comb, collapsed

    return pre


def make_decode_hc_post_inline(capacity, config):
    """Build BF16 residual expansion with FP32 combination arithmetic."""
    hidden = config["hidden_size"]
    hc = config["hc_mult"]

    @pl.jit.inline
    def post(
        sublayer: pl.Tensor[[capacity, hidden], pl.BF16],
        residual: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        post_mix: pl.Tensor[[capacity, hc], pl.FP32],
        comb: pl.Tensor[[capacity, hc, hc], pl.FP32],
        output: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(residual, [capacity, hc * hidden])
        output_flat = pl.reshape(output, [capacity, hc * hidden])
        comb_flat = pl.reshape(comb, [capacity, hc * hc])
        for block in pl.spmd(active * hc * (hidden // D_TILE), name_hint="a2_hc_post"):
            token = block // (hc * (hidden // D_TILE))
            stream = block // (hidden // D_TILE) % hc
            offset = block % (hidden // D_TILE) * D_TILE
            value = pl.cast(sublayer[token : token + 1, offset : offset + D_TILE], pl.FP32)
            coefficient = pl.read(post_mix, [token, stream])
            merged = pl.mul(value, coefficient)
            for source_stream in pl.unroll(hc):
                source_offset = source_stream * hidden + offset
                source = flat[token : token + 1, source_offset : source_offset + D_TILE]
                source = pl.cast(source, pl.FP32)
                coefficient = pl.read(comb_flat, [token, source_stream * hc + stream])
                term = pl.mul(source, coefficient)
                merged = pl.add(merged, term)
            destination = stream * hidden + offset
            merged_bf16 = pl.cast(merged, pl.BF16, mode="rint")
            output_flat[token : token + 1, destination : destination + D_TILE] = merged_bf16
        return output

    return post


def make_decode_hc(capacity, config):
    """Build the standalone HC operator diagnostic entry."""
    hidden = config["hidden_size"]
    hc = config["hc_mult"]
    channels = (hc + 2) * hc
    pre = make_decode_hc_pre_inline(capacity, config)
    post_fn = make_decode_hc_post_inline(capacity, config)

    @pl.jit
    def hc_entry(
        x: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        function: pl.Tensor[[channels, hc * hidden], pl.FP32],
        scale: pl.Tensor[[3], pl.FP32],
        base: pl.Tensor[[channels], pl.FP32],
        incoming: pl.Tensor[[capacity, hc], pl.FP32],
        sublayer: pl.Tensor[[capacity, hidden], pl.BF16],
        next_pre: pl.InOut[pl.Tensor[[capacity, hc], pl.FP32]],
        post: pl.InOut[pl.Tensor[[capacity, hc], pl.FP32]],
        comb: pl.InOut[pl.Tensor[[capacity, hc, hc], pl.FP32]],
        collapsed: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        merged: pl.InOut[pl.Tensor[[capacity, hc, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        collapsed_fp32 = pl.create_tensor([capacity, hidden], dtype=pl.FP32)
        pre(
            x,
            function,
            scale,
            base,
            incoming,
            next_pre,
            post,
            comb,
            collapsed,
            collapsed_fp32,
            active,
        )
        post_fn(sublayer, x, post, comb, merged, active)
        return next_pre, post, comb, collapsed, merged

    return hc_entry


def golden_hc_pre(tensors, *, config, decode=False):
    """Compute phase-specific HC projection, Sinkhorn mixes and incoming collapse."""
    import torch

    active = int(tensors["active"])
    if not 0 <= active <= tensors["x"].shape[0]:
        raise ValueError("HC active rows must fit the actual capacity")
    if not active:
        return
    x = tensors["x"][:active].float()
    hc, hidden = config["hc_mult"], config["hidden_size"]
    flat = x.reshape(active, hc * hidden)
    weight = tensors["function"].float()
    if not decode:
        bits = weight.contiguous().view(torch.int32).to(torch.int64)
        rounded_bits = (((bits + 2048) >> 12) << 12).to(torch.int32)
        weight = rounded_bits.contiguous().view(torch.float32)
    squares = torch.zeros(active, dtype=torch.float32)
    for column in range(0, flat.shape[1], D_TILE):
        chunk = flat[:, column : column + D_TILE]
        squares = squares + (chunk * chunk).sum(dim=1)
    inverse = torch.rsqrt(squares * (1.0 / flat.shape[1]) + config["rms_norm_eps"])
    channels = (hc + 2) * hc
    projected = torch.zeros((active, channels), dtype=torch.float32)
    projection_tile = D_TILE if decode else PROJ_K_TILE
    for column in range(0, flat.shape[1], projection_tile):
        source = flat[:, column:column + projection_tile]
        weights = weight[:, column:column + projection_tile]
        product = (source[:, None] * weights[None]).sum(-1) if decode else source @ weights.T
        projected = projected + product
    projected = projected * inverse[:, None]
    scale, base = tensors["scale"], tensors["base"]
    pre_logits = projected[:, :hc] * scale[0] + base[:hc]
    post_logits = projected[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc]
    pre = torch.reciprocal(torch.exp(-pre_logits) + 1.0) + config["hc_eps"]
    post = torch.reciprocal(torch.exp(-post_logits) + 1.0) * 2.0
    logits = projected[:, 2 * hc :].reshape(active, hc, hc) * scale[2] + base[2 * hc :].reshape(hc, hc)
    repeats = MIX_TILE // hc
    expanded = logits.repeat(1, repeats, repeats)
    shifted = expanded - expanded.amax(dim=-1, keepdim=True)
    exponential = torch.exp(shifted)
    probability = exponential / (exponential.sum(dim=-1, keepdim=True) * (hc / MIX_TILE))
    probability = probability + config["hc_eps"]
    denominator = probability.sum(dim=-2, keepdim=True) * (hc / MIX_TILE) + config["hc_eps"]
    probability = probability / denominator
    for _ in range(config["hc_sinkhorn_iters"] - 1):
        denominator = probability.sum(dim=-1, keepdim=True) * (hc / MIX_TILE) + config["hc_eps"]
        probability = probability / denominator
        denominator = probability.sum(dim=-2, keepdim=True) * (hc / MIX_TILE) + config["hc_eps"]
        probability = probability / denominator
    comb = probability[:, :hc, :hc]
    collapsed = torch.zeros((active, hidden), dtype=torch.float32)
    for stream in range(hc):
        collapsed = collapsed + x[:, stream] * tensors["incoming"][:active, stream, None]
    for name, value in (
        ("next_pre", pre),
        ("post", post),
        ("comb", comb),
        ("collapsed", collapsed),
    ):
        tensors[name][:active] = value.to(tensors[name].dtype)


def golden_hc_post(tensors):
    """Expand the sublayer output and mix residual streams in FP32 order."""
    active = int(tensors["active"])
    x = tensors["x"][:active].float()
    streams = x.shape[1]
    for stream in range(streams):
        value = tensors["sublayer"][:active].float() * tensors["post"][:active, stream, None]
        for source in range(streams):
            term = x[:, source] * tensors["comb"][:active, source, stream, None]
            value = value + term
        tensors["merged"][:active, stream] = value.to(tensors["merged"].dtype)


def golden_hc(tensors, *, config):
    """Evaluate Prefill HC coefficients, incoming collapse and residual mixing."""
    golden_hc_pre(tensors, config=config)
    golden_hc_post(tensors)


def hc_prefix_compare(active):
    """Use the shared floating comparator and exact untouched InOut tails."""
    from golden.validation import ratio_allclose

    numeric = ratio_allclose(rtol=0.05, atol=0.05, max_error_ratio=0.0)
    exact = ratio_allclose(rtol=0.0, atol=0.0, max_error_ratio=0.0)

    def compare(actual, expected, **context):
        for label, region, comparator in (
            ("active", slice(None, active), numeric),
            ("untouched", slice(active, None), exact),
        ):
            if actual[region].numel():
                passed, detail = comparator(actual[region], expected[region], **context)
                if not passed:
                    return False, label + ": " + detail
        return True, ""

    return compare


def hc_golden_cases(*, build=True):
    """Cover empty, singleton, seven-row tail and full-capacity four-stream HC."""
    from functools import partial
    import torch
    from golden import ScalarSpec, TensorSpec

    capacity, hidden, streams = 16, 256, 4
    channels = (streams + 2) * streams
    config = dict(hidden_size=hidden, hc_mult=streams, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-20)
    generator = torch.Generator().manual_seed(417)
    x = torch.randn((capacity, streams, hidden), generator=generator).to(torch.bfloat16)
    function = torch.randn((channels, streams * hidden), generator=generator) * 0.03 + 2.0**-20
    incoming = torch.rand((capacity, streams), generator=generator)
    sublayer = torch.randn((capacity, hidden), generator=generator).to(torch.bfloat16)
    for active in (0, 1, 7, capacity):
        specs = [
            TensorSpec("x", [capacity, streams, hidden], torch.bfloat16, init_value=x),
            TensorSpec("function", [channels, streams * hidden], torch.float32, init_value=function),
            TensorSpec("scale", [3], torch.float32, init_value=torch.tensor([0.3, -0.2, 0.1])),
            TensorSpec("base", [channels], torch.float32, init_value=torch.linspace(-0.2, 0.2, channels)),
            TensorSpec("incoming", [capacity, streams], torch.float32, init_value=incoming),
            TensorSpec("sublayer", [capacity, hidden], torch.bfloat16, init_value=sublayer),
            TensorSpec("next_pre", [capacity, streams], torch.float32, init_value=-23.0),
            TensorSpec("post", [capacity, streams], torch.float32, init_value=-29.0),
            TensorSpec("comb", [capacity, streams, streams], torch.float32, init_value=-31.0),
            TensorSpec("collapsed", [capacity, hidden], torch.bfloat16, init_value=-37.0),
            TensorSpec("merged", [capacity, streams, hidden], torch.bfloat16, init_value=-41.0),
            ScalarSpec("active", torch.int32, active, compile_runtime=True),
        ]
        outputs = ("next_pre", "post", "comb", "collapsed", "merged")
        case = dict(
            name=f"hc-four-stream-active{active}",
            specs=specs,
            golden_fn=partial(golden_hc, config=config),
            compare_fn={name: hc_prefix_compare(active) for name in outputs},
        )
        if build:
            case["fn"] = make_hc(capacity, config)
        yield case


def validate(*, device=0, sim=False, compile_only=False, cpu_only=False):
    """Run bounded HC goldens; CPU mode checks only the reference implementation."""
    import torch
    from golden import run

    if cpu_only and compile_only:
        raise ValueError("CPU golden checks and kernel compilation are separate modes")
    records = []
    outputs = ("next_pre", "post", "comb", "collapsed", "merged")
    for case in hc_golden_cases(build=not cpu_only):
        name = case.pop("name")
        if cpu_only:
            values = {
                spec.name: spec.create_tensor().clone()
                if hasattr(spec, "create_tensor")
                else spec.to_python()
                for spec in case["specs"]
            }
            before = {key: value.clone() for key, value in values.items() if isinstance(value, torch.Tensor)}
            case["golden_fn"](values)
            active = int(values["active"])
            for key in before:
                if key not in outputs and not torch.equal(values[key], before[key]):
                    raise RuntimeError("HC golden mutated input " + key)
            for key in outputs:
                if not torch.isfinite(values[key]).all() or not torch.equal(
                    values[key][active:], before[key][active:]
                ):
                    raise RuntimeError("HC golden changed an inactive tail or produced a nonfinite result")
            if active:
                sums = values["comb"][:active].sum(dim=-2)
                torch.testing.assert_close(sums, torch.ones_like(sums), rtol=0.0, atol=1e-5)
            records.append(dict(name=name, cpu_golden="passed", device="not_run"))
            continue
        result = run(
            **case,
            config=dict(backend_type="a2a3", execution_mode=bool(sim), device_id=device),
            compile_only=compile_only,
            rtol=0.05,
            atol=0.05,
        )
        if not result.passed:
            raise RuntimeError(f"{name}: {result.error}")
        records.append(dict(name=name, compiled=True, device="not_run" if compile_only else "passed"))
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
    print(
        json.dumps(
            validate(device=args.device, sim=args.sim, compile_only=args.compile_only, cpu_only=args.cpu_only)
        )
    )

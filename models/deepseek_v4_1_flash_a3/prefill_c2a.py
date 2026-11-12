# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill compressed attention, source observation and cache readback."""

import pypto.language as pl
from .compressor import COMPRESSOR_PROJECTION_HEAD_DIM as HEAD_DIM, INDEX_WIDTH, RING_ROWS_DYN, compressor_ratio2_prefill, make_compressed_cache_publish, make_fp32_projection
from .gemm import N_TILE, VEC_M_TILE, make_blocked_weight_pack as make_pack_weight, make_dot
from .hc import make_hc_post_inline
from .operator_config import B_DYN, Q_START_DYN, STATE_BLOCKS_DYN
from .prefill_indexer import INDEX_HEADS, QUANT_ROWS_TILE, WEIGHT_ROWS_TILE, make_empty_selection, make_sparse_controls
from .prefill_sparse_attention import make_consumer_sparse_core
from .prefill_swa import GEOMETRY, make_builders
from .qkv_proj_rope import make_bf16_projection_staged
from .quant import make_dequantize, make_quantize
from .rmsnorm import make_blocked_rmsnorm as make_norm, make_prefill_norm
from .rope import make_adjacent_rope as make_rope


def make_readback_entries(entries, active_rows):
    compute_HC_post = pl.jit.inline(entries["HC_post"]._func)
    assert compute_HC_post._func is entries["HC_post"]._func

    @pl.jit
    def readback_HC_post(
        sublayer: pl.InOut[pl.Tensor[[16, 5120], pl.BF16]],
        residual: pl.InOut[pl.Tensor[[16, 4, 5120], pl.BF16]],
        post_mix: pl.InOut[pl.Tensor[[16, 4], pl.FP32]],
        comb: pl.InOut[pl.Tensor[[16, 4, 4], pl.FP32]],
        output: pl.InOut[pl.Tensor[[16, 4, 5120], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_HC_post(sublayer, residual, post_mix, comb, output, active)
        return (sublayer, residual, post_mix, comb, output)

    compute_attention_norm = pl.jit.inline(entries["attention_norm"]._func)
    assert compute_attention_norm._func is entries["attention_norm"]._func

    @pl.jit
    def readback_attention_norm(
        collapsed: pl.InOut[pl.Tensor[[16, 5120], pl.BF16]],
        weight: pl.InOut[pl.Tensor[[5120], pl.BF16]],
        normed: pl.InOut[pl.Tensor[[16, 5120], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_attention_norm(collapsed, weight, normed, active)
        return (collapsed, weight, normed)

    compute_c2_project_pool_publish = pl.jit.inline(entries["c2_project_pool_publish"]._func)
    assert compute_c2_project_pool_publish._func is entries["c2_project_pool_publish"]._func

    @pl.jit
    def readback_c2_project_pool_publish(
        x: pl.InOut[pl.Tensor[[16, 5120], pl.BF16]],
        compressor_wkv: pl.InOut[pl.Tensor[[5120, 512], pl.BF16]],
        compressor_wgate: pl.InOut[pl.Tensor[[5120, 512], pl.BF16]],
        compressor_norm: pl.InOut[pl.Tensor[[512], pl.BF16]],
        index_wk: pl.InOut[pl.Tensor[[512, 128], pl.BF16]],
        index_norm: pl.InOut[pl.Tensor[[128], pl.BF16]],
        source_cos: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        source_sin: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        positions32: pl.InOut[pl.Tensor[[16], pl.INT32]],
        compressed_slots: pl.InOut[pl.Tensor[[16], pl.INT64]],
        query_start: pl.InOut[pl.Tensor[[2], pl.INT32]],
        requests: pl.InOut[pl.Tensor[[16], pl.INT32]],
        state_block_table: pl.InOut[pl.Tensor[[1, 1], pl.INT32]],
        state: pl.InOut[pl.Tensor[[2, 32, 1024], pl.FP32]],
        compressed_cache: pl.InOut[pl.Tensor[[1, 128, 1, 512], pl.BF16]],
        index_cache: pl.InOut[pl.Tensor[[1, 128, 1, 128], pl.INT8]],
        index_scale: pl.InOut[pl.Tensor[[1, 128, 1, 1], pl.FP16]],
        kv_projected: pl.InOut[pl.Tensor[[16, 512], pl.FP32]],
        scores_projected: pl.InOut[pl.Tensor[[16, 512], pl.FP32]],
        compressed: pl.InOut[pl.Tensor[[16, 512], pl.BF16]],
        index_key: pl.InOut[pl.Tensor[[16, 128], pl.BF16]],
        compressed_rotated: pl.InOut[pl.Tensor[[16, 512], pl.BF16]],
        key_payload: pl.InOut[pl.Tensor[[16, 128], pl.INT8]],
        key_scale_f32_8: pl.InOut[pl.Tensor[[16, 8], pl.FP32]],
        wk_input_readback: pl.InOut[pl.Tensor[[16, 512], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_c2_project_pool_publish(
            x,
            compressor_wkv,
            compressor_wgate,
            compressor_norm,
            index_wk,
            index_norm,
            source_cos,
            source_sin,
            positions32,
            compressed_slots,
            query_start,
            requests,
            state_block_table,
            state,
            compressed_cache,
            index_cache,
            index_scale,
            kv_projected,
            scores_projected,
            compressed,
            index_key,
            compressed_rotated,
            key_payload,
            key_scale_f32_8,
            wk_input_readback,
            active,
        )
        return (
            x,
            compressor_wkv,
            compressor_wgate,
            compressor_norm,
            index_wk,
            index_norm,
            source_cos,
            source_sin,
            positions32,
            compressed_slots,
            query_start,
            requests,
            state_block_table,
            state,
            compressed_cache,
            index_cache,
            index_scale,
            kv_projected,
            scores_projected,
            compressed,
            index_key,
            compressed_rotated,
            key_payload,
            key_scale_f32_8,
            wk_input_readback,
        )

    compute_empty_selection = pl.jit.inline(entries["empty_selection"]._func)
    assert compute_empty_selection._func is entries["empty_selection"]._func

    @pl.jit
    def readback_empty_selection(
        raw_selected: pl.InOut[pl.Tensor[[16, 512], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_empty_selection(raw_selected, active)
        return (raw_selected,)

    compute_grouped_wo_a = pl.jit.inline(entries["grouped_wo_a"]._func)
    assert compute_grouped_wo_a._func is entries["grouped_wo_a"]._func

    @pl.jit
    def readback_grouped_wo_a(
        x: pl.InOut[pl.Tensor[[16, 32768], pl.BF16]],
        weight: pl.InOut[pl.Tensor[[8, 1024, 4096], pl.BF16]],
        output: pl.InOut[pl.Tensor[[16, 8192], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_grouped_wo_a(x, weight, output, active)
        return (x, weight, output)

    compute_index_projection = pl.jit.inline(entries["index_projection"]._func)
    assert compute_index_projection._func is entries["index_projection"]._func

    @pl.jit
    def readback_index_projection(
        x: pl.InOut[pl.Tensor[[16, 5120], pl.BF16]],
        query_latent: pl.InOut[pl.Tensor[[16, 1280], pl.BF16]],
        index_wq_b: pl.InOut[pl.Tensor[[1280, 4096], pl.INT8]],
        index_wq_b_scale: pl.InOut[pl.Tensor[[4096], pl.FP32]],
        weights_proj: pl.InOut[pl.Tensor[[5120, 32], pl.BF16]],
        cosine: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        sine: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        query: pl.InOut[pl.Tensor[[16, 32, 128], pl.BF16]],
        weights: pl.InOut[pl.Tensor[[16, 32], pl.FP16]],
        projected: pl.InOut[pl.Tensor[[16, 4096], pl.BF16]],
        scaled_weights: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_index_projection(
            x,
            query_latent,
            index_wq_b,
            index_wq_b_scale,
            weights_proj,
            cosine,
            sine,
            query,
            weights,
            projected,
            scaled_weights,
            active,
        )
        return (
            x,
            query_latent,
            index_wq_b,
            index_wq_b_scale,
            weights_proj,
            cosine,
            sine,
            query,
            weights,
            projected,
            scaled_weights,
        )

    compute_index_quantize = pl.jit.inline(entries["index_quantize"]._func)
    assert compute_index_quantize._func is entries["index_quantize"]._func

    @pl.jit
    def readback_index_quantize(
        query: pl.InOut[pl.Tensor[[16, 32, 128], pl.BF16]],
        payload: pl.InOut[pl.Tensor[[16, 32, 128], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[16, 32], pl.FP16]],
        raw_scale_f32: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        effective_scale_f32: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_index_quantize(query, payload, scales, raw_scale_f32, effective_scale_f32, active)
        return (query, payload, scales, raw_scale_f32, effective_scale_f32)

    compute_inverse_rope = pl.jit.inline(entries["inverse_rope"]._func)
    assert compute_inverse_rope._func is entries["inverse_rope"]._func

    @pl.jit
    def readback_inverse_rope(
        x: pl.InOut[pl.Tensor[[16, 32768], pl.BF16]],
        cosine: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        sine: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        output: pl.InOut[pl.Tensor[[16, 32768], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_inverse_rope(x, cosine, sine, output, active)
        return (x, cosine, sine, output)

    compute_kv_norm = pl.jit.inline(entries["kv_norm"]._func)
    assert compute_kv_norm._func is entries["kv_norm"]._func

    @pl.jit
    def readback_kv_norm(
        x: pl.InOut[pl.Tensor[[16, 512], pl.BF16]],
        weight: pl.InOut[pl.Tensor[[512], pl.BF16]],
        output: pl.InOut[pl.Tensor[[16, 512], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_kv_norm(x, weight, output, active)
        return (x, weight, output)

    compute_kv_rope = pl.jit.inline(entries["kv_rope"]._func)
    assert compute_kv_rope._func is entries["kv_rope"]._func

    @pl.jit
    def readback_kv_rope(
        x: pl.InOut[pl.Tensor[[16, 512], pl.BF16]],
        cosine: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        sine: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        output: pl.InOut[pl.Tensor[[16, 512], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_kv_rope(x, cosine, sine, output, active)
        return (x, cosine, sine, output)

    compute_narrow_sparse_BF16 = pl.jit.inline(entries["narrow_sparse_BF16"]._func)
    assert compute_narrow_sparse_BF16._func is entries["narrow_sparse_BF16"]._func

    @pl.jit
    def readback_narrow_sparse_BF16(
        source: pl.InOut[pl.Tensor[[16, 32768], pl.FP32]],
        output: pl.InOut[pl.Tensor[[16, 32768], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_narrow_sparse_BF16(source, output, active)
        return (source, output)

    compute_q_norm = pl.jit.inline(entries["q_norm"]._func)
    assert compute_q_norm._func is entries["q_norm"]._func

    @pl.jit
    def readback_q_norm(
        x: pl.InOut[pl.Tensor[[16, 1280], pl.BF16]],
        weight: pl.InOut[pl.Tensor[[1280], pl.BF16]],
        output: pl.InOut[pl.Tensor[[16, 1280], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_q_norm(x, weight, output, active)
        return (x, weight, output)

    compute_q_rope = pl.jit.inline(entries["q_rope"]._func)
    assert compute_q_rope._func is entries["q_rope"]._func

    @pl.jit
    def readback_q_rope(
        x: pl.InOut[pl.Tensor[[16, 32768], pl.BF16]],
        cosine: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        sine: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        output: pl.InOut[pl.Tensor[[16, 32768], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_q_rope(x, cosine, sine, output, active)
        return (x, cosine, sine, output)

    compute_sparse_controls = pl.jit.inline(entries["sparse_controls"]._func)
    assert compute_sparse_controls._func is entries["sparse_controls"]._func

    @pl.jit
    def readback_sparse_controls(
        raw_selected: pl.InOut[pl.Tensor[[16, 512], pl.INT32]],
        positions: pl.InOut[pl.Tensor[[16], pl.INT64]],
        request_ids: pl.InOut[pl.Tensor[[16], pl.INT32]],
        long_block_table: pl.InOut[pl.Tensor[[1, 2], pl.INT32]],
        window_block_table: pl.InOut[pl.Tensor[[1, 2], pl.INT32]],
        selected: pl.InOut[pl.Tensor[[16, 512], pl.INT32]],
        selected_lengths: pl.InOut[pl.Tensor[[16, 1], pl.INT32]],
        compressed_indices: pl.InOut[pl.Tensor[[16, 512], pl.INT32]],
        window_indices: pl.InOut[pl.Tensor[[16, 128], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_sparse_controls(
            raw_selected,
            positions,
            request_ids,
            long_block_table,
            window_block_table,
            selected,
            selected_lengths,
            compressed_indices,
            window_indices,
            active,
        )
        return (
            raw_selected,
            positions,
            request_ids,
            long_block_table,
            window_block_table,
            selected,
            selected_lengths,
            compressed_indices,
            window_indices,
        )

    compute_sparse_core = pl.jit.inline(entries["sparse_core"]._func)
    assert compute_sparse_core._func is entries["sparse_core"]._func

    @pl.jit
    def readback_sparse_core(
        query: pl.InOut[pl.Tensor[[active_rows, 64, 512], pl.FP32]],
        window: pl.InOut[pl.Tensor[[2, 128, 512], pl.FP32]],
        window_indices: pl.InOut[pl.Tensor[[active_rows, 128], pl.INT32]],
        compressed: pl.InOut[pl.Tensor[[1, 128, 512], pl.FP32]],
        compressed_indices: pl.InOut[pl.Tensor[[active_rows, 512], pl.INT32]],
        sink: pl.InOut[pl.Tensor[[64], pl.FP32]],
        extents: pl.InOut[pl.Tensor[[active_rows, 2], pl.INT32]],
        output: pl.InOut[pl.Tensor[[active_rows, 64, 512], pl.FP32]],
    ):
        compute_sparse_core(
            query, window, window_indices, compressed, compressed_indices, sink, extents, output
        )
        return (query, window, window_indices, compressed, compressed_indices, sink, extents, output)

    compute_swa_publish = pl.jit.inline(entries["swa_publish"]._func)
    assert compute_swa_publish._func is entries["swa_publish"]._func

    @pl.jit
    def readback_swa_publish(
        value: pl.InOut[pl.Tensor[[16, 512], pl.BF16]],
        slots: pl.InOut[pl.Tensor[[16], pl.INT64]],
        cache: pl.InOut[pl.Tensor[[2, 128, 1, 512], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_swa_publish(value, slots, cache, active)
        return (value, slots, cache)

    compute_wkv = pl.jit.inline(entries["wkv"]._func)
    assert compute_wkv._func is entries["wkv"]._func

    @pl.jit
    def readback_wkv(
        x: pl.InOut[pl.Tensor[[16, 5120], pl.BF16]],
        weight: pl.InOut[pl.Tensor[[5120, 512], pl.INT8]],
        channel_scale: pl.InOut[pl.Tensor[[512], pl.FP32]],
        output: pl.InOut[pl.Tensor[[16, 512], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_wkv(x, weight, channel_scale, output, active)
        return (x, weight, channel_scale, output)

    compute_wo_b = pl.jit.inline(entries["wo_b"]._func)
    assert compute_wo_b._func is entries["wo_b"]._func

    @pl.jit
    def readback_wo_b(
        x: pl.InOut[pl.Tensor[[16, 8192], pl.BF16]],
        weight: pl.InOut[pl.Tensor[[8192, 5120], pl.BF16]],
        output: pl.InOut[pl.Tensor[[16, 5120], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_wo_b(x, weight, output, active)
        return (x, weight, output)

    compute_wq_a = pl.jit.inline(entries["wq_a"]._func)
    assert compute_wq_a._func is entries["wq_a"]._func

    @pl.jit
    def readback_wq_a(
        x: pl.InOut[pl.Tensor[[16, 5120], pl.BF16]],
        weight: pl.InOut[pl.Tensor[[5120, 1280], pl.INT8]],
        channel_scale: pl.InOut[pl.Tensor[[1280], pl.FP32]],
        output: pl.InOut[pl.Tensor[[16, 1280], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_wq_a(x, weight, channel_scale, output, active)
        return (x, weight, channel_scale, output)

    compute_wq_b = pl.jit.inline(entries["wq_b"]._func)
    assert compute_wq_b._func is entries["wq_b"]._func

    @pl.jit
    def readback_wq_b(
        x: pl.InOut[pl.Tensor[[16, 1280], pl.BF16]],
        weight: pl.InOut[pl.Tensor[[1280, 32768], pl.INT8]],
        channel_scale: pl.InOut[pl.Tensor[[32768], pl.FP32]],
        output: pl.InOut[pl.Tensor[[16, 32768], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_wq_b(x, weight, channel_scale, output, active)
        return (x, weight, channel_scale, output)

    return {
        "HC_post": readback_HC_post,
        "attention_norm": readback_attention_norm,
        "c2_project_pool_publish": readback_c2_project_pool_publish,
        "empty_selection": readback_empty_selection,
        "grouped_wo_a": readback_grouped_wo_a,
        "index_projection": readback_index_projection,
        "index_quantize": readback_index_quantize,
        "inverse_rope": readback_inverse_rope,
        "kv_norm": readback_kv_norm,
        "kv_rope": readback_kv_rope,
        "narrow_sparse_BF16": readback_narrow_sparse_BF16,
        "q_norm": readback_q_norm,
        "q_rope": readback_q_rope,
        "sparse_controls": readback_sparse_controls,
        "sparse_core": readback_sparse_core,
        "swa_publish": readback_swa_publish,
        "wkv": readback_wkv,
        "wo_b": readback_wo_b,
        "wq_a": readback_wq_a,
        "wq_b": readback_wq_b,
    }


def make_wk_input_observed_inline(capacity):
    project = pl.jit.inline(make_bf16_projection_staged(HEAD_DIM, INDEX_WIDTH)._func)

    @pl.jit.inline
    def observe(
        source: pl.Tensor[[capacity, HEAD_DIM], pl.BF16],
        weight: pl.Tensor[[HEAD_DIM, INDEX_WIDTH], pl.BF16],
        output: pl.Tensor[[capacity, INDEX_WIDTH], pl.BF16],
        consumer_input: pl.Tensor[[capacity, HEAD_DIM], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd((active + VEC_M_TILE - 1) // VEC_M_TILE, name_hint="L2_observe_actual_WK_input"):
            row = pl.tile.get_block_idx() * VEC_M_TILE
            rows = pl.min(VEC_M_TILE, active - row)
            value = pl.slice(source, [VEC_M_TILE, HEAD_DIM], [row, 0], valid_shape=[rows, HEAD_DIM])
            consumer_input[row : row + VEC_M_TILE, :] = pl.set_validshape(value, rows, HEAD_DIM)
        project(source, weight, output, active)
        return output, consumer_input

    return observe


def make_indexer_cache_publish_128_observed(capacity, blocks):
    """Publish width-128 rows without the generic 256-wide quantizer assumption."""
    rows = blocks * 128

    @pl.jit.inline
    def publish(
        value: pl.Tensor[[capacity, INDEX_WIDTH], pl.BF16],
        slots: pl.Tensor[[capacity], pl.INT64],
        cache: pl.Tensor[[blocks, 128, 1, INDEX_WIDTH], pl.INT8],
        scales: pl.Tensor[[blocks, 128, 1, 1], pl.FP16],
        payload: pl.Tensor[[capacity, INDEX_WIDTH], pl.INT8],
        row_scales: pl.Tensor[[capacity, 8], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd((active + VEC_M_TILE - 1) // VEC_M_TILE, name_hint="a2_c1_index_quant"):
            block = pl.tile.get_block_idx()
            t0 = block * VEC_M_TILE
            rows_valid = pl.min(VEC_M_TILE, active - t0)
            source = pl.slice(
                value, [VEC_M_TILE, INDEX_WIDTH], [t0, 0], valid_shape=[rows_valid, INDEX_WIDTH]
            )
            source = pl.set_validshape(
                pl.fillpad(source, pad_value=pl.PadValue.zero), VEC_M_TILE, INDEX_WIDTH
            )
            value_fp32 = pl.cast(source, pl.FP32)
            maximum_row = pl.reshape(pl.row_max(pl.abs(value_fp32)), [1, VEC_M_TILE])
            maximum_row = pl.maximum(maximum_row, 1e-30)
            scale = pl.reshape(pl.mul(maximum_row, 1.0 / 127.0), [VEC_M_TILE, 1])
            reciprocal_maximum = pl.full([1, VEC_M_TILE], dtype=pl.FP32, value=0.0)
            for reciprocal_lane in pl.range(VEC_M_TILE):
                maximum_scalar: pl.Scalar[pl.FP32] = pl.read(maximum_row, [0, reciprocal_lane])
                reciprocal_scalar: pl.Scalar[pl.FP32] = 1.0 / maximum_scalar
                pl.write(reciprocal_maximum, [0, reciprocal_lane], reciprocal_scalar)
            inverse_row = pl.mul(reciprocal_maximum, 127.0)
            inverse = pl.reshape(inverse_row, [VEC_M_TILE, 1])
            normalized = pl.row_expand_mul(value_fp32, inverse)
            rounded = pl.cast(pl.minimum(pl.maximum(normalized, -127.0), 127.0), pl.INT32, mode="rint")
            payload[t0 : t0 + VEC_M_TILE, :] = pl.set_validshape(
                pl.cast(rounded, pl.INT8, mode="trunc"), rows_valid, INDEX_WIDTH
            )
            row_scales[t0 : t0 + VEC_M_TILE, :] = pl.set_validshape(
                pl.row_expand_mul(pl.full([VEC_M_TILE, 8], dtype=pl.FP32, value=1.0), scale), rows_valid, 8
            )
        cache_flat = pl.reshape(cache, [rows, INDEX_WIDTH])
        scale_flat = pl.reshape(scales, [rows, 1])
        with pl.spmd(active, name_hint="a2_c1_index_publish"):
            token = pl.tile.get_block_idx()
            slot_i64 = pl.read(slots, [token])
            if slot_i64 >= 0:
                slot = pl.cast(slot_i64, pl.INDEX)
                cache_flat[slot : slot + 1, :] = payload[token : token + 1, :]
        with pl.spmd(1, name_hint="a2_c1_index_scale_publish"):
            _worker = pl.tile.get_block_idx()
            for token in pl.range(active):
                slot_i64 = pl.read(slots, [token])
                if slot_i64 >= 0:
                    slot = pl.cast(slot_i64, pl.INDEX)
                    pl.write(scale_flat, [slot, 0], pl.cast(pl.read(row_scales, [token, 0]), pl.FP16))
        return (cache, scales, payload, row_scales)

    return publish


def make_c2_index_publish_projected_observed_inline(capacity, config, cache_blocks=2):
    """Pool projected FP32 rows, then normalize, rotate and publish caches."""
    pairs = config["qk_rope_head_dim"] // 2
    eps = config["rms_norm_eps"]
    project_index = make_wk_input_observed_inline(capacity)
    norm_index = make_norm(capacity, INDEX_WIDTH, eps)
    rope_index = make_rope(capacity, 1, INDEX_WIDTH, config["qk_rope_head_dim"])
    rope_compressed = make_rope(capacity, 1, HEAD_DIM, config["qk_rope_head_dim"])
    publish_compressed = make_compressed_cache_publish(capacity, cache_blocks, HEAD_DIM)
    publish_index = make_indexer_cache_publish_128_observed(capacity, cache_blocks)

    @pl.jit.inline
    def publish(
        kv_proj: pl.Tensor[[capacity, HEAD_DIM], pl.FP32],
        score_proj: pl.Tensor[[capacity, HEAD_DIM], pl.FP32],
        compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16],
        index_wk: pl.Tensor[[HEAD_DIM, INDEX_WIDTH], pl.BF16],
        index_norm: pl.Tensor[[INDEX_WIDTH], pl.BF16],
        source_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        source_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        positions: pl.Tensor[[capacity], pl.INT32],
        compressed_slots: pl.Tensor[[capacity], pl.INT64],
        query_start: pl.Tensor[[Q_START_DYN], pl.INT32],
        requests: pl.Tensor[[capacity], pl.INT32],
        state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
        state: pl.Tensor[[STATE_BLOCKS_DYN, RING_ROWS_DYN, 2 * HEAD_DIM], pl.FP32],
        compressed_cache: pl.Tensor[[cache_blocks, 128, 1, HEAD_DIM], pl.BF16],
        index_cache: pl.Tensor[[cache_blocks, 128, 1, INDEX_WIDTH], pl.INT8],
        index_scale: pl.Tensor[[cache_blocks, 128, 1, 1], pl.FP16],
        compressed: pl.Tensor[[capacity, HEAD_DIM], pl.BF16],
        index_key: pl.Tensor[[capacity, INDEX_WIDTH], pl.BF16],
        compressed_rotated: pl.Tensor[[capacity, HEAD_DIM], pl.BF16],
        key_payload: pl.Tensor[[capacity, INDEX_WIDTH], pl.INT8],
        key_scale_f32_8: pl.Tensor[[capacity, 8], pl.FP32],
        wk_input_readback: pl.Tensor[[capacity, HEAD_DIM], pl.BF16],
        active: pl.Scalar[pl.INT32],
        projections_ready: pl.Scalar[pl.TASK_ID],
    ):
        compressor_ratio2_prefill(
            kv_proj,
            score_proj,
            query_start,
            positions,
            requests,
            state_block_table,
            state,
            compressor_norm,
            compressed,
            active,
            projections_ready,
        )
        rope_compressed(compressed, source_cos, source_sin, compressed_rotated, active)
        publish_compressed(compressed_rotated, compressed_slots, compressed_cache, active)
        projected_index = pl.create_tensor([capacity, INDEX_WIDTH], dtype=pl.BF16)
        project_index(compressed, index_wk, projected_index, wk_input_readback, active)
        norm_index(projected_index, index_norm, index_key, active)
        rope_index(index_key, source_cos, source_sin, index_key, active)
        publish_index(
            index_key, compressed_slots, index_cache, index_scale, key_payload, key_scale_f32_8, active
        )
        return (
            state,
            compressed,
            compressed_cache,
            index_key,
            index_cache,
            index_scale,
            compressed_rotated,
            key_payload,
            key_scale_f32_8,
            wk_input_readback,
        )

    return publish


def make_indexer_projection_observed(capacity, config):
    """Project and rotate index queries and emit scaled FP16 head weights."""
    hidden = config["hidden_size"]
    latent = config["q_lora_rank"]
    rope_width = config["qk_rope_head_dim"]
    pairs = rope_width // 2
    index_width = INDEX_HEADS * INDEX_WIDTH
    packed_rows = latent * (index_width // N_TILE)
    weights_scale = INDEX_WIDTH ** (-0.5) * INDEX_HEADS ** (-0.5)
    quantize = make_quantize(capacity, latent)
    pack = make_pack_weight(latent, index_width)
    dot = make_dot(capacity, latent, index_width)
    dequantize = make_dequantize(capacity, index_width)
    rotate = make_rope(capacity, INDEX_HEADS, INDEX_WIDTH, rope_width)
    project_weights = pl.jit.inline(make_bf16_projection_staged(hidden, INDEX_HEADS)._func)

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
        projected: pl.Tensor[[capacity, index_width], pl.BF16],
        scaled_weights: pl.Tensor[[capacity, INDEX_HEADS], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        latent_i8 = pl.create_tensor([capacity, latent], dtype=pl.INT8)
        latent_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize(query_latent, latent_i8, latent_scale, active)
        packed = pl.create_tensor([packed_rows, N_TILE], dtype=pl.INT8)
        pack(index_wq_b, packed)
        accumulator = pl.create_tensor([capacity, index_width], dtype=pl.INT32)
        dot(latent_i8, packed, accumulator, active)
        dequantize(accumulator, index_wq_b_scale, latent_scale, projected, active)
        query_flat = pl.reshape(query, [capacity, index_width])
        rotate(projected, cosine, sine, query_flat, active)
        projected_weights = pl.create_tensor([capacity, INDEX_HEADS], dtype=pl.BF16)
        project_weights(x, weights_proj, projected_weights, active)
        with pl.spmd((active + WEIGHT_ROWS_TILE - 1) // WEIGHT_ROWS_TILE, name_hint="a2_index_weights"):
            block = pl.tile.get_block_idx()
            row = block * WEIGHT_ROWS_TILE
            rows = pl.min(WEIGHT_ROWS_TILE, active - row)
            value = pl.slice(
                projected_weights, [WEIGHT_ROWS_TILE, INDEX_HEADS], [row, 0], valid_shape=[rows, INDEX_HEADS]
            )
            scaled = pl.mul(pl.cast(value, pl.FP32), weights_scale)
            scaled_weights[row : row + WEIGHT_ROWS_TILE, :] = pl.set_validshape(scaled, rows, INDEX_HEADS)
            narrowed = pl.cast(scaled, pl.FP16, mode="rint")
            weights[row : row + WEIGHT_ROWS_TILE, :] = pl.set_validshape(narrowed, rows, INDEX_HEADS)
        return (query, weights, projected, scaled_weights)

    return indexer_projection


def make_indexer_query_quantize_observed(capacity):
    """Quantize each index head using its rounded FP16 scale."""

    @pl.jit
    def indexer_query_quantize(
        query: pl.Tensor[[capacity, INDEX_HEADS, INDEX_WIDTH], pl.BF16],
        payload: pl.InOut[pl.Tensor[[capacity, INDEX_HEADS, INDEX_WIDTH], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, INDEX_HEADS], pl.FP16]],
        raw_scale_f32: pl.Tensor[[capacity, INDEX_HEADS], pl.FP32],
        effective_scale_f32: pl.Tensor[[capacity, INDEX_HEADS], pl.FP32],
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
            scale = pl.maximum(rounded_scale, 2.0 ** (-24))
            normalized = pl.row_expand_div(value, scale)
            rounded = pl.cast(normalized, pl.INT32, mode="rint")
            bounded = pl.minimum(pl.maximum(rounded, -127), 127)
            quantized = pl.cast(bounded, pl.INT8, mode="trunc")
            output[row : row + QUANT_ROWS_TILE, :] = pl.set_validshape(quantized, rows, INDEX_WIDTH)
            scale_row = pl.reshape(pl.cast(scale, pl.FP16, mode="rint"), [1, QUANT_ROWS_TILE])
            token = row // INDEX_HEADS
            head = row % INDEX_HEADS
            scales[token : token + 1, head : head + QUANT_ROWS_TILE] = scale_row
            raw_scale_row = pl.reshape(raw_scale, [1, QUANT_ROWS_TILE])
            raw_scale_f32[token : token + 1, head : head + QUANT_ROWS_TILE] = raw_scale_row
            effective_scale_row = pl.reshape(scale, [1, QUANT_ROWS_TILE])
            effective_scale_f32[token : token + 1, head : head + QUANT_ROWS_TILE] = effective_scale_row
        return (payload, scales, raw_scale_f32, effective_scale_f32)

    return indexer_query_quantize


def make_c2_program(config, *, project_inputs=True):
    """Pool and publish C2 rows, optionally projecting the source in DSL."""
    if type(project_inputs) is not bool:
        raise ValueError("The compressor projection boundary must be selected by the host")
    project = make_fp32_projection(5120, 512)
    publish_c2 = make_c2_index_publish_projected_observed_inline(16, config, cache_blocks=1)

    @pl.jit
    def c2_observed_program(
        x: pl.InOut[pl.Tensor[[16, 5120], pl.BF16]],
        compressor_wkv: pl.InOut[pl.Tensor[[5120, 512], pl.BF16]],
        compressor_wgate: pl.InOut[pl.Tensor[[5120, 512], pl.BF16]],
        compressor_norm: pl.InOut[pl.Tensor[[512], pl.BF16]],
        index_wk: pl.InOut[pl.Tensor[[512, 128], pl.BF16]],
        index_norm: pl.InOut[pl.Tensor[[128], pl.BF16]],
        source_cos: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        source_sin: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        positions32: pl.InOut[pl.Tensor[[16], pl.INT32]],
        compressed_slots: pl.InOut[pl.Tensor[[16], pl.INT64]],
        query_start: pl.InOut[pl.Tensor[[2], pl.INT32]],
        requests: pl.InOut[pl.Tensor[[16], pl.INT32]],
        state_block_table: pl.InOut[pl.Tensor[[1, 1], pl.INT32]],
        state: pl.InOut[pl.Tensor[[2, 32, 1024], pl.FP32]],
        compressed_cache: pl.InOut[pl.Tensor[[1, 128, 1, 512], pl.BF16]],
        index_cache: pl.InOut[pl.Tensor[[1, 128, 1, 128], pl.INT8]],
        index_scale: pl.InOut[pl.Tensor[[1, 128, 1, 1], pl.FP16]],
        kv_projected: pl.InOut[pl.Tensor[[16, 512], pl.FP32]],
        scores_projected: pl.InOut[pl.Tensor[[16, 512], pl.FP32]],
        compressed: pl.InOut[pl.Tensor[[16, 512], pl.BF16]],
        index_key: pl.InOut[pl.Tensor[[16, 128], pl.BF16]],
        compressed_rotated: pl.InOut[pl.Tensor[[16, 512], pl.BF16]],
        key_payload: pl.InOut[pl.Tensor[[16, 128], pl.INT8]],
        key_scale_f32_8: pl.InOut[pl.Tensor[[16, 8], pl.FP32]],
        wk_input_readback: pl.InOut[pl.Tensor[[16, 512], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        if project_inputs:
            kv_ready = project(x, compressor_wkv, kv_projected, active)
            scores_ready = project(x, compressor_wgate, scores_projected, active)
            ready = pl.system.task_dummy(deps=[kv_ready, scores_ready])
            publish_c2(
                kv_projected,
                scores_projected,
                compressor_norm,
                index_wk,
                index_norm,
                source_cos,
                source_sin,
                positions32,
                compressed_slots,
                query_start,
                requests,
                state_block_table,
                state,
                compressed_cache,
                index_cache,
                index_scale,
                compressed,
                index_key,
                compressed_rotated,
                key_payload,
                key_scale_f32_8,
                wk_input_readback,
                active,
                ready,
            )
        else:
            ready = pl.system.task_dummy(deps=[])
            publish_c2(
                kv_projected,
                scores_projected,
                compressor_norm,
                index_wk,
                index_norm,
                source_cos,
                source_sin,
                positions32,
                compressed_slots,
                query_start,
                requests,
                state_block_table,
                state,
                compressed_cache,
                index_cache,
                index_scale,
                compressed,
                index_key,
                compressed_rotated,
                key_payload,
                key_scale_f32_8,
                wk_input_readback,
                active,
                ready,
            )
        return (
            x,
            compressor_wkv,
            compressor_wgate,
            compressor_norm,
            index_wk,
            index_norm,
            source_cos,
            source_sin,
            positions32,
            compressed_slots,
            query_start,
            requests,
            state_block_table,
            state,
            compressed_cache,
            index_cache,
            index_scale,
            kv_projected,
            scores_projected,
            compressed,
            index_key,
            compressed_rotated,
            key_payload,
            key_scale_f32_8,
            wk_input_readback,
        )

    return c2_observed_program


def make_entries(active, *, compressor_projected=False):
    """Build complete query, Full2 producer, sparse and BF16 output stages."""
    if type(active) is not int or active not in (1, 7, 16):
        raise ValueError("Original bounded active1/7/16 required")
    if type(compressor_projected) is not bool:
        raise ValueError("Select the actual product compressor projection boundary explicitly")
    config = dict(GEOMETRY)
    base = make_builders(16, config)
    names = (
        "wq_a",
        "q_norm",
        "wq_b",
        "q_rope",
        "wkv",
        "kv_norm",
        "kv_rope",
        "inverse_rope",
        "grouped_wo_a",
        "wo_b",
    )
    entries = {name: base[name] for name in names}

    @pl.jit
    def narrow_sparse_BF16(
        source: pl.InOut[pl.Tensor[[16, 32768], pl.FP32]],
        output: pl.InOut[pl.Tensor[[16, 32768], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(active * 128, name_hint="L2_sparse_native_BF16_boundary"):
            token = block // 128
            column = block % 128 * 256
            values = source[token : token + 1, column : column + 256]
            output[token : token + 1, column : column + 256] = pl.cast(values, pl.BF16, mode="rint")
        return source, output

    entries.update(
        attention_norm=make_prefill_norm(16, 5120),
        c2_project_pool_publish=make_c2_program(config, project_inputs=not compressor_projected),
        swa_publish=pl.jit(make_compressed_cache_publish(16, 2, 512)._func),
        index_projection=make_indexer_projection_observed(16, config),
        index_quantize=make_indexer_query_quantize_observed(16),
        empty_selection=make_empty_selection(16),
        sparse_controls=make_sparse_controls(16, 1, 2, 2, long_page_rows=64, compress_ratio=2),
        sparse_core=make_consumer_sparse_core(active, config, compressed_blocks=1),
        narrow_sparse_BF16=narrow_sparse_BF16,
        HC_post=pl.jit(make_hc_post_inline(16, config)._func),
    )
    return make_readback_entries(entries, active)


def make_physical_compressed_entries(
    capacity,
    config,
    cache_blocks,
    requests,
    table_pages,
    *,
    compress_ratio=2,
    cache_layout=None,
    layer_id=2,
    network_cache=False,
):
    """Build paged physical-layer C2, index packing and full-head compressed attention entries."""
    from .indexer import INDEXER_SELECT_KEY_CAPACITY as KEY_CAPACITY, make_prefill_indexer_select as make_indexer_select
    from .rope import make_rope_tables
    from .rmsnorm import make_blocked_rmsnorm as make_norm
    from .rope import make_adjacent_rope as make_rope
    from .compressor import make_c2_pool_observed
    from .compressor import make_index_cache_publish_observed
    from .prefill_indexer import make_index_key_gather, make_physical_index_query_entries
    from .prefill_indexer import make_sparse_index_pack
    from .compressor import make_c2_projection
    from .qkv_proj_rope import make_bf16_projection_staged
    from .swa import make_swa_leaves, make_swa_read

    publish, _gather, _attend = make_swa_leaves(capacity, cache_blocks, config)
    sparse_config = dict(config, sliding_window=640)
    theta, scaling = config["compress_rope_theta"], config.get("rope_scaling")
    entries = dict(
        rope_tables=make_rope_tables(capacity, 64, theta, scaling),
        source_rope_tables=make_rope_tables(capacity, 64, theta, scaling),
        c2_projection=make_c2_projection(capacity, config["hidden_size"]),
        c2_pool=make_c2_pool_observed(capacity, cache_blocks, requests),
        compressed_rope=pl.jit(make_rope(capacity, 1, 512, 64)._func),
        compressed_publish=pl.jit(publish._func),
        index_wk=make_bf16_projection_staged(512, 128, capacity=capacity),
        index_norm=pl.jit(make_norm(capacity, 128, config["rms_norm_eps"])._func),
        index_rope=pl.jit(make_rope(capacity, 1, 128, 64)._func),
        index_publish=make_index_cache_publish_observed(capacity, cache_blocks),
        sparse_index_pack=make_sparse_index_pack(capacity),
        compressed_attention=make_swa_read(capacity, cache_blocks, sparse_config),
    )
    entries.update(make_physical_index_query_entries(capacity, config, cache_blocks, requests, table_pages))
    if compress_ratio not in (1, 2):
        raise ValueError("Physical compressed attention supports ratio1 or ratio2")
    if compress_ratio == 1:
        from .indexer import make_candidate_block_pool, make_candidate_pool_publish
        from .cache import make_paged_cache_copy, physical_page_geometry

        geometry = (
            None
            if network_cache
            else physical_page_geometry(cache_layout, f"model.layers.{layer_id}.self_attn.", cache_blocks)
        )
        del entries["c2_projection"]
        del entries["c2_pool"]
        entries.update(
            compressor_projection=make_bf16_projection_staged(5120, 512, capacity=capacity),
            compressor_norm=pl.jit(make_norm(capacity, 512, config["rms_norm_eps"])._func),
            index_publish=make_index_cache_publish_observed(capacity, cache_blocks, page_rows=128),
            index_key_gather=make_index_key_gather(cache_blocks, requests, table_pages, page_rows=128),
            index_select=make_indexer_select(capacity, KEY_CAPACITY, compress_ratio=1),
            sparse_controls=make_sparse_controls(
                capacity, requests, table_pages, table_pages, long_page_rows=128, compress_ratio=1
            ),
            candidate_pool=make_candidate_block_pool(capacity),
            candidate_publish=make_candidate_pool_publish(capacity),
        )
        dtypes = {"torch.bfloat16": pl.BF16, "torch.int8": pl.INT8, "torch.float16": pl.FP16}
        for name, descriptor in (geometry or {}).items():
            payload, stride, offset = descriptor["payload"], descriptor["stride"], descriptor["offset"]
            dtype = dtypes[descriptor["dtype"]]
            entries["cache_load_" + name] = make_paged_cache_copy(
                cache_blocks, payload, stride, payload, source_offset=offset, dtype=dtype
            )
            entries["cache_store_" + name] = make_paged_cache_copy(
                cache_blocks, payload, payload, stride, destination_offset=offset, dtype=dtype
            )
    if network_cache:
        from .cache import resolve_layer_role
        from .cache import make_network_cache_entries, network_cache_geometry

        role = resolve_layer_role(layer_id)
        if compress_ratio != role["ratio"]:
            raise ValueError("Network producer ratio differs from its original physical role")
        geometry = network_cache_geometry(cache_layout, role, cache_blocks)
        entries.update(make_network_cache_entries(capacity, config, geometry, role))
    return entries

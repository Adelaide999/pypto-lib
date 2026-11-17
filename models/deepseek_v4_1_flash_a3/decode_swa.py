# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode attention composition and SWA block without distributed MoE."""

import pypto.language as pl
from .hc import make_decode_hc_post_inline as make_hc_post_inline, make_decode_hc_pre_inline as make_hc_pre_inline
from .o_proj import make_decode_output_projection_bf16 as make_output_projection_bf16
from .qkv_proj_rope import make_decode_qkv_w8a8_inline as make_qkv_w8a8_inline
from .rmsnorm import DECODE_D as D, DECODE_EPS as EPS, NORM_T_TILE, rms_norm_tiled
from .swa import make_decode_swa_inline as make_swa_inline
from .qkv_proj_rope import make_decode_qkv_w8a8_kn_inline


def make_attention_block(capacity, config):
    """Build the layer-zero attention block with delayed HC coefficients."""
    hidden = config["hidden_size"]
    latent = config["q_lora_rank"]
    head_dim = config["head_dim"]
    heads = config["num_attention_heads"]
    width = heads * head_dim
    pairs = config["qk_rope_head_dim"] // 2
    window = config["sliding_window"]
    groups = config["o_groups"]
    group_input = width // groups
    rank = config["o_lora_rank"]
    hc = config["hc_mult"]
    channels = (hc + 2) * hc
    if hidden != D or config["rms_norm_eps"] != EPS:
        raise ValueError("The shared model RMSNorm configuration must match this block")
    pre = make_hc_pre_inline(capacity, config)
    attention = make_attention_inline(capacity, config)
    post_fn = make_hc_post_inline(capacity, config)

    @pl.jit
    def attention_block(
        x_hc: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        hc_function: pl.Tensor[[channels, hc * hidden], pl.FP32],
        hc_scale: pl.Tensor[[3], pl.FP32],
        hc_base: pl.Tensor[[channels], pl.FP32],
        incoming: pl.Tensor[[capacity, hc], pl.FP32],
        norm_weight: pl.Tensor[[hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden * (latent // 64), 64], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm_weight: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent * (width // 64), 64], pl.INT8],
        wq_b_scale: pl.Tensor[[width], pl.FP32],
        wkv: pl.Tensor[[hidden * (head_dim // 64), 64], pl.INT8],
        wkv_scale: pl.Tensor[[head_dim], pl.FP32],
        kv_norm_weight: pl.Tensor[[head_dim], pl.BF16],
        wo_a: pl.Tensor[[groups, group_input, rank], pl.BF16],
        wo_b: pl.Tensor[[groups * rank, hidden], pl.BF16],
        rope_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        query_latent: pl.InOut[pl.Tensor[[capacity, latent], pl.BF16]],
        query: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        window_kv: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        cache: pl.InOut[pl.Tensor[[2, 128, 1, head_dim], pl.BF16]],
        cache_lens: pl.InOut[pl.Tensor[[2], pl.INT32]],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        next_pre: pl.InOut[pl.Tensor[[capacity, hc], pl.FP32]],
        post: pl.InOut[pl.Tensor[[capacity, hc], pl.FP32]],
        comb: pl.InOut[pl.Tensor[[capacity, hc, hc], pl.FP32]],
        collapsed: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        normed: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        hidden_output: pl.InOut[pl.Tensor[[capacity, hc, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):

        collapsed_fp32 = pl.create_tensor([capacity, hidden], dtype=pl.FP32)
        pre(
            x_hc,
            hc_function,
            hc_scale,
            hc_base,
            incoming,
            next_pre,
            post,
            comb,
            collapsed,
            collapsed_fp32,
            active,
        )
        for block in pl.spmd((active + NORM_T_TILE - 1) // NORM_T_TILE, name_hint="a2_block_norm"):
            rms_norm_tiled(collapsed, norm_weight, normed, block, active)
        attention(
            normed,
            wq_a,
            wq_a_scale,
            q_norm_weight,
            wq_b,
            wq_b_scale,
            wkv,
            wkv_scale,
            kv_norm_weight,
            wo_a,
            wo_b,
            rope_cos,
            rope_sin,
            slots,
            indices,
            sink,
            positions,
            query_latent,
            query,
            window_kv,
            cache,
            cache_lens,
            output,
            active,
        )
        post_fn(output, x_hc, post, comb, hidden_output, active)
        return (
            positions,
            query_latent,
            query,
            window_kv,
            cache,
            cache_lens,
            output,
            next_pre,
            post,
            comb,
            collapsed,
            normed,
            hidden_output,
        )

    return attention_block


def make_attention_inline(capacity, config):
    """Build QKV, paged SWA, inverse RoPE and both output projections."""
    hidden = config["hidden_size"]
    latent = config["q_lora_rank"]
    head_dim = config["head_dim"]
    heads = config["num_attention_heads"]
    width = heads * head_dim
    pairs = config["qk_rope_head_dim"] // 2
    window = config["sliding_window"]
    groups = config["o_groups"]
    group_input = width // groups
    rank = config["o_lora_rank"]
    qkv = make_qkv_w8a8_inline(capacity, config)
    swa = make_swa_inline(capacity, 2, config)
    project = make_output_projection_bf16(
        capacity,
        heads,
        head_dim,
        config["qk_rope_head_dim"],
        groups,
        rank,
        hidden,
    )

    @pl.jit.inline
    def attention_inline(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden * (latent // 64), 64], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm_weight: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent * (width // 64), 64], pl.INT8],
        wq_b_scale: pl.Tensor[[width], pl.FP32],
        wkv: pl.Tensor[[hidden * (head_dim // 64), 64], pl.INT8],
        wkv_scale: pl.Tensor[[head_dim], pl.FP32],
        kv_norm_weight: pl.Tensor[[head_dim], pl.BF16],
        wo_a: pl.Tensor[[groups, group_input, rank], pl.BF16],
        wo_b: pl.Tensor[[groups * rank, hidden], pl.BF16],
        rope_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        positions: pl.Tensor[[capacity], pl.INT64],
        query_latent: pl.Tensor[[capacity, latent], pl.BF16],
        query: pl.Tensor[[capacity, width], pl.BF16],
        window_kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        cache: pl.Tensor[[2, 128, 1, head_dim], pl.BF16],
        cache_lens: pl.Tensor[[2], pl.INT32],
        output: pl.Tensor[[capacity, hidden], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        flat_cache = pl.reshape(cache, [2, 128, head_dim])
        qkv(
            x,
            wq_a,
            wq_a_scale,
            q_norm_weight,
            wq_b,
            wq_b_scale,
            wkv,
            wkv_scale,
            kv_norm_weight,
            rope_cos,
            rope_sin,
            positions,
            query_latent,
            query,
            window_kv,
            flat_cache,
            cache_lens,
            active,
        )
        attended = pl.create_tensor([capacity, width], dtype=pl.BF16)
        swa(query, window_kv, slots, indices, sink, cache, attended, active)
        rotated = pl.create_tensor([capacity, width], dtype=pl.BF16)
        grouped = pl.create_tensor([capacity, groups * rank], dtype=pl.BF16)
        project(attended, wo_a, wo_b, rope_cos, rope_sin, rotated, grouped, output, active)
        return positions, query_latent, query, window_kv, cache, cache_lens, output

    return attention_inline


def make_attention(capacity, config):
    """Build the standalone JIT wrapper for shared inline attention."""
    hidden = config["hidden_size"]
    latent = config["q_lora_rank"]
    head_dim = config["head_dim"]
    heads = config["num_attention_heads"]
    width = heads * head_dim
    pairs = config["qk_rope_head_dim"] // 2
    window = config["sliding_window"]
    groups = config["o_groups"]
    group_input = width // groups
    rank = config["o_lora_rank"]
    compute = make_attention_inline(capacity, config)

    @pl.jit
    def attention(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden * (latent // 64), 64], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm_weight: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent * (width // 64), 64], pl.INT8],
        wq_b_scale: pl.Tensor[[width], pl.FP32],
        wkv: pl.Tensor[[hidden * (head_dim // 64), 64], pl.INT8],
        wkv_scale: pl.Tensor[[head_dim], pl.FP32],
        kv_norm_weight: pl.Tensor[[head_dim], pl.BF16],
        wo_a: pl.Tensor[[groups, group_input, rank], pl.BF16],
        wo_b: pl.Tensor[[groups * rank, hidden], pl.BF16],
        rope_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        query_latent: pl.InOut[pl.Tensor[[capacity, latent], pl.BF16]],
        query: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        window_kv: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        cache: pl.InOut[pl.Tensor[[2, 128, 1, head_dim], pl.BF16]],
        cache_lens: pl.InOut[pl.Tensor[[2], pl.INT32]],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):

        compute(
            x,
            wq_a,
            wq_a_scale,
            q_norm_weight,
            wq_b,
            wq_b_scale,
            wkv,
            wkv_scale,
            kv_norm_weight,
            wo_a,
            wo_b,
            rope_cos,
            rope_sin,
            slots,
            indices,
            sink,
            positions,
            query_latent,
            query,
            window_kv,
            cache,
            cache_lens,
            output,
            active,
        )
        return positions, query_latent, query, window_kv, cache, cache_lens, output

    return attention


def make_attention_kn_inline(capacity, config):
    """Build QKV, paged SWA, inverse RoPE and both output projections."""
    hidden = config["hidden_size"]
    latent = config["q_lora_rank"]
    head_dim = config["head_dim"]
    heads = config["num_attention_heads"]
    width = heads * head_dim
    pairs = config["qk_rope_head_dim"] // 2
    window = config["sliding_window"]
    groups = config["o_groups"]
    group_input = width // groups
    rank = config["o_lora_rank"]
    qkv = make_decode_qkv_w8a8_kn_inline(capacity, config)
    swa = make_swa_inline(capacity, 2, config)
    project = make_output_projection_bf16(
        capacity,
        heads,
        head_dim,
        config["qk_rope_head_dim"],
        groups,
        rank,
        hidden,
    )

    @pl.jit.inline
    def attention_inline(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden, latent], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm_weight: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent, width], pl.INT8],
        wq_b_scale: pl.Tensor[[width], pl.FP32],
        wkv: pl.Tensor[[hidden, head_dim], pl.INT8],
        wkv_scale: pl.Tensor[[head_dim], pl.FP32],
        kv_norm_weight: pl.Tensor[[head_dim], pl.BF16],
        wo_a: pl.Tensor[[groups, group_input, rank], pl.BF16],
        wo_b: pl.Tensor[[groups * rank, hidden], pl.BF16],
        rope_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        positions: pl.Tensor[[capacity], pl.INT64],
        query_latent: pl.Tensor[[capacity, latent], pl.BF16],
        query: pl.Tensor[[capacity, width], pl.BF16],
        window_kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        cache: pl.Tensor[[2, 128, 1, head_dim], pl.BF16],
        cache_lens: pl.Tensor[[2], pl.INT32],
        output: pl.Tensor[[capacity, hidden], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        flat_cache = pl.reshape(cache, [2, 128, head_dim])
        qkv(
            x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
            kv_norm_weight, rope_cos, rope_sin,
            positions, query_latent, query, window_kv, flat_cache, cache_lens, active,
        )
        attended = pl.create_tensor([capacity, width], dtype=pl.BF16)
        swa(query, window_kv, slots, indices, sink, cache, attended, active)
        rotated = pl.create_tensor([capacity, width], dtype=pl.BF16)
        grouped = pl.create_tensor([capacity, groups * rank], dtype=pl.BF16)
        project(attended, wo_a, wo_b, rope_cos, rope_sin, rotated, grouped, output, active)
        return positions, query_latent, query, window_kv, cache, cache_lens, output

    return attention_inline


def make_attention_kn(capacity, config):
    """Build the standalone JIT wrapper for shared inline attention."""
    hidden = config["hidden_size"]
    latent = config["q_lora_rank"]
    head_dim = config["head_dim"]
    heads = config["num_attention_heads"]
    width = heads * head_dim
    pairs = config["qk_rope_head_dim"] // 2
    window = config["sliding_window"]
    groups = config["o_groups"]
    group_input = width // groups
    rank = config["o_lora_rank"]
    compute = make_attention_kn_inline(capacity, config)

    @pl.jit
    def attention(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden, latent], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm_weight: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent, width], pl.INT8],
        wq_b_scale: pl.Tensor[[width], pl.FP32],
        wkv: pl.Tensor[[hidden, head_dim], pl.INT8],
        wkv_scale: pl.Tensor[[head_dim], pl.FP32],
        kv_norm_weight: pl.Tensor[[head_dim], pl.BF16],
        wo_a: pl.Tensor[[groups, group_input, rank], pl.BF16],
        wo_b: pl.Tensor[[groups * rank, hidden], pl.BF16],
        rope_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        query_latent: pl.InOut[pl.Tensor[[capacity, latent], pl.BF16]],
        query: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        window_kv: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        cache: pl.InOut[pl.Tensor[[2, 128, 1, head_dim], pl.BF16]],
        cache_lens: pl.InOut[pl.Tensor[[2], pl.INT32]],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):

        compute(
            x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
            kv_norm_weight, wo_a, wo_b,
            rope_cos, rope_sin, slots, indices, sink,
            positions, query_latent, query, window_kv, cache, cache_lens, output, active,
        )
        return positions, query_latent, query, window_kv, cache, cache_lens, output

    return attention


def make_attention_kn_block(capacity, config):
    """Build the layer-zero attention block with delayed HC coefficients."""
    hidden = config["hidden_size"]
    latent = config["q_lora_rank"]
    head_dim = config["head_dim"]
    heads = config["num_attention_heads"]
    width = heads * head_dim
    pairs = config["qk_rope_head_dim"] // 2
    window = config["sliding_window"]
    groups = config["o_groups"]
    group_input = width // groups
    rank = config["o_lora_rank"]
    hc = config["hc_mult"]
    channels = (hc + 2) * hc
    if hidden != D or config["rms_norm_eps"] != EPS:
        raise ValueError("The shared model RMSNorm configuration must match this block")
    pre = make_hc_pre_inline(capacity, config)
    attention = make_attention_kn_inline(capacity, config)
    post_fn = make_hc_post_inline(capacity, config)

    @pl.jit
    def attention_block(
        x_hc: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        hc_function: pl.Tensor[[channels, hc * hidden], pl.FP32],
        hc_scale: pl.Tensor[[3], pl.FP32],
        hc_base: pl.Tensor[[channels], pl.FP32],
        incoming: pl.Tensor[[capacity, hc], pl.FP32],
        norm_weight: pl.Tensor[[hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden, latent], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm_weight: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent, width], pl.INT8],
        wq_b_scale: pl.Tensor[[width], pl.FP32],
        wkv: pl.Tensor[[hidden, head_dim], pl.INT8],
        wkv_scale: pl.Tensor[[head_dim], pl.FP32],
        kv_norm_weight: pl.Tensor[[head_dim], pl.BF16],
        wo_a: pl.Tensor[[groups, group_input, rank], pl.BF16],
        wo_b: pl.Tensor[[groups * rank, hidden], pl.BF16],
        rope_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        query_latent: pl.InOut[pl.Tensor[[capacity, latent], pl.BF16]],
        query: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        window_kv: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        cache: pl.InOut[pl.Tensor[[2, 128, 1, head_dim], pl.BF16]],
        cache_lens: pl.InOut[pl.Tensor[[2], pl.INT32]],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        next_pre: pl.InOut[pl.Tensor[[capacity, hc], pl.FP32]],
        post: pl.InOut[pl.Tensor[[capacity, hc], pl.FP32]],
        comb: pl.InOut[pl.Tensor[[capacity, hc, hc], pl.FP32]],
        collapsed: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        normed: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        hidden_output: pl.InOut[pl.Tensor[[capacity, hc, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):

        collapsed_fp32 = pl.create_tensor([capacity, hidden], dtype=pl.FP32)
        pre(
            x_hc, hc_function, hc_scale, hc_base, incoming, next_pre, post, comb,
            collapsed, collapsed_fp32, active,
        )
        for block in pl.spmd((active + NORM_T_TILE - 1) // NORM_T_TILE, name_hint="a2_block_norm"):
            rms_norm_tiled(collapsed, norm_weight, normed, block, active)
        attention(
            normed, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
            kv_norm_weight, wo_a, wo_b,
            rope_cos, rope_sin, slots, indices, sink,
            positions, query_latent, query, window_kv, cache, cache_lens, output, active,
        )
        post_fn(output, x_hc, post, comb, hidden_output, active)
        return (
            positions, query_latent, query, window_kv, cache, cache_lens, output,
            next_pre, post, comb, collapsed, normed, hidden_output,
        )

    return attention_block

def golden_swa(tensors):
    """Compose Decode HC, W8A8 QKV, paged SWA and BF16 output projection goldens."""
    from dataclasses import asdict
    import torch
    from .config import FLASH
    from .hc import golden_hc_pre, golden_hc_post
    from .o_proj import decode_output_projection_bf16_reference
    from .qkv_proj_rope import decode_qkv_reference
    from .rmsnorm import decode_rms_norm_reference
    from .swa import golden_swa as golden_attention

    active = int(tensors["active"])
    if not 0 <= active <= tensors["x_hc"].shape[0]:
        raise ValueError("SWA active rows must fit the actual capacity")
    tensors["output"][active:] = 0
    if not active:
        return
    config = asdict(FLASH)
    hc_values = dict(
        x=tensors["x_hc"], function=tensors["hc_function"], scale=tensors["hc_scale"],
        base=tensors["hc_base"], incoming=tensors["incoming"], next_pre=tensors["next_pre"],
        post=tensors["post"], comb=tensors["comb"], collapsed=tensors["collapsed"], active=active,
    )
    golden_hc_pre(hc_values, config=config, decode=True)
    normalized = decode_rms_norm_reference(
        tensors["collapsed"][:active], tensors["norm_weight"], FLASH.rms_norm_eps,
    )
    tensors["normed"][:active] = normalized
    projected = decode_qkv_reference(
        normalized, tensors, tensors["rope_cos"][:active], tensors["rope_sin"][:active],
        config, weight_layout="kn",
    )
    for name in ("query_latent", "query", "window_kv"):
        tensors[name][:active] = projected[name]
    attended = torch.empty_like(tensors["query"])
    attention_values = dict(
        query=tensors["query"], kv=tensors["window_kv"], slots=tensors["slots"],
        indices=tensors["indices"], sink=tensors["sink"], cache=tensors["cache"], output=attended, active=active,
    )
    golden_attention(attention_values, config=config)
    output = decode_output_projection_bf16_reference(
        attended[:active], tensors["wo_a"], tensors["wo_b"],
        tensors["rope_cos"][:active], tensors["rope_sin"][:active],
        FLASH.num_attention_heads, FLASH.head_dim, FLASH.qk_rope_head_dim,
    )
    tensors["output"][:active] = output["output"]
    hc_values.update(sublayer=tensors["output"], merged=tensors["hidden_output"])
    golden_hc_post(hc_values)


def build_swa_specs(active=7, *, seed=417):
    """Build full-width KN Decode operands, paged history and inactive sentinels."""
    import torch
    from golden import ScalarSpec, TensorSpec
    from .config import FLASH

    capacity, hidden, streams = 16, FLASH.hidden_size, FLASH.hc_mult
    latent, heads, dim = FLASH.q_lora_rank, FLASH.num_attention_heads, FLASH.head_dim
    groups, rank, window = FLASH.o_groups, FLASH.o_lora_rank, FLASH.sliding_window
    width, pairs = heads * dim, FLASH.qk_rope_head_dim // 2
    channels = (streams + 2) * streams
    if not 0 <= active <= capacity:
        raise ValueError("SWA active rows must be in [0, 16]")

    def random_tensor(shape, dtype, offset, scale=1.0):
        generator = torch.Generator().manual_seed(seed + offset)
        if dtype == torch.int8:
            return torch.randint(-127, 128, shape, dtype=dtype, generator=generator)
        return (torch.randn(shape, generator=generator) * scale).to(dtype)

    def random_spec(name, shape, dtype, offset, scale=1.0):
        return TensorSpec(name, shape, dtype, init_value=lambda: random_tensor(shape, dtype, offset, scale))

    positions = torch.arange(capacity, dtype=torch.int64) + 127
    slots = torch.stack((positions // 128, positions % 128), dim=-1)
    slots[2::3] = -1
    slots[active:] = 999
    indices = torch.stack([torch.arange(position - window + 1, position + 1) for position in positions])
    indices[0] = -1
    indices[:, ::5] = -1
    indices[active:] = 999
    angles = positions.float()[:, None] * torch.linspace(0.0001, 0.1, pairs)[None]
    specs = [
        random_spec("x_hc", [capacity, streams, hidden], torch.bfloat16, 0, 0.25),
        random_spec("hc_function", [channels, streams * hidden], torch.float32, 1, 0.01),
        TensorSpec("hc_scale", [3], torch.float32, init_value=torch.tensor([0.3, -0.2, 0.1])),
        TensorSpec("hc_base", [channels], torch.float32, init_value=torch.linspace(-0.2, 0.2, channels)),
        TensorSpec("incoming", [capacity, streams], torch.float32, init_value=1.0 / streams),
        TensorSpec("norm_weight", [hidden], torch.bfloat16, init_value=1.0),
        random_spec("wq_a", [hidden, latent], torch.int8, 2),
        TensorSpec("wq_a_scale", [latent], torch.float32, init_value=0.001),
        TensorSpec("q_norm_weight", [latent], torch.bfloat16, init_value=1.0),
        random_spec("wq_b", [latent, width], torch.int8, 3),
        TensorSpec("wq_b_scale", [width], torch.float32, init_value=0.001),
        random_spec("wkv", [hidden, dim], torch.int8, 4),
        TensorSpec("wkv_scale", [dim], torch.float32, init_value=0.001),
        TensorSpec("kv_norm_weight", [dim], torch.bfloat16, init_value=1.0),
        random_spec("wo_a", [groups, width // groups, rank], torch.bfloat16, 5, 0.01),
        random_spec("wo_b", [groups * rank, hidden], torch.bfloat16, 6, 0.01),
        TensorSpec("rope_cos", [capacity, pairs], torch.float32, init_value=angles.cos()),
        TensorSpec("rope_sin", [capacity, pairs], torch.float32, init_value=angles.sin()),
        TensorSpec("slots", [capacity, 2], torch.int64, init_value=slots),
        TensorSpec("indices", [capacity, window], torch.int32, init_value=indices.int()),
        random_spec("sink", [heads], torch.float32, 7),
        TensorSpec("positions", [capacity], torch.int64, init_value=positions),
        TensorSpec("query_latent", [capacity, latent], torch.bfloat16, init_value=-23.0),
        TensorSpec("query", [capacity, width], torch.bfloat16, init_value=-29.0),
        TensorSpec("window_kv", [capacity, dim], torch.bfloat16, init_value=-31.0),
        random_spec("cache", [2, 128, 1, dim], torch.bfloat16, 8, 0.1),
        TensorSpec("cache_lens", [2], torch.int32, init_value=torch.tensor([128, 17], dtype=torch.int32)),
        TensorSpec("output", [capacity, hidden], torch.bfloat16, init_value=-37.0),
        TensorSpec("next_pre", [capacity, streams], torch.float32, init_value=-41.0),
        TensorSpec("post", [capacity, streams], torch.float32, init_value=-43.0),
        TensorSpec("comb", [capacity, streams, streams], torch.float32, init_value=-47.0),
        TensorSpec("collapsed", [capacity, hidden], torch.bfloat16, init_value=-53.0),
        TensorSpec("normed", [capacity, hidden], torch.bfloat16, init_value=-59.0),
        TensorSpec("hidden_output", [capacity, streams, hidden], torch.bfloat16, init_value=-61.0),
        ScalarSpec("active", torch.int32, active, compile_runtime=True),
    ]
    return specs


def golden_cache_compare(active):
    """Translate active page/offset pairs into the harness's mapped-pool contract."""
    import torch
    from golden import mapped_pool_ratio_allclose

    def compare(actual, expected, **context):
        inputs = dict(context.get("inputs", {}))
        slots = inputs.get("slots")
        if slots is None or slots.ndim != 2 or slots.shape[1] != 2:
            return False, "SWA cache comparison requires page/offset slots"
        mapping = torch.full((slots.shape[0],), -1, dtype=torch.int64)
        pairs = slots[:active].cpu()
        published = pairs[:, 0] >= 0
        if ((pairs[published, 0] >= actual.shape[0]) | (pairs[published, 1] < 0)
                | (pairs[published, 1] >= actual.shape[1])).any():
            return False, "SWA publication slot is outside the allocated cache"
        mapping[:active][published] = pairs[published, 0] * actual.shape[1] + pairs[published, 1]
        inputs["published_rows"] = mapping
        mapped = mapped_pool_ratio_allclose(
            "published_rows", mapping_shape=tuple(mapping.shape), block_size=actual.shape[1],
            pool_name="cache", atol=0.05, rtol=0.05, max_error_ratio=0.0,
        )
        return mapped(actual, expected, **dict(context, inputs=inputs))

    return compare


def main(argv=None):
    """Validate the existing full-width Decode SWA block with the Golden Harness."""
    import argparse
    from dataclasses import asdict
    from golden import ratio_allclose, run
    from .config import FLASH
    from .qkv_proj_rope import golden_prefix_compare
    from .utils import validate_golden_active

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3"], default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--active", type=int, choices=range(17))
    parser.add_argument("--seed", type=int, default=417)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data")
    parser.add_argument("--runtime-dir")
    args = parser.parse_args(argv)
    if (args.golden_data or args.runtime_dir) and args.active is None:
        parser.error("replay requires one explicit --active case")
    validate_golden_active(args.golden_data, args.active)
    entry = make_attention_kn_block(16, asdict(FLASH))
    for active in (0, 1, 7, 16) if args.active is None else (args.active,):
        numeric = golden_prefix_compare(active)
        exact = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
        result = run(
            fn=entry,
            specs=build_swa_specs(active, seed=args.seed),
            golden_fn=golden_swa,
            compare_fn={
                "query_latent": numeric, "query": numeric, "window_kv": numeric,
                "cache": golden_cache_compare(active), "positions": exact, "cache_lens": exact,
                "output": numeric, "next_pre": numeric, "post": numeric, "comb": numeric,
                "collapsed": numeric, "normed": numeric, "hidden_output": numeric,
            },
            config=dict(backend_type=args.platform, execution_mode=args.sim, device_id=args.device),
            compile_only=args.compile_only,
            save_data=args.save_data,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
        if args.save_data and not args.compile_only:
            print(f"[GOLDEN] data: {result.work_dir / 'data'}")


if __name__ == "__main__":
    main()

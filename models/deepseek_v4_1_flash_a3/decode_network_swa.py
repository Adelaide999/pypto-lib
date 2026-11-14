# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Full-width layer-zero A2 W8A8 attention with BF16 paged SWA."""

from __future__ import annotations

import pypto.language as pl

from .decode_network_decode_qkv_w8a8 import make_qkv_w8a8_inline
from .decode_network_decode_output_proj import make_output_projection_bf16


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


# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""BF16 paged SWA cache publication, window gather and sink attention."""

import pypto.language as pl


# tiling
HEAD_TILE = 16
WINDOW_TILE = 128


def make_swa_inline(capacity, cache_blocks, config):
    """Build a BF16 PA_BBND SWA leaf for the full configured query width."""
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    window = config["sliding_window"]
    if window % WINDOW_TILE:
        raise ValueError("A2 SWA requires a window divisible by its 128-token softmax tile")
    width = heads * head_dim
    cache_rows = cache_blocks * 128
    score_scale = head_dim ** -0.5

    @pl.jit.inline
    def publish(
        kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(cache, [cache_rows, head_dim])
        for token in pl.spmd(active, name_hint="a2_swa_publish"):
            page = pl.read(slots, [token, 0])
            offset = pl.read(slots, [token, 1])
            if page >= 0:
                address = page * 128 + offset
                row = pl.cast(address, pl.INDEX)
                flat[row:row + 1, :] = kv[token:token + 1, :]
        return cache

    @pl.jit.inline
    def gather(
        cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        selected: pl.Tensor[[capacity * window, head_dim], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(cache, [cache_rows, head_dim])
        for block in pl.spmd(active * window, name_hint="a2_swa_gather"):
            token = block // window
            column = block % window
            index = pl.read(indices, [token, column])
            if index >= 0:
                row = pl.cast(index, pl.INDEX)
                selected[block:block + 1, :] = flat[row:row + 1, :]
            else:
                selected[block:block + 1, :] = pl.full([1, head_dim], dtype=pl.BF16, value=0.0)
        return selected

    @pl.jit.inline
    def attend(
        query: pl.Tensor[[capacity, width], pl.BF16],
        selected: pl.Tensor[[capacity * window, head_dim], pl.BF16],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        output: pl.Tensor[[capacity, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        qflat = pl.reshape(query, [capacity * heads, head_dim])
        oflat = pl.reshape(output, [capacity * heads, head_dim])
        for block in pl.spmd(active * (heads // HEAD_TILE), name_hint="a2_swa_attention"):
            token = block // (heads // HEAD_TILE)
            head = block % (heads // HEAD_TILE) * HEAD_TILE
            query_row = token * heads + head
            # Include the zero-value sink before narrowing exp(scores-max) to
            # BF16, matching the native softmax's rounding boundary.
            maximum = pl.reshape(sink[head:head + HEAD_TILE], [1, HEAD_TILE])
            denominator = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=1.0)
            numerator = pl.full([HEAD_TILE, head_dim], dtype=pl.FP32, value=0.0)
            for part in pl.range(window // WINDOW_TILE):
                cache_row = token * window + part * WINDOW_TILE
                q = qflat[query_row:query_row + HEAD_TILE, :]
                kv = selected[cache_row:cache_row + WINDOW_TILE, :]
                dot = pl.matmul(q, kv, b_trans=True)
                scores = pl.mul(dot, score_scale)
                index_tile = indices[token:token + 1, part * WINDOW_TILE:part * WINDOW_TILE + WINDOW_TILE]
                index_float = pl.cast(index_tile, pl.FP32)
                shifted_index = pl.add(index_float, 1.0)
                positive_index = pl.maximum(shifted_index, 0.0)
                valid = pl.minimum(positive_index, 1.0)
                invalid = pl.sub(valid, 1.0)
                bias = pl.mul(invalid, 1e30)
                scores = pl.col_expand_add(scores, bias)
                row_max = pl.row_max(scores)
                local_max = pl.reshape(row_max, [1, HEAD_TILE])
                next_max = pl.maximum(maximum, local_max)
                max_difference = pl.sub(maximum, next_max)
                correction = pl.exp(max_difference)
                max_column = pl.reshape(next_max, [HEAD_TILE, 1])
                shifted = pl.row_expand_sub(scores, max_column)
                exponential = pl.exp(shifted)
                probabilities = pl.col_expand_mul(exponential, valid)
                row_sum = pl.row_sum(probabilities)
                probability_sum = pl.reshape(row_sum, [1, HEAD_TILE])
                old_denominator = pl.mul(denominator, correction)
                denominator = pl.add(old_denominator, probability_sum)
                weights = pl.cast(probabilities, pl.BF16, mode="rint")
                weighted = pl.matmul(weights, kv)
                correction_column = pl.reshape(correction, [HEAD_TILE, 1])
                corrected = pl.row_expand_mul(numerator, correction_column)
                numerator = pl.add(corrected, weighted)
                maximum = next_max
            one = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=1.0)
            normalized = pl.div(one, denominator, high_precision=True)
            normalized_column = pl.reshape(normalized, [HEAD_TILE, 1])
            result = pl.row_expand_mul(numerator, normalized_column)
            oflat[query_row:query_row + HEAD_TILE, :] = pl.cast(result, pl.BF16, mode="rint")
        return output

    @pl.jit.inline
    def swa_inline(
        query: pl.Tensor[[capacity, width], pl.BF16],
        kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
        output: pl.Tensor[[capacity, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        publish(kv, slots, cache, active)
        selected = pl.create_tensor([capacity * window, head_dim], dtype=pl.BF16)
        gather(cache, indices, selected, active)
        attend(query, selected, indices, sink, output, active)
        return cache, output

    return swa_inline


def make_swa(capacity, cache_blocks, config):
    """Build the standalone JIT wrapper for shared inline SWA compute."""
    head_dim = config["head_dim"]
    heads = config["num_attention_heads"]
    width = heads * head_dim
    window = config["sliding_window"]
    compute = make_swa_inline(capacity, cache_blocks, config)

    @pl.jit
    def swa(
        query: pl.Tensor[[capacity, width], pl.BF16],
        kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        cache: pl.InOut[pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16]],
        output: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute(query, kv, slots, indices, sink, cache, output, active)
        return cache, output

    return swa


# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""BF16 sliding-window cache publication and gather."""


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

import pypto.language as pl

from .operator_config import CONTRACT

BLOCK_SIZE = 128

def write_swa_cache_reference(
    cache: torch.Tensor,
    values: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> torch.Tensor:
    """Write BF16 rows into a ``[blocks, 128, 1, head_dim]`` cache."""

    import torch

    if cache.dtype != torch.bfloat16 or values.dtype != torch.bfloat16:
        raise TypeError("A2/A3 SWA cache and values must be BF16")
    result = cache.clone()
    flat = result.view(-1, result.shape[-1])
    for row, slot in zip(values, slot_mapping.reshape(-1, 2).tolist()):
        page, offset = slot
        if page >= 0:
            flat[page * CONTRACT.swa_block_size + offset].copy_(row)
    return result


def gather_swa_cache_reference(cache: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather physical cache rows; negative indices produce zero rows."""

    import torch

    flat = cache.reshape(-1, cache.shape[-1])
    result = torch.zeros(*indices.shape, cache.shape[-1], dtype=cache.dtype, device=cache.device)
    valid = indices >= 0
    if valid.any():
        result[valid] = flat[indices[valid].long()]
    return result


def make_swa_cache_publish(
    capacity: int,
    cache_blocks: int,
    head_dim: int,
    *,
    decode_state: bool = False,
):
    cache_rows = cache_blocks * BLOCK_SIZE

    if decode_state:

        @pl.jit.inline
        def publish(
            values: pl.Tensor[[capacity, head_dim], pl.BF16],
            slots: pl.Tensor[[capacity, 2], pl.INT64],
            positions: pl.Tensor[[capacity], pl.INT64],
            request_ids: pl.Tensor[[capacity], pl.INT32],
            cache_source_ids: pl.Tensor[[capacity], pl.INT32],
            finished: pl.Tensor[[capacity], pl.INT8],
            cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
            active: pl.Scalar[pl.INT32],
        ):
            flat = pl.reshape(cache, [cache_rows, head_dim])
            for token in pl.spmd(capacity, name_hint="dsv41_a2a3_swa_publish"):
                position = pl.read(positions, [token])
                request_id = pl.read(request_ids, [token])
                source_id = pl.read(cache_source_ids, [token])
                done = pl.read(finished, [token])
                if token < active and position >= 0 and request_id >= 0 and source_id >= 0 and done == 0:
                    page = pl.read(slots, [token, 0])
                    offset = pl.read(slots, [token, 1])
                    if page >= 0:
                        physical = pl.cast(page * 128 + offset, pl.INDEX)
                        flat[physical:physical + 1, :] = values[token:token + 1, :]
            return cache

    else:

        @pl.jit.inline
        def publish(
            values: pl.Tensor[[capacity, head_dim], pl.BF16],
            slots: pl.Tensor[[capacity, 2], pl.INT64],
            cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
            active: pl.Scalar[pl.INT32],
        ):
            flat = pl.reshape(cache, [cache_rows, head_dim])
            for token in pl.spmd(capacity, name_hint="dsv41_a2a3_swa_publish"):
                if token < active:
                    page = pl.read(slots, [token, 0])
                    offset = pl.read(slots, [token, 1])
                    if page >= 0:
                        physical = pl.cast(page * 128 + offset, pl.INDEX)
                        flat[physical:physical + 1, :] = values[token:token + 1, :]
            return cache

    return publish


def make_swa_cache_gather(
    capacity: int,
    window: int,
    cache_blocks: int,
    head_dim: int,
    *,
    decode_state: bool = False,
):
    cache_rows = cache_blocks * BLOCK_SIZE

    if decode_state:

        @pl.jit.inline
        def gather(
            cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
            indices: pl.Tensor[[capacity, window], pl.INT32],
            finished: pl.Tensor[[capacity], pl.INT8],
            output: pl.Tensor[[capacity * window, head_dim], pl.BF16],
            active: pl.Scalar[pl.INT32],
        ):
            flat = pl.reshape(cache, [cache_rows, head_dim])
            for block in pl.spmd(capacity * window, name_hint="dsv41_a2a3_swa_gather"):
                token = block // window
                column = block % window
                if token < active and pl.read(finished, [token]) == 0:
                    index = pl.read(indices, [token, column])
                    if index >= 0:
                        physical = pl.cast(index, pl.INDEX)
                        output[block:block + 1, :] = flat[physical:physical + 1, :]
                    else:
                        output[block:block + 1, :] = pl.full([1, head_dim], dtype=pl.BF16, value=0.0)
            return output

    else:

        @pl.jit.inline
        def gather(
            cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
            indices: pl.Tensor[[capacity, window], pl.INT32],
            output: pl.Tensor[[capacity * window, head_dim], pl.BF16],
            active: pl.Scalar[pl.INT32],
        ):
            flat = pl.reshape(cache, [cache_rows, head_dim])
            for block in pl.spmd(capacity * window, name_hint="dsv41_a2a3_swa_gather"):
                token = block // window
                column = block % window
                if token < active:
                    index = pl.read(indices, [token, column])
                    if index >= 0:
                        physical = pl.cast(index, pl.INDEX)
                        output[block:block + 1, :] = flat[physical:physical + 1, :]
                    else:
                        output[block:block + 1, :] = pl.full([1, head_dim], dtype=pl.BF16, value=0.0)
            return output

    return gather


__all__ = [
    "gather_swa_cache_reference",
    "make_swa_cache_gather",
    "make_swa_cache_publish",
    "write_swa_cache_reference",
]


# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2/A3 BF16 sink-aware sliding-window attention kernel and reference."""


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

import pypto.language as pl


def swa_attention_reference(
    query: torch.Tensor,
    selected: torch.Tensor,
    indices: torch.Tensor,
    sink: torch.Tensor,
    head_dim: int,
    *,
    valid_tokens: int | None = None,
) -> torch.Tensor:
    """Compute sink-aware attention over gathered BF16 SWA rows."""

    import torch

    if query.ndim != 2 or query.shape[-1] % head_dim:
        raise ValueError("query must be [T, H*D] with a divisible head dimension")
    tokens = query.shape[0]
    heads = query.shape[-1] // head_dim
    if selected.ndim == 2:
        if selected.shape[0] != tokens * indices.shape[1]:
            raise ValueError("flat selected rows do not match indices")
        selected = selected.reshape(tokens, indices.shape[1], head_dim)
    if selected.shape[:2] != indices.shape or selected.shape[-1] != head_dim:
        raise ValueError("selected and indices shape mismatch")
    if sink.shape != (heads,):
        raise ValueError(f"sink must have shape {(heads,)}, got {tuple(sink.shape)}")
    active = tokens if valid_tokens is None else int(valid_tokens)
    output = torch.zeros_like(query)
    if active == 0:
        return output
    q = query[:active].float().reshape(active, heads, head_dim)
    k = selected[:active].float()
    score = torch.matmul(q, k.transpose(-1, -2)) * (head_dim ** -0.5)
    valid = indices[:active].ge(0).unsqueeze(1)
    score = score.masked_fill(~valid, float("-inf"))
    sink_score = sink.float().reshape(1, heads, 1).expand(active, -1, -1)
    all_score = torch.cat((score, sink_score), dim=-1)
    probability = torch.softmax(all_score, dim=-1)
    result = torch.matmul(probability[..., :-1], k)
    output[:active] = result.reshape(active, heads * head_dim).to(torch.bfloat16)
    return output


def make_swa_attention(
    capacity: int,
    heads: int,
    head_dim: int,
    window: int,
    *,
    decode_state: bool = False,
):
    """Build the sink-aware A2/A3 attention reduction kernel.

    The default keeps the prefill/layer ABI. Decode requests opt into the
    request-state controls explicitly so existing callers retain their
    public signature.
    """

    if capacity <= 0 or heads <= 0 or head_dim <= 0 or window <= 0:
        raise ValueError("attention dimensions must be positive")
    if heads % 16:
        raise ValueError("A2/A3 attention uses a 16-head vector tile")
    publish = make_swa_cache_publish(capacity, 2, head_dim, decode_state=True)
    gather = make_swa_cache_gather(capacity, window, 2, head_dim, decode_state=True)
    head_tile = 16
    scale = head_dim ** -0.5

    @pl.jit.inline
    def decode_attend(
        query: pl.Tensor[[capacity, heads * head_dim], pl.BF16],
        window_kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        positions: pl.Tensor[[capacity], pl.INT64],
        request_ids: pl.Tensor[[capacity], pl.INT32],
        cache_source_ids: pl.Tensor[[capacity], pl.INT32],
        finished: pl.Tensor[[capacity], pl.INT8],
        cache: pl.Tensor[[2, 128, 1, head_dim], pl.BF16],
        output: pl.Tensor[[capacity, heads * head_dim], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        publish(window_kv, slots, positions, request_ids, cache_source_ids, finished, cache, active)
        selected_flat = pl.create_tensor([capacity * window, head_dim], dtype=pl.BF16)
        gather(cache, indices, finished, selected_flat, active)
        qflat = pl.reshape(query, [capacity * heads, head_dim])
        oflat = pl.reshape(output, [capacity * heads, head_dim])
        kflat = selected_flat
        for block in pl.spmd((active + 0) * (heads // head_tile), name_hint="dsv41_a2a3_swa_attention"):
            token = block // (heads // head_tile)
            head = (block % (heads // head_tile)) * head_tile
            q = qflat[token * heads + head:token * heads + head + head_tile, :]
            k = kflat[token * window:token * window + window, :]
            scores = pl.mul(pl.matmul(q, k, b_trans=True, out_dtype=pl.FP32), scale)
            idx = pl.cast(indices[token:token + 1, :], pl.FP32)
            valid = pl.minimum(pl.maximum(pl.add(idx, 1.0), 0.0), 1.0)
            scores = pl.col_expand_add(scores, pl.mul(pl.sub(valid, 1.0), 1.0e30))
            maximum = pl.reshape(pl.row_max(scores), [1, head_tile])
            probabilities = pl.col_expand_mul(
                pl.exp(pl.row_expand_sub(scores, pl.reshape(maximum, [head_tile, 1]))), valid
            )
            denominator = pl.row_sum(probabilities)
            sink_score = pl.reshape(sink[head:head + head_tile], [1, head_tile])
            final_max = pl.maximum(maximum, sink_score)
            correction = pl.exp(pl.sub(maximum, final_max))
            denominator = pl.add(
                pl.mul(pl.reshape(denominator, [1, head_tile]), correction),
                pl.exp(pl.sub(sink_score, final_max)),
            )
            weighted = pl.matmul(pl.cast(probabilities, pl.BF16, mode="rint"), k, out_dtype=pl.FP32)
            result = pl.row_expand_mul(weighted, pl.reshape(pl.div(correction, denominator), [head_tile, 1]))
            oflat[token * heads + head:token * heads + head + head_tile, :] = pl.cast(
                result, pl.BF16, mode="rint"
            )
        return output, cache, selected_flat

    if decode_state:
        return decode_attend

    @pl.jit.inline
    def attend(
        query: pl.Tensor[[capacity, heads * head_dim], pl.BF16],
        window_kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        cache: pl.Tensor[[2, 128, 1, head_dim], pl.BF16],
        output: pl.Tensor[[capacity, heads * head_dim], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        positions = pl.full([capacity], dtype=pl.INT64, value=0)
        request_ids = pl.full([capacity], dtype=pl.INT32, value=0)
        cache_source_ids = pl.full([capacity], dtype=pl.INT32, value=0)
        finished = pl.full([capacity], dtype=pl.INT8, value=0)
        return decode_attend(
            query,
            window_kv,
            slots,
            indices,
            sink,
            positions,
            request_ids,
            cache_source_ids,
            finished,
            cache,
            output,
            active,
        )

    return attend


__all__ = [
    "gather_swa_cache_reference",
    "make_swa_attention",
    "swa_attention_reference",
    "write_swa_cache_reference",
]


# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Delayed HC, RMSNorm, A2 W8A8 attention and BF16 residual expansion."""

import pypto.language as pl

from .decode_network_decode_hc import make_hc_pre_inline, make_hc_post_inline
from .decode_network_decode_rmsnorm import D, EPS, NORM_T_TILE, rms_norm_tiled


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


# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Complete physical L0 Decode composition for the two-rank EP2 gate.

The attention and FFN residual streams stay rank-local.  The MoE child owns
the only cross-rank transport and receives the normalized FFN stream.  This
entry is a product path: all model computation is emitted through PyPTO.
"""


from dataclasses import asdict

import pypto.language as pl
import pypto.language.distributed as pld

from .config import FLASH
from .decode_network_moe import (
    AUX_PAD,
    COUNT_PAD,
    SIGNAL_PAD,
    D,
    EP_SIZE,
    LANE_ROWS,
    MOE_INTER,
    N_EXPERTS,
    N_LOCAL_EXPERTS,
    T,
    TOPK,
    decode_ep2_rank,
)
from .decode_network_decode_hc import make_hc_post_inline, make_hc_pre_inline
from .decode_network_decode_rmsnorm import NORM_T_TILE, rms_norm_tiled_with_fp32


CONFIG = asdict(FLASH)
HC = FLASH.hc_mult
CHANNELS = FLASH.mix_hc
HIDDEN = FLASH.hidden_size
LATENT = FLASH.q_lora_rank
WIDTH = FLASH.num_attention_heads * FLASH.head_dim
ROPE_PAIRS = FLASH.qk_rope_head_dim // 2
GROUP_INPUT = WIDTH // FLASH.o_groups
O_RANK = FLASH.o_lora_rank
WINDOW = FLASH.sliding_window

ATTENTION = pl.jit.inline(make_attention_block(T, CONFIG)._func)
FFN_PRE = make_hc_pre_inline(T, CONFIG)
FFN_POST = make_hc_post_inline(T, CONFIG)
EP2_RANK = pl.jit.inline(decode_ep2_rank._func)


@pl.jit(auto_scope=False)
def decode_layer_ep2_rank(
    x_hc: pl.Tensor[[T, HC, HIDDEN], pl.BF16],
    attn_hc_function: pl.Tensor[[CHANNELS, HC * HIDDEN], pl.FP32],
    attn_hc_scale: pl.Tensor[[3], pl.FP32],
    attn_hc_base: pl.Tensor[[CHANNELS], pl.FP32],
    attn_incoming: pl.Tensor[[T, HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[HIDDEN], pl.BF16],
    wq_a: pl.Tensor[[HIDDEN, LATENT], pl.INT8],
    wq_a_scale: pl.Tensor[[LATENT], pl.FP32],
    q_norm_weight: pl.Tensor[[LATENT], pl.BF16],
    wq_b: pl.Tensor[[LATENT, WIDTH], pl.INT8],
    wq_b_scale: pl.Tensor[[WIDTH], pl.FP32],
    wkv: pl.Tensor[[HIDDEN, FLASH.head_dim], pl.INT8],
    wkv_scale: pl.Tensor[[FLASH.head_dim], pl.FP32],
    kv_norm_weight: pl.Tensor[[FLASH.head_dim], pl.BF16],
    wo_a: pl.Tensor[[FLASH.o_groups, GROUP_INPUT, O_RANK], pl.BF16],
    wo_b: pl.Tensor[[FLASH.o_groups * O_RANK, HIDDEN], pl.BF16],
    rope_cos: pl.Tensor[[T, ROPE_PAIRS], pl.FP32],
    rope_sin: pl.Tensor[[T, ROPE_PAIRS], pl.FP32],
    slots: pl.Tensor[[T, 2], pl.INT64],
    indices: pl.Tensor[[T, WINDOW], pl.INT32],
    sink: pl.Tensor[[FLASH.num_attention_heads], pl.FP32],
    positions: pl.InOut[pl.Tensor[[T], pl.INT64]],
    query_latent: pl.InOut[pl.Tensor[[T, LATENT], pl.BF16]],
    query: pl.InOut[pl.Tensor[[T, WIDTH], pl.BF16]],
    window_kv: pl.InOut[pl.Tensor[[T, FLASH.head_dim], pl.BF16]],
    cache: pl.InOut[pl.Tensor[[2, 128, 1, FLASH.head_dim], pl.BF16]],
    cache_lens: pl.InOut[pl.Tensor[[2], pl.INT32]],
    attn_output: pl.InOut[pl.Tensor[[T, HIDDEN], pl.BF16]],
    attn_next_pre: pl.InOut[pl.Tensor[[T, HC], pl.FP32]],
    attn_post: pl.InOut[pl.Tensor[[T, HC], pl.FP32]],
    attn_comb: pl.InOut[pl.Tensor[[T, HC, HC], pl.FP32]],
    attn_collapsed: pl.InOut[pl.Tensor[[T, HIDDEN], pl.BF16]],
    attn_normed: pl.InOut[pl.Tensor[[T, HIDDEN], pl.BF16]],
    after_attention: pl.InOut[pl.Tensor[[T, HC, HIDDEN], pl.BF16]],
    ffn_hc_function: pl.Tensor[[CHANNELS, HC * HIDDEN], pl.FP32],
    ffn_hc_scale: pl.Tensor[[3], pl.FP32],
    ffn_hc_base: pl.Tensor[[CHANNELS], pl.FP32],
    ffn_norm_weight: pl.Tensor[[HIDDEN], pl.BF16],
    ffn_next_pre: pl.InOut[pl.Tensor[[T, HC], pl.FP32]],
    ffn_post: pl.InOut[pl.Tensor[[T, HC], pl.FP32]],
    ffn_comb: pl.InOut[pl.Tensor[[T, HC, HC], pl.FP32]],
    ffn_collapsed: pl.InOut[pl.Tensor[[T, HIDDEN], pl.BF16]],
    ffn_normed: pl.InOut[pl.Tensor[[T, HIDDEN], pl.BF16]],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    text_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    vision_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, 2560], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    shared_gate_up: pl.Tensor[[D, MOE_INTER * 2], pl.INT8],
    shared_gate_up_scale: pl.Tensor[[MOE_INTER * 2], pl.FP32],
    shared_down: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_down_scale: pl.Tensor[[D], pl.FP32],
    lane_x: pld.DistributedTensor[[LANE_ROWS, D], pl.INT8],
    lane_aux: pld.DistributedTensor[[LANE_ROWS, AUX_PAD], pl.FP32],
    lane_count: pld.DistributedTensor[[EP_SIZE, COUNT_PAD], pl.INT32],
    lane_ready: pld.DistributedTensor[[EP_SIZE, SIGNAL_PAD], pl.INT32],
    slot_out: pld.DistributedTensor[[T * TOPK, D], pl.FP32],
    return_ready: pld.DistributedTensor[[EP_SIZE, SIGNAL_PAD], pl.INT32],
    logits: pl.InOut[pl.Tensor[[T, N_EXPERTS], pl.FP32]],
    route_ids: pl.InOut[pl.Tensor[[T, TOPK], pl.INT32]],
    route_weights: pl.InOut[pl.Tensor[[T, TOPK], pl.FP32]],
    moe_output: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    output: pl.Out[pl.Tensor[[T, HC, HIDDEN], pl.BF16]],
    active: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """Run one rank's attention, FFN residual, and EP2 MoE composition."""

    ATTENTION(
        x_hc,
        attn_hc_function,
        attn_hc_scale,
        attn_hc_base,
        attn_incoming,
        attn_norm_weight,
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
        attn_output,
        attn_next_pre,
        attn_post,
        attn_comb,
        attn_collapsed,
        attn_normed,
        after_attention,
        active,
    )

    ffn_collapsed_fp32 = pl.create_tensor([T, HIDDEN], dtype=pl.FP32)
    FFN_PRE(
        after_attention,
        ffn_hc_function,
        ffn_hc_scale,
        ffn_hc_base,
        attn_next_pre,
        ffn_next_pre,
        ffn_post,
        ffn_comb,
        ffn_collapsed, ffn_collapsed_fp32,
        active,
    )
    ffn_router_normed = pl.create_tensor([T, HIDDEN], dtype=pl.FP32)
    for block in pl.spmd((active + NORM_T_TILE - 1) // NORM_T_TILE, name_hint="dsv41_decode_ffn_norm"):
        # The fused reference normalizes the BF16 HC collapse and exposes its
        # FP32 result to the router.  Keep the BF16 store boundary before both
        # outputs so expert input and router input share the same source.
        rms_norm_tiled_with_fp32(
            ffn_collapsed, ffn_norm_weight, ffn_normed, ffn_router_normed, block, active
        )

    EP2_RANK(
        ffn_normed,
        ffn_router_normed,
        gate_weight,
        text_bias,
        vision_bias,
        input_ids,
        routed_w1,
        routed_w1_scale,
        routed_w3,
        routed_w3_scale,
        routed_w2,
        routed_w2_scale,
        shared_gate_up,
        shared_gate_up_scale,
        shared_down,
        shared_down_scale,
        lane_x,
        lane_aux,
        lane_count,
        lane_ready,
        slot_out,
        return_ready,
        logits,
        route_ids,
        route_weights,
        moe_output,
        active,
        rank,
        epoch,
    )
    FFN_POST(moe_output, after_attention, ffn_post, ffn_comb, output, active)
    return output


@pl.jit.host
def decode_layer_ep2(
    x_hc: pl.Tensor[[EP_SIZE, T, HC, HIDDEN], pl.BF16],
    attn_hc_function: pl.Tensor[[EP_SIZE, CHANNELS, HC * HIDDEN], pl.FP32],
    attn_hc_scale: pl.Tensor[[EP_SIZE, 3], pl.FP32],
    attn_hc_base: pl.Tensor[[EP_SIZE, CHANNELS], pl.FP32],
    attn_incoming: pl.Tensor[[EP_SIZE, T, HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[EP_SIZE, HIDDEN], pl.BF16],
    wq_a: pl.Tensor[[EP_SIZE, HIDDEN, LATENT], pl.INT8],
    wq_a_scale: pl.Tensor[[EP_SIZE, LATENT], pl.FP32],
    q_norm_weight: pl.Tensor[[EP_SIZE, LATENT], pl.BF16],
    wq_b: pl.Tensor[[EP_SIZE, LATENT, WIDTH], pl.INT8],
    wq_b_scale: pl.Tensor[[EP_SIZE, WIDTH], pl.FP32],
    wkv: pl.Tensor[[EP_SIZE, HIDDEN, FLASH.head_dim], pl.INT8],
    wkv_scale: pl.Tensor[[EP_SIZE, FLASH.head_dim], pl.FP32],
    kv_norm_weight: pl.Tensor[[EP_SIZE, FLASH.head_dim], pl.BF16],
    wo_a: pl.Tensor[[EP_SIZE, FLASH.o_groups, GROUP_INPUT, O_RANK], pl.BF16],
    wo_b: pl.Tensor[[EP_SIZE, FLASH.o_groups * O_RANK, HIDDEN], pl.BF16],
    rope_cos: pl.Tensor[[EP_SIZE, T, ROPE_PAIRS], pl.FP32],
    rope_sin: pl.Tensor[[EP_SIZE, T, ROPE_PAIRS], pl.FP32],
    slots: pl.Tensor[[EP_SIZE, T, 2], pl.INT64],
    indices: pl.Tensor[[EP_SIZE, T, WINDOW], pl.INT32],
    sink: pl.Tensor[[EP_SIZE, FLASH.num_attention_heads], pl.FP32],
    positions: pl.InOut[pl.Tensor[[EP_SIZE, T], pl.INT64]],
    query_latent: pl.InOut[pl.Tensor[[EP_SIZE, T, LATENT], pl.BF16]],
    query: pl.InOut[pl.Tensor[[EP_SIZE, T, WIDTH], pl.BF16]],
    window_kv: pl.InOut[pl.Tensor[[EP_SIZE, T, FLASH.head_dim], pl.BF16]],
    cache: pl.InOut[pl.Tensor[[EP_SIZE, 2, 128, 1, FLASH.head_dim], pl.BF16]],
    cache_lens: pl.InOut[pl.Tensor[[EP_SIZE, 2], pl.INT32]],
    attn_output: pl.InOut[pl.Tensor[[EP_SIZE, T, HIDDEN], pl.BF16]],
    attn_next_pre: pl.InOut[pl.Tensor[[EP_SIZE, T, HC], pl.FP32]],
    attn_post: pl.InOut[pl.Tensor[[EP_SIZE, T, HC], pl.FP32]],
    attn_comb: pl.InOut[pl.Tensor[[EP_SIZE, T, HC, HC], pl.FP32]],
    attn_collapsed: pl.InOut[pl.Tensor[[EP_SIZE, T, HIDDEN], pl.BF16]],
    attn_normed: pl.InOut[pl.Tensor[[EP_SIZE, T, HIDDEN], pl.BF16]],
    after_attention: pl.InOut[pl.Tensor[[EP_SIZE, T, HC, HIDDEN], pl.BF16]],
    ffn_hc_function: pl.Tensor[[EP_SIZE, CHANNELS, HC * HIDDEN], pl.FP32],
    ffn_hc_scale: pl.Tensor[[EP_SIZE, 3], pl.FP32],
    ffn_hc_base: pl.Tensor[[EP_SIZE, CHANNELS], pl.FP32],
    ffn_norm_weight: pl.Tensor[[EP_SIZE, HIDDEN], pl.BF16],
    ffn_next_pre: pl.InOut[pl.Tensor[[EP_SIZE, T, HC], pl.FP32]],
    ffn_post: pl.InOut[pl.Tensor[[EP_SIZE, T, HC], pl.FP32]],
    ffn_comb: pl.InOut[pl.Tensor[[EP_SIZE, T, HC, HC], pl.FP32]],
    ffn_collapsed: pl.InOut[pl.Tensor[[EP_SIZE, T, HIDDEN], pl.BF16]],
    ffn_normed: pl.InOut[pl.Tensor[[EP_SIZE, T, HIDDEN], pl.BF16]],
    gate_weight: pl.Tensor[[EP_SIZE, N_EXPERTS, D], pl.FP32],
    text_bias: pl.Tensor[[EP_SIZE, N_EXPERTS], pl.FP32],
    vision_bias: pl.Tensor[[EP_SIZE, N_EXPERTS], pl.FP32],
    input_ids: pl.Tensor[[EP_SIZE, T], pl.INT64],
    routed_w1: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, D, 2560], pl.INT8],
    routed_w2_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, D], pl.FP32],
    shared_gate_up: pl.Tensor[[EP_SIZE, D, MOE_INTER * 2], pl.INT8],
    shared_gate_up_scale: pl.Tensor[[EP_SIZE, MOE_INTER * 2], pl.FP32],
    shared_down: pl.Tensor[[EP_SIZE, MOE_INTER, D], pl.INT8],
    shared_down_scale: pl.Tensor[[EP_SIZE, D], pl.FP32],
    logits: pl.Out[pl.Tensor[[EP_SIZE, T, N_EXPERTS], pl.FP32]],
    route_ids: pl.Out[pl.Tensor[[EP_SIZE, T, TOPK], pl.INT32]],
    route_weights: pl.Out[pl.Tensor[[EP_SIZE, T, TOPK], pl.FP32]],
    moe_output: pl.Out[pl.Tensor[[EP_SIZE, T, D], pl.BF16]],
    output: pl.Out[pl.Tensor[[EP_SIZE, T, HC, HIDDEN], pl.BF16]],
    active: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """Launch the complete L0 Decode layer across the two EP ranks."""

    lane_x_buf = pld.alloc_window_buffer([LANE_ROWS, D], dtype=pl.INT8)
    lane_aux_buf = pld.alloc_window_buffer([LANE_ROWS, AUX_PAD], dtype=pl.FP32)
    lane_count_buf = pld.alloc_window_buffer([EP_SIZE, COUNT_PAD], dtype=pl.INT32)
    lane_ready_buf = pld.alloc_window_buffer([EP_SIZE, SIGNAL_PAD], dtype=pl.INT32)
    slot_out_buf = pld.alloc_window_buffer([T * TOPK, D], dtype=pl.FP32)
    return_ready_buf = pld.alloc_window_buffer([EP_SIZE, SIGNAL_PAD], dtype=pl.INT32)
    for rank in pl.range(EP_SIZE):
        lane_x = pld.window(lane_x_buf, [LANE_ROWS, D], dtype=pl.INT8)
        lane_aux = pld.window(lane_aux_buf, [LANE_ROWS, AUX_PAD], dtype=pl.FP32)
        lane_count = pld.window(lane_count_buf, [EP_SIZE, COUNT_PAD], dtype=pl.INT32)
        lane_ready = pld.window(lane_ready_buf, [EP_SIZE, SIGNAL_PAD], dtype=pl.INT32)
        slot_out = pld.window(slot_out_buf, [T * TOPK, D], dtype=pl.FP32)
        return_ready = pld.window(return_ready_buf, [EP_SIZE, SIGNAL_PAD], dtype=pl.INT32)
        decode_layer_ep2_rank(
            x_hc[rank],
            attn_hc_function[rank],
            attn_hc_scale[rank],
            attn_hc_base[rank],
            attn_incoming[rank],
            attn_norm_weight[rank],
            wq_a[rank],
            wq_a_scale[rank],
            q_norm_weight[rank],
            wq_b[rank],
            wq_b_scale[rank],
            wkv[rank],
            wkv_scale[rank],
            kv_norm_weight[rank],
            wo_a[rank],
            wo_b[rank],
            rope_cos[rank],
            rope_sin[rank],
            slots[rank],
            indices[rank],
            sink[rank],
            positions[rank],
            query_latent[rank],
            query[rank],
            window_kv[rank],
            cache[rank],
            cache_lens[rank],
            attn_output[rank],
            attn_next_pre[rank],
            attn_post[rank],
            attn_comb[rank],
            attn_collapsed[rank],
            attn_normed[rank],
            after_attention[rank],
            ffn_hc_function[rank],
            ffn_hc_scale[rank],
            ffn_hc_base[rank],
            ffn_norm_weight[rank],
            ffn_next_pre[rank],
            ffn_post[rank],
            ffn_comb[rank],
            ffn_collapsed[rank],
            ffn_normed[rank],
            gate_weight[rank],
            text_bias[rank],
            vision_bias[rank],
            input_ids[rank],
            routed_w1[rank],
            routed_w1_scale[rank],
            routed_w3[rank],
            routed_w3_scale[rank],
            routed_w2[rank],
            routed_w2_scale[rank],
            shared_gate_up[rank],
            shared_gate_up_scale[rank],
            shared_down[rank],
            shared_down_scale[rank],
            lane_x,
            lane_aux,
            lane_count,
            lane_ready,
            slot_out,
            return_ready,
            logits[rank],
            route_ids[rank],
            route_weights[rank],
            moe_output[rank],
            output[rank],
            active,
            rank,
            epoch,
            device=rank,
        )


def product_contract() -> dict[str, object]:
    """Return the complete L0 EP2 composition boundary."""

    return {
        "status": "implementation_ready_not_run",
        "layer_id": 0,
        "topology": {"dp": 2, "tp": 1, "ep": 2, "pp": 1, "pcp": 1, "dcp": 1},
        "stages": [
            "mHC attention pre",
            "RMSNorm/QKV/RoPE/SWA cache read-write",
            "attention mHC residual expansion",
            "mHC FFN pre and RMSNorm",
            "PyPTO routed/shared EP2 dispatch-compute-combine",
            "mHC FFN output restoration",
        ],
        "product_compute": "PyPTO only",
        "reference_process": "independent vLLM-Ascend process",
        "formal_accuracy": "not_run",
    }


__all__ = ["decode_layer_ep2", "decode_layer_ep2_rank", "product_contract"]

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared V4.1 Q/KV projection, normalization, and RoPE stages."""


import pypto.language as pl
import torch

from .activation_quant import (
    _quantize_prefill_rows,
    _tile_rsqrt_for_norm,
    normalize_quantize_prefill_projection,
    normalize_quantize_projection_activation,
)
from .config import FLASH as M, T_DYN, TP_SIZE
from .gemm import (
    gemm_int8,
    gemm_int8_pair_columns,
    gemm_int8_tiled,
    golden_dynamic_int8,
    prefill_gemm_int8_tiled,
)
from .rmsnorm import golden_rms_norm, rms_norm
from .rope import make_pipelined_rope, make_prefill_kv_norm_rope


# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
EPS = M.rms_norm_eps

# tiling
Q_A_N_TILE = 128
Q_B_N_TILE = 256
KV_N_TILE = 64
PREFILL_QKV_MIN_ROWS = 512
PREFILL_Q_B_MIN_ROWS = 1024
PREFILL_Q_A_M_TILE = 128
PREFILL_Q_A_K_TILE = 512
PREFILL_Q_B_KV_M_TILE = 64
PREFILL_Q_B_KV_N_TILE = 256
PREFILL_Q_B_K_TILE = 128
PREFILL_Q_B_KV_K_TILE = 256
ROPE_WORKER_TILE = 32

rotate_prefill_q = make_pipelined_rope(
    T_DYN, LOCAL_H, HEAD_DIM, inverse=False, name_hint="attention_q_rope", allow_early_resolve=False, inplace=True,
)
normalize_rotate_prefill_kv = make_prefill_kv_norm_rope(T_DYN, HEAD_DIM, EPS)


@pl.jit.inline(auto_scope=False)
def normalize_rope_kv(
    source: pl.Tensor,
    norm_weight: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    """Apply BF16 RMSNorm and rotate the KV tail for Decode rows."""
    blocks = (num_tokens + 7) // 8
    workers = pl.min(blocks, pl.system.available_cluster_count())
    weight_row = pl.tensor.view(norm_weight, [1, HEAD_DIM])
    for worker in pl.spmd(workers, name_hint="norm_rope_kv"):
        for block in pl.range(worker, blocks, workers):
            row = block * 8
            rows = pl.max(0, pl.min(8, num_tokens - row))
            raw = pl.load(source, [row, 0], [8, HEAD_DIM], valid_shape=[rows, HEAD_DIM])
            padded = pl.set_validshape(pl.fillpad(raw, pad_value=pl.PadValue.zero), 8, HEAD_DIM)
            value = pl.cast(padded, pl.FP32)
            square = pl.mul(value, value)
            norm_scratch = pl.create_tile([8, HEAD_DIM], dtype=pl.FP32)
            row_sum = pl.row_sum(square, tmp_tile=norm_scratch)
            mean = pl.mul(row_sum, 1.0 / HEAD_DIM)
            rms_argument = pl.add(mean, EPS)
            rsqrt_scratch = pl.create_tile([8, 1], dtype=pl.FP32)
            row_inverse = _tile_rsqrt_for_norm(rms_argument, rsqrt_scratch)
            gamma = pl.cast(pl.load(weight_row, [0, 0], [1, HEAD_DIM]), pl.FP32)
            normalized = pl.col_expand_mul(pl.row_expand_mul(value, row_inverse), gamma)
            normalized_bf16 = pl.cast(normalized, pl.BF16, mode="rint")
            prefix = pl.tile.slice(
                normalized_bf16,
                [8, HEAD_DIM - ROPE_DIM],
                [0, 0],
                valid_shape=[rows, HEAD_DIM - ROPE_DIM],
            )
            pl.store(prefix, [row, 0], output)
            tail_bf16 = pl.tile.slice(normalized_bf16, [8, ROPE_DIM], [0, HEAD_DIM - ROPE_DIM])
            tail = pl.cast(tail_bf16, pl.FP32)
            even = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            cosine = pl.load(rope_cos, [row, 0], [8, ROPE_DIM // 2], valid_shape=[rows, ROPE_DIM // 2])
            sine = pl.load(rope_sin, [row, 0], [8, ROPE_DIM // 2], valid_shape=[rows, ROPE_DIM // 2])
            cosine = pl.set_validshape(pl.fillpad(cosine, pad_value=pl.PadValue.zero), 8, ROPE_DIM // 2)
            sine = pl.set_validshape(pl.fillpad(sine, pad_value=pl.PadValue.zero), 8, ROPE_DIM // 2)
            rope_even_cos = pl.mul(even, cosine)
            rope_odd_sin = pl.mul(odd, sine)
            real = pl.sub(rope_even_cos, rope_odd_sin)
            rope_even_sin = pl.mul(even, sine)
            rope_odd_cos = pl.mul(odd, cosine)
            imag = pl.add(rope_even_sin, rope_odd_cos)
            real_lanes = pl.tile.full([8, ROPE_DIM], dtype=pl.FP32, value=0.0)
            real_lanes = pl.tile.scatter_mask(real_lanes, real, mask_pattern=pl.tile.MaskPattern.P0101)
            imag_lanes = pl.tile.full([8, ROPE_DIM], dtype=pl.FP32, value=0.0)
            imag_lanes = pl.tile.scatter_mask(imag_lanes, imag, mask_pattern=pl.tile.MaskPattern.P1010)
            real_bits = pl.reinterpret_view(real_lanes, pl.INT16)
            imag_bits = pl.reinterpret_view(imag_lanes, pl.INT16)
            rotated_bits = pl.or_(real_bits, imag_bits)
            rotated = pl.reinterpret_view(rotated_bits, pl.FP32)
            rounded = pl.cast(rotated, pl.BF16, mode="rint")
            pl.store(pl.set_validshape(rounded, rows, ROPE_DIM), [row, HEAD_DIM - ROPE_DIM], output)
    return output


@pl.jit.inline(auto_scope=False)
def _rotate_rope(
    query: pl.Tensor,
    output: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    row_start: pl.Scalar[pl.INDEX],
    num_tokens: pl.Scalar[pl.INT32],
    heads: pl.constexpr,
    head_dim: pl.constexpr,
    name_hint: pl.constexpr,
    inverse: pl.constexpr,
    copy_prefix: pl.constexpr,
):
    """Rotate adjacent-pair tails and copy the leading channels for out-of-place output."""
    rope_blocks = (num_tokens + 15) // 16 * heads
    rope_workers = pl.max(1, pl.min(pl.min(rope_blocks, ROPE_WORKER_TILE), 2 * pl.system.available_cluster_count()))
    for worker in pl.spmd(rope_workers, name_hint=name_hint):
        for block in pl.range(worker, rope_blocks, rope_workers):
            t0 = block // heads * 16
            output_t0 = row_start + t0
            n0 = block % heads * head_dim
            rows = pl.max(pl.min(16, num_tokens - t0), 0)
            if copy_prefix:
                prefix = pl.load(query, [output_t0, n0], [16, head_dim - ROPE_DIM], valid_shape=[rows, head_dim - ROPE_DIM])
                output = pl.store(prefix, [output_t0, n0], output)
            tail_bf16 = pl.load(query, [output_t0, n0 + head_dim - ROPE_DIM], [16, ROPE_DIM], valid_shape=[rows, ROPE_DIM])
            tail = pl.cast(tail_bf16, pl.FP32)
            even = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            cosine = pl.load(rope_cos, [output_t0, 0], [16, ROPE_DIM // 2], valid_shape=[rows, ROPE_DIM // 2])
            sine = pl.load(rope_sin, [output_t0, 0], [16, ROPE_DIM // 2], valid_shape=[rows, ROPE_DIM // 2])
            cosine = pl.fillpad(cosine, pad_value=pl.PadValue.zero)
            cosine = pl.set_validshape(cosine, 16, ROPE_DIM // 2)
            sine = pl.fillpad(sine, pad_value=pl.PadValue.zero)
            sine = pl.set_validshape(sine, 16, ROPE_DIM // 2)
            if inverse:
                sine = pl.mul(sine, -1.0)
            rope_even_cos = pl.mul(even, cosine)
            rope_odd_sin = pl.mul(odd, sine)
            re = pl.sub(rope_even_cos, rope_odd_sin)
            rope_even_sin = pl.mul(even, sine)
            rope_odd_cos = pl.mul(odd, cosine)
            im = pl.add(rope_even_sin, rope_odd_cos)
            real_lanes = pl.tile.full([16, ROPE_DIM], dtype=pl.FP32, value=0.0)
            real_lanes = pl.tile.scatter_mask(real_lanes, re, mask_pattern=pl.tile.MaskPattern.P0101)
            imag_lanes = pl.tile.full([16, ROPE_DIM], dtype=pl.FP32, value=0.0)
            imag_lanes = pl.tile.scatter_mask(imag_lanes, im, mask_pattern=pl.tile.MaskPattern.P1010)
            real_bits = pl.reinterpret_view(real_lanes, pl.INT16)
            imag_bits = pl.reinterpret_view(imag_lanes, pl.INT16)
            rotated_bits = pl.or_(real_bits, imag_bits)
            rotated = pl.reinterpret_view(rotated_bits, pl.FP32)
            rounded = pl.cast(rotated, pl.BF16, mode="rint")
            valid = pl.set_validshape(rounded, rows, ROPE_DIM)
            output = pl.store(valid, [output_t0, n0 + head_dim - ROPE_DIM], output)
    return output


def make_q_proj_rope(*, prefill):
    if prefill:
        @pl.jit.inline(auto_scope=False)
        def q_proj_rope(
            payload: pl.Tensor,
            scales: pl.Tensor,
            wq_a: pl.Tensor,
            wq_a_scale: pl.Tensor,
            q_norm_weight: pl.Tensor,
            wq_b: pl.Tensor,
            wq_b_scale: pl.Tensor,
            rope_cos: pl.Tensor,
            rope_sin: pl.Tensor,
            query_latent: pl.Tensor,
            query: pl.Tensor,
            num_tokens: pl.Scalar[pl.INT32],
        ):
            tokens = pl.tensor.dim(query_latent, 0)
            projected = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
            if num_tokens >= PREFILL_QKV_MIN_ROWS:
                prefill_gemm_int8_tiled(
                    payload, scales, wq_a, wq_a_scale, projected, num_tokens, D, Q_LORA, "proj_q_a",
                    Q_A_N_TILE, PREFILL_Q_A_K_TILE, False, PREFILL_Q_A_M_TILE,
                )
            else:
                gemm_int8(
                    payload, scales, wq_a, wq_a_scale, projected, num_tokens,
                    D, Q_LORA, "proj_q_a", Q_A_N_TILE, 256, False,
                )
            padded_rows = (tokens + 15) // 16 * 16
            qb_payload = pl.create_tensor([padded_rows, Q_LORA], dtype=pl.INT8)
            qb_scales = pl.create_tensor([padded_rows, 1], dtype=pl.FP32)
            if num_tokens >= 64:
                normalize_quantize_prefill_projection(
                    projected, q_norm_weight, query_latent, qb_payload, qb_scales,
                    num_tokens, padded_rows, Q_LORA, EPS,
                )
            else:
                for worker in pl.spmd((num_tokens + 7) // 8, name_hint="norm_q_latent"):
                    rms_norm(projected, q_norm_weight, query_latent, worker, num_tokens, width=Q_LORA)
                _quantize_prefill_rows(
                    query_latent, qb_payload, qb_scales, num_tokens, padded_rows, Q_LORA, Q_LORA,
                )
            if num_tokens >= PREFILL_Q_B_MIN_ROWS:
                prefill_gemm_int8_tiled(
                    qb_payload, qb_scales, wq_b, wq_b_scale, query, num_tokens,
                    Q_LORA, LOCAL_H * HEAD_DIM, "proj_q_b",
                    PREFILL_Q_B_KV_N_TILE, PREFILL_Q_B_K_TILE, False, PREFILL_Q_B_KV_M_TILE,
                )
            elif num_tokens >= PREFILL_QKV_MIN_ROWS:
                prefill_gemm_int8_tiled(
                    qb_payload, qb_scales, wq_b, wq_b_scale, query, num_tokens,
                    Q_LORA, LOCAL_H * HEAD_DIM, "proj_q_b",
                    PREFILL_Q_B_KV_N_TILE, PREFILL_Q_B_KV_K_TILE, False, PREFILL_Q_B_KV_M_TILE,
                )
            else:
                gemm_int8(
                    qb_payload, qb_scales, wq_b, wq_b_scale, query, num_tokens,
                    Q_LORA, LOCAL_H * HEAD_DIM, "proj_q_b", Q_B_N_TILE, 256, False,
                )
            if num_tokens >= 64:
                query = rotate_prefill_q(query, rope_cos, rope_sin, query, num_tokens)
            else:
                query = _rotate_rope(
                    query, query, rope_cos, rope_sin, 0, num_tokens,
                    LOCAL_H, HEAD_DIM, "attention_q_rope", False, False,
                )
            return query_latent, query, qb_payload, qb_scales
    else:
        @pl.jit.inline(auto_scope=False)
        def q_proj_rope(
            payload: pl.Tensor,
            scales: pl.Tensor,
            wq_a: pl.Tensor,
            wq_a_scale: pl.Tensor,
            q_norm_weight: pl.Tensor,
            wq_b: pl.Tensor,
            wq_b_scale: pl.Tensor,
            rope_cos: pl.Tensor,
            rope_sin: pl.Tensor,
            query_latent: pl.Tensor,
            query: pl.Tensor,
            num_tokens: pl.Scalar[pl.INT32],
        ):
            tokens = pl.tensor.dim(query_latent, 0)
            projected = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
            gemm_int8_pair_columns(
                payload, scales, wq_a, wq_a_scale, projected, num_tokens,
                D, Q_LORA, "proj_q_a_decode_pair_k512", 128, 512, False, 16,
                rope_cos, rope_sin, False, 0, 0,
            )
            payload_rows = pl.tensor.dim(projected, 0)
            padded_rows = (payload_rows + 15) // 16 * 16
            qb_payload = pl.create_tensor([padded_rows, Q_LORA], dtype=pl.INT8)
            qb_scales = pl.create_tensor([padded_rows, 1], dtype=pl.FP32)
            normalize_quantize_projection_activation(
                projected, q_norm_weight, query_latent, qb_payload, qb_scales,
                num_tokens, Q_LORA, EPS,
            )
            gemm_int8_pair_columns(
                qb_payload, qb_scales, wq_b, wq_b_scale, query, num_tokens,
                Q_LORA, LOCAL_H * HEAD_DIM, "proj_q_b_decode_pair", Q_B_N_TILE,
                256, False, 16, rope_cos, rope_sin, True,
                ROPE_DIM, HEAD_DIM - ROPE_DIM - Q_B_N_TILE,
            )
            return query_latent, query, qb_payload, qb_scales

    return q_proj_rope


q_proj_rope = make_q_proj_rope(prefill=True)
decode_q_proj_rope = make_q_proj_rope(prefill=False)


def make_kv_proj_rope(*, prefill):
    if prefill:
        @pl.jit.inline(auto_scope=False)
        def kv_proj_rope(
            payload: pl.Tensor,
            scales: pl.Tensor,
            wkv: pl.Tensor,
            wkv_scale: pl.Tensor,
            kv_norm_weight: pl.Tensor,
            rope_cos: pl.Tensor,
            rope_sin: pl.Tensor,
            kv_projection: pl.Tensor,
            window_kv: pl.Tensor,
            num_tokens: pl.Scalar[pl.INT32],
        ):
            if num_tokens >= PREFILL_QKV_MIN_ROWS:
                prefill_gemm_int8_tiled(
                    payload, scales, wkv, wkv_scale, kv_projection, num_tokens, D, HEAD_DIM, "proj_kv",
                    PREFILL_Q_B_KV_N_TILE, PREFILL_Q_B_KV_K_TILE, False, PREFILL_Q_B_KV_M_TILE,
                )
            else:
                gemm_int8(
                    payload, scales, wkv, wkv_scale, kv_projection, num_tokens,
                    D, HEAD_DIM, "proj_kv", KV_N_TILE, 256, False,
                )
            if num_tokens >= 64:
                window_kv = normalize_rotate_prefill_kv(
                    kv_projection, kv_norm_weight,
                    rope_cos, rope_sin, window_kv, num_tokens,
                )
            else:
                for worker in pl.spmd((num_tokens + 7) // 8, name_hint="norm_kv"):
                    rms_norm(kv_projection, kv_norm_weight, window_kv, worker, num_tokens, width=HEAD_DIM)
                window_kv = _rotate_rope(
                    window_kv, window_kv, rope_cos, rope_sin, 0, num_tokens,
                    1, HEAD_DIM, "rope_kv", False, False,
                )
            return kv_projection, window_kv
    else:
        @pl.jit.inline(auto_scope=False)
        def kv_proj_rope(
            payload: pl.Tensor,
            scales: pl.Tensor,
            wkv: pl.Tensor,
            wkv_scale: pl.Tensor,
            kv_norm_weight: pl.Tensor,
            rope_cos: pl.Tensor,
            rope_sin: pl.Tensor,
            kv_projection: pl.Tensor,
            window_kv: pl.Tensor,
            num_tokens: pl.Scalar[pl.INT32],
        ):
            gemm_int8_tiled(
                payload, scales, wkv, wkv_scale, kv_projection, num_tokens,
                D, HEAD_DIM, "proj_kv_decode_k512_n128", 128, 512, False, 16,
            )
            normalize_rope_kv(
                kv_projection, kv_norm_weight, rope_cos, rope_sin, window_kv, num_tokens,
            )
            return kv_projection, window_kv

    return kv_proj_rope


kv_proj_rope = make_kv_proj_rope(prefill=True)
decode_kv_proj_rope = make_kv_proj_rope(prefill=False)


def golden_rope_interleave(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply adjacent-pair rotary embedding to the final dimension."""
    value = x.float().unflatten(-1, (-1, 2))
    real, imag = value.unbind(dim=-1)
    if inverse:
        sin = -sin
    while cos.ndim < real.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    rotated = torch.stack((real * cos - imag * sin, imag * cos + real * sin), dim=-1)
    return rotated.flatten(-2).to(x.dtype)


def golden_rope_tail(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Rotate the trailing 2*cos.shape[-1] lanes as adjacent (real, imag) pairs."""
    rope_dim = cos.shape[-1] * 2
    rotated = golden_rope_interleave(x[..., -rope_dim:], cos, sin, inverse)
    return torch.cat((x[..., :-rope_dim], rotated), dim=-1)


def golden_qkv_proj_rope(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor | None,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor | None,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor | None,
    kv_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    sequential_query: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project normalized query latent and shared KV, then rotate their RoPE tails."""
    from .quant import unpack_fp8_nz

    if wq_a.dtype == torch.float8_e4m3fn:
        wq_a = unpack_fp8_nz(wq_a)
    if wq_b.dtype == torch.float8_e4m3fn:
        wq_b = unpack_fp8_nz(wq_b)
    if wkv.dtype == torch.float8_e4m3fn:
        wkv = unpack_fp8_nz(wkv)

    def project(value, weight, scale):
        if sequential_query and scale is not None:
            return golden_dynamic_int8(value, weight, scale)
        return golden_dynamic_int8(value, weight, scale)

    if wq_a_scale is not None:
        projected = golden_dynamic_int8(x, wq_a, wq_a_scale)
    else:
        projected = golden_dynamic_int8(x, wq_a, wq_a_scale)
    qr = golden_rms_norm(projected, q_norm_weight)
    head_dim = wkv.shape[-1]
    num_heads = wq_b.shape[-1] // head_dim
    q = project(qr, wq_b, wq_b_scale).unflatten(-1, (num_heads, head_dim))
    kv = golden_rms_norm(project(x, wkv, wkv_scale), kv_norm_weight)
    q = golden_rope_tail(q, cos, sin)
    kv = golden_rope_tail(kv, cos, sin)
    return q, kv, qr

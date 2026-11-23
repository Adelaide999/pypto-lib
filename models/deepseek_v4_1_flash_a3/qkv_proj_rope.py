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

from .activation_quant import quantize_projection_activation
from .blocked_rmsnorm import make_blocked_rmsnorm
from .config import FLASH as M, T_DYN, TP_SIZE
from .gemm import gemm_int8, golden_dynamic_int8
from .rmsnorm import golden_rms_norm, rms_norm
from .rope import make_pipelined_rope, make_pipelined_token_rope


# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
EPS = M.rms_norm_eps

# tiling
Q_A_N_TILE = 64
Q_B_N_TILE = 256
KV_N_TILE = 64
ROPE_WORKER_TILE = 32

rotate_prefill_q = make_pipelined_rope(
    T_DYN, LOCAL_H, HEAD_DIM, inverse=False, name_hint="attention_q_rope", allow_early_resolve=False, inplace=True,
)
rotate_prefill_kv = make_pipelined_token_rope(T_DYN, HEAD_DIM, name_hint="rope_kv", inplace=True)

normalize_q = make_blocked_rmsnorm(T_DYN, Q_LORA, EPS, dynamic_rows=True)
normalize_kv = make_blocked_rmsnorm(T_DYN, HEAD_DIM, EPS, dynamic_rows=True)


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
        """Project INT8 Q-A/Q-B weights, normalize latent rows, and rotate query heads."""
        tokens = pl.tensor.dim(query_latent, 0)
        projected = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
        gemm_int8(payload, scales, wq_a, wq_a_scale, projected, num_tokens, D, Q_LORA, "proj_q_a", 128, 256, False)
        if prefill and num_tokens >= 64:
            normalize_q(projected, q_norm_weight, query_latent, num_tokens)
        else:
            for worker in pl.spmd((num_tokens + 7) // 8, name_hint="norm_q_latent"):
                rms_norm(projected, q_norm_weight, query_latent, worker, num_tokens, width=Q_LORA)
        qb_payload, qb_scales = quantize_projection_activation(query_latent, num_tokens, Q_LORA, prefill)
        gemm_int8(qb_payload, qb_scales, wq_b, wq_b_scale, query, num_tokens, Q_LORA, LOCAL_H * HEAD_DIM, "proj_q_b", Q_B_N_TILE, 256, False)
        if prefill and num_tokens >= 64:
            query = rotate_prefill_q(query, rope_cos, rope_sin, query, num_tokens)
        else:
            query = _rotate_rope(query, query, rope_cos, rope_sin, 0, num_tokens, LOCAL_H, HEAD_DIM, "attention_q_rope", False, False)
        return query_latent, query, qb_payload, qb_scales

    return q_proj_rope


q_proj_rope = make_q_proj_rope(prefill=True)
decode_q_proj_rope = make_q_proj_rope(prefill=False)


def make_kv_proj_rope(*, prefill):
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
        """Project INT8 KV weights, normalize latent rows, and rotate their tails."""
        gemm_int8(payload, scales, wkv, wkv_scale, kv_projection, num_tokens, D, HEAD_DIM, "proj_kv", KV_N_TILE, 256, False)
        if prefill and num_tokens >= 64:
            normalize_kv(kv_projection, kv_norm_weight, window_kv, num_tokens)
        else:
            for worker in pl.spmd((num_tokens + 7) // 8, name_hint="norm_kv"):
                rms_norm(kv_projection, kv_norm_weight, window_kv, worker, num_tokens, width=HEAD_DIM)
        if prefill and num_tokens >= 64:
            window_kv = rotate_prefill_kv(window_kv, rope_cos, rope_sin, window_kv, num_tokens)
        else:
            window_kv = _rotate_rope(
                window_kv, window_kv, rope_cos, rope_sin, 0, num_tokens,
                1, HEAD_DIM, "rope_kv", False, False,
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

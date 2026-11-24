# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile and benchmark the 1122 Decode Q projection path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pypto.language as pl
import torch

from golden import TensorSpec, ratio_allclose, run
from models.deepseek_v4_1_flash_a3.activation_quant import quantize_projection_activation
from models.deepseek_v4_1_flash_a3.config import FLASH as M
from models.deepseek_v4_1_flash_a3.gemm import gemm_int8, golden_dynamic_int8
from models.deepseek_v4_1_flash_a3.qkv_proj_rope import (
    _rotate_rope,
    decode_q_proj_rope,
    golden_rope_tail,
)
from models.deepseek_v4_1_flash_a3.rmsnorm import golden_rms_norm, rms_norm


CAPACITY = 64
TOKENS = 1
D = M.hidden_size
Q_LORA = M.q_lora_rank
LOCAL_H = M.num_attention_heads // 4
HEAD_DIM = M.head_dim
QUERY_WIDTH = LOCAL_H * HEAD_DIM
ROPE_DIM = M.qk_rope_head_dim


@pl.jit
def l2_decode_q_proj(
    payload: pl.Tensor[[CAPACITY, D], pl.INT8],
    scales: pl.Tensor[[CAPACITY, 1], pl.FP32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8],
    wq_a_scale: pl.Tensor[[1, Q_LORA], pl.FP32],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, QUERY_WIDTH], pl.INT8],
    wq_b_scale: pl.Tensor[[1, QUERY_WIDTH], pl.FP32],
    rope_cos: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    counts: pl.Tensor[[1], pl.INT32],
    query_latent: pl.Out[pl.Tensor[[CAPACITY, Q_LORA], pl.BF16]],
    query: pl.Out[pl.Tensor[[CAPACITY, QUERY_WIDTH], pl.BF16]],
):
    count = pl.read(counts, [0])
    return decode_q_proj_rope(
        payload, scales, wq_a, wq_a_scale, q_norm_weight,
        wq_b, wq_b_scale, rope_cos, rope_sin, query_latent, query, count,
    )


@pl.jit
def l2_decode_q_proj_baseline(
    payload: pl.Tensor[[CAPACITY, D], pl.INT8],
    scales: pl.Tensor[[CAPACITY, 1], pl.FP32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8],
    wq_a_scale: pl.Tensor[[1, Q_LORA], pl.FP32],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, QUERY_WIDTH], pl.INT8],
    wq_b_scale: pl.Tensor[[1, QUERY_WIDTH], pl.FP32],
    rope_cos: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    counts: pl.Tensor[[1], pl.INT32],
    query_latent: pl.Out[pl.Tensor[[CAPACITY, Q_LORA], pl.BF16]],
    query: pl.Out[pl.Tensor[[CAPACITY, QUERY_WIDTH], pl.BF16]],
):
    count = pl.read(counts, [0])
    projected = pl.create_tensor([CAPACITY, Q_LORA], dtype=pl.BF16)
    gemm_int8(
        payload, scales, wq_a, wq_a_scale, projected, count,
        D, Q_LORA, "baseline_proj_q_a", 128, 256, False,
    )
    for worker in pl.spmd((count + 7) // 8, name_hint="baseline_norm_q"):
        rms_norm(projected, q_norm_weight, query_latent, worker, count, width=Q_LORA)
    qb_payload, qb_scales = quantize_projection_activation(query_latent, count, Q_LORA, False)
    gemm_int8(
        qb_payload, qb_scales, wq_b, wq_b_scale, query, count,
        Q_LORA, QUERY_WIDTH, "baseline_proj_q_b", 256, 256, False,
    )
    query = _rotate_rope(
        query, query, rope_cos, rope_sin, 0, count,
        LOCAL_H, HEAD_DIM, "baseline_attention_q_rope", False, False,
    )
    return query_latent, query


def golden(values: dict[str, torch.Tensor]) -> None:
    source = values["payload"].float() * values["scales"]
    projected = golden_dynamic_int8(source, values["wq_a"], values["wq_a_scale"])
    latent = golden_rms_norm(projected, values["q_norm_weight"])
    query = golden_dynamic_int8(latent, values["wq_b"], values["wq_b_scale"])
    query = golden_rope_tail(query, values["rope_cos"], values["rope_sin"])
    values["query_latent"][:TOKENS].copy_(latent[:TOKENS])
    values["query"][:TOKENS].copy_(query[:TOKENS])


def make_values() -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(1122)
    return {
        "payload": torch.randint(-127, 128, (CAPACITY, D), dtype=torch.int8, generator=gen),
        "scales": torch.full((CAPACITY, 1), 0.02 / 127, dtype=torch.float32),
        "wq_a": torch.randint(-127, 128, (D, Q_LORA), dtype=torch.int8, generator=gen),
        "wq_a_scale": torch.full((1, Q_LORA), 0.02 / 127, dtype=torch.float32),
        "q_norm_weight": torch.ones(Q_LORA, dtype=torch.bfloat16),
        "wq_b": torch.randint(-127, 128, (Q_LORA, QUERY_WIDTH), dtype=torch.int8, generator=gen),
        "wq_b_scale": torch.full((1, QUERY_WIDTH), 0.02 / 127, dtype=torch.float32),
        "rope_cos": torch.ones((CAPACITY, ROPE_DIM // 2), dtype=torch.float32),
        "rope_sin": torch.zeros((CAPACITY, ROPE_DIM // 2), dtype=torch.float32),
        "counts": torch.tensor([TOKENS], dtype=torch.int32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--mode", choices=("baseline", "candidate"), default="candidate")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    values = make_values()
    specs = [
        TensorSpec(name, list(value.shape), value.dtype, init_value=value)
        for name, value in values.items()
    ]
    specs.extend([
        TensorSpec("query_latent", [CAPACITY, Q_LORA], torch.bfloat16),
        TensorSpec("query", [CAPACITY, QUERY_WIDTH], torch.bfloat16),
    ])
    result = run(
        fn=l2_decode_q_proj if args.mode == "candidate" else l2_decode_q_proj_baseline,
        specs=specs,
        golden_fn=golden,
        compare_fn={
            "query_latent": ratio_allclose(atol=0.003, rtol=0.03, max_error_ratio=0.005),
            "query": ratio_allclose(atol=0.003, rtol=0.03, max_error_ratio=0.005),
        },
        config={"backend_type": args.platform, "execution_mode": False, "device_id": args.device},
        compile_only=args.compile_only,
    )
    report = {
        "passed": result.passed,
        "mode": args.mode,
        "build": str(result.work_dir),
        "error": str(result.error) if result.error else None,
        "bench": str(result.bench) if result.bench else None,
        "compile_only": args.compile_only,
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()

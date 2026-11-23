"""Compile and benchmark the 1122 Decode KV projection path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pypto.language as pl
import torch

from golden import TensorSpec, ratio_allclose, run
from models.deepseek_v4_1_flash_a3.config import FLASH as M
from models.deepseek_v4_1_flash_a3.gemm import golden_dynamic_int8
from models.deepseek_v4_1_flash_a3.qkv_proj_rope import (
    _rotate_rope,
    decode_kv_proj_rope,
    golden_rope_tail,
)
from models.deepseek_v4_1_flash_a3.gemm import gemm_int8
from models.deepseek_v4_1_flash_a3.rmsnorm import golden_rms_norm, rms_norm


CAPACITY = 64
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim


def make_l2(fn):
    @pl.jit
    def l2(
        payload: pl.Tensor[[CAPACITY, D], pl.INT8],
        scales: pl.Tensor[[CAPACITY, 1], pl.FP32],
        wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8],
        wkv_scale: pl.Tensor[[1, HEAD_DIM], pl.FP32],
        kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
        rope_cos: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
        counts: pl.Tensor[[1], pl.INT32],
        kv_projection: pl.Out[pl.Tensor[[CAPACITY, HEAD_DIM], pl.BF16]],
        window_kv: pl.Out[pl.Tensor[[CAPACITY, HEAD_DIM], pl.BF16]],
    ):
        count = pl.read(counts, [0])
        return fn(
            payload, scales, wkv, wkv_scale, kv_norm_weight,
            rope_cos, rope_sin, kv_projection, window_kv, count,
        )

    return l2


l2_decode_kv_proj = make_l2(decode_kv_proj_rope)


@pl.jit.inline(auto_scope=False)
def decode_kv_proj_baseline(
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
    gemm_int8(
        payload, scales, wkv, wkv_scale, kv_projection, num_tokens,
        D, HEAD_DIM, "baseline_proj_kv", 64, 256, False,
    )
    for worker in pl.spmd((num_tokens + 7) // 8, name_hint="baseline_norm_kv"):
        rms_norm(kv_projection, kv_norm_weight, window_kv, worker, num_tokens, width=HEAD_DIM)
    window_kv = _rotate_rope(
        window_kv, window_kv, rope_cos, rope_sin, 0, num_tokens,
        1, HEAD_DIM, "baseline_rope_kv", False, False,
    )
    return kv_projection, window_kv


l2_decode_kv_proj_baseline = make_l2(decode_kv_proj_baseline)


def make_values(tokens: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(1122)
    return {
        "payload": torch.randint(-127, 128, (CAPACITY, D), dtype=torch.int8, generator=gen),
        "scales": torch.full((CAPACITY, 1), 0.02 / 127, dtype=torch.float32),
        "wkv": torch.randint(-127, 128, (D, HEAD_DIM), dtype=torch.int8, generator=gen),
        "wkv_scale": torch.full((1, HEAD_DIM), 0.02 / 127, dtype=torch.float32),
        "kv_norm_weight": torch.ones(HEAD_DIM, dtype=torch.bfloat16),
        "rope_cos": torch.ones((CAPACITY, ROPE_DIM // 2), dtype=torch.float32),
        "rope_sin": torch.zeros((CAPACITY, ROPE_DIM // 2), dtype=torch.float32),
        "counts": torch.tensor([tokens], dtype=torch.int32),
    }


def golden(values: dict[str, torch.Tensor]) -> None:
    source = values["payload"].float() * values["scales"]
    projection = golden_dynamic_int8(source, values["wkv"], values["wkv_scale"])
    window = golden_rms_norm(projection, values["kv_norm_weight"])
    window = golden_rope_tail(window, values["rope_cos"], values["rope_sin"])
    tokens = int(values["counts"].item())
    values["kv_projection"][:tokens].copy_(projection[:tokens])
    values["window_kv"][:tokens].copy_(window[:tokens])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--mode", choices=("baseline", "candidate"), default="candidate")
    parser.add_argument("--tokens", type=int, choices=(1, 16, CAPACITY), default=1)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    values = make_values(args.tokens)
    specs = [
        TensorSpec(name, list(value.shape), value.dtype, init_value=value)
        for name, value in values.items()
    ]
    specs.extend([
        TensorSpec("kv_projection", [CAPACITY, HEAD_DIM], torch.bfloat16),
        TensorSpec("window_kv", [CAPACITY, HEAD_DIM], torch.bfloat16),
    ])
    result = run(
        fn=l2_decode_kv_proj if args.mode == "candidate" else l2_decode_kv_proj_baseline,
        specs=specs,
        golden_fn=golden,
        compare_fn={
            "kv_projection": ratio_allclose(
                atol=0.003, rtol=0.03, max_error_ratio=0.005, valid_rows=args.tokens,
            ),
            "window_kv": ratio_allclose(
                atol=0.003, rtol=0.03, max_error_ratio=0.005, valid_rows=args.tokens,
            ),
        },
        config={"backend_type": args.platform, "execution_mode": False, "device_id": args.device},
        compile_only=args.compile_only,
    )
    report = {
        "passed": result.passed,
        "mode": args.mode,
        "tokens": args.tokens,
        "compile_only": args.compile_only,
        "build": str(result.work_dir),
        "error": str(result.error) if result.error else None,
        "bench": str(result.bench) if result.bench else None,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()

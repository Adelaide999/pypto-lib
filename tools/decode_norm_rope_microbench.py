"""Validate the Decode KV RMSNorm and RoPE fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pypto.language as pl
import torch

from golden import TensorSpec, ratio_allclose, run
from models.deepseek_v4_1_flash_a3.config import FLASH as M
from models.deepseek_v4_1_flash_a3.qkv_proj_rope import _rotate_rope, normalize_rope_kv
from models.deepseek_v4_1_flash_a3.qkv_proj_rope import golden_rope_tail
from models.deepseek_v4_1_flash_a3.rmsnorm import golden_rms_norm, rms_norm


CAPACITY = 64
WIDTH = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
TOKENS = 1


@pl.jit
def l2_decode_norm_rope(
    source: pl.Tensor[[CAPACITY, WIDTH], pl.BF16],
    norm_weight: pl.Tensor[[WIDTH], pl.BF16],
    rope_cos: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    counts: pl.Tensor[[1], pl.INT32],
    output: pl.Out[pl.Tensor[[CAPACITY, WIDTH], pl.BF16]],
):
    count = pl.read(counts, [0])
    normalize_rope_kv(source, norm_weight, rope_cos, rope_sin, output, count)
    return output


@pl.jit
def l2_decode_norm_rope_baseline(
    source: pl.Tensor[[CAPACITY, WIDTH], pl.BF16],
    norm_weight: pl.Tensor[[WIDTH], pl.BF16],
    rope_cos: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    counts: pl.Tensor[[1], pl.INT32],
    output: pl.Out[pl.Tensor[[CAPACITY, WIDTH], pl.BF16]],
):
    count = pl.read(counts, [0])
    with pl.spmd((count + 7) // 8, name_hint="baseline_norm_kv"):
        worker = pl.tile.get_block_idx()
        rms_norm(source, norm_weight, output, worker, count, width=WIDTH)
    _rotate_rope(output, output, rope_cos, rope_sin, 0, count, 1, WIDTH, "baseline_rope_kv", False, False)
    return output


def make_values(tokens: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1122)
    angles = torch.randn(CAPACITY, ROPE_DIM // 2, generator=generator)
    return {
        "source": torch.randn(CAPACITY, WIDTH, dtype=torch.bfloat16, generator=generator),
        "norm_weight": torch.randn(WIDTH, dtype=torch.bfloat16, generator=generator),
        "rope_cos": torch.cos(angles),
        "rope_sin": torch.sin(angles),
        "counts": torch.tensor([tokens], dtype=torch.int32),
    }


def golden(values: dict[str, torch.Tensor]) -> None:
    active = int(values["counts"].item())
    normalized = golden_rms_norm(values["source"][:active], values["norm_weight"])
    values["output"][:active].copy_(golden_rope_tail(
        normalized, values["rope_cos"][:active], values["rope_sin"][:active],
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "candidate"), default="candidate")
    parser.add_argument("--tokens", type=int, choices=(1, 16, CAPACITY), default=TOKENS)
    parser.add_argument("--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    values = make_values(args.tokens)
    specs = [
        TensorSpec(name, list(value.shape), value.dtype, init_value=value)
        for name, value in values.items()
    ]
    specs.append(TensorSpec("output", [CAPACITY, WIDTH], torch.bfloat16))
    result = run(
        fn=l2_decode_norm_rope if args.mode == "candidate" else l2_decode_norm_rope_baseline,
        specs=specs,
        golden_fn=golden,
        compare_fn={"output": ratio_allclose(valid_rows=args.tokens)},
        config={"backend_type": args.platform, "execution_mode": False, "device_id": args.device},
    )
    report = {
        "passed": result.passed,
        "mode": args.mode,
        "tokens": args.tokens,
        "build": str(result.work_dir),
        "error": str(result.error) if result.error else None,
        "bench": str(result.bench) if result.bench else None,
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()

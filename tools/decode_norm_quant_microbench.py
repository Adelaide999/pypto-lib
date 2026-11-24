# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate the Decode RMSNorm and INT8 activation quantization fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pypto.language as pl
import torch

from golden import TensorSpec, ratio_allclose, run
from models.deepseek_v4_1_flash_a3.activation_quant import normalize_quantize_projection_activation, quantize_projection_activation
from models.deepseek_v4_1_flash_a3.config import FLASH as M
from models.deepseek_v4_1_flash_a3.gemm import golden_activation_int8
from models.deepseek_v4_1_flash_a3.rmsnorm import golden_rms_norm, rms_norm


CAPACITY = 64
WIDTH = M.q_lora_rank
TOKENS = 1


@pl.jit
def l2_decode_norm_quant(
    source: pl.Tensor[[CAPACITY, WIDTH], pl.BF16],
    norm_weight: pl.Tensor[[WIDTH], pl.BF16],
    counts: pl.Tensor[[1], pl.INT32],
    normalized: pl.Out[pl.Tensor[[CAPACITY, WIDTH], pl.BF16]],
    payload: pl.Out[pl.Tensor[[CAPACITY, WIDTH], pl.INT8]],
    scale: pl.Out[pl.Tensor[[CAPACITY, 1], pl.FP32]],
):
    count = pl.read(counts, [0])
    normalize_quantize_projection_activation(
        source, norm_weight, normalized, payload, scale, count, WIDTH, M.rms_norm_eps,
    )
    return normalized, payload, scale


@pl.jit
def l2_decode_norm_quant_baseline(
    source: pl.Tensor[[CAPACITY, WIDTH], pl.BF16],
    norm_weight: pl.Tensor[[WIDTH], pl.BF16],
    counts: pl.Tensor[[1], pl.INT32],
    normalized: pl.Out[pl.Tensor[[CAPACITY, WIDTH], pl.BF16]],
    payload: pl.Out[pl.Tensor[[CAPACITY, WIDTH], pl.INT8]],
    scale: pl.Out[pl.Tensor[[CAPACITY, 1], pl.FP32]],
):
    count = pl.read(counts, [0])
    with pl.spmd((count + 7) // 8, name_hint="baseline_norm"):
        worker = pl.tile.get_block_idx()
        rms_norm(source, norm_weight, normalized, worker, count, width=WIDTH)
    payload_tmp, scale_tmp = quantize_projection_activation(normalized, count, WIDTH, False)
    with pl.spmd(CAPACITY // 16, name_hint="baseline_quant_copy"):
        worker = pl.tile.get_block_idx()
        row = worker * 16
        for k0 in pl.range(0, WIDTH, 256):
            value = pl.load(payload_tmp, [row, k0], [16, 256])
            pl.store(value, [row, k0], payload)
        factor = pl.load(scale_tmp, [row, 0], [16, 1])
        pl.store(factor, [row, 0], scale)
    return normalized, payload, scale


def make_values(tokens: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1122)
    return {
        "source": torch.randn(CAPACITY, WIDTH, dtype=torch.bfloat16, generator=generator),
        "norm_weight": torch.randn(WIDTH, dtype=torch.bfloat16, generator=generator),
        "counts": torch.tensor([tokens], dtype=torch.int32),
    }


def golden(values: dict[str, torch.Tensor]) -> None:
    active = int(values["counts"].item())
    normalized = golden_rms_norm(values["source"][:active], values["norm_weight"])
    payload = golden_activation_int8(normalized)
    maximum = normalized.float().abs().amax(-1, keepdim=True).clamp_min(1e-30)
    scale = maximum * (1.0 / 127.0)
    values["normalized"][:active].copy_(normalized)
    values["payload"][:active].copy_(payload)
    values["scale"][:active].copy_(scale)


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
    specs.extend([
        TensorSpec("normalized", [CAPACITY, WIDTH], torch.bfloat16),
        TensorSpec("payload", [CAPACITY, WIDTH], torch.int8),
        TensorSpec("scale", [CAPACITY, 1], torch.float32),
    ])
    result = run(
        fn=l2_decode_norm_quant if args.mode == "candidate" else l2_decode_norm_quant_baseline,
        specs=specs,
        golden_fn=golden,
        compare_fn={
            "normalized": ratio_allclose(valid_rows=args.tokens),
            "payload": ratio_allclose(rtol=0.0, atol=0.0, valid_rows=args.tokens),
            "scale": ratio_allclose(rtol=0.0, atol=0.0, valid_rows=args.tokens),
        },
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

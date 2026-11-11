# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Engram INT8 lookup, BF16 projection, original-basis hidden and gate readbacks."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a2"

import pypto.language as pl

from .qkv_proj_rope import make_bf16_projection_staged

from .engram_ecotech import (
    COLS, D, HC, K, N, ROTATION_TILE, WIDTH,
    make_original_basis_gate, make_original_basis_hidden, make_resident_int8_lookup,
)


def make_engram_ecotech_observed(capacity, table_rows, rms_norm_eps=1e-20):
    if capacity < 1 or capacity % 16 or table_rows < 1 or rms_norm_eps <= 0:
        raise ValueError("Use a positive padded capacity, resident row count and source norm epsilon")
    gather = pl.jit.inline(make_resident_int8_lookup(capacity, table_rows)._func)
    project = pl.jit.inline(make_bf16_projection_staged(K, N)._func)
    restore = make_original_basis_hidden(capacity)
    gate = make_original_basis_gate(capacity, rms_norm_eps)

    @pl.jit
    def engram_ecotech_observed(
        staging_ids: pl.InOut[pl.Tensor[[capacity, COLS], pl.INT32]],
        table: pl.InOut[pl.Tensor[[table_rows, WIDTH], pl.INT8]],
        table_scales: pl.InOut[pl.Tensor[[table_rows, 8], pl.FP32]],
        wkv: pl.InOut[pl.Tensor[[K, N], pl.BF16]],
        channel_weight: pl.InOut[pl.Tensor[[HC, D], pl.FP32]],
        hidden: pl.InOut[pl.Tensor[[capacity, HC, D], pl.BF16]],
        rotation_block: pl.InOut[pl.Tensor[[ROTATION_TILE, ROTATION_TILE], pl.FP32]],
        token_mask: pl.InOut[pl.Tensor[[capacity], pl.INT32]],
        lookup: pl.InOut[pl.Tensor[[capacity, K], pl.BF16]],
        kv: pl.InOut[pl.Tensor[[capacity, N], pl.BF16]],
        original: pl.InOut[pl.Tensor[[capacity, HC, D], pl.FP32]],
        probabilities: pl.InOut[pl.Tensor[[capacity, HC], pl.FP32]],
        output: pl.InOut[pl.Tensor[[capacity, HC, D], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        gather(staging_ids, table, table_scales, lookup, active)
        project(lookup, wkv, kv, active)
        restore(hidden, rotation_block, original, active)
        gate(hidden, original, kv, channel_weight, token_mask, probabilities, output, active)
        return (
            staging_ids, table, table_scales, wkv, channel_weight, hidden, rotation_block, token_mask,
            lookup, kv, original, probabilities, output,
        )

    return engram_ecotech_observed


def engram_observed_golden_cases():
    """Bound full-width resident composition and exact readonly input readback."""
    import torch
    from golden import ScalarSpec, TensorSpec
    from golden.validation import ratio_allclose
    from .engram_ecotech import golden_engram_ecotech, golden_masked_gate_compare
    from .qkv_proj_rope import golden_prefix_compare

    capacity, rows, active = 16, 19, 3
    generator = torch.Generator().manual_seed(733)
    ids = (torch.arange(capacity * COLS).reshape(capacity, COLS) % (rows + 2) - 1).to(torch.int32)
    table = (torch.arange(rows * WIDTH).reshape(rows, WIDTH) % 256 - 128).to(torch.int8)
    scales = torch.empty((rows, 8), dtype=torch.float32).uniform_(0.001, 0.02, generator=generator)
    def wkv():
        weight_generator = torch.Generator().manual_seed(734)
        return torch.empty((K, N), dtype=torch.bfloat16).uniform_(-0.02, 0.02, generator=weight_generator)

    channels = torch.empty((HC, D), dtype=torch.float32).uniform_(-1.0, 1.0, generator=generator)
    hidden = torch.empty((capacity, HC, D), dtype=torch.bfloat16).uniform_(-0.5, 0.5, generator=generator)
    rotation = torch.linalg.qr(torch.randn((32, 32), generator=generator)).Q.contiguous()
    mask = (torch.arange(capacity) % 3 != 1).to(torch.int32)
    specs = [
        TensorSpec("staging_ids", [capacity, COLS], torch.int32, init_value=ids),
        TensorSpec("table", [rows, WIDTH], torch.int8, init_value=table),
        TensorSpec("table_scales", [rows, 8], torch.float32, init_value=scales),
        TensorSpec("wkv", [K, N], torch.bfloat16, init_value=wkv),
        TensorSpec("channel_weight", [HC, D], torch.float32, init_value=channels),
        TensorSpec("hidden", [capacity, HC, D], torch.bfloat16, init_value=hidden),
        TensorSpec("rotation_block", [32, 32], torch.float32, init_value=rotation),
        TensorSpec("token_mask", [capacity], torch.int32, init_value=mask),
        TensorSpec("lookup", [capacity, K], torch.bfloat16, init_value=-12.5),
        TensorSpec("kv", [capacity, N], torch.bfloat16, init_value=-12.5),
        TensorSpec("original", [capacity, HC, D], torch.float32, init_value=-12.5),
        TensorSpec("probabilities", [capacity, HC], torch.float32, init_value=-12.5),
        TensorSpec("output", [capacity, HC, D], torch.bfloat16, init_value=-12.5),
        ScalarSpec("active", torch.int32, active, compile_runtime=True),
    ]
    readonly = ("staging_ids", "table", "table_scales", "wkv", "channel_weight", "hidden",
                "rotation_block", "token_mask")
    exact = ratio_allclose(rtol=0.0, atol=0.0, max_error_ratio=0.0)
    compare = {name: exact for name in readonly}
    compare["lookup"] = golden_prefix_compare(active, rtol=0.0, atol=0.0)
    compare.update({name: golden_prefix_compare(active) for name in ("kv", "original")})
    compare.update({name: golden_masked_gate_compare(active) for name in ("probabilities", "output")})
    yield dict(name="engram-observed-readonly-3", fn=make_engram_ecotech_observed(capacity, rows),
               specs=specs, golden_fn=golden_engram_ecotech, compare_fn=compare)


def validate_engram_observed(*, device=0, compile_only=False, platform="a2a3", sim=False):
    """Run the existing full-width readback composite through the Golden Harness."""
    from golden import run

    for case in engram_observed_golden_cases():
        name = case.pop("name")
        result = run(
            **case,
            config={"backend_type": platform, "execution_mode": sim, "device_id": device},
            compile_only=compile_only,
        )
        if not result.passed:
            raise RuntimeError(f"{name}: {result.error}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    validate_engram_observed(device=args.device, compile_only=args.compile_only, platform=args.platform, sim=args.sim)

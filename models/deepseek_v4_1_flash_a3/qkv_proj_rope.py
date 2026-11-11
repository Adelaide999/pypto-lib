# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2 single-layer dependencies extracted from the shared model operators."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

from math import gcd

import pypto.language as pl
from .config import FLASH as M
from .operator_config import T_DYN


# model config
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim


# tiling
M_TILE = 16
N_TILE = 128
K_TILE = 64
BF16_N_TILE = 16
BF16_K_TILE = 512


def _make_rope_block(heads, inverse, head_dim, rope_dim):
    sign = -1.0 if inverse else 1.0
    nope_dim = head_dim - rope_dim

    @pl.jit.inline
    def rotate_block(
        x: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        cos: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        output: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        block: pl.Scalar[pl.INT32],
    ):
        t = block // heads
        h = block % heads
        base = h * head_dim
        output[t : t + 1, base : base + nope_dim] = x[t : t + 1, base : base + nope_dim]
        tail = pl.cast(x[t : t + 1, base + nope_dim : base + head_dim], pl.FP32)
        even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
        odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
        c = cos[t : t + 1, :]
        s = pl.mul(sin[t : t + 1, :], sign)
        re = pl.sub(pl.mul(even, c), pl.mul(odd, s))
        im = pl.add(pl.mul(even, s), pl.mul(odd, c))
        rotated = pl.full([1, rope_dim], dtype=pl.FP32, value=0.0)
        rotated = pl.tensor.scatter(re, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
        rotated = pl.tensor.scatter(im, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
        output[t : t + 1, base + nope_dim : base + head_dim] = pl.cast(
            rotated, pl.BF16, mode="rint"
        )
        return output

    return rotate_block


def make_rope(
    heads,
    inverse=False,
    *,
    head_dim=HEAD_DIM,
    rope_dim=ROPE_DIM,
    max_workers=None,
    name_hint="attention_rope",
):
    """Specialize adjacent-pair RoPE with direct or capped strided scheduling."""
    rotate_block = _make_rope_block(heads, inverse, head_dim, rope_dim)
    worker_limit = 2**31 - 1 if max_workers is None else max_workers

    @pl.jit.inline
    def rotate(
        x: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        cos: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        output: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        blocks = num_tokens * heads
        workers = pl.min(blocks, worker_limit)
        for worker in pl.spmd(workers, name_hint=name_hint):
            for block in pl.range(worker, blocks, workers):
                rotate_block(x, cos, sin, output, block)
        return output

    return rotate


def make_bf16_projection_staged(width, output_width, *, capacity=None):
    """Accumulate BF16 Cube slabs by paired output owner and narrow once.

    The BF16 operand passes through Vector before Cube starts its result transfer
    (hw-native-sys/pypto#2829). Store FP32 values through Vector as well: direct
    Acc-to-GM stores were unstable in the concurrent A5 prefill workload.
    """
    n_tile = gcd(BF16_N_TILE, output_width)
    k_tile = gcd(BF16_K_TILE, width)
    n_owner_tile = gcd(32, output_width)
    n_owners = output_width // n_owner_tile
    n_pairs = max(1, n_owners // 2)
    shift_enabled = int(4 <= n_owners <= 2 * (width // k_tile) + 1)
    shift_step = (width // k_tile) // n_pairs * shift_enabled
    extent = T_DYN if capacity is None else capacity

    @pl.jit
    def project(
        source: pl.Tensor[[extent, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.BF16],
        output: pl.InOut[pl.Tensor[[extent, output_width], pl.BF16]],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        tokens = pl.tensor.dim(source, 0)
        blocks = (num_tokens + M_TILE - 1) // M_TILE * (output_width // n_tile)
        accumulation = pl.create_tensor([tokens, output_width], dtype=pl.FP32)
        with pl.spmd(blocks, name_hint="prefill_bf16_projection_cube") as cube_tid:
            block = pl.tile.get_block_idx()
            row = block // (output_width // n_tile) * M_TILE
            col = block % (output_width // n_tile) * n_tile
            rows = pl.min(M_TILE, num_tokens - row)
            owner = col // n_owner_tile
            parity = owner % 2 * shift_enabled
            phase = shift_step * ((owner // 2) % n_pairs)
            acc = pl.create_tensor([M_TILE, n_tile], dtype=pl.FP32)
            for step in pl.range(width // k_tile):
                directed = step + parity * (width // k_tile - 1 - 2 * step)
                physical = (directed + phase) % (width // k_tile)
                k = physical * k_tile
                a = pl.slice(source, [M_TILE, k_tile], [row, k], valid_shape=[rows, k_tile])
                a = pl.cast(a, pl.FP32)
                a = pl.cast(a, pl.BF16, mode="rint")
                b = weight[k:k + k_tile, col:col + n_tile]
                acc = pl.matmul_acc(acc, a, b, init_cond=(step == 0))
            transferred = pl.mul(acc, 1.0)
            transferred = pl.set_validshape(transferred, rows, n_tile)
            accumulation[row:row + M_TILE, col:col + n_tile] = transferred
        for block in pl.spmd(blocks, name_hint="prefill_bf16_projection_cast", deps=[cube_tid]):
            row = block // (output_width // n_tile) * M_TILE
            col = block % (output_width // n_tile) * n_tile
            rows = pl.min(M_TILE, num_tokens - row)
            result = pl.slice(accumulation, [M_TILE, n_tile], [row, col], valid_shape=[rows, n_tile])
            narrowed = pl.cast(result, pl.BF16, mode="rint")
            output[row:row + M_TILE, col:col + n_tile] = narrowed
        return output

    return project


def golden_bf16_projection(tensors):
    """Evaluate FP32 matrix multiplication and one BF16 rounding boundary."""
    active = int(tensors["num_tokens"])
    source = tensors["source"][:active].float()
    weight = tensors["weight"].float()
    tensors["output"][:active] = (source @ weight).to(tensors["output"].dtype)


def golden_prefix_compare(active, *, rtol=0.05, atol=0.05):
    """Compare the numeric prefix and require an exact inactive sentinel tail."""
    from golden.validation import ratio_allclose

    numeric = ratio_allclose(rtol=rtol, atol=atol, max_error_ratio=0.0)
    exact = ratio_allclose(rtol=0.0, atol=0.0, max_error_ratio=0.0)

    def compare(actual, expected, **context):
        for part, selection in (("active", slice(None, active)), ("inactive", slice(active, None))):
            comparator = numeric if part == "active" else exact
            if actual[selection].numel():
                passed, detail = comparator(actual[selection], expected[selection], **context)
                if not passed:
                    return False, f"{part}: {detail}"
        return True, ""

    return compare


def bf16_projection_golden_cases():
    """Bound dynamic/static extents, empty work, partial tiles and BF16 rounding."""
    import torch
    from golden import ScalarSpec, TensorSpec

    generator = torch.Generator().manual_seed(731)
    for static, active in ((False, 0), (False, 17), (True, 1), (True, 32)):
        capacity, width, output_width = 32, 512, 128
        source = torch.empty((capacity, width), dtype=torch.bfloat16).uniform_(-0.5, 0.5, generator=generator)
        weight = torch.empty((width, output_width), dtype=torch.bfloat16).uniform_(-0.5, 0.5, generator=generator)
        extent = capacity if static else None
        specs = [
            TensorSpec("source", [capacity, width], torch.bfloat16, init_value=source),
            TensorSpec("weight", [width, output_width], torch.bfloat16, init_value=weight),
            TensorSpec("output", [capacity, output_width], torch.bfloat16, init_value=-12.5),
            ScalarSpec("num_tokens", torch.int32, active, compile_runtime=True),
        ]
        yield dict(name=f"projection-{'static' if static else 'dynamic'}-{active}",
                   fn=make_bf16_projection_staged(width, output_width, capacity=extent), specs=specs,
                   golden_fn=golden_bf16_projection, compare_fn={"output": golden_prefix_compare(active)})


def validate_bf16_projection(*, device=0, compile_only=False, platform="a2a3", sim=False):
    """Run bounded owning projection cases through the Golden Harness."""
    from golden import run

    for case in bf16_projection_golden_cases():
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
    validate_bf16_projection(device=args.device, compile_only=args.compile_only, platform=args.platform, sim=args.sim)

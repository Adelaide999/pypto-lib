"""Single-card Decode INT8 GEMM comparison for the 1122 A3 branch.

The candidate keeps the Decode-only full-K schedule used by the current-base
experiments. It consumes a prepacked [panel*K, N_TILE] weight buffer, so the
benchmark measures the device kernel and excludes one-time host packing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pypto.language as pl
import torch

from golden import TensorSpec, ratio_allclose, run
from models.deepseek_v4_1_flash_a3.gemm import gemm_int8_pair_columns, gemm_int8_tiled


TOKENS = 1
CAPACITY = 16
WIDTH = 1280
OUTPUT_WIDTH = 8192
N_TILE = 256
K_TILE = 256
PANELS = OUTPUT_WIDTH // N_TILE


@pl.jit.inline(auto_scope=False)
def gemm_int8_full_k(
    payload: pl.Tensor,
    activation_scale: pl.Tensor,
    packed_weight: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    """Reuse one full-K activation tile across a group of output panels."""
    row_tile = 16
    column_group = 2
    row_blocks = (num_tokens + row_tile - 1) // row_tile
    column_blocks = (PANELS + column_group - 1) // column_group
    blocks = row_blocks * column_blocks
    workers = pl.max(1, pl.min(blocks, pl.system.available_cluster_count()))
    for worker in pl.spmd(workers, name_hint="decode_int8_full_k"):
        for block in pl.range(worker, blocks, workers):
            row = block // column_blocks * row_tile
            first_panel = block % column_blocks * column_group
            last_panel = pl.min(first_panel + column_group, PANELS)
            valid_rows = pl.max(0, pl.min(row_tile, num_tokens - row))
            source = pl.load(
                payload,
                [row, 0],
                [row_tile, WIDTH],
                valid_shape=[valid_rows, WIDTH],
                target_memory=pl.MemorySpace.Mat,
            )
            for panel in pl.pipeline(first_panel, last_panel, stage=1):
                col = panel * N_TILE
                weight = pl.load(
                    packed_weight,
                    [panel * WIDTH, 0],
                    [WIDTH, N_TILE],
                    target_memory=pl.MemorySpace.Mat,
                )
                product = pl.matmul(source, weight, out_dtype=pl.INT32)
                value = pl.cast(product, pl.FP32)
                a_scale = pl.load(
                    activation_scale,
                    [row, 0],
                    [row_tile, 1],
                    valid_shape=[valid_rows, 1],
                )
                b_scale = pl.load(weight_scale, [0, col], [1, N_TILE])
                value = pl.row_expand_mul(value, a_scale)
                value = pl.col_expand_mul(value, b_scale)
                value = pl.cast(value, pl.BF16, mode="rint")
                output = pl.store(
                    pl.set_validshape(value, valid_rows, N_TILE),
                    [row, col],
                    output,
                )
    return output


@pl.jit.inline(auto_scope=False)
def gemm_int8_full_k_canonical(
    payload: pl.Tensor,
    activation_scale: pl.Tensor,
    weight: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    """Reuse one full-K activation tile with the existing [K, N] ABI."""
    row_tile = 16
    column_group = 2
    row_blocks = (num_tokens + row_tile - 1) // row_tile
    column_blocks = (PANELS + column_group - 1) // column_group
    blocks = row_blocks * column_blocks
    workers = pl.max(1, pl.min(blocks, pl.system.available_cluster_count()))
    for worker in pl.spmd(workers, name_hint="decode_int8_full_k_canonical"):
        for block in pl.range(worker, blocks, workers):
            row = block // column_blocks * row_tile
            first_panel = block % column_blocks * column_group
            last_panel = pl.min(first_panel + column_group, PANELS)
            valid_rows = pl.max(0, pl.min(row_tile, num_tokens - row))
            source = pl.load(
                payload,
                [row, 0],
                [row_tile, WIDTH],
                valid_shape=[valid_rows, WIDTH],
                target_memory=pl.MemorySpace.Mat,
            )
            for panel in pl.pipeline(first_panel, last_panel, stage=1):
                col = panel * N_TILE
                panel_weight = pl.load(
                    weight,
                    [0, col],
                    [WIDTH, N_TILE],
                    target_memory=pl.MemorySpace.Mat,
                )
                product = pl.matmul(source, panel_weight, out_dtype=pl.INT32)
                value = pl.cast(product, pl.FP32)
                a_scale = pl.load(
                    activation_scale,
                    [row, 0],
                    [row_tile, 1],
                    valid_shape=[valid_rows, 1],
                )
                b_scale = pl.load(weight_scale, [0, col], [1, N_TILE])
                value = pl.row_expand_mul(value, a_scale)
                value = pl.col_expand_mul(value, b_scale)
                value = pl.cast(value, pl.BF16, mode="rint")
                output = pl.store(
                    pl.set_validshape(value, valid_rows, N_TILE),
                    [row, col],
                    output,
                )
    return output


@pl.jit
def l2_baseline(
    payload: pl.Tensor[[CAPACITY, WIDTH], pl.INT8],
    activation_scale: pl.Tensor[[CAPACITY, 1], pl.FP32],
    weight: pl.Tensor[[WIDTH, OUTPUT_WIDTH], pl.INT8],
    weight_scale: pl.Tensor[[1, OUTPUT_WIDTH], pl.FP32],
    counts: pl.Tensor[[1], pl.INT32],
    output: pl.Out[pl.Tensor[[CAPACITY, OUTPUT_WIDTH], pl.BF16]],
):
    count = pl.read(counts, [0])
    return gemm_int8_tiled(
        payload,
        activation_scale,
        weight,
        weight_scale,
        output,
        count,
        WIDTH,
        OUTPUT_WIDTH,
        "decode_int8_baseline",
        N_TILE,
        K_TILE,
        False,
        16,
    )


@pl.jit
def l2_candidate(
    payload: pl.Tensor[[CAPACITY, WIDTH], pl.INT8],
    activation_scale: pl.Tensor[[CAPACITY, 1], pl.FP32],
    packed_weight: pl.Tensor[[WIDTH * PANELS, N_TILE], pl.INT8],
    weight_scale: pl.Tensor[[1, OUTPUT_WIDTH], pl.FP32],
    counts: pl.Tensor[[1], pl.INT32],
    output: pl.Out[pl.Tensor[[CAPACITY, OUTPUT_WIDTH], pl.BF16]],
):
    count = pl.read(counts, [0])
    return gemm_int8_full_k(payload, activation_scale, packed_weight, weight_scale, output, count)


@pl.jit
def l2_candidate_canonical(
    payload: pl.Tensor[[CAPACITY, WIDTH], pl.INT8],
    activation_scale: pl.Tensor[[CAPACITY, 1], pl.FP32],
    weight: pl.Tensor[[WIDTH, OUTPUT_WIDTH], pl.INT8],
    weight_scale: pl.Tensor[[1, OUTPUT_WIDTH], pl.FP32],
    counts: pl.Tensor[[1], pl.INT32],
    output: pl.Out[pl.Tensor[[CAPACITY, OUTPUT_WIDTH], pl.BF16]],
):
    count = pl.read(counts, [0])
    return gemm_int8_full_k_canonical(payload, activation_scale, weight, weight_scale, output, count)


@pl.jit
def l2_pair_columns(
    payload: pl.Tensor[[CAPACITY, WIDTH], pl.INT8],
    activation_scale: pl.Tensor[[CAPACITY, 1], pl.FP32],
    weight: pl.Tensor[[WIDTH, OUTPUT_WIDTH], pl.INT8],
    weight_scale: pl.Tensor[[1, OUTPUT_WIDTH], pl.FP32],
    counts: pl.Tensor[[1], pl.INT32],
    output: pl.Out[pl.Tensor[[CAPACITY, OUTPUT_WIDTH], pl.BF16]],
):
    count = pl.read(counts, [0])
    return gemm_int8_pair_columns(
        payload, activation_scale, weight, weight_scale, output, count,
        WIDTH, OUTPUT_WIDTH, "decode_int8_pair_columns", N_TILE, K_TILE, False, 16,
    )


def _golden(values: dict[str, torch.Tensor], packed: bool) -> None:
    payload = values["payload"].long()
    if packed:
        packed_weight = values["packed_weight"]
        weight = packed_weight.view(PANELS, WIDTH, N_TILE).permute(1, 0, 2).reshape(WIDTH, OUTPUT_WIDTH).long()
    else:
        weight = values["weight"].long()
    result = payload @ weight
    result = result.float() * values["activation_scale"] * values["weight_scale"]
    values["output"][:TOKENS].copy_(result[:TOKENS].bfloat16())


def _make_values() -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(1122)
    payload = torch.randint(-127, 128, (CAPACITY, WIDTH), dtype=torch.int8, generator=gen)
    activation_scale = torch.full((CAPACITY, 1), 0.02 / 127, dtype=torch.float32)
    weight = torch.randint(-127, 128, (WIDTH, OUTPUT_WIDTH), dtype=torch.int8, generator=gen)
    weight_scale = torch.full((1, OUTPUT_WIDTH), 0.02 / 127, dtype=torch.float32)
    packed = torch.empty((WIDTH * PANELS, N_TILE), dtype=torch.int8)
    for panel in range(PANELS):
        packed[panel * WIDTH : (panel + 1) * WIDTH] = weight[:, panel * N_TILE : (panel + 1) * N_TILE]
    return {
        "payload": payload,
        "activation_scale": activation_scale,
        "weight": weight,
        "packed_weight": packed,
        "weight_scale": weight_scale,
        "counts": torch.tensor([TOKENS], dtype=torch.int32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "candidate", "candidate-canonical", "pair"), required=True)
    parser.add_argument("--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    values = _make_values()
    if args.mode == "baseline":
        specs = [
            TensorSpec("payload", [CAPACITY, WIDTH], torch.int8, init_value=values["payload"]),
            TensorSpec("activation_scale", [CAPACITY, 1], torch.float32, init_value=values["activation_scale"]),
            TensorSpec("weight", [WIDTH, OUTPUT_WIDTH], torch.int8, init_value=values["weight"]),
            TensorSpec("weight_scale", [1, OUTPUT_WIDTH], torch.float32, init_value=values["weight_scale"]),
            TensorSpec("counts", [1], torch.int32, init_value=values["counts"]),
        ]
        fn = l2_baseline
    elif args.mode == "candidate":
        specs = [
            TensorSpec("payload", [CAPACITY, WIDTH], torch.int8, init_value=values["payload"]),
            TensorSpec("activation_scale", [CAPACITY, 1], torch.float32, init_value=values["activation_scale"]),
            TensorSpec("packed_weight", [WIDTH * PANELS, N_TILE], torch.int8, init_value=values["packed_weight"]),
            TensorSpec("weight_scale", [1, OUTPUT_WIDTH], torch.float32, init_value=values["weight_scale"]),
            TensorSpec("counts", [1], torch.int32, init_value=values["counts"]),
        ]
        fn = l2_candidate
    elif args.mode == "pair":
        specs = [
            TensorSpec("payload", [CAPACITY, WIDTH], torch.int8, init_value=values["payload"]),
            TensorSpec("activation_scale", [CAPACITY, 1], torch.float32, init_value=values["activation_scale"]),
            TensorSpec("weight", [WIDTH, OUTPUT_WIDTH], torch.int8, init_value=values["weight"]),
            TensorSpec("weight_scale", [1, OUTPUT_WIDTH], torch.float32, init_value=values["weight_scale"]),
            TensorSpec("counts", [1], torch.int32, init_value=values["counts"]),
        ]
        fn = l2_pair_columns
    else:
        specs = [
            TensorSpec("payload", [CAPACITY, WIDTH], torch.int8, init_value=values["payload"]),
            TensorSpec("activation_scale", [CAPACITY, 1], torch.float32, init_value=values["activation_scale"]),
            TensorSpec("weight", [WIDTH, OUTPUT_WIDTH], torch.int8, init_value=values["weight"]),
            TensorSpec("weight_scale", [1, OUTPUT_WIDTH], torch.float32, init_value=values["weight_scale"]),
            TensorSpec("counts", [1], torch.int32, init_value=values["counts"]),
        ]
        fn = l2_candidate_canonical
    specs.append(TensorSpec("output", [CAPACITY, OUTPUT_WIDTH], torch.bfloat16))
    result = run(
        fn=fn,
        specs=specs,
        golden_fn=lambda tensors: _golden(tensors, args.mode == "candidate"),
        compare_fn={"output": ratio_allclose(atol=0.003, rtol=0.03, max_error_ratio=0.005)},
        config={"backend_type": args.platform, "execution_mode": False, "device_id": args.device},
        compile_only=args.compile_only,
    )
    report = {
        "mode": args.mode,
        "passed": result.passed,
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

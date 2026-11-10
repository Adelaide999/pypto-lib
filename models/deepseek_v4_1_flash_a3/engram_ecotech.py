# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""INT8/F32 Engram lookup, BF16 projection and block32 original-basis gating."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

import pypto.language as pl

from .qkv_proj_rope import make_bf16_projection_staged


# model config
D, HC, COLS, WIDTH = 5120, 4, 24, 256
K, N = COLS * WIDTH, (HC + 1) * D

# tiling
T_TILE, D_TILE, ROTATION_TILE = 16, 512, 32
RESTORE_M_TILE = 16


def make_resident_int8_lookup(capacity, table_rows, columns=COLS, head_start=0, row_start=0):
    """Fuse guarded INT8 lookup and group32 BF16 RNE in token-column SPMD tasks."""

    @pl.jit
    def lookup_rows(
        staging_ids: pl.Tensor[[capacity, COLS], pl.INT32],
        table: pl.Tensor[[table_rows, WIDTH], pl.INT8],
        table_scales: pl.Tensor[[table_rows, 8], pl.FP32],
        lookup: pl.InOut[pl.Tensor[[capacity, columns * WIDTH], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for task in pl.spmd(active * columns, name_hint="engram_ecotech_i8_lookup"):
            token = task // columns
            column = task % columns
            global_row = pl.cast(pl.read(staging_ids, [token, head_start + column]), pl.INT64)
            local_row = global_row - row_start
            row = pl.cast(local_row, pl.INDEX)
            for group in pl.range(8):
                decoded = pl.full([1, 32], dtype=pl.BF16, value=0.)
                if local_row >= 0:
                    if local_row < table_rows:
                        code_half = pl.cast(table[row:row + 1, group * 32:group * 32 + 32], pl.FP16)
                        code_fp32 = pl.cast(code_half, pl.FP32)
                        scale = pl.read(table_scales, [row, group])
                        decoded = pl.cast(pl.mul(code_fp32, scale), pl.BF16, mode="rint")
                start = column * WIDTH + group * 32
                lookup[token:token + 1, start:start + 32] = decoded
        return lookup

    return lookup_rows


def make_original_basis_hidden(capacity):
    """Restore FP32 hidden with independent block32 FP32 matrix products."""

    @pl.jit.inline
    def restore(
        hidden: pl.Tensor[[capacity, HC, D], pl.BF16],
        rotation_block: pl.Tensor[[ROTATION_TILE, ROTATION_TILE], pl.FP32],
        original: pl.Tensor[[capacity, HC, D], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        flat_hidden = pl.reshape(hidden, [capacity * HC * (D // ROTATION_TILE), ROTATION_TILE])
        flat_original = pl.reshape(original, [capacity * HC * (D // ROTATION_TILE), ROTATION_TILE])
        blocks = active * HC * (D // ROTATION_TILE // RESTORE_M_TILE)
        for task in pl.spmd(blocks, name_hint="engram_ecotech_original_basis_cube"):
            row = task * RESTORE_M_TILE
            source_bf16 = flat_hidden[row:row + RESTORE_M_TILE, :]
            source_fp32 = pl.cast(source_bf16, pl.FP32)
            rotation_fp32 = rotation_block[:, :]
            restored = pl.matmul(source_fp32, rotation_fp32, out_dtype=pl.FP32, b_trans=True)
            flat_original[row:row + RESTORE_M_TILE, :] = pl.mul(restored, 1.0)
        return original

    return restore


def make_original_basis_gate(capacity, eps):
    """Compute the original-basis gate and add the already rotated value."""

    @pl.jit.inline
    def gate(
        hidden: pl.Tensor[[capacity, HC, D], pl.BF16],
        original: pl.Tensor[[capacity, HC, D], pl.FP32],
        kv: pl.Tensor[[capacity, N], pl.BF16],
        channel_weight: pl.Tensor[[HC, D], pl.FP32],
        token_mask: pl.Tensor[[capacity], pl.INT32],
        probabilities: pl.Tensor[[capacity, HC], pl.FP32],
        output: pl.Tensor[[capacity, HC, D], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        flat_original = pl.reshape(original, [capacity, HC * D])
        flat_hidden = pl.reshape(hidden, [capacity, HC * D])
        flat_output = pl.reshape(output, [capacity, HC * D])
        mask = pl.reshape(token_mask, [1, capacity])
        for task in pl.spmd((active + T_TILE - 1) // T_TILE, name_hint="engram_ecotech_gate"):
            t0 = task * T_TILE
            rows = pl.min(T_TILE, active - t0)
            # One token-tile owner writes every probability cache line.
            for branch in pl.range(HC):
                sx = pl.full([1, T_TILE], dtype=pl.FP32, value=0.)
                sk = pl.full([1, T_TILE], dtype=pl.FP32, value=0.)
                dot = pl.full([1, T_TILE], dtype=pl.FP32, value=0.)
                for d0 in pl.range(0, D, D_TILE):
                    x = pl.slice(flat_original, [T_TILE, D_TILE], [t0, branch * D + d0], valid_shape=[rows, D_TILE])
                    key_bf16 = pl.slice(kv, [T_TILE, D_TILE], [t0, branch * D + d0], valid_shape=[rows, D_TILE])
                    key = pl.cast(key_bf16, pl.FP32)
                    weight = channel_weight[branch:branch + 1, d0:d0 + D_TILE]
                    sx = pl.add(sx, pl.reshape(pl.row_sum(pl.mul(x, x)), [1, T_TILE]))
                    sk = pl.add(sk, pl.reshape(pl.row_sum(pl.mul(key, key)), [1, T_TILE]))
                    weighted_x = pl.col_expand_mul(x, weight)
                    dot = pl.add(dot, pl.reshape(pl.row_sum(pl.mul(weighted_x, key)), [1, T_TILE]))
                inv_x = pl.rsqrt(pl.add(pl.mul(sx, 1. / D), eps), high_precision=True)
                inv_key = pl.rsqrt(pl.add(pl.mul(sk, 1. / D), eps), high_precision=True)
                inv = pl.mul(inv_x, inv_key)
                dot = pl.mul(pl.mul(dot, inv), D**-0.5)
                magnitude = pl.sqrt(pl.maximum(pl.abs(dot), 1e-6))
                bits = pl.cast(pl.reinterpret_view(dot, pl.INT32), pl.FP32)
                negative = pl.maximum(pl.minimum(bits, 0.), -1.)
                signed = pl.mul(magnitude, pl.add(pl.mul(negative, 2.), 1.))
                eligible = pl.cast(pl.slice(mask, [1, T_TILE], [0, t0], valid_shape=[1, rows]), pl.FP32)
                probability = pl.recip(pl.add(pl.exp(pl.neg(signed)), 1.))
                probability = pl.reshape(pl.mul(probability, eligible), [T_TILE, 1])
                for row in pl.range(rows):
                    pl.write(probabilities, [t0 + row, branch], pl.read(probability, [row, 0]))
                for d0 in pl.range(0, D, D_TILE):
                    x_bf16 = pl.slice(flat_hidden, [T_TILE, D_TILE], [t0, branch * D + d0], valid_shape=[rows, D_TILE])
                    value_bf16 = pl.slice(kv, [T_TILE, D_TILE], [t0, HC * D + d0], valid_shape=[rows, D_TILE])
                    value = pl.cast(value_bf16, pl.FP32)
                    result = pl.add(pl.cast(x_bf16, pl.FP32), pl.row_expand_mul(value, probability))
                    result = pl.cast(result, pl.BF16, mode="rint")
                    start = branch * D + d0
                    flat_output[t0:t0 + T_TILE, start:start + D_TILE] = pl.set_validshape(result, rows, D_TILE)
        return probabilities, output

    return gate


def make_engram_ecotech(capacity, table_rows, rms_norm_eps=1e-20):
    """Compose resident lookup, shared BF16 projection and rotated-residual gating."""
    if capacity < 1 or capacity % 16 or table_rows < 1 or rms_norm_eps <= 0:
        raise ValueError("Use a positive padded capacity, resident row count and source norm epsilon")
    gather = pl.jit.inline(make_resident_int8_lookup(capacity, table_rows)._func)
    project = pl.jit.inline(make_bf16_projection_staged(K, N)._func)
    restore = make_original_basis_hidden(capacity)
    gate = make_original_basis_gate(capacity, rms_norm_eps)

    @pl.jit
    def engram_ecotech(
        staging_ids: pl.Tensor[[capacity, COLS], pl.INT32],
        table: pl.Tensor[[table_rows, WIDTH], pl.INT8],
        table_scales: pl.Tensor[[table_rows, 8], pl.FP32],
        wkv: pl.Tensor[[K, N], pl.BF16],
        channel_weight: pl.Tensor[[HC, D], pl.FP32],
        hidden: pl.Tensor[[capacity, HC, D], pl.BF16],
        rotation_block: pl.Tensor[[ROTATION_TILE, ROTATION_TILE], pl.FP32],
        token_mask: pl.Tensor[[capacity], pl.INT32],
        lookup: pl.InOut[pl.Tensor[[capacity, K], pl.BF16]],
        kv: pl.InOut[pl.Tensor[[capacity, N], pl.BF16]],
        original: pl.InOut[pl.Tensor[[capacity, HC, D], pl.FP32]],
        output: pl.InOut[pl.Tensor[[capacity, HC, D], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        gather(staging_ids, table, table_scales, lookup, active)
        project(lookup, wkv, kv, active)
        restore(hidden, rotation_block, original, active)
        probabilities = pl.create_tensor([capacity, HC], dtype=pl.FP32)
        gate(hidden, original, kv, channel_weight, token_mask, probabilities, output, active)
        return lookup, kv, original, output

    return engram_ecotech


def reference_int8_rows(ids, table, table_scales, *, row_start=0):
    """Select bounded table rows with zero codes/scales outside the shard."""
    import torch

    local = ids.long().reshape(-1) - row_start
    valid = (local >= 0) & (local < table.shape[0])
    bounded = local.clamp(0, table.shape[0] - 1)
    codes = torch.where(valid[:, None], table[bounded], 0)
    scales = torch.where(valid[:, None], table_scales[bounded], 0.0)
    return codes, scales


def golden_int8_lookup(tensors, *, head_start=0, row_start=0):
    """Evaluate guarded shard selection, group32 scaling and inactive rows."""
    active = int(tensors["active"])
    width = tensors["lookup"].shape[1]
    columns = width // WIDTH
    ids = tensors["staging_ids"][:active, head_start:head_start + columns]
    codes, scales = reference_int8_rows(ids, tensors["table"], tensors["table_scales"], row_start=row_start)
    values = codes.float().reshape(active, columns, 8, 32) * scales.reshape(active, columns, 8, 1)
    tensors["lookup"][:active] = values.reshape(active, width).to(tensors["lookup"].dtype)


def golden_original_basis_hidden(tensors):
    """Restore block32 hidden by independent FP32 matrix multiplication."""
    active = int(tensors["active"])
    hidden = tensors["hidden"][:active].float().reshape(-1, ROTATION_TILE)
    restored = hidden @ tensors["rotation_block"].float().T
    tensors["original"][:active] = restored.reshape(active, HC, D)


def golden_original_basis_gate(tensors, *, eps=1e-20):
    """Evaluate RMS-normalized channel products, signed-root sigmoid and residual."""
    import torch

    active = int(tensors["active"])
    original = tensors["original"][:active].float()
    key = tensors["kv"][:active, :HC * D].float().reshape(active, HC, D)
    inv_x = torch.rsqrt(original.square().mean(-1) + eps)
    inv_key = torch.rsqrt(key.square().mean(-1) + eps)
    dot = (original * tensors["channel_weight"].float() * key).sum(-1)
    dot = dot * inv_x * inv_key * D**-0.5
    signed = torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot)
    probability = torch.sigmoid(signed) * tensors["token_mask"][:active, None].float()
    tensors["probabilities"][:active] = probability
    value = tensors["kv"][:active, HC * D:].float()[:, None, :]
    result = tensors["hidden"][:active].float() + probability[:, :, None] * value
    tensors["output"][:active] = result.to(tensors["output"].dtype)


def golden_engram_ecotech(tensors, *, eps=1e-20):
    """Compose independent INT8, projection, original and gate references."""
    import torch
    from .qkv_proj_rope import golden_bf16_projection

    active = int(tensors["active"])
    ids = tensors["staging_ids"][:active]
    codes, scales = reference_int8_rows(ids, tensors["table"], tensors["table_scales"])
    decoded = codes.float().reshape(active, COLS, 8, 32) * scales.reshape(active, COLS, 8, 1)
    tensors["lookup"][:active] = decoded.reshape(active, K).to(tensors["lookup"].dtype)
    projection = dict(source=tensors["lookup"], weight=tensors["wkv"], output=tensors["kv"], num_tokens=active)
    golden_bf16_projection(projection)
    golden_original_basis_hidden(tensors)
    values = dict(tensors)
    if "probabilities" not in values:
        values["probabilities"] = torch.empty((tensors["hidden"].shape[0], HC), dtype=torch.float32)
    golden_original_basis_gate(values, eps=eps)


def golden_masked_gate_compare(active):
    """Require exact ineligible gates/residuals alongside numeric prefix checks."""
    from golden.validation import ratio_allclose
    from .qkv_proj_rope import golden_prefix_compare

    numeric = golden_prefix_compare(active)
    exact = ratio_allclose(rtol=0.0, atol=0.0, max_error_ratio=0.0)

    def compare(actual, expected, **context):
        passed, detail = numeric(actual, expected, **context)
        if not passed:
            return passed, detail
        token_mask = context["inputs"].get("token_mask")
        if token_mask is None:
            token_mask = context["expected_outputs"]["token_mask"]
        masked = token_mask[:active] == 0
        if masked.any():
            return exact(actual[:active][masked], expected[:active][masked], **context)
        return True, ""

    return compare


def engram_int8_golden_cases():
    """Bound full signed INT8, group32 ties, row guards and TP shard offsets."""
    from functools import partial
    import torch
    from golden import ScalarSpec, TensorSpec
    from .qkv_proj_rope import golden_prefix_compare

    capacity, rows, row_start = 16, 19, 1_000_000_000
    table = (torch.arange(rows * WIDTH).reshape(rows, WIDTH) % 256 - 128).to(torch.int8)
    table[:, :8] = torch.tensor([-128, -1, 1, 127, 2, -2, 64, -64], dtype=torch.int8)
    group_scales = torch.tensor([1.00390625, 0.007843137, 0.33333334, 1e-4, 2.5, 0.0, 1.99609375, 0.03125])
    table_scales = group_scales.repeat(rows, 1)
    table_scales[1::2, 0] = 1.01171875
    choices = torch.tensor([-1, 0, rows - 1, rows, 1], dtype=torch.int32) + row_start
    ids = choices[torch.arange(capacity * COLS).reshape(capacity, COLS) % choices.numel()]
    for tp, active in ((1, 0), (1, 7), (2, 7), (4, 1), (4, 16)):
        columns = COLS // tp
        head_start = COLS - columns
        specs = [
            TensorSpec("staging_ids", [capacity, COLS], torch.int32, init_value=ids),
            TensorSpec("table", [rows, WIDTH], torch.int8, init_value=table),
            TensorSpec("table_scales", [rows, 8], torch.float32, init_value=table_scales),
            TensorSpec("lookup", [capacity, columns * WIDTH], torch.bfloat16, init_value=-12.5),
            ScalarSpec("active", torch.int32, active, compile_runtime=True),
        ]
        reference = partial(golden_int8_lookup, head_start=head_start, row_start=row_start)
        exact_lookup = golden_prefix_compare(active, rtol=0.0, atol=0.0)
        yield dict(name=f"int8-fused-tp{tp}-{active}",
                   fn=make_resident_int8_lookup(capacity, rows, columns, head_start, row_start),
                   specs=specs, golden_fn=reference,
                   compare_fn={"lookup": exact_lookup})


def validate_engram_int8(*, device=0, compile_only=False, platform="a2a3", sim=False):
    """Run the owning fused lookup and group32 Golden cases."""
    from golden import run

    for case in engram_int8_golden_cases():
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
    validate_engram_int8(device=args.device, compile_only=args.compile_only, platform=args.platform, sim=args.sim)

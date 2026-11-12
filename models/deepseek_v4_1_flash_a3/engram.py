# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Engram lookup, projection, gating, Torch goldens and validation."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"
import pypto.language as pl
import torch
from .qkv_proj_rope import make_bf16_projection_staged

D, HC, COLS, ROW_WIDTH = 5120, 4, 24, 256
K, N = COLS * ROW_WIDTH, (HC + 1) * D
TILE, KT, NT, DT = 16, 128, 256, 512


# Decoder aliases retain the standalone A2/A3 contract names while sharing
# the same dimensions as the prefill Engram implementation above.
HIDDEN, HASH_COLS, HEAD_DIM = D, COLS, ROW_WIDTH
LOOKUP_WIDTH, KV_WIDTH = K, N
M_TILE, GATE_TILE = TILE, DT


# model config
ENGRAM_ECOTECH_D, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_COLS, WIDTH = 5120, 4, 24, 256
ENGRAM_ECOTECH_K, ENGRAM_ECOTECH_N = ENGRAM_ECOTECH_COLS * WIDTH, (ENGRAM_ECOTECH_HC + 1) * ENGRAM_ECOTECH_D

# tiling
T_TILE, D_TILE, ROTATION_TILE = 16, 512, 32
RESTORE_M_TILE = 16


# tiling
ENGRAM_D_TILE = 256


def make_engram(capacity, table_rows):
    if capacity % TILE or table_rows < 1:
        raise ValueError("Engram needs padded token capacity and a nonempty resident table")

    @pl.jit
    def engram(
        hash_ids: pl.Tensor[[capacity, COLS], pl.INT32],
        table: pl.Tensor[[table_rows, ROW_WIDTH], pl.BF16],
        wkv: pl.Tensor[[K, N], pl.BF16],
        channel_weight: pl.Tensor[[HC, D], pl.FP32],
        hidden: pl.Tensor[[capacity, HC, D], pl.BF16],
        token_mask: pl.Tensor[[capacity], pl.INT32],
        lookup: pl.InOut[pl.Tensor[[capacity, K], pl.BF16]],
        kv: pl.InOut[pl.Tensor[[capacity, N], pl.BF16]],
        output: pl.InOut[pl.Tensor[[capacity, HC, D], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for token in pl.spmd(active, name_hint="a2_engram_lookup"):
            for column in pl.range(COLS):
                row = pl.read(hash_ids, [token, column])
                lookup[token : token + 1, column * ROW_WIDTH : (column + 1) * ROW_WIDTH] = pl.slice(
                    table, [1, ROW_WIDTH], [row, 0]
                )

        blocks = pl.system.available_cluster_count()
        for core in pl.spmd(blocks, name_hint="a2_engram_projection"):
            for task in pl.range(core, ((active + TILE - 1) // TILE) * (N // NT), blocks):
                t0 = (task // (N // NT)) * TILE
                n0 = (task % (N // NT)) * NT
                rows = pl.min(TILE, active - t0)
                a = pl.slice(lookup, [TILE, KT], [t0, 0], valid_shape=[rows, KT])
                b = wkv[:KT, n0 : n0 + NT]
                acc = pl.matmul(a, b, out_dtype=pl.FP32)
                for k0 in pl.pipeline(KT, K, KT, stage=2):
                    a = pl.slice(lookup, [TILE, KT], [t0, k0], valid_shape=[rows, KT])
                    b = wkv[k0 : k0 + KT, n0 : n0 + NT]
                    acc = pl.matmul_acc(acc, a, b)
                # ReplicatedLinear returns BF16 before the FP32 gate arithmetic.
                kv[t0 : t0 + TILE, n0 : n0 + NT] = pl.set_validshape(
                    pl.cast(acc, pl.BF16, mode="rint"), rows, NT
                )

        flat_hidden = pl.reshape(hidden, [capacity, HC * D])
        flat_output = pl.reshape(output, [capacity, HC * D])
        mask = pl.reshape(token_mask, [1, capacity])
        for task in pl.spmd(((active + TILE - 1) // TILE) * HC, name_hint="a2_engram_gate"):
            t0, copy = (task // HC) * TILE, task % HC
            rows = pl.min(TILE, active - t0)
            sx = pl.full([1, TILE], dtype=pl.FP32, value=0.0)
            sk = pl.full([1, TILE], dtype=pl.FP32, value=0.0)
            dot = pl.full([1, TILE], dtype=pl.FP32, value=0.0)
            for d0 in pl.range(0, D, DT):
                x = pl.cast(
                    pl.slice(flat_hidden, [TILE, DT], [t0, copy * D + d0], valid_shape=[rows, DT]), pl.FP32
                )
                key = pl.cast(pl.slice(kv, [TILE, DT], [t0, copy * D + d0], valid_shape=[rows, DT]), pl.FP32)
                weight = channel_weight[copy : copy + 1, d0 : d0 + DT]
                sx = pl.add(sx, pl.reshape(pl.row_sum(pl.mul(x, x)), [1, TILE]))
                sk = pl.add(sk, pl.reshape(pl.row_sum(pl.mul(key, key)), [1, TILE]))
                dot = pl.add(
                    dot, pl.reshape(pl.row_sum(pl.mul(pl.col_expand_mul(x, weight), key)), [1, TILE])
                )
            inv = pl.mul(
                pl.rsqrt(pl.add(pl.mul(sx, 1.0 / D), 1e-20)), pl.rsqrt(pl.add(pl.mul(sk, 1.0 / D), 1e-20))
            )
            dot = pl.mul(pl.mul(dot, inv), D**-0.5)
            magnitude = pl.sqrt(pl.maximum(pl.abs(dot), 1e-6))
            # Signed FP32 bits preserve -0 without A2-unsupported INT32 bitwise AND.
            bits = pl.cast(pl.reinterpret_view(dot, pl.INT32), pl.FP32)
            negative = pl.maximum(pl.minimum(bits, 0.0), -1.0)
            signed = pl.mul(magnitude, pl.add(pl.mul(negative, 2.0), 1.0))
            gate = pl.recip(pl.add(pl.exp(pl.neg(signed)), 1.0))
            eligible = pl.cast(pl.slice(mask, [1, TILE], [0, t0], valid_shape=[1, rows]), pl.FP32)
            gate = pl.reshape(pl.mul(gate, eligible), [TILE, 1])
            for d0 in pl.range(0, D, DT):
                x = pl.cast(
                    pl.slice(flat_hidden, [TILE, DT], [t0, copy * D + d0], valid_shape=[rows, DT]), pl.FP32
                )
                value = pl.cast(pl.slice(kv, [TILE, DT], [t0, HC * D + d0], valid_shape=[rows, DT]), pl.FP32)
                result = pl.cast(pl.add(x, pl.row_expand_mul(value, gate)), pl.BF16, mode="rint")
                flat_output[t0 : t0 + TILE, copy * D + d0 : copy * D + d0 + DT] = pl.set_validshape(
                    result, rows, DT
                )
        return lookup, kv, output

    return engram


def make_lookup(capacity: int, table_rows: int):
    @pl.jit
    def lookup(
        hash_ids: pl.Tensor[[capacity, HASH_COLS], pl.INT32],
        table: pl.Tensor[[table_rows, HEAD_DIM], pl.BF16],
        output: pl.InOut[pl.Tensor[[capacity, LOOKUP_WIDTH], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for token in pl.spmd(capacity, name_hint="dsv41_a2a3_engram_lookup"):
            if token < active:
                for column in pl.range(HASH_COLS):
                    row = pl.read(hash_ids, [token, column])
                    output[token : token + 1, column * HEAD_DIM : (column + 1) * HEAD_DIM] = table[
                        row : row + 1, :
                    ]
        return output

    return lookup


def make_gate(capacity: int, eps: float = 1.0e-20):
    @pl.jit
    def gate(
        hidden: pl.Tensor[[capacity, HC, HIDDEN], pl.BF16],
        kv: pl.Tensor[[capacity, KV_WIDTH], pl.BF16],
        channel_weight: pl.Tensor[[HC, HIDDEN], pl.FP32],
        token_mask: pl.Tensor[[capacity], pl.INT32],
        output: pl.InOut[pl.Tensor[[capacity, HC, HIDDEN], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        hidden_flat = pl.reshape(hidden, [capacity, HC * HIDDEN])
        output_flat = pl.reshape(output, [capacity, HC * HIDDEN])
        mask_flat = pl.reshape(token_mask, [1, capacity])
        for block in pl.spmd(((active + M_TILE - 1) // M_TILE) * HC, name_hint="dsv41_a2a3_engram_gate"):
            token = (block // HC) * M_TILE
            copy = block % HC
            rows = pl.min(M_TILE, active - token)
            source_rms = pl.full([1, M_TILE], dtype=pl.FP32, value=0.0)
            key_rms = pl.full([1, M_TILE], dtype=pl.FP32, value=0.0)
            dot = pl.full([1, M_TILE], dtype=pl.FP32, value=0.0)
            for d0 in pl.range(0, HIDDEN, GATE_TILE):
                x = pl.cast(
                    pl.slice(
                        hidden_flat,
                        [M_TILE, GATE_TILE],
                        [token, copy * HIDDEN + d0],
                        valid_shape=[rows, GATE_TILE],
                    ),
                    pl.FP32,
                )
                key = pl.cast(
                    pl.slice(
                        kv, [M_TILE, GATE_TILE], [token, copy * HIDDEN + d0], valid_shape=[rows, GATE_TILE]
                    ),
                    pl.FP32,
                )
                weight = channel_weight[copy : copy + 1, d0 : d0 + GATE_TILE]
                source_rms = pl.add(source_rms, pl.reshape(pl.row_sum(pl.mul(x, x)), [1, M_TILE]))
                key_rms = pl.add(key_rms, pl.reshape(pl.row_sum(pl.mul(key, key)), [1, M_TILE]))
                dot = pl.add(
                    dot, pl.reshape(pl.row_sum(pl.col_expand_mul(pl.mul(x, key), weight)), [1, M_TILE])
                )
            inv = pl.mul(
                pl.rsqrt(pl.add(pl.mul(source_rms, 1.0 / HIDDEN), eps)),
                pl.rsqrt(pl.add(pl.mul(key_rms, 1.0 / HIDDEN), eps)),
            )
            dot = pl.mul(pl.mul(dot, inv), HIDDEN**-0.5)
            magnitude = pl.sqrt(pl.maximum(pl.abs(dot), 1.0e-6))
            bits = pl.cast(pl.reinterpret_view(dot, pl.INT32), pl.FP32)
            negative = pl.maximum(pl.minimum(bits, 0.0), -1.0)
            signed = pl.mul(magnitude, pl.add(pl.mul(negative, 2.0), 1.0))
            gate_value = pl.recip(pl.add(pl.exp(pl.neg(signed)), 1.0))
            mask = pl.cast(pl.slice(mask_flat, [1, M_TILE], [0, token], valid_shape=[1, rows]), pl.FP32)
            gate_value = pl.reshape(pl.mul(gate_value, pl.reshape(mask, [1, M_TILE])), [M_TILE, 1])
            for d0 in pl.range(0, HIDDEN, GATE_TILE):
                x = pl.cast(
                    pl.slice(
                        hidden_flat,
                        [M_TILE, GATE_TILE],
                        [token, copy * HIDDEN + d0],
                        valid_shape=[rows, GATE_TILE],
                    ),
                    pl.FP32,
                )
                value = pl.cast(
                    pl.slice(
                        kv, [M_TILE, GATE_TILE], [token, HC * HIDDEN + d0], valid_shape=[rows, GATE_TILE]
                    ),
                    pl.FP32,
                )
                result = pl.cast(pl.add(x, pl.row_expand_mul(value, gate_value)), pl.BF16, mode="rint")
                output_flat[token : token + M_TILE, copy * HIDDEN + d0 : copy * HIDDEN + d0 + GATE_TILE] = (
                    pl.set_validshape(result, rows, GATE_TILE)
                )
        return output

    return gate


def make_resident_int8_lookup(capacity, table_rows, columns=ENGRAM_ECOTECH_COLS, head_start=0, row_start=0):
    """Fuse guarded INT8 lookup and group32 BF16 RNE in token-column SPMD tasks."""

    @pl.jit
    def lookup_rows(
        staging_ids: pl.Tensor[[capacity, ENGRAM_ECOTECH_COLS], pl.INT32],
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
                decoded = pl.full([1, 32], dtype=pl.BF16, value=0.0)
                if local_row >= 0:
                    if local_row < table_rows:
                        code_half = pl.cast(table[row : row + 1, group * 32 : group * 32 + 32], pl.FP16)
                        code_fp32 = pl.cast(code_half, pl.FP32)
                        scale = pl.read(table_scales, [row, group])
                        decoded = pl.cast(pl.mul(code_fp32, scale), pl.BF16, mode="rint")
                start = column * WIDTH + group * 32
                lookup[token : token + 1, start : start + 32] = decoded
        return lookup

    return lookup_rows


def make_original_basis_hidden(capacity):
    """Restore FP32 hidden with independent block32 FP32 matrix products."""

    @pl.jit.inline
    def restore(
        hidden: pl.Tensor[[capacity, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], pl.BF16],
        rotation_block: pl.Tensor[[ROTATION_TILE, ROTATION_TILE], pl.FP32],
        original: pl.Tensor[[capacity, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        flat_hidden = pl.reshape(
            hidden, [capacity * ENGRAM_ECOTECH_HC * (ENGRAM_ECOTECH_D // ROTATION_TILE), ROTATION_TILE]
        )
        flat_original = pl.reshape(
            original, [capacity * ENGRAM_ECOTECH_HC * (ENGRAM_ECOTECH_D // ROTATION_TILE), ROTATION_TILE]
        )
        blocks = active * ENGRAM_ECOTECH_HC * (ENGRAM_ECOTECH_D // ROTATION_TILE // RESTORE_M_TILE)
        for task in pl.spmd(blocks, name_hint="engram_ecotech_original_basis_cube"):
            row = task * RESTORE_M_TILE
            source_bf16 = flat_hidden[row : row + RESTORE_M_TILE, :]
            source_fp32 = pl.cast(source_bf16, pl.FP32)
            rotation_fp32 = rotation_block[:, :]
            restored = pl.matmul(source_fp32, rotation_fp32, out_dtype=pl.FP32, b_trans=True)
            flat_original[row : row + RESTORE_M_TILE, :] = pl.mul(restored, 1.0)
        return original

    return restore


def make_original_basis_gate(capacity, eps):
    """Compute the original-basis gate and add the already rotated value."""

    @pl.jit.inline
    def gate(
        hidden: pl.Tensor[[capacity, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], pl.BF16],
        original: pl.Tensor[[capacity, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], pl.FP32],
        kv: pl.Tensor[[capacity, ENGRAM_ECOTECH_N], pl.BF16],
        channel_weight: pl.Tensor[[ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], pl.FP32],
        token_mask: pl.Tensor[[capacity], pl.INT32],
        probabilities: pl.Tensor[[capacity, ENGRAM_ECOTECH_HC], pl.FP32],
        output: pl.Tensor[[capacity, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        flat_original = pl.reshape(original, [capacity, ENGRAM_ECOTECH_HC * ENGRAM_ECOTECH_D])
        flat_hidden = pl.reshape(hidden, [capacity, ENGRAM_ECOTECH_HC * ENGRAM_ECOTECH_D])
        flat_output = pl.reshape(output, [capacity, ENGRAM_ECOTECH_HC * ENGRAM_ECOTECH_D])
        mask = pl.reshape(token_mask, [1, capacity])
        for task in pl.spmd((active + T_TILE - 1) // T_TILE, name_hint="engram_ecotech_gate"):
            t0 = task * T_TILE
            rows = pl.min(T_TILE, active - t0)
            # One token-tile owner writes every probability cache line.
            for branch in pl.range(ENGRAM_ECOTECH_HC):
                sx = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
                sk = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
                dot = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
                for d0 in pl.range(0, ENGRAM_ECOTECH_D, D_TILE):
                    x = pl.slice(
                        flat_original,
                        [T_TILE, D_TILE],
                        [t0, branch * ENGRAM_ECOTECH_D + d0],
                        valid_shape=[rows, D_TILE],
                    )
                    key_bf16 = pl.slice(
                        kv, [T_TILE, D_TILE], [t0, branch * ENGRAM_ECOTECH_D + d0], valid_shape=[rows, D_TILE]
                    )
                    key = pl.cast(key_bf16, pl.FP32)
                    weight = channel_weight[branch : branch + 1, d0 : d0 + D_TILE]
                    sx = pl.add(sx, pl.reshape(pl.row_sum(pl.mul(x, x)), [1, T_TILE]))
                    sk = pl.add(sk, pl.reshape(pl.row_sum(pl.mul(key, key)), [1, T_TILE]))
                    weighted_x = pl.col_expand_mul(x, weight)
                    dot = pl.add(dot, pl.reshape(pl.row_sum(pl.mul(weighted_x, key)), [1, T_TILE]))
                inv_x = pl.rsqrt(pl.add(pl.mul(sx, 1.0 / ENGRAM_ECOTECH_D), eps), high_precision=True)
                inv_key = pl.rsqrt(pl.add(pl.mul(sk, 1.0 / ENGRAM_ECOTECH_D), eps), high_precision=True)
                inv = pl.mul(inv_x, inv_key)
                dot = pl.mul(pl.mul(dot, inv), ENGRAM_ECOTECH_D**-0.5)
                magnitude = pl.sqrt(pl.maximum(pl.abs(dot), 1e-6))
                bits = pl.cast(pl.reinterpret_view(dot, pl.INT32), pl.FP32)
                negative = pl.maximum(pl.minimum(bits, 0.0), -1.0)
                signed = pl.mul(magnitude, pl.add(pl.mul(negative, 2.0), 1.0))
                eligible = pl.cast(pl.slice(mask, [1, T_TILE], [0, t0], valid_shape=[1, rows]), pl.FP32)
                probability = pl.recip(pl.add(pl.exp(pl.neg(signed)), 1.0))
                probability = pl.reshape(pl.mul(probability, eligible), [T_TILE, 1])
                for row in pl.range(rows):
                    pl.write(probabilities, [t0 + row, branch], pl.read(probability, [row, 0]))
                for d0 in pl.range(0, ENGRAM_ECOTECH_D, D_TILE):
                    x_bf16 = pl.slice(
                        flat_hidden,
                        [T_TILE, D_TILE],
                        [t0, branch * ENGRAM_ECOTECH_D + d0],
                        valid_shape=[rows, D_TILE],
                    )
                    value_bf16 = pl.slice(
                        kv,
                        [T_TILE, D_TILE],
                        [t0, ENGRAM_ECOTECH_HC * ENGRAM_ECOTECH_D + d0],
                        valid_shape=[rows, D_TILE],
                    )
                    value = pl.cast(value_bf16, pl.FP32)
                    result = pl.add(pl.cast(x_bf16, pl.FP32), pl.row_expand_mul(value, probability))
                    result = pl.cast(result, pl.BF16, mode="rint")
                    start = branch * ENGRAM_ECOTECH_D + d0
                    flat_output[t0 : t0 + T_TILE, start : start + D_TILE] = pl.set_validshape(
                        result, rows, D_TILE
                    )
        return probabilities, output

    return gate


def make_engram_ecotech(capacity, table_rows, rms_norm_eps=1e-20):
    """Compose resident lookup, shared BF16 projection and rotated-residual gating."""
    if capacity < 1 or capacity % 16 or table_rows < 1 or rms_norm_eps <= 0:
        raise ValueError("Use a positive padded capacity, resident row count and source norm epsilon")
    gather = pl.jit.inline(make_resident_int8_lookup(capacity, table_rows)._func)
    project = pl.jit.inline(make_bf16_projection_staged(ENGRAM_ECOTECH_K, ENGRAM_ECOTECH_N)._func)
    restore = make_original_basis_hidden(capacity)
    gate = make_original_basis_gate(capacity, rms_norm_eps)

    @pl.jit
    def engram_ecotech(
        staging_ids: pl.Tensor[[capacity, ENGRAM_ECOTECH_COLS], pl.INT32],
        table: pl.Tensor[[table_rows, WIDTH], pl.INT8],
        table_scales: pl.Tensor[[table_rows, 8], pl.FP32],
        wkv: pl.Tensor[[ENGRAM_ECOTECH_K, ENGRAM_ECOTECH_N], pl.BF16],
        channel_weight: pl.Tensor[[ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], pl.FP32],
        hidden: pl.Tensor[[capacity, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], pl.BF16],
        rotation_block: pl.Tensor[[ROTATION_TILE, ROTATION_TILE], pl.FP32],
        token_mask: pl.Tensor[[capacity], pl.INT32],
        lookup: pl.InOut[pl.Tensor[[capacity, ENGRAM_ECOTECH_K], pl.BF16]],
        kv: pl.InOut[pl.Tensor[[capacity, ENGRAM_ECOTECH_N], pl.BF16]],
        original: pl.InOut[pl.Tensor[[capacity, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], pl.FP32]],
        output: pl.InOut[pl.Tensor[[capacity, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        gather(staging_ids, table, table_scales, lookup, active)
        project(lookup, wkv, kv, active)
        restore(hidden, rotation_block, original, active)
        probabilities = pl.create_tensor([capacity, ENGRAM_ECOTECH_HC], dtype=pl.FP32)
        gate(hidden, original, kv, channel_weight, token_mask, probabilities, output, active)
        return lookup, kv, original, output

    return engram_ecotech


def make_engram_ecotech_observed(capacity, table_rows, rms_norm_eps=1e-20):
    if capacity < 1 or capacity % 16 or table_rows < 1 or rms_norm_eps <= 0:
        raise ValueError("Use a positive padded capacity, resident row count and source norm epsilon")
    gather = pl.jit.inline(make_resident_int8_lookup(capacity, table_rows)._func)
    project = pl.jit.inline(make_bf16_projection_staged(ENGRAM_ECOTECH_K, ENGRAM_ECOTECH_N)._func)
    restore = make_original_basis_hidden(capacity)
    gate = make_original_basis_gate(capacity, rms_norm_eps)

    @pl.jit
    def engram_ecotech_observed(
        staging_ids: pl.InOut[pl.Tensor[[capacity, ENGRAM_ECOTECH_COLS], pl.INT32]],
        table: pl.InOut[pl.Tensor[[table_rows, WIDTH], pl.INT8]],
        table_scales: pl.InOut[pl.Tensor[[table_rows, 8], pl.FP32]],
        wkv: pl.InOut[pl.Tensor[[ENGRAM_ECOTECH_K, ENGRAM_ECOTECH_N], pl.BF16]],
        channel_weight: pl.InOut[pl.Tensor[[ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], pl.FP32]],
        hidden: pl.InOut[pl.Tensor[[capacity, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], pl.BF16]],
        rotation_block: pl.InOut[pl.Tensor[[ROTATION_TILE, ROTATION_TILE], pl.FP32]],
        token_mask: pl.InOut[pl.Tensor[[capacity], pl.INT32]],
        lookup: pl.InOut[pl.Tensor[[capacity, ENGRAM_ECOTECH_K], pl.BF16]],
        kv: pl.InOut[pl.Tensor[[capacity, ENGRAM_ECOTECH_N], pl.BF16]],
        original: pl.InOut[pl.Tensor[[capacity, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], pl.FP32]],
        probabilities: pl.InOut[pl.Tensor[[capacity, ENGRAM_ECOTECH_HC], pl.FP32]],
        output: pl.InOut[pl.Tensor[[capacity, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        gather(staging_ids, table, table_scales, lookup, active)
        project(lookup, wkv, kv, active)
        restore(hidden, rotation_block, original, active)
        gate(hidden, original, kv, channel_weight, token_mask, probabilities, output, active)
        return (
            staging_ids,
            table,
            table_scales,
            wkv,
            channel_weight,
            hidden,
            rotation_block,
            token_mask,
            lookup,
            kv,
            original,
            probabilities,
            output,
        )

    return engram_ecotech_observed


def make_prefill_engram_hash(capacity, vocab, layers, cache_rows, max_blocks, image_id, image_pad_id, pad_id):
    """Publish compressed slot history and compute both layers' exact INT64 rolling hashes."""

    @pl.jit
    def hash_history(
        token_ids: pl.Tensor[[capacity], pl.INT64],
        positions: pl.Tensor[[capacity], pl.INT64],
        query_start: pl.Tensor[[capacity + 1], pl.INT32],
        slots: pl.Tensor[[capacity], pl.INT64],
        block_table: pl.Tensor[[capacity, max_blocks], pl.INT32],
        lookback_ids: pl.Tensor[[capacity, 3], pl.INT64],
        token_map: pl.Tensor[[vocab], pl.INT32],
        multipliers: pl.Tensor[[layers, 4], pl.INT64],
        primes: pl.Tensor[[layers, ENGRAM_ECOTECH_COLS], pl.INT64],
        offsets: pl.Tensor[[layers, ENGRAM_ECOTECH_COLS], pl.INT64],
        cache: pl.InOut[pl.Tensor[[cache_rows], pl.INT32]],
        hashes: pl.InOut[pl.Tensor[[capacity, layers, ENGRAM_ECOTECH_COLS], pl.INT32]],
        token_mask: pl.InOut[pl.Tensor[[capacity], pl.INT32]],
        active: pl.Scalar[pl.INT32],
        requests: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(1, name_hint="prefill_engram_history_publish") as publish_tid:
            publish_block = pl.tile.get_block_idx()
            for token in pl.range(publish_block, active):
                token_id = pl.read(token_ids, [token])
                eligible = (token_id != image_id) and (token_id != image_pad_id)
                pl.write(token_mask, [token], pl.cast(eligible, pl.INT32))
                compressed = pl.read(token_map, [pl.cast(token_id, pl.INDEX)])
                if not eligible:
                    compressed = pl.cast(-1, pl.INT32)
                slot = pl.cast(pl.read(slots, [token]), pl.INDEX)
                if slot >= 0:
                    if slot < cache_rows:
                        pl.write(cache, [slot], compressed)
        for hash_block in pl.spmd(1, name_hint="prefill_engram_hash", deps=[publish_tid]):
            for task in pl.range(hash_block, active * layers):
                token, layer = task // layers, task % layers
                request: pl.Scalar[pl.INDEX] = 0
                for candidate in pl.range(requests):
                    query_begin = pl.read(query_start, [candidate])
                    end = pl.read(query_start, [candidate + 1])
                    if token >= query_begin:
                        if token < end:
                            request = candidate
                begin = pl.cast(pl.read(query_start, [request]), pl.INDEX)
                chunk_start = pl.read(positions, [begin])
                position = pl.read(positions, [token])
                rolling: pl.Scalar[pl.INT64] = 0
                blocked: pl.Scalar[pl.INT32] = 0
                for shift in pl.range(4):
                    history_position = position - shift
                    value: pl.Scalar[pl.INT32] = pad_id
                    source_dead: pl.Scalar[pl.INT32] = 0
                    if history_position >= chunk_start:
                        source_token = pl.max(token - shift, 0)
                        source_id = pl.read(token_ids, [source_token])
                        value = pl.read(token_map, [pl.cast(source_id, pl.INDEX)])
                        if source_id == image_id or source_id == image_pad_id:
                            source_dead = pl.cast(1, pl.INT32)
                    else:
                        history_column = pl.cast(chunk_start - 1 - history_position, pl.INDEX)
                        lookback_id: pl.Scalar[pl.INT64] = -1
                        if history_column >= 0:
                            if history_column < 3:
                                lookback_id = pl.read(lookback_ids, [request, history_column])
                        if lookback_id >= 0:
                            value = pl.read(token_map, [pl.cast(lookback_id, pl.INDEX)])
                            if lookback_id == image_id or lookback_id == image_pad_id:
                                source_dead = pl.cast(1, pl.INT32)
                        else:
                            bounded_position = pl.min(pl.max(history_position, 0), max_blocks * 128 - 1)
                            block_column = pl.cast(bounded_position // 128, pl.INDEX)
                            block = pl.read(block_table, [request, block_column])
                            history_slot = pl.cast(block, pl.INT64) * 128 + bounded_position % 128
                            bounded_slot = pl.cast(pl.min(pl.max(history_slot, 0), cache_rows - 1), pl.INDEX)
                            value = pl.read(cache, [bounded_slot])
                    if history_position < 0 or source_dead != 0 or value == -1:
                        blocked = pl.cast(1, pl.INT32)
                    if blocked != 0:
                        value = pl.cast(pad_id, pl.INT32)
                    multiplier = pl.read(multipliers, [layer, shift])
                    product = pl.cast(value, pl.INT64) * multiplier
                    rolling = rolling ^ product
                    if shift > 0:
                        for head in pl.range(8):
                            column = (shift - 1) * 8 + head
                            prime = pl.read(primes, [layer, column])
                            offset = pl.read(offsets, [layer, column])
                            row_id = pl.cast(rolling % prime + offset, pl.INT32)
                            pl.write(hashes, [token, layer, column], row_id)
        return cache, hashes, token_mask

    return hash_history


def make_prefill_engram_layout(capacity, tp):
    """Unpack TP column buckets and shard complete rows in original token order."""
    columns = ENGRAM_ECOTECH_COLS // tp
    local_width = columns * WIDTH

    @pl.jit
    def lookup_layout(
        packet: pl.Tensor[[tp * capacity, local_width], pl.BF16],
        token_mask: pl.Tensor[[capacity], pl.INT32],
        full: pl.InOut[pl.Tensor[[capacity, ENGRAM_ECOTECH_K], pl.BF16]],
        local: pl.InOut[pl.Tensor[[capacity, ENGRAM_ECOTECH_K], pl.BF16]],
        local_mask: pl.InOut[pl.Tensor[[capacity], pl.INT32]],
        physical: pl.Scalar[pl.INT32],
        active: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(active * ENGRAM_ECOTECH_COLS, name_hint="prefill_engram_tp_unpack") as unpack_tid:
            unpack_task = pl.tile.get_block_idx()
            unpack_token = unpack_task // ENGRAM_ECOTECH_COLS
            unpack_column = unpack_task % ENGRAM_ECOTECH_COLS
            unpack_peer = unpack_column // columns
            unpack_local_column = unpack_column % columns
            unpack_source = unpack_peer * capacity + unpack_token
            unpack_start = unpack_local_column * WIDTH
            unpack_values = pl.load(
                packet, [unpack_source, unpack_start], [1, WIDTH], target_memory=pl.MemorySpace.Vec
            )
            pl.store(unpack_values, [unpack_token, unpack_column * WIDTH], full)
        for shard_task in pl.spmd(
            physical * ENGRAM_ECOTECH_COLS, name_hint="prefill_engram_sp_shard", deps=[unpack_tid]
        ):
            shard_token = shard_task // ENGRAM_ECOTECH_COLS
            shard_column = shard_task % ENGRAM_ECOTECH_COLS
            shard_source = tp_rank * physical + shard_token
            shard_start = shard_column * WIDTH
            if shard_source < active:
                shard_values = pl.load(
                    full, [shard_source, shard_start], [1, WIDTH], target_memory=pl.MemorySpace.Vec
                )
                pl.store(shard_values, [shard_token, shard_start], local)
            else:
                shard_zeros = pl.tile.full([1, WIDTH], dtype=pl.BF16, value=0.0)
                pl.store(shard_zeros, [shard_token, shard_start], local)
        for mask_block in pl.spmd(1, name_hint="prefill_engram_mask_publish", deps=[unpack_tid]):
            for mask_token in pl.range(mask_block, physical):
                mask_source = tp_rank * physical + mask_token
                eligible: pl.Scalar[pl.INT32] = 0
                if mask_source < active:
                    eligible = pl.read(token_mask, [mask_source])
                pl.write(local_mask, [mask_token], eligible)
        return full, local, local_mask

    return lookup_layout


def make_prefill_engram_gate(capacity, eps):
    """Restore hidden and apply FP32 channel products and the original-basis gate."""
    restore = make_original_basis_hidden(capacity)
    gate = make_original_basis_gate(capacity, eps)

    @pl.jit
    def original_basis_gate(
        hidden: pl.Tensor[[capacity, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], pl.BF16],
        projected: pl.Tensor[[capacity, ENGRAM_ECOTECH_N], pl.BF16],
        q_weight: pl.Tensor[[ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], pl.BF16],
        k_weight: pl.Tensor[[ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], pl.BF16],
        rotation: pl.Tensor[[32, 32], pl.FP32],
        token_mask: pl.Tensor[[capacity], pl.INT32],
        original: pl.InOut[pl.Tensor[[capacity, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], pl.FP32]],
        probabilities: pl.InOut[pl.Tensor[[capacity, ENGRAM_ECOTECH_HC], pl.FP32]],
        output: pl.InOut[pl.Tensor[[capacity, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        channel_weight = pl.create_tensor([ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], dtype=pl.FP32)
        for task in pl.spmd(
            ENGRAM_ECOTECH_HC * (ENGRAM_ECOTECH_D // ENGRAM_D_TILE), name_hint="prefill_engram_channel_weight"
        ):
            branch, start = (
                task // (ENGRAM_ECOTECH_D // ENGRAM_D_TILE),
                task % (ENGRAM_ECOTECH_D // ENGRAM_D_TILE) * ENGRAM_D_TILE,
            )
            q = pl.cast(q_weight[branch : branch + 1, start : start + ENGRAM_D_TILE], pl.FP32)
            k = pl.cast(k_weight[branch : branch + 1, start : start + ENGRAM_D_TILE], pl.FP32)
            channel_weight[branch : branch + 1, start : start + ENGRAM_D_TILE] = pl.mul(q, k)
        restore(hidden, rotation, original, active)
        gate(hidden, original, projected, channel_weight, token_mask, probabilities, output, active)
        return original, probabilities, output

    return original_basis_gate


def make_prefill_engram_entries(capacity, constants, config, shards, cache_rows, max_blocks, tp=4):
    """Construct bounded hash/layout diagnostics and ready-row projection/gate entries."""
    if tp not in (1, 2, 4) or len(shards) != tp or ENGRAM_ECOTECH_COLS % tp:
        raise ValueError("Require complete TP1/2/4 Engram column ownership")
    columns = ENGRAM_ECOTECH_COLS // tp
    entries = {}
    for rank, shard in enumerate(shards):
        if shard["head_range"] != [rank * columns, (rank + 1) * columns]:
            raise ValueError("Require contiguous whole-column Engram shards")
    image_id = config["image_token_id"]
    image_pad_id = config.get("image_pad_token_id", image_id + 1)
    hash_args = (capacity, len(constants["token_map"]), len(constants["layer_ids"]), cache_rows, max_blocks)
    entries["engram_hash"] = make_prefill_engram_hash(*hash_args, image_id, image_pad_id, constants["pad_id"])
    entries["engram_layout"] = make_prefill_engram_layout(capacity, tp)
    entries["engram_projection"] = make_bf16_projection_staged(
        ENGRAM_ECOTECH_K, ENGRAM_ECOTECH_N, capacity=capacity
    )
    entries["engram_gate"] = make_prefill_engram_gate(capacity, config["rms_norm_eps"])
    return entries


def engram_reference(
    hash_ids: torch.Tensor,
    table: torch.Tensor,
    wkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    hidden: torch.Tensor,
    token_mask: torch.Tensor,
    eps: float = 1.0e-20,
) -> dict[str, torch.Tensor]:
    """CPU reference mirroring vLLM-Ascend's unrotated decoder Engram path."""

    lookup = table[hash_ids.long()].reshape(hash_ids.shape[0], LOOKUP_WIDTH)
    kv = torch.matmul(lookup.float(), wkv.float()).to(torch.bfloat16)
    key = kv[:, : HC * HIDDEN].reshape(kv.shape[0], HC, HIDDEN).float()
    value = kv[:, HC * HIDDEN :]
    source = hidden.float()
    channel_weight = q_weight.float() * k_weight.float()
    source_rstd = torch.rsqrt(source.square().mean(-1, keepdim=True) + eps)
    key_rstd = torch.rsqrt(key.square().mean(-1, keepdim=True) + eps)
    dot = (source * channel_weight.reshape(1, HC, HIDDEN) * key).sum(-1, keepdim=True)
    dot = dot * source_rstd * key_rstd * (HIDDEN**-0.5)
    signed_root = torch.copysign(dot.abs().clamp_min(1.0e-6).sqrt(), dot)
    gate = torch.sigmoid(signed_root) * token_mask.reshape(-1, 1, 1).float()
    output = (source + gate * value.float().unsqueeze(1)).to(torch.bfloat16)
    return {"lookup": lookup, "kv": kv, "output": output}


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
    ids = tensors["staging_ids"][:active, head_start : head_start + columns]
    codes, scales = reference_int8_rows(ids, tensors["table"], tensors["table_scales"], row_start=row_start)
    values = codes.float().reshape(active, columns, 8, 32) * scales.reshape(active, columns, 8, 1)
    tensors["lookup"][:active] = values.reshape(active, width).to(tensors["lookup"].dtype)


def golden_original_basis_hidden(tensors):
    """Restore block32 hidden by independent FP32 matrix multiplication."""
    active = int(tensors["active"])
    hidden = tensors["hidden"][:active].float().reshape(-1, ROTATION_TILE)
    restored = hidden @ tensors["rotation_block"].float().T
    tensors["original"][:active] = restored.reshape(active, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D)


def golden_original_basis_gate(tensors, *, eps=1e-20):
    """Evaluate RMS-normalized channel products, signed-root sigmoid and residual."""
    import torch

    active = int(tensors["active"])
    original = tensors["original"][:active].float()
    key = (
        tensors["kv"][:active, : ENGRAM_ECOTECH_HC * ENGRAM_ECOTECH_D]
        .float()
        .reshape(active, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D)
    )
    inv_x = torch.rsqrt(original.square().mean(-1) + eps)
    inv_key = torch.rsqrt(key.square().mean(-1) + eps)
    dot = (original * tensors["channel_weight"].float() * key).sum(-1)
    dot = dot * inv_x * inv_key * ENGRAM_ECOTECH_D**-0.5
    signed = torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot)
    probability = torch.sigmoid(signed) * tensors["token_mask"][:active, None].float()
    tensors["probabilities"][:active] = probability
    value = tensors["kv"][:active, ENGRAM_ECOTECH_HC * ENGRAM_ECOTECH_D :].float()[:, None, :]
    result = tensors["hidden"][:active].float() + probability[:, :, None] * value
    tensors["output"][:active] = result.to(tensors["output"].dtype)


def golden_engram_ecotech(tensors, *, eps=1e-20):
    """Compose independent INT8, projection, original and gate references."""
    import torch
    from .qkv_proj_rope import golden_bf16_projection

    active = int(tensors["active"])
    ids = tensors["staging_ids"][:active]
    codes, scales = reference_int8_rows(ids, tensors["table"], tensors["table_scales"])
    decoded = codes.float().reshape(active, ENGRAM_ECOTECH_COLS, 8, 32) * scales.reshape(
        active, ENGRAM_ECOTECH_COLS, 8, 1
    )
    tensors["lookup"][:active] = decoded.reshape(active, ENGRAM_ECOTECH_K).to(tensors["lookup"].dtype)
    projection = dict(
        source=tensors["lookup"], weight=tensors["wkv"], output=tensors["kv"], num_tokens=active
    )
    golden_bf16_projection(projection)
    golden_original_basis_hidden(tensors)
    values = dict(tensors)
    if "probabilities" not in values:
        values["probabilities"] = torch.empty(
            (tensors["hidden"].shape[0], ENGRAM_ECOTECH_HC), dtype=torch.float32
        )
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


def golden_prefill_engram_hash(tensors, *, image_id, image_pad_id, pad_id):
    """Use the independent host history/hash contract on bounded inputs."""
    from .engram_host import HashLayout, hash_ids, update_slot_history

    active, requests = int(tensors["active"]), int(tensors["requests"])
    tokens = tensors["token_ids"][:active]
    dead = (tokens == image_id) | (tokens == image_pad_id)
    tensors["token_mask"][:active] = (~dead).to(tensors["token_mask"].dtype)
    update_slot_history(tokens, tensors["token_map"], dead, tensors["slots"][:active], tensors["cache"])
    if not active:
        return
    lookback = tensors["lookback_ids"][:requests]
    lookback_dead = (lookback == image_id) | (lookback == image_pad_id)
    layers = tensors["multipliers"].shape[0]
    layout = HashLayout(
        layer_ids=tuple(range(layers)),
        multipliers=tensors["multipliers"],
        primes=tensors["primes"],
        offsets=tensors["offsets"],
        max_ngram_size=4,
        n_heads=8,
        compressed_vocab_size=int(tensors["token_map"].max()) + 1,
        pad_token_id=pad_id,
        vocab_size=tensors["token_map"].numel(),
    )
    result = hash_ids(
        tokens,
        tensors["token_map"],
        tensors["positions"][:active],
        tensors["query_start"][: requests + 1],
        layout=layout,
        dead_mask=dead,
        lookback_token_ids=lookback,
        lookback_dead_mask=lookback_dead,
        hash_cache=tensors["cache"],
        block_table=tensors["block_table"][:requests],
    )
    tensors["hashes"][:active] = result


def golden_prefill_engram_layout(tensors):
    """Evaluate TP column concatenation and contiguous SP token selection."""
    capacity = tensors["full"].shape[0]
    tp = tensors["packet"].shape[0] // capacity
    active, physical, rank = (int(tensors[name]) for name in ("active", "physical", "tp_rank"))
    buckets = tensors["packet"].reshape(tp, capacity, ENGRAM_ECOTECH_COLS // tp, WIDTH)
    full = buckets.permute(1, 0, 2, 3).reshape(capacity, ENGRAM_ECOTECH_K)
    tensors["full"][:active] = full[:active]
    tensors["local"][:physical] = 0
    tensors["local_mask"][:physical] = 0
    start = rank * physical
    valid = max(0, min(physical, active - start))
    tensors["local"][:valid] = full[start : start + valid]
    tensors["local_mask"][:valid] = tensors["token_mask"][start : start + valid]


def golden_prefill_engram_gate(tensors, *, eps=1e-20):
    """Compose owning restore/gate goldens with FP32 q*k channel products."""

    values = dict(tensors, kv=tensors["projected"], rotation_block=tensors["rotation"])
    values["channel_weight"] = tensors["q_weight"].float() * tensors["k_weight"].float()
    golden_original_basis_hidden(values)
    golden_original_basis_gate(values, eps=eps)


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
    ids = choices[
        torch.arange(capacity * ENGRAM_ECOTECH_COLS).reshape(capacity, ENGRAM_ECOTECH_COLS) % choices.numel()
    ]
    for tp, active in ((1, 0), (1, 7), (2, 7), (4, 1), (4, 16)):
        columns = ENGRAM_ECOTECH_COLS // tp
        head_start = ENGRAM_ECOTECH_COLS - columns
        specs = [
            TensorSpec("staging_ids", [capacity, ENGRAM_ECOTECH_COLS], torch.int32, init_value=ids),
            TensorSpec("table", [rows, WIDTH], torch.int8, init_value=table),
            TensorSpec("table_scales", [rows, 8], torch.float32, init_value=table_scales),
            TensorSpec("lookup", [capacity, columns * WIDTH], torch.bfloat16, init_value=-12.5),
            ScalarSpec("active", torch.int32, active, compile_runtime=True),
        ]
        reference = partial(golden_int8_lookup, head_start=head_start, row_start=row_start)
        exact_lookup = golden_prefix_compare(active, rtol=0.0, atol=0.0)
        yield dict(
            name=f"int8-fused-tp{tp}-{active}",
            fn=make_resident_int8_lookup(capacity, rows, columns, head_start, row_start),
            specs=specs,
            golden_fn=reference,
            compare_fn={"lookup": exact_lookup},
        )


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


def engram_observed_golden_cases():
    """Bound full-width resident composition and exact readonly input readback."""
    import torch
    from golden import ScalarSpec, TensorSpec
    from golden.validation import ratio_allclose
    from .qkv_proj_rope import golden_prefix_compare

    capacity, rows, active = 16, 19, 3
    generator = torch.Generator().manual_seed(733)
    ids = (
        torch.arange(capacity * ENGRAM_ECOTECH_COLS).reshape(capacity, ENGRAM_ECOTECH_COLS) % (rows + 2) - 1
    ).to(torch.int32)
    table = (torch.arange(rows * WIDTH).reshape(rows, WIDTH) % 256 - 128).to(torch.int8)
    scales = torch.empty((rows, 8), dtype=torch.float32).uniform_(0.001, 0.02, generator=generator)

    def wkv():
        weight_generator = torch.Generator().manual_seed(734)
        return torch.empty((ENGRAM_ECOTECH_K, ENGRAM_ECOTECH_N), dtype=torch.bfloat16).uniform_(
            -0.02, 0.02, generator=weight_generator
        )

    channels = torch.empty((ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D), dtype=torch.float32).uniform_(
        -1.0, 1.0, generator=generator
    )
    hidden = torch.empty((capacity, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D), dtype=torch.bfloat16).uniform_(
        -0.5, 0.5, generator=generator
    )
    rotation = torch.linalg.qr(torch.randn((32, 32), generator=generator)).Q.contiguous()
    mask = (torch.arange(capacity) % 3 != 1).to(torch.int32)
    specs = [
        TensorSpec("staging_ids", [capacity, ENGRAM_ECOTECH_COLS], torch.int32, init_value=ids),
        TensorSpec("table", [rows, WIDTH], torch.int8, init_value=table),
        TensorSpec("table_scales", [rows, 8], torch.float32, init_value=scales),
        TensorSpec("wkv", [ENGRAM_ECOTECH_K, ENGRAM_ECOTECH_N], torch.bfloat16, init_value=wkv),
        TensorSpec(
            "channel_weight", [ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], torch.float32, init_value=channels
        ),
        TensorSpec(
            "hidden", [capacity, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], torch.bfloat16, init_value=hidden
        ),
        TensorSpec("rotation_block", [32, 32], torch.float32, init_value=rotation),
        TensorSpec("token_mask", [capacity], torch.int32, init_value=mask),
        TensorSpec("lookup", [capacity, ENGRAM_ECOTECH_K], torch.bfloat16, init_value=-12.5),
        TensorSpec("kv", [capacity, ENGRAM_ECOTECH_N], torch.bfloat16, init_value=-12.5),
        TensorSpec(
            "original", [capacity, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], torch.float32, init_value=-12.5
        ),
        TensorSpec("probabilities", [capacity, ENGRAM_ECOTECH_HC], torch.float32, init_value=-12.5),
        TensorSpec(
            "output", [capacity, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], torch.bfloat16, init_value=-12.5
        ),
        ScalarSpec("active", torch.int32, active, compile_runtime=True),
    ]
    readonly = (
        "staging_ids",
        "table",
        "table_scales",
        "wkv",
        "channel_weight",
        "hidden",
        "rotation_block",
        "token_mask",
    )
    exact = ratio_allclose(rtol=0.0, atol=0.0, max_error_ratio=0.0)
    compare = {name: exact for name in readonly}
    compare["lookup"] = golden_prefix_compare(active, rtol=0.0, atol=0.0)
    compare.update({name: golden_prefix_compare(active) for name in ("kv", "original")})
    compare.update({name: golden_masked_gate_compare(active) for name in ("probabilities", "output")})
    yield dict(
        name="engram-observed-readonly-3",
        fn=make_engram_ecotech_observed(capacity, rows),
        specs=specs,
        golden_fn=golden_engram_ecotech,
        compare_fn=compare,
    )


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


def prefill_engram_hash_golden_cases():
    """Bound empty, dead-token, request, lookback/cache and page-edge history."""
    from functools import partial
    import torch
    from golden import ScalarSpec, TensorSpec
    from golden.validation import ratio_allclose

    capacity, vocab, layers, cache_rows, max_blocks = 16, 32, 2, 512, 3
    token_map = torch.arange(vocab, dtype=torch.int32) * 3 % 23
    multipliers = (2**40 + torch.tensor([[3, 5, 7, 11], [13, 17, 19, 23]])).to(torch.int64)
    primes = torch.tensor([97, 101] * 12, dtype=torch.int64).repeat(layers, 1)
    offsets = (
        torch.arange(layers * ENGRAM_ECOTECH_COLS, dtype=torch.int64).reshape(layers, ENGRAM_ECOTECH_COLS)
        * 103
    )
    reference = partial(golden_prefill_engram_hash, image_id=30, image_pad_id=31, pad_id=2)
    for active in (0, 1, 7, 16):
        requests = 2 if active == 7 else int(active > 0)
        tokens = torch.tensor([3, 30, 5, 6, 31, 8, 9] + list(range(10, 19)), dtype=torch.int64)
        positions = torch.arange(capacity, dtype=torch.int64)
        query = torch.full((capacity + 1,), active, dtype=torch.int32)
        query[0] = 0
        slots = torch.arange(capacity, dtype=torch.int64)
        if active == 7:
            query[1] = 3
            positions[:7] = torch.tensor([127, 128, 129, 4, 5, 6, 7])
            slots[:7] = torch.tensor([255, 256, 257, -1, cache_rows + 4, 133, 134])
        block_table = torch.tensor([1, 2, 0], dtype=torch.int32).repeat(capacity, 1)
        lookback = torch.full((capacity, 3), -1, dtype=torch.int64)
        lookback[0] = torch.tensor([7, 30, -1])
        lookback[1] = torch.tensor([-1, 9, 10])
        cache = torch.arange(cache_rows, dtype=torch.int32) % 23
        cache[131] = -1
        specs = [
            TensorSpec("token_ids", [capacity], torch.int64, init_value=tokens),
            TensorSpec("positions", [capacity], torch.int64, init_value=positions),
            TensorSpec("query_start", [capacity + 1], torch.int32, init_value=query),
            TensorSpec("slots", [capacity], torch.int64, init_value=slots),
            TensorSpec("block_table", [capacity, max_blocks], torch.int32, init_value=block_table),
            TensorSpec("lookback_ids", [capacity, 3], torch.int64, init_value=lookback),
            TensorSpec("token_map", [vocab], torch.int32, init_value=token_map),
            TensorSpec("multipliers", [layers, 4], torch.int64, init_value=multipliers),
            TensorSpec("primes", [layers, ENGRAM_ECOTECH_COLS], torch.int64, init_value=primes),
            TensorSpec("offsets", [layers, ENGRAM_ECOTECH_COLS], torch.int64, init_value=offsets),
            TensorSpec("cache", [cache_rows], torch.int32, init_value=cache),
            TensorSpec("hashes", [capacity, layers, ENGRAM_ECOTECH_COLS], torch.int32, init_value=-23),
            TensorSpec("token_mask", [capacity], torch.int32, init_value=-23),
            ScalarSpec("active", torch.int32, active, compile_runtime=True),
            ScalarSpec("requests", torch.int32, requests, compile_runtime=True),
        ]
        exact = ratio_allclose(rtol=0.0, atol=0.0, max_error_ratio=0.0)
        fn = make_prefill_engram_hash(capacity, vocab, layers, cache_rows, max_blocks, 30, 31, 2)
        yield dict(
            name=f"hash-history-{active}",
            fn=fn,
            specs=specs,
            golden_fn=reference,
            compare_fn={name: exact for name in ("cache", "hashes", "token_mask")},
        )


def prefill_engram_layout_golden_cases():
    """Bound TP1/2/4 column packets, empty owners and masked/inactive SP rows."""
    import torch
    from golden import ScalarSpec, TensorSpec
    from golden.validation import ratio_allclose

    capacity = 16
    for tp in (1, 2, 4):
        width = ENGRAM_ECOTECH_K // tp
        packet = (torch.arange(capacity * ENGRAM_ECOTECH_K).reshape(tp * capacity, width) % 251 - 125).to(
            torch.bfloat16
        )
        mask = (torch.arange(capacity) % 3 != 1).to(torch.int32)
        for active, rank in ((0, 0), (1, tp - 1), (7, tp - 1), (16, 0)):
            physical = max(1, (active + tp - 1) // tp)
            specs = [
                TensorSpec("packet", [tp * capacity, width], torch.bfloat16, init_value=packet),
                TensorSpec("token_mask", [capacity], torch.int32, init_value=mask),
                TensorSpec("full", [capacity, ENGRAM_ECOTECH_K], torch.bfloat16, init_value=-12.5),
                TensorSpec("local", [capacity, ENGRAM_ECOTECH_K], torch.bfloat16, init_value=-12.5),
                TensorSpec("local_mask", [capacity], torch.int32, init_value=-23),
                ScalarSpec("physical", torch.int32, physical, compile_runtime=True),
                ScalarSpec("active", torch.int32, active, compile_runtime=True),
                ScalarSpec("tp_rank", torch.int32, rank, compile_runtime=True),
            ]
            exact = ratio_allclose(rtol=0.0, atol=0.0, max_error_ratio=0.0)
            yield dict(
                name=f"layout-tp{tp}-{active}-rank{rank}",
                fn=make_prefill_engram_layout(capacity, tp),
                specs=specs,
                golden_fn=golden_prefill_engram_layout,
                compare_fn={name: exact for name in ("full", "local", "local_mask")},
            )


def prefill_engram_gate_golden_cases():
    """Bound original/probability readbacks, signed/zero gates and inactive tails."""
    import torch
    from golden import ScalarSpec, TensorSpec
    from .qkv_proj_rope import golden_prefix_compare

    capacity = 16
    generator = torch.Generator().manual_seed(732)
    hidden = torch.empty((capacity, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D), dtype=torch.bfloat16).uniform_(
        -0.5, 0.5, generator=generator
    )
    projected = torch.empty((capacity, ENGRAM_ECOTECH_N), dtype=torch.bfloat16).uniform_(
        -0.5, 0.5, generator=generator
    )
    q = torch.empty((ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D), dtype=torch.bfloat16).uniform_(
        -1.0, 1.0, generator=generator
    )
    k = torch.empty((ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D), dtype=torch.bfloat16).uniform_(
        -1.0, 1.0, generator=generator
    )
    rotation = torch.linalg.qr(torch.randn((32, 32), generator=generator)).Q.contiguous()
    projected[0, :ENGRAM_ECOTECH_D] = 0
    hidden[0, 1] = 0
    mask = (torch.arange(capacity) % 3 != 1).to(torch.int32)
    for active in (0, 1, 7, 16):
        specs = [
            TensorSpec(
                "hidden", [capacity, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], torch.bfloat16, init_value=hidden
            ),
            TensorSpec("projected", [capacity, ENGRAM_ECOTECH_N], torch.bfloat16, init_value=projected),
            TensorSpec("q_weight", [ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], torch.bfloat16, init_value=q),
            TensorSpec("k_weight", [ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], torch.bfloat16, init_value=k),
            TensorSpec("rotation", [32, 32], torch.float32, init_value=rotation),
            TensorSpec("token_mask", [capacity], torch.int32, init_value=mask),
            TensorSpec(
                "original", [capacity, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], torch.float32, init_value=-12.5
            ),
            TensorSpec("probabilities", [capacity, ENGRAM_ECOTECH_HC], torch.float32, init_value=-12.5),
            TensorSpec(
                "output", [capacity, ENGRAM_ECOTECH_HC, ENGRAM_ECOTECH_D], torch.bfloat16, init_value=-12.5
            ),
            ScalarSpec("active", torch.int32, active, compile_runtime=True),
        ]
        compare = {"original": golden_prefix_compare(active)}
        compare.update({name: golden_masked_gate_compare(active) for name in ("probabilities", "output")})
        yield dict(
            name=f"gate-original-probability-{active}",
            fn=make_prefill_engram_gate(capacity, 1e-20),
            specs=specs,
            golden_fn=golden_prefill_engram_gate,
            compare_fn=compare,
        )


def validate_prefill_engram(*, device=0, compile_only=False, platform="a2a3", sim=False):
    """Run the owning hash, TP/SP layout and original-basis Golden cases."""
    from itertools import chain
    from golden import run

    cases = chain(
        prefill_engram_hash_golden_cases(),
        prefill_engram_layout_golden_cases(),
        prefill_engram_gate_golden_cases(),
    )
    for case in cases:
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
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    options = vars(parser.parse_args())
    validate_engram_int8(**options)
    validate_engram_observed(**options)
    validate_prefill_engram(**options)

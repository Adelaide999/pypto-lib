# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-card BF16 Engram in the unrotated checkpoint basis.

The caller supplies resident table rows and their indices. Host hashing,
large-table offload and cross-rank gathers remain outside this kernel.
"""
import torch
import pypto.language as pl

D, HC, COLS, ROW_WIDTH = 5120, 4, 24, 256
K, N = COLS * ROW_WIDTH, (HC + 1) * D
TILE, KT, NT, DT = 16, 128, 256, 512


def make_engram(capacity, table_rows):
    if capacity % TILE or table_rows < 1:
        raise ValueError('Engram needs padded token capacity and a nonempty resident table')

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
        for token in pl.spmd(active, name_hint='a2_engram_lookup'):
            for column in pl.range(COLS):
                row = pl.read(hash_ids, [token, column])
                lookup[token:token + 1, column * ROW_WIDTH:(column + 1) * ROW_WIDTH] = pl.slice(table, [1, ROW_WIDTH], [row, 0])

        blocks = pl.system.available_cluster_count()
        for core in pl.spmd(blocks, name_hint='a2_engram_projection'):
            for task in pl.range(core, ((active + TILE - 1) // TILE) * (N // NT), blocks):
                t0 = (task // (N // NT)) * TILE
                n0 = (task % (N // NT)) * NT
                rows = pl.min(TILE, active - t0)
                a = pl.slice(lookup, [TILE, KT], [t0, 0], valid_shape=[rows, KT])
                b = wkv[:KT, n0:n0 + NT]
                acc = pl.matmul(a, b, out_dtype=pl.FP32)
                for k0 in pl.pipeline(KT, K, KT, stage=2):
                    a = pl.slice(lookup, [TILE, KT], [t0, k0], valid_shape=[rows, KT])
                    b = wkv[k0:k0 + KT, n0:n0 + NT]
                    acc = pl.matmul_acc(acc, a, b)
                # ReplicatedLinear returns BF16 before the FP32 gate arithmetic.
                kv[t0:t0 + TILE, n0:n0 + NT] = pl.set_validshape(pl.cast(acc, pl.BF16, mode='rint'), rows, NT)

        flat_hidden = pl.reshape(hidden, [capacity, HC * D])
        flat_output = pl.reshape(output, [capacity, HC * D])
        mask = pl.reshape(token_mask, [1, capacity])
        for task in pl.spmd(((active + TILE - 1) // TILE) * HC, name_hint='a2_engram_gate'):
            t0, copy = (task // HC) * TILE, task % HC
            rows = pl.min(TILE, active - t0)
            sx = pl.full([1, TILE], dtype=pl.FP32, value=0.)
            sk = pl.full([1, TILE], dtype=pl.FP32, value=0.)
            dot = pl.full([1, TILE], dtype=pl.FP32, value=0.)
            for d0 in pl.range(0, D, DT):
                x = pl.cast(pl.slice(flat_hidden, [TILE, DT], [t0, copy * D + d0], valid_shape=[rows, DT]), pl.FP32)
                key = pl.cast(pl.slice(kv, [TILE, DT], [t0, copy * D + d0], valid_shape=[rows, DT]), pl.FP32)
                weight = channel_weight[copy:copy + 1, d0:d0 + DT]
                sx = pl.add(sx, pl.reshape(pl.row_sum(pl.mul(x, x)), [1, TILE]))
                sk = pl.add(sk, pl.reshape(pl.row_sum(pl.mul(key, key)), [1, TILE]))
                dot = pl.add(dot, pl.reshape(pl.row_sum(pl.mul(pl.col_expand_mul(x, weight), key)), [1, TILE]))
            inv = pl.mul(pl.rsqrt(pl.add(pl.mul(sx, 1. / D), 1e-20)), pl.rsqrt(pl.add(pl.mul(sk, 1. / D), 1e-20)))
            dot = pl.mul(pl.mul(dot, inv), D**-0.5)
            magnitude = pl.sqrt(pl.maximum(pl.abs(dot), 1e-6))
            # Signed FP32 bits preserve -0 without A2-unsupported INT32 bitwise AND.
            bits = pl.cast(pl.reinterpret_view(dot, pl.INT32), pl.FP32)
            negative = pl.maximum(pl.minimum(bits, 0.), -1.)
            signed = pl.mul(magnitude, pl.add(pl.mul(negative, 2.), 1.))
            gate = pl.recip(pl.add(pl.exp(pl.neg(signed)), 1.))
            eligible = pl.cast(pl.slice(mask, [1, TILE], [0, t0], valid_shape=[1, rows]), pl.FP32)
            gate = pl.reshape(pl.mul(gate, eligible), [TILE, 1])
            for d0 in pl.range(0, D, DT):
                x = pl.cast(pl.slice(flat_hidden, [TILE, DT], [t0, copy * D + d0], valid_shape=[rows, DT]), pl.FP32)
                value = pl.cast(pl.slice(kv, [TILE, DT], [t0, HC * D + d0], valid_shape=[rows, DT]), pl.FP32)
                result = pl.cast(pl.add(x, pl.row_expand_mul(value, gate)), pl.BF16, mode='rint')
                flat_output[t0:t0 + TILE, copy * D + d0:copy * D + d0 + DT] = pl.set_validshape(result, rows, DT)
        return lookup, kv, output

    return engram


# Decoder aliases retain the standalone A2/A3 contract names while sharing
# the same dimensions as the prefill Engram implementation above.
HIDDEN, HASH_COLS, HEAD_DIM = D, COLS, ROW_WIDTH
LOOKUP_WIDTH, KV_WIDTH = K, N
M_TILE, GATE_TILE = TILE, DT


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
    key = kv[:, :HC * HIDDEN].reshape(kv.shape[0], HC, HIDDEN).float()
    value = kv[:, HC * HIDDEN:]
    source = hidden.float()
    channel_weight = q_weight.float() * k_weight.float()
    source_rstd = torch.rsqrt(source.square().mean(-1, keepdim=True) + eps)
    key_rstd = torch.rsqrt(key.square().mean(-1, keepdim=True) + eps)
    dot = (source * channel_weight.reshape(1, HC, HIDDEN) * key).sum(-1, keepdim=True)
    dot = dot * source_rstd * key_rstd * (HIDDEN ** -0.5)
    signed_root = torch.copysign(dot.abs().clamp_min(1.0e-6).sqrt(), dot)
    gate = torch.sigmoid(signed_root) * token_mask.reshape(-1, 1, 1).float()
    output = (source + gate * value.float().unsqueeze(1)).to(torch.bfloat16)
    return {"lookup": lookup, "kv": kv, "output": output}


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
                    output[token:token + 1, column * HEAD_DIM:(column + 1) * HEAD_DIM] = table[
                        row:row + 1, :
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
        for block in pl.spmd(((active + M_TILE - 1) // M_TILE) * HC,
                             name_hint="dsv41_a2a3_engram_gate"):
            token = (block // HC) * M_TILE
            copy = block % HC
            rows = pl.min(M_TILE, active - token)
            source_rms = pl.full([1, M_TILE], dtype=pl.FP32, value=0.0)
            key_rms = pl.full([1, M_TILE], dtype=pl.FP32, value=0.0)
            dot = pl.full([1, M_TILE], dtype=pl.FP32, value=0.0)
            for d0 in pl.range(0, HIDDEN, GATE_TILE):
                x = pl.cast(pl.slice(hidden_flat, [M_TILE, GATE_TILE], [token, copy * HIDDEN + d0], valid_shape=[rows, GATE_TILE]), pl.FP32)
                key = pl.cast(pl.slice(kv, [M_TILE, GATE_TILE], [token, copy * HIDDEN + d0], valid_shape=[rows, GATE_TILE]), pl.FP32)
                weight = channel_weight[copy:copy + 1, d0:d0 + GATE_TILE]
                source_rms = pl.add(source_rms, pl.reshape(pl.row_sum(pl.mul(x, x)), [1, M_TILE]))
                key_rms = pl.add(key_rms, pl.reshape(pl.row_sum(pl.mul(key, key)), [1, M_TILE]))
                dot = pl.add(dot, pl.reshape(pl.row_sum(pl.col_expand_mul(pl.mul(x, key), weight)), [1, M_TILE]))
            inv = pl.mul(
                pl.rsqrt(pl.add(pl.mul(source_rms, 1.0 / HIDDEN), eps)),
                pl.rsqrt(pl.add(pl.mul(key_rms, 1.0 / HIDDEN), eps)),
            )
            dot = pl.mul(pl.mul(dot, inv), HIDDEN ** -0.5)
            magnitude = pl.sqrt(pl.maximum(pl.abs(dot), 1.0e-6))
            bits = pl.cast(pl.reinterpret_view(dot, pl.INT32), pl.FP32)
            negative = pl.maximum(pl.minimum(bits, 0.0), -1.0)
            signed = pl.mul(magnitude, pl.add(pl.mul(negative, 2.0), 1.0))
            gate_value = pl.recip(pl.add(pl.exp(pl.neg(signed)), 1.0))
            mask = pl.cast(pl.slice(mask_flat, [1, M_TILE], [0, token], valid_shape=[1, rows]), pl.FP32)
            gate_value = pl.reshape(pl.mul(gate_value, pl.reshape(mask, [1, M_TILE])), [M_TILE, 1])
            for d0 in pl.range(0, HIDDEN, GATE_TILE):
                x = pl.cast(pl.slice(hidden_flat, [M_TILE, GATE_TILE], [token, copy * HIDDEN + d0], valid_shape=[rows, GATE_TILE]), pl.FP32)
                value = pl.cast(pl.slice(kv, [M_TILE, GATE_TILE], [token, HC * HIDDEN + d0], valid_shape=[rows, GATE_TILE]), pl.FP32)
                result = pl.cast(pl.add(x, pl.row_expand_mul(value, gate_value)), pl.BF16, mode="rint")
                output_flat[token:token + M_TILE, copy * HIDDEN + d0:copy * HIDDEN + d0 + GATE_TILE] = pl.set_validshape(result, rows, GATE_TILE)
        return output

    return gate


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

from .decode_network_qkv_proj_rope import make_bf16_projection_staged


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
    from .decode_network_qkv_proj_rope import golden_bf16_projection

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
    from .decode_network_qkv_proj_rope import golden_prefix_compare

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
    from .decode_network_qkv_proj_rope import golden_prefix_compare

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


# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
ENGRAM_K = K
ENGRAM_N = N


"""Complete EP2 Decode composition for an SWA layer with Engram state.

Layer 1 owns the sliding-window cache and consumes an Engram lookup prepared
at the Decode input boundary.  The lookup rows are frozen by the independent
reference capture; projection, gate, mHC, attention and MoE remain PyPTO
operations in this product entry.
"""


import pypto.language.distributed as pld

from .decode_network_swa import (
    AUX_PAD,
    ATTENTION,
    CHANNELS,
    COUNT_PAD,
    SIGNAL_PAD,
    D,
    EP_SIZE,
    EP2_RANK,
    FFN_POST,
    FFN_PRE,
    HC,
    LATENT,
    LANE_ROWS,
    MOE_INTER,
    N_EXPERTS,
    N_LOCAL_EXPERTS,
    O_RANK,
    ROPE_PAIRS,
    T,
    TOPK,
    WIDTH,
    WINDOW,
)
from .decode_network_decode_rmsnorm import NORM_T_TILE, rms_norm_tiled_with_fp32


ENGRAM_TABLE_ROWS = T * COLS


@pl.jit.inline
def _engram_linear(
    value: pl.Tensor[[T, ENGRAM_K], pl.BF16],
    weight: pl.Tensor[[ENGRAM_K, ENGRAM_N], pl.BF16],
    output: pl.InOut[pl.Tensor[[T, ENGRAM_N], pl.BF16]],
    active: pl.Scalar[pl.INT32],
):
    """Project frozen Engram lookup rows in the checkpoint BF16 basis."""

    n_tile = 64
    blocks = ENGRAM_N // n_tile
    output_2d = pl.reshape(output, [T, ENGRAM_N])
    for block in pl.spmd(((active + 15) // 16) * blocks, name_hint="dsv41_a2a3_engram_bf16_linear"):
        t0 = (block // blocks) * 16
        n0 = (block % blocks) * n_tile
        rows = pl.min(16, active - t0)
        acc = pl.create_tensor([16, n_tile], dtype=pl.FP32)
        for k0 in pl.pipeline(0, ENGRAM_K, 256, stage=2):
            lhs = pl.slice(value, [16, 256], [t0, k0], valid_shape=[rows, 256])
            lhs = pl.set_validshape(pl.fillpad(lhs, pad_value=pl.PadValue.zero), 16, 256)
            rhs = pl.slice(weight, [256, n_tile], [k0, n0])
            acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(k0 == 0))
        output_2d[t0:t0 + 16, n0:n0 + n_tile] = pl.set_validshape(
            pl.cast(acc, pl.BF16, mode="rint"), rows, n_tile
        )


ENGRAM_LINEAR = _engram_linear
ENGRAM_GATE = pl.jit.inline(auto_scope=False)(make_gate(T)._func)
@pl.jit(auto_scope=False)
def decode_layer_ep2_engram_rank(
    x_hc: pl.Tensor[[T, HC, HIDDEN], pl.BF16],
    engram_hash_ids: pl.Tensor[[T, COLS], pl.INT32],
    engram_table: pl.Tensor[[ENGRAM_TABLE_ROWS, ROW_WIDTH], pl.BF16],
    engram_wkv: pl.Tensor[[ENGRAM_K, ENGRAM_N], pl.BF16],
    engram_channel_weight: pl.Tensor[[HC, HIDDEN], pl.FP32],
    engram_mask: pl.Tensor[[T], pl.INT32],
    engram_lookup: pl.Tensor[[T, ENGRAM_K], pl.BF16],
    engram_kv: pl.InOut[pl.Tensor[[T, ENGRAM_N], pl.BF16]],
    engram_hidden: pl.InOut[pl.Tensor[[T, HC, HIDDEN], pl.BF16]],
    attn_hc_function: pl.Tensor[[CHANNELS, HC * HIDDEN], pl.FP32],
    attn_hc_scale: pl.Tensor[[3], pl.FP32],
    attn_hc_base: pl.Tensor[[CHANNELS], pl.FP32],
    attn_incoming: pl.Tensor[[T, HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[HIDDEN], pl.BF16],
    wq_a: pl.Tensor[[HIDDEN, LATENT], pl.INT8],
    wq_a_scale: pl.Tensor[[LATENT], pl.FP32],
    q_norm_weight: pl.Tensor[[LATENT], pl.BF16],
    wq_b: pl.Tensor[[LATENT, WIDTH], pl.INT8],
    wq_b_scale: pl.Tensor[[WIDTH], pl.FP32],
    wkv: pl.Tensor[[HIDDEN, 512], pl.INT8],
    wkv_scale: pl.Tensor[[512], pl.FP32],
    kv_norm_weight: pl.Tensor[[512], pl.BF16],
    wo_a: pl.Tensor[[8, WIDTH // 8, O_RANK], pl.BF16],
    wo_b: pl.Tensor[[8 * O_RANK, HIDDEN], pl.BF16],
    rope_cos: pl.Tensor[[T, ROPE_PAIRS], pl.FP32],
    rope_sin: pl.Tensor[[T, ROPE_PAIRS], pl.FP32],
    slots: pl.Tensor[[T, 2], pl.INT64],
    indices: pl.Tensor[[T, WINDOW], pl.INT32],
    sink: pl.Tensor[[64], pl.FP32],
    positions: pl.InOut[pl.Tensor[[T], pl.INT64]],
    query_latent: pl.InOut[pl.Tensor[[T, LATENT], pl.BF16]],
    query: pl.InOut[pl.Tensor[[T, WIDTH], pl.BF16]],
    window_kv: pl.InOut[pl.Tensor[[T, 512], pl.BF16]],
    cache: pl.InOut[pl.Tensor[[2, 128, 1, 512], pl.BF16]],
    cache_lens: pl.InOut[pl.Tensor[[2], pl.INT32]],
    attn_output: pl.InOut[pl.Tensor[[T, HIDDEN], pl.BF16]],
    attn_next_pre: pl.InOut[pl.Tensor[[T, HC], pl.FP32]],
    attn_post: pl.InOut[pl.Tensor[[T, HC], pl.FP32]],
    attn_comb: pl.InOut[pl.Tensor[[T, HC, HC], pl.FP32]],
    attn_collapsed: pl.InOut[pl.Tensor[[T, HIDDEN], pl.BF16]],
    attn_normed: pl.InOut[pl.Tensor[[T, HIDDEN], pl.BF16]],
    after_attention: pl.InOut[pl.Tensor[[T, HC, HIDDEN], pl.BF16]],
    ffn_hc_function: pl.Tensor[[CHANNELS, HC * HIDDEN], pl.FP32],
    ffn_hc_scale: pl.Tensor[[3], pl.FP32],
    ffn_hc_base: pl.Tensor[[CHANNELS], pl.FP32],
    ffn_norm_weight: pl.Tensor[[HIDDEN], pl.BF16],
    ffn_next_pre: pl.InOut[pl.Tensor[[T, HC], pl.FP32]],
    ffn_post: pl.InOut[pl.Tensor[[T, HC], pl.FP32]],
    ffn_comb: pl.InOut[pl.Tensor[[T, HC, HC], pl.FP32]],
    ffn_collapsed: pl.InOut[pl.Tensor[[T, HIDDEN], pl.BF16]],
    ffn_normed: pl.InOut[pl.Tensor[[T, HIDDEN], pl.BF16]],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    text_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    vision_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, 2560], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    shared_gate_up: pl.Tensor[[D, MOE_INTER * 2], pl.INT8],
    shared_gate_up_scale: pl.Tensor[[MOE_INTER * 2], pl.FP32],
    shared_down: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_down_scale: pl.Tensor[[D], pl.FP32],
    lane_x: pld.DistributedTensor[[LANE_ROWS, D], pl.INT8],
    lane_aux: pld.DistributedTensor[[LANE_ROWS, AUX_PAD], pl.FP32],
    lane_count: pld.DistributedTensor[[EP_SIZE, COUNT_PAD], pl.INT32],
    lane_ready: pld.DistributedTensor[[EP_SIZE, SIGNAL_PAD], pl.INT32],
    slot_out: pld.DistributedTensor[[T * TOPK, D], pl.FP32],
    return_ready: pld.DistributedTensor[[EP_SIZE, SIGNAL_PAD], pl.INT32],
    logits: pl.InOut[pl.Tensor[[T, N_EXPERTS], pl.FP32]],
    route_ids: pl.InOut[pl.Tensor[[T, TOPK], pl.INT32]],
    route_weights: pl.InOut[pl.Tensor[[T, TOPK], pl.FP32]],
    moe_output: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    output: pl.Out[pl.Tensor[[T, HC, HIDDEN], pl.BF16]],
    active: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """Run Engram followed by the shared complete layer body."""

    # The resident hash table is a reference-side capture boundary.  Consume
    # its frozen BF16 rows here and keep both projection and gate in PyPTO.
    # This avoids dynamic row indexing in the device kernel while preserving
    # the real lookup values and the model's BF16 narrowing points.
    _engram_linear(engram_lookup, engram_wkv, engram_kv, active)
    ENGRAM_GATE(x_hc, engram_kv, engram_channel_weight, engram_mask, engram_hidden, active)
    ATTENTION(
        engram_hidden,
        attn_hc_function,
        attn_hc_scale,
        attn_hc_base,
        attn_incoming,
        attn_norm_weight,
        wq_a,
        wq_a_scale,
        q_norm_weight,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        kv_norm_weight,
        wo_a,
        wo_b,
        rope_cos,
        rope_sin,
        slots,
        indices,
        sink,
        positions,
        query_latent,
        query,
        window_kv,
        cache,
        cache_lens,
        attn_output,
        attn_next_pre,
        attn_post,
        attn_comb,
        attn_collapsed,
        attn_normed,
        after_attention,
        active,
    )
    ffn_collapsed_fp32 = pl.create_tensor([T, HIDDEN], dtype=pl.FP32)
    FFN_PRE(
        after_attention,
        ffn_hc_function,
        ffn_hc_scale,
        ffn_hc_base,
        attn_next_pre,
        ffn_next_pre,
        ffn_post,
        ffn_comb,
        ffn_collapsed, ffn_collapsed_fp32,
        active,
    )
    ffn_router_normed = pl.create_tensor([T, HIDDEN], dtype=pl.FP32)
    for block in pl.spmd((active + NORM_T_TILE - 1) // NORM_T_TILE, name_hint="dsv41_decode_ffn_norm"):
        # The fused reference normalizes the BF16 HC collapse and exposes its
        # FP32 result to the router.  Keep the BF16 store boundary before both
        # outputs so expert input and router input share the same source.
        rms_norm_tiled_with_fp32(
            ffn_collapsed, ffn_norm_weight, ffn_normed, ffn_router_normed, block, active
        )
    EP2_RANK(
        ffn_normed,
        ffn_router_normed,
        gate_weight,
        text_bias,
        vision_bias,
        input_ids,
        routed_w1,
        routed_w1_scale,
        routed_w3,
        routed_w3_scale,
        routed_w2,
        routed_w2_scale,
        shared_gate_up,
        shared_gate_up_scale,
        shared_down,
        shared_down_scale,
        lane_x,
        lane_aux,
        lane_count,
        lane_ready,
        slot_out,
        return_ready,
        logits,
        route_ids,
        route_weights,
        moe_output,
        active,
        rank,
        epoch,
    )
    FFN_POST(moe_output, after_attention, ffn_post, ffn_comb, output, active)
    return output

@pl.jit.host
def decode_layer_ep2_engram(
    x_hc: pl.Tensor[[EP_SIZE, T, HC, HIDDEN], pl.BF16],
    engram_hash_ids: pl.Tensor[[EP_SIZE, T, COLS], pl.INT32],
    engram_table: pl.Tensor[[EP_SIZE, ENGRAM_TABLE_ROWS, ROW_WIDTH], pl.BF16],
    engram_wkv: pl.Tensor[[EP_SIZE, ENGRAM_K, ENGRAM_N], pl.BF16],
    engram_channel_weight: pl.Tensor[[EP_SIZE, HC, HIDDEN], pl.FP32],
    engram_mask: pl.Tensor[[EP_SIZE, T], pl.INT32],
    engram_lookup: pl.Tensor[[EP_SIZE, T, ENGRAM_K], pl.BF16],
    engram_kv: pl.InOut[pl.Tensor[[EP_SIZE, T, ENGRAM_N], pl.BF16]],
    engram_hidden: pl.InOut[pl.Tensor[[EP_SIZE, T, HC, HIDDEN], pl.BF16]],
    attn_hc_function: pl.Tensor[[EP_SIZE, CHANNELS, HC * HIDDEN], pl.FP32],
    attn_hc_scale: pl.Tensor[[EP_SIZE, 3], pl.FP32],
    attn_hc_base: pl.Tensor[[EP_SIZE, CHANNELS], pl.FP32],
    attn_incoming: pl.Tensor[[EP_SIZE, T, HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[EP_SIZE, HIDDEN], pl.BF16],
    wq_a: pl.Tensor[[EP_SIZE, HIDDEN, LATENT], pl.INT8],
    wq_a_scale: pl.Tensor[[EP_SIZE, LATENT], pl.FP32],
    q_norm_weight: pl.Tensor[[EP_SIZE, LATENT], pl.BF16],
    wq_b: pl.Tensor[[EP_SIZE, LATENT, WIDTH], pl.INT8],
    wq_b_scale: pl.Tensor[[EP_SIZE, WIDTH], pl.FP32],
    wkv: pl.Tensor[[EP_SIZE, HIDDEN, 512], pl.INT8],
    wkv_scale: pl.Tensor[[EP_SIZE, 512], pl.FP32],
    kv_norm_weight: pl.Tensor[[EP_SIZE, 512], pl.BF16],
    wo_a: pl.Tensor[[EP_SIZE, 8, WIDTH // 8, O_RANK], pl.BF16],
    wo_b: pl.Tensor[[EP_SIZE, 8 * O_RANK, HIDDEN], pl.BF16],
    rope_cos: pl.Tensor[[EP_SIZE, T, ROPE_PAIRS], pl.FP32],
    rope_sin: pl.Tensor[[EP_SIZE, T, ROPE_PAIRS], pl.FP32],
    slots: pl.Tensor[[EP_SIZE, T, 2], pl.INT64],
    indices: pl.Tensor[[EP_SIZE, T, WINDOW], pl.INT32],
    sink: pl.Tensor[[EP_SIZE, 64], pl.FP32],
    positions: pl.InOut[pl.Tensor[[EP_SIZE, T], pl.INT64]],
    query_latent: pl.InOut[pl.Tensor[[EP_SIZE, T, LATENT], pl.BF16]],
    query: pl.InOut[pl.Tensor[[EP_SIZE, T, WIDTH], pl.BF16]],
    window_kv: pl.InOut[pl.Tensor[[EP_SIZE, T, 512], pl.BF16]],
    cache: pl.InOut[pl.Tensor[[EP_SIZE, 2, 128, 1, 512], pl.BF16]],
    cache_lens: pl.InOut[pl.Tensor[[EP_SIZE, 2], pl.INT32]],
    attn_output: pl.InOut[pl.Tensor[[EP_SIZE, T, HIDDEN], pl.BF16]],
    attn_next_pre: pl.InOut[pl.Tensor[[EP_SIZE, T, HC], pl.FP32]],
    attn_post: pl.InOut[pl.Tensor[[EP_SIZE, T, HC], pl.FP32]],
    attn_comb: pl.InOut[pl.Tensor[[EP_SIZE, T, HC, HC], pl.FP32]],
    attn_collapsed: pl.InOut[pl.Tensor[[EP_SIZE, T, HIDDEN], pl.BF16]],
    attn_normed: pl.InOut[pl.Tensor[[EP_SIZE, T, HIDDEN], pl.BF16]],
    after_attention: pl.InOut[pl.Tensor[[EP_SIZE, T, HC, HIDDEN], pl.BF16]],
    ffn_hc_function: pl.Tensor[[EP_SIZE, CHANNELS, HC * HIDDEN], pl.FP32],
    ffn_hc_scale: pl.Tensor[[EP_SIZE, 3], pl.FP32],
    ffn_hc_base: pl.Tensor[[EP_SIZE, CHANNELS], pl.FP32],
    ffn_norm_weight: pl.Tensor[[EP_SIZE, HIDDEN], pl.BF16],
    ffn_next_pre: pl.InOut[pl.Tensor[[EP_SIZE, T, HC], pl.FP32]],
    ffn_post: pl.InOut[pl.Tensor[[EP_SIZE, T, HC], pl.FP32]],
    ffn_comb: pl.InOut[pl.Tensor[[EP_SIZE, T, HC, HC], pl.FP32]],
    ffn_collapsed: pl.InOut[pl.Tensor[[EP_SIZE, T, HIDDEN], pl.BF16]],
    ffn_normed: pl.InOut[pl.Tensor[[EP_SIZE, T, HIDDEN], pl.BF16]],
    gate_weight: pl.Tensor[[EP_SIZE, N_EXPERTS, D], pl.FP32],
    text_bias: pl.Tensor[[EP_SIZE, N_EXPERTS], pl.FP32],
    vision_bias: pl.Tensor[[EP_SIZE, N_EXPERTS], pl.FP32],
    input_ids: pl.Tensor[[EP_SIZE, T], pl.INT64],
    routed_w1: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, D, 2560], pl.INT8],
    routed_w2_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, D], pl.FP32],
    shared_gate_up: pl.Tensor[[EP_SIZE, D, MOE_INTER * 2], pl.INT8],
    shared_gate_up_scale: pl.Tensor[[EP_SIZE, MOE_INTER * 2], pl.FP32],
    shared_down: pl.Tensor[[EP_SIZE, MOE_INTER, D], pl.INT8],
    shared_down_scale: pl.Tensor[[EP_SIZE, D], pl.FP32],
    logits: pl.Out[pl.Tensor[[EP_SIZE, T, N_EXPERTS], pl.FP32]],
    route_ids: pl.Out[pl.Tensor[[EP_SIZE, T, TOPK], pl.INT32]],
    route_weights: pl.Out[pl.Tensor[[EP_SIZE, T, TOPK], pl.FP32]],
    moe_output: pl.Out[pl.Tensor[[EP_SIZE, T, D], pl.BF16]],
    output: pl.Out[pl.Tensor[[EP_SIZE, T, HC, HIDDEN], pl.BF16]],
    active: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """Launch the L1 Engram composition over the two EP ranks."""

    lane_x_buf = pld.alloc_window_buffer([LANE_ROWS, D], dtype=pl.INT8)
    lane_aux_buf = pld.alloc_window_buffer([LANE_ROWS, AUX_PAD], dtype=pl.FP32)
    lane_count_buf = pld.alloc_window_buffer([EP_SIZE, COUNT_PAD], dtype=pl.INT32)
    lane_ready_buf = pld.alloc_window_buffer([EP_SIZE, SIGNAL_PAD], dtype=pl.INT32)
    slot_out_buf = pld.alloc_window_buffer([T * TOPK, D], dtype=pl.FP32)
    return_ready_buf = pld.alloc_window_buffer([EP_SIZE, SIGNAL_PAD], dtype=pl.INT32)
    for rank in pl.range(EP_SIZE):
        lane_x = pld.window(lane_x_buf, [LANE_ROWS, D], dtype=pl.INT8)
        lane_aux = pld.window(lane_aux_buf, [LANE_ROWS, AUX_PAD], dtype=pl.FP32)
        lane_count = pld.window(lane_count_buf, [EP_SIZE, COUNT_PAD], dtype=pl.INT32)
        lane_ready = pld.window(lane_ready_buf, [EP_SIZE, SIGNAL_PAD], dtype=pl.INT32)
        slot_out = pld.window(slot_out_buf, [T * TOPK, D], dtype=pl.FP32)
        return_ready = pld.window(return_ready_buf, [EP_SIZE, SIGNAL_PAD], dtype=pl.INT32)
        decode_layer_ep2_engram_rank(
            x_hc[rank],
            engram_hash_ids[rank], engram_table[rank], engram_wkv[rank],
            engram_channel_weight[rank], engram_mask[rank], engram_lookup[rank], engram_kv[rank], engram_hidden[rank],
            attn_hc_function[rank], attn_hc_scale[rank], attn_hc_base[rank], attn_incoming[rank],
            attn_norm_weight[rank], wq_a[rank], wq_a_scale[rank], q_norm_weight[rank], wq_b[rank],
            wq_b_scale[rank], wkv[rank], wkv_scale[rank], kv_norm_weight[rank], wo_a[rank], wo_b[rank],
            rope_cos[rank], rope_sin[rank], slots[rank], indices[rank], sink[rank], positions[rank],
            query_latent[rank], query[rank], window_kv[rank], cache[rank], cache_lens[rank],
            attn_output[rank], attn_next_pre[rank], attn_post[rank], attn_comb[rank], attn_collapsed[rank],
            attn_normed[rank], after_attention[rank], ffn_hc_function[rank], ffn_hc_scale[rank],
            ffn_hc_base[rank], ffn_norm_weight[rank], ffn_next_pre[rank], ffn_post[rank], ffn_comb[rank],
            ffn_collapsed[rank], ffn_normed[rank], gate_weight[rank], text_bias[rank], vision_bias[rank],
            input_ids[rank], routed_w1[rank], routed_w1_scale[rank], routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank], shared_gate_up[rank], shared_gate_up_scale[rank],
            shared_down[rank], shared_down_scale[rank], lane_x, lane_aux, lane_count, lane_ready, slot_out,
            return_ready, logits[rank], route_ids[rank], route_weights[rank], moe_output[rank], output[rank],
            active, rank, epoch, device=rank,
        )


__all__ = ["decode_layer_ep2_engram", "decode_layer_ep2_engram_rank"]


def make_decode_engram():
    """Apply checkpoint Engram projection and the original-basis residual gate."""

    restore = make_original_basis_hidden(T)
    gate = make_original_basis_gate(T, 1e-20)

    @pl.jit
    def decode_engram(
        hidden: pl.Tensor[[T, HC, HIDDEN], pl.BF16],
        lookup: pl.Tensor[[T, ENGRAM_K], pl.BF16],
        weight: pl.Tensor[[ENGRAM_K, ENGRAM_N], pl.BF16],
        channel_weight: pl.Tensor[[HC, HIDDEN], pl.FP32],
        rotation: pl.Tensor[[32, 32], pl.FP32],
        mask: pl.Tensor[[T], pl.INT32],
        output: pl.InOut[pl.Tensor[[T, HC, HIDDEN], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        projected = pl.create_tensor([T, ENGRAM_N], dtype=pl.BF16)
        _engram_linear(lookup, weight, projected, active)
        original = pl.create_tensor([T, HC, HIDDEN], dtype=pl.FP32)
        restore(hidden, rotation, original, active)
        probabilities = pl.create_tensor([T, HC], dtype=pl.FP32)
        gate(hidden, original, projected, channel_weight, mask, probabilities, output, active)
        return output

    return decode_engram

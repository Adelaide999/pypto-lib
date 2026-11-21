# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Rotary embeddings, tables, goldens and phase-specific rotations."""

import math
import pypto.language as pl
import torch as decode_torch
import torch

# tiling
ROPE_WIDE_HEAD_ROWS_TILE = 32
ROPE_NARROW_HEAD_ROWS_TILE = 64
ROPE_WORKER_LIMIT = 48
ROPE_TABLE_ROWS_TILE = 64


def make_decode_adjacent_rope(capacity, heads, head_dim, rope_dim):
    """Rotate the final channels of each head in adjacent pairs."""
    output_width = heads * head_dim
    nope_dim = head_dim - rope_dim
    pair_width = rope_dim // 2

    @pl.jit.inline
    def rope(
        x: pl.Tensor[[capacity, output_width], pl.BF16],
        cosine: pl.Tensor[[capacity, pair_width], pl.FP32],
        sine: pl.Tensor[[capacity, pair_width], pl.FP32],
        output: pl.Tensor[[capacity, output_width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(active * heads, name_hint="a2_adjacent_rope"):
            t = block // heads
            base = (block % heads) * head_dim
            output[t : t + 1, base : base + nope_dim] = x[t : t + 1, base : base + nope_dim]
            tail = pl.cast(x[t : t + 1, base + nope_dim : base + head_dim], pl.FP32)
            even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            c = cosine[t : t + 1, :]
            s = sine[t : t + 1, :]
            even_c = pl.mul(even, c)
            odd_s = pl.mul(odd, s)
            odd_c = pl.mul(odd, c)
            even_s = pl.mul(even, s)
            real = pl.sub(even_c, odd_s)
            imag = pl.add(odd_c, even_s)
            rotated = pl.full([1, rope_dim], dtype=pl.FP32, value=0.0)
            rotated = pl.tensor.scatter(real, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
            rotated = pl.tensor.scatter(imag, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
            output[t : t + 1, base + nope_dim : base + head_dim] = pl.cast(rotated, pl.BF16, mode="rint")
        return output

    return rope


def make_rope(capacity: int, heads: int, head_dim: int, rope_dim: int):
    if rope_dim % 2 or capacity <= 0:
        raise ValueError("RoPE capacity and dimension must be positive and even")
    width = heads * head_dim
    nope = head_dim - rope_dim
    pairs = rope_dim // 2

    @pl.jit.inline
    def rope(
        value: pl.Tensor[[capacity, width], pl.BF16],
        cosine: pl.Tensor[[capacity, pairs], pl.FP32],
        sine: pl.Tensor[[capacity, pairs], pl.FP32],
        output: pl.Tensor[[capacity, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(active * heads, name_hint="dsv41_a2a3_rope"):
            token = block // heads
            base = (block % heads) * head_dim
            output[token : token + 1, base : base + nope] = value[token : token + 1, base : base + nope]
            tail = pl.cast(value[token : token + 1, base + nope : base + head_dim], pl.FP32)
            even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            c = cosine[token : token + 1, :]
            s = sine[token : token + 1, :]
            real = pl.sub(pl.mul(even, c), pl.mul(odd, s))
            imag = pl.add(pl.mul(even, s), pl.mul(odd, c))
            rotated = pl.full([1, rope_dim], dtype=pl.FP32, value=0.0)
            rotated = pl.tensor.scatter(real, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
            rotated = pl.tensor.scatter(imag, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
            output[token : token + 1, base + nope : base + head_dim] = pl.cast(rotated, pl.BF16, mode="rint")
        return output

    return rope


def make_rope_tables(capacity, rope_dim, theta, scaling=None):
    """Build plain or YaRN RoPE tables from actual token positions."""
    pairs = rope_dim // 2
    exponent_scale = 2.0 * math.log(theta) / rope_dim
    factor, low, high = 1.0, 0, 1
    if scaling is not None:
        factor = float(scaling["factor"])
        original = scaling["original_max_position_embeddings"]
        fast = scaling.get("beta_fast", 32)
        slow = scaling.get("beta_slow", 1)
        low = max(0, math.floor(rope_dim * math.log(original / (fast * 2 * math.pi)) / (2 * math.log(theta))))
        high = min(
            rope_dim - 1,
            math.ceil(rope_dim * math.log(original / (slow * 2 * math.pi)) / (2 * math.log(theta))),
        )
    use_yarn = scaling is not None
    ramp_start = float(low)
    inverse_ramp_width = 1.0 / max(high - low, 0.001)
    interpolation_delta = 1.0 / factor - 1.0
    tile_rows = math.gcd(capacity, ROPE_TABLE_ROWS_TILE)
    rows = tile_rows if pairs == 32 and capacity >= 128 and tile_rows >= 8 else 1
    worker_limit = ROPE_WORKER_LIMIT if rows > 1 else capacity

    @pl.jit
    def rope_tables(
        positions: pl.Tensor[[capacity], pl.INT64],
        cosine: pl.InOut[pl.Tensor[[capacity, pairs], pl.FP32]],
        sine: pl.InOut[pl.Tensor[[capacity, pairs], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        blocks = (active + rows - 1) // rows
        workers = pl.min(blocks, worker_limit)
        if rows > 1:
            position_rows = pl.reshape(positions, [1, capacity])
        for worker in pl.spmd(workers, name_hint="prefill_rope_tables"):
            dimension_ids = pl.tile.arange(0, [1, pairs], dtype=pl.INT32)
            dimensions = pl.cast(dimension_ids, pl.FP32)
            exponents = pl.mul(dimensions, exponent_scale)
            powers = pl.exp(exponents)
            ones = pl.tile.full([1, pairs], dtype=pl.FP32, value=1.0)
            frequencies = pl.div(ones, powers, high_precision=True)
            if use_yarn:
                ramp_offset = pl.sub(dimensions, ramp_start)
                ramp_unbounded = pl.mul(ramp_offset, inverse_ramp_width)
                ramp_nonnegative = pl.maximum(ramp_unbounded, 0.0)
                ramp = pl.minimum(ramp_nonnegative, 1.0)
                interpolation = pl.mul(ramp, interpolation_delta)
                multiplier = pl.add(interpolation, 1.0)
                frequencies = pl.mul(frequencies, multiplier)
            if rows == 1:
                position = pl.cast(pl.read(positions, [worker]), pl.FP32)
                angles = pl.mul(frequencies, position)
                pl.store(pl.cos(angles), [worker, 0], cosine)
                pl.store(pl.sin(angles), [worker, 0], sine)
            else:
                first = worker * blocks // workers
                last = (worker + 1) * blocks // workers
                template = pl.tile.full([rows, pairs], dtype=pl.FP32, value=1.0)
                frequency_tile = pl.col_expand_mul(template, frequencies)
                for block in pl.range(first, last):
                    offset = block * rows
                    valid = pl.min(rows, active - offset)
                    loaded_positions = pl.load(position_rows, [0, offset], [1, rows])
                    packed_positions = pl.tile.full([1, rows], dtype=pl.FP32, value=0.0)
                    for row in pl.range(valid):
                        position_i64 = pl.tile.read(loaded_positions, [0, row])
                        position_fp32 = pl.cast(position_i64, pl.FP32)
                        pl.tile.write(packed_positions, [0, row], position_fp32)
                    row_positions = pl.reshape(packed_positions, [rows, 1])
                    angle_tile = pl.row_expand_mul(frequency_tile, row_positions)
                    cos_block = pl.cos(angle_tile)
                    sin_block = pl.sin(angle_tile)
                    cos_tile = pl.set_validshape(cos_block, valid, pairs)
                    sin_tile = pl.set_validshape(sin_block, valid, pairs)
                    pl.store(cos_tile, [offset, 0], cosine)
                    pl.store(sin_tile, [offset, 0], sine)
        return cosine, sine

    return rope_tables


def make_pipelined_rope(capacity, heads, head_dim, *, inverse, name_hint, allow_early_resolve):
    """Rotate groups of heads sharing one token's coefficients."""
    width = heads * head_dim
    nope = head_dim - 64
    rows = min(heads, 32)
    elements = rows * 64
    direction = -1.0 if inverse else 1.0

    @pl.jit.inline
    def rope(
        x: pl.Tensor[[capacity, width], pl.BF16],
        cosine: pl.Tensor[[capacity, 32], pl.FP32],
        sine: pl.Tensor[[capacity, 32], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        tokens = pl.tensor.dim(x, 0)
        source = pl.reshape(x, [tokens * heads, head_dim])
        target = pl.reshape(output, [tokens * heads, head_dim])
        blocks = active * (heads // rows)
        workers = pl.min(blocks, ROPE_WORKER_LIMIT)
        for worker in pl.spmd(workers, name_hint=name_hint, allow_early_resolve=allow_early_resolve):
            lanes = pl.cast(pl.tile.arange(0, [1, elements], dtype=pl.INT32), pl.FP32)
            pairs = pl.cast(pl.mul(lanes, 0.5), pl.INT32, mode="trunc")
            parity = pl.sub(lanes, pl.mul(pl.cast(pairs, pl.FP32), 2.0))
            swap_idx = pl.cast(pl.sub(pl.add(lanes, 1.0), pl.mul(parity, 2.0)), pl.INT32)
            coeff_idx = pl.tile.slice(pairs, [1, 64], [0, 0])
            head_parity = pl.tile.slice(parity, [1, 64], [0, 0])
            sign = pl.mul(pl.sub(pl.mul(head_parity, 2.0), 1.0), direction)
            start = worker * blocks // workers
            end = (worker + 1) * blocks // workers
            # One token per tile: expand the 32 pairs once, then broadcast
            # over the heads. Smaller coefficient buffers fit both UB slots.
            for block in pl.pipeline(start, end, stage=2):
                row = block * rows
                token = row // heads
                block_value = pl.load(source, [row, 0], [rows, head_dim])
                value = pl.cast(pl.tile.slice(block_value, [rows, 64], [0, nope]), pl.FP32)
                swap_tmp = pl.create_tile([1, elements], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                swapped = pl.tile.gather(pl.reshape(value, [1, elements]), swap_idx, swap_tmp)
                coeff_tmp = pl.create_tile([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                c = pl.tile.gather(pl.load(cosine, [token, 0], [1, 32]), coeff_idx, coeff_tmp)
                s = pl.mul(pl.tile.gather(pl.load(sine, [token, 0], [1, 32]), coeff_idx, coeff_tmp), sign)
                direct = pl.col_expand_mul(value, c)
                cross = pl.col_expand_mul(pl.reshape(swapped, [rows, 64]), s)
                rotated = pl.cast(pl.add(direct, cross), pl.BF16, mode="rint")
                combined = pl.tile.assemble(block_value, rotated, [0, nope])
                pl.store(combined, [row, 0], target)
        return output

    return rope


def make_adjacent_rope(
    capacity, heads, head_dim, rope_dim, *, inverse=False, name_hint="a2_adjacent_rope",
    allow_early_resolve=False,
):
    """Rotate the final channels of each head in adjacent pairs."""
    if capacity >= 1024 and rope_dim == 64 and (heads, head_dim) in {(16, 512), (64, 512), (32, 128)}:
        return make_pipelined_rope(
            capacity, heads, head_dim, inverse=inverse, name_hint=name_hint, allow_early_resolve=allow_early_resolve,
        )
    output_width = heads * head_dim
    nope_dim = head_dim - rope_dim
    pair_width = rope_dim // 2
    max_rows_tile = ROPE_NARROW_HEAD_ROWS_TILE if head_dim <= 128 else ROPE_WIDE_HEAD_ROWS_TILE
    if max_rows_tile % heads == 0:
        row_tile = math.gcd(capacity * heads, max_rows_tile)
    else:
        row_tile = math.gcd(heads, max_rows_tile)
    heads_per_tile_token = min(heads, row_tile)
    token_tile = row_tile // heads_per_tile_token
    table_elements = token_tile * pair_width
    elements = row_tile * rope_dim
    token_elements = heads_per_tile_token * rope_dim
    pair_stride = float(pair_width)
    direction = -1.0 if inverse else 1.0

    @pl.jit.inline
    def rope(
        x: pl.Tensor[[capacity, output_width], pl.BF16],
        cosine: pl.Tensor[[capacity, pair_width], pl.FP32],
        sine: pl.Tensor[[capacity, pair_width], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, output_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        x_heads = pl.reshape(x, [capacity * heads, head_dim])
        output_heads = pl.reshape(output, [capacity * heads, head_dim])
        blocks = (active * heads + row_tile - 1) // row_tile
        workers = pl.min(blocks, ROPE_WORKER_LIMIT)
        blocks_per_worker = (blocks + ROPE_WORKER_LIMIT - 1) // ROPE_WORKER_LIMIT
        for worker in pl.spmd(workers, name_hint=name_hint, allow_early_resolve=allow_early_resolve):
            # Map the flattened tile lanes to adjacent values and their shared token tables.
            lanes = pl.cast(pl.arange(0, [1, elements], dtype=pl.INT32), pl.FP32)
            pairs = pl.cast(pl.mul(lanes, 0.5), pl.INT32, mode="trunc")
            pair_ids = pl.cast(pairs, pl.FP32)
            pair_starts = pl.mul(pair_ids, 2.0)
            parity = pl.sub(lanes, pair_starts)
            sign = pl.mul(pl.sub(pl.mul(parity, 2.0), 1.0), direction)
            swap = pl.sub(pl.add(lanes, 1.0), pl.mul(parity, 2.0))
            swap_index = pl.cast(swap, pl.INT32)
            head_row = pl.cast(pl.mul(lanes, 1.0 / rope_dim), pl.INT32, mode="trunc")
            token_row = pl.cast(pl.mul(lanes, 1.0 / token_elements), pl.INT32, mode="trunc")
            head_offset = pl.mul(pl.cast(head_row, pl.FP32), pair_stride)
            token_offset = pl.mul(pl.cast(token_row, pl.FP32), pair_stride)
            pair_in_head = pl.sub(pair_ids, head_offset)
            coefficient_index = pl.cast(pl.add(pair_in_head, token_offset), pl.INT32)
            start = worker * blocks_per_worker
            end = pl.min(blocks, start + blocks_per_worker)
            for block in pl.range(start, end):
                h0 = block * row_tile
                rows = pl.min(row_tile, active * heads - h0)
                tail = pl.slice(x_heads, [row_tile, rope_dim], [h0, nope_dim], valid_shape=[rows, rope_dim])
                tail = pl.set_validshape(tail, row_tile, rope_dim)
                values = pl.cast(pl.reshape(tail, [1, elements]), pl.FP32)
                swapped = pl.gather(values, dim=-1, index=swap_index)
                t = h0 // heads
                c_table = pl.reshape(cosine[t:t + token_tile, :], [1, table_elements])
                s_table = pl.reshape(sine[t:t + token_tile, :], [1, table_elements])
                c = pl.gather(c_table, dim=-1, index=coefficient_index)
                s = pl.gather(s_table, dim=-1, index=coefficient_index)
                direct = pl.mul(values, c)
                cross = pl.mul(swapped, pl.mul(s, sign))
                rotated = pl.cast(pl.add(direct, cross), pl.BF16, mode="rint")
                result = pl.reshape(rotated, [row_tile, rope_dim])
                prefix = pl.slice(x_heads, [row_tile, nope_dim], [h0, 0], valid_shape=[rows, nope_dim])
                combined = pl.concat(prefix, result)
                output_heads[h0 : h0 + row_tile, :] = pl.set_validshape(combined, rows, head_dim)
        return output

    return rope


def make_rope_kernel(capacity: int, heads: int, head_dim: int, rope_dim: int):
    rotate = make_rope(capacity, heads, head_dim, rope_dim)
    pairs = rope_dim // 2
    width = heads * head_dim

    @pl.jit
    def kernel(
        value: pl.Tensor[[capacity, width], pl.BF16],
        cosine: pl.Tensor[[capacity, pairs], pl.FP32],
        sine: pl.Tensor[[capacity, pairs], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        rotate(value, cosine, sine, output, active)
        return output

    return kernel


def make_decode_rope(capacity: int, heads: int, head_dim: int, rope_dim: int):
    if rope_dim % 2 or capacity <= 0:
        raise ValueError("RoPE capacity and dimension must be positive and even")
    width = heads * head_dim
    nope = head_dim - rope_dim
    pairs = rope_dim // 2

    @pl.jit.inline
    def rope(
        value: pl.Tensor[[capacity, width], pl.BF16],
        cosine: pl.Tensor[[capacity, pairs], pl.FP32],
        sine: pl.Tensor[[capacity, pairs], pl.FP32],
        output: pl.Tensor[[capacity, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(active * heads, name_hint="dsv41_a2a3_rope"):
            token = block // heads
            base = (block % heads) * head_dim
            output[token : token + 1, base : base + nope] = value[token : token + 1, base : base + nope]
            tail = pl.cast(value[token : token + 1, base + nope : base + head_dim], pl.FP32)
            even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            c = cosine[token : token + 1, :]
            s = sine[token : token + 1, :]
            real = pl.sub(pl.mul(even, c), pl.mul(odd, s))
            imag = pl.add(pl.mul(even, s), pl.mul(odd, c))
            rotated = pl.full([1, rope_dim], dtype=pl.FP32, value=0.0)
            rotated = pl.tensor.scatter(real, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
            rotated = pl.tensor.scatter(imag, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
            output[token : token + 1, base + nope : base + head_dim] = pl.cast(rotated, pl.BF16, mode="rint")
        return output

    return rope


def rope_reference(
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    heads: int,
    head_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    if rope_dim % 2 or value.shape[-1] != heads * head_dim:
        raise ValueError("RoPE dimensions must be even and match heads * head_dim")
    output = value.clone()
    nope = head_dim - rope_dim
    shaped = value.view(value.shape[0], heads, head_dim)
    tail = shaped[..., nope:]
    even, odd = tail[..., 0::2].float(), tail[..., 1::2].float()
    c, s = cosine.float().unsqueeze(1), sine.float().unsqueeze(1)
    rotated = torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2)
    output.view(value.shape[0], heads, head_dim)[..., nope:] = rotated.to(torch.bfloat16)
    return output


def decode_rope_reference(
    value: decode_torch.Tensor,
    cosine: decode_torch.Tensor,
    sine: decode_torch.Tensor,
    heads: int,
    head_dim: int,
    rope_dim: int,
) -> decode_torch.Tensor:
    import torch

    if rope_dim % 2 or value.shape[-1] != heads * head_dim:
        raise ValueError("RoPE dimensions must be even and match heads * head_dim")
    output = value.clone()
    nope = head_dim - rope_dim
    shaped = value.view(value.shape[0], heads, head_dim)
    tail = shaped[..., nope:]
    even, odd = tail[..., 0::2].float(), tail[..., 1::2].float()
    c, s = cosine.float().unsqueeze(1), sine.float().unsqueeze(1)
    rotated = torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2)
    output.view(value.shape[0], heads, head_dim)[..., nope:] = rotated.to(torch.bfloat16)
    return output

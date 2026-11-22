# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pipelined rotary position embedding for the A3 attention projections."""

import pypto.language as pl


ROPE_WORKER_LIMIT = 48


def make_pipelined_rope(capacity, heads, head_dim, *, inverse, name_hint, allow_early_resolve, inplace=False):
    """Rotate heads sharing a token's coefficients. In-place callers pass x as output."""
    width = heads * head_dim
    nope = head_dim - 64
    load_start = nope if inplace else 0
    load_width = head_dim - load_start
    tail_start = nope - load_start
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
                block_value = pl.load(source, [row, load_start], [rows, load_width])
                value = pl.cast(pl.tile.slice(block_value, [rows, 64], [0, tail_start]), pl.FP32)
                swap_tmp = pl.create_tile([1, elements], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                swapped = pl.tile.gather(pl.reshape(value, [1, elements]), swap_idx, swap_tmp)
                coeff_tmp = pl.create_tile([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                c = pl.tile.gather(pl.load(cosine, [token, 0], [1, 32]), coeff_idx, coeff_tmp)
                s = pl.mul(pl.tile.gather(pl.load(sine, [token, 0], [1, 32]), coeff_idx, coeff_tmp), sign)
                direct = pl.col_expand_mul(value, c)
                cross = pl.col_expand_mul(pl.reshape(swapped, [rows, 64]), s)
                rotated = pl.cast(pl.add(direct, cross), pl.BF16, mode="rint")
                if inplace:
                    pl.store(rotated, [row, nope], target)
                else:
                    combined = pl.tile.assemble(block_value, rotated, [0, nope])
                    pl.store(combined, [row, 0], target)
        return output

    return rope

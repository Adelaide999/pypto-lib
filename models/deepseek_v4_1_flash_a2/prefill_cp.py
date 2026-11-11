# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pack CP heads and padded SP rows for TP collectives."""

import pypto.language as pl


# tiling
D_TILE = 512


def make_sp_masked_zero(capacity, width):
    """Zero masked physical SP rows and preserve valid and inactive rows."""
    @pl.jit
    def masked_zero(
        value: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        valid: pl.Scalar[pl.INT32],
        physical: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd((physical - valid) * (width // D_TILE), name_hint="prefill_sp_masked_zero"):
            token = valid + block // (width // D_TILE)
            column = block % (width // D_TILE) * D_TILE
            tile = pl.full([1, D_TILE], dtype=pl.BF16, value=0.0)
            value[token:token + 1, column:column + D_TILE] = tile
        return value

    return masked_zero


def make_head_pack(capacity, heads, head_dim, tp):
    """Pack all local CP heads into fixed equal peer buckets."""
    local_width = heads // tp * head_dim

    @pl.jit
    def head_pack(
        value: pl.Tensor[[capacity, heads * head_dim], pl.BF16],
        packed: pl.InOut[pl.Tensor[[tp * capacity, local_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(tp * capacity * (local_width // D_TILE), name_hint="prefill_cp_head_pack"):
            column = block % (local_width // D_TILE) * D_TILE
            row = block // (local_width // D_TILE)
            peer = row // capacity
            token = row % capacity
            tile = pl.full([1, D_TILE], dtype=pl.BF16, value=0.0)
            if token < active:
                source = peer * local_width + column
                tile = value[token:token + 1, source:source + D_TILE]
            packed[row:row + 1, column:column + D_TILE] = tile
        return packed

    return head_pack


def make_sp_gather_unpack(capacity, width, tp, dtype=pl.BF16, tile=D_TILE):
    """Compact fixed peer buckets into contiguous physical token order."""
    scalar_rows = width < 8

    @pl.jit
    def gather_unpack(
        packet: pl.Tensor[[tp * capacity, width], dtype],
        output: pl.InOut[pl.Tensor[[capacity, width], dtype]],
        physical: pl.Scalar[pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        if scalar_rows:
            for _block in pl.spmd(1, name_hint="prefill_sp_state_unpack"):
                for token in pl.range(active):
                    source = token // physical * capacity + token % physical
                    for column in pl.range(width):
                        state_value = pl.read(packet, [source, column])
                        pl.write(output, [token, column], state_value)
        else:
            for block in pl.spmd(active * (width // tile), name_hint="prefill_sp_gather_unpack"):
                token = block // (width // tile)
                column = block % (width // tile) * tile
                source = token // physical * capacity + token % physical
                vector_value = packet[source:source + 1, column:column + tile]
                output[token:token + 1, column:column + tile] = vector_value
        return output

    return gather_unpack


def make_sp_reduce_pack(capacity, hidden, tp):
    """Pad full Wo-B partial sums into equal reduce-scatter peer buckets."""
    @pl.jit
    def reduce_pack(
        value: pl.Tensor[[capacity, hidden], pl.BF16],
        packed: pl.InOut[pl.Tensor[[tp * capacity, hidden], pl.BF16]],
        physical: pl.Scalar[pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(tp * capacity * (hidden // D_TILE), name_hint="prefill_sp_reduce_pack"):
            row = block // (hidden // D_TILE)
            column = block % (hidden // D_TILE) * D_TILE
            peer = row // capacity
            local = row % capacity
            source = peer * physical + local
            tile = pl.full([1, D_TILE], dtype=pl.BF16, value=0.0)
            if local < physical:
                if source < active:
                    tile = value[source:source + 1, column:column + D_TILE]
            packed[row:row + 1, column:column + D_TILE] = tile
        return packed

    return reduce_pack

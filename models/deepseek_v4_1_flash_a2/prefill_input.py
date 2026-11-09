# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded token embedding lookup and initial mHC lane expansion."""

import pypto.language as pl


# tiling
D_TILE = 512
LOOKUP_PAD = 8


def make_prefill_input(capacity, row_count, hidden, hc):
    """Build lookup over exact checkpoint rows carrying their original token IDs."""
    if hidden % D_TILE:
        raise ValueError("Embedding width must be divisible by the column tile")

    @pl.jit
    def prefill_input(
        input_ids: pl.Tensor[[capacity], pl.INT64],
        global_ids: pl.Tensor[[row_count], pl.INT64],
        rows: pl.Tensor[[row_count, hidden], pl.BF16],
        lookup: pl.InOut[pl.Tensor[[capacity, LOOKUP_PAD], pl.INT32]],
        hidden_hc: pl.InOut[pl.Tensor[[capacity, hc, hidden], pl.BF16]],
        incoming: pl.InOut[pl.Tensor[[capacity, hc], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        for _block in pl.spmd(1, name_hint="a2_prefill_token_lookup"):
            for token in pl.range(active):
                token_id = pl.read(input_ids, [token])
                matched = pl.cast(-1, pl.INT32)
                for row in pl.range(row_count):
                    global_id = pl.read(global_ids, [row])
                    if token_id == global_id:
                        matched = pl.cast(row, pl.INT32)
                for column in pl.range(LOOKUP_PAD):
                    if column == 0:
                        pl.write(lookup, [token, column], matched)
                    else:
                        pl.write(lookup, [token, column], pl.cast(0, pl.INT32))
                if matched >= 0:
                    for channel in pl.range(hc):
                        if channel == 0:
                            pl.write(incoming, [token, channel], pl.cast(1.0, pl.FP32))
                        else:
                            pl.write(incoming, [token, channel], pl.cast(0.0, pl.FP32))

        for block in pl.spmd(active * hc * (hidden // D_TILE), name_hint="a2_prefill_embed_expand"):
            token = block // (hc * (hidden // D_TILE))
            channel = block // (hidden // D_TILE) % hc
            offset = block % (hidden // D_TILE) * D_TILE
            source = pl.read(lookup, [token, 0])
            if source >= 0:
                value = rows[source:source + 1, offset:offset + D_TILE]
                hidden_hc[token:token + 1, channel:channel + 1, offset:offset + D_TILE] = pl.reshape(value, [1, 1, D_TILE])
        return lookup, hidden_hc, incoming

    return prefill_input


def make_hc_widen(tokens, hidden, hc):
    """Widen the BF16 decoder state for the final FP32 mHC collapse."""

    @pl.jit
    def hc_widen(
        hidden_hc: pl.Tensor[[tokens, hc, hidden], pl.BF16],
        widened: pl.Out[pl.Tensor[[tokens, hc, hidden], pl.FP32]],
    ):
        for block in pl.spmd(tokens * hc * (hidden // D_TILE), name_hint="a2_prefill_hc_widen"):
            token = block // (hc * (hidden // D_TILE))
            channel = block // (hidden // D_TILE) % hc
            offset = block % (hidden // D_TILE) * D_TILE
            value = hidden_hc[token:token + 1, channel:channel + 1, offset:offset + D_TILE]
            widened[token:token + 1, channel:channel + 1, offset:offset + D_TILE] = pl.cast(value, pl.FP32)
        return widened

    return hc_widen

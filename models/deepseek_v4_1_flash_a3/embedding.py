# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Vocabulary lookup and initial HC streams shared by Prefill and Decode."""

import pypto.language as pl


# tiling
D_TILE = 512


def make_vocab_embedding(capacity, hidden, vocabulary, tp):
    """Copy owned vocabulary rows and zero other active rows before TP reduction."""
    if capacity < 1 or hidden % D_TILE or tp not in (1, 2, 4) or vocabulary % tp:
        raise ValueError("Require aligned hidden width and complete TP1/2/4 vocabulary shards")
    local_rows = vocabulary // tp

    @pl.jit
    def embedding(
        token_ids: pl.Tensor[[capacity], pl.INT64],
        live: pl.Tensor[[capacity], pl.INT32],
        weight: pl.Tensor[[local_rows, hidden], pl.BF16],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        vocab_start: pl.Scalar[pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(active * (hidden // D_TILE), name_hint="prefill_vocab_embedding"):
            token = block // (hidden // D_TILE)
            column = block % (hidden // D_TILE) * D_TILE
            global_id = pl.cast(pl.read(token_ids, [token]), pl.INT32)
            local_id = global_id - vocab_start
            tile = pl.full([1, D_TILE], dtype=pl.BF16, value=0.0)
            if pl.read(live, [token]) > 0:
                if local_id >= 0:
                    if local_id < local_rows:
                        row = pl.cast(local_id, pl.INDEX)
                        tile = weight[row:row + 1, column:column + D_TILE]
            output[token:token + 1, column:column + D_TILE] = tile
        return output

    return embedding


def make_initial_hc(capacity, hidden, hc_mult):
    """Shard tokens, replicate BF16 hidden lanes and set the initial one-hot pre-mix."""
    if capacity < 1 or hidden % D_TILE or hc_mult < 1:
        raise ValueError("Require aligned hidden width and positive HC width")

    @pl.jit
    def initial_hc(
        embedding: pl.Tensor[[capacity, hidden], pl.BF16],
        streams: pl.InOut[pl.Tensor[[capacity, hc_mult, hidden], pl.BF16]],
        premix: pl.InOut[pl.Tensor[[capacity, hc_mult], pl.FP32]],
        tp_rank: pl.Scalar[pl.INT32],
        physical: pl.Scalar[pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        flattened = pl.reshape(streams, [capacity, hc_mult * hidden])
        for block in pl.spmd(physical * hc_mult * (hidden // D_TILE), name_hint="prefill_initial_hc"):
            column = block % (hidden // D_TILE) * D_TILE
            lane = block // (hidden // D_TILE) % hc_mult
            token = block // (hidden // D_TILE) // hc_mult
            source = tp_rank * physical + token
            tile = pl.full([1, D_TILE], dtype=pl.BF16, value=0.0)
            if source < active:
                tile = embedding[source:source + 1, column:column + D_TILE]
            offset = lane * hidden + column
            flattened[token:token + 1, offset:offset + D_TILE] = tile
        for instance in pl.spmd(1, name_hint="prefill_initial_premix"):
            for token in pl.range(physical):
                for lane in pl.unroll(hc_mult):
                    value = pl.cast(0.0, pl.FP32)
                    if lane == 0:
                        value = pl.cast(1.0, pl.FP32)
                    pl.write(premix, [token, lane], value)
        return streams, premix

    return initial_hc

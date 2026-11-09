# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Resident Mini-B IO entries with actual readonly argument readback ports."""

import pypto.language as pl

from .output_head import make_lm_head, NORM_D_TILE, NORM_T_TILE, mhc_pre, rms_norm_tiled
from .prefill_input import make_hc_widen

D = 5120
HC = 4
VOCAB = 129280
D_TILE = 512


def make_resident_embedding(capacity):
    """Gather original vocabulary rows and initialize delayed mHC controls."""
    if capacity not in (1, 7, 16):
        raise ValueError("Require the bounded Mini-B token capacity")

    @pl.jit
    def resident_embedding(
        input_ids: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        weight: pl.InOut[pl.Tensor[[VOCAB, D], pl.BF16]],
        valid_ids: pl.InOut[pl.Tensor[[capacity], pl.INT32]],
        hidden_hc: pl.InOut[pl.Tensor[[capacity, HC, D], pl.BF16]],
        incoming: pl.InOut[pl.Tensor[[capacity, HC], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        # Scalar stores write a whole cache line. One instance owns all the
        # short control rows, including adjacent tokens in the same line.
        for _block in pl.spmd(1, name_hint="mini_b_embedding_controls"):
            for token in pl.range(active):
                source = pl.read(input_ids, [token])
                valid = pl.cast(0, pl.INT32)
                if source >= 0 and source < VOCAB:
                    valid = pl.cast(1, pl.INT32)
                    for channel in pl.range(HC):
                        if channel == 0:
                            pl.write(incoming, [token, channel], pl.cast(1.0, pl.FP32))
                        else:
                            pl.write(incoming, [token, channel], pl.cast(0.0, pl.FP32))
                pl.write(valid_ids, [token], valid)

        for block in pl.spmd(active * HC * (D // D_TILE), name_hint="mini_b_embedding_gather"):
            token = block // (HC * (D // D_TILE))
            channel = block // (D // D_TILE) % HC
            offset = block % (D // D_TILE) * D_TILE
            source = pl.read(input_ids, [token])
            if source >= 0 and source < VOCAB:
                value = weight[source:source + 1, offset:offset + D_TILE]
                hidden_hc[token:token + 1, channel:channel + 1, offset:offset + D_TILE] = pl.reshape(
                    value, [1, 1, D_TILE])
        return input_ids, weight, valid_ids, hidden_hc, incoming

    return resident_embedding


def make_final_norm_bound(active):
    """Specialize original final_norm extent and retain nested inline result."""
    @pl.jit.inline
    def final_norm_bound(
        hidden: pl.Tensor[[active, HC, D], pl.FP32],
        pre_mix: pl.Tensor[[active, HC], pl.FP32],
        weight: pl.Tensor[[D], pl.BF16],
        collapsed: pl.Out[pl.Tensor[[active, D], pl.BF16]],
        normalized: pl.Out[pl.Tensor[[active, D], pl.BF16]],
    ):
        """Reuse the upstream final HC collapse and tiled RMSNorm without collectives."""
        mhc_pre(hidden, pre_mix, collapsed)
        tokens = active
        ones_matrix = pl.create_tensor([1, D], dtype=pl.BF16)
        unweighted = pl.create_tensor([active, D], dtype=pl.BF16)
        for chunk in pl.spmd(D // NORM_D_TILE, name_hint="a2_final_norm_ones"):
            offset = chunk * NORM_D_TILE
            ones_matrix[:, offset:offset + NORM_D_TILE] = pl.full([1, NORM_D_TILE], dtype=pl.BF16, value=1.0)
        ones = pl.reshape(ones_matrix, [D])
        for block in pl.spmd((tokens + NORM_T_TILE - 1) // NORM_T_TILE, name_hint="a2_final_norm"):
            # RMSNorm.forward_native rounds x / rms to BF16 before multiplying gamma.
            norm_result = rms_norm_tiled(collapsed, ones, unweighted, block, tokens)
            t0 = block * NORM_T_TILE
            rows = pl.min(NORM_T_TILE, tokens - t0)
            for offset in pl.range(0, D, NORM_D_TILE):
                value = pl.slice(unweighted, [NORM_T_TILE, NORM_D_TILE], [t0, offset], valid_shape=[rows, NORM_D_TILE])
                gamma = pl.reshape(pl.cast(weight[offset:offset + NORM_D_TILE], pl.FP32), [1, NORM_D_TILE])
                scaled = pl.col_expand_mul(pl.cast(value, pl.FP32), gamma)
                normalized[t0:t0 + NORM_T_TILE, offset:offset + NORM_D_TILE] = pl.set_validshape(
                    pl.cast(scaled, pl.BF16, mode="rint"), rows, NORM_D_TILE)
        return collapsed, normalized
    return final_norm_bound


def make_terminal_entries(active):
    """Keep existing terminal math and expose its readonly device arguments."""
    if active not in (1, 7, 16):
        raise ValueError("Require active1/7/16")
    widen = pl.jit.inline(make_hc_widen(active, D, HC)._func)
    head = pl.jit.inline(make_lm_head(active)._func)

    @pl.jit
    def terminal_widen(
        hidden_hc: pl.InOut[pl.Tensor[[active, HC, D], pl.BF16]],
        widened: pl.Out[pl.Tensor[[active, HC, D], pl.FP32]],
    ):
        widened = widen(hidden_hc, widened)
        return hidden_hc, widened

    norm = make_final_norm_bound(active)

    @pl.jit
    def terminal_norm(
        hidden: pl.InOut[pl.Tensor[[active, HC, D], pl.FP32]],
        pre_mix: pl.InOut[pl.Tensor[[active, HC], pl.FP32]],
        weight: pl.InOut[pl.Tensor[[D], pl.BF16]],
        collapsed: pl.Out[pl.Tensor[[active, D], pl.BF16]],
        normalized: pl.Out[pl.Tensor[[active, D], pl.BF16]],
    ):
        collapsed, normalized = norm(hidden, pre_mix, weight, collapsed, normalized)
        return hidden, pre_mix, weight, collapsed, normalized

    @pl.jit
    def terminal_head(
        hidden: pl.InOut[pl.Tensor[[active, D], pl.BF16]],
        weight: pl.InOut[pl.Tensor[[VOCAB, D], pl.BF16]],
        row_indices: pl.InOut[pl.Tensor[[16], pl.INT32]],
        logits: pl.Out[pl.Tensor[[16, VOCAB], pl.FP32]],
        sampled_ids: pl.Out[pl.Tensor[[16, 8], pl.INT32]],
    ):
        logits, sampled_ids = head(hidden, weight, row_indices, logits, sampled_ids)
        return hidden, weight, row_indices, logits, sampled_ids

    return {"widen": terminal_widen, "norm": terminal_norm, "head": terminal_head}

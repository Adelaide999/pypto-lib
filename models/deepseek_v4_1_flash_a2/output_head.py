# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-card A2 final HC/RMSNorm and full-vocabulary BF16 LM head."""

import pypto.language as pl

from .config import FLASH as M
from .operator_config import T_DYN
from .hc_collapse import mhc_pre
from .sampling import greedy_sample, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD
from .rmsnorm import rms_norm_tiled, NORM_T_TILE, NORM_D_TILE


D = M.hidden_size
HC = M.hc_mult
VOCAB = M.vocab_size
K_TILE = 256
N_TILE = 128


@pl.jit
def final_norm(
    hidden: pl.Tensor[[T_DYN, HC, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC], pl.FP32],
    weight: pl.Tensor[[D], pl.BF16],
    collapsed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    normalized: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    """Reuse the upstream final HC collapse and tiled RMSNorm without collectives."""
    hidden.bind_dynamic(0, T_DYN)
    collapsed.bind_dynamic(0, T_DYN)
    normalized.bind_dynamic(0, T_DYN)
    mhc_pre(hidden, pre_mix, collapsed)
    tokens = pl.tensor.dim(hidden, 0)
    ones_matrix = pl.create_tensor([1, D], dtype=pl.BF16)
    unweighted = pl.create_tensor([T_DYN, D], dtype=pl.BF16)
    for chunk in pl.spmd(D // NORM_D_TILE, name_hint="a2_final_norm_ones"):
        offset = chunk * NORM_D_TILE
        ones_matrix[:, offset:offset + NORM_D_TILE] = pl.full([1, NORM_D_TILE], dtype=pl.BF16, value=1.0)
    ones = pl.reshape(ones_matrix, [D])
    for block in pl.spmd((tokens + NORM_T_TILE - 1) // NORM_T_TILE, name_hint="a2_final_norm"):
        # RMSNorm.forward_native rounds x / rms to BF16 before multiplying gamma.
        rms_norm_tiled(collapsed, ones, unweighted, block, tokens)
        t0 = block * NORM_T_TILE
        rows = pl.min(NORM_T_TILE, tokens - t0)
        for offset in pl.range(0, D, NORM_D_TILE):
            value = pl.slice(unweighted, [NORM_T_TILE, NORM_D_TILE], [t0, offset], valid_shape=[rows, NORM_D_TILE])
            gamma = pl.reshape(pl.cast(weight[offset:offset + NORM_D_TILE], pl.FP32), [1, NORM_D_TILE])
            scaled = pl.col_expand_mul(pl.cast(value, pl.FP32), gamma)
            normalized[t0:t0 + NORM_T_TILE, offset:offset + NORM_D_TILE] = pl.set_validshape(
                pl.cast(scaled, pl.BF16, mode="rint"), rows, NORM_D_TILE)
    return collapsed, normalized


def make_lm_head(capacity):
    """Specialize the upstream LM-head projection for TP1, without communication."""

    @pl.jit
    def lm_head(
        hidden: pl.Tensor[[capacity, D], pl.BF16],
        weight: pl.Tensor[[VOCAB, D], pl.BF16],
        row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
        logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
        sampled_ids: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    ):
        selected = pl.create_tensor([MAX_LOGIT_ROWS, D], dtype=pl.BF16)
        for row in pl.spmd(MAX_LOGIT_ROWS, name_hint="a2_head_select"):
            source = pl.read(row_indices, [row])
            for offset in pl.range(0, D, K_TILE):
                if source >= 0 and source < capacity:
                    value = pl.slice(hidden, [1, K_TILE], [source, offset])
                else:
                    value = pl.full([1, K_TILE], dtype=pl.BF16, value=0.0)
                selected[row:row + 1, offset:offset + K_TILE] = value

        blocks = pl.system.available_cluster_count()
        for core in pl.spmd(blocks, name_hint="a2_head_matmul"):
            for column in pl.range(core, VOCAB // N_TILE, blocks):
                n0 = column * N_TILE
                operand = selected[:, :K_TILE]
                parameter = weight[n0:n0 + N_TILE, :K_TILE]
                accumulator = pl.matmul(operand, parameter, b_trans=True, out_dtype=pl.FP32)
                for k0 in pl.pipeline(K_TILE, D, K_TILE, stage=2):
                    operand = selected[:, k0:k0 + K_TILE]
                    parameter = weight[n0:n0 + N_TILE, k0:k0 + K_TILE]
                    accumulator = pl.matmul_acc(accumulator, operand, parameter, b_trans=True)
                # Native vLLM's unquantized BF16 head rounds before greedy sampling.
                narrowed = pl.cast(accumulator, pl.BF16, mode="rint")
                logits[:, n0:n0 + N_TILE] = pl.cast(narrowed, pl.FP32)
        greedy_sample(logits, sampled_ids)
        return logits, sampled_ids

    return lm_head

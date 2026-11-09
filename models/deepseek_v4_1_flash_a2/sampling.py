# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Select the first maximum token from full-vocabulary logits."""

import pypto.language as pl
from .config import FLASH as M

VOCAB = M.vocab_size
MAX_LOGIT_ROWS = 16
SAMPLED_IDS_PAD = 8

# tiling
GREEDY_VOCAB_CHUNK = 256
GREEDY_NUM_VOCAB_CHUNKS = VOCAB // GREEDY_VOCAB_CHUNK
GREEDY_CHUNK_PAD = 512
GREEDY_TOPK = 16


@pl.jit.inline
def greedy_sample(
    logits: pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    sampled_ids: pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
):
    """Select the first maximum token id from each full-vocabulary logits row."""
    for row in pl.spmd(MAX_LOGIT_ROWS, name_hint="lm_head_greedy_sample"):
        chunk_idx_init = pl.arange(0, [1, GREEDY_VOCAB_CHUNK], dtype=pl.UINT32)
        chunk_maxima = pl.create_tensor([1, GREEDY_CHUNK_PAD], dtype=pl.FP32)
        chunk_maxima[:, :] = pl.full(
            [1, GREEDY_CHUNK_PAD],
            dtype=pl.FP32,
            value=-3.402823e38,
        )
        for chunk in pl.range(GREEDY_NUM_VOCAB_CHUNKS):
            chunk_start = chunk * GREEDY_VOCAB_CHUNK
            scores = logits[
                row : row + 1,
                chunk_start : chunk_start + GREEDY_VOCAB_CHUNK,
            ]
            sorted_pairs = pl.sort32(scores, chunk_idx_init)
            sorted_pairs = pl.mrgsort(sorted_pairs, block_len=64)
            sorted_pairs = pl.mrgsort(
                sorted_pairs[:, 0:GREEDY_VOCAB_CHUNK],
                sorted_pairs[:, GREEDY_VOCAB_CHUNK : 2 * GREEDY_VOCAB_CHUNK],
            )
            top_pair = sorted_pairs[:, 0 : 2 * GREEDY_TOPK]
            top_values = pl.gather(top_pair, mask_pattern=pl.tile.MaskPattern.P0101)
            pl.write(chunk_maxima, [0, chunk], pl.read(top_values, [0, 0]))

        maxima_idx_init = pl.arange(0, [1, GREEDY_CHUNK_PAD], dtype=pl.UINT32)
        sorted_maxima = pl.sort32(chunk_maxima, maxima_idx_init)
        sorted_maxima = pl.mrgsort(sorted_maxima, block_len=64)
        sorted_maxima = pl.mrgsort(sorted_maxima, block_len=256)
        top_maximum_pair = sorted_maxima[:, 0 : 2 * GREEDY_TOPK]
        top_maximum_values = pl.gather(
            top_maximum_pair,
            mask_pattern=pl.tile.MaskPattern.P0101,
        )
        best_value = pl.read(top_maximum_values, [0, 0])

        # Reverse scans leave the lowest matching index selected, matching
        # torch.argmax's first-occurrence tie behavior.
        winning_chunk = pl.cast(0, pl.INT32)
        for chunk in pl.range(GREEDY_NUM_VOCAB_CHUNKS):
            scan_chunk = GREEDY_NUM_VOCAB_CHUNKS - 1 - chunk
            if pl.read(chunk_maxima, [0, scan_chunk]) == best_value:
                winning_chunk = pl.cast(scan_chunk, pl.INT32)

        chunk_base = winning_chunk * pl.cast(GREEDY_VOCAB_CHUNK, pl.INT32)
        winning_scores = pl.slice(
            logits,
            [1, GREEDY_VOCAB_CHUNK],
            [pl.cast(row, pl.INDEX), pl.cast(chunk_base, pl.INDEX)],
        )
        winning_offset = pl.cast(0, pl.INT32)
        for offset in pl.range(GREEDY_VOCAB_CHUNK):
            scan_offset = GREEDY_VOCAB_CHUNK - 1 - offset
            if pl.read(winning_scores, [0, scan_offset]) == best_value:
                winning_offset = pl.cast(scan_offset, pl.INT32)
        sampled_row = pl.create_tensor([1, SAMPLED_IDS_PAD], dtype=pl.INT32)
        sampled_row[:, :] = pl.full(
            [1, SAMPLED_IDS_PAD],
            dtype=pl.INT32,
            value=0,
        )
        pl.write(sampled_row, [0, 0], chunk_base + winning_offset)
        sampled_ids[row : row + 1, :] = sampled_row

    return sampled_ids

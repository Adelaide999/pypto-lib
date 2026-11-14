# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unified A2/A3 Decode layer entry points for the supported physical layers."""

from __future__ import annotations

from .config import FLASH, AttentionMode
from .decode_network_swa import (
    AUX_PAD,
    ATTENTION,
    CHANNELS,
    COUNT_PAD,
    D,
    EP_SIZE,
    EP2_RANK,
    FFN_POST,
    FFN_PRE,
    GROUP_INPUT,
    HC,
    HIDDEN,
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
    decode_layer_ep2,
    decode_layer_ep2_rank,
    product_contract,
)
from .decode_network_decode_c1a import (
    decode_layer_ep2_c1a_full,
    decode_layer_ep2_c1a_full_rank,
    decode_layer_ep2_c1a_reindex,
    decode_layer_ep2_c1a_reindex_rank,
    decode_layer_ep2_c1a_reuse,
    decode_layer_ep2_c1a_reuse_rank,
)
from .decode_network_engram import decode_layer_ep2_engram, decode_layer_ep2_engram_rank
from .decode_network_decode_c2a import decode_layer_ep2_sparse, decode_layer_ep2_sparse_rank


C2A_LAYERS = tuple(layer for layer in range(40) if FLASH.compress_ratios[layer] == 2)
C1A_FULL_LAYERS = (20,)
C1A_REINDEX_LAYERS = tuple(layer for layer in FLASH.index_source_layer_ids if layer > 20)
C1A_REUSE_LAYERS = tuple(
    layer for layer in range(20, 40) if FLASH.layer_config(layer).mode == AttentionMode.REUSE
)


LAYER_ENTRIES = {
    0: decode_layer_ep2,
    1: decode_layer_ep2_engram,
    **dict.fromkeys(C2A_LAYERS, decode_layer_ep2_sparse),
    **dict.fromkeys(C1A_FULL_LAYERS, decode_layer_ep2_c1a_full),
    **dict.fromkeys(C1A_REINDEX_LAYERS, decode_layer_ep2_c1a_reindex),
    **dict.fromkeys(C1A_REUSE_LAYERS, decode_layer_ep2_c1a_reuse),
}


def get_layer_entry(layer_id: int):
    """Return the canonical PyPTO entry for one supported physical layer."""

    try:
        return LAYER_ENTRIES[layer_id]
    except KeyError as error:
        supported = ", ".join(str(value) for value in sorted(LAYER_ENTRIES))
        raise ValueError(f"unsupported Decode layer {layer_id}; expected one of {supported}") from error


__all__ = [
    "C1A_FULL_LAYERS",
    "C1A_REINDEX_LAYERS",
    "C1A_REUSE_LAYERS",
    "C2A_LAYERS",
    "AUX_PAD",
    "ATTENTION",
    "CHANNELS",
    "COUNT_PAD",
    "D",
    "EP_SIZE",
    "EP2_RANK",
    "FFN_POST",
    "FFN_PRE",
    "GROUP_INPUT",
    "HC",
    "HIDDEN",
    "LATENT",
    "LANE_ROWS",
    "MOE_INTER",
    "N_EXPERTS",
    "N_LOCAL_EXPERTS",
    "O_RANK",
    "ROPE_PAIRS",
    "T",
    "TOPK",
    "WIDTH",
    "WINDOW",
    "LAYER_ENTRIES",
    "decode_layer_ep2",
    "decode_layer_ep2_rank",
    "decode_layer_ep2_c1a_full",
    "decode_layer_ep2_c1a_full_rank",
    "decode_layer_ep2_c1a_reindex",
    "decode_layer_ep2_c1a_reindex_rank",
    "decode_layer_ep2_c1a_reuse",
    "decode_layer_ep2_c1a_reuse_rank",
    "decode_layer_ep2_engram",
    "decode_layer_ep2_engram_rank",
    "decode_layer_ep2_sparse",
    "decode_layer_ep2_sparse_rank",
    "get_layer_entry",
    "product_contract",
]

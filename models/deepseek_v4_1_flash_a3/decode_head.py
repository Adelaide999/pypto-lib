# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Final Decode HC collapse, RMSNorm and full-vocabulary projection."""

from dataclasses import asdict

import pypto.language as pl

from .config import FLASH
from .hc import make_hc_collapse_inline
from .decode_projection import make_bf16_linear
from .decode_rmsnorm import NORM_T_TILE, rms_norm_tiled


def make_decode_head(capacity=16):
    """Compose the existing output operators after the final physical layer."""
    hidden, hc, vocabulary = FLASH.hidden_size, FLASH.hc_mult, FLASH.vocab_size
    collapse = make_hc_collapse_inline(capacity, asdict(FLASH))
    project = pl.jit.inline(make_bf16_linear(capacity, hidden, vocabulary)._func)

    @pl.jit
    def decode_head(
        streams: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        premix: pl.Tensor[[capacity, hc], pl.FP32],
        norm_weight: pl.Tensor[[hidden], pl.BF16],
        head_weight: pl.Tensor[[hidden, vocabulary], pl.BF16],
        logits: pl.InOut[pl.Tensor[[capacity, vocabulary], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        collapsed = pl.create_tensor([capacity, hidden], dtype=pl.BF16)
        collapse(streams, premix, collapsed, active)
        normalized = pl.create_tensor([capacity, hidden], dtype=pl.BF16)
        for block in pl.spmd((active + NORM_T_TILE - 1) // NORM_T_TILE, name_hint="decode_final_norm"):
            rms_norm_tiled(collapsed, norm_weight, normalized, block, active)
        project(normalized, head_weight, logits, active)
        return logits

    return decode_head

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Collapse four mHC streams with FP32 mixing and BF16 output."""

import pypto.language as pl
from .config import FLASH as M

# model config
D = M.hidden_size
HC_DIM = M.hc_dim

# tiling
D_TILE = 2560


@pl.jit.inline
def mhc_pre(
    x_hc: pl.Tensor,
    pre_mix: pl.Tensor,
    output: pl.Tensor,
):
    t_dim = pl.tensor.dim(x_hc, 0)
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    for t in pl.spmd(t_dim, name_hint="mhc_pre", allow_early_resolve=True):
        pre0 = pl.read(pre_mix, [t, 0])
        pre1 = pl.read(pre_mix, [t, 1])
        pre2 = pl.read(pre_mix, [t, 2])
        pre3 = pl.read(pre_mix, [t, 3])
        for d0 in pl.pipeline(0, D, D_TILE, stage=2):
            x0 = pl.load(x_flat, [t, d0], [1, D_TILE])
            x1 = pl.load(x_flat, [t, D + d0], [1, D_TILE])
            x2 = pl.load(x_flat, [t, 2 * D + d0], [1, D_TILE])
            x3 = pl.load(x_flat, [t, 3 * D + d0], [1, D_TILE])
            y0 = pl.mul(x0, pre0)
            y1 = pl.mul(x1, pre1)
            y2 = pl.mul(x2, pre2)
            y3 = pl.mul(x3, pre3)
            y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
            y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
            pl.store(y_bf16, [t, d0], output)
    return output

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""BF16 SwiGLU with explicit sigmoid and SiLU rounding boundaries."""

import pypto.language as pl


def make_swiglu(capacity: int, intermediate: int, clamp_limit: float = 10.0):
    """Evaluate the BF16 SwiGLU boundary in small vector tiles."""

    if intermediate % 256:
        raise ValueError("SwiGLU intermediate width must be divisible by 256")

    @pl.jit
    def activation(
        gate_up: pl.Tensor[[capacity, intermediate * 2], pl.BF16],
        output: pl.InOut[pl.Tensor[[capacity, intermediate], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        tiles = intermediate // 256
        for block in pl.spmd(((active + 7) // 8) * tiles,
                             name_hint="dsv41_a2a3_swiglu_v2"):
            token = (block // tiles) * 8
            column = (block % tiles) * 256
            rows = pl.min(8, active - token)
            # Keep the full vector tile in the traced arithmetic.  On A2/A3
            # a partial ``valid_shape`` changes the elementwise lowering and
            # can perturb the last valid row's BF16 rounding.
            gate = pl.cast(pl.slice(gate_up, [8, 256], [token, column]), pl.FP32)
            up = pl.cast(pl.slice(gate_up, [8, 256], [token, intermediate + column]), pl.FP32)
            gate = pl.minimum(gate, clamp_limit)
            up = pl.maximum(pl.minimum(up, clamp_limit), -clamp_limit)
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate)), 1.0))
            sigmoid_bf16 = pl.cast(sigmoid, pl.BF16, mode="rint")
            silu = pl.mul(gate, pl.cast(sigmoid_bf16, pl.FP32))
            silu_bf16 = pl.cast(silu, pl.BF16, mode="rint")
            activated = pl.mul(pl.cast(silu_bf16, pl.FP32), up)
            output[token:token + 8, column:column + 256] = pl.set_validshape(
                pl.cast(activated, pl.BF16, mode="rint"), rows, 256
            )
        return output

    return activation

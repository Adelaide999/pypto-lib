# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""BF16 SwiGLU rounding and device activation finite checks."""

import pypto.language as pl
from .runtime_config import SPMD_WORKER_LIMIT


# tiling
FINITE_ROWS_TILE = 8
FINITE_D_TILE = 256


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


def make_finite_check(capacity: int, hidden: int = 5120, hc_mult: int = 4):
    """Flag nonfinite active residual/pre-mix rows without copying activations."""
    width = hc_mult * hidden
    if capacity <= 0 or hidden <= 0 or hc_mult <= 0 or width % 256:
        raise ValueError("Finite checks require positive dimensions and a 256-aligned residual width")
    expected_count = float(width)
    row_blocks = (capacity + FINITE_ROWS_TILE - 1) // FINITE_ROWS_TILE
    workers = min(row_blocks, SPMD_WORKER_LIMIT)

    @pl.jit
    def finite_check(
        hidden_state: pl.Tensor[[capacity, hc_mult, hidden], pl.BF16],
        pre_mix: pl.Tensor[[capacity, hc_mult], pl.FP32],
        status: pl.Out[pl.Tensor[[capacity], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        flattened = pl.reshape(hidden_state, [capacity, width])
        status_rows = pl.reshape(status, [1, capacity])
        for worker in pl.spmd(workers, name_hint="dsv41_activation_finite_check"):
            for block in pl.range(worker, row_blocks, workers):
                first = block * FINITE_ROWS_TILE
                rows = pl.min(FINITE_ROWS_TILE, capacity - first)
                finite_counts = pl.full([1, FINITE_ROWS_TILE], dtype=pl.FP32, value=0.0)
                if first < active:
                    for column in pl.range(0, width, FINITE_D_TILE):
                        source = pl.slice(flattened, [FINITE_ROWS_TILE, FINITE_D_TILE], [first, column], valid_shape=[rows, FINITE_D_TILE])
                        values = pl.cast(source, pl.FP32)
                        magnitude = pl.abs(values)
                        finite = pl.cmp(magnitude, 3.4028234663852886e38, cmp_type=3)
                        counts = pl.reshape(pl.row_sum(finite), [1, FINITE_ROWS_TILE])
                        finite_counts = pl.add(finite_counts, counts)
                flags = pl.full([1, FINITE_ROWS_TILE], dtype=pl.INT32, value=1)
                for lane in pl.range(rows):
                    token = first + lane
                    valid = pl.cast(1, pl.INT32)
                    if token < active:
                        if pl.read(finite_counts, [0, lane]) != expected_count:
                            valid = pl.cast(0, pl.INT32)
                        for mix_lane in pl.range(hc_mult):
                            value = pl.read(pre_mix, [token, mix_lane])
                            if not (value <= 3.4028234663852886e38 and value >= -3.4028234663852886e38):
                                valid = pl.cast(0, pl.INT32)
                    pl.write(flags, [0, lane], valid)
                status_rows[0:1, first:first + FINITE_ROWS_TILE] = pl.set_validshape(flags, 1, rows)
        return status

    return finite_check


def golden_finite_check(args):
    """Mark nonfinite active rows and accept every inactive row."""
    import torch

    active = int(args["active"])
    args["status"].fill_(1)
    hidden = args["hidden_state"][:active].flatten(1)
    finite = torch.isfinite(hidden).all(dim=1) & torch.isfinite(args["pre_mix"][:active]).all(dim=1)
    args["status"][:active].copy_(finite.to(torch.int32))


def build_finite_check_specs(capacity=16, active=16, *, hidden=5120, hc_mult=4):
    """Build finite, NaN, infinity, inactive and partial-row fixtures."""
    import torch
    from golden import ScalarSpec, TensorSpec

    if not 0 <= active <= capacity:
        raise ValueError("Active rows must fit the finite-check capacity")

    def hidden_values():
        value = torch.full((capacity, hc_mult, hidden), 0.125, dtype=torch.bfloat16)
        for token, poison in enumerate((float("nan"), float("inf"), -float("inf"))):
            if token < active:
                value[token, token % hc_mult, (token * 257) % hidden] = poison
        if active < capacity:
            value[active:].fill_(float("nan"))
        return value

    def mix_values():
        value = torch.full((capacity, hc_mult), torch.finfo(torch.float32).max)
        for token, poison in enumerate((float("nan"), float("inf"), -float("inf")), start=4):
            if token < active:
                value[token, token % hc_mult] = poison
        if active < capacity:
            value[active:].fill_(float("inf"))
        return value

    return [
        TensorSpec("hidden_state", [capacity, hc_mult, hidden], torch.bfloat16, init_value=hidden_values),
        TensorSpec("pre_mix", [capacity, hc_mult], torch.float32, init_value=mix_values),
        TensorSpec("status", [capacity], torch.int32),
        ScalarSpec("active", torch.int32, active, compile_runtime=True),
    ]

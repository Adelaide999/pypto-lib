# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A3 shared W8A8 SwiGLU expert."""

import argparse
import json
from pathlib import Path

import pypto.language as pl
import torch
import torch.nn.functional as F

from config import FLASH as M, MOE_TOKENS
from gemm import gemm_dynamic_int8, gemm_int8


# model config
T = MOE_TOKENS
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit

# tiling
SH_M_TILE = 16
T_PAD = (T + SH_M_TILE - 1) // SH_M_TILE * SH_M_TILE


@pl.jit.inline(auto_scope=False)
def expert_shared(
    x_local: pl.Tensor,
    x_local_scale: pl.Tensor,
    shared_w1: pl.Tensor,
    shared_w1_scale: pl.Tensor,
    shared_w3: pl.Tensor,
    shared_w3_scale: pl.Tensor,
    shared_w2: pl.Tensor,
    shared_w2_scale: pl.Tensor,
    sh: pl.Tensor,
):
    """Compute the shared INT8 SwiGLU expert with BF16 activation rounding."""
    scale = pl.reshape(x_local_scale, [T_PAD, 1])
    gate = pl.create_tensor([T_PAD, MOE_INTER], dtype=pl.BF16)
    up = pl.create_tensor([T_PAD, MOE_INTER], dtype=pl.BF16)
    gemm_int8(x_local, scale, shared_w1, shared_w1_scale, gate, T, D, MOE_INTER, "shared_gate")
    gemm_int8(x_local, scale, shared_w3, shared_w3_scale, up, T, D, MOE_INTER, "shared_up")
    hidden = pl.create_tensor([T_PAD, MOE_INTER], dtype=pl.BF16)
    for row in pl.spmd(T_PAD, name_hint="shared_swiglu"):
        g = pl.cast(pl.load(gate, [row, 0], [1, MOE_INTER]), pl.FP32)
        u = pl.cast(pl.load(up, [row, 0], [1, MOE_INTER]), pl.FP32)
        g = pl.minimum(g, SWIGLU_LIMIT)
        u = pl.minimum(pl.maximum(u, -SWIGLU_LIMIT), SWIGLU_LIMIT)
        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(g)), 1.0))
        value = pl.mul(pl.mul(g, sigmoid), u)
        hidden = pl.store(pl.cast(value, pl.BF16, mode="rint"), [row, 0], hidden)
    return gemm_dynamic_int8(hidden, shared_w2, shared_w2_scale, sh, T, MOE_INTER, D, "shared_down")



def golden_expert_shared(payload, activation_scale, w1, scale1, w3, scale3, w2, scale2):
    """Evaluate the checkpoint W8A8 shared expert with BF16 activation boundaries."""
    from gemm import golden_dynamic_int8

    source = payload.long()
    factor = activation_scale.float().reshape(-1, 1)
    gate = (((source @ w1.long()).float()) * factor * scale1.float()).bfloat16().float()
    up = (((source @ w3.long()).float()) * factor * scale3.float()).bfloat16().float()
    hidden = F.silu(gate.clamp(max=SWIGLU_LIMIT)) * up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    return golden_dynamic_int8(hidden.bfloat16(), w2, scale2)


@pl.jit
def l2_expert_shared(
    source: pl.Tensor[[T_PAD, D], pl.INT8],
    scale: pl.Tensor[[1, T_PAD], pl.FP32],
    w1: pl.Tensor[[D, MOE_INTER], pl.INT8],
    scale1: pl.Tensor[[1, MOE_INTER], pl.FP32],
    w3: pl.Tensor[[D, MOE_INTER], pl.INT8],
    scale3: pl.Tensor[[1, MOE_INTER], pl.FP32],
    w2: pl.Tensor[[MOE_INTER, D], pl.INT8],
    scale2: pl.Tensor[[1, D], pl.FP32],
    output: pl.Out[pl.Tensor[[T_PAD, D], pl.BF16]],
):
    expert_shared(source, scale, w1, scale1, w3, scale3, w2, scale2, output)
    return output


def golden_l2_expert_shared(tensors):
    """Evaluate all three W8A8 projections and both SwiGLU rounding boundaries."""
    values = [tensors[name] for name in ('source', 'scale', 'w1', 'scale1', 'w3', 'scale3', 'w2', 'scale2')]
    tensors['output'].copy_(golden_expert_shared(*values))


if __name__ == '__main__':
    from golden import TensorSpec, ratio_allclose, run
    from gemm import golden_activation_int8

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p', '--platform', choices=['a2a3'], default='a2a3')
    parser.add_argument('-d', '--device', type=int, default=0)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--layer', type=int, choices=range(M.num_hidden_layers), default=0)
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    gen = torch.Generator().manual_seed(41)
    hidden = torch.randn(T_PAD, D, generator=gen).bfloat16()
    hidden[0].zero_()
    source = golden_activation_int8(hidden)
    factor = (hidden.float().abs().amax(-1) / 127).clamp_min(1e-30).reshape(1, T_PAD)
    values = {'source': source, 'scale': factor}
    if args.checkpoint:
        from weights import Checkpoint, ResidentWeights

        weights = ResidentWeights(Checkpoint(args.checkpoint))
    for number, width, columns in ((1, D, MOE_INTER), (3, D, MOE_INTER), (2, MOE_INTER, D)):
        if args.checkpoint:
            stem = f'layers.{args.layer}.ffn.shared_experts.w{number}'
            weight, scale = weights.projection(stem)
        else:
            weight = torch.randint(-127, 128, (width, columns), dtype=torch.int8, generator=gen)
            scale = torch.full((1, columns), 0.02 / 127)
        values[f'w{number}'], values[f'scale{number}'] = weight, scale
    specs = [TensorSpec(name, list(value.shape), value.dtype, init_value=value) for name, value in values.items()]
    specs.append(TensorSpec('output', [T_PAD, D], torch.bfloat16))
    result = run(
        fn=l2_expert_shared,
        specs=specs,
        golden_fn=golden_l2_expert_shared,
        compare_fn={'output': ratio_allclose(atol=0.003, rtol=0.03, max_error_ratio=0.005)},
        config={'backend_type': args.platform, 'execution_mode': False, 'device_id': args.device},
        compile_only=args.compile_only,
    )
    if args.report:
        report = {'layer': args.layer, 'passed': result.passed, 'compile_only': args.compile_only,
                  'build': str(result.work_dir), 'error': str(result.error) if result.error else None}
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    if not result.passed:
        raise SystemExit(result.error or 1)

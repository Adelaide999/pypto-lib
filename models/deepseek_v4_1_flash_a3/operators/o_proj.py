# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A3 inverse RoPE, grouped BF16 output projections and TP reductions."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "models.deepseek_v4_1_flash_a3.operators"


from .gemm import gemm_bf16_to_bf16
from .qkv_proj_rope import _rope_inplace

import torch


from .qkv_proj_rope import golden_rope_interleave as rope_interleave


import pypto.language as pl
import pypto.language.distributed as pld

from .config import (
    PREFILL_MAX_TOKENS,
    FLASH as M, SP_T_DYN, T_DYN, TP_SIZE,
)


# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
LOCAL_O_GROUPS = M.o_groups // TP_SIZE
LOCAL_O_WIDTH = M.o_groups // TP_SIZE * M.o_lora_rank
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
O_LORA = M.o_lora_rank
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim


# tiling
O_A_N_TILE = 128
O_B_N_TILE = 128
K_TILE = 128
D_TILE = 512


@pl.jit.inline(auto_scope=False)
def project_output(
    attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    sequential: pl.constexpr,
):
    """Apply inverse RoPE and grouped BF16 O-A/O-B projections."""
    tokens = pl.tensor.dim(output, 0)
    rotated = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    for block in pl.spmd((num_tokens + 15) // 16 * LOCAL_H, name_hint="o_projection_copy"):
        row = block // LOCAL_H * 16
        col = block % LOCAL_H * HEAD_DIM
        rows = pl.min(16, num_tokens - row)
        value = pl.load(attended, [row, col], [16, HEAD_DIM], valid_shape=[rows, HEAD_DIM])
        rotated = pl.store(value, [row, col], rotated)
    rotated = _rope_inplace(rotated, rope_cos, rope_sin, 0, num_tokens, LOCAL_H, HEAD_DIM, "inverse_rope_o", True)
    projected = pl.create_tensor([tokens, LOCAL_O_WIDTH], dtype=pl.BF16)
    ready = pl.system.task_dummy(deps=[])
    for group in pl.parallel(LOCAL_O_GROUPS):
        source_col = pl.cast(group * O_GROUP_IN, pl.INT32)
        output_col = pl.cast(group * O_LORA, pl.INT32)
        gemm_bf16_to_bf16(
            rotated, wo_a[group], projected, num_tokens, ready,
            O_GROUP_IN, O_LORA, O_A_N_TILE, K_TILE, 24, "proj_o_a", source_col, output_col, sequential,
        )
    gemm_bf16_to_bf16(projected, wo_b, output, num_tokens, ready, LOCAL_O_WIDTH, D, O_B_N_TILE, K_TILE, 24, "proj_o_b", 0, 0, sequential)
    return output


@pl.jit.inline(auto_scope=False)
def o_proj(
    attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Apply the Cube output projections."""
    return project_output(attended, wo_a, wo_b, rope_cos, rope_sin, output, num_tokens, False)


def golden_grouped_output(grouped, wo_a, *, output_dtype=None):
    """Apply grouped BF16 O-A with FP32 accumulation and BF16 rounding."""
    value = torch.einsum("tgk,gkn->tgn", grouped.float(), wo_a.float())
    return value.to(output_dtype or torch.bfloat16)


def golden_o_proj(attended, cos, sin, wo_a, wo_b, *, output_dtype=None, sequential_projection=False):
    """Evaluate inverse RoPE and the checkpoint's two BF16 output projections."""
    rope_dim = cos.shape[-1] * 2
    tail = rope_interleave(attended[..., -rope_dim:], cos, sin, inverse=True)
    rotated = torch.cat((attended[..., :-rope_dim], tail), dim=-1)
    groups = wo_a.shape[0]
    grouped = rotated.flatten(-2).unflatten(-1, (groups, -1))
    latent = golden_grouped_output(grouped, wo_a)
    result = latent.flatten(-2).float() @ wo_b.float()
    return result.to(output_dtype or torch.bfloat16)


@pl.jit.inline(auto_scope=False)
def retire_prefill_signals(
    completion_anchor: pl.Tensor,
    arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Clear ready and consumed signals at a serialized entry boundary."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_signal_retire", allow_early_resolve=False):
        anchor = pl.read(completion_anchor, [0, 0, 0])
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                arrived, peer=group_base + tp_rank, offsets=[peer, 0],
                value=0, op=pld.NotifyOp.Set,
            )
        pl.write(completion_anchor, [0, 0, 0], anchor)
    return completion_anchor


@pl.jit.inline(auto_scope=False)
def prefill_sp_output_reduce_scatter(
    partial: pl.Tensor[[T_DYN, D], pl.BF16],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[SP_T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Sum BF16 head-TP contributions in FP32 for this rank's token rows."""
    local_tokens = pl.tensor.dim(output, 0)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_reduce_reuse", allow_early_resolve=False) as reuse_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(
                output_arrived,
                offsets=[peer, 0], expected=(attention_epoch - 1) * 2, cmp=pld.WaitCmp.Ge,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_reduce_publish", deps=[reuse_tid]) as publish_tid:
        if num_tokens > 0:
            pld.tensor.put(
                dst=output_window, peer=group_base + tp_rank, src=partial,
                dst_offsets=[0, 0], src_offsets=[0, 0], shape=[num_tokens, D],
                chunk_rows=1, chunk_cols=D_TILE, pipeline=True,
            )
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                output_arrived,
                peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_reduce_wait", deps=[publish_tid],
               allow_early_resolve=False) as wait_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2 - 1, cmp=pld.WaitCmp.Ge)
    with pl.spmd(64, name_hint="sp_reduce_scatter", deps=[wait_tid]) as reduce_tid:
        worker = pl.tile.get_block_idx()
        for tile in pl.range(worker, local_tokens * (D // D_TILE), 64):
            local_row = tile // (D // D_TILE)
            col = tile % (D // D_TILE) * D_TILE
            row = tp_rank * local_tokens + local_row
            acc = pl.tile.full([1, D_TILE], dtype=pl.FP32, value=0.0)
            if row < num_tokens:
                for peer in pl.range(TP_SIZE):
                    value = pld.tile.remote_load(
                        output_window,
                        peer=group_base + peer, offsets=[row, col], shape=[1, D_TILE],
                    )
                    acc = pl.add(acc, pl.cast(value, pl.FP32))
            result = pl.cast(acc, pl.BF16, mode="rint")
            output = pl.store(result, [local_row, col], output)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_reduce_release", deps=[reduce_tid]) as release_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                output_arrived,
                peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_reduce_consumed", deps=[release_tid],
               allow_early_resolve=False):
        for peer in pl.range(TP_SIZE):
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2, cmp=pld.WaitCmp.Ge)
        anchor = pl.read(output, [0, 0])
        pl.write(output, [0, 0], anchor)
    return output


@pl.jit.inline(auto_scope=False)
def prefill_tp_output_all_reduce(
    output_partial: pl.Tensor[[T_DYN, D], pl.BF16],
    output_window: pld.DistributedTensor,
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_reuse", allow_early_resolve=False) as reuse_tid:
        for peer in pl.range(TP_SIZE):
            previous_epoch = (attention_epoch - 1) * 2
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=previous_epoch, cmp=pld.WaitCmp.Ge)
    with pl.spmd(64, name_hint="prefill_tp_publish", deps=[reuse_tid]) as publish_tid:
        worker = pl.tile.get_block_idx()
        for tile in pl.range(worker, num_tokens * (D // 512), 64):
            row = tile // (D // 512)
            col = tile % (D // 512) * 512
            pld.tensor.put(
                dst=output_window, peer=group_base + tp_rank, src=output_partial,
                dst_offsets=[row, col], src_offsets=[row, col], shape=[1, 512],
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_ready", deps=[publish_tid]) as ready_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                output_arrived,
                peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_wait", deps=[ready_tid],
               allow_early_resolve=False) as wait_tid:
        for peer in pl.range(TP_SIZE):
            ready_epoch = attention_epoch * 2 - 1
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=ready_epoch, cmp=pld.WaitCmp.Ge)
    with pl.spmd(64, name_hint="prefill_tp_reduce", deps=[wait_tid]) as reduce_tid:
        worker = pl.tile.get_block_idx()
        output_rows = pl.tensor.dim(output, 0)
        for tile in pl.range(worker, output_rows * (D // 512), 64):
            row = tile // (D // 512)
            col = tile % (D // 512) * 512
            acc = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
            if row < num_tokens:
                for peer in pl.range(TP_SIZE):
                    value = pld.tile.remote_load(
                        output_window,
                        peer=group_base + peer, offsets=[row, col], shape=[1, 512],
                    )
                    acc = pl.add(acc, pl.cast(value, pl.FP32))
            output = pl.store(pl.cast(acc, pl.BF16, mode="rint"), [row, col], output)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_release", deps=[reduce_tid]) as release_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                output_arrived,
                peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_consumed", deps=[release_tid],
               allow_early_resolve=False):
        for peer in pl.range(TP_SIZE):
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2, cmp=pld.WaitCmp.Ge)
        anchor = pl.read(output, [0, 0])
        pl.write(output, [0, 0], anchor)
    return output


@pl.jit
def l2_o_proj(
    attended: pl.Tensor[[32, LOCAL_H * HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.BF16],
    rope_cos: pl.Tensor[[32, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[32, ROPE_DIM // 2], pl.FP32],
    counts: pl.Tensor[[1], pl.INT32],
    output: pl.InOut[pl.Tensor[[32, D], pl.BF16]],
):
    num_tokens = pl.read(counts, [0])
    o_proj(attended, wo_a, wo_b, rope_cos, rope_sin, output, num_tokens)
    return output


def golden_l2_o_proj(tensors):
    """Project the valid local-head rows and preserve padded output rows."""
    count = int(tensors["counts"][0])
    attended = tensors["attended"][:count].reshape(count, LOCAL_H, HEAD_DIM)
    cos, sin = tensors["rope_cos"][:count], tensors["rope_sin"][:count]
    result = golden_o_proj(attended, cos, sin, tensors["wo_a"], tensors["wo_b"])
    tensors["output"].zero_()
    tensors["output"][:count].copy_(result)


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    from golden import TensorSpec, ratio_allclose, run
    from .rope_tables import prepare_forward_rope_tables

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3"], default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layer", type=int, default=0, choices=range(M.num_hidden_layers))
    parser.add_argument("--tp-rank", type=int, choices=range(TP_SIZE))
    parser.add_argument("--num-tokens", type=int, default=17, choices=range(1, 33))
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    checkpoint = None
    if args.checkpoint is not None:
        from .weights import Checkpoint

        checkpoint = Checkpoint(args.checkpoint)
    cos, sin = prepare_forward_rope_tables(8200)
    generator = torch.Generator().manual_seed(41)
    attended = torch.randn(32, LOCAL_H * HEAD_DIM, generator=generator).bfloat16()
    attended[0].zero_()
    reports = []
    for lane in range(TP_SIZE) if args.tp_rank is None else [args.tp_rank]:
        if checkpoint is None:
            wo_a = torch.randn(LOCAL_O_GROUPS, O_GROUP_IN, O_LORA, generator=generator) / O_GROUP_IN**0.5
            wo_b = torch.randn(LOCAL_O_WIDTH, D, generator=generator) / LOCAL_O_WIDTH**0.5
            wo_a, wo_b = wo_a.bfloat16(), wo_b.bfloat16()
        else:
            prefix = f"layers.{args.layer}.attn."
            wo_a = checkpoint.tensor(prefix + "wo_a.weight").reshape(M.o_groups, O_LORA, O_GROUP_IN)
            wo_a = wo_a[lane * LOCAL_O_GROUPS:(lane + 1) * LOCAL_O_GROUPS].transpose(-1, -2).contiguous()
            wo_b = checkpoint.tensor(prefix + "wo_b.weight")
            wo_b = wo_b[:, lane * LOCAL_O_WIDTH:(lane + 1) * LOCAL_O_WIDTH].t().contiguous()
        values = {"attended": attended, "wo_a": wo_a, "wo_b": wo_b,
                  "rope_cos": cos[0, -32:].contiguous(), "rope_sin": sin[0, -32:].contiguous(),
                  "counts": torch.tensor([args.num_tokens], dtype=torch.int32)}
        specs = [TensorSpec(name, list(value.shape), value.dtype, init_value=value) for name, value in values.items()]
        specs.append(TensorSpec("output", [32, D], torch.bfloat16, init_value=torch.zeros))
        print(f"A3_OUTPUT_PROJECTION layer={args.layer} tp_rank={lane}", flush=True)
        result = run(
            fn=l2_o_proj,
            specs=specs,
            golden_fn=golden_l2_o_proj,
            compare_fn={"output": ratio_allclose(atol=0.003, rtol=1.0 / 128, max_error_ratio=0.0)},
            config={"backend_type": args.platform, "execution_mode": False, "device_id": args.device},
            compile_only=args.compile_only,
        )
        reports.append({"layer": args.layer, "tp_rank": lane, "passed": result.passed,
                        "compile_only": args.compile_only, "build": str(result.work_dir)})
        if args.report:
            args.report.write_text(json.dumps(reports, indent=2) + "\n")
        if not result.passed:
            raise SystemExit(result.error or 1)

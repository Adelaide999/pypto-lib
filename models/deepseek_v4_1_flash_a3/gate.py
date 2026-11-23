# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4.1 MoE FFN router (decode): RMSNorm + gate + topk + normalize."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

import pypto.language as pl
from .gemm import quantize_activation
import torch
import torch.nn.functional as F

from .config import FLASH as M

from .config import MOE_TOKENS, EP_SIZE, N_LOCAL_EXPERTS


FP32_NEG_INF = -3.4028234663852886e38


# model config
T = MOE_TOKENS
D = M.hidden_size
N_EXPERTS = EP_SIZE * N_LOCAL_EXPERTS
TOPK = M.num_experts_per_tok
ROUTE_SCALE = M.routed_scaling_factor
VOCAB = M.vocab_size
N_HASH_LAYERS = 0

# tiling
GATE_INPUT_TILE = 4
# Dense Prefill slabs use one 64-row panel per gate worker.  The 64-row
# panel was the best A3 gate shape in the single-card sweep while retaining
# the 16-row capacity ABI for decode-sized slabs.
GATE_M_TILE = 64 if T >= 64 else 32 if T >= 32 else 16
GATE_N_TILE = 64 if T >= 64 else 16
ROUTE_SELECT_T_TILE = 16
ROUTE_SELECT_WORKERS = 48
GATE_INPUT_WORKERS = 40
GATE_QUANT_WORKERS = 48
GATE_WORKERS = 48
T_PAD = ((T + GATE_M_TILE - 1) // GATE_M_TILE) * GATE_M_TILE
# The 256-wide K tile halves the number of Cube launches for the 5120-wide gate.
GATE_D_TILE = 256
QUANT_TILE = 256
QUANT_TASK_TILE = QUANT_TILE * 4
MX_GROUP = 32
MX_SCALE_GROUPS = 1
SCORE_PAD = 256 if M.name == "flash" else 384
TOPK_PAD = 8
SORT_PAD = TOPK_PAD * 2
QUANT_GROUPS = QUANT_TILE // 32
QUANT_SCALE_TMP = ((64 + (GATE_M_TILE // 16) * QUANT_GROUPS + 31) // 32) * 32


if M.name == "flash":
    @pl.jit.inline
    def route_topk_row(
        score_row: pl.Tensor[[1, 256], pl.FP32],
        idx_init: pl.Tensor[[1, 256], pl.UINT32],
    ) -> pl.Tensor[[1, TOPK_PAD], pl.INT32]:
        """Sort one Flash router row and return its leading expert ids."""

        sorted_32 = pl.sort32(score_row, idx_init)
        sorted_64 = pl.mrgsort(sorted_32, block_len=64)
        prefix = SORT_PAD
        sorted_pairs = pl.mrgsort(
            sorted_64[:, 0:prefix], sorted_64[:, 256 : 256 + prefix],
        )
        return pl.gather(
            sorted_pairs[:, 0:SORT_PAD],
            mask_pattern=pl.tile.MaskPattern.P1010,
            output_dtype=pl.INT32,
        )
else:
    @pl.jit.inline
    def route_topk_row(
        score_row: pl.Tensor[[1, 384], pl.FP32],
        idx_init: pl.Tensor[[1, 384], pl.UINT32],
    ) -> pl.Tensor[[1, TOPK_PAD], pl.INT32]:
        """Sort one Pro router row and return its leading expert ids."""

        sorted_32 = pl.sort32(score_row, idx_init)
        sorted_64 = pl.mrgsort(sorted_32, block_len=64)
        prefix = SORT_PAD
        sorted_pairs = pl.mrgsort(
            sorted_64[:, 0:prefix],
            sorted_64[:, 256 : 256 + prefix],
            sorted_64[:, 512 : 512 + prefix],
        )
        return pl.gather(
            sorted_pairs[:, 0:SORT_PAD],
            mask_pattern=pl.tile.MaskPattern.P1010,
            output_dtype=pl.INT32,
        )


def golden_gate_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    accumulation_dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round the exact BF16 dot to FP32 before sqrt-softplus routing and biased selection."""
    source = x.to(accumulation_dtype)
    matrix = weight.to(accumulation_dtype)
    logits = F.linear(source, matrix).float()
    scores = F.softplus(logits / M.gate_temperature).sqrt()
    indices = torch.argsort(scores + correction_bias.float(), dim=-1, descending=True, stable=True)
    indices = indices[:, :M.num_experts_per_tok]
    weights = scores.gather(-1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights * M.routed_scaling_factor, indices.to(torch.int32)


@pl.jit.inline
def gate_normalized(
    x_normed: pl.Tensor[[T, D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    x_norm_mx: pl.Tensor[[T_PAD, D], pl.INT8],
    x_norm_scale: pl.Tensor[[1, T_PAD * MX_SCALE_GROUPS], pl.FP32],
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
):
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)

    quantized, quant_scale = quantize_activation(x_normed, num_tokens, D)
    scale_view = pl.reshape(x_norm_scale, [T_PAD, 1])
    quant_blocks = T_PAD // 16
    quant_workers = pl.min(quant_blocks, GATE_QUANT_WORKERS)
    for worker in pl.spmd(quant_workers, name_hint="gate_quant_copy"):
        for block in pl.range(worker, quant_blocks, quant_workers):
            row = block * 16
            codes = pl.load(quantized, [row, 0], [16, D])
            x_norm_mx = pl.store(codes, [row, 0], x_norm_mx)
            factor = pl.load(quant_scale, [row, 0], [16, 1])
            scale_view = pl.store(factor, [row, 0], scale_view)
    xg_buf = pl.create_tensor([T_PAD, D], dtype=pl.FP32)
    input_rows = pl.max(active_tokens, pl.cast(1, pl.INDEX))
    input_rows = (input_rows + GATE_M_TILE - 1) // GATE_M_TILE * GATE_M_TILE
    input_blocks = input_rows // GATE_INPUT_TILE
    input_workers = pl.min(input_blocks, GATE_INPUT_WORKERS)
    for worker in pl.spmd(input_workers, name_hint="gate_normalized_input"):
        for block in pl.range(worker, input_blocks, input_workers):
            t0 = block * GATE_INPUT_TILE
            valid_rows = pl.min(GATE_INPUT_TILE, pl.max(active_tokens - t0, 0))
            normalized = pl.slice(x_normed, [GATE_INPUT_TILE, D], [t0, 0], valid_shape=[valid_rows, D])
            normalized = pl.fillpad(normalized, pad_value=pl.PadValue.zero)
            normalized = pl.set_validshape(normalized, GATE_INPUT_TILE, D)
            xg_buf[t0 : t0 + GATE_INPUT_TILE, :] = pl.cast(normalized, pl.FP32)

    return gate_from_fp32(
        xg_buf, gate_w, gate_bias, num_tokens, indices, weights,
    )


@pl.jit.inline
def gate_from_fp32(
    xg_buf: pl.Tensor[[T_PAD, D], pl.FP32],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
):
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)
    task_tokens = pl.max(active_tokens, pl.cast(1, pl.INDEX))
    task_gate_tiles = (task_tokens + GATE_M_TILE - 1) // GATE_M_TILE

    biased_scores_buf = pl.create_tensor([T_PAD, SCORE_PAD], dtype=pl.FP32)
    if active_tokens < T:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_pre_route"):
            for zt in pl.range(active_tokens, T):
                for zk in pl.range(TOPK):
                    zero_index = pl.cast(0, pl.INT32)
                    pl.write(indices, [zt, zk], zero_index)
                    zero_weight = pl.cast(0.0, pl.FP32)
                    pl.write(weights, [zt, zk], zero_weight)
    if N_EXPERTS < SCORE_PAD:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_score_pad"):
            biased_pad = pl.full([T_PAD, SCORE_PAD - N_EXPERTS], dtype=pl.FP32, value=FP32_NEG_INF)
            biased_scores_buf[:, N_EXPERTS:SCORE_PAD] = biased_pad

    route_scores_buf = pl.create_tensor([T_PAD, SCORE_PAD], dtype=pl.FP32)
    # Reserve the available clusters before any local Cube-to-Vector transfer.
    gate_bias_matrix = pl.reshape(gate_bias, [1, N_EXPERTS])
    gate_tasks = task_gate_tiles * ((N_EXPERTS + GATE_N_TILE - 1) // GATE_N_TILE)
    gate_workers = pl.min(gate_tasks, pl.min(GATE_WORKERS, pl.system.available_cluster_count()))
    for gate_worker in pl.spmd(gate_workers, name_hint="gate"):
        # The gate matrix is streamed once per expert tile. Bypassing L2 avoids
        # evicting the activations and routing buffers used by the following sort.
        pl.set_cache_policy(gate_w, pl.CachePolicy.BYPASS)
        for gb_idx in pl.range(gate_worker, gate_tasks, gate_workers):
            tg = gb_idx // ((N_EXPERTS + GATE_N_TILE - 1) // GATE_N_TILE)
            nb = gb_idx % ((N_EXPERTS + GATE_N_TILE - 1) // GATE_N_TILE)
            t1 = tg * GATE_M_TILE
            n0 = nb * GATE_N_TILE
            gate_logits_tile = pl.tile.create(
                [GATE_M_TILE, GATE_N_TILE],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Acc,
                compact=True,
            )
            for kb in pl.pipeline(0, D // GATE_D_TILE, stage=1):
                gd_kd = kb * GATE_D_TILE
                gd_x = pl.tile.load(xg_buf, [t1, gd_kd], [GATE_M_TILE, GATE_D_TILE])
                valid_n = pl.min(GATE_N_TILE, N_EXPERTS - n0)
                gd_w = pl.tile.load(
                    gate_w,
                    [n0, gd_kd],
                    [GATE_N_TILE, GATE_D_TILE],
                    valid_shape=[valid_n, GATE_D_TILE],
                    target_memory=pl.MemorySpace.Mat,
                )
                gd_w = pl.tile.transpose_view(gd_w)
                gate_logits_tile = pl.matmul_acc(
                    gate_logits_tile, gd_x, gd_w, init_cond=(kb == 0),
                )
            gp_relu = pl.maximum(gate_logits_tile, 0.0)
            gp_abs = pl.abs(gate_logits_tile)
            gp_neg_abs = pl.neg(gp_abs)
            gp_exp_abs = pl.exp(gp_neg_abs)
            gp_exp_plus = pl.add(gp_exp_abs, 1.0)
            gp_softplus_tail = pl.log(gp_exp_plus)
            gp_softplus = pl.add(gp_relu, gp_softplus_tail)
            gp_score = pl.sqrt(gp_softplus)
            route_scores_buf = pl.store(gp_score, [t1, n0], route_scores_buf)
            bias_valid_n = pl.min(GATE_N_TILE, N_EXPERTS - n0)
            gp_bias_row = pl.tile.load(
                gate_bias_matrix,
                [0, n0],
                [1, GATE_N_TILE],
                valid_shape=[1, bias_valid_n],
                target_memory=pl.MemorySpace.Vec,
            )
            gp_biased = pl.col_expand_add(gp_score, gp_bias_row)
            biased_scores_buf = pl.store(gp_biased, [t1, n0], biased_scores_buf)

    # V4.1 uses score-only top-k: the V4 hash-route loop is dropped so that PTOAS
    # does not emit an invalid function for a route_hash path that does no work.
    task_route_tiles = (task_tokens + ROUTE_SELECT_T_TILE - 1) // ROUTE_SELECT_T_TILE
    route_workers = pl.min(task_route_tiles, ROUTE_SELECT_WORKERS)
    for route_worker in pl.spmd(route_workers, name_hint="route_sort"):
        for ts_idx in pl.range(route_worker, task_route_tiles, route_workers):
            t1 = ts_idx * ROUTE_SELECT_T_TILE
            # ptoas pto.tmrgsort requires a single source row. Keep the
            # row-wise sort inside a larger worker tile to amortize launch and
            # synchronization overhead across the prefill batch.
            topk_idx_tile = pl.create_tensor([ROUTE_SELECT_T_TILE, TOPK_PAD], dtype=pl.INT32)
            sr_idx_init = pl.create_tensor([1, SCORE_PAD], dtype=pl.UINT32)
            sr_index_range = pl.arange(0, [1, SCORE_PAD], dtype=pl.UINT32)
            sr_idx_init[:, :] = sr_index_range
            for sr_tt in pl.range(ROUTE_SELECT_T_TILE):
                sr_t = t1 + sr_tt
                sr_row = pl.slice(biased_scores_buf, [1, SCORE_PAD], [sr_t, 0])
                sr_i = route_topk_row(sr_row, sr_idx_init)
                topk_idx_tile[sr_tt : sr_tt + 1, :] = sr_i

            local_scores = pl.create_tensor([ROUTE_SELECT_T_TILE, SCORE_PAD], dtype=pl.FP32)
            local_scores[:, :] = route_scores_buf[t1 : t1 + ROUTE_SELECT_T_TILE, :]
            gather_all = pl.gather(local_scores, dim=-1, index=topk_idx_tile)
            gather_valid = pl.set_validshape(gather_all, ROUTE_SELECT_T_TILE, TOPK)
            topk_vals_pad = pl.fillpad(gather_valid, pad_value=pl.PadValue.zero)
            topk_sum = pl.row_sum(topk_vals_pad)
            denom = pl.reshape(topk_sum, [ROUTE_SELECT_T_TILE, 1])
            denom = pl.add(denom, 1e-20)
            topk_normalized = pl.row_expand_div(topk_vals_pad, denom)
            normalized_weights = pl.mul(topk_normalized, ROUTE_SCALE)
            valid_rows = pl.min(ROUTE_SELECT_T_TILE, active_tokens - t1)
            if valid_rows > 0:
                indices[t1 : t1 + ROUTE_SELECT_T_TILE, :] = pl.set_validshape(
                    topk_idx_tile, valid_rows, TOPK,
                )
                weights[t1 : t1 + ROUTE_SELECT_T_TILE, :] = pl.set_validshape(
                    normalized_weights, valid_rows, TOPK,
                )

    return weights


@pl.jit
def l2_gate(
    source: pl.Tensor[[T, D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    codes: pl.Out[pl.Tensor[[T_PAD, D], pl.INT8]],
    scales: pl.Out[pl.Tensor[[1, T_PAD], pl.FP32]],
    indices: pl.Out[pl.Tensor[[T, TOPK], pl.INT32]],
    weights: pl.Out[pl.Tensor[[T, TOPK], pl.FP32]],
):
    gate_normalized(source, gate_weight, gate_bias, num_tokens, codes, scales, indices, weights)
    return codes, scales, indices, weights


def golden_gate(tensors):
    """Validate exact activation codes, FP32 scales, all six routes and inactive zeros."""
    from .gemm import golden_activation_int8

    active = int(tensors["num_tokens"])
    source = tensors["source"].clone()
    source[active:] = 0
    tensors["codes"].zero_()
    tensors["codes"][:T].copy_(golden_activation_int8(source))
    scale = (source.float().abs().amax(-1) * (1.0 / 127.0)).clamp_min(1e-30)
    tensors["scales"].zero_()
    tensors["scales"][:, :T].copy_(scale.reshape(1, T))
    tensors["indices"].zero_()
    tensors["weights"].zero_()
    if active:
        weights, indices = golden_gate_reference(source[:active], tensors["gate_weight"], tensors["gate_bias"])
        tensors["indices"][:active] = indices
        tensors["weights"][:active] = weights


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    from golden import ScalarSpec, TensorSpec, ratio_allclose, run
    from .weights import Checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3"], default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--input-data", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    checkpoint = Checkpoint(args.checkpoint)
    prefix = f"layers.{args.layer}.ffn.gate."
    gate_weight = checkpoint.tensor(prefix + "weight")
    gate_bias = checkpoint.tensor(prefix + "bias")
    source = torch.randn(T, D, generator=torch.Generator().manual_seed(4103)).bfloat16()
    records = []
    cases = [("empty", 0), ("single", 1), ("tail", 17), ("full", T), ("ties", 17)]
    selected_rows = []
    if args.input_data is not None:
        captured = torch.load(args.input_data, map_location="cpu", weights_only=True)
        if not torch.equal(captured["gate_weight"], gate_weight.float()):
            raise ValueError("captured gate weights differ from the selected checkpoint layer")
        if "actual_indices" in captured:
            different = (captured["actual_indices"] != captured["expected_indices"]).any(-1)
            selected_rows = different.nonzero().flatten().tolist()
        if not selected_rows:
            selected_rows = list(range(min(T, captured["source"].shape[0])))
        selected_rows = selected_rows[:T]
        source.zero_()
        source[:len(selected_rows)] = captured["source"][selected_rows]
        gate_bias = captured["gate_bias"]
        cases = [("captured", len(selected_rows))]
    for label, count in cases:
        x = torch.zeros_like(source) if label == "ties" else source
        bias = torch.zeros_like(gate_bias) if label == "ties" else gate_bias
        specs = [TensorSpec("source", [T, D], torch.bfloat16, init_value=x),
                 TensorSpec("gate_weight", [N_EXPERTS, D], torch.float32, init_value=gate_weight),
                 TensorSpec("gate_bias", [N_EXPERTS], torch.float32, init_value=bias),
                 ScalarSpec("num_tokens", torch.int32, count),
                 TensorSpec("codes", [T_PAD, D], torch.int8),
                 TensorSpec("scales", [1, T_PAD], torch.float32),
                 TensorSpec("indices", [T, TOPK], torch.int32),
                 TensorSpec("weights", [T, TOPK], torch.float32)]
        result = run(
            l2_gate,
            specs,
            golden_fn=golden_gate,
            config=dict(backend_type=args.platform, execution_mode=False, device_id=args.device),
            compare_fn={"codes": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
                        "scales": ratio_allclose(atol=0.0, rtol=1e-6, max_error_ratio=0.0),
                        "indices": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
                        "weights": ratio_allclose(atol=1e-6, rtol=3e-5, max_error_ratio=0.0)},
            compile_only=args.compile_only,
        )
        records.append(dict(case=label, count=count, passed=result.passed, work_dir=str(result.work_dir)))
        if not result.passed:
            raise RuntimeError(result.error)
    if args.report:
        args.report.write_text(json.dumps(dict(checkpoint=checkpoint.identity, cases=records,
                                              captured_rows=selected_rows), indent=2) + "\n")

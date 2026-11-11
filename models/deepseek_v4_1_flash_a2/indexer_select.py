# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2/A3 INT8 QLI scoring and Top-K selection.

The A2/A3 DeepSeek V4.1 indexer does not have the A5 FP8/MX path.  Its
selection ABI is nevertheless simple and stable: quantized per-head query
rows are compared with a shared INT8 index-key cache, the two FP16 scales are
applied in FP32, and the positive head scores are multiplied by the FP16
per-head weights.  This module implements that boundary directly in PyPTO.

``indices`` intentionally keeps the score order returned by QLI.  vLLM's
``prepare_indexer_indices`` performs the later visibility filter and
chronological sort; keeping that step separate makes the score/index boundary
independently checkable on A2/A3 and lets the sparse-attention consumer use
either ordering (attention is permutation invariant).
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a2"

import torch
import pypto.language as pl


INDEX_HEADS = 32
INDEX_HEAD_DIM = 128
INDEX_TOPK = 512
KEY_CAPACITY = 512
KEY_TILE = 32
HEAD_TILE = 16
# A2/A3 sort32 is most reliable at this width.  The public kernel signature
# currently fixes the candidate row to this size.
SORT_LEAF = 512
NEGATIVE_INF = -3.0e38


def indexer_select_reference(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    *,
    topk: int = INDEX_TOPK,
    compress_ratio: int = 2,
) -> dict[str, torch.Tensor]:
    """Ideal visibility-aware oracle without the executable kernel's FP16 boundaries.

    ``query`` is ``[T, H, 128]`` INT8, ``key`` is ``[K, 128]`` INT8, and the
    two scale tensors are FP16.  ``positions`` is the raw token position used
    by vLLM to derive the visible compressed-key prefix.
    """

    if query.ndim != 3 or tuple(query.shape[1:]) != (INDEX_HEADS, INDEX_HEAD_DIM):
        raise ValueError("query must have shape [T, 32, 128]")
    if key.ndim != 2 or key.shape[-1] != INDEX_HEAD_DIM:
        raise ValueError("key must have shape [K, 128]")
    tokens, heads, _ = query.shape
    key_rows = key.shape[0]
    if query_scale.shape != (tokens, heads) or weights.shape != (tokens, heads):
        raise ValueError("query_scale and weights must have shape [T, 32]")
    if key_scale.numel() != key_rows or positions.shape != (tokens,):
        raise ValueError("key_scale must be [K] and positions must be [T]")
    if not 1 <= topk <= key_rows:
        raise ValueError("topk must be in [1, key.shape[0]]")
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")

    dots = torch.einsum("thd,kd->thk", query.to(torch.int32), key.to(torch.int32)).float()
    dots.mul_(query_scale.float().unsqueeze(-1))
    dots.mul_(key_scale.float().reshape(1, 1, key_rows))
    scores = torch.relu(dots)
    scores.mul_(weights.float().unsqueeze(-1))
    scores = scores.sum(dim=1)

    indices = torch.full((tokens, topk), -1, dtype=torch.int32)
    visible = ((positions.to(torch.int64) + 1) // compress_ratio).clamp(min=0, max=key_rows)
    for token in range(tokens):
        count = min(int(visible[token]), topk)
        if count:
            # QLI returns score order.  ``sorted=False`` matches the native
            # operator's contract; ties are not expected for random fixtures.
            picked = torch.topk(scores[token, : int(visible[token])], count, sorted=False).indices
            indices[token, :count] = picked.to(torch.int32)
    return {"scores": scores, "indices": indices}


def make_indexer_select(
    capacity: int,
    key_capacity: int,
    *,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
    topk: int = INDEX_TOPK,
    compress_ratio: int = 2,
):
    """Build a fixed-bucket PyPTO QLI score/Top-K kernel."""

    if capacity <= 0 or capacity % HEAD_TILE:
        raise ValueError("capacity must be a positive multiple of 16")
    if heads != INDEX_HEADS or head_dim != INDEX_HEAD_DIM:
        raise ValueError("A2/A3 QLI uses 32 heads x 128")
    # The native A2/A3 QLI handoff allocates one 512-entry candidate row.
    # Smaller visible prefixes are represented by score padding and are
    # filtered by ``prepare_indexer_indices`` in the next boundary.  The
    # fixed signature is deliberate: larger or variable rows are not stable
    # on the current A2/A3 sort32/mrgsort backend.
    if key_capacity != KEY_CAPACITY:
        raise ValueError(f"A2/A3 QLI uses fixed key_capacity={KEY_CAPACITY}")
    if topk != INDEX_TOPK:
        raise ValueError(f"A2/A3 QLI uses fixed topk={INDEX_TOPK}")
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")

    @pl.jit.inline
    def select_one(
        query: pl.Tensor[[capacity, heads, head_dim], pl.INT8],
        query_scale: pl.Tensor[[capacity, heads], pl.FP16],
        key: pl.Tensor[[key_capacity, head_dim], pl.INT8],
        key_scale: pl.Tensor[[key_capacity], pl.FP16],
        weights: pl.Tensor[[capacity, heads], pl.FP16],
        positions: pl.Tensor[[capacity], pl.INT32],
        running_pairs: pl.Tensor[[capacity, 2 * topk], pl.FP32],
        scores: pl.Tensor[[capacity, key_capacity], pl.FP32],
        indices: pl.Tensor[[capacity, topk], pl.INT32],
    ):
        token = pl.tile.get_block_idx()
        # The A2/A3 QLI cache ABI is one fixed 512-entry candidate row.  Sort
        # the complete row here; ``prepare_indexer_indices`` applies the
        # request-specific visible-prefix filter before cache gather.  Keeping
        # this boundary fixed avoids dynamic tile tail writes on A2/A3 while
        # preserving the final selected set.
        visible = key_capacity
        # QLI writes a score tile for every visible key.  Invalid rows never
        # enter the sort, and score padding stays at the hardware minimum.
        # Materialize that floor explicitly: dynamic ``valid_shape`` on a
        # sort32 input is not reliable on the A2/A3 backend for partial rows.
        pl.store(
            pl.tile.full([1, key_capacity], dtype=pl.FP32, value=NEGATIVE_INF),
            [token, 0],
            scores,
        )
        for k0 in pl.range(0, visible, KEY_TILE):
            valid = pl.min(KEY_TILE, visible - k0)
            key_tile = key[k0:k0 + KEY_TILE, :]
            scale_tile = pl.reshape(pl.cast(key_scale[k0:k0 + KEY_TILE], pl.FP32), [1, KEY_TILE])
            q_tile = pl.reshape(query[token:token + 1, :, :], [heads, head_dim])
            dots = pl.matmul(q_tile, key_tile, b_trans=True, out_dtype=pl.INT32)
            dots_fp32 = pl.cast(dots, pl.FP32, mode="none")
            dots_scaled = pl.mul(pl.maximum(dots_fp32, 0.0), 0.0009765625)
            dots_half = pl.cast(dots_scaled, pl.FP16, mode="rint")
            q_scale_row = pl.cast(query_scale[token:token + 1, :], pl.FP32)
            weight_row = pl.cast(weights[token:token + 1, :], pl.FP32)
            combined = pl.mul(q_scale_row, weight_row)
            combined_half = pl.cast(combined, pl.FP16, mode="rint")
            weight_ones = pl.full([HEAD_TILE, heads], dtype=pl.FP16, value=1.0)
            weight_matrix = pl.col_expand_mul(weight_ones, combined_half)
            reduced = pl.matmul(weight_matrix, dots_half, out_dtype=pl.FP32)
            reduced_vector = pl.mul(reduced, 1.0)
            reduced_row = pl.slice(reduced_vector, [1, KEY_TILE], [0, 0])
            total = pl.mul(reduced_row, scale_tile)
            scores[token:token + 1, k0:k0 + KEY_TILE] = pl.fillpad(
                pl.set_validshape(total, 1, valid), pad_value=pl.PadValue.min
            )

        # Match QLI's score-order result with a bounded merge sort.  The
        # payload carries absolute key positions, so no host-side offset is
        # needed for the fixed flattened cache used by the A2/A3 path.
        # The A2/A3 ABI uses one 512-entry QLI candidate row.  Always execute
        # the fixed sort once and represent a short visible prefix with the
        # explicit score floor above; a dynamic ``pl.range(leaf_count)`` leaves
        # the pair arena uninitialized for partial prefixes on A2/A3.
        leaf0 = 0
        row_raw = pl.load(scores, [token, leaf0], [1, SORT_LEAF])
        row = pl.tile.fillpad(row_raw, pad_value=pl.PadValue.min)
        row = pl.maximum(row, pl.tile.full([1, SORT_LEAF], dtype=pl.FP32, value=NEGATIVE_INF))
        ramp = pl.tile.arange(0, [1, SORT_LEAF], dtype=pl.INT32)
        leaf_indices = pl.tile.add(ramp, pl.cast(leaf0, pl.INT32))
        sort_tmp = pl.tile.create([1, 2 * SORT_LEAF], dtype=pl.FP32)
        leaf_pairs = pl.tile.sort32(
            row, pl.reinterpret_view(leaf_indices, pl.UINT32), tmp=sort_tmp
        )
        leaf_pairs = pl.tile.mrgsort(leaf_pairs, block_len=64)
        leaf_pairs = pl.tile.mrgsort(leaf_pairs, block_len=256)
        leaf_pairs = pl.tile.slice(leaf_pairs, [1, 2 * topk], [0, 0])
        pl.store(leaf_pairs, [token, 0], running_pairs)

        # Materialize the complete padding row in one UB tile.  Explicitly
        # initialize every lane because the backend may preserve stale pair
        # lanes when only a dynamic prefix is written.
        selected = pl.tile.full([1, topk], dtype=pl.INT32, value=-1)
        if visible > 0:
            pairs = pl.load(running_pairs, [token, 0], [1, 2 * topk])
            picked = pl.tile.gather_mask(pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            selected = picked
        pl.store(selected, [token, 0], indices)
        return scores, indices

    @pl.jit
    def kernel(
        query: pl.Tensor[[capacity, heads, head_dim], pl.INT8],
        query_scale: pl.Tensor[[capacity, heads], pl.FP16],
        key: pl.Tensor[[key_capacity, head_dim], pl.INT8],
        key_scale: pl.Tensor[[key_capacity], pl.FP16],
        weights: pl.Tensor[[capacity, heads], pl.FP16],
        positions: pl.Tensor[[capacity], pl.INT32],
        scores: pl.Out[pl.Tensor[[capacity, key_capacity], pl.FP32]],
        indices: pl.Out[pl.Tensor[[capacity, topk], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        running_pairs = pl.create_tensor([capacity, 2 * topk], dtype=pl.FP32)
        for _worker in pl.spmd(active, name_hint="dsv41_a2a3_qli_select"):
            select_one(
                query, query_scale, key, key_scale, weights, positions,
                running_pairs, scores, indices,
            )
        return scores, indices

    return kernel


def golden_indexer_select(tensors):
    """Evaluate all 512 candidates with the kernel's two FP16 narrowing boundaries."""
    active = int(tensors["active"])
    if not 0 <= active <= tensors["query"].shape[0]:
        raise ValueError("Indexer active rows must fit the actual capacity")
    if not active:
        return
    key = tensors["key"].to(torch.int32)
    for token in range(active):
        dots = tensors["query"][token].to(torch.int32) @ key.T
        positive = dots.float().clamp_min(0.0) * 0.0009765625
        narrowed_dots = positive.to(torch.float16).float()
        combined = tensors["query_scale"][token].float() * tensors["weights"][token].float()
        narrowed_weight = combined.to(torch.float16).float()
        reduced = narrowed_weight[None] @ narrowed_dots
        scores = reduced[0] * tensors["key_scale"].float()
        tensors["scores"][token] = scores
        tensors["indices"][token] = torch.argsort(scores, descending=True).to(torch.int32)


def indexer_active_compare(active, *, exact=False):
    """Compare the defined prefix; pure Out inactive rows have no preservation contract."""
    from golden.validation import ratio_allclose

    tolerance = 0.0 if exact else 0.05
    comparator = ratio_allclose(rtol=tolerance, atol=tolerance, max_error_ratio=0.0)

    def compare(actual, expected, **context):
        if not active:
            return True, "No active pure-Out rows; inactive rows are undefined"
        return comparator(actual[:active], expected[:active], **context)

    return compare


def indexer_select_golden_cases(*, build=True):
    """Use unique positive scores, negative-dot heads and ignored visibility positions."""
    from golden import ScalarSpec, TensorSpec

    capacity = 16
    rows = torch.arange(KEY_CAPACITY, dtype=torch.int32)
    key = torch.ones((KEY_CAPACITY, INDEX_HEAD_DIM), dtype=torch.int8)
    key[:, 0] = (rows // 4).to(torch.int8)
    key[:, 1] = (rows % 4).to(torch.int8)
    query = torch.ones((capacity, INDEX_HEADS, INDEX_HEAD_DIM), dtype=torch.int8)
    query[:, :, 0] = 4
    query[:, ::3] *= -1
    heads = torch.arange(INDEX_HEADS, dtype=torch.float32)
    query_scale = ((heads % 7 + 8.0) / 17.0).repeat(capacity, 1).to(torch.float16)
    weights = ((heads % 5 + 3.0) / 11.0).repeat(capacity, 1).to(torch.float16)
    positions = torch.tensor([-1, 0, 1, 31, 127, 511, 1023, 2047] * 2, dtype=torch.int32)
    for ratio in (1, 2):
        for active in (0, 1, 7, capacity):
            specs = [
                TensorSpec("query", [capacity, INDEX_HEADS, INDEX_HEAD_DIM], torch.int8, init_value=query),
                TensorSpec("query_scale", [capacity, INDEX_HEADS], torch.float16, init_value=query_scale),
                TensorSpec("key", [KEY_CAPACITY, INDEX_HEAD_DIM], torch.int8, init_value=key),
                TensorSpec("key_scale", [KEY_CAPACITY], torch.float16, init_value=0.75),
                TensorSpec("weights", [capacity, INDEX_HEADS], torch.float16, init_value=weights),
                TensorSpec("positions", [capacity], torch.int32, init_value=positions),
                TensorSpec("scores", [capacity, KEY_CAPACITY], torch.float32, init_value=-53.0),
                TensorSpec("indices", [capacity, INDEX_TOPK], torch.int32, init_value=-59),
                ScalarSpec("active", torch.int32, active, compile_runtime=True),
            ]
            case = dict(name=f"indexer-fixed512-ratio{ratio}-active{active}", specs=specs,
                        golden_fn=golden_indexer_select,
                        compare_fn=dict(scores=indexer_active_compare(active),
                                        indices=indexer_active_compare(active, exact=True)))
            if build:
                case["fn"] = make_indexer_select(capacity, KEY_CAPACITY, compress_ratio=ratio)
            yield case


def validate(*, device=0, sim=False, compile_only=False, cpu_only=False):
    """Run fixed-row kernel goldens without applying the later visibility filter."""
    from golden import run

    if cpu_only and compile_only:
        raise ValueError("CPU golden checks and kernel compilation are separate modes")
    records = []
    for case in indexer_select_golden_cases(build=not cpu_only):
        name = case.pop("name")
        if cpu_only:
            values = {spec.name: spec.create_tensor().clone() if hasattr(spec, "create_tensor") else spec.to_python()
                      for spec in case["specs"]}
            before = {key: value.clone() for key, value in values.items() if isinstance(value, torch.Tensor)}
            case["golden_fn"](values)
            active = int(values["active"])
            for key in before:
                if key not in ("scores", "indices") and not torch.equal(values[key], before[key]):
                    raise RuntimeError("Indexer golden mutated input " + key)
            if active:
                scores = values["scores"][:active]
                ordered = torch.sort(scores, dim=1).values
                if not (torch.diff(ordered, dim=1) > 0).all():
                    raise RuntimeError("The exact-index fixture must contain no tied scores")
                expected = torch.arange(KEY_CAPACITY - 1, -1, -1, dtype=torch.int32).repeat(active, 1)
                if not torch.equal(values["indices"][:active], expected):
                    raise RuntimeError("The fixed 512-row selection changed its exact score ordering")
                if not torch.isfinite(scores).all():
                    raise RuntimeError("Indexer golden produced nonfinite scores")
            records.append(dict(name=name, cpu_golden="passed", device="not_run", inactive_Out_rows="undefined"))
            continue
        result = run(
            **case,
            config=dict(backend_type="a2a3", execution_mode=bool(sim), device_id=device),
            compile_only=compile_only,
            rtol=0.05,
            atol=0.05,
        )
        if not result.passed:
            raise RuntimeError(f"{name}: {result.error}")
        records.append(dict(name=name, compiled=True, device="not_run" if compile_only else "passed",
                            inactive_Out_rows="undefined"))
    return records


__all__ = ["INDEX_HEADS", "INDEX_HEAD_DIM", "INDEX_TOPK", "indexer_select_reference", "make_indexer_select",
           "golden_indexer_select", "indexer_select_golden_cases", "validate"]


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3"], default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(device=args.device, sim=args.sim, compile_only=args.compile_only, cpu_only=args.cpu_only)))

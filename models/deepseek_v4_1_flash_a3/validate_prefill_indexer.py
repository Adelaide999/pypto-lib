# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate packed Prefill Indexer results against independent request-local Torch references."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

import pypto.language as pl
from golden import ScalarSpec, TensorSpec, ratio_allclose, run

from .activation_quant import quantize_projection_activation
from .config import CMP_BLOCKS_DYN, INDEX_BLOCKS_DYN, INDEX_POOL_BLOCKS_DYN, ORI_BLOCKS_DYN, T_DYN
from .prefill_indexer import (
    D, HEAD_DIM, IDX_PACKED, IDX_SCALES, INDEX_DIM, INDEX_H, INDEX_QUERY_WIDTH, INDEX_TOPK, Q_LORA, ROPE_DIM,
    golden_prefill_indexer, golden_select_candidate_block_pool, index_pool_block_capacity, prefill_indexer,
)
from .quant import dequantize_mxfp4_cache, dequantize_mxfp8_cache, quantize_mxfp4_cache, quantize_mxfp8_cache
from .sparse_attention import KEY_TILE, LOCAL_H, TOKEN_TILE, attend_packed, golden_paged_sparse_attention


# Logical table capacity, followed by (request slot, visible compressed keys) for each query.
CASES = {
    "packed-short": (1048576, [(2, n) for n in (0, 1, 511, 512, 513, 1023, 1024, 1025)]),
    "packed-mixed": (1048576, [(2, n) for n in (32767, 32768, 32769, 33005)] +
                     [(0, n) for n in (511, 512, 513, 2047, 2048, 2049)]),
    "interleaved": (4096, [(1, 2047), (0, 2048), (1, 2049), (2, 4095), (0, 1025), (1, 1)]),
    "same-bucket": (65536, [(slot, n + slot) for n in (1023, 1024, 2048, 4095) for slot in (2, 0)]),
    "mixed-buckets": (65536, [(1, 0), (0, 65535), (2, 16383), (1, 512), (0, 65536),
                              (2, 16384), (1, 513), (2, 16385), (0, 32769)]),
    "c2a-mixed-buckets": (65536, [(2, 511), (1, 16385), (0, 32769), (2, 512),
                                  (0, 32768), (1, 16384), (2, 513), (0, 65025)]),
    "pool-interleaved": (65536, [(2, 65535), (0, 513), (2, 65536), (0, 1025), (2, 32769)]),
    "c2a-packed": (524288, [(2, n) for n in (0, 1, 511, 512, 513)] + [(1, 32768), (1, 32769)]),
    "pool-sparse": (1048576, [(1, 33005), (1, 33117), (0, 1048573), (0, 1048576)]),
    "pool-holes": (1048576, [(2, 0), (2, 513), (2, 32769), (1, 1048573)]),
    "pool-dense": (8192, [(1, 511), (1, 513), (1, 8001), (0, 8192)]),
    "published-pool": (65536, [(2, 32767), (2, 32768), (2, 32769), (1, 1025), (1, 513)]),
    "published-interleaved": (65536, [(2, 32769), (1, 512), (2, 33005), (1, 1025), (2, 65535)]),
    "many-slots": (65536, [(32, 32769), (1, 1025), (31, 16385), (32, 33005), (34, 0), (1, 513)]),
    "stream-ties": (65536, [(1, 32767), (1, 32768), (1, 32769), (1, 33280)]),
    "band-ties": (131072, [(2, 65535), (0, 16383), (2, 65536), (0, 16384), (2, 65537)]),
    "stream-empty-prefix": (65536, [(1, 0)] * 64 + [(1, 32769)]),
    "c2a-stream-short-prefix": (65536, [(1, 512)] * 64 + [(1, 32769)]),
    "c1a-million": (1048576, [(1, 1048575), (1, 1048576)]),
    "c2a-million": (524288, [(2, 512), (1, 524287), (1, 524288)]),
    "empty": (4096, []),
}
_POOL = False
_C2A = False
_ATTEND = False
_PUBLISHED = False


@pl.jit(auto_scope=False)
def check_indexer(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    latent: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    requests: pl.Tensor[[T_DYN], pl.INT32],
    lengths: pl.Tensor[[T_DYN], pl.INT32],
    cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_PACKED], pl.UINT8],
    scales: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_SCALES], pl.UINT8],
    table: pl.Tensor,
    cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    wq: pl.Tensor[[Q_LORA, INDEX_QUERY_WIDTH], pl.INT8],
    wscale: pl.Tensor[[1, INDEX_QUERY_WIDTH], pl.FP32],
    wp: pl.Tensor[[D, INDEX_H], pl.BF16, pl.ND],
    ids: pl.InOut[pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    counts: pl.InOut[pl.Tensor[[T_DYN], pl.INT32]],
    indices: pl.InOut[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    attn_q: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    wi: pl.Tensor[[T_DYN, 128], pl.INT32],
    window: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
    ws: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    compressed: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    cs: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 16], pl.BF16],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    attended: pl.InOut[pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16]],
    active: pl.Scalar[pl.INT32],
):
    x.bind_dynamic(0, T_DYN)
    ids.bind_dynamic(1, INDEX_POOL_BLOCKS_DYN)
    cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    compressed.bind_dynamic(0, CMP_BLOCKS_DYN)
    window.bind_dynamic(0, ORI_BLOCKS_DYN)
    payload, scales_q = quantize_projection_activation(latent, active, Q_LORA, True, True)
    initial = pl.system.task_dummy(deps=[])
    if _PUBLISHED:
        first_output = pl.create_tensor([pl.tensor.dim(x, 0), INDEX_TOPK], dtype=pl.INT32)
        initial = prefill_indexer(
            x, payload, scales_q, requests, lengths, cache, scales, table, sin, cos, wq, wscale, wp,
            ids, counts, first_output, active, initial, pool=False, c2a=False,
        )
    selected = prefill_indexer(
        x, payload, scales_q, requests, lengths, cache, scales, table, cos, sin, wq, wscale, wp,
        ids, counts, indices, active, initial, pool=_POOL, c2a=_C2A,
    )
    if _ATTEND:
        if active > 0:
            live = pl.cast(pl.min(active, pl.tensor.dim(x, 0)), pl.INT32)
            attend_packed(
                attn_q, wi, window, ws, indices, compressed, cs, sink, attended,
                live, selected, KEY_TILE, TOKEN_TILE, 0,
            )
    return ids, counts, indices, attended


def make_specs(case):
    width, runs = CASES[case]
    n = len(runs)
    rows = n + 2
    generator = torch.Generator().manual_seed(7991)

    def random(*shape):
        return torch.randn(shape, generator=generator).bfloat16()

    # A wide logical table with a bounded physical cache exercises paging without large fixtures.
    pages = 263
    permutation = torch.randperm(pages, generator=generator)
    logical = torch.arange(width // 128)
    slots = max(3, max((slot + 1 for slot, _ in runs), default=0))
    table = torch.stack([permutation[(logical + slot * 29) % pages] for slot in range(slots)]).int()
    if case != "pool-sparse":
        table[:, 1] = -1
        table[:, 4] = pages
    if case == "stream-ties":
        table.fill_(-1)
        for page in (0, 253, 255, 256, 257, 259):
            table[1, page] = permutation[page]
    cache, scales = quantize_mxfp4_cache(random(pages, 128, 1, INDEX_DIM), 32, "e8m0", round_to_even=True)
    latent = random(rows, Q_LORA)
    if case in {"stream-ties", "band-ties"}:
        latent.zero_()
    req = torch.tensor([slot for slot, _ in runs] + [-1, -1], dtype=torch.int32)
    lengths = torch.tensor([length for _, length in runs] + [width, width], dtype=torch.int32)
    capacity = index_pool_block_capacity(width)
    ids = torch.full((rows, capacity), -17, dtype=torch.int32)
    counts = torch.full((rows,), -17, dtype=torch.int32)
    if _POOL and not _PUBLISHED:
        for token, (_, length) in enumerate(runs):
            blocks = (length + 7) // 8
            stride = 1 if case == "pool-dense" else max(2, blocks // 93)
            selected = sorted(set(range(0, blocks, stride)) | {blocks - 1}) if blocks else []
            if case == "pool-holes" and blocks > 4:
                selected = sorted(set(selected) | {0, 16, 64}) if blocks > 64 else selected
                selected = [block for block in selected if block < blocks - 1]
            ids[token, :len(selected)] = torch.tensor(selected)
            counts[token] = len(selected)
    angle = torch.randn((rows, ROPE_DIM // 2), generator=generator)
    window, ws = quantize_mxfp8_cache(random(1, 128, 1, HEAD_DIM))
    compressed_pages = pages if _ATTEND else 1
    compressed, cs = quantize_mxfp4_cache(random(compressed_pages, 128, 1, HEAD_DIM), 16, "bf16")
    values = dict(
        x=random(rows, D), latent=latent, requests=req, lengths=lengths, cache=cache, scales=scales, table=table,
        cos=angle.cos(), sin=angle.sin(), wq=torch.randint(-8, 9, (Q_LORA, INDEX_QUERY_WIDTH),
                                                           dtype=torch.int8, generator=generator),
        wscale=torch.full((1, INDEX_QUERY_WIDTH), 0.002), wp=random(D, INDEX_H).mul_(0.125),
        ids=ids, counts=counts, indices=torch.full((rows, INDEX_TOPK), -17, dtype=torch.int32),
        attn_q=random(rows, LOCAL_H * HEAD_DIM), wi=torch.arange(128, dtype=torch.int32).repeat(rows, 1),
        window=window.view(torch.uint8), ws=ws, compressed=compressed, cs=cs, sink=torch.zeros(LOCAL_H),
        attended=torch.full((rows, LOCAL_H * HEAD_DIM), -17, dtype=torch.bfloat16),
    )
    return [TensorSpec(k, list(v.shape), v.dtype, init_value=v) for k, v in values.items()] + [
        ScalarSpec("active", torch.int32, n),
    ]


def golden(t):
    n = int(t["active"])
    if _PUBLISHED or (not _C2A and not _POOL):
        t["ids"].fill_(-1)
        t["counts"].zero_()
    keys = dequantize_mxfp4_cache(t["cache"], t["scales"], 32, "e8m0").bfloat16()
    for slot in t["requests"][:n].unique().tolist():
        rows = torch.where(t["requests"][:n] == slot)[0]
        lengths = t["lengths"][rows]
        if _PUBLISHED:
            initial_scores, _ = golden_prefill_indexer(
                t["x"][rows], t["latent"][rows], torch.zeros_like(lengths), keys, t["table"][slot:slot + 1],
                lengths, t["wq"], t["wscale"], t["wp"], t["sin"][rows], t["cos"][rows],
            )
            ids, counts = golden_select_candidate_block_pool(initial_scores, lengths)
            t["ids"][rows] = F.pad(ids, (0, t["ids"].shape[1] - ids.shape[1]), value=-1)
            t["counts"][rows] = counts
        scores, chosen = golden_prefill_indexer(
            t["x"][rows], t["latent"][rows], torch.zeros_like(lengths), keys, t["table"][slot:slot + 1],
            lengths, t["wq"], t["wscale"], t["wp"], t["cos"][rows], t["sin"][rows], c2a=_C2A,
            pool_block_ids=t["ids"][rows] if _POOL else None, pool_block_counts=t["counts"][rows] if _POOL else None,
        )
        t["indices"][rows] = chosen
        if not _C2A and not _POOL:
            ids, counts = golden_select_candidate_block_pool(scores, lengths)
            t["ids"][rows] = F.pad(ids, (0, t["ids"].shape[1] - ids.shape[1]), value=-1)
            t["counts"][rows] = counts
    if _ATTEND and n:
        window = dequantize_mxfp8_cache(t["window"].view(torch.float8_e4m3fn), t["ws"]).bfloat16()
        compressed = dequantize_mxfp4_cache(t["compressed"], t["cs"], 16, "bf16").bfloat16()
        q = t["attn_q"][:n].reshape(n, LOCAL_H, HEAD_DIM)
        expected = golden_paged_sparse_attention(q, window, t["wi"][:n], compressed, t["indices"][:n], t["sink"])
        t["attended"][:n] = expected.flatten(1)


def main():
    global _POOL, _C2A, _ATTEND, _PUBLISHED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="interleaved")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--runtime-dir", help="replay a compatible precompiled validation graph")
    parser.add_argument("--golden-data")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    args = parser.parse_args()
    torch.set_num_threads(4)
    _PUBLISHED = args.case.startswith("published-")
    _POOL = args.case.startswith("pool-") or _PUBLISHED
    _C2A, _ATTEND = args.case.startswith("c2a-"), args.attention
    exact = ratio_allclose(atol=0, rtol=0, max_error_ratio=0)
    cfg = dict(backend_type="a2a3", execution_mode=False, device_id=args.device, ring_heap=1024 * 1024**2,
               enable_chip_swimlane=args.enable_chip_swimlane)
    if args.output:
        cfg.update(save_kernels=True, save_kernels_dir=args.output)
    result = run(
        check_indexer,
        make_specs(args.case),
        golden_fn=golden,
        config=cfg,
        compare_fn=dict(ids=exact, counts=exact, indices=exact,
                        attended=ratio_allclose(atol=0.003, rtol=0.015, max_error_ratio=0.001)),
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()

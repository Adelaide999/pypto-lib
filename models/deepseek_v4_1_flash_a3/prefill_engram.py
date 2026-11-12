# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill Engram hash/history, TP lookup layout, SP sharding and original-basis gate."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

import pypto.language as pl

from .engram_ecotech import (
    COLS, D, HC, K, N, WIDTH,
    make_original_basis_gate, make_original_basis_hidden,
)
from .qkv_proj_rope import make_bf16_projection_staged


# tiling
D_TILE = 256


def make_prefill_engram_hash(capacity, vocab, layers, cache_rows, max_blocks, image_id, image_pad_id, pad_id):
    """Publish compressed slot history and compute both layers' exact INT64 rolling hashes."""

    @pl.jit
    def hash_history(
        token_ids: pl.Tensor[[capacity], pl.INT64],
        positions: pl.Tensor[[capacity], pl.INT64],
        query_start: pl.Tensor[[capacity + 1], pl.INT32],
        slots: pl.Tensor[[capacity], pl.INT64],
        block_table: pl.Tensor[[capacity, max_blocks], pl.INT32],
        lookback_ids: pl.Tensor[[capacity, 3], pl.INT64],
        token_map: pl.Tensor[[vocab], pl.INT32],
        multipliers: pl.Tensor[[layers, 4], pl.INT64],
        primes: pl.Tensor[[layers, COLS], pl.INT64],
        offsets: pl.Tensor[[layers, COLS], pl.INT64],
        cache: pl.InOut[pl.Tensor[[cache_rows], pl.INT32]],
        hashes: pl.InOut[pl.Tensor[[capacity, layers, COLS], pl.INT32]],
        token_mask: pl.InOut[pl.Tensor[[capacity], pl.INT32]],
        active: pl.Scalar[pl.INT32],
        requests: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(1, name_hint="prefill_engram_history_publish") as publish_tid:
            publish_block = pl.tile.get_block_idx()
            for token in pl.range(publish_block, active):
                token_id = pl.read(token_ids, [token])
                eligible = (token_id != image_id) and (token_id != image_pad_id)
                pl.write(token_mask, [token], pl.cast(eligible, pl.INT32))
                compressed = pl.read(token_map, [pl.cast(token_id, pl.INDEX)])
                if not eligible:
                    compressed = pl.cast(-1, pl.INT32)
                slot = pl.cast(pl.read(slots, [token]), pl.INDEX)
                if slot >= 0:
                    if slot < cache_rows:
                        pl.write(cache, [slot], compressed)
        for hash_block in pl.spmd(1, name_hint="prefill_engram_hash", deps=[publish_tid]):
            for task in pl.range(hash_block, active * layers):
                token, layer = task // layers, task % layers
                request: pl.Scalar[pl.INDEX] = 0
                for candidate in pl.range(requests):
                    query_begin = pl.read(query_start, [candidate])
                    end = pl.read(query_start, [candidate + 1])
                    if token >= query_begin:
                        if token < end:
                            request = candidate
                begin = pl.cast(pl.read(query_start, [request]), pl.INDEX)
                chunk_start = pl.read(positions, [begin])
                position = pl.read(positions, [token])
                rolling: pl.Scalar[pl.INT64] = 0
                blocked: pl.Scalar[pl.INT32] = 0
                for shift in pl.range(4):
                    history_position = position - shift
                    value: pl.Scalar[pl.INT32] = pad_id
                    source_dead: pl.Scalar[pl.INT32] = 0
                    if history_position >= chunk_start:
                        source_token = pl.max(token - shift, 0)
                        source_id = pl.read(token_ids, [source_token])
                        value = pl.read(token_map, [pl.cast(source_id, pl.INDEX)])
                        if source_id == image_id or source_id == image_pad_id:
                            source_dead = pl.cast(1, pl.INT32)
                    else:
                        history_column = pl.cast(chunk_start - 1 - history_position, pl.INDEX)
                        lookback_id: pl.Scalar[pl.INT64] = -1
                        if history_column >= 0:
                            if history_column < 3:
                                lookback_id = pl.read(lookback_ids, [request, history_column])
                        if lookback_id >= 0:
                            value = pl.read(token_map, [pl.cast(lookback_id, pl.INDEX)])
                            if lookback_id == image_id or lookback_id == image_pad_id:
                                source_dead = pl.cast(1, pl.INT32)
                        else:
                            bounded_position = pl.min(pl.max(history_position, 0), max_blocks * 128 - 1)
                            block_column = pl.cast(bounded_position // 128, pl.INDEX)
                            block = pl.read(block_table, [request, block_column])
                            history_slot = pl.cast(block, pl.INT64) * 128 + bounded_position % 128
                            bounded_slot = pl.cast(pl.min(pl.max(history_slot, 0), cache_rows - 1), pl.INDEX)
                            value = pl.read(cache, [bounded_slot])
                    if history_position < 0 or source_dead != 0 or value == -1:
                        blocked = pl.cast(1, pl.INT32)
                    if blocked != 0:
                        value = pl.cast(pad_id, pl.INT32)
                    multiplier = pl.read(multipliers, [layer, shift])
                    product = pl.cast(value, pl.INT64) * multiplier
                    rolling = rolling ^ product
                    if shift > 0:
                        for head in pl.range(8):
                            column = (shift - 1) * 8 + head
                            prime = pl.read(primes, [layer, column])
                            offset = pl.read(offsets, [layer, column])
                            row_id = pl.cast(rolling % prime + offset, pl.INT32)
                            pl.write(hashes, [token, layer, column], row_id)
        return cache, hashes, token_mask

    return hash_history


def make_prefill_engram_layout(capacity, tp):
    """Unpack TP column buckets and shard complete rows in original token order."""
    columns = COLS // tp
    local_width = columns * WIDTH

    @pl.jit
    def lookup_layout(
        packet: pl.Tensor[[tp * capacity, local_width], pl.BF16],
        token_mask: pl.Tensor[[capacity], pl.INT32],
        full: pl.InOut[pl.Tensor[[capacity, K], pl.BF16]],
        local: pl.InOut[pl.Tensor[[capacity, K], pl.BF16]],
        local_mask: pl.InOut[pl.Tensor[[capacity], pl.INT32]],
        physical: pl.Scalar[pl.INT32],
        active: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(active * COLS, name_hint="prefill_engram_tp_unpack") as unpack_tid:
            unpack_task = pl.tile.get_block_idx()
            unpack_token = unpack_task // COLS
            unpack_column = unpack_task % COLS
            unpack_peer = unpack_column // columns
            unpack_local_column = unpack_column % columns
            unpack_source = unpack_peer * capacity + unpack_token
            unpack_start = unpack_local_column * WIDTH
            unpack_values = pl.load(packet, [unpack_source, unpack_start], [1, WIDTH], target_memory=pl.MemorySpace.Vec)
            pl.store(unpack_values, [unpack_token, unpack_column * WIDTH], full)
        for shard_task in pl.spmd(physical * COLS, name_hint="prefill_engram_sp_shard", deps=[unpack_tid]):
            shard_token = shard_task // COLS
            shard_column = shard_task % COLS
            shard_source = tp_rank * physical + shard_token
            shard_start = shard_column * WIDTH
            if shard_source < active:
                shard_values = pl.load(full, [shard_source, shard_start], [1, WIDTH], target_memory=pl.MemorySpace.Vec)
                pl.store(shard_values, [shard_token, shard_start], local)
            else:
                shard_zeros = pl.tile.full([1, WIDTH], dtype=pl.BF16, value=0.)
                pl.store(shard_zeros, [shard_token, shard_start], local)
        for mask_block in pl.spmd(1, name_hint="prefill_engram_mask_publish", deps=[unpack_tid]):
            for mask_token in pl.range(mask_block, physical):
                mask_source = tp_rank * physical + mask_token
                eligible: pl.Scalar[pl.INT32] = 0
                if mask_source < active:
                    eligible = pl.read(token_mask, [mask_source])
                pl.write(local_mask, [mask_token], eligible)
        return full, local, local_mask

    return lookup_layout


def make_prefill_engram_gate(capacity, eps):
    """Restore hidden and apply FP32 channel products and the original-basis gate."""
    restore = make_original_basis_hidden(capacity)
    gate = make_original_basis_gate(capacity, eps)

    @pl.jit
    def original_basis_gate(
        hidden: pl.Tensor[[capacity, HC, D], pl.BF16],
        projected: pl.Tensor[[capacity, N], pl.BF16],
        q_weight: pl.Tensor[[HC, D], pl.BF16],
        k_weight: pl.Tensor[[HC, D], pl.BF16],
        rotation: pl.Tensor[[32, 32], pl.FP32],
        token_mask: pl.Tensor[[capacity], pl.INT32],
        original: pl.InOut[pl.Tensor[[capacity, HC, D], pl.FP32]],
        probabilities: pl.InOut[pl.Tensor[[capacity, HC], pl.FP32]],
        output: pl.InOut[pl.Tensor[[capacity, HC, D], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        channel_weight = pl.create_tensor([HC, D], dtype=pl.FP32)
        for task in pl.spmd(HC * (D // D_TILE), name_hint="prefill_engram_channel_weight"):
            branch, start = task // (D // D_TILE), task % (D // D_TILE) * D_TILE
            q = pl.cast(q_weight[branch:branch + 1, start:start + D_TILE], pl.FP32)
            k = pl.cast(k_weight[branch:branch + 1, start:start + D_TILE], pl.FP32)
            channel_weight[branch:branch + 1, start:start + D_TILE] = pl.mul(q, k)
        restore(hidden, rotation, original, active)
        gate(hidden, original, projected, channel_weight, token_mask, probabilities, output, active)
        return original, probabilities, output

    return original_basis_gate


def make_prefill_engram_entries(capacity, constants, config, shards, cache_rows, max_blocks, tp=4):
    """Construct bounded hash/layout diagnostics and ready-row projection/gate entries."""
    if tp not in (1, 2, 4) or len(shards) != tp or COLS % tp:
        raise ValueError("Require complete TP1/2/4 Engram column ownership")
    columns = COLS // tp
    entries = {}
    for rank, shard in enumerate(shards):
        if shard["head_range"] != [rank * columns, (rank + 1) * columns]:
            raise ValueError("Require contiguous whole-column Engram shards")
    image_id = config["image_token_id"]
    image_pad_id = config.get("image_pad_token_id", image_id + 1)
    hash_args = (capacity, len(constants["token_map"]), len(constants["layer_ids"]), cache_rows, max_blocks)
    entries["engram_hash"] = make_prefill_engram_hash(*hash_args, image_id, image_pad_id, constants["pad_id"])
    entries["engram_layout"] = make_prefill_engram_layout(capacity, tp)
    entries["engram_projection"] = make_bf16_projection_staged(K, N, capacity=capacity)
    entries["engram_gate"] = make_prefill_engram_gate(capacity, config["rms_norm_eps"])
    return entries


def golden_prefill_engram_hash(tensors, *, image_id, image_pad_id, pad_id):
    """Use the independent host history/hash contract on bounded inputs."""
    from .engram_hash import HashLayout, hash_ids, update_slot_history

    active, requests = int(tensors["active"]), int(tensors["requests"])
    tokens = tensors["token_ids"][:active]
    dead = (tokens == image_id) | (tokens == image_pad_id)
    tensors["token_mask"][:active] = (~dead).to(tensors["token_mask"].dtype)
    update_slot_history(tokens, tensors["token_map"], dead, tensors["slots"][:active], tensors["cache"])
    if not active:
        return
    lookback = tensors["lookback_ids"][:requests]
    lookback_dead = (lookback == image_id) | (lookback == image_pad_id)
    layers = tensors["multipliers"].shape[0]
    layout = HashLayout(
        layer_ids=tuple(range(layers)), multipliers=tensors["multipliers"],
        primes=tensors["primes"], offsets=tensors["offsets"], max_ngram_size=4, n_heads=8,
        compressed_vocab_size=int(tensors["token_map"].max()) + 1, pad_token_id=pad_id,
        vocab_size=tensors["token_map"].numel(),
    )
    result = hash_ids(
        tokens, tensors["token_map"], tensors["positions"][:active], tensors["query_start"][:requests + 1],
        layout=layout, dead_mask=dead, lookback_token_ids=lookback, lookback_dead_mask=lookback_dead,
        hash_cache=tensors["cache"], block_table=tensors["block_table"][:requests],
    )
    tensors["hashes"][:active] = result


def golden_prefill_engram_layout(tensors):
    """Evaluate TP column concatenation and contiguous SP token selection."""
    capacity = tensors["full"].shape[0]
    tp = tensors["packet"].shape[0] // capacity
    active, physical, rank = (int(tensors[name]) for name in ("active", "physical", "tp_rank"))
    buckets = tensors["packet"].reshape(tp, capacity, COLS // tp, WIDTH)
    full = buckets.permute(1, 0, 2, 3).reshape(capacity, K)
    tensors["full"][:active] = full[:active]
    tensors["local"][:physical] = 0
    tensors["local_mask"][:physical] = 0
    start = rank * physical
    valid = max(0, min(physical, active - start))
    tensors["local"][:valid] = full[start:start + valid]
    tensors["local_mask"][:valid] = tensors["token_mask"][start:start + valid]


def golden_prefill_engram_gate(tensors, *, eps=1e-20):
    """Compose owning restore/gate goldens with FP32 q*k channel products."""
    from .engram_ecotech import golden_original_basis_gate, golden_original_basis_hidden

    values = dict(tensors, kv=tensors["projected"], rotation_block=tensors["rotation"])
    values["channel_weight"] = tensors["q_weight"].float() * tensors["k_weight"].float()
    golden_original_basis_hidden(values)
    golden_original_basis_gate(values, eps=eps)


def prefill_engram_hash_golden_cases():
    """Bound empty, dead-token, request, lookback/cache and page-edge history."""
    from functools import partial
    import torch
    from golden import ScalarSpec, TensorSpec
    from golden.validation import ratio_allclose

    capacity, vocab, layers, cache_rows, max_blocks = 16, 32, 2, 512, 3
    token_map = torch.arange(vocab, dtype=torch.int32) * 3 % 23
    multipliers = (2**40 + torch.tensor([[3, 5, 7, 11], [13, 17, 19, 23]])).to(torch.int64)
    primes = torch.tensor([97, 101] * 12, dtype=torch.int64).repeat(layers, 1)
    offsets = torch.arange(layers * COLS, dtype=torch.int64).reshape(layers, COLS) * 103
    reference = partial(golden_prefill_engram_hash, image_id=30, image_pad_id=31, pad_id=2)
    for active in (0, 1, 7, 16):
        requests = 2 if active == 7 else int(active > 0)
        tokens = torch.tensor([3, 30, 5, 6, 31, 8, 9] + list(range(10, 19)), dtype=torch.int64)
        positions = torch.arange(capacity, dtype=torch.int64)
        query = torch.full((capacity + 1,), active, dtype=torch.int32)
        query[0] = 0
        slots = torch.arange(capacity, dtype=torch.int64)
        if active == 7:
            query[1] = 3
            positions[:7] = torch.tensor([127, 128, 129, 4, 5, 6, 7])
            slots[:7] = torch.tensor([255, 256, 257, -1, cache_rows + 4, 133, 134])
        block_table = torch.tensor([1, 2, 0], dtype=torch.int32).repeat(capacity, 1)
        lookback = torch.full((capacity, 3), -1, dtype=torch.int64)
        lookback[0] = torch.tensor([7, 30, -1])
        lookback[1] = torch.tensor([-1, 9, 10])
        cache = torch.arange(cache_rows, dtype=torch.int32) % 23
        cache[131] = -1
        specs = [
            TensorSpec("token_ids", [capacity], torch.int64, init_value=tokens),
            TensorSpec("positions", [capacity], torch.int64, init_value=positions),
            TensorSpec("query_start", [capacity + 1], torch.int32, init_value=query),
            TensorSpec("slots", [capacity], torch.int64, init_value=slots),
            TensorSpec("block_table", [capacity, max_blocks], torch.int32, init_value=block_table),
            TensorSpec("lookback_ids", [capacity, 3], torch.int64, init_value=lookback),
            TensorSpec("token_map", [vocab], torch.int32, init_value=token_map),
            TensorSpec("multipliers", [layers, 4], torch.int64, init_value=multipliers),
            TensorSpec("primes", [layers, COLS], torch.int64, init_value=primes),
            TensorSpec("offsets", [layers, COLS], torch.int64, init_value=offsets),
            TensorSpec("cache", [cache_rows], torch.int32, init_value=cache),
            TensorSpec("hashes", [capacity, layers, COLS], torch.int32, init_value=-23),
            TensorSpec("token_mask", [capacity], torch.int32, init_value=-23),
            ScalarSpec("active", torch.int32, active, compile_runtime=True),
            ScalarSpec("requests", torch.int32, requests, compile_runtime=True),
        ]
        exact = ratio_allclose(rtol=0.0, atol=0.0, max_error_ratio=0.0)
        fn = make_prefill_engram_hash(capacity, vocab, layers, cache_rows, max_blocks, 30, 31, 2)
        yield dict(name=f"hash-history-{active}", fn=fn, specs=specs, golden_fn=reference,
                   compare_fn={name: exact for name in ("cache", "hashes", "token_mask")})


def prefill_engram_layout_golden_cases():
    """Bound TP1/2/4 column packets, empty owners and masked/inactive SP rows."""
    import torch
    from golden import ScalarSpec, TensorSpec
    from golden.validation import ratio_allclose

    capacity = 16
    for tp in (1, 2, 4):
        width = K // tp
        packet = (torch.arange(capacity * K).reshape(tp * capacity, width) % 251 - 125).to(torch.bfloat16)
        mask = (torch.arange(capacity) % 3 != 1).to(torch.int32)
        for active, rank in ((0, 0), (1, tp - 1), (7, tp - 1), (16, 0)):
            physical = max(1, (active + tp - 1) // tp)
            specs = [
                TensorSpec("packet", [tp * capacity, width], torch.bfloat16, init_value=packet),
                TensorSpec("token_mask", [capacity], torch.int32, init_value=mask),
                TensorSpec("full", [capacity, K], torch.bfloat16, init_value=-12.5),
                TensorSpec("local", [capacity, K], torch.bfloat16, init_value=-12.5),
                TensorSpec("local_mask", [capacity], torch.int32, init_value=-23),
                ScalarSpec("physical", torch.int32, physical, compile_runtime=True),
                ScalarSpec("active", torch.int32, active, compile_runtime=True),
                ScalarSpec("tp_rank", torch.int32, rank, compile_runtime=True),
            ]
            exact = ratio_allclose(rtol=0.0, atol=0.0, max_error_ratio=0.0)
            yield dict(name=f"layout-tp{tp}-{active}-rank{rank}", fn=make_prefill_engram_layout(capacity, tp),
                       specs=specs, golden_fn=golden_prefill_engram_layout,
                       compare_fn={name: exact for name in ("full", "local", "local_mask")})


def prefill_engram_gate_golden_cases():
    """Bound original/probability readbacks, signed/zero gates and inactive tails."""
    import torch
    from golden import ScalarSpec, TensorSpec
    from .engram_ecotech import golden_masked_gate_compare
    from .qkv_proj_rope import golden_prefix_compare

    capacity = 16
    generator = torch.Generator().manual_seed(732)
    hidden = torch.empty((capacity, HC, D), dtype=torch.bfloat16).uniform_(-0.5, 0.5, generator=generator)
    projected = torch.empty((capacity, N), dtype=torch.bfloat16).uniform_(-0.5, 0.5, generator=generator)
    q = torch.empty((HC, D), dtype=torch.bfloat16).uniform_(-1.0, 1.0, generator=generator)
    k = torch.empty((HC, D), dtype=torch.bfloat16).uniform_(-1.0, 1.0, generator=generator)
    rotation = torch.linalg.qr(torch.randn((32, 32), generator=generator)).Q.contiguous()
    projected[0, :D] = 0
    hidden[0, 1] = 0
    mask = (torch.arange(capacity) % 3 != 1).to(torch.int32)
    for active in (0, 1, 7, 16):
        specs = [
            TensorSpec("hidden", [capacity, HC, D], torch.bfloat16, init_value=hidden),
            TensorSpec("projected", [capacity, N], torch.bfloat16, init_value=projected),
            TensorSpec("q_weight", [HC, D], torch.bfloat16, init_value=q),
            TensorSpec("k_weight", [HC, D], torch.bfloat16, init_value=k),
            TensorSpec("rotation", [32, 32], torch.float32, init_value=rotation),
            TensorSpec("token_mask", [capacity], torch.int32, init_value=mask),
            TensorSpec("original", [capacity, HC, D], torch.float32, init_value=-12.5),
            TensorSpec("probabilities", [capacity, HC], torch.float32, init_value=-12.5),
            TensorSpec("output", [capacity, HC, D], torch.bfloat16, init_value=-12.5),
            ScalarSpec("active", torch.int32, active, compile_runtime=True),
        ]
        compare = {"original": golden_prefix_compare(active)}
        compare.update({name: golden_masked_gate_compare(active) for name in ("probabilities", "output")})
        yield dict(name=f"gate-original-probability-{active}", fn=make_prefill_engram_gate(capacity, 1e-20),
                   specs=specs, golden_fn=golden_prefill_engram_gate, compare_fn=compare)


def validate_prefill_engram(*, device=0, compile_only=False, platform="a2a3", sim=False):
    """Run the owning hash, TP/SP layout and original-basis Golden cases."""
    from itertools import chain
    from golden import run

    cases = chain(prefill_engram_hash_golden_cases(), prefill_engram_layout_golden_cases(),
                  prefill_engram_gate_golden_cases())
    for case in cases:
        name = case.pop("name")
        result = run(
            **case,
            config={"backend_type": platform, "execution_mode": sim, "device_id": device},
            compile_only=compile_only,
        )
        if not result.passed:
            raise RuntimeError(f"{name}: {result.error}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    validate_prefill_engram(device=args.device, compile_only=args.compile_only, platform=args.platform, sim=args.sim)

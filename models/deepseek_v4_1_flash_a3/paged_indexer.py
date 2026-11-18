# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded index-cache reads and global QLI selection over original A3 byte arenas."""

import pypto.language as pl

from .indexer import NEGATIVE_INF
from .quant import make_quantize


TILE_ROWS = 2048
TOPK = 512
BLOCK_SIZE = 8
CANDIDATE_BLOCKS = 2048


def make_paged_index_publish(capacity, geometry):
    """Reuse the observed quantizer and publish directly to native strided pages."""
    codes, scales = geometry["codes"], geometry["scales"]
    blocks, code_stride, scale_stride = codes["blocks"], codes["stride"], scales["stride"]
    code_offset, scale_offset = codes["offset"], scales["offset"]
    quantize = make_quantize(capacity, 128, rows_tile=16)

    @pl.jit
    def publish(
        value: pl.Tensor[[capacity, 128], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT32],
        cache: pl.InOut[pl.Tensor[[blocks, code_stride], pl.INT8]],
        scale_cache: pl.InOut[pl.Tensor[[blocks, scale_stride], pl.FP16]],
        payload: pl.InOut[pl.Tensor[[capacity, 128], pl.INT8]],
        row_scales: pl.InOut[pl.Tensor[[capacity, 8], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        quantize(value, payload, row_scales, active)
        with pl.spmd(1, name_hint="paged_index_publication"):
            _writer = pl.tile.get_block_idx()
            scale_values = pl.reshape(row_scales, [1, capacity * 8])
            narrowed_scales = pl.cast(scale_values[:, :], pl.FP16, mode="rint")
            for token in pl.range(active):
                block = pl.cast(pl.read(slots, [token, 0]), pl.INDEX)
                row = pl.cast(pl.read(slots, [token, 1]), pl.INDEX)
                if block >= 0 and row >= 0:
                    column = code_offset + row * 128
                    cache[block:block + 1, column:column + 128] = payload[token:token + 1, :]
                    scale = pl.read(narrowed_scales, [0, token * 8])
                    pl.write(scale_cache, [block, scale_offset + row], scale)
        return cache, scale_cache, payload, row_scales

    return publish


def make_paged_index_gather(geometry, requests, table_pages):
    """Read one bounded tile, optionally from eight-position source candidate blocks."""
    codes, scales = geometry["codes"], geometry["scales"]
    blocks, page_rows = codes["blocks"], codes["rows"]
    code_stride, scale_stride = codes["stride"], scales["stride"]
    code_offset, scale_offset = codes["offset"], scales["offset"]

    @pl.jit
    def gather(
        cache: pl.Tensor[[blocks, code_stride], pl.INT8],
        scales: pl.Tensor[[blocks, scale_stride], pl.FP16],
        table: pl.Tensor[[requests, table_pages], pl.INT32],
        candidates: pl.Tensor[[CANDIDATE_BLOCKS], pl.INT32],
        key: pl.InOut[pl.Tensor[[TILE_ROWS, 128], pl.INT8]],
        key_scales: pl.InOut[pl.Tensor[[TILE_ROWS], pl.FP16]],
        logical_ids: pl.InOut[pl.Tensor[[TILE_ROWS], pl.INT32]],
        request: pl.Scalar[pl.INT32],
        length: pl.Scalar[pl.INT32],
        base: pl.Scalar[pl.INT32],
        filtered: pl.Scalar[pl.INT32],
    ):
        # One writer also prevents adjacent FP16 scale/INT32 control races.
        with pl.spmd(1, name_hint="paged_index_tile"):
            _writer = pl.tile.get_block_idx()
            for row in pl.range(TILE_ROWS):
                logical = base + row
                if filtered != 0:
                    candidate = pl.read(candidates, [(base + row) // BLOCK_SIZE])
                    logical = candidate * BLOCK_SIZE + (base + row) % BLOCK_SIZE
                    if candidate < 0:
                        logical = -1
                scale = pl.cast(0.0, pl.FP16)
                zero_values = pl.full([1, 128], dtype=pl.FP16, value=0.0)
                values = pl.cast(zero_values, pl.INT8, mode="trunc")
                if logical >= 0 and logical < length:
                    block = pl.cast(pl.read(table, [request, logical // page_rows]), pl.INDEX)
                    offset = pl.cast(logical % page_rows, pl.INDEX)
                    column = code_offset + offset * 128
                    values = cache[block:block + 1, column:column + 128]
                    scale = pl.read(scales, [block, scale_offset + offset])
                else:
                    logical = -1
                key[row:row + 1, :] = values
                pl.write(key_scales, [row], scale)
                pl.write(logical_ids, [row], pl.cast(logical, pl.INT32))
        return key, key_scales, logical_ids

    return gather


def make_paged_topk_merge(capacity, ratio):
    """Merge tile TopK into a fixed 512-position global selection, with original causal IDs."""

    @pl.jit
    def merge(
        scores: pl.Tensor[[capacity, TILE_ROWS], pl.FP32],
        logical_ids: pl.Tensor[[TILE_ROWS], pl.INT32],
        positions: pl.Tensor[[capacity], pl.INT32],
        running: pl.InOut[pl.Tensor[[capacity, TOPK * 2], pl.FP32]],
        selected: pl.InOut[pl.Tensor[[capacity, TOPK], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(active, name_hint="paged_qli_topk_merge"):
            token = pl.tile.get_block_idx()
            visible = pl.cast((pl.read(positions, [token]) + 1) // ratio, pl.INT32)
            ids = pl.reshape(pl.load(logical_ids, [0], [TILE_ROWS]), [1, TILE_ROWS])
            one = pl.tile.full([1, TILE_ROWS], dtype=pl.INT32, value=1)
            positive = pl.minimum(pl.maximum(pl.add(ids, 1), 0), 1)
            causal = pl.minimum(pl.maximum(pl.mul(pl.sub(ids, visible), -1), 0), 1)
            valid = pl.mul(positive, causal)
            raw = pl.load(scores, [token, 0], [1, TILE_ROWS])
            row = pl.add(pl.mul(raw, pl.cast(valid, pl.FP32)),
                         pl.mul(pl.cast(pl.sub(one, valid), pl.FP32), NEGATIVE_INF))
            ids = pl.sub(pl.mul(pl.add(ids, 1), valid), 1)
            pairs = pl.tile.sort32(row, pl.reinterpret_view(ids, pl.UINT32))
            pairs = pl.tile.mrgsort(pairs, block_len=64)
            pairs = pl.tile.mrgsort(pairs, block_len=256)
            pairs = pl.tile.mrgsort(pairs, block_len=1024)
            # Merge two sorted TopK runs; pad to the existing 2048-element sort geometry.
            old = pl.load(running, [token, 0], [1, TOPK * 2])
            values = pl.tile.full([1, TILE_ROWS], dtype=pl.FP32, value=NEGATIVE_INF)
            indices = pl.tile.full([1, TILE_ROWS], dtype=pl.INT32, value=-1)
            old_values = pl.tile.gather_mask(old, mask_pattern=pl.tile.MaskPattern.P0101)
            old_ids = pl.tile.gather_mask(old, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            new = pl.tile.slice(pairs, [1, TOPK * 2], [0, 0])
            new_values = pl.tile.gather_mask(new, mask_pattern=pl.tile.MaskPattern.P0101)
            new_ids = pl.tile.gather_mask(new, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            values = pl.tile.assemble(values, old_values, [0, 0])
            values = pl.tile.assemble(values, new_values, [0, TOPK])
            indices = pl.tile.assemble(indices, old_ids, [0, 0])
            indices = pl.tile.assemble(indices, new_ids, [0, TOPK])
            merged = pl.tile.sort32(values, pl.reinterpret_view(indices, pl.UINT32))
            merged = pl.tile.mrgsort(merged, block_len=64)
            merged = pl.tile.mrgsort(merged, block_len=256)
            merged = pl.tile.mrgsort(merged, block_len=1024)
            leading = pl.tile.slice(merged, [1, TOPK * 2], [0, 0])
            pl.store(leading, [token, 0], running)
            picked = pl.tile.gather_mask(leading, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            pl.store(picked, [token, 0], selected)
        return running, selected

    return merge


def make_paged_candidate_merge(capacity):
    """Globally keep 2048 eight-position block maxima, pinning the newest visible block."""
    sort_rows = 8192
    tile_blocks = TILE_ROWS // BLOCK_SIZE

    @pl.jit
    def merge(
        scores: pl.Tensor[[capacity, TILE_ROWS], pl.FP32],
        positions: pl.Tensor[[capacity], pl.INT32],
        running: pl.InOut[pl.Tensor[[capacity, CANDIDATE_BLOCKS * 2], pl.FP32]],
        candidates: pl.InOut[pl.Tensor[[capacity, CANDIDATE_BLOCKS], pl.INT32]],
        base: pl.Scalar[pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(active, name_hint="paged_candidate_merge"):
            token = pl.tile.get_block_idx()
            visible = pl.cast(pl.read(positions, [token]) + 1, pl.INT32)
            ramp = pl.tile.arange(0, [1, TILE_ROWS], dtype=pl.INT32)
            absolute = pl.add(ramp, base)
            valid = pl.minimum(pl.maximum(pl.mul(pl.sub(absolute, visible), -1), 0), 1)
            raw = pl.load(scores, [token, 0], [1, TILE_ROWS])
            masked = pl.add(pl.mul(raw, pl.cast(valid, pl.FP32)),
                pl.mul(pl.cast(pl.add(pl.mul(valid, -1), 1), pl.FP32), NEGATIVE_INF))
            grouped = pl.reshape(masked, [tile_blocks, BLOCK_SIZE])
            reduce_tmp = pl.tile.create([tile_blocks, BLOCK_SIZE], dtype=pl.FP32)
            maxima = pl.reshape(pl.row_max(grouped, tmp_tile=reduce_tmp), [1, tile_blocks])
            ids = pl.add(pl.tile.arange(0, [1, tile_blocks], dtype=pl.INT32), pl.cast(base // BLOCK_SIZE, pl.INT32))
            last = pl.cast((visible - 1) // BLOCK_SIZE, pl.INT32)
            past_last = pl.cast(last + 1, pl.INT32)
            block_valid = pl.minimum(pl.maximum(pl.mul(pl.sub(ids, past_last), -1), 0), 1)
            distance = pl.sub(ids, last)
            pinned = pl.add(pl.mul(pl.minimum(pl.maximum(distance, pl.mul(distance, -1)), 1), -1), 1)
            maxima = pl.add(pl.mul(maxima, pl.cast(pl.add(pl.mul(pinned, -1), 1), pl.FP32)),
                            pl.mul(pl.cast(pinned, pl.FP32), 1.0e30))
            ids = pl.sub(pl.mul(pl.add(ids, 1), block_valid), 1)
            old = pl.load(running, [token, 0], [1, CANDIDATE_BLOCKS * 2])
            values = pl.tile.full([1, sort_rows], dtype=pl.FP32, value=NEGATIVE_INF)
            all_ids = pl.tile.full([1, sort_rows], dtype=pl.INT32, value=-1)
            values = pl.tile.assemble(values,
                pl.tile.gather_mask(old, mask_pattern=pl.tile.MaskPattern.P0101), [0, 0])
            all_ids = pl.tile.assemble(all_ids,
                pl.tile.gather_mask(old, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32), [0, 0])
            values = pl.tile.assemble(values, maxima, [0, CANDIDATE_BLOCKS])
            all_ids = pl.tile.assemble(all_ids, ids, [0, CANDIDATE_BLOCKS])
            pairs = pl.tile.sort32(values, pl.reinterpret_view(all_ids, pl.UINT32))
            pairs = pl.tile.mrgsort(pairs, block_len=64)
            pairs = pl.tile.mrgsort(pairs, block_len=256)
            pairs = pl.tile.mrgsort(pairs, block_len=1024)
            pairs = pl.tile.mrgsort(pairs, block_len=4096)
            leading = pl.tile.slice(pairs, [1, CANDIDATE_BLOCKS * 2], [0, 0])
            pl.store(leading, [token, 0], running)
            picked = pl.tile.gather_mask(leading, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            pl.store(picked, [token, 0], candidates)
        return running, candidates

    return merge


def make_paged_candidate_publish(capacity, shared_rows=8192):
    """Publish the full pruned pool; retain the observed native candidate-length buffer."""

    @pl.jit
    def publish(
        candidates: pl.Tensor[[capacity, CANDIDATE_BLOCKS], pl.INT32],
        shared: pl.InOut[pl.Tensor[[shared_rows, 1, CANDIDATE_BLOCKS], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(shared, [shared_rows, CANDIDATE_BLOCKS])
        with pl.spmd(active, name_hint="paged_candidates_publish"):
            token = pl.tile.get_block_idx()
            flat[token:token + 1, :] = candidates[token:token + 1, :]
        return shared

    return publish


def make_paged_index_entries(capacity, geometry, role, requests, table_pages):
    entries = dict(index_key_gather=make_paged_index_gather(geometry, requests, table_pages),
                   paged_topk_merge=make_paged_topk_merge(capacity, role["ratio"]))
    if role["mode"] == "full":
        entries["index_publish"] = make_paged_index_publish(capacity, geometry)
    if role["candidate_write"]:
        entries.update(paged_candidate_merge=make_paged_candidate_merge(capacity),
                       paged_candidate_publish=make_paged_candidate_publish(capacity))
    return entries


def run_paged_selection(query, query_scales, weights, positions, starts, lengths,
                        codes, scales, table, state, execute, rank, case, capacity, start, active):
    """Stream fixed-size tiles on device; transfer only bounded selection/candidate scratch."""
    import torch

    role = state["role"]
    selected = torch.full((capacity, TOPK), -1, dtype=torch.int32)
    candidates = torch.full((capacity, CANDIDATE_BLOCKS), -1, dtype=torch.int32)

    def pad(value):
        padded = torch.zeros((capacity, *value.shape[1:]), dtype=value.dtype)
        padded[:len(value)].copy_(value)
        return padded

    def pairs(count):
        value = torch.full((capacity, count * 2), NEGATIVE_INF, dtype=torch.float32)
        value.view(torch.int32)[:, 1::2].fill_(-1)
        return value

    for request, (begin, end) in enumerate(zip(starts.tolist(), starts.tolist()[1:])):
        left, right = max(begin, start), min(end, start + active)
        if left >= right:
            continue
        offset, count = left - start, right - left
        # Candidate-filtered requests have a different block list for every query row.
        segments = [(offset + row, 1) for row in range(count)] if role["candidate_filter"] else [(offset, count)]
        for first, rows in segments:
            running, block_running = pairs(TOPK), pairs(CANDIDATE_BLOCKS)
            picked = torch.full((capacity, TOPK), -1, dtype=torch.int32)
            blocks = torch.full((capacity, CANDIDATE_BLOCKS), -1, dtype=torch.int32)
            source = torch.full((CANDIDATE_BLOCKS,), -1, dtype=torch.int32)
            visible = min(lengths[request], max(0, (int(positions[first:first + rows].max()) + 1) // role["ratio"]))
            span = visible
            if role["candidate_filter"]:
                source.copy_(state["live"]["shared"]["candidates"][first].reshape(-1))
                span = int((source >= 0).sum()) * BLOCK_SIZE
            for base in range(0, span, TILE_ROWS):
                key = torch.zeros((TILE_ROWS, 128), dtype=torch.int8)
                key_scales = torch.zeros(TILE_ROWS, dtype=torch.float16)
                logical = torch.full((TILE_ROWS,), -1, dtype=torch.int32)
                label = case + f"-request-{request}-query-{first}-tile-{base}"
                execute("index_key_gather", rank, label,
                    [codes, scales, table, source, key, key_scales, logical, request,
                     lengths[request], base, int(role["candidate_filter"])],
                    dict(key=key, scales=key_scales, logical_ids=logical), "index_select")
                scores = torch.zeros((capacity, TILE_ROWS), dtype=torch.float32)
                unused = torch.full((capacity, TOPK), -1, dtype=torch.int32)
                # The existing QLI kernel retains all A3 FP16/FP32 rounding boundaries.
                score_positions = torch.full((capacity,), TILE_ROWS * role["ratio"] - 1, dtype=torch.int32)
                execute("index_select", rank, label,
                    [pad(query[first:first + rows]), pad(query_scales[first:first + rows]), key, key_scales,
                     pad(weights[first:first + rows]), score_positions, scores, unused, rows],
                    dict(scores=scores, selected=unused), "index_select")
                query_positions = pad(positions[first:first + rows]).to(torch.int32)
                execute("paged_topk_merge", rank, label,
                    [scores, logical, query_positions, running, picked, rows],
                    dict(running=running, selected=picked), "index_select")
                if role["candidate_write"]:
                    execute("paged_candidate_merge", rank, label,
                        [scores, query_positions, block_running, blocks, base, rows],
                        dict(running=block_running, candidates=blocks), "candidate_pool")
            selected[first:first + rows].copy_(picked[:rows])
            candidates[first:first + rows].copy_(blocks[:rows])
    return selected, candidates


def golden_paged_gather(args, geometry):
    """Gather from the original physical page strides, including candidate holes and tails."""
    args["key"].zero_()
    args["key_scales"].zero_()
    args["logical_ids"].fill_(-1)
    code, scale = geometry["codes"], geometry["scales"]
    for row in range(TILE_ROWS):
        logical = int(args["base"]) + row
        if int(args["filtered"]):
            block = int(args["candidates"][logical // BLOCK_SIZE])
            logical = block * BLOCK_SIZE + logical % BLOCK_SIZE if block >= 0 else -1
        if not 0 <= logical < int(args["length"]):
            continue
        block = int(args["table"][int(args["request"]), logical // code["rows"]])
        offset = logical % code["rows"]
        column = code["offset"] + offset * 128
        args["key"][row].copy_(args["cache"][block, column:column + 128])
        args["key_scales"][row] = args["scales"][block, scale["offset"] + offset]
        args["logical_ids"][row] = logical


def golden_paged_publish(args, geometry):
    import torch

    active = int(args["active"])
    value = args["value"][:active].float()
    maximum = value.abs().amax(dim=-1, keepdim=True)
    scale = maximum * (1.0 / 127.0)
    normalized = value * (127.0 / maximum.clamp_min(1e-30))
    codes = torch.round(normalized.clamp(-127, 127)).to(torch.int32).half().to(torch.int8)
    args["payload"][:active].copy_(codes)
    args["row_scales"][:active].copy_(scale.expand(-1, 8))
    for token in range(active):
        block, row = args["slots"][token].tolist()
        if block < 0 or row < 0:
            continue
        offset = geometry["codes"]["offset"] + row * 128
        args["cache"][block, offset:offset + 128].copy_(codes[token])
        args["scale_cache"][block, geometry["scales"]["offset"] + row] = scale[token, 0].half()


def golden_paged_candidate_publish(args):
    active = int(args["active"])
    args["shared"][:active, 0, :].copy_(args["candidates"][:active])


def _golden_merge_pairs(running, token, scores, ids, count):
    import torch

    old = running[token]
    values = torch.cat((old[::2], scores))
    logical = torch.cat((old.view(torch.int32)[1::2], ids))
    chosen = torch.argsort(values, descending=True, stable=True)[:count]
    values, logical = values[chosen], logical[chosen]
    logical[values <= NEGATIVE_INF / 2] = -1
    old[::2].copy_(values)
    old.view(torch.int32)[1::2].copy_(logical)
    return logical


def golden_paged_topk_merge(args, ratio):
    for token in range(int(args["active"])):
        ids = args["logical_ids"].clone()
        visible = (int(args["positions"][token]) + 1) // ratio
        valid = (ids >= 0) & (ids < visible)
        scores = args["scores"][token].clone().masked_fill_(~valid, NEGATIVE_INF)
        ids[~valid] = -1
        args["selected"][token].copy_(_golden_merge_pairs(args["running"], token, scores, ids, TOPK))


def golden_paged_candidate_merge(args):
    import torch

    base = int(args["base"])
    for token in range(int(args["active"])):
        visible = int(args["positions"][token]) + 1
        logical = torch.arange(TILE_ROWS, dtype=torch.int32) + base
        scores = args["scores"][token].clone().masked_fill_(logical >= visible, NEGATIVE_INF)
        maxima = scores.reshape(-1, BLOCK_SIZE).max(dim=1).values
        blocks = torch.arange(TILE_ROWS // BLOCK_SIZE, dtype=torch.int32) + base // BLOCK_SIZE
        maxima[blocks == (visible - 1) // BLOCK_SIZE] = 1.0e30
        blocks[blocks >= (visible + BLOCK_SIZE - 1) // BLOCK_SIZE] = -1
        args["candidates"][token].copy_(
            _golden_merge_pairs(args["running"], token, maxima, blocks, CANDIDATE_BLOCKS))


def validate_paged_selection(*, cpu_only=False):
    """Check the stream scheduler against dense QLI and global candidate oracles on CPU."""
    if not cpu_only:
        raise ValueError("Use leaf golden cases for device execution; this validates the CPU stream oracle")
    import torch
    from .indexer import golden_indexer_select

    torch.manual_seed(191)
    records = []
    capacity = 16
    for ratio, length in ((2, 2049), (1, 16385)):
        page_rows = 128 // ratio
        pages = (length + page_rows - 1) // page_rows
        blocks = pages + 2
        geometry = dict(codes=dict(blocks=blocks, rows=page_rows, stride=page_rows * 128 + 32, offset=32),
                        scales=dict(blocks=blocks, rows=page_rows, stride=page_rows + 16, offset=16))
        key = torch.randint(-7, 8, (pages * page_rows, 128), dtype=torch.int8)
        key_scale = torch.rand(pages * page_rows).half()
        codes = torch.zeros(blocks, geometry["codes"]["stride"], dtype=torch.int8)
        scales = torch.zeros(blocks, geometry["scales"]["stride"], dtype=torch.float16)
        table = torch.arange(pages, 0, -1).int().unsqueeze(0)
        for page in range(pages):
            physical = int(table[0, page])
            codes[physical, 32:].copy_(key[page * page_rows:(page + 1) * page_rows].flatten())
            scales[physical, 16:].copy_(key_scale[page * page_rows:(page + 1) * page_rows])
        query = torch.randint(-7, 8, (capacity, 32, 128), dtype=torch.int8)
        query_scales, weights = torch.rand(capacity, 32).half(), torch.rand(capacity, 32).half()
        positions = torch.zeros(capacity, dtype=torch.int64)
        positions[0] = length * ratio - 1
        operations = []

        def execute(name, _rank, _case, args, _outputs, _role):
            operations.append(name)
            if name == "index_key_gather":
                names = ("cache", "scales", "table", "candidates", "key", "key_scales", "logical_ids",
                         "request", "length", "base", "filtered")
                golden_paged_gather(dict(zip(names, args)), geometry)
            elif name == "index_select":
                names = ("query", "query_scale", "key", "key_scale", "weights", "positions",
                         "scores", "indices", "active")
                golden_indexer_select(dict(zip(names, args)), compress_ratio=ratio)
            elif name == "paged_topk_merge":
                names = ("scores", "logical_ids", "positions", "running", "selected", "active")
                golden_paged_topk_merge(dict(zip(names, args)), ratio)
            elif name == "paged_candidate_merge":
                names = ("scores", "positions", "running", "candidates", "base", "active")
                golden_paged_candidate_merge(dict(zip(names, args)))
            else:
                raise ValueError(name)

        role = dict(ratio=ratio, candidate_filter=False, candidate_write=ratio == 1)
        state = dict(role=role, live=dict(shared={}))
        selected, candidates = run_paged_selection(query, query_scales, weights, positions, torch.tensor([0, 1]),
            [length], codes, scales, table, state, execute, 0, "golden", capacity, 0, 1)
        dense_scores = torch.zeros(capacity, len(key))
        dense_ids = torch.full((capacity, TOPK), -1, dtype=torch.int32)
        golden_indexer_select(dict(query=query, query_scale=query_scales, key=key, key_scale=key_scale,
            weights=weights, positions=positions, scores=dense_scores, indices=dense_ids, active=1),
            compress_ratio=ratio)
        if set(selected[0].tolist()) != set(dense_ids[0].tolist()):
            raise RuntimeError("Streamed QLI selection differs from the dense causal oracle")
        if ratio == 1:
            masked = dense_scores[0].clone()
            masked[length:] = NEGATIVE_INF
            maxima = masked.reshape(-1, BLOCK_SIZE).max(dim=1).values
            maxima[(length - 1) // BLOCK_SIZE] = 1.0e30
            expected = torch.argsort(maxima, descending=True, stable=True)[:CANDIDATE_BLOCKS].int()
            if not torch.equal(candidates[0], expected):
                raise RuntimeError("Global source candidate pool differs from the dense block oracle")
            state["role"] = dict(role, candidate_write=False, candidate_filter=True)
            state["live"]["shared"]["candidates"] = candidates[:, None, :]
            filtered, _ = run_paged_selection(query, query_scales, weights, positions, torch.tensor([0, 1]),
                [length], codes, scales, table, state, execute, 0, "filtered", capacity, 0, 1)
            allowed = torch.zeros(len(key), dtype=torch.bool)
            for block in expected:
                allowed[int(block) * BLOCK_SIZE:(int(block) + 1) * BLOCK_SIZE] = True
            allowed[length:] = False
            restricted = dense_scores[0].clone().masked_fill_(~allowed, NEGATIVE_INF)
            expected_ids = torch.argsort(restricted, descending=True)[:TOPK].int()
            if set(filtered[0].tolist()) != set(expected_ids.tolist()):
                raise RuntimeError("Candidate consumer selected a position outside the restricted QLI oracle")
        records.append(dict(ratio=ratio, length=length, cpu_golden="passed", kernel="not_run",
                            tile_reads=operations.count("index_key_gather")))
    return records


def paged_selection_golden_cases():
    """Cover strided page reads, causal merges, global candidates and inactive tails."""
    from functools import partial
    import torch
    from golden import ScalarSpec, TensorSpec
    from golden.validation import ratio_allclose

    capacity = 16
    exact = ratio_allclose(rtol=0.0, atol=0.0, max_error_ratio=0.0)

    def tensor(name, value):
        return TensorSpec(name, list(value.shape), value.dtype, init_value=value)

    def scalar(name, value):
        return ScalarSpec(name, torch.int32, value, compile_runtime=True)

    def pairs(count):
        value = torch.full((capacity, count * 2), NEGATIVE_INF)
        value.view(torch.int32)[:, 1::2].fill_(-1)
        return value

    for ratio in (1, 2):
        rows, blocks = 128 // ratio, 6
        geometry = dict(codes=dict(blocks=blocks, rows=rows, stride=rows * 128 + 32, offset=32),
                        scales=dict(blocks=blocks, rows=rows, stride=rows + 16, offset=16))
        codes = (torch.arange(blocks * geometry["codes"]["stride"]) % 251 - 125).to(torch.int8)
        codes = codes.reshape(blocks, -1)
        scales = (torch.arange(blocks * geometry["scales"]["stride"]) % 127).half().reshape(blocks, -1)
        table = torch.tensor([[4, 1, 3, 2]], dtype=torch.int32)
        candidates = torch.full((CANDIDATE_BLOCKS,), -1, dtype=torch.int32)
        candidates[:3] = torch.tensor([3, 0, 17])
        value = ((torch.arange(capacity * 128).reshape(capacity, 128) % 251 - 125) / 32).bfloat16()
        value[0].zero_()
        slots = torch.tensor([[1 + row // rows, row % rows] for row in range(capacity)], dtype=torch.int32)
        slots[2] = torch.tensor([-1, -1])
        slots[3] = torch.tensor([4, rows - 1])
        for active in (0, 7):
            specs = [tensor("value", value), tensor("slots", slots), tensor("cache", codes),
                     tensor("scale_cache", scales),
                     tensor("payload", torch.full((capacity, 128), -23, dtype=torch.int8)),
                     tensor("row_scales", torch.full((capacity, 8), -23.0)), scalar("active", active)]
            yield dict(name=f"paged-publish-c{ratio}-active{active}",
                fn=make_paged_index_publish(capacity, geometry), specs=specs,
                golden_fn=partial(golden_paged_publish, geometry=geometry),
                compare_fn={name: exact for name in ("cache", "scale_cache", "payload", "row_scales")})
        for filtered in (0, 1):
            specs = [tensor("cache", codes), tensor("scales", scales), tensor("table", table),
                     tensor("candidates", candidates),
                     tensor("key", torch.full((TILE_ROWS, 128), -23, dtype=torch.int8)),
                     tensor("key_scales", torch.full((TILE_ROWS,), -23, dtype=torch.float16)),
                     tensor("logical_ids", torch.full((TILE_ROWS,), -23, dtype=torch.int32)),
                     scalar("request", 0), scalar("length", rows * 3 + 7), scalar("base", 0),
                     scalar("filtered", filtered)]
            yield dict(name=f"paged-gather-c{ratio}-filtered{filtered}",
                fn=make_paged_index_gather(geometry, 1, 4), specs=specs,
                golden_fn=partial(golden_paged_gather, geometry=geometry),
                compare_fn={name: exact for name in ("key", "key_scales", "logical_ids")})
        scores = torch.arange(capacity * TILE_ROWS).reshape(capacity, TILE_ROWS).float() / 8192
        logical = torch.arange(TILE_ROWS, dtype=torch.int32) + TILE_ROWS
        positions = torch.full((capacity,), 3001 * ratio - 1, dtype=torch.int32)
        for active in (0, 2):
            running = pairs(TOPK)
            running[:, ::2] = torch.arange(TOPK).float() / 2048
            running.view(torch.int32)[:, 1::2] = torch.arange(TOPK, dtype=torch.int32)
            specs = [tensor("scores", scores), tensor("logical_ids", logical), tensor("positions", positions),
                     tensor("running", running),
                     tensor("selected", torch.full((capacity, TOPK), -23, dtype=torch.int32)), scalar("active", active)]
            yield dict(name=f"paged-topk-c{ratio}-active{active}", fn=make_paged_topk_merge(capacity, ratio),
                specs=specs, golden_fn=partial(golden_paged_topk_merge, ratio=ratio),
                compare_fn={name: exact for name in ("running", "selected")})
    scores = torch.arange(capacity * TILE_ROWS).reshape(capacity, TILE_ROWS).float()
    positions = torch.full((capacity,), CANDIDATE_BLOCKS * BLOCK_SIZE + 136, dtype=torch.int32)
    for active in (0, 2):
        running = pairs(CANDIDATE_BLOCKS)
        running[:, ::2] = torch.arange(CANDIDATE_BLOCKS).float() / 2048
        running.view(torch.int32)[:, 1::2] = torch.arange(CANDIDATE_BLOCKS, dtype=torch.int32)
        specs = [tensor("scores", scores), tensor("positions", positions), tensor("running", running),
                 tensor("candidates", torch.full((capacity, CANDIDATE_BLOCKS), -23, dtype=torch.int32)),
                 scalar("base", CANDIDATE_BLOCKS * BLOCK_SIZE), scalar("active", active)]
        yield dict(name=f"paged-candidates-active{active}", fn=make_paged_candidate_merge(capacity),
            specs=specs, golden_fn=golden_paged_candidate_merge,
            compare_fn={name: exact for name in ("running", "candidates")})
        shared = torch.full((32, 1, CANDIDATE_BLOCKS), -23, dtype=torch.int32)
        candidates = torch.arange(capacity * CANDIDATE_BLOCKS, dtype=torch.int32).reshape(capacity, -1)
        yield dict(name=f"paged-candidates-publish-active{active}",
            fn=make_paged_candidate_publish(capacity, shared_rows=32),
            specs=[tensor("candidates", candidates), tensor("shared", shared), scalar("active", active)],
            golden_fn=golden_paged_candidate_publish, compare_fn={"shared": exact})


def main():
    import argparse
    import json
    from golden import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--cpu-oracle", action="store_true")
    parser.add_argument("--sim", action="store_true")
    args = parser.parse_args()
    if args.cpu_oracle:
        print(json.dumps(validate_paged_selection(cpu_only=True)))
        return
    for case in paged_selection_golden_cases():
        name = case.pop("name")
        result = run(**case, compile_only=args.compile_only,
                     config=dict(backend_type=args.platform, execution_mode=args.sim, device_id=args.device))
        if not result.passed:
            raise RuntimeError(f"Paged selection failed: {name}: {result.error}")


if __name__ == "__main__":
    main()

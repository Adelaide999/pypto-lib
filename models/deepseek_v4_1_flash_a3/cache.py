# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cache layouts, source ownership, geometry and legal state lifetimes."""

import json
import math
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import numpy as np
import pypto.language as pl
import torch
from .indexer import INDEXER_SELECT_KEY_CAPACITY
from .runtime_config import MAX_MODEL_CONTEXT_TOKENS, SPMD_WORKER_LIMIT
from .swa import GATHER_TILE, make_swa_leaves


# tiling
# A3 cache payloads are contiguous 128-row pages.  A 512-element transfer
# reduces launch work for the large BF16 and INT8 planes while remaining
# compatible with the smaller FP16 scale plane through the payload clamp.
COPY_TILE = 512


# Logical Decode pools are views over the Prefill allocation.  The map is
# deliberately explicit because compressed, SWA, index and state planes do
# not share the same page geometry on A3.
A3_DECODE_CACHE_SOURCES = {
    "window_cache_pool": "swa_cache",
    "compressed_cache_pool": "long_kv_cache",
    "index_cache_pool": "indexer.k_cache",
    "compressor_state_pool": "compressor.state_cache",
}


def decode_cache_views(prefill_handles, source_map=None):
    """Create Decode ``DeviceTensor`` aliases for Prefill cache allocations.

    ``prefill_handles`` is a rank-local mapping of logical cache names to
    ``DeviceTensor`` objects.  The returned tensors keep the source data
    pointer and owning buffer; no H2D copy or second allocation is performed.
    A caller with a phase-specific ABI may provide ``source_map`` whose values
    are either source names or ``(source_name, shape)`` pairs.
    """
    from pypto.runtime import DeviceTensor

    mapping = A3_DECODE_CACHE_SOURCES if source_map is None else source_map
    result = {}
    for decode_name, descriptor in mapping.items():
        if isinstance(descriptor, tuple):
            source_name, shape = descriptor
        else:
            source_name, shape = descriptor, None
        try:
            source = prefill_handles[source_name]
        except KeyError as error:
            raise KeyError(f"Prefill cache handle is missing for {source_name}: {decode_name}") from error
        if type(source) is not DeviceTensor:
            raise TypeError(f"Prefill cache {source_name} must be a DeviceTensor")
        if shape is None:
            shape = source.shape
        shape = tuple(shape)
        if (not shape or any(type(size) is not int or size <= 0 for size in shape)
                or source.nbytes < math.prod(shape) * source.dtype.itemsize):
            raise ValueError(f"Decode cache view {decode_name} exceeds {source_name} allocation")
        result[decode_name] = DeviceTensor(
            source.data_ptr,
            shape,
            source.dtype,
            buffer=source.buffer,
            byte_offset=source.byte_offset,
        )
    return result


def make_paged_cache_copy(
    blocks,
    payload,
    source_stride,
    destination_stride,
    source_offset=0,
    destination_offset=0,
    dtype=pl.BF16,
    copy_tile=None,
):
    """Copy selected page payloads in one dispatch while preserving unrelated bytes."""
    if blocks < 1 or payload < 1:
        raise ValueError("Require positive pages and payload")
    if copy_tile is None:
        copy_tile = next((tile for tile in (COPY_TILE, 256, 128, 64) if payload % tile == 0), None)
    if copy_tile not in (64, 128, 256, 512) or payload % copy_tile:
        raise ValueError("Require positive pages with aligned contiguous payloads")
    if not 0 <= source_offset <= source_stride - payload:
        raise ValueError("Source cache payload exceeds its physical page")
    if not 0 <= destination_offset <= destination_stride - payload:
        raise ValueError("Destination cache payload exceeds its physical page")

    @pl.jit
    def paged_cache_copy(
        source: pl.Tensor[[blocks, source_stride], dtype],
        destination: pl.InOut[pl.Tensor[[blocks, destination_stride], dtype]],
        pages: pl.Tensor[[blocks], pl.INT32],
        active_pages: pl.Scalar[pl.INT32],
    ):
        block_count = active_pages * (payload // copy_tile)
        for worker in pl.spmd(pl.min(block_count, SPMD_WORKER_LIMIT), name_hint="prefill_paged_cache_copy"):
            for block in pl.range(worker, block_count, SPMD_WORKER_LIMIT):
                page_index = block // (payload // copy_tile)
                page = pl.cast(pl.read(pages, [page_index]), pl.INDEX)
                offset = block % (payload // copy_tile) * copy_tile
                source_column = source_offset + offset
                destination_column = destination_offset + offset
                value = source[page:page + 1, source_column:source_column + copy_tile]
                destination[page:page + 1, destination_column:destination_column + copy_tile] = value
        return destination

    return paged_cache_copy


def cache_copy_pages(blocks, pages):
    """Build a bounded page list with unique in-range physical indices."""
    pages = tuple(pages)
    if len(pages) != len(set(pages)) or any(type(page) is not int or not 0 <= page < blocks for page in pages):
        raise ValueError("Cache copy pages must be unique physical indices inside the allocation")
    indices = torch.full((blocks,), -1, dtype=torch.int32)
    indices[:len(pages)] = torch.tensor(pages, dtype=torch.int32)
    return indices, len(pages)


def physical_page_geometry(layout, prefix, blocks):
    """Validate contiguous page payloads in the original shared raw cache ABI."""
    views = layout["views"]
    long = views[prefix + "long_kv_cache"]
    if views[prefix + "swa_cache"] != long:
        raise ValueError("Ratio1 long and SWA must retain the same typed physical arena")
    descriptors = dict(
        long=long, codes=views[prefix + "indexer.k_cache"][0], scales=views[prefix + "indexer.k_cache"][1]
    )
    geometry = {}
    for name, descriptor in descriptors.items():
        width = 512 if name == "long" else 128 if name == "codes" else 1
        dtype = "torch.bfloat16" if name == "long" else "torch.int8" if name == "codes" else "torch.float16"
        size = 1 if name == "codes" else 2
        shape, stride = descriptor["shape"], descriptor["stride"]
        key = descriptor["storage_key"]
        backing = layout["raw"][key]
        offset = descriptor["storage_offset_bytes"] - backing["storage_offset_bytes"]
        if shape != [blocks, 128, 1, width] or stride[1:] != [width, width, 1]:
            raise ValueError("Cache payload must retain its original 128-row typed page")
        if descriptor["dtype"] != dtype or offset < 0 or offset % size:
            raise ValueError("Cache payload dtype or byte offset differs")
        if backing["shape"] != [blocks * stride[0] * size] or backing["stride"] != [1]:
            raise ValueError("Raw cache backing does not cover the complete strided pages")
        payload = 128 * width
        if offset // size + payload > stride[0]:
            raise ValueError("Cache payload escapes its page")
        geometry[name] = dict(
            storage_key=key,
            dtype=dtype,
            stride=stride[0],
            payload=payload,
            offset=offset // size,
            width=width,
        )
    return geometry


def physical_consumer_geometry(layout, own_prefix, source_prefix, blocks):
    """Bind the shared ratio1 payload while preserving every source-owned raw page."""
    own = layout["views"][own_prefix + "swa_cache"]
    source = layout["views"][source_prefix + "long_kv_cache"]
    key = own["storage_key"]
    backing = layout["raw"][key]
    stride = own["stride"][0]
    if (
        source["storage_key"] != key
        or own["shape"] != [blocks, 128, 1, 512]
        or own["dtype"] != source["dtype"]
        or own["dtype"] != "torch.bfloat16"
        or own["stride"] != source["stride"]
        or own["stride"][1:] != [512, 512, 1]
        or own["storage_offset_bytes"] != source["storage_offset_bytes"]
        or source["shape"][1:] != [128, 1, 512]
    ):
        raise ValueError("Require the original shared source and own BF16 page payloads")
    offset = own["storage_offset_bytes"] - backing["storage_offset_bytes"]
    first_owned = source["shape"][0]
    if (
        offset < 0
        or offset % 2
        or offset // 2 + 65536 > stride
        or not 0 < first_owned < blocks
        or backing["shape"] != [blocks * stride * 2]
        or backing["stride"] != [1]
    ):
        raise ValueError("Missing readonly source prefix or complete raw page backing")
    return dict(storage_key=key, stride=stride, payload=65536, offset=offset // 2, first_owned=first_owned)


def physical_index_geometry(layout, prefix, blocks):
    """Validate readonly source index payloads in the shared consumer backing."""
    geometry = {}
    for name, descriptor in zip(("codes", "scales"), layout["views"][prefix + "indexer.k_cache"]):
        width, size = (128, 1) if name == "codes" else (1, 2)
        dtype = "torch.int8" if name == "codes" else "torch.float16"
        backing = layout["raw"][descriptor["storage_key"]]
        offset = descriptor["storage_offset_bytes"] - backing["storage_offset_bytes"]
        stride = descriptor["stride"][0]
        if (
            descriptor["dtype"] != dtype
            or (not 0 < descriptor["shape"][0] <= blocks or descriptor["shape"][1:] != [128, 1, width])
            or descriptor["stride"][1:] != [width, width, 1]
            or offset < 0
            or offset % size
            or backing["shape"] != [blocks * stride * size]
            or backing["stride"] != [1]
            or offset // size + 128 * width > stride
        ):
            raise ValueError("Readonly source index page geometry differs")
        geometry[name] = dict(
            storage_key=descriptor["storage_key"],
            dtype=dtype,
            stride=stride,
            payload=128 * width,
            offset=offset // size,
        )
    return geometry


@lru_cache(maxsize=1)
def _layer_plan():
    with Path(__file__).with_name("prefill_layer_plan.json").open(encoding="utf-8") as stream:
        plan = json.load(stream)
    roles = plan["layers"]
    if [role["physical_layer"] for role in roles] != list(range(40)):
        raise ValueError("Require the complete original physical-layer plan")
    return roles


def resolve_layer_role(layer_id, role=None):
    """Resolve original pure-data semantics without importing a reference model."""
    if type(layer_id) is not int or not 0 <= layer_id < 40:
        raise ValueError("Require an original physical layer in 0..39")
    expected = _layer_plan()[layer_id]
    if role is not None and any(role.get(name) != value for name, value in expected.items()):
        raise ValueError("Supplied role differs from the original physical-layer plan")
    return dict(expected)


def validate_live_state(role, original, live, epoch, token_map_id, *, candidate_check=None):
    """Reject stale selections and missing actual arenas before any layer mutation."""
    if role["engram_slot"] is not None:
        prepared = live.get("network_engram")
        if prepared is not None:
            if (
                prepared["shared_history_owner"] != 1
                or prepared["physical_layer"] != role["physical_layer"]
                or prepared["slot"] != role["engram_slot"]
                or prepared["epoch"] != epoch
                or prepared["token_map_id"] != token_map_id
            ):
                raise ValueError(
                    "The network layer requires its current slot from shared model-level history"
                )
        elif "hash_cache" not in live or live.get("hash_owner") != role["physical_layer"]:
            raise ValueError("Engram requires its own persistent request hash-cache state")
    if role["mode"] == "swa":
        if "cache" not in live:
            raise ValueError("SWA requires its request-owned live cache")
        return
    if not all(name in live for name in ("raw", "layout", "shared", "versions")):
        raise ValueError(
            "Compressed attention requires actual live backing, layout, selection and owner versions"
        )
    metadata = original["cache_metadata"]
    if metadata.get("physical_layer_id", role["physical_layer"]) != role["physical_layer"]:
        raise ValueError("Cache metadata differs from the requested physical layer")
    if role["mode"] != "full":
        if metadata["source_physical_layer_id"] != role["kv_source"]:
            raise ValueError("Readonly cache metadata differs from the original KV source")
        required = [("kv", role["kv_source"])]
        if role["mode"] != "reindex":
            required.append(("index", role["index_source"]))
        if role["candidate_filter"]:
            required.append(("candidate", role["candidate_source"]))
        for name, owner in required:
            if tuple(live["versions"].get(name, ())) != (owner, epoch, token_map_id):
                raise ValueError("Missing or stale live " + name + " ownership for this token map")
    if role["candidate_filter"]:
        if candidate_check is None:
            validate_live_refresh_candidates(
                original, live["shared"], role["kv_source"], packed_requests=live.get("packed_requests", False)
            )
        else:
            candidate_check(original, live["shared"], role["kv_source"])


def publish_live_versions(role, live, epoch, token_map_id):
    """Publish owner versions only after the layer has completed its real computation."""
    for name, writes, owner in (
        ("kv", role["kv_write"], role["kv_source"]),
        ("index", role["index_selection_write"], role["index_source"]),
        ("candidate", role["candidate_write"], role["candidate_source"]),
    ):
        if writes:
            live["versions"][name] = (owner, epoch, token_map_id)


def invalidate_live_versions(role, live):
    """Clear pending publications so a failed retry cannot retain a valid owner tag."""
    for name, writes in (
        ("kv", role["kv_write"]),
        ("index", role["index_selection_write"]),
        ("candidate", role["candidate_write"]),
    ):
        if writes:
            live["versions"].pop(name, None)


DTYPES = {
    "torch.bfloat16": torch.bfloat16,
    "torch.int8": torch.int8,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
}
PTO_DTYPES = {"torch.bfloat16": pl.BF16, "torch.int8": pl.INT8, "torch.float16": pl.FP16}


def network_cache_geometry(layout, role, blocks):
    """Bind complete raw strides and original typed plane offsets without alias remapping."""
    role = resolve_layer_role(role["physical_layer"], role)
    if role["mode"] not in ("full", "reuse", "reindex") or role["ratio"] not in (1, 2):
        raise ValueError("The network cache adapter requires an original compressed role")
    root = f"model.layers.{role['kv_source']}.self_attn."
    own_root = f"model.layers.{role['physical_layer']}.self_attn."
    views = layout["views"]
    descriptors = dict(
        long=views[root + "long_kv_cache"],
        own=views[own_root + "swa_cache"],
        codes=views[root + "indexer.k_cache"][0],
        scales=views[root + "indexer.k_cache"][1],
    )
    result = {}
    for name, descriptor in descriptors.items():
        rows = 128 if name == "own" else 128 // role["ratio"]
        width = 512 if name in ("long", "own") else 128 if name == "codes" else 1
        dtype = (
            "torch.bfloat16"
            if name in ("long", "own")
            else "torch.int8"
            if name == "codes"
            else "torch.float16"
        )
        size = 1 if name == "codes" else 2
        shape, stride = descriptor["shape"], descriptor["stride"]
        backing = layout["raw"][descriptor["storage_key"]]
        offset = descriptor["storage_offset_bytes"] - backing["storage_offset_bytes"]
        if (
            descriptor["dtype"] != dtype
            or shape != [blocks, rows, 1, width]
            or stride[1:] != [width, width, 1]
            or offset < 0
            or offset % size
            or backing["shape"] != [blocks * stride[0] * size]
            or backing["stride"] != [1]
            or offset // size + rows * width > stride[0]
        ):
            raise ValueError("Original full-network " + name + " plane geometry is unsupported")
        result[name] = dict(
            storage_key=descriptor["storage_key"],
            dtype=dtype,
            blocks=blocks,
            stride=stride[0],
            offset=offset // size,
            rows=rows,
            width=width,
            payload=rows * width,
        )
    if role["ratio"] == 2 and role["mode"] == "full":
        ring = views[root + "compressor.state_cache"]
        backing = layout["raw"][ring["storage_key"]]
        if (
            ring["dtype"] != "torch.float32"
            or ring["shape"] != [blocks, 32, 1, 1024]
            or ring["stride"] != [32768, 1024, 1024, 1]
            or ring["storage_offset_bytes"] != 0
            or backing["shape"] != [blocks * 131072]
        ):
            raise ValueError("C2 requires its actual contiguous32x1024 FP32 ring plane")
        result["ring"] = dict(
            storage_key=ring["storage_key"],
            dtype="torch.float32",
            blocks=blocks,
            stride=32768,
            offset=0,
            rows=32,
            width=1024,
            payload=32768,
        )
    return result


def raw_cache_matrix(raw, descriptor):
    """Expose one complete contiguous raw arena in the plane's scalar dtype."""
    value = raw[descriptor["storage_key"]]
    return value.view(DTYPES[descriptor["dtype"]]).reshape(descriptor["blocks"], descriptor["stride"])


def owned_cache_pages(original, cache_name, plane):
    """Validate actual owner pages, slots and the reserved untouched page."""
    source = original["cache_metadata"]
    groups = [group for group in source["actual_group_requests"] if cache_name in group["layer_names"]]
    if len(groups) != 1:
        raise ValueError("Require one original scheduler group per cache plane")
    group = groups[0]
    pages = sorted(set(group["live_block_ids"]))
    if group["reserved_untouched_block_id"] in pages or any(
        type(page) is not int or not 0 < page < plane["blocks"] for page in pages
    ):
        raise ValueError("Cache owner pages escape the real allocation or include the untouched reservation")
    metadata = source["actual_metadata"][cache_name]["global_metadata"]
    table_pages = {int(page) for row in metadata["block_table"].tolist() for page in row if int(page) > 0}
    if not table_pages.issubset(set(pages)):
        raise ValueError("Native block table refers to a foreign cache-owner page")
    if cache_name.endswith("compressor.state_cache"):
        return pages
    slots = metadata["slot_mapping"]
    if slots.dtype not in (torch.int32, torch.int64) or list(slots.shape) != [len(original["positions"]), 2]:
        raise ValueError("Network publication requires actual native page/offset slot pairs")
    for page, offset in slots.tolist():
        if page < 0 or offset < 0:
            if (page, offset) != (-1, -1):
                raise ValueError("A padded native slot must retain both negative fields")
        elif page not in pages or not 0 <= offset < plane["rows"]:
            raise ValueError("Publication would write a foreign page or leave its typed payload")
    return pages


def make_network_cache_publish(capacity, plane):
    """Publish current BF16 rows into their real page stride and payload offset."""
    blocks, stride, offset = plane["blocks"], plane["stride"], plane["offset"]
    width = plane["width"]
    T_TILE = 4

    @pl.jit
    def publish(
        value: pl.Tensor[[capacity, width], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        raw: pl.InOut[pl.Tensor[[blocks, stride], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for group in pl.spmd((active + T_TILE - 1) // T_TILE, name_hint="network_cache_plane_publish"):
            first = group * T_TILE
            rows = pl.min(T_TILE, active - first)
            payload = pl.slice(value, [T_TILE, width], [first, 0], valid_shape=[rows, width])
            payload = pl.set_validshape(payload, T_TILE, width)
            for lane in pl.unroll(T_TILE):
                token = first + lane
                if token < active:
                    page = pl.read(slots, [token, 0])
                    row = pl.read(slots, [token, 1])
                    if page >= 0:
                        page_index = pl.cast(page, pl.INDEX)
                        column = pl.cast(offset + row * width, pl.INDEX)
                        raw[page_index : page_index + 1, column : column + width] = payload[lane : lane + 1, :]
        return raw

    return publish


def make_network_sparse_read(capacity, config, geometry):
    """Read the original cache planes and attend over selected KV rows."""
    own, long = geometry["own"], geometry["long"]
    heads, dim = config["num_attention_heads"], config["head_dim"]
    own_stride, long_stride = own["stride"], long["stride"]
    own_offset, long_offset = own["offset"], long["offset"]
    # Decode compiles a 16-row network shape and normally has one active row.
    # The pipelined reader below is tuned for large Prefill batches.  Keeping
    # its fixed 24-worker staging ring for Decode creates buffers for 4,608
    # head rows even though at most 16 rows can be live.  Use the dynamic
    # reader for this small shape so its scratch follows the active rows.
    if (capacity <= 16 or (heads, dim) != (64, 512)
            or any(v % dim for v in (own_stride, long_stride, own_offset, long_offset))):
        return _make_general_network_sparse_read(capacity, config, geometry)

    blocks = own["blocks"]
    own_pitch, long_pitch = own_stride // dim, long_stride // dim
    own_offset, long_offset = own_offset // dim, long_offset // dim
    window = 640
    width = heads * dim
    scale = dim**-0.5
    workers = 24
    K_TILE = 128
    GATHER_T_TILE = K_TILE // 2
    D_TILE = dim // 2
    lane_heads = heads // 2
    pv_delay = 2
    slots = pv_delay + 1
    staging_rows = workers * slots * heads
    kv_ready, score_ready, prob_ready, pv_ready = 0, 1, 2, 3

    @pl.jit
    def sparse_read(
        query: pl.Tensor[[capacity, width], pl.BF16],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        long_raw: pl.Tensor[[blocks, long_stride], pl.BF16],
        own_raw: pl.Tensor[[blocks, own_stride], pl.BF16],
        output: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        own_pool = pl.reshape(own_raw, [blocks * own_pitch, dim])
        long_pool = pl.reshape(long_raw, [blocks * long_pitch, dim])
        kv_transfer = pl.create_tensor([workers * slots * K_TILE, dim], dtype=pl.BF16)
        qflat = pl.reshape(query, [capacity * heads, dim])
        oflat = pl.reshape(output, [capacity * heads, dim])
        sink_col = pl.reshape(sink, [heads, 1])
        scores = pl.create_tensor([staging_rows, K_TILE], dtype=pl.FP32)
        weights = pl.create_tensor([staging_rows, K_TILE], dtype=pl.BF16)
        projections = pl.create_tensor([staging_rows, dim], dtype=pl.FP16)
        ffts = pl.create_tensor([256], dtype=pl.INT64)
        scheduled_parts = pl.create_tensor([capacity, 1], dtype=pl.INT32)
        groups = (active + 7) // 8
        for count_worker in pl.spmd(pl.min(groups, 48), name_hint="count_sparse_kv_parts"):
            key_columns = pl.tile.arange(0, [1, window], dtype=pl.INT32)
            part_ids = pl.cast(pl.add(pl.tile.shrs(key_columns, 7), 1), pl.FP32)
            for group in pl.range(count_worker, groups, 48):
                token_start = group * 8
                rows = pl.min(active - token_start, 8)
                sparse_ids = pl.tile.load(indices, [token_start, 0], [8, window], valid_shape=[rows, window])
                float_ids = pl.cast(sparse_ids, pl.FP32)
                valid_ids = pl.minimum(pl.maximum(pl.add(float_ids, 1.0), 0.0), 1.0)
                marked = pl.col_expand_mul(valid_ids, part_ids)
                mx_tmp = pl.create_tile([8, window], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                last_part = pl.cast(pl.row_max(marked, mx_tmp), pl.INT32)
                pl.store(pl.set_validshape(last_part, rows, 1), [token_start, 0], scheduled_parts)
        with pl.spmd(pl.min(workers, active), name_hint="prefill_sparse_qk_pv_pipe"):
            worker = pl.tile.get_block_idx()
            pl.system.set_ffts(ffts)
            for token in pl.range(worker, active, workers):
                live_parts = pl.read(scheduled_parts, [token, 0])
                kv = pl.create_tile([slots * K_TILE, dim], dtype=pl.BF16, target_memory=pl.MemorySpace.Mat)
                q = pl.load(qflat, [token * heads, 0], [heads, dim], target_memory=pl.MemorySpace.Mat)
                for step in pl.range(live_parts + pv_delay):
                    if step < live_parts:
                        src_row = (worker * slots + step % slots) * K_TILE
                        l1_row = step % slots * K_TILE
                        stage_row = (worker * slots + step % slots) * heads
                        pl.system.sync_wait(kv_ready, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                        kv = pl.gather_row(kv, kv_transfer, [l1_row, 0], [src_row, 0], [K_TILE, dim])
                        kt = pl.tile.slice(pl.tile.transpose_view(kv), [dim, K_TILE], [0, l1_row])
                        score = pl.matmul(q, kt, out_dtype=pl.FP32)
                        pl.store(score, [stage_row, 0], scores)
                        pl.system.sync_set(score_ready, pipe=pl.PipeType.FIX,
                                           ffts_mode=2, core_type=pl.KernelType.AIC)
                    if step >= pv_delay:
                        part = step - pv_delay
                        l1_row = part % slots * K_TILE
                        stage_row = (worker * slots + part % slots) * heads
                        pl.system.sync_wait(prob_ready, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                        p = pl.load(weights, [stage_row, 0], [heads, K_TILE], target_memory=pl.MemorySpace.Mat)
                        v = pl.tile.slice(kv, [K_TILE, dim], [l1_row, 0])
                        pv = pl.matmul(p, v, out_dtype=pl.FP32)
                        pl.store(pv, [stage_row, 0], projections)
                        pl.system.sync_set(pv_ready, pipe=pl.PipeType.FIX,
                                           ffts_mode=2, core_type=pl.KernelType.AIC)

                for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    pl.system.set_ffts(ffts)
                    head = lane * lane_heads
                    initial_m = pl.load(sink_col, [head, 0], [lane_heads, 1], target_memory=pl.MemorySpace.Vec)
                    initial_l = pl.tile.adds(pl.tile.muls(initial_m, 0.0), 1.0)
                    initial_left = pl.tile.full([lane_heads, D_TILE], dtype=pl.FP32, value=0.0)
                    initial_right = pl.tile.full([lane_heads, D_TILE], dtype=pl.FP32, value=0.0)
                    # The softmax stage leads the PV merge by two steps.
                    for step, (m, l, left, right, alpha_old, alpha_new) in pl.range(
                        live_parts + pv_delay + 1,
                        init_values=(initial_m, initial_l, initial_left, initial_right, initial_l, initial_l),
                    ):
                        # After the final score, reuse GM slot zero for the next token.
                        gather_current = step < live_parts and (step != 0 or token == worker)
                        next_token = token + workers
                        gather_next = False
                        if step == live_parts + 1 and next_token < active:
                            next_parts = pl.read(scheduled_parts, [next_token, 0])
                            gather_next = next_parts > 0
                        gather_part = step
                        gather_token = token
                        if gather_next:
                            gather_part = 0
                            gather_token = next_token
                        do_gather = gather_current or gather_next
                        if do_gather:
                            gather_offset = lane * GATHER_T_TILE
                            kv_row = (worker * slots + gather_part % slots) * K_TILE + gather_offset
                            ids = pl.load(indices, [gather_token, gather_part * K_TILE + gather_offset], [1, GATHER_T_TILE])
                            valid = pl.minimum(pl.maximum(pl.add(ids, 1), 0), 1)
                            # Masked lanes use a selected row from the same plane.
                            last_id = pl.tile.read(ids, [0, GATHER_T_TILE - 1])
                            first_id = pl.tile.read(ids, [0, 0])
                            picked_id = pl.max(last_id, first_id)
                            if picked_id < 0:
                                for column in pl.range(1, GATHER_T_TILE - 1):
                                    picked_id = pl.max(picked_id, pl.tile.read(ids, [0, column]))
                            if picked_id < 0:
                                pad = pl.tile.full([GATHER_T_TILE, dim], dtype=pl.BF16, value=0.0)
                                pl.store(pad, [kv_row, 0], kv_transfer)
                            else:
                                picked = pl.add(pl.maximum(ids, 0), pl.mul(pl.sub(valid, 1), -picked_id))
                                picked_page = pl.tile.shrs(picked, 7)
                                if gather_part == 0:
                                    address = pl.add(pl.add(picked, own_offset), pl.mul(picked_page, own_pitch - 128))
                                    own_kv = pl.tile.mgather(own_pool, address)
                                    pl.store(own_kv, [kv_row, 0], kv_transfer)
                                else:
                                    address = pl.add(pl.add(picked, long_offset), pl.mul(picked_page, long_pitch - 128))
                                    if last_id < 0:
                                        # Gather the prefix through the last valid long-cache slot.
                                        ends = pl.tile.arange(1, [1, GATHER_T_TILE], dtype=pl.INT32)
                                        marked_ends = pl.mul(ends, valid)
                                        reduce_ones = pl.tile.full([8, GATHER_T_TILE], dtype=pl.INT32, value=1)
                                        end_matrix = pl.col_expand_mul(reduce_ones, marked_ends)
                                        reduce_tmp = pl.tile.full([8, GATHER_T_TILE], dtype=pl.INT32, value=0)
                                        end_tile = pl.row_max(end_matrix, reduce_tmp)
                                        end = pl.cast(pl.tile.read(end_tile, [0, 0]), pl.INDEX)
                                        bounded = pl.tile.set_validshape(address, 1, end)
                                        partial_kv = pl.tile.mgather(long_pool, bounded)
                                        zeroed_kv = pl.tile.fillpad_inplace(partial_kv)
                                        full_kv = pl.tile.set_validshape(zeroed_kv, GATHER_T_TILE, dim)
                                        pl.store(full_kv, [kv_row, 0], kv_transfer)
                                    else:
                                        long_kv = pl.tile.mgather(long_pool, address)
                                        pl.store(long_kv, [kv_row, 0], kv_transfer)
                            pl.system.sync_set(kv_ready, pipe=pl.PipeType.MTE3,
                                               ffts_mode=2, core_type=pl.KernelType.AIV)
                        if step > 0 and step <= live_parts:
                            part = step - 1
                            stage_row = (worker * slots + part % slots) * heads + head
                            pl.system.sync_wait(score_ready, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                            s = pl.load(scores, [stage_row, 0], [lane_heads, K_TILE],
                                        target_memory=pl.MemorySpace.Vec)
                            ids = pl.load(indices, [token, part * K_TILE], [1, K_TILE],
                                          target_memory=pl.MemorySpace.Vec)
                            fids = pl.cast(ids, pl.FP32)
                            valid = pl.minimum(pl.maximum(pl.add(fids, 1.0), 0.0), 1.0)
                            bias = pl.mul(pl.sub(valid, 1.0), 1e30)
                            logits = pl.col_expand_add(pl.mul(s, scale), bias)
                            tmp = pl.create_tile([lane_heads, K_TILE], dtype=pl.FP32,
                                                 target_memory=pl.MemorySpace.Vec)
                            next_m = pl.maximum(pl.row_max(logits, tmp), m)
                            alpha = pl.exp(pl.sub(m, next_m))
                            exp = pl.exp(pl.row_expand_sub(logits, next_m))
                            prob = pl.col_expand_mul(exp, valid)
                            local_l = pl.row_sum(prob, tmp)
                            pl.store(pl.cast(prob, pl.BF16, mode="rint"), [stage_row, 0], weights)
                            next_l = pl.add(pl.mul(alpha, l), local_l)
                            pl.system.sync_set(prob_ready, pipe=pl.PipeType.MTE3,
                                               ffts_mode=2, core_type=pl.KernelType.AIV)
                            softmax_m, softmax_l, softmax_alpha = pl.yield_(next_m, next_l, alpha)
                        else:
                            softmax_m, softmax_l, softmax_alpha = pl.yield_(m, l, initial_l)
                        if step >= pv_delay + 1:
                            part = step - pv_delay - 1
                            stage_row = (worker * slots + part % slots) * heads + head
                            pl.system.sync_wait(pv_ready, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                            pv_left = pl.load(projections, [stage_row, 0], [lane_heads, D_TILE])
                            pv_left = pl.cast(pv_left, pl.FP32)
                            next_left = pl.add(pl.row_expand_mul(left, alpha_old), pv_left)
                            pv_right = pl.load(projections, [stage_row, D_TILE], [lane_heads, D_TILE])
                            pv_right = pl.cast(pv_right, pl.FP32)
                            next_right = pl.add(pl.row_expand_mul(right, alpha_old), pv_right)
                            merge_left, merge_right = pl.yield_(next_left, next_right)
                        else:
                            merge_left, merge_right = pl.yield_(left, right)
                        final_m, final_l, final_left, final_right, final_alpha_old, final_alpha_new = pl.yield_(
                            softmax_m, softmax_l, merge_left, merge_right, alpha_new, softmax_alpha,
                        )
                    row = token * heads + head
                    pl.store(pl.cast(pl.row_expand_div(final_left, final_l), pl.BF16, mode="rint"),
                             [row, 0], oflat)
                    pl.store(pl.cast(pl.row_expand_div(final_right, final_l), pl.BF16, mode="rint"),
                             [row, D_TILE], oflat)
        return output

    return sparse_read


def _make_general_network_sparse_read(capacity, config, geometry):
    """Gather from cache layouts that cannot be viewed as contiguous KV rows."""
    long, own = geometry["long"], geometry["own"]
    blocks, long_stride, own_stride = long["blocks"], long["stride"], own["stride"]
    long_offset, own_offset = long["offset"], own["offset"]
    heads, head_dim = config["num_attention_heads"], config["head_dim"]
    width = heads * head_dim
    sparse_config = dict(config, sliding_window=640)
    _publish, _gather, attend = make_swa_leaves(capacity, blocks, sparse_config)

    @pl.jit
    def sparse_read(
        query: pl.Tensor[[capacity, width], pl.BF16],
        indices: pl.Tensor[[capacity, 640], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        long_raw: pl.Tensor[[blocks, long_stride], pl.BF16],
        own_raw: pl.Tensor[[blocks, own_stride], pl.BF16],
        output: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        selected_rows = pl.max(active, 1) * 640
        selected = pl.create_tensor([selected_rows, head_dim], dtype=pl.BF16)
        block_count = active * (640 // GATHER_TILE)
        for block_worker in pl.spmd(pl.min(block_count, SPMD_WORKER_LIMIT), name_hint='network_distinct_cache_gather'):
            for block in pl.range(block_worker, block_count, SPMD_WORKER_LIMIT):
                token = block // (640 // GATHER_TILE)
                begin = block % (640 // GATHER_TILE) * GATHER_TILE
                element = token * 640 + begin
                ids = pl.tile.load(indices, [token, begin], [1, GATHER_TILE])
                head = pl.tile.read(ids, [0, 0])
                tail = pl.tile.read(ids, [0, GATHER_TILE - 1])
                page_row = pl.cast(head % 128, pl.INDEX)
                is_run = False
                if head >= 0 and tail - head == GATHER_TILE - 1 and page_row <= 128 - GATHER_TILE:
                    is_run = True
                    for group_lane in pl.unroll(1, GATHER_TILE - 1):
                        is_run = is_run and pl.tile.read(ids, [0, group_lane]) == head + group_lane
                if is_run:
                    run_page = pl.cast(head // 128, pl.INDEX)
                    run_offset = page_row * head_dim
                    run_span = GATHER_TILE * head_dim
                    if begin < 128:
                        run = pl.tile.load(own_raw, [run_page, own_offset + run_offset], [1, run_span])
                    else:
                        run = pl.tile.load(long_raw, [run_page, long_offset + run_offset], [1, run_span])
                    run_rows = pl.tile.reshape(run, [GATHER_TILE, head_dim])
                    pl.tile.store(run_rows, [element, 0], selected)
                else:
                    values = pl.tile.full([GATHER_TILE, head_dim], dtype=pl.BF16, value=0.0)
                    for group_lane in pl.range(GATHER_TILE):
                        index = pl.tile.read(ids, [0, group_lane])
                        if index >= 0:
                            page = pl.cast(index // 128, pl.INDEX)
                            column_offset = pl.cast(index % 128, pl.INDEX) * head_dim
                            if begin < 128:
                                values = pl.tile.gather_row(
                                    values, own_raw, [group_lane, 0], [page, own_offset + column_offset], [1, head_dim],
                                )
                            else:
                                values = pl.tile.gather_row(
                                    values, long_raw, [group_lane, 0], [page, long_offset + column_offset], [1, head_dim],
                                )
                    pl.tile.store(values, [element, 0], selected)
        attend(query, selected, indices, sink, output, active)
        return output

    return sparse_read


def make_network_cache_entries(capacity, config, geometry, role, requests=1, table_pages=1):
    """Build only actual cache-addressing and owner-page transfer entries."""
    role = resolve_layer_role(role["physical_layer"], role)
    entries = dict(
        network_swa_publish=make_network_cache_publish(capacity, geometry["own"]),
        network_compressed_attention=make_network_sparse_read(capacity, config, geometry),
    )
    if role["mode"] == "full":
        entries["network_compressed_publish"] = make_network_cache_publish(capacity, geometry["long"])
    if role["mode"] == "reuse":
        return entries
    if config.get("packed_requests", False):
        from .paged_indexer import make_paged_index_entries

        entries.update(make_paged_index_entries(capacity, geometry, role, requests, table_pages))
        return entries
    for name in ("codes", "scales"):
        plane = geometry[name]
        payload, stride, offset = plane["payload"], plane["stride"], plane["offset"]
        tile = min(COPY_TILE, payload)
        dtype = PTO_DTYPES[plane["dtype"]]
        entries["network_load_" + name] = make_paged_cache_copy(
            plane["blocks"], payload, stride, payload, source_offset=offset, dtype=dtype, copy_tile=tile
        )
        if role["mode"] == "full":
            entries["network_store_" + name] = make_paged_cache_copy(
                plane["blocks"], payload, payload, stride, destination_offset=offset, dtype=dtype, copy_tile=tile
            )
    return entries


def load_network_index(raw, geometry, pages, execute, rank, case):
    """Stage actual index-owner pages in their native64/128-row typed shapes."""
    result = {}
    page_ids, active_pages = cache_copy_pages(geometry["codes"]["blocks"], pages)
    for name in ("codes", "scales"):
        plane = geometry[name]
        dense = torch.empty((plane["blocks"], plane["payload"]), dtype=DTYPES[plane["dtype"]])
        physical = raw_cache_matrix(raw, plane)
        if active_pages:
            execute("network_load_" + name, rank, case, [physical, dense, page_ids, active_pages],
                    {"payload": dense}, "physical_cache_load")
        result[name] = dense.reshape(plane["blocks"], plane["rows"], 1, plane["width"])
    return result


def store_network_index(raw, geometry, pages, dense, execute, rank, case):
    """Store only real index-owner pages and preserve aliased ring/SWA pages."""
    page_ids, active_pages = cache_copy_pages(geometry["codes"]["blocks"], pages)
    for name in ("codes", "scales"):
        plane = geometry[name]
        physical = raw_cache_matrix(raw, plane)
        source = dense[name].reshape(plane["blocks"], plane["payload"])
        if active_pages:
            execute("network_store_" + name, rank, case, [source, physical, page_ids, active_pages],
                    {"raw": physical}, "physical_cache_store")


def validate_network_sparse_indices(indices, original, geometry, active):
    """Reject physical IDs outside the real request's long or own-SWA page maps."""
    starts = original["metadata"]["query_start_loc"].tolist()
    start = original["cp_token_range"][0]
    layer = original["physical_layer_id"]
    role = resolve_layer_role(layer, original.get("role"))
    root = f"model.layers.{role['kv_source']}.self_attn."
    own_root = f"model.layers.{layer}.self_attn."
    metadata = original["cache_metadata"]["actual_metadata"]
    long_table = metadata[root + "long_kv_cache"]["global_metadata"]["block_table"].numpy()
    own_table = metadata[own_root + "swa_cache"]["global_metadata"]["block_table"].numpy()
    values = indices[:active].numpy()
    if (values < -1).any():
        raise ValueError("Sparse padding ID must remain exactly -1")
    if len(values) and (start < starts[0] or start + len(values) > starts[-1]):
        raise ValueError("Sparse query rows fall outside the original request ranges")
    planes = ((values[:, :128], own_table, geometry["own"]["rows"]),
              (values[:, 128:], long_table, geometry["long"]["rows"]))
    for request, (begin, end) in enumerate(zip(starts, starts[1:])):
        first, last = max(0, begin - start), min(len(values), end - start)
        if first >= last:
            continue
        for lanes, table, rows in planes:
            ids = lanes[first:last]
            pages, offsets = np.divmod(ids[ids >= 0], 128)
            if (offsets >= rows).any() or not np.isin(pages, table[request]).all():
                raise ValueError("Sparse read would leave its original request-owned typed cache plane")


def attach_network_cache_geometry(provider):
    """Bind original compressed-role addressing without changing computation or owner versions."""
    if provider.scheduler_capacity != 8192:
        raise ValueError("The existing shared publication entries require the original8192-row ABI")
    for rank in range(provider.topology["ep"]):
        for case, layout in provider.layouts[rank].items():
            for layer in range(40):
                live = provider.state_handles_for_layer(layer)[rank][case]
                live["shared"] = provider.shared[rank][case]
                live["packed_requests"] = provider.packed_requests
            for layer in range(2, 40):
                role = resolve_layer_role(layer)
                blocks = layout["views"][f"model.layers.{layer}.self_attn.swa_cache"]["shape"][0]
                geometry = network_cache_geometry(layout, role, blocks)
                provider.state_handles_for_layer(layer)[rank][case]["network_geometry"] = deepcopy(geometry)
                provider.geometry[rank][case][layer] = []
    return dict(
        attached_layers=list(range(2, 40)),
        entry_ABI="distinct_actual_raw_cache_planes",
        roles="original_full_reuse_reindex",
        compile="not_run",
        NPU="not_run",
        numerical="not_run",
    )


def validate_live_refresh_candidates(original, shared, source_id, *, packed_requests=False):
    """Validate full short-history coverage or an actual bounded, pinned global source pool."""
    starts = original["metadata"]["query_start_loc"].tolist()
    metadata = original["cache_metadata"]["actual_metadata"]
    prefix = f"model.layers.{source_id}.self_attn."
    lengths = metadata[prefix + "indexer.k_cache"]["global_metadata"]["cache_seq_lens"].tolist()
    limit = MAX_MODEL_CONTEXT_TOKENS if packed_requests else INDEXER_SELECT_KEY_CAPACITY
    if any(length < 0 or length > limit for length in lengths):
        raise ValueError("Refresh history exceeds the supported index-key capacity")
    start = original["cp_token_range"][0]
    candidates = shared["candidates"]
    for local in range(original["actual_local_query_rows"]):
        token = start + local
        request = next(i for i in range(len(starts) - 1) if starts[i] <= token < starts[i + 1])
        visible = min(lengths[request], int(original["positions"][token]) + 1)
        values = candidates[local].reshape(-1).tolist()
        blocks = (visible + 7) // 8
        if not packed_requests:
            if not set(range(blocks)).issubset(values):
                raise ValueError("This input requires candidate pruning; no dense substitute is supported")
            continue
        available = [value for value in values if value >= 0]
        if (len(available) != min(blocks, 2048) or len(available) != len(set(available))
                or values != available + [-1] * (len(values) - len(available))
                or any(value >= blocks for value in available)
                or (blocks and available[0] != blocks - 1)):
            raise ValueError("Candidate pool is stale, unpinned, duplicated or outside its request history")
        if blocks <= 2048 and set(available) != set(range(blocks)):
            raise ValueError("Short-history candidate publication lost full block coverage")


def golden_paged_cache_copy(tensors, *, payload, source_offset=0, destination_offset=0):
    """Copy selected payloads and retain all unselected pages and padding."""
    pages = tensors["pages"][:int(tensors["active_pages"])].long()
    tensors["destination"][pages, destination_offset:destination_offset + payload] = (
        tensors["source"][pages, source_offset:source_offset + payload]
    )


def paged_cache_copy_cases():
    """Cover sparse owner lists, empty work and both raw/dense transfer directions."""
    from functools import partial
    from golden import ScalarSpec, TensorSpec

    blocks = 8
    for dtype, payload in ((torch.int8, 128 * 128), (torch.float16, 64),
                           (torch.float16, 128), (torch.bfloat16, 512)):
        for store in (False, True):
            source_stride = payload if store else payload + 256
            destination_stride = payload + 256 if store else payload
            source_offset, destination_offset = (0, 128) if store else (128, 0)
            pages, active = cache_copy_pages(blocks, [6, 1, 4] if store else [1, 6])
            dtype_name = str(dtype).split(".")[-1]
            specs = [
                TensorSpec("source", [blocks, source_stride], dtype,
                           init_value=lambda s=source_stride: torch.arange(blocks * s).reshape(blocks, s) % 127),
                TensorSpec("destination", [blocks, destination_stride], dtype, init_value=-17),
                TensorSpec("pages", [blocks], torch.int32, init_value=pages),
                ScalarSpec("active_pages", torch.int32, active, compile_runtime=True),
            ]
            golden = partial(golden_paged_cache_copy, payload=payload,
                             source_offset=source_offset, destination_offset=destination_offset)
            yield dict(name=f"paged-{dtype_name}-{payload}-{'store' if store else 'load'}",
                       fn=make_paged_cache_copy(blocks, payload, source_stride, destination_stride,
                                               source_offset, destination_offset, dtype=PTO_DTYPES[str(dtype)],
                                               copy_tile=min(COPY_TILE, payload)),
                       specs=specs, golden_fn=golden)
    pages, active = cache_copy_pages(blocks, [])
    yield dict(name="paged-empty", fn=make_paged_cache_copy(blocks, 128, 256, 256), specs=[
        TensorSpec("source", [blocks, 256], torch.bfloat16, init_value=3),
        TensorSpec("destination", [blocks, 256], torch.bfloat16, init_value=-17),
        TensorSpec("pages", [blocks], torch.int32, init_value=pages),
        ScalarSpec("active_pages", torch.int32, active, compile_runtime=True),
    ], golden_fn=partial(golden_paged_cache_copy, payload=128))


def golden_network_sparse_read(tensors, *, geometry):
    """Evaluate two paged KV planes and a logit-only sink."""
    active = int(tensors["active"])
    indices = tensors["indices"][:active].long()
    sink = tensors["sink"]
    query = tensors["query"][:active].float().reshape(active, sink.numel(), 512)
    planes = []
    for name, part in (("own", indices[:, :128]), ("long", indices[:, 128:])):
        layout = geometry[name]
        offset, rows = layout["offset"], layout["rows"]
        values = tensors[f"{name}_raw"][:, offset:offset + rows * 512].reshape(layout["blocks"], rows, 512)
        address = part.clamp_min(0)
        tile = values[address // 128, address % 128].float()
        planes.append(tile.masked_fill(part[:, :, None] < 0, 0.0))
    kv = torch.cat(planes, dim=1)
    logits = (query @ kv.transpose(1, 2)) * (512**-0.5)
    logits = logits.masked_fill(indices[:, None, :] < 0, -torch.inf)
    logits = torch.cat((logits, sink[None, :, None].expand(active, -1, -1)), dim=-1)
    result = logits.softmax(dim=-1)[..., :-1] @ kv
    tensors["output"][:active] = result.flatten(1).to(torch.bfloat16)


def network_sparse_read_cases():
    """Cover padded/cache-aligned pages, sparse holes and slot reuse."""
    from functools import partial
    from golden import ScalarSpec, TensorSpec
    from golden.validation import ratio_allclose

    capacity, active, blocks, heads, dim = 32, 29, 4, 64, 512
    config = dict(num_attention_heads=heads, head_dim=dim, sliding_window=128, index_topk=512)
    numeric = ratio_allclose(atol=0.004, rtol=0.02, max_error_ratio=0.0, valid_rows=active)

    def compare(actual, expected, **kwargs):
        if not torch.equal(actual[active:], expected[active:]):
            return False, "Inactive sparse attention output changed"
        if torch.count_nonzero(actual[0]) or torch.count_nonzero(actual[24]):
            return False, "Fully masked sparse attention output is nonzero"
        return numeric(actual, expected, **kwargs)

    for ratio, offset in ((1, 0), (2, dim), (2, 128)):
        rng = torch.Generator().manual_seed(2411)
        ids = torch.full((capacity, 640), -1, dtype=torch.int32)
        specs = []
        geometry = {}
        for name, column, width, rows in (("long", 128, 512, 128 // ratio), ("own", 0, 128, 128)):
            stride = 130 * dim + offset
            geometry[name] = dict(blocks=blocks, stride=stride, offset=offset, rows=rows)
            raw = torch.full((blocks, stride), 29, dtype=torch.bfloat16)
            data = torch.randn(blocks, rows * dim, generator=rng, dtype=torch.bfloat16) * 0.5
            raw[:, offset:offset + rows * dim] = data
            page = torch.randint(blocks, (active, width), generator=rng)
            row = torch.randint(rows, (active, width), generator=rng)
            chosen = (page * 128 + row).int()
            chosen[:, ::7] = -1
            chosen[1, :4] = torch.tensor([rows - 1, 128, 128, -1], dtype=torch.int32)
            chosen[2, 29:] = -1
            chosen[3, :64] = -1
            chosen[3, 8:10] = 1
            chosen[0] = chosen[24] = -1
            ids[:active, column:column + width] = chosen
            specs.append(TensorSpec(f"{name}_raw", list(raw.shape), raw.dtype, init_value=raw))
        query = torch.randn(capacity, heads * dim, generator=rng, dtype=torch.bfloat16) * 0.25
        sink = torch.randn(heads, generator=rng)
        inputs = [
            TensorSpec("query", list(query.shape), query.dtype, init_value=query),
            TensorSpec("indices", list(ids.shape), ids.dtype, init_value=ids),
            TensorSpec("sink", [heads], sink.dtype, init_value=sink),
            *specs,
            TensorSpec("output", list(query.shape), query.dtype, init_value=-299),
            ScalarSpec("active", torch.int32, active, compile_runtime=True),
        ]
        yield dict(
            name=f"sparse-read-ratio{ratio}-offset{offset}",
            fn=make_network_sparse_read(capacity, config, geometry),
            specs=inputs,
            golden_fn=partial(golden_network_sparse_read, geometry=geometry),
            compare_fn={"output": compare},
        )


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--sparse-read", action="store_true", help="Validate the two-plane sparse attention reader")
    args = parser.parse_args()
    cases = network_sparse_read_cases() if args.sparse_read else paged_cache_copy_cases()
    for case in cases:
        name = case.pop("name")
        result = run(
            **case,
            config={"backend_type": args.platform, "execution_mode": args.sim, "device_id": args.device},
            compile_only=args.compile_only,
            rtol=0.0,
            atol=0.0,
        )
        if not result.passed:
            raise RuntimeError(f"{name}: {result.error}")
        print(f"[cache] {name} PASS", flush=True)

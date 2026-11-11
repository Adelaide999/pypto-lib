# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Address distinct network cache planes and reuse the existing sparse attention arithmetic."""

from copy import deepcopy

import torch
import pypto.language as pl

from .prefill_cache_layout import make_paged_cache_copy
from .prefill_layer_state import resolve_layer_role
from .swa import make_swa_leaves


DTYPES = {"torch.bfloat16": torch.bfloat16, "torch.int8": torch.int8,
          "torch.float16": torch.float16, "torch.float32": torch.float32}
PTO_DTYPES = {"torch.bfloat16": pl.BF16, "torch.int8": pl.INT8, "torch.float16": pl.FP16}


def network_cache_geometry(layout, role, blocks):
    """Bind complete raw strides and original typed plane offsets without alias remapping."""
    role = resolve_layer_role(role["physical_layer"], role)
    if role["mode"] not in ("full", "reuse", "reindex") or role["ratio"] not in (1, 2):
        raise ValueError("The network cache adapter requires an original compressed role")
    root = f"model.layers.{role['kv_source']}.self_attn."
    own_root = f"model.layers.{role['physical_layer']}.self_attn."
    views = layout["views"]
    descriptors = dict(long=views[root + "long_kv_cache"], own=views[own_root + "swa_cache"],
                       codes=views[root + "indexer.k_cache"][0], scales=views[root + "indexer.k_cache"][1])
    result = {}
    for name, descriptor in descriptors.items():
        rows = 128 if name == "own" else 128 // role["ratio"]
        width = 512 if name in ("long", "own") else 128 if name == "codes" else 1
        dtype = "torch.bfloat16" if name in ("long", "own") else "torch.int8" if name == "codes" else "torch.float16"
        size = 1 if name == "codes" else 2
        shape, stride = descriptor["shape"], descriptor["stride"]
        backing = layout["raw"][descriptor["storage_key"]]
        offset = descriptor["storage_offset_bytes"] - backing["storage_offset_bytes"]
        if (descriptor["dtype"] != dtype or shape != [blocks, rows, 1, width]
                or stride[1:] != [width, width, 1] or offset < 0 or offset % size
                or backing["shape"] != [blocks * stride[0] * size] or backing["stride"] != [1]
                or offset // size + rows * width > stride[0]):
            raise ValueError("Original full-network " + name + " plane geometry is unsupported")
        result[name] = dict(storage_key=descriptor["storage_key"], dtype=dtype, blocks=blocks,
            stride=stride[0], offset=offset // size, rows=rows, width=width, payload=rows * width)
    if role["ratio"] == 2 and role["mode"] == "full":
        ring = views[root + "compressor.state_cache"]
        backing = layout["raw"][ring["storage_key"]]
        if (ring["dtype"] != "torch.float32" or ring["shape"] != [blocks, 32, 1, 1024]
                or ring["stride"] != [32768, 1024, 1024, 1] or ring["storage_offset_bytes"] != 0
                or backing["shape"] != [blocks * 131072]):
            raise ValueError("C2 requires its actual contiguous32x1024 FP32 ring plane")
        result["ring"] = dict(storage_key=ring["storage_key"], dtype="torch.float32", blocks=blocks,
                              stride=32768, offset=0, rows=32, width=1024, payload=32768)
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
    if (group["reserved_untouched_block_id"] in pages
            or any(type(page) is not int or not 0 < page < plane["blocks"] for page in pages)):
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

    @pl.jit
    def publish(
        value: pl.Tensor[[capacity, width], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        raw: pl.InOut[pl.Tensor[[blocks, stride], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for token in pl.spmd(active, name_hint="network_cache_plane_publish"):
            page = pl.read(slots, [token, 0])
            row = pl.read(slots, [token, 1])
            if page >= 0:
                page_index = pl.cast(page, pl.INDEX)
                column = pl.cast(offset + row * width, pl.INDEX)
                raw[page_index:page_index + 1, column:column + width] = value[token:token + 1, :]
        return raw

    return publish


def make_network_sparse_read(capacity, config, geometry):
    """Gather distinct cache planes and call the unchanged640-row attention leaf."""
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
        selected = pl.create_tensor([capacity * 640, head_dim], dtype=pl.BF16)
        for element in pl.spmd(active * 640, name_hint="network_distinct_cache_gather"):
            token = element // 640
            lane = element % 640
            index = pl.read(indices, [token, lane])
            if index >= 0:
                page = pl.cast(index // 128, pl.INDEX)
                row = pl.cast(index % 128, pl.INDEX)
                if lane < 128:
                    column = own_offset + row * head_dim
                    selected[element:element + 1, :] = own_raw[page:page + 1, column:column + head_dim]
                else:
                    column = long_offset + row * head_dim
                    selected[element:element + 1, :] = long_raw[page:page + 1, column:column + head_dim]
            else:
                selected[element:element + 1, :] = pl.full([1, head_dim], dtype=pl.BF16, value=0.0)
        attend(query, selected, indices, sink, output, active)
        return output

    return sparse_read


def make_network_cache_entries(capacity, config, geometry, role):
    """Build only actual cache-addressing and owner-page transfer entries."""
    role = resolve_layer_role(role["physical_layer"], role)
    entries = dict(network_swa_publish=make_network_cache_publish(capacity, geometry["own"]),
                   network_compressed_attention=make_network_sparse_read(capacity, config, geometry))
    if role["mode"] == "full":
        entries["network_compressed_publish"] = make_network_cache_publish(capacity, geometry["long"])
    if role["mode"] == "reuse":
        return entries
    for name in ("codes", "scales"):
        plane = geometry[name]
        payload, stride, offset = plane["payload"], plane["stride"], plane["offset"]
        tile = min(128, payload)
        dtype = PTO_DTYPES[plane["dtype"]]
        entries["network_load_" + name] = make_paged_cache_copy(
            1, payload, stride, payload, source_offset=offset, dtype=dtype, copy_tile=tile)
        if role["mode"] == "full":
            entries["network_store_" + name] = make_paged_cache_copy(
                1, payload, payload, stride, destination_offset=offset, dtype=dtype, copy_tile=tile)
    return entries


def load_network_index(raw, geometry, pages, execute, rank, case):
    """Stage actual index-owner pages in their native64/128-row typed shapes."""
    result = {}
    for name in ("codes", "scales"):
        plane = geometry[name]
        dense = torch.empty((plane["blocks"], plane["payload"]), dtype=DTYPES[plane["dtype"]])
        physical = raw_cache_matrix(raw, plane)
        for page in pages:
            value = dense[page:page + 1]
            execute("network_load_" + name, rank, case + f"-page-{page}",
                    [physical[page:page + 1], value], {"payload": value}, "physical_cache_load")
        result[name] = dense.reshape(plane["blocks"], plane["rows"], 1, plane["width"])
    return result


def store_network_index(raw, geometry, pages, dense, execute, rank, case):
    """Store only real index-owner pages and preserve aliased ring/SWA pages."""
    for name in ("codes", "scales"):
        plane = geometry[name]
        physical = raw_cache_matrix(raw, plane)
        source = dense[name].reshape(plane["blocks"], plane["payload"])
        for page in pages:
            value = physical[page:page + 1]
            execute("network_store_" + name, rank, case + f"-page-{page}",
                    [source[page:page + 1], value], {"raw": value}, "physical_cache_store")


def validate_network_sparse_indices(indices, original, geometry, active):
    """Reject physical IDs outside the real request's long or own-SWA page maps."""
    starts = original["metadata"]["query_start_loc"].tolist()
    start = original["cp_token_range"][0]
    layer = original["physical_layer_id"]
    role = resolve_layer_role(layer, original.get("role"))
    root = f"model.layers.{role['kv_source']}.self_attn."
    own_root = f"model.layers.{layer}.self_attn."
    metadata = original["cache_metadata"]["actual_metadata"]
    long_table = metadata[root + "long_kv_cache"]["global_metadata"]["block_table"].tolist()
    own_table = metadata[own_root + "swa_cache"]["global_metadata"]["block_table"].tolist()
    for local, values in enumerate(indices[:active].tolist()):
        token = start + local
        request = next(index for index in range(len(starts) - 1) if starts[index] <= token < starts[index + 1])
        for lane, value in enumerate(values):
            if value < 0:
                if value != -1:
                    raise ValueError("Sparse padding ID must remain exactly -1")
                continue
            page, offset = divmod(value, 128)
            kind = "own" if lane < 128 else "long"
            table = own_table if lane < 128 else long_table
            if page not in table[request] or not 0 <= offset < geometry[kind]["rows"]:
                raise ValueError("Sparse read would leave its original request-owned typed cache plane")


def attach_network_cache_geometry(provider):
    """Bind original compressed-role addressing without changing computation or owner versions."""
    if provider.scheduler_capacity != 8192:
        raise ValueError("The existing shared publication entries require the original8192-row ABI")
    for rank in range(provider.topology["ep"]):
        for case, layout in provider.layouts[rank].items():
            for layer in range(40):
                provider.state_handles_for_layer(layer)[rank][case]["shared"] = provider.shared[rank][case]
            for layer in range(2, 40):
                role = resolve_layer_role(layer)
                blocks = layout["views"][f"model.layers.{layer}.self_attn.swa_cache"]["shape"][0]
                geometry = network_cache_geometry(layout, role, blocks)
                provider.state_handles_for_layer(layer)[rank][case]["network_geometry"] = deepcopy(geometry)
                provider.geometry[rank][case][layer] = []
    return dict(attached_layers=list(range(2, 40)), entry_ABI="distinct_actual_raw_cache_planes",
                roles="original_full_reuse_reindex", compile="not_run", NPU="not_run", numerical="not_run")

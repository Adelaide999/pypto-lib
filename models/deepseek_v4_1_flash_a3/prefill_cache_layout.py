# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Copy typed cache page payloads between dense and shared physical backings."""

import pypto.language as pl


# tiling
COPY_TILE = 128


def make_paged_cache_copy(blocks, payload, source_stride, destination_stride,
                         source_offset=0, destination_offset=0, dtype=pl.BF16, first_page=0, copy_tile=COPY_TILE):
    """Copy one contiguous payload per page while preserving unrelated bytes."""
    if blocks < 1 or payload < 1 or copy_tile not in (64, 128) or payload % copy_tile:
        raise ValueError("Require positive pages with aligned contiguous payloads")
    if not 0 <= source_offset <= source_stride - payload:
        raise ValueError("Source cache payload exceeds its physical page")
    if not 0 <= destination_offset <= destination_stride - payload:
        raise ValueError("Destination cache payload exceeds its physical page")

    if not 0 <= first_page < blocks:
        raise ValueError("Require a nonempty owned page range")

    @pl.jit
    def paged_cache_copy(
        source: pl.Tensor[[blocks, source_stride], dtype],
        destination: pl.InOut[pl.Tensor[[blocks, destination_stride], dtype]],
    ):
        for block in pl.spmd((blocks - first_page) * (payload // copy_tile), name_hint="prefill_paged_cache_copy"):
            page = first_page + block // (payload // copy_tile)
            offset = block % (payload // copy_tile) * copy_tile
            source_column = source_offset + offset
            destination_column = destination_offset + offset
            value = source[page:page + 1, source_column:source_column + copy_tile]
            destination[page:page + 1, destination_column:destination_column + copy_tile] = value
        return destination

    return paged_cache_copy


def physical_page_geometry(layout, prefix, blocks):
    """Validate contiguous page payloads in the original shared raw cache ABI."""
    views = layout["views"]
    long = views[prefix + "long_kv_cache"]
    if views[prefix + "swa_cache"] != long:
        raise ValueError("Ratio1 long and SWA must retain the same typed physical arena")
    descriptors = dict(long=long, codes=views[prefix + "indexer.k_cache"][0],
                       scales=views[prefix + "indexer.k_cache"][1])
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
        geometry[name] = dict(storage_key=key, dtype=dtype, stride=stride[0], payload=payload,
                              offset=offset // size, width=width)
    return geometry


def physical_consumer_geometry(layout, own_prefix, source_prefix, blocks):
    """Bind the shared ratio1 payload while preserving every source-owned raw page."""
    own = layout["views"][own_prefix + "swa_cache"]
    source = layout["views"][source_prefix + "long_kv_cache"]
    key = own["storage_key"]
    backing = layout["raw"][key]
    stride = own["stride"][0]
    if (source["storage_key"] != key or own["shape"] != [blocks, 128, 1, 512]
            or own["dtype"] != source["dtype"] or own["dtype"] != "torch.bfloat16"
            or own["stride"] != source["stride"] or own["stride"][1:] != [512, 512, 1]
            or own["storage_offset_bytes"] != source["storage_offset_bytes"]
            or source["shape"][1:] != [128, 1, 512]):
        raise ValueError("Require the original shared source and own BF16 page payloads")
    offset = own["storage_offset_bytes"] - backing["storage_offset_bytes"]
    first_owned = source["shape"][0]
    if (offset < 0 or offset % 2 or offset // 2 + 65536 > stride or not 0 < first_owned < blocks
            or backing["shape"] != [blocks * stride * 2] or backing["stride"] != [1]):
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
        if (descriptor["dtype"] != dtype or (not 0 < descriptor["shape"][0] <= blocks or descriptor["shape"][1:] != [128, 1, width])
                or descriptor["stride"][1:] != [width, width, 1] or offset < 0 or offset % size
                or backing["shape"] != [blocks * stride * size] or backing["stride"] != [1]
                or offset // size + 128 * width > stride):
            raise ValueError("Readonly source index page geometry differs")
        geometry[name] = dict(storage_key=descriptor["storage_key"], dtype=dtype,
                              stride=stride, payload=128 * width, offset=offset // size)
    return geometry

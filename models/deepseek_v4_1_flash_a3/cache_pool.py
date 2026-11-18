# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Budget unique physical cache arenas and lease pages to request generations."""

from copy import deepcopy


def cache_page_bytes(layout):
    """Count each physical storage key once across all of its typed aliases."""
    sizes = {}
    element_bytes = {"torch.int8": 1, "torch.uint8": 1, "torch.float16": 2,
                     "torch.bfloat16": 2, "torch.float32": 4, "torch.int32": 4, "torch.int64": 8}
    for value in layout["views"].values():
        for plane in value if isinstance(value, list) else [value]:
            stride = plane["stride"][0] * element_bytes[plane["dtype"]]
            key = plane["storage_key"]
            if key in sizes and sizes[key] != stride:
                raise ValueError("Typed aliases disagree on their physical page byte stride")
            if type(stride) is not int or not 0 < stride < 2**63:
                raise ValueError("Cache page addressing exceeds signed INT64")
            sizes[key] = stride
    if set(sizes) != set(layout["raw"]):
        raise ValueError("Typed pages must cover every physical allocation")
    return sizes


def cache_capacity_from_hbm(layouts, *, usable_bytes, weight_bytes, scratch_bytes,
                            communication_bytes, runtime_reserve_bytes, history_page_bytes=512):
    """Take the tightest rank after resident weights and peak temporary allocations."""
    world = len(layouts)
    vectors = (usable_bytes, weight_bytes, scratch_bytes, communication_bytes, runtime_reserve_bytes)
    if type(history_page_bytes) is not int or history_page_bytes < 0:
        raise ValueError("History page byte count must be nonnegative")
    if any(len(values) != world or any(type(n) is not int or n < 0 for n in values) for values in vectors):
        raise ValueError("HBM planning requires nonnegative byte counts for every rank")
    counts = []
    for rank, layout in enumerate(layouts):
        remaining = usable_bytes[rank] - sum(values[rank] for values in vectors[1:])
        count = remaining // (sum(cache_page_bytes(layout).values()) + history_page_bytes) - 2
        if count < 1:
            raise ValueError(f"Resident model and scratch leave no cache capacity on rank {rank}")
        counts.append(count)
    return dict(page_capacity=min(counts), rank_page_capacities=counts,
                history_page_bytes=history_page_bytes,
                accounting="unique_storage_keys_and_history_with_two_reserved_pages")


class CachePagePool:
    """Retain fixed backing geometry while appending per-request logical pages."""

    def __init__(self, templates, plan, *, page_capacity, table_pages, page_rows=128, swa_pages=None):
        if (len(templates) != plan.ep or type(page_capacity) is not int or page_capacity < 1
                or type(table_pages) is not int or table_pages < 1 or page_rows != 128):
            raise ValueError("Require all rank templates and positive physical/logical page capacities")
        self.plan, self.page_capacity = plan, page_capacity
        self.table_pages, self.page_rows = table_pages, page_rows
        self.swa_pages = table_pages if swa_pages is None else swa_pages
        if type(self.swa_pages) is not int or not 1 <= self.swa_pages <= table_pages:
            raise ValueError("SWA retention must fit the logical page-table capacity")
        self.reserved = page_capacity + 1
        self.templates = templates
        self.free = [list(range(page_capacity, 0, -1)) for _ in range(plan.dp)]
        self.leases = {}
        self.layouts = []
        for template in templates:
            layout = deepcopy(template["layout"])
            sizes = cache_page_bytes(layout)
            for value in layout["views"].values():
                for plane in value if isinstance(value, list) else [value]:
                    count = page_capacity + 2
                    size = count * sizes[plane["storage_key"]]
                    if size >= 2**63:
                        raise ValueError("Cache backing byte extent exceeds signed INT64")
                    plane["shape"][0], plane["storage_bytes"] = count, size
            for key, raw in layout["raw"].items():
                raw["shape"] = [(page_capacity + 2) * sizes[key]]
                raw["storage_bytes"] = raw["shape"][0]
            self.layouts.append(layout)

    def reserve(self, batch):
        """Validate the entire dispatch before modifying any request page lease."""
        required = [0] * self.plan.dp
        changes = []
        for row in batch.requests:
            handle = row.handle
            rank = self.plan.rank(handle.dp_rank)
            groups = self.templates[rank]["groups"]
            owned = self.leases.get(handle, [[] for _ in groups])
            for index, group in enumerate(groups):
                ring = all(name.endswith("compressor.state_cache") for name in group["names"])
                swa = all(name.endswith("swa_cache") for name in group["names"])
                needed = (1 if ring else self.swa_pages if swa else
                          (row.sequence_length + self.page_rows - 1) // self.page_rows)
                if needed > self.table_pages:
                    raise ValueError("Request exceeds its independent logical page-table capacity")
                additional = max(0, needed - len(owned[index]))
                required[handle.dp_rank] += additional
                changes.append((handle, index, len(owned[index]), additional))
        if any(count > len(self.free[dp]) for dp, count in enumerate(required)):
            raise MemoryError("Cache page budget exhausted; queue or reject before device dispatch")
        previous = dict(new_handles=set(), updates=[])
        for handle, index, logical_start, additional in changes:
            if handle not in self.leases:
                groups = self.templates[self.plan.rank(handle.dp_rank)]["groups"]
                self.leases[handle] = [[] for _ in groups]
                previous["new_handles"].add(handle)
            pages = [self.free[handle.dp_rank].pop() for _ in range(additional)]
            self.leases[handle][index].extend(pages)
            if pages:
                previous["updates"].append((handle, index, logical_start, pages))
        return previous

    def rollback(self, previous):
        for handle, group, logical_start, pages in reversed(previous["updates"]):
            del self.leases[handle][group][logical_start:]
            self.free[handle.dp_rank].extend(reversed(pages))
        for handle in previous["new_handles"]:
            del self.leases[handle]

    def pages(self, handle):
        if handle not in self.leases:
            raise ValueError("Request generation does not own cache pages")
        return tuple(tuple(pages) for pages in self.leases[handle])

    def group_pages(self, handle, group):
        if handle not in self.leases:
            raise ValueError("Request generation does not own cache pages")
        return tuple(self.leases[handle][group])

    def physical_page(self, handle, group, logical_page):
        if handle not in self.leases:
            raise ValueError("Request generation does not own cache pages")
        pages = self.leases[handle][group]
        names = self.templates[self.plan.rank(handle.dp_rank)]["groups"][group]["names"]
        if all(name.endswith("swa_cache") for name in names):
            return pages[logical_page % len(pages)]
        return pages[logical_page]

    def release(self, handle, reset):
        """Reset every owned physical page before making any page available for reuse."""
        pages = self.pages(handle)
        reset(handle, pages)
        self.free[handle.dp_rank].extend(page for group in pages for page in group)
        del self.leases[handle]

    def summary(self):
        return dict(page_capacity=self.page_capacity, logical_table_pages=self.table_pages,
                    swa_retained_pages=self.swa_pages,
                    live_pages=[self.page_capacity - len(pages) for pages in self.free],
                    unique_page_bytes=[sum(cache_page_bytes(layout).values()) for layout in self.layouts],
                    reserved_untouched_block_id=self.reserved)

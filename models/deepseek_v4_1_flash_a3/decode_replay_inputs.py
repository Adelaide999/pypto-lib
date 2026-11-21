# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Stable input storage and per-token bindings for single-request Decode replay."""

from dataclasses import dataclass
from math import prod

import torch

from .indexer import INDEXER_SELECT_KEY_CAPACITY
from .utils import window_schedule_from_controls


# These are the token-dependent inputs of the existing one-token layer composition.
# The remaining operands are device intermediates, weights or fixed layout controls.
INPUTS = {
    "lookup": {"token_ids": "full_ids", "live": "live"},
    "rope_tables": {"positions": "positions"},
    "source_rope_tables": {"positions": "source_positions"},
    "ffn_dispatch_l2": {"token_ids": "local_ids"},
    "cache_publish": {"slots": "swa_slots"},
    "swa_read": {"indices": "window"},
    "network_swa_publish": {"slots": "swa_slots"},
    "network_compressed_publish": {"slots": "long_slots"},
    "index_publish": {"slots": "index_slots"},
    "c2_pool": {"positions": "positions"},
    "index_select": {"positions": "query_positions"},
    "sparse_controls": {"positions": "query_positions"},
    "candidate_pool": {"compressed_lengths": "visible_lengths"},
    "engram_projection": {"source": "engram"},
    "engram_gate": {"token_mask": "engram_mask"},
}
SCALAR_REF = -(1 << 60)


@dataclass(frozen=True)
class Binding:
    field: str
    layer: int
    dtype: torch.dtype
    shape: tuple


class DecodeReplayInputs:
    """Pack variable operands into one aligned host/device buffer per worker."""

    def __init__(self, rank):
        self.rank = rank
        self.offsets = {}
        self.size = 0
        self.host = None
        self.examples = {}

    def reserve(self, binding):
        if binding not in self.offsets:
            offset = (self.size + 63) // 64 * 64
            self.offsets[binding] = offset
            self.size = offset + prod(binding.shape) * binding.dtype.itemsize
        return self.offsets[binding]

    def tensor(self, name, argument, layer, value, original):
        field = INPUTS.get(name, {}).get(argument)
        if field is None:
            return None
        if field in ("full_ids", "local_ids", "live", "positions", "query_positions", "engram_mask"):
            layer = 0
        binding = Binding(field, layer, value.dtype, tuple(value.shape))
        if binding not in self.examples:
            self.examples[binding] = original.clone()
        elif not torch.equal(self.examples[binding], original):
            raise ValueError(f"Conflicting Decode input {binding} on rank {self.rank}")
        return -1 - self.reserve(binding)

    def scalar(self, field, layer):
        return SCALAR_REF + self.reserve(Binding(field, layer, torch.int64, (1,)))

    def update(self, provider, settings, lookup, tp_base, ep_base, cache_blocks):
        if self.host is None:
            self.host = torch.zeros((self.size + 63) // 64 * 64, dtype=torch.uint8)
            self.views = {key: self.host.narrow(0, offset, prod(key.shape) * key.dtype.itemsize)
                          .view(key.dtype).view(key.shape) for key, offset in self.offsets.items()}
            self.signal_rows = {}
            for field in ("tp_signal", "ep_signal"):
                keys = [key for key in self.views if key.field == field]
                self.signal_rows[field] = (torch.tensor([self.offsets[k] // 8 for k in keys], dtype=torch.int64),
                                           torch.tensor([k.layer for k in keys], dtype=torch.int64))
            self.token_views = [(k, v) for k, v in self.views.items() if k.field not in self.signal_rows]
        words = self.host.view(torch.int64)
        for field, base in (("tp_signal", tp_base), ("ep_signal", ep_base)):
            offsets, relative = self.signal_rows[field]
            words.index_copy_(0, offsets, relative + base)
        engine = settings["DP_engines"][self.rank // provider.topology["tp"]]
        masks = [m for req in engine["requests"] for m in req["live_token_mask"]]
        ready = lookup["cases"]["fresh"][self.rank]
        for key, dst in self.token_views:
            field, layer = key.field, key.layer
            row = provider.batch_for_layer(layer)["cases"]["fresh"][self.rank]
            pos = row["positions"]
            count = row["actual_local_query_rows"]
            own = f"model.layers.{layer}.self_attn."
            role = row["role"]
            source = f"model.layers.{role['kv_source']}.self_attn."
            meta = row["cache_metadata"]["actual_metadata"]
            if field == "engram":
                dst.copy_(ready["local"][layer])
            elif field == "engram_mask":
                dst.copy_(ready["local_mask"])
            elif field == "window":
                controls = row["metadata"]
                schedule = window_schedule_from_controls(controls["query_start_loc"].tolist(),
                    controls["seq_lens"].tolist(), controls["block_table"].tolist(),
                    cache_blocks=cache_blocks, capacity=provider.capacity)
                dst.fill_(-1)
                if count:
                    dst[0].copy_(torch.tensor(schedule["physical_window_indices"][0], dtype=dst.dtype))
            elif field.endswith("_slots"):
                suffix = {"swa_slots": "swa_cache", "long_slots": "long_kv_cache",
                          "index_slots": "indexer.k_cache"}[field]
                metadata = meta[own + suffix]["global_metadata"]
                slots = metadata["slot_mapping"]
                if len(pos):
                    page, index = slots[0].tolist()
                    limit = 128 if field == "swa_slots" else 128 // role["ratio"]
                    if (page, index) != (-1, -1) and (not 0 <= index < limit
                            or page not in metadata["block_table"][0].tolist()):
                        raise ValueError("Decode publication escapes its physical cache plane")
                dst.fill_(-1)
                dst[:len(slots)].copy_(slots)
            elif field in ("cache_length", "candidate_blocks"):
                length, = meta[source + "indexer.k_cache"]["global_metadata"]["cache_seq_lens"].tolist()
                if not 0 <= length <= INDEXER_SELECT_KEY_CAPACITY:
                    raise ValueError("Decode index history exceeds the compiled capacity")
                dst.fill_(length if field == "cache_length" else (min(length, int(pos[0]) + 1) + 7) // 8)
            else:
                dst.zero_()
                if field == "source_positions" and role["ratio"] == 2:
                    values = meta[own + "compressor.state_cache"]["global_metadata"]["c2_source_positions"]
                elif field in ("positions", "source_positions"):
                    values = pos
                elif field == "query_positions":
                    values = pos[:count]
                elif field == "visible_lengths":
                    values = (pos[:count] + 1) // role["ratio"]
                elif field == "full_ids":
                    values = row["full_input_ids"]
                elif field == "local_ids":
                    values = row["input_ids"]
                elif field == "live":
                    values = torch.tensor(masks, dtype=dst.dtype)
                else:
                    raise ValueError(f"Unknown Decode replay operand {field}")
                dst[:len(values)].copy_(values)
        for key, expected in self.examples.items():
            if not torch.equal(self.views[key], expected):
                raise ValueError(f"Captured Decode input differs from its dynamic binding: rank {self.rank} {key}")
        self.examples.clear()
        return self.host

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One static checkpoint, host preparation and device bank per A3 generation."""

from time import perf_counter

from .cache import decode_cache_views
from .decode_network_weights import DecodeNetworkWeights
from .weights import ResidentNetworkWeights


class A3ResidentWeights:
    """Own canonical operands and identical Prefill/Decode allocation mappings."""

    def __init__(self, checkpoint, topology, model_config, runtime):
        self.checkpoint, self.topology = str(checkpoint), dict(topology)
        self.world, self.runtime = int(topology["ep"]), runtime
        self.prefill = ResidentNetworkWeights(checkpoint, topology, model_config, runtime)
        self.decode = DecodeNetworkWeights(self)
        self.prefill_handles = [dict() for _ in range(self.world)]
        self.decode_handles = [dict() for _ in range(self.world)]
        self.cache_handles = [dict() for _ in range(self.world)]
        self.cache_alias_receipts = []
        self.initialization_timings = {}
        self.ready, self._closed = False, False
        self._frozen_counters = None

    @property
    def reader(self):
        return self.prefill.reader

    @property
    def layout(self):
        return self.prefill.layout

    @property
    def load_seconds(self):
        return self.prefill.load_seconds

    @property
    def load_calls(self):
        return self.prefill.load_calls

    @property
    def layer_receipts(self):
        return self.prefill.layer_receipts

    @property
    def device_upload_calls(self):
        return self.runtime.stats["device_weight_upload_calls"]

    def counters(self):
        return dict(checkpoint_read_calls=self.reader.read_calls,
                    checkpoint_file_open_calls=self.reader.file_open_calls + int(self.ready),
                    host_weight_preparation_calls=sum(self.reader.preparations.values()),
                    device_weight_upload_calls=self.device_upload_calls)

    def preload_once(self, progress=None):
        """Read and prepare each logical weight once for both phases' common ABI."""
        if self._closed:
            raise RuntimeError("A3 resident weights are closed")
        if self.ready:
            self.assert_static_unchanged()
            return dict(self.initialization_timings)
        begin = perf_counter()
        self.prefill.preload(progress)
        rotation_begin = perf_counter()
        rotation, _ = self.runtime.load_rotation(self.checkpoint, 5120)
        self.reader.read_calls += 1
        self.reader.read_seconds += perf_counter() - rotation_begin
        self.reader.consumed["optional/global_rotation"] = dict(
            file="optional/quarot.safetensors", read_calls=1, prepared_shape=[32, 32],
            prepared_dtype=str(rotation.dtype), selected_rows=None,
        )
        self.reader.preparations["engram.rotation"] = 1
        self.runtime.pin(rotation, range(self.world), readonly=True, name="engram.rotation")
        for rank in range(self.world):
            handles = self.prefill_handles[rank]
            for (owner_rank, _), owner in self.runtime.pinned.items():
                if owner_rank == rank and owner["readonly"]:
                    handles[owner["name"]] = self.runtime.handle(owner["source"], rank)
            for (layer, owner_rank, expert), values in self.prefill.experts.items():
                if owner_rank == rank:
                    prefix = f"layers.{layer}.ffn.{expert if expert is not None else 'shared'}"
                    for name, handle in zip(("gate_up", "gate_scale", "down", "down_scale"), values):
                        handles[prefix + "." + name] = handle
            self.decode_handles[rank].update(handles)
        self.reader.seal()
        self.ready = True
        self._frozen_counters = self.counters()
        initialization_seconds = perf_counter() - begin
        upload_seconds = self.runtime.stats["device_weight_upload_seconds"]
        self.initialization_timings = dict(
            **self._frozen_counters,
            checkpoint_read_seconds=self.reader.read_seconds,
            host_weight_preparation_seconds=initialization_seconds - self.reader.read_seconds - upload_seconds,
            device_weight_upload_seconds=upload_seconds,
            static_initialization_seconds=initialization_seconds,
        )
        return dict(self.initialization_timings)

    def assert_static_unchanged(self):
        """Reject checkpoint reads, host packing or static uploads in a token step."""
        if not self.ready or self.counters() != self._frozen_counters:
            raise RuntimeError("Static weights changed after generation session preload")

    def for_layer(self, layer):
        return self.prefill.for_layer(layer)

    def raw_for_layer(self, layer):
        return self.prefill.raw_for_layer(layer)

    def projection_for_layer(self, layer):
        return self.prefill.projection_for_layer(layer)

    def head_for_rank(self, rank):
        return self.prefill.head_for_rank(rank)

    def release_current(self):
        """Keep the complete bank alive across physical layers and tokens."""

    def bind_cache_views(self, rank, prefill_handles, source_map=None):
        """Retain exact aliases of the source pools, including byte offsets and owners."""
        if not 0 <= rank < self.world:
            raise ValueError("Cache rank is outside the resident world")
        views = decode_cache_views(prefill_handles, source_map)
        self.cache_handles[rank].update(views)
        mapping = source_map if source_map is not None else {}
        for name, view in views.items():
            descriptor = mapping.get(name, name)
            source_name = descriptor[0] if isinstance(descriptor, tuple) else descriptor
            source = prefill_handles[source_name]
            self.cache_alias_receipts.append(dict(
                rank=rank, name=name, source_pointer=source.data_ptr, view_pointer=view.data_ptr,
                same_buffer=source.buffer is view.buffer, byte_offset=view.byte_offset,
                view_bytes=view.nbytes, source_bytes=source.nbytes,
            ))
        return views

    def summary(self):
        counts = self.counters()
        shared = [all(handle.data_ptr == self.decode_handles[rank][name].data_ptr
                      and handle.buffer is self.decode_handles[rank][name].buffer
                      for name, handle in handles.items())
                  for rank, handles in enumerate(self.prefill_handles)]
        return dict(
            owner="A3ResidentWeights", checkpoint=self.checkpoint, topology=self.topology,
            prefill_ready=self.ready, decode_ready=self.ready, **counts,
            device_weight_upload_call_count=self.device_upload_calls,
            device_weight_bytes=list(self.runtime.stats["weight_bytes"]),
            initialization_timings=self.initialization_timings,
            logical_checkpoint_tensors=len(self.reader.consumed),
            max_logical_tensor_read_calls=max((v["read_calls"] for v in self.reader.consumed.values()), default=0),
            host_preparation_groups=len(self.reader.preparations),
            shared_weight_allocation_identity=shared,
            prefill_handle_count=[len(row) for row in self.prefill_handles],
            decode_handle_count=[len(row) for row in self.decode_handles],
            cache_aliases=list(self.cache_alias_receipts),
            routed_expert_ownership=[self.layout.ownership(rank) for rank in range(self.world)],
            extra_device_representations=0, closed=self._closed,
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.prefill.close()
        for handles in self.prefill_handles + self.decode_handles + self.cache_handles:
            handles.clear()

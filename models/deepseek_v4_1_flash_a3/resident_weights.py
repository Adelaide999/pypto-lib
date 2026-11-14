# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generation-scoped A3 Prefill/Decode weight ownership.

The phase-specific readers retain their numerical ABI adapters, while this
owner controls their lifetime and the accounting shared by a complete
generation.  A Decode reader may therefore be attached to an already prepared
Prefill session instead of being constructed for every token.
"""

from __future__ import annotations

from time import perf_counter

from .cache import decode_cache_views


class A3ResidentWeights:
    """Own all static A3 operands for one Prefill/Decode generation session."""

    def __init__(self, checkpoint, topology, model_config, runtime):
        from .decode_network_weights import DecodeNetworkWeights
        from .weights import ResidentNetworkWeights

        self.checkpoint = str(checkpoint)
        self.topology = dict(topology)
        self.world = int(self.topology["ep"])
        self.runtime = runtime
        self.prefill = ResidentNetworkWeights(checkpoint, topology, model_config, runtime)
        self.decode = DecodeNetworkWeights(checkpoint, topology["ep"])
        self.prefill_handles = [dict() for _ in range(topology["ep"])]
        self.decode_handles = [dict() for _ in range(topology["ep"])]
        self.cache_handles = [dict() for _ in range(topology["ep"])]
        self.initialization_timings = {}
        self._prefill_ready = False
        self._decode_ready = False
        self._closed = False

    @property
    def reader(self):
        """Expose the canonical checkpoint reader for Prefill diagnostics."""
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
        """Expose Prefill's per-layer checkpoint accounting to runtime reports."""
        return self.prefill.layer_receipts

    @property
    def static_names(self):
        return self.decode.static_names

    @property
    def device_upload_calls(self):
        """Expose Decode's one-time device upload accounting to its driver."""
        return self.decode.device_upload_calls

    @property
    def image_ids(self):
        return self.decode.image_ids

    @property
    def token_map(self):
        return self.decode.token_map

    @token_map.setter
    def token_map(self, value):
        self.decode.token_map = value

    def preload_prefill(self, progress=None):
        """Upload Prefill operands once and keep their runtime allocations alive."""
        if self._closed:
            raise RuntimeError("A3 resident weights are closed")
        if self._prefill_ready:
            return dict(self.initialization_timings)
        begin = perf_counter()
        self.prefill.preload(progress)
        self._prefill_ready = True
        self.initialization_timings.update(
            checkpoint_read_seconds=self.prefill.reader.read_seconds,
            checkpoint_read_calls=self.prefill.reader.read_calls,
            host_weight_preparation_seconds=self.prefill.load_seconds,
            prefill_weight_initialization_seconds=perf_counter() - begin,
        )
        return dict(self.initialization_timings)

    def preload_decode(self, runtime=None):
        """Prepare Decode's ABI adapter once for the same generation owner.

        ``runtime`` is accepted for compatibility with the existing Decode
        network.  A caller may provide the session runtime when its compiled
        entries support direct resident aliases; otherwise the Decode network
        owns only the execution wrapper while this object still guarantees a
        single Decode preload across all steps.
        """
        if self._closed:
            raise RuntimeError("A3 resident weights are closed")
        if self._decode_ready:
            return dict(self.initialization_timings)
        if not self._prefill_ready:
            self.preload_prefill()
        begin = perf_counter()
        timings = self.decode.preload(runtime) if runtime is not None else None
        if timings:
            self.initialization_timings.update(timings)
        self.initialization_timings["decode_weight_initialization_seconds"] = perf_counter() - begin
        self._decode_ready = True
        return dict(self.initialization_timings)

    def preload(self, runtime=None):
        """Compatibility entry point used by the standalone Decode network.

        The generation path owns both phases from one resident bank.  Keep the
        method name accepted by the older Decode driver, but always route it
        through the generation-scoped Decode preload so repeated Decode steps
        cannot allocate or upload another bank.
        """
        return self.preload_decode(runtime)

    def layer(self, layer):
        if not self._decode_ready:
            raise RuntimeError("Decode weights were not preloaded for this session")
        return self.decode.layer(layer)

    def for_layer(self, layer):
        return self.prefill.for_layer(layer)

    def raw_for_layer(self, layer):
        return self.prefill.raw_for_layer(layer)

    def projection_for_layer(self, layer):
        return self.prefill.projection_for_layer(layer)

    def head_for_rank(self, rank):
        return self.prefill.head_for_rank(rank)

    def embedding(self):
        return self.decode.embedding()

    def head(self):
        return self.decode.head()

    def engram(self, layer):
        return self.decode.engram(layer)

    def engram_rows(self, tokens, histories):
        return self.decode.engram_rows(tokens, histories)

    def release_current(self):
        """Keep every static operand alive until the session closes."""

    def bind_cache_views(self, rank, prefill_handles, source_map=None):
        """Bind Decode cache names to the Prefill rank-local allocation."""
        if not 0 <= rank < len(self.cache_handles):
            raise ValueError("cache rank is outside the resident world")
        views = decode_cache_views(prefill_handles, source_map)
        self.cache_handles[rank].update(views)
        return views

    def summary(self):
        checkpoint_reads = self.prefill.reader.read_calls + self.decode.reader.read_calls
        return dict(
            owner="A3ResidentWeights",
            checkpoint=self.checkpoint,
            topology=self.topology,
            prefill_ready=self._prefill_ready,
            decode_ready=self._decode_ready,
            cache_aliases=[sorted(row) for row in self.cache_handles],
            checkpoint_read_calls=checkpoint_reads,
            device_weight_upload_calls=self.prefill.runtime.stats.get("weight_bytes", []),
            device_weight_upload_call_count=self.decode.device_upload_calls,
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.decode.close()
        self.prefill.close()
        for handles in self.prefill_handles + self.decode_handles + self.cache_handles:
            handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


__all__ = ["A3ResidentWeights"]

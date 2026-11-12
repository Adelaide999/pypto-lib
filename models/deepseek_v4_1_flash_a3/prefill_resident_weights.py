# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Load forty A3 layer banks and every owned W8A8 expert into generation-scoped HBM."""

from time import perf_counter

from .prefill_layer_state import resolve_layer_role
from .prefill_swa import prepare_product_operands
from .runtime_config import ExpertParallelConfig
from .weights import CheckpointOperands


class ResidentCheckpointOperands(CheckpointOperands):
    """Cache model boundary weights while streaming expert source tensors once."""

    def __init__(self, root):
        super().__init__(root, diagnostics=False)
        self.boundaries = {}

    def load(self, name):
        if name in self.boundaries:
            return self.boundaries[name]
        value = super().load(name)
        if name in ("embed.weight", "norm.weight", "head.weight"):
            self.boundaries[name] = value
        return value


class ResidentNetworkWeights:
    """Retain complete compute weights, with routed experts sharded by their actual EP owner."""

    def __init__(self, checkpoint, topology, model_config, runtime):
        self.reader = ResidentCheckpointOperands(checkpoint)
        self.layout = ExpertParallelConfig(topology["dp"], topology["tp"], topology["ep"], 384)
        self.model_config, self.runtime = model_config, runtime
        self.layers, self.experts, self.heads = {}, {}, {}
        self.layer_receipts = {}
        self.load_seconds, self.load_calls = 0.0, 0
        self.ready = False

    def preload(self, progress=None):
        """Stream each checkpoint operand into its persistent compute owner before token execution."""
        if self.ready or self.layers:
            raise ValueError("Resident weight preload is a single generation-scoped operation")
        begin = perf_counter()
        world, tp = self.layout.ep, self.layout.tp
        ranks = range(world)
        for layer in range(40):
            if progress is not None:
                progress(dict(stage="loading_resident_weights", layer=layer, epoch=0))
            prefix = f"layers.{layer}."
            names = [name for name in self.reader.index if name.startswith(prefix)
                     and ".ffn.experts." not in name and ".engram.embed." not in name]
            raw = {name: self.reader.load(name) for name in names}
            operands = prepare_product_operands(raw, layer, role=resolve_layer_role(layer))
            raw[prefix + "ffn.gate.weight"] = raw[prefix + "ffn.gate.weight"].float()
            for name in ("hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "ffn_norm.weight",
                         "ffn.gate.weight", "ffn.gate.bias", "ffn.gate.bias_vl"):
                self.runtime.pin(raw[prefix + name], ranks, readonly=True, name=prefix + name)
            for name, value in operands.items():
                if name in ("wo_a", "wo_a_grouped_prefill", "wo_b"):
                    continue
                self.runtime.pin(value, ranks, readonly=True, name=prefix + name)
            for rank in ranks:
                lane = rank % tp
                groups, width = 8 // tp, 8192 // tp
                for name, value in (
                    ("wo_a", operands["wo_a_grouped_prefill"][lane * groups:(lane + 1) * groups]),
                    ("wo_b", operands["wo_b"][lane * width:(lane + 1) * width]),
                ):
                    self.runtime.pin(value, [rank], readonly=True, name=prefix + name)
            role = resolve_layer_role(layer)
            if role["engram_slot"] is not None:
                engram = prefix + "engram"
                projection = raw[engram + ".wkv.weight"].T.contiguous()
                operands["engram_projection"] = projection
                self.runtime.pin(projection, ranks, readonly=True, name=engram + ".wkv.kn")
                for name in ("q_weight", "k_weight"):
                    self.runtime.pin(raw[engram + "." + name], ranks, readonly=True, name=engram + "." + name)
            self.layers[layer] = dict(raw=raw, operands=operands,
                                      expert=lambda rank, expert, layer=layer: self.expert(layer, rank, expert))
            shared = self.reader.expert(prefix + "ffn.shared_experts")
            for rank in ranks:
                self.experts[layer, rank, None] = tuple(self.runtime.weight(value, rank) for value in shared)
            del shared
            for rank in ranks:
                for expert, _ in self.layout.ownership(rank):
                    values = self.reader.expert(prefix + f"ffn.experts.{expert}")
                    self.experts[layer, rank, expert] = tuple(self.runtime.weight(value, rank) for value in values)
            self.layer_receipts[layer] = dict(original_names=names, policy="generation-scoped resident compute weights",
                                              routed_experts=384, shared_replicas=world)
        embedding, norm, head = (self.reader.load(name) for name in ("embed.weight", "norm.weight", "head.weight"))
        rows = embedding.shape[0] // tp
        for rank in ranks:
            lane = rank % tp
            self.runtime.pin(embedding.narrow(0, lane * rows, rows), [rank], readonly=True, name="embed.weight")
            self.runtime.pin(norm, [rank], readonly=True, name="norm.weight")
            if lane not in self.heads:
                self.heads[lane] = head.narrow(0, lane * rows, rows).T.contiguous()
            self.runtime.pin(self.heads[lane], [rank], readonly=True, name="head.weight.kn")
        self.ready = True
        self.load_seconds = perf_counter() - begin
        self.load_calls = self.reader.read_calls
        self.runtime.stats.update(weight_preload_seconds=self.load_seconds,
                                  checkpoint_preload_read_calls=self.reader.read_calls,
                                  checkpoint_preload_read_seconds=self.reader.read_seconds)

    def for_layer(self, layer):
        if not self.ready or layer not in self.layers:
            raise ValueError("Every resident layer must be loaded before execution")
        return self.layers[layer]

    def raw_for_layer(self, layer):
        return self.for_layer(layer)["raw"]

    def projection_for_layer(self, layer):
        return self.for_layer(layer)["operands"]["engram_projection"]

    def expert(self, layer, rank, expert):
        if expert is not None and self.layout.owner(expert) != rank:
            raise ValueError("Resident expert belongs to another EP rank")
        return self.experts[layer, rank, expert]

    def head_for_rank(self, rank):
        return self.heads[rank % self.layout.tp]

    def release_current(self):
        """Preserve generation-scoped layers across all layer and token boundaries."""

    def close(self):
        self.layers.clear()
        self.experts.clear()
        self.heads.clear()
        self.reader.boundaries.clear()

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile and reuse the existing A3 Prefill entries across all original layers."""

from hashlib import sha256
import json

from pathlib import Path
from time import perf_counter

from pypto import ir

from .prefill_embedding import make_input_entries
from .engram import make_prefill_engram_entries
from .prefill_layer import make_prefill_layer_entries
from .prefill_layer_state import resolve_layer_role
from .prefill_l2 import REGIONS, make_network_l2_entries
from .prefill_network_cache import network_cache_geometry
from .prefill_output import make_output_entries
from .runtime_utils import Executable, tensor_sha, write_json


def network_layer_geometry(provider, topology, layer_id):
    """Check the live routed row bound without rebuilding any operator IR."""
    geometry = provider.runtime_geometry()
    batch = provider.batch_for_layer(layer_id)
    total = max(sum(record["SP_valid_rows"] for record in records) for records in batch["cases"].values())
    # One expert can receive every valid token once; SP shards partition each DP chunk.
    bound = geometry["capacity"] * topology["dp"]
    if total > bound:
        raise ValueError("Actual routed token rows exceed the compiled per-DP chunk capacity")
    expert_capacity = max(16, (bound + 15) // 16 * 16)
    return geometry, expert_capacity


def make_network_layer_entries(provider, config, topology, layer_id, definitions_cache=None):
    """Build the exact layer specialization from its validated request/cache geometry."""
    geometry, expert_capacity = network_layer_geometry(provider, topology, layer_id)
    batch = provider.batch_for_layer(layer_id)
    layout = provider.layouts[0][next(iter(batch["cases"]))]
    role = resolve_layer_role(layer_id)
    planes = {} if role["mode"] == "swa" else network_cache_geometry(layout, role, geometry["cache_blocks"])
    # Physical ownership is checked above and by the runtime; only byte geometry enters child IR.
    semantic_role = {
        name: value
        for name, value in role.items()
        if name not in ("physical_layer", "kv_source", "index_source", "candidate_source", "engram_slot")
    }
    plane_types = {
        name: {key: value for key, value in plane.items() if key != "storage_key"}
        for name, plane in planes.items()
    }
    key = json.dumps(
        [geometry, expert_capacity, config, topology, semantic_role, plane_types], sort_keys=True
    )
    if definitions_cache is not None and key in definitions_cache:
        return definitions_cache[key], geometry, expert_capacity
    entries = make_prefill_layer_entries(
        geometry["capacity"],
        config,
        expert_capacity,
        layer_id,
        geometry["cache_blocks"],
        geometry["requests"],
        geometry["table_pages"],
        ep=topology["ep"],
        tp=topology["tp"],
        cache_layout=layout,
        role=role,
        network_cache=True,
    )
    definitions = make_network_l2_entries(entries, geometry["capacity"])
    if definitions_cache is not None:
        definitions_cache[key] = definitions
    return definitions, geometry, expert_capacity


def _compile_signature(provider, model_config, topology, engram_contracts):
    """Freeze all compile-time geometry and constants, excluding token values and live cache bytes."""

    def encode(value):
        if hasattr(value, "dtype") and hasattr(value, "shape"):
            return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=tensor_sha(value))
        raise TypeError("Unsupported compile-time configuration value: " + type(value).__name__)

    batch = provider.hash_batch()
    max_blocks = max(
        1,
        max(
            record["metadata"]["block_table"].shape[1]
            for records in batch["cases"].values()
            for record in records
        ),
    )
    signature = dict(
        geometry=provider.runtime_geometry(),
        layouts=provider.layouts,
        model=model_config,
        topology=topology,
        history_rows=provider.history.cache_rows,
        max_blocks=max_blocks,
        engram={
            layer: {name: contract[name] for name in ("model_config", "hash_constants", "engram_shards")}
            for layer, contract in engram_contracts.items()
        },
    )
    return sha256(json.dumps(signature, sort_keys=True, default=encode).encode()).hexdigest()


class PrefillProgram:
    """Own compiled entries; share only structurally equal IR with identical argument names."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stages, self._buckets = {}, {}
        self._specializations, self._layer_definitions = {}, {}
        self.records = []
        self.compile_seconds = 0.0
        self.ready = False

    def bind(self, stage, definitions):
        """Compile one stage or verify its existing specialization before reuse."""
        if not stage or "/" in stage or "\\" in stage or ".." in stage:
            raise ValueError("Compiled stages require a safe unique name")
        if stage in self.stages and self.stages[stage].keys() != definitions.keys():
            raise ValueError("The compiled stage entry set changed: " + stage)
        if self.ready and stage not in self.stages:
            raise ValueError("The compiled network does not cover stage: " + stage)
        entries = {}
        for name, function in definitions.items():
            if not name or "/" in name or "\\" in name or ".." in name:
                raise ValueError("Compiled entries require safe names")
            identity = id(function)
            if identity not in self._specializations:
                program = function.specialize()
                key = (ir.structural_hash(program), tuple(function.param_names))
                # Retain the function so an object ID cannot be recycled during this program's lifetime.
                self._specializations[identity] = (function, program, key)
            _, program, key = self._specializations[identity]
            bucket = self._buckets.get(key, [])
            match = next((item for item in bucket if ir.structural_equal(item["ir"], program)), None)
            if stage in self.stages:
                if match is None or match["entry"] is not self.stages[stage][name]:
                    raise ValueError(
                        "The compiled stage geometry or computation changed: " + stage + "/" + name
                    )
            elif match is None:
                if self.ready:
                    raise ValueError("The compiled network does not cover stage: " + stage + "/" + name)
                directory = self.directory / f"entry-{len(self.records):04d}-{name}"
                begin = perf_counter()
                entry = Executable(function, directory)
                seconds = perf_counter() - begin
                self.compile_seconds += seconds
                match = dict(ir=program, entry=entry)
                self._buckets.setdefault(key, []).append(match)
                self.records.append(
                    dict(first_stage=stage, name=name, seconds=seconds, artifacts=entry.artifacts)
                )
            entries[name] = match["entry"]
        self.stages[stage] = entries
        return entries

    def entries_for_stage(self, stage):
        """Use a stage only after the complete program's compile-time signature has been admitted."""
        if not self.ready or stage not in self.stages:
            raise ValueError("The compiled network does not cover stage: " + stage)
        return self.stages[stage]

    def compile(self, provider, model_config, topology, engram_contracts):
        """Compile embedding, forty complete layers, both Engram slots and the final head."""
        signature = _compile_signature(provider, model_config, topology, engram_contracts)
        if self.ready:
            if signature != self._signature:
                raise ValueError("The compiled network geometry or constants changed")
            return self
        begin = perf_counter()
        text = dict(model_config["text_config"], image_sentinel_base_id=model_config["image_token_id"])
        geometry = provider.runtime_geometry()
        try:
            self.bind(
                "embedding",
                make_network_l2_entries(
                    make_input_entries(geometry["capacity"], text, topology["tp"]),
                    geometry["capacity"],
                    "embedding",
                ),
            )
            for layer in range(40):
                definitions, _, _ = make_network_layer_entries(
                    provider, text, topology, layer, self._layer_definitions
                )
                self.bind(f"layer{layer}", definitions)
            batch = provider.hash_batch()
            max_blocks = max(
                1,
                max(
                    record["metadata"]["block_table"].shape[1]
                    for records in batch["cases"].values()
                    for record in records
                ),
            )
            for layer in (1, 14):
                contract = engram_contracts[layer]
                config = dict(
                    contract["model_config"]["text_config"], image_token_id=model_config["image_token_id"]
                )
                config["image_pad_token_id"] = model_config.get(
                    "image_pad_token_id", config["image_token_id"] + 1
                )
                definitions = make_prefill_engram_entries(
                    geometry["capacity"],
                    contract["hash_constants"],
                    config,
                    contract["engram_shards"],
                    provider.history.cache_rows,
                    max_blocks,
                    topology["tp"],
                )
                self.bind(
                    f"engram{layer}",
                    {name: definitions[name] for name in ("engram_projection", "engram_gate")},
                )
            self.bind(
                "output",
                make_network_l2_entries(
                    make_output_entries(geometry["capacity"], text, topology["tp"]),
                    geometry["capacity"],
                    "output",
                ),
            )
            self._signature = signature
            self.ready = True
        finally:
            self.prepare_seconds = perf_counter() - begin
            write_json(self.directory / "compilation.json", self.summary())
        return self

    def summary(self):
        """Report actual compiler work separately from device execution and accuracy."""
        return dict(
            complete=self.ready,
            layers=[layer for layer in range(40) if f"layer{layer}" in self.stages],
            execution_model="host_hccl_with_l2_compute_regions",
            single_l2=False,
            l2_regions={name: [child for child, _ in steps] for name, steps in REGIONS.items()},
            stages={stage: list(entries) for stage, entries in self.stages.items()},
            unique_entries=len(self.records),
            stage_bindings=sum(map(len, self.stages.values())),
            layer_definition_specializations=len(self._layer_definitions),
            specialized_functions=len(self._specializations),
            compile_seconds=self.compile_seconds,
            compile_phase="pypto_ir_pto_cpp_generation",
            device_binary_build_and_weight_loading="not_measured_here",
            prepare_seconds=getattr(self, "prepare_seconds", None),
            entries=self.records,
            device_execution="not_run",
            numerical_acceptance="not_run",
        )

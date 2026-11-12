# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Whole-network Prefill, reusable stage compilation and runtime orchestration."""

import argparse
import json
import sys
from collections.abc import Mapping
from contextlib import ExitStack
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"
import torch
from pypto import ir
from .cache import attach_network_cache_geometry, network_cache_geometry, resolve_layer_role
from .engram import make_prefill_engram_entries
from .input_pack import make_input_entries
from .lm_head import make_output_entries
from .prefill_layer import REGIONS, execute_region, make_network_l2_entries, make_prefill_layer_entries, run_prefill_layer_inmemory
from .prefill_network_engram import prepare_network_engram, validate_host_lookup
from .prefill_network_input import run_input_boundary
from .runtime_utils import Executable, file_sha, loaded_source_files, tensor_sha, write_json
from .state import NetworkCacheProvider


def compile_prefill(inputs, *, cache, config, output, weights=None):
    """Compile every original network stage from canonical request/cache geometry.

    This uses metadata and storage-free meta tensors to derive exact shapes.
    It does not read checkpoint payloads, allocate cache backing storage,
    initialize CANN, execute kernels or establish accuracy.
    The returned program can be supplied to ``prefill_fwd`` in this process.
    Raw text or token IDs require ``weights`` for checkpoint metadata and use
    the same full-prompt cache geometry as automatic execution.
    """

    if isinstance(inputs, (str, list, tuple)):
        from .prefill_prepare import prepare_token_chunks

        if weights is None:
            raise ValueError("Automatic compile-only inputs require an explicit checkpoint via weights")
        with TemporaryDirectory(prefix="ds41-prefill-compile-") as temporary:
            chunks = prepare_token_chunks(
                inputs,
                checkpoint=weights,
                cache=cache,
                config=config,
                output=Path(temporary) / "prepared",
                chunk_size=config.get("chunk_size", 16),
                dp_rank=config.get("dp_rank", 0),
            )
            try:
                first = next(chunks)
                return compile_prefill(
                    first["inputs"], cache=first["cache"], config=first["contract"], output=output
                )
            finally:
                chunks.close()
    operands = _merge_inputs(inputs, cache)
    directory = Path(output).resolve()
    provider = prepare_network_provider(operands, config, directory.name, geometry_only=True)
    return PrefillProgram(directory).compile(
        provider, config["model_config"], config["topology"], operands["engram_contracts"]
    )


def _merge_inputs(inputs, cache):
    overlap = set(inputs) & set(cache)
    if overlap:
        raise ValueError("Input and cache configuration have duplicate fields: " + ", ".join(sorted(overlap)))
    return dict(inputs, **cache)


def prefill_fwd(
    inputs,
    *,
    weights,
    cache,
    config,
    devices,
    output,
    compiled=None,
    continuation=None,
    progress=None,
    engram_lookup=None,
):
    """Execute the existing network using explicit checkpoint and cache-state providers.

    ``inputs`` accepts raw completion text, 1..2048 token IDs, or prepared
    backbone tensors and their input contract. Automatic inputs use one DP
    engine and chunks of up to 16 tokens; config may set chunk_size/dp_rank.
    Embedding and initial HC execute inside this forward unless explicit
    prepared backbone tensors are supplied.
    ``cache`` supplies the canonical all-rank cache ABI, source descriptors and
    Engram contracts. No historical task release or layer receipt is required.
    ``continuation`` accepts one prepared chunk or a lazy iterable of chunks;
    each chunk follows the preceding input's hash and next epoch.
    ``engram_lookup`` contains host-gathered BF16 rows for both slots.
    Each prepared continuation supplies its own rows in inputs["engram_lookup"].
    The forward performs only device projection/gating on these ready rows.
    Ordinary execution preloads all compute weights and retains workers,
    caches and communication groups until the complete stream ends.
    """
    from .runtime_config import configure_devices

    chunks = None
    try:
        configure_devices(devices, world=config["topology"]["ep"])
        if isinstance(inputs, (str, list, tuple)):
            from .prefill_prepare import prepare_token_chunks

            if continuation is not None or Path(output).exists():
                raise ValueError(
                    "Automatic Prefill needs a new output directory and owns its continuation stream"
                )
            chunks = prepare_token_chunks(
                inputs,
                checkpoint=weights,
                cache=cache,
                config=config,
                output=Path(output) / "prepared",
                chunk_size=config.get("chunk_size", 16),
                dp_rank=config.get("dp_rank", 0),
                prepare_engram=True,
            )
            first = next(chunks)
            inputs, cache, config = first["inputs"], first["cache"], first["contract"]
            output, continuation = Path(output) / "network", chunks

        if engram_lookup is not None:
            if "engram_lookup" in inputs:
                raise ValueError("Pass host Engram rows once, as an argument or prepared input field")
            inputs = dict(inputs, engram_lookup=engram_lookup)
        return run_network_runtime(
            _merge_inputs(inputs, cache),
            contract=config,
            checkpoint=weights,
            devices=devices,
            output=output,
            compiled=compiled,
            continuation=continuation,
            progress=progress,
        )
    finally:
        if chunks is not None:
            chunks.close()


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


def network_parameter_abi(compiled):
    """Validate the complete saved signature and returned assembly write slots without loading runtime."""
    from pypto import ir
    from pypto.ir.compiled_artifact import CompiledArtifact
    from pypto.ir.compiled_program import extract_param_infos

    def unique_members(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate compiled metadata member: " + key)
            result[key] = value
        return result

    path = Path(compiled.output_dir).resolve() / "compiled_meta.json"
    digest = file_sha(path)
    metadata = json.loads(path.read_text(), object_pairs_hook=unique_members)
    if type(metadata.get("schema")) is not int or metadata["schema"] != 3:
        raise ValueError("Unsupported network compiled parameter metadata schema")
    # Both public reload paths read metadata only; load/callable access is excluded.
    validated = ir.CompiledProgram.from_dir(path.parent)
    artifact = CompiledArtifact.from_dir(path.parent)
    if artifact.metadata is None or artifact.metadata.record() != metadata:
        raise ValueError("Compiled parameter metadata is incomplete, unknown or noncanonical")
    params, lowered = metadata["params"], metadata["lowered_params"]
    names = [item["name"] for item in params]
    pure_outputs = [index for index, item in enumerate(params) if item["direction"] == "Out"]
    for entry in (compiled, validated):
        if (
            names != entry.param_names
            or pure_outputs != entry.output_indices
            or bool(metadata["num_return_types"]) != entry.has_return
            or metadata["platform"] != entry.platform
            or metadata["backend_type"] != entry.backend_type.name
        ):
            raise ValueError("Saved JIT ABI differs from the exact compiled entry")
    if len(params) != len(lowered):
        raise ValueError("Lowering changed the external JIT parameter count")
    aliases = metadata["return_aliases"]
    directions, proofs = [], []
    for index, (logical, actual) in enumerate(zip(params, lowered)):
        if any(logical[field] != actual[field] for field in ("name", "shape", "dtype", "layout")):
            raise ValueError("Lowering changed external JIT storage slot: " + logical["name"])
        before, after = logical["direction"], actual["direction"]
        if before not in {"In", "Out", "InOut"} or after not in {"In", "Out", "InOut"}:
            raise ValueError("Unknown network JIT parameter direction")
        if logical["shape"] is None and (before != "In" or after != "In"):
            raise ValueError("Network scalar parameters must remain In")
        alias_positions = [] if aliases is None else [i for i, value in enumerate(aliases) if value == index]
        if before == "In" and after in {"Out", "InOut"}:
            if logical["shape"] is None or not alias_positions:
                raise ValueError("Lowered write lacks an exact tensor return alias: " + logical["name"])
            effective, source = after, "same-slot-lowered-write-and-validated-return-alias"
        elif before == "Out" and after == "InOut":
            if logical["shape"] is None or not alias_positions:
                raise ValueError(
                    "Lowered output readback lacks an exact tensor return alias: " + logical["name"]
                )
            effective, source = after, "same-slot-output-readback-and-validated-return-alias"
        elif before == after:
            effective, source = before, "explicit-logical-direction"
        else:
            raise ValueError("Unsupported logical/lowered JIT direction change: " + logical["name"])
        directions.append(effective)
        if effective != "In":
            proofs.append(
                dict(
                    index=index,
                    name=logical["name"],
                    logical_direction=before,
                    lowered_direction=after,
                    effective_direction=effective,
                    return_positions=alias_positions,
                    source=source,
                )
            )
    live_checked = compiled.program is not None
    if live_checked:
        original, original_outputs, returns = extract_param_infos(compiled.program)
        records = [
            dict(
                name=item.name,
                direction=item.direction.name,
                shape=None if item.shape is None else list(item.shape),
                dtype=str(item.dtype),
                layout=item.layout,
            )
            for item in original
        ]
        if (
            records != params
            or original_outputs != pure_outputs
            or len(returns) != metadata["num_return_types"]
        ):
            raise ValueError("Saved JIT ABI differs from the original live Program")
        functions = list(compiled.program.functions.values())
        entries = [item for item in functions if item.func_type == ir.FunctionType.Orchestration]
        if len(entries) != 1:
            if not entries and len(functions) == 1:
                entries = functions
            else:
                raise ValueError("Network JIT requires one unambiguous live Program entry")
        traced = ir.returned_param_indices(entries[0], compiled.program)
        known = len(traced) == len(returns) and all(
            type(index) is int and 0 <= index < len(params) and params[index]["shape"] is not None
            for index in traced
        )
        live_aliases = list(traced) if known else None
        if live_aliases != aliases:
            raise ValueError("Saved return aliases differ from the original live Program")
    if file_sha(path) != digest:
        raise ValueError("Compiled parameter metadata changed during ABI validation")
    return dict(
        path=str(path),
        sha256=digest,
        schema=3,
        params=params,
        lowered_params=lowered,
        return_aliases=aliases,
        num_return_types=metadata["num_return_types"],
        effective_directions=directions,
        writable_proofs=proofs,
        live_program_aliases_checked=live_checked,
        parameter_mapping="same ordered name/shape/dtype/layout slots",
        validation="public CompiledProgram.from_dir and CompiledArtifact.from_dir; no runtime load",
    )


class NetworkExecutor:
    """Dispatch shared compiled stages through persistent device workers and HCCL groups."""

    def __init__(
        self,
        topology,
        provider,
        ownership,
        memory,
        reports,
        program,
        namespace="chunk0",
        resident=None,
    ):
        self.topology = topology
        self.provider, self.ownership, self.memory = provider, ownership, memory
        self.reports = reports
        self.namespace = namespace
        self.program = program
        self.resident = resident
        self.current_layer, self.current_entries = None, None
        self.compiled = {}
        self.resident_abis = {}
        # The public ChipWorker retains registrations until close (64 slots).
        # IO entries use only CPU boundaries; give them independent workers and
        # buffers so resident expert operands keep their original live owners.
        self.staged_entries = {
            id(entry.compiled)
            for stage in ("embedding", "output")
            for entry in program.entries_for_stage(stage).values()
        }
        direct_entries = {
            id(entry.compiled)
            for stage, entries in program.stages.items()
            if stage not in ("embedding", "output")
            for name, entry in entries.items()
            if not name.startswith("engram_lookup_")
        }
        if len(direct_entries) > 64 or len(self.staged_entries) > 64:
            raise ValueError("The network entry partition exceeds public worker registration capacity")

    def execute_entry(self, entry, rank, arguments):
        """Bind one existing compiled entry to the generation's persistent worker."""
        if self.resident is None:
            raise ValueError("Ordinary execution requires an initialized resident runtime")
        key = id(entry.compiled)
        if key not in self.resident_abis:
            self.resident_abis[key] = network_parameter_abi(entry.compiled)
        self.resident.execute(
            entry.compiled, rank, arguments, self.resident_abis[key], staged=key in self.staged_entries
        )

    def exchange(self, stage, values, collective, group, *, packets=False):
        if not stage or "/" in stage or ".." in stage:
            raise ValueError("HCCL stages require distinct safe chunk/layer identities")
        transport = (
            values if packets else [{case: {"value": value} for case, value in row.items()} for row in values]
        )
        result, receipts = self.resident.exchange(stage, transport, collective, group)
        for rank, receipt in enumerate(receipts):
            self.reports[rank]["communication"].append(receipt)
        if packets:
            return result
        return [{case: packet["value"] for case, packet in row.items()} for row in result]

    def precheck(self):
        world = self.topology["ep"]
        values = [{"precheck": torch.tensor([[rank]], dtype=torch.int32)} for rank in range(world)]
        for group in ("ep", "tp", "dp"):
            observed = self.exchange(
                self.namespace + "-network-startup-" + group, values, "all_gather", group
            )
            for rank in range(world):
                tp = self.topology["tp"]
                expected = (
                    list(range(world))
                    if group == "ep"
                    else list(range(rank // tp * tp, rank // tp * tp + tp))
                )
                if group == "dp":
                    expected = list(range(rank % tp, world, tp))
                if observed[rank]["precheck"].flatten().tolist() != expected:
                    raise RuntimeError("Actual network HCCL group precheck failed: " + group)

    def for_layer(self, layer_id):
        geometry, expert_capacity = network_layer_geometry(self.provider, self.topology, layer_id)
        batch = self.provider.batch_for_layer(layer_id)
        if self.current_layer != layer_id:
            self.current_entries = None
            self.current_entries = self.program.entries_for_stage(f"layer{layer_id}")
            self.compiled[layer_id] = {name: entry.artifacts for name, entry in self.current_entries.items()}
            self.current_layer = layer_id

        def execute(name, rank, case, arguments, outputs, responsibility=None):
            self.execute_entry(self.current_entries[name], rank, arguments)

        return dict(
            topology=self.topology,
            ownership=self.ownership,
            enable_expert_parallel=True,
            capacity=geometry["capacity"],
            expert_capacity=expert_capacity,
            cache_blocks=geometry["cache_blocks"],
            requests=geometry["requests"],
            table_pages=geometry["table_pages"],
            memory=self.memory,
            execute=execute,
            execute_l2=execute,
            exchange=self.exchange,
            epoch=batch["epoch"],
            token_map_id=batch["token_map_id"],
            output_sentinel=-37.0,
            stage_namespace=f"{self.namespace}-layer{layer_id}",
        )


def embed_network_input(input_contract, weights, executor, config, output):
    """Run the shared vocabulary lookup and initial HC directly into the backbone."""
    directory = Path(output)
    directory.mkdir(exist_ok=False)
    entries = executor.program.entries_for_stage("embedding")

    def execute(name, rank, arguments, outputs):
        executor.execute_entry(entries[name], rank, arguments)

    def exchange(stage, packets):
        return executor.exchange(executor.namespace + "-" + stage, packets, "all_gather", "tp", packets=True)

    begin = perf_counter()
    boundaries = run_input_boundary(
        input_contract,
        config=config,
        capacity=executor.provider.capacity,
        reader=weights.reader,
        execute=execute,
        exchange=exchange,
        execute_l2=execute,
    )
    write_json(
        directory / "execution.json",
        dict(
            status="executed_not_independently_compared",
            seconds=perf_counter() - begin,
            ranks=len(boundaries),
            layers_executed=[],
        ),
    )
    hidden, premix = [], []
    for rank, original in enumerate(boundaries):
        record = executor.provider.batch_for_layer(0)["cases"]["fresh"][rank]
        if any(original[name] != record[name] for name in ("SP_physical_rows", "SP_valid_rows")):
            raise ValueError("Embedding and cache metadata disagree on rank-local token ownership")
        local_hidden = torch.zeros(
            (executor.provider.capacity, config["hc_mult"], config["hidden_size"]), dtype=torch.bfloat16
        )
        local_pre = torch.zeros((executor.provider.capacity, config["hc_mult"]), dtype=torch.float32)
        physical = original["SP_physical_rows"]
        local_hidden[:physical].copy_(original["hidden"])
        local_pre[:physical].copy_(original["pre_mix"])
        hidden.append(dict(fresh=local_hidden))
        premix.append(dict(fresh=local_pre))
    return hidden, premix


def run_output_boundary(result, *, batch, topology, config, weights, executor, output):
    """Consume actual layer39 local delayed HC state and restore complete SP/TP vocabulary output."""
    capacity, world, tp = executor.provider.capacity, topology["ep"], topology["tp"]
    directory = Path(output)
    directory.mkdir(exist_ok=False)
    entries = executor.program.entries_for_stage("output")
    norm = weights.reader.load("norm.weight")
    head = weights.reader.load("head.weight")
    if (
        norm.dtype != torch.bfloat16
        or head.dtype != torch.bfloat16
        or tuple(head.shape) != (config["vocab_size"], 5120)
    ):
        raise ValueError("Actual complete output norm/head checkpoint ABI differs")

    def execute(name, rank, case, arguments, outputs, responsibility=None):
        executor.execute_entry(entries[name], rank, arguments)

    packets = [{} for _ in range(world)]
    for case, records in batch["cases"].items():
        for rank, record in enumerate(records):
            physical = record["SP_physical_rows"]
            collapsed = torch.full((capacity, 5120), -17, dtype=torch.bfloat16)
            execute(
                "collapse",
                rank,
                case,
                [result["hidden"][rank][case], result["pre_mix"][rank][case], collapsed, physical],
                dict(local_collapsed=collapsed),
            )
            packets[rank][case] = collapsed
    gathered = executor.exchange(executor.namespace + "-output-SP-gather", packets, "all_gather", "tp")
    logits_packets = [{} for _ in range(world)]
    for case, records in batch["cases"].items():
        for rank, record in enumerate(records):
            physical, active = record["SP_physical_rows"], len(record["positions"])
            collapsed = torch.full((capacity, 5120), -17, dtype=torch.bfloat16)
            normalized = torch.full((capacity, 5120), -19, dtype=torch.bfloat16)
            local_logits = torch.full((capacity, config["vocab_size"] // tp), -23, dtype=torch.bfloat16)
            count = config["vocab_size"] // tp
            weight = weights.head_for_rank(rank)
            if tuple(weight.shape) != (5120, count):
                raise ValueError("Output head must use canonical KN vocabulary shard layout")
            execute_region(
                {"execute_l2": execute},
                "head_l2",
                rank,
                case,
                (
                    (
                        "sp_unpack",
                        [gathered[rank][case], collapsed, max(physical, 1), active],
                        dict(collapsed=collapsed),
                        "output",
                    ),
                    ("norm", [collapsed, norm, normalized, active], dict(normalized=normalized), "output"),
                    (
                        "projection",
                        [normalized, weight, local_logits, active],
                        dict(local_logits=local_logits),
                        "output",
                    ),
                ),
            )
            logits_packets[rank][case] = local_logits
    gathered_logits = executor.exchange(
        executor.namespace + "-output-vocabulary-gather", logits_packets, "all_gather", "tp"
    )
    files, actual_logits = {}, [{} for _ in range(world)]
    for case, records in batch["cases"].items():
        for rank, record in enumerate(records):
            logits = torch.full((capacity, config["vocab_size"]), -29, dtype=torch.bfloat16)
            execute(
                "vocab_unpack",
                rank,
                case,
                [gathered_logits[rank][case], logits, len(record["positions"])],
                dict(logits=logits),
            )
            path = directory / f"rank-{rank}-{case}-network-logits.pt"
            torch.save(
                dict(
                    logits=logits,
                    active=len(record["positions"]),
                    positions=record["positions"],
                    DP_rank=rank // tp,
                ),
                path,
            )
            files[str(path)] = file_sha(path)
            actual_logits[rank][case] = logits[: len(record["positions"])].clone()
    write_json(
        directory / "execution.json",
        dict(
            outcome="executed_actual_layer39_output_boundary_not_compared",
            files=files,
            native_reference="not_run",
            full40_accuracy="not_run",
        ),
    )
    return dict(files=files, logits=actual_logits)


def _network_owner_snapshot(provider):
    """Bind persistent allocation identities across continuous request chunks."""
    result = {"provider": id(provider), "history": id(provider.history), "arrays": {}}
    for family, values in (
        ("raw", provider.arenas),
        ("shared", provider.shared),
        ("history", provider.history.caches),
    ):
        for rank, cases in enumerate(values):
            for case, tensors in cases.items():
                items = tensors.items() if isinstance(tensors, dict) else (("history", tensors),)
                for name, value in items:
                    key = f"{family}/{rank}/{case}/{name}"
                    record = dict(
                        object_id=id(value),
                        storage=value.untyped_storage().data_ptr(),
                        shape=list(value.shape),
                        dtype=str(value.dtype),
                    )
                    result["arrays"][key] = record
    return result


def prepare_network_provider(inputs, contract, allocation_id, *, geometry_only=False):
    """Allocate canonical cache owners and validate the complete original network ABI."""
    topology = contract["topology"]
    if topology != dict(dp=4, tp=4, ep=16, pp=1, pcp=1, dcp=1):
        raise ValueError("The current network runtime requires DP4/TP4/EP16/PP1/PCP1/DCP1")
    if (
        not all(contract["required_modes"].values())
        or contract["capacity"] != 16
        or contract["scheduler_capacity"] != 8192
    ):
        raise ValueError("Retain the original model modes and bounded actual scheduler ABI")
    if contract.get("fresh_shared_initialization") != {"topk_indices": -1}:
        raise ValueError("Freeze the same deterministic initial TopK storage as the independent reference")
    engram_contracts = inputs.get("engram_contracts")
    if not isinstance(engram_contracts, dict) or set(engram_contracts) != {1, 14}:
        raise ValueError("Real independent TP4 complete-table Engram slot contracts are still required")
    for item in engram_contracts.values():
        if item["topology"] != topology or len(item["engram_shards"]) != 4:
            raise ValueError("A TP2 or diagnostic Engram contract cannot be reused as TP4")

    model_config = contract["model_config"]
    provider = NetworkCacheProvider(
        {"fresh": inputs["cache_ABI"]},
        {"fresh": inputs["input_contract"]},
        topology=topology,
        model_config=model_config,
        capacity=16,
        scheduler_capacity=8192,
        allocation_id=allocation_id,
        sources_by_case={"fresh": inputs["cache_ABI_sources"]},
        geometry_only=geometry_only,
    )
    attach_network_cache_geometry(provider)
    provider.require_network_geometry()
    for cases in provider.shared:
        cases["fresh"]["topk_indices"].fill_(-1)
    provider.begin_chunk(epoch=0, token_map_id=contract["input_contract_sha256"])
    return provider


def run_network_runtime(
    inputs,
    *,
    contract,
    output,
    devices,
    checkpoint=None,
    continuation=None,
    compiled=None,
    progress=None,
):
    """Execute a fresh chunk and a lazy stream of continuations with shared live state.

    ``continuation`` is one chunk mapping or an iterable of chunk mappings.
    Each item supplies inputs, its contract and every rank's metadata sources.
    The iterable advances only after the preceding complete chunk returns.
    """
    if "engram_lookup" not in inputs:
        raise ValueError("Prepare Engram rows on the host before calling the forward")
    if continuation is None:
        following_chunks = iter(())
    elif isinstance(continuation, Mapping):
        following_chunks = iter((continuation,))
    elif isinstance(continuation, (str, bytes)):
        raise ValueError("Continuation must contain chunk mappings")
    else:
        try:
            following_chunks = iter(continuation)
        except TypeError as error:
            raise ValueError("Continuation must be a chunk mapping or iterable") from error
    topology = contract["topology"]
    if topology != dict(dp=4, tp=4, ep=16, pp=1, pcp=1, dcp=1):
        raise ValueError("The current network runtime requires DP4/TP4/EP16/PP1/PCP1/DCP1")
    from .runtime_config import configure_devices

    assigned = configure_devices(devices, world=topology["ep"])
    output = Path(output).resolve()
    if checkpoint is not None:
        contract = dict(contract, checkpoint=str(Path(checkpoint).resolve()))
    from .moe_transport import CANNMemory

    output.mkdir(parents=True, exist_ok=False)
    if progress is not None:
        progress(dict(stage="compiling_network", epoch=0))
    provider = prepare_network_provider(inputs, contract, output.name, geometry_only=True)
    validate_host_lookup(inputs["engram_lookup"], inputs["engram_contracts"][1], provider.hash_batch())
    program = compiled if compiled is not None else PrefillProgram(output / "compiled")
    if compiled is not None and not compiled.ready:
        raise ValueError("The supplied Prefill program is incomplete")
    program.compile(provider, contract["model_config"], topology, inputs["engram_contracts"])
    provider = prepare_network_provider(inputs, contract, output.name)
    with ExitStack() as stack:
        from .prefill_resident import ResidentNetworkRuntime
        from .weights import ResidentNetworkWeights

        resident = stack.enter_context(ResidentNetworkRuntime(assigned, topology))
        weights = ResidentNetworkWeights(
            contract["checkpoint"], topology, contract["model_config"], resident
        )
        stack.callback(weights.close)
        weights.preload(progress)
        for family, values in (("raw", provider.arenas), ("shared", provider.shared)):
            for rank, cases in enumerate(values):
                for case, tensors in cases.items():
                    items = tensors.items() if isinstance(tensors, dict) else (("history", tensors),)
                    for name, value in items:
                        resident.pin(value, [rank], readonly=False, name=f"{family}/{rank}/{case}/{name}")
        memory = CANNMemory(diagnostics=False)
        ownership = [weights.layout.ownership(rank) for rank in range(16)]
        first = _run_network_chunk(
            inputs,
            contract=contract,
            output=output,
            provider=provider,
            weights=weights,
            memory=memory,
            ownership=ownership,
            assigned=assigned,
            engram_contracts=inputs["engram_contracts"],
            program=program,
            epoch=0,
            progress=progress,
            resident=resident,
        )
        if continuation is None:
            return first
        completed = [first]
        previous_contract = contract
        fixed = ("topology", "checkpoint", "model_config", "required_modes", "capacity", "scheduler_capacity")
        for epoch, chunk in enumerate(following_chunks, start=1):
            if (
                not isinstance(chunk, Mapping)
                or not {"inputs", "contract", "metadata_sources"} <= chunk.keys()
            ):
                raise ValueError("Each continuation needs inputs, contract and metadata_sources")
            following, next_contract = chunk["inputs"], chunk["contract"]
            required = set(fixed) | {"input_contract_path", "input_contract_sha256"}
            if (
                not isinstance(following, Mapping)
                or "input_contract" not in following
                or not isinstance(next_contract, Mapping)
                or not required <= next_contract.keys()
            ):
                raise ValueError("Continuation needs prepared inputs and a complete network contract")
            if checkpoint is not None:
                next_contract = dict(next_contract, checkpoint=contract["checkpoint"])
            if any(next_contract[name] != contract[name] for name in fixed):
                raise ValueError("Continuation cannot change numerical/model/topology configuration")
            input_path = Path(next_contract["input_contract_path"])
            actual_input = following["input_contract"]
            if (
                not isinstance(actual_input, Mapping)
                or not {"epoch", "predecessor_input_sha256"} <= actual_input.keys()
            ):
                raise ValueError("Continuation input must identify its epoch and predecessor")
            if (
                file_sha(input_path) != next_contract["input_contract_sha256"]
                or json.loads(input_path.read_text()) != actual_input
                or actual_input["predecessor_input_sha256"] != previous_contract["input_contract_sha256"]
                or type(actual_input["epoch"]) is not int
                or actual_input["epoch"] != epoch
            ):
                raise ValueError("Continuation needs the next epoch and exact preceding input identity")
            sources = chunk["metadata_sources"]
            if not isinstance(sources, (list, tuple)) or len(sources) != topology["ep"]:
                raise ValueError("Continuation needs real original-builder metadata for every rank")
            metadata, receipts = [], []
            for rank, record in enumerate(sources):
                if not isinstance(record, Mapping) or not {"rank", "path", "sha256"} <= record.keys():
                    raise ValueError("Each continuation metadata source needs rank, path and sha256")
                if (
                    type(record["rank"]) is not int
                    or record["rank"] != rank
                    or file_sha(record["path"]) != record["sha256"]
                ):
                    raise ValueError("Actual continuation metadata source changed")
                metadata.append(torch.load(record["path"], map_location="cpu", weights_only=True))
                receipts.append(dict(rank=rank, metadata_sha256=record["sha256"]))
            if "engram_lookup" not in following:
                raise ValueError("Each continuation requires fresh host-prepared Engram rows")
            following = dict(following, cache_ABI=metadata)
            before = _network_owner_snapshot(provider)
            provider.begin_chunk(
                epoch=epoch,
                token_map_id=next_contract["input_contract_sha256"],
                input_by_case={"fresh": actual_input},
                metadata_by_case={"fresh": metadata},
                metadata_sources_by_case={"fresh": receipts},
            )
            after = _network_owner_snapshot(provider)
            if before != after:
                raise RuntimeError("Chunk admission changed persistent raw/shared/history allocation identities")
            validate_host_lookup(
                following["engram_lookup"], inputs["engram_contracts"][1], provider.hash_batch()
            )
            program.compile(provider, next_contract["model_config"], topology, inputs["engram_contracts"])
            chunk_output = output / f"continuation-chunk{epoch}"
            chunk_output.mkdir(exist_ok=False)
            write_json(
                chunk_output / "own-state-continuity.json",
                dict(
                    before=before,
                    after=after,
                    allocation_identity_exact=True,
                    bytes_compared=False,
                    previous_complete_output=True,
                    scope="Persistent allocation identities across request chunks",
                ),
            )
            result = _run_network_chunk(
                following,
                contract=next_contract,
                output=chunk_output,
                provider=provider,
                weights=weights,
                memory=memory,
                ownership=ownership,
                assigned=assigned,
                engram_contracts=inputs["engram_contracts"],
                program=program,
                epoch=epoch,
                progress=progress,
                resident=resident,
            )
            completed.append(result)
            previous_contract = next_contract
        if len(completed) == 1:
            return first
        return dict(completed[-1], completed_prefill_chunks=completed, complete_network_acceptance="not_run")


def _run_network_chunk(
    inputs,
    *,
    contract,
    output,
    provider,
    weights,
    memory,
    ownership,
    assigned,
    engram_contracts,
    program,
    epoch,
    progress=None,
    resident=None,
):
    """Execute embedding, the complete backbone and the final head with persistent owners."""
    begin = perf_counter()
    initial_load_seconds, initial_load_calls = weights.load_seconds, weights.load_calls
    initial_read_seconds, initial_read_calls = weights.reader.read_seconds, weights.reader.read_calls
    topology, model_config = contract["topology"], contract["model_config"]
    text = dict(model_config["text_config"], image_sentinel_base_id=model_config["image_token_id"])
    namespace = f"chunk{epoch}"
    if progress is not None:
        progress(dict(stage="executing_embedding", epoch=epoch))
    runtime_path = output / "runtime_config.json"
    runtime = dict(
        topology=topology,
        assigned_devices=assigned,
        logical_rank_mapping=list(range(16)),
        source_root=str(Path(__file__).resolve().parents[2]),
        source_files={},
        groups=dict(
            ep=[list(range(16))],
            tp=[list(range(dp * 4, dp * 4 + 4)) for dp in range(4)],
            dp=[[dp * 4 + lane for dp in range(4)] for lane in range(4)],
        ),
        status="executing_not_compared",
        epoch=epoch,
        namespace=namespace,
        input_contract_sha256=contract["input_contract_sha256"],
        fresh_shared_initialization=contract["fresh_shared_initialization"],
        fresh_initialization_applied=epoch == 0,
    )
    loaded_source_files(runtime)
    write_json(runtime_path, runtime)
    reports = [dict(rank=rank, communication=[]) for rank in range(16)]
    executor = NetworkExecutor(
        topology,
        provider,
        ownership,
        memory,
        reports,
        program,
        namespace,
        resident=resident,
    )
    executor.precheck()
    hidden, premix = embed_network_input(
        inputs["input_contract"], weights, executor, text, output / "embedding"
    )
    hash_batch = provider.hash_batch()
    common_engram = engram_contracts[1]
    if progress is not None:
        progress(dict(stage="preparing_engram", epoch=epoch))
    captured_constants = provider.hash_constants(layer_ids=[1, 14], pad_id=text["engram_pad_token_id"])
    for item in engram_contracts.values():
        for name in ("token_map", "multipliers", "primes", "offsets"):
            if not torch.equal(item["hash_constants"][name], captured_constants[name]):
                raise ValueError("TP4 Engram contract differs from actual captured hash constants")
        if (
            tuple(item["hash_constants"]["layer_ids"]) != (1, 14)
            or item["hash_constants"]["pad_id"] != captured_constants["pad_id"]
        ):
            raise ValueError("TP4 Engram contract differs from the original slot/pad layout")
    eng_runtime = dict(
        topology=topology,
        assigned_devices=assigned,
        capacity=16,
        stage_namespace=namespace + "-engram",
        output=output / "engram",
        reports=reports,
        memory=memory,
        exchange=executor.exchange,
        entries_for_stage=program.entries_for_stage,
        execute_entry=executor.execute_entry,
        projection_for_layer=weights.projection_for_layer,
    )
    eng_runtime["load_rotation"] = resident.load_rotation
    prepared_engram = prepare_network_engram(
        common_engram,
        hash_batch,
        provider.history,
        engram_contracts,
        weights.raw_for_layer,
        eng_runtime,
        lookup=inputs["engram_lookup"],
    )
    for layer in (1, 14):
        prepared_engram.attach(layer, provider.state_handles_for_layer(layer))
    def runtime_for_layer(layer):
        if progress is not None:
            progress(dict(stage="executing_layer", epoch=epoch, layer=layer))
        actual = executor.for_layer(layer)
        actual["required_modes"] = contract["required_modes"]
        actual["engram"] = prepared_engram.apply
        return actual

    completed_layers = []

    def observe_layer(layer, result):
        provider.observe_layer(layer, result)
        completed_layers.append(layer)
        if progress is not None:
            progress(dict(stage="layer_complete", epoch=epoch, layer=layer))
        print(dict(stage="full40_layer_complete", layer=layer), flush=True)
        weights.release_current()

    try:
        result = run_prefill_network_inmemory(
            hidden,
            premix,
            batch_for_layer=provider.batch_for_layer,
            cache_for_layer=provider.cache_for_layer,
            weights_for_layer=weights.for_layer,
            runtime_for_layer=runtime_for_layer,
            observe_layer=observe_layer,
        )
        if progress is not None:
            progress(dict(stage="executing_output_head", epoch=epoch))
        output_result = run_output_boundary(
            result,
            batch=provider.batch_for_layer(39),
            topology=topology,
            config=text,
            weights=weights,
            executor=executor,
            output=output / "actual-layer39-output",
        )
        loaded_source_files(runtime)
        runtime.update(
            status="executed_full40_not_compared",
            completed_layers=list(range(40)),
            actual_Layer39_boundary=True,
            final_output_files=output_result["files"],
            formal_accuracy="not_run",
        )
        runtime["resident_runtime"] = resident.summary()
        write_json(runtime_path, runtime)
        write_json(
            output / "execution.json",
            dict(
                at=datetime.now(timezone.utc).isoformat(),
                outcome="executed_actual_full40_Prefill_pending_independent_comparison",
                reports=reports,
                compiled=executor.compiled,
                compilation=program.summary(),
                timings=dict(
                    chunk_seconds=perf_counter() - begin,
                    weight_load_seconds=weights.load_seconds - initial_load_seconds,
                    weight_load_calls=weights.load_calls - initial_load_calls,
                    checkpoint_read_seconds=weights.reader.read_seconds - initial_read_seconds,
                    checkpoint_read_calls=weights.reader.read_calls - initial_read_calls,
                ),
                consumed_checkpoint=weights.reader.consumed,
                layer_host_weight_accounting=weights.layer_receipts,
                memory_samples=memory.samples,
                memory_intervals=memory.intervals,
                capacity_verified_by_execution=False,
                formal_accuracy="not_run",
                complete_network_acceptance="not_run",
            ),
        )
        return dict(runtime, logits=output_result["logits"], compilation=program.summary())
    except BaseException as error:
        write_json(
            output / "failure.json",
            dict(
                at=datetime.now(timezone.utc).isoformat(),
                error_type=type(error).__name__,
                error=str(error),
                last_attempted_layer=executor.current_layer,
                completed_layers=completed_layers,
                cached_chunk_state="May be mutated; preserve and discard before any retry",
                complete_network_acceptance="not_run",
            ),
        )
        raise


def run_prefill_network_inmemory(
    local_hidden,
    local_pre,
    *,
    batch_for_layer,
    cache_for_layer,
    weights_for_layer,
    runtime_for_layer,
    observe_layer=None,
):
    """Chain all original backbone roles using actual local product activations.

    Providers own allocation, weights, chunk metadata, queue-mapped runtime and
    persistent shared aliases. Final delayed collapse/norm/head is a separate
    caller-owned output stage, not a substitute for the layer39 boundary.
    """
    from .cache import resolve_layer_role

    namespaces = set()
    token_identity = None
    for layer_id in range(40):
        role = resolve_layer_role(layer_id)
        batch, runtime = batch_for_layer(layer_id), runtime_for_layer(layer_id)
        identity = (batch["epoch"], batch["token_map_id"])
        if token_identity is None:
            token_identity = identity
        if identity != token_identity:
            raise ValueError("A network chunk must preserve its request/token ownership through all layers")
        namespace = runtime.get("stage_namespace")
        if not namespace or namespace in namespaces or "/" in namespace or ".." in namespace:
            raise ValueError("Network layers require distinct safe chunk/layer stage namespaces")
        namespaces.add(namespace)
        result = run_prefill_layer_inmemory(
            role,
            batch,
            local_hidden,
            local_pre,
            cache_for_layer(layer_id),
            weights_for_layer(layer_id),
            runtime,
        )
        local_hidden, local_pre = result["hidden"], result["pre_mix"]
        if observe_layer is not None:
            observe_layer(layer_id, result)
        del result
    return {"hidden": local_hidden, "pre_mix": local_pre}


# Preserve the published entry name for callers of the original package API.
run_prefill = prefill_fwd


def load_configuration(path):
    """Read paths relative to a JSON configuration without importing the device runtime."""
    path = Path(path).resolve()
    settings = json.loads(path.read_text(encoding="utf-8"))
    automatic = set(settings) & {"prompt", "token_ids"}
    if len(automatic) > 1:
        raise ValueError("Choose either prompt or token_ids for automatic Prefill")
    if automatic:
        if "inputs" in settings:
            raise ValueError("Automatic Prefill cannot also supply prepared inputs")
        if "prompt" in settings and (not isinstance(settings["prompt"], str) or not settings["prompt"]):
            raise ValueError("prompt must contain raw completion text")
        if "token_ids" in settings and (
            not isinstance(settings["token_ids"], list)
            or not 1 <= len(settings["token_ids"]) <= 2048
            or any(type(token) is not int or token < 0 for token in settings["token_ids"])
        ):
            raise ValueError("token_ids must contain 1..2048 nonnegative integers")
        if settings.get("continuations"):
            raise ValueError("Automatic Prefill owns its continuation descriptors")
        for name, default, lower, upper in (("chunk_size", 16, 1, 16), ("dp_rank", 0, 0, 3)):
            value = settings.get(name, default)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in {lower}..{upper}")
            settings[name] = value
    required = {"checkpoint", "cache_state", "contract", "devices", "output"}
    if not automatic:
        required.add("inputs")
    missing = required - settings.keys()
    if missing:
        raise ValueError("Missing Prefill configuration fields: " + ", ".join(sorted(missing)))
    for name in ("checkpoint", "inputs", "cache_state", "contract", "output"):
        if name not in settings:
            continue
        value = Path(settings[name])
        settings[name] = value if value.is_absolute() else (path.parent / value).resolve()
    chunks = settings.get("continuations", [])
    if not isinstance(chunks, list):
        raise ValueError("continuations must be a list of prepared chunk descriptors")
    for chunk in chunks:
        if not isinstance(chunk, dict) or not {"inputs", "contract", "metadata_sources"} <= chunk.keys():
            raise ValueError("Each continuation needs inputs, contract and metadata_sources paths")
        for name in ("inputs", "contract", "metadata_sources"):
            if not isinstance(chunk[name], str) or not chunk[name]:
                raise ValueError("Continuation paths must be nonempty strings")
            value = Path(chunk[name])
            chunk[name] = value if value.is_absolute() else (path.parent / value).resolve()
    devices = settings["devices"]
    if (
        not isinstance(devices, list)
        or len(devices) != 16
        or any(type(value) is not int or value < 0 for value in devices)
        or len(set(devices)) != 16
    ):
        raise ValueError("The current full-network configuration requires 16 distinct physical device IDs")
    return settings


def load_continuations(chunks):
    """Load each prepared chunk and its rank metadata descriptors on demand."""
    import torch

    for chunk in chunks:
        metadata_path = Path(chunk["metadata_sources"])
        sources = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(sources, list) or len(sources) != 16:
            raise ValueError("Continuation metadata_sources must describe all 16 ranks")
        for rank, source in enumerate(sources):
            if not isinstance(source, dict) or not {"rank", "path", "sha256"} <= source.keys():
                raise ValueError("Each continuation metadata source needs rank, path and sha256")
            if type(source["rank"]) is not int or source["rank"] != rank:
                raise ValueError("Continuation metadata sources must retain rank order")
            value = Path(source["path"])
            source["path"] = str(value if value.is_absolute() else (metadata_path.parent / value).resolve())
        inputs = torch.load(chunk["inputs"], map_location="cpu", weights_only=True)
        contract = json.loads(Path(chunk["contract"]).read_text(encoding="utf-8"))
        yield dict(inputs=inputs, contract=contract, metadata_sources=sources)


def main():
    """Load an explicit full-network configuration and run the shared product path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("-p", "--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("--devices", help="Comma-separated physical device IDs; overrides the JSON mapping")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check-config", action="store_true", help="Validate the configuration without device execution"
    )
    parser.add_argument(
        "--compile-only", action="store_true", help="Compile the complete network without NPU execution"
    )
    args = parser.parse_args()
    settings = load_configuration(args.config)
    if args.devices is not None:
        settings["devices"] = [int(value) for value in args.devices.split(",")]
    if args.checkpoint is not None:
        settings["checkpoint"] = args.checkpoint.resolve()
    if args.output is not None:
        settings["output"] = args.output.resolve()
    # Bind device visibility before loading the compiler, Torch or CANN.
    from .runtime_config import configure_devices, validate_devices

    if args.check_config:
        validate_devices(settings["devices"], world=16)
        print(json.dumps({"configuration": "valid", "execution": "not_run", "platform": args.platform}))
        return
    if args.compile_only:
        validate_devices(settings["devices"], world=16)
    else:
        configure_devices(settings["devices"], world=16)
    import torch

    contract = json.loads(settings["contract"].read_text(encoding="utf-8"))
    if "prompt" in settings or "token_ids" in settings:
        inputs = settings["prompt"] if "prompt" in settings else settings["token_ids"]
        contract = dict(contract, chunk_size=settings["chunk_size"], dp_rank=settings["dp_rank"])
    else:
        inputs = torch.load(settings["inputs"], map_location="cpu", weights_only=True)
    cache = torch.load(settings["cache_state"], map_location="cpu", weights_only=True)
    if args.compile_only:
        program = compile_prefill(
            inputs, cache=cache, config=contract, output=settings["output"], weights=settings["checkpoint"]
        )
        print(json.dumps(program.summary()))
        return
    continuation = load_continuations(settings["continuations"]) if settings.get("continuations") else None
    if isinstance(inputs, dict):
        from .engram_host import HostEngram

        with HostEngram(settings["checkpoint"], cache["engram_contracts"], settings["output"].name) as host:
            ready = host.lookup(inputs, cache=cache, config=contract)
            source_chunks = continuation

            def prepared_continuations():
                for chunk in source_chunks:
                    payloads = [
                        torch.load(row["path"], map_location="cpu", weights_only=True)
                        for row in chunk["metadata_sources"]
                    ]
                    ready = host.lookup(
                        chunk["inputs"], cache=dict(cache, cache_ABI=payloads), config=chunk["contract"]
                    )
                    yield dict(chunk, inputs=dict(chunk["inputs"], engram_lookup=ready))

            result = prefill_fwd(
                inputs,
                weights=settings["checkpoint"],
                cache=cache,
                config=contract,
                devices=settings["devices"],
                output=settings["output"],
                engram_lookup=ready,
                continuation=prepared_continuations() if source_chunks is not None else None,
            )
    else:
        result = prefill_fwd(
            inputs,
            weights=settings["checkpoint"],
            cache=cache,
            config=contract,
            devices=settings["devices"],
            output=settings["output"],
        )
    print(
        json.dumps(
            dict(
                completed_layers=result["completed_layers"],
                final_output_files=result["final_output_files"],
                numerical_acceptance=result["formal_accuracy"],
            )
        )
    )


if __name__ == "__main__":
    main()

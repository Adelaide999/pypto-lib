# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared TP/EP network programs, request execution and generation session."""

import json
import sys
from collections.abc import Mapping
from contextlib import ExitStack
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig
from .cache import attach_network_cache_geometry, network_cache_geometry, resolve_layer_role
from .engram import make_prefill_engram_entries
from .input_pack import make_input_entries
from .lm_head import make_output_entries
from .prefill_layer import (
    REGIONS, execute_region, region_arguments, make_network_l2_entries,
    make_prefill_layer_entries, run_prefill_layer_inmemory,
)
from .prefill_network_engram import prepare_network_engram, validate_host_lookup
from .utils import file_sha, tensor_sha, write_json
from .state import NetworkCacheProvider


class Executable:
    """Compile one production operator definition and retain its compiler artifacts."""

    def __init__(self, function, directory):
        print({"stage": "compile", "operator": directory.name}, flush=True)
        config = RunConfig(arch=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                           save_kernels=True, save_kernels_dir=str(directory))
        self.compiled = function.compile(config=config)
        self.artifacts = {}
        for pattern in ("*.pto", "*.cpp", "*.so", "kernel_config.py"):
            self.artifacts.update({str(path): file_sha(path) for path in sorted(directory.rglob(pattern))})
        if not any(path.endswith(".pto") for path in self.artifacts):
            raise RuntimeError("PyPTO compilation produced no retained PTO artifact")
        print({"stage": "compiled", "operator": directory.name, "directory": str(directory)}, flush=True)


def loaded_source_files(runtime):
    root = Path(runtime["source_root"])
    model = root / "models/deepseek_v4_1_flash_a3"
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path is not None:
            path = Path(path).resolve()
            if path.is_relative_to(model) and path.suffix == ".py":
                runtime["source_files"][str(path.relative_to(root))] = file_sha(path)


def run_input_boundary(settings, *, config, capacity, reader, execute, exchange, execute_l2=None):
    """Embed the actual request tokens and return rank-local HC inputs in memory."""
    topology = settings["topology"]
    dp, tp, world = topology["dp"], topology["tp"], topology["ep"]
    hidden, vocabulary, hc_mult = config["hidden_size"], config["vocab_size"], config["hc_mult"]
    if world != dp * tp or len(settings["DP_engines"]) != dp:
        raise ValueError("Input request engines must match the complete network topology")
    if any(not 0 <= engine["active"] <= capacity for engine in settings["DP_engines"]):
        raise ValueError("Actual request tokens exceed the compiled Prefill capacity")
    weight = reader.load("embed.weight")
    if tuple(weight.shape) != (vocabulary, hidden) or weight.dtype != torch.bfloat16:
        raise ValueError("Actual checkpoint embedding shape or dtype differs")
    packets, rows_by_rank = [], []
    for rank in range(world):
        engine = settings["DP_engines"][rank // tp]
        tokens = [token for request in engine["requests"] for token in request["token_ids"]]
        masks = [value for request in engine["requests"] for value in request["live_token_mask"]]
        if len(tokens) != engine["active"] or len(masks) != len(tokens):
            raise ValueError("Request/token ownership differs from the actual prepared boundary")
        if any(token < 0 or token >= vocabulary for token in tokens):
            raise ValueError("Prepared token ID is outside the original vocabulary")
        ids, live = torch.zeros(capacity, dtype=torch.int64), torch.zeros(capacity, dtype=torch.int32)
        ids[:len(tokens)] = torch.tensor(tokens, dtype=torch.int64)
        live[:len(tokens)] = torch.tensor(masks, dtype=torch.int32)
        output = torch.zeros((capacity, hidden), dtype=torch.bfloat16)
        start, count = rank % tp * (vocabulary // tp), vocabulary // tp
        local_weight = weight.narrow(0, start, count).contiguous()
        execute("lookup", rank, [ids, live, local_weight, output, start, len(tokens)], dict(lookup=output))
        packets.append({"network-input": {"value": output}})
        rows_by_rank.append(len(tokens))
    gathered = exchange("input-vocabulary-TP-gather", packets)
    result = []
    for rank in range(world):
        active = rows_by_rank[rank]
        physical = max(1, (active + tp - 1) // tp)
        embedding = torch.zeros((capacity, hidden), dtype=torch.bfloat16)
        streams = torch.zeros((capacity, hc_mult, hidden), dtype=torch.bfloat16)
        premix = torch.zeros((capacity, hc_mult), dtype=torch.float32)
        operations = (
            ("reduce", [gathered[rank]["network-input"]["value"], embedding, active], dict(embedding=embedding), "input"),
            ("initial_hc", [embedding, streams, premix, rank % tp, physical, active],
             dict(hidden=streams, pre_mix=premix), "input"),
        )
        if execute_l2 is None:
            for name, arguments, outputs, _ in operations:
                execute(name, rank, arguments, outputs)
        else:
            arguments, outputs, _ = region_arguments("input_l2", operations)
            execute_l2("input_l2", rank, arguments, outputs)
        valid = max(0, min(physical, active - rank % tp * physical))
        result.append(dict(hidden=streams[:(active + tp - 1) // tp].clone(),
            pre_mix=premix[:(active + tp - 1) // tp].clone(),
            SP_physical_rows=(active + tp - 1) // tp, SP_kernel_rows=physical, SP_valid_rows=valid, DP_rank=rank // tp, TP_rank=rank % tp,
            global_token_start=rank % tp * physical))
    return result


def _merge_inputs(inputs, cache):
    overlap = set(inputs) & set(cache)
    if overlap:
        raise ValueError("Input and cache configuration have duplicate fields: " + ", ".join(sorted(overlap)))
    return dict(inputs, **cache)


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


class NetworkProgram:
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

    def execute_entry(self, entry, rank, arguments):
        """Bind one existing compiled entry to the generation's persistent worker."""
        if self.resident is None:
            raise ValueError("Ordinary execution requires an initialized resident runtime")
        key = id(entry.compiled)
        if key not in self.resident_abis:
            self.resident_abis[key] = network_parameter_abi(entry.compiled)
        self.resident.execute(entry.compiled, rank, arguments, self.resident_abis[key])

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
    if (topology.get("ep") != 16 or topology.get("tp") not in (1, 2, 4)
            or topology.get("dp") * topology.get("tp") != 16
            or any(topology.get(name) != 1 for name in ("pp", "pcp", "dcp"))):
        raise ValueError("The network runtime requires EP16 with TP1/2/4 and PP1/PCP1/DCP1")
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
        raise ValueError("Real complete-table Engram slot contracts are required")
    for item in engram_contracts.values():
        if item["topology"] != topology or len(item["engram_shards"]) != topology["tp"]:
            raise ValueError("Engram contract topology differs from the shared network")

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


class GenerationSession:
    """Own both phases' workers, weights, cache aliases and monotonically advancing state."""

    def __init__(self, stack, *, resident, weights, provider, program, contract, engram_contracts,
                 memory, assigned, output, request_inputs=None):
        self._stack = stack
        self.runtime = self.resident = resident
        self.weights, self.provider, self.program = weights, provider, program
        self.contract, self.engram_contracts = contract, engram_contracts
        self.memory, self.assigned, self.output = memory, assigned, output
        self.request_inputs = request_inputs
        self.ownership = [weights.layout.ownership(rank) for rank in range(weights.world)]
        self.epoch, self.closed, self.failed = 0, False, False
        self.cache_views = weights.cache_handles
        for rank, cases in enumerate(provider.arenas):
            sources = {case + "/" + name: resident.handle(value, rank)
                       for case, tensors in cases.items() for name, value in tensors.items()}
            weights.bind_cache_views(rank, sources, {name: name for name in sources})

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self._stack.close()
            finally:
                if self.request_inputs is not None:
                    self.request_inputs.close()

    def release_runtime(self):
        """End the session explicitly; generation never releases a worker between phases."""
        self.close()

    def advance(self, chunk, *, output, progress=None, phase="prefill"):
        """Admit fresh controls and run the next chunk over the existing weights and caches."""
        if self.closed or self.failed:
            raise RuntimeError("A closed or failed generation session cannot advance")
        following, contract = chunk["inputs"], chunk["contract"]
        fixed = ("topology", "checkpoint", "model_config", "required_modes", "capacity", "scheduler_capacity")
        if any(contract[name] != self.contract[name] for name in fixed):
            raise ValueError("Continuation cannot change the session's model or topology")
        actual, epoch = following["input_contract"], self.epoch + 1
        input_path = Path(contract["input_contract_path"])
        if (actual["epoch"] != epoch or type(actual["epoch"]) is not int
                or actual["predecessor_input_sha256"] != self.contract["input_contract_sha256"]
                or file_sha(input_path) != contract["input_contract_sha256"]
                or json.loads(input_path.read_text()) != actual):
            raise ValueError("Continuation needs the exact next epoch and preceding input identity")
        sources = chunk["metadata_sources"]
        if len(sources) != self.weights.world:
            raise ValueError("Continuation metadata must cover every active and empty rank")
        metadata, receipts = [], []
        for rank, record in enumerate(sources):
            if record["rank"] != rank or file_sha(record["path"]) != record["sha256"]:
                raise ValueError("Continuation metadata source changed")
            metadata.append(chunk["cache"]["cache_ABI"][rank] if "cache" in chunk else
                            torch.load(record["path"], map_location="cpu", weights_only=True))
            receipts.append(dict(rank=rank, metadata_sha256=record["sha256"]))
        if "engram_lookup" not in following:
            raise ValueError("Each continuation requires fresh Engram lookup rows")
        self.weights.assert_static_unchanged()
        cache_uploads = self.runtime.stats["cache_h2d_bytes"]
        before = _network_owner_snapshot(self.provider)
        try:
            self.provider.begin_chunk(epoch=epoch, token_map_id=contract["input_contract_sha256"],
                input_by_case={"fresh": actual}, metadata_by_case={"fresh": metadata},
                metadata_sources_by_case={"fresh": receipts})
            after = _network_owner_snapshot(self.provider)
            if before != after:
                raise RuntimeError("Chunk admission replaced a persistent cache or history allocation")
            validate_host_lookup(following["engram_lookup"], self.engram_contracts[1], self.provider.hash_batch())
            self.program.compile(self.provider, contract["model_config"], contract["topology"], self.engram_contracts)
            output = Path(output)
            output.mkdir(parents=True, exist_ok=False)
            result = run_network_chunk(following, contract=contract, output=output, provider=self.provider,
                weights=self.weights, memory=self.memory, ownership=self.ownership, assigned=self.assigned,
                engram_contracts=self.engram_contracts, program=self.program, epoch=epoch, progress=progress,
                resident=self.runtime, phase=phase)
            self.weights.assert_static_unchanged()
            if cache_uploads != self.runtime.stats["cache_h2d_bytes"]:
                raise RuntimeError("A token step uploaded persistent cache bytes")
        except BaseException:
            self.failed = True
            raise
        self.epoch, self.contract = epoch, contract
        return result

    def summary(self):
        return dict(resident_runtime=self.runtime.summary(), resident_weights=self.weights.summary(),
                    provider_allocation_id=self.provider.allocation_id, epoch=self.epoch,
                    cache_aliases=self.weights.cache_alias_receipts, program=self.program.summary(),
                    closed=self.closed, failed=self.failed)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def run_network_runtime(inputs, *, contract, output, devices, checkpoint=None, continuation=None,
                        compiled=None, progress=None, keep_runtime=False, request_inputs=None):
    """Initialize one complete network session and retain it through both phases."""
    from .moe_transport import CANNMemory
    from .prefill_resident import ResidentNetworkRuntime
    from .resident_weights import A3ResidentWeights
    from .runtime_config import configure_devices

    if "engram_lookup" not in inputs:
        raise ValueError("Prepare Engram lookup rows before calling the network")
    topology = contract["topology"]
    assigned = configure_devices(devices, world=topology["ep"])
    output = Path(output).resolve()
    if checkpoint is not None:
        contract = dict(contract, checkpoint=str(Path(checkpoint).resolve()))
    output.mkdir(parents=True, exist_ok=False)
    if progress is not None:
        progress(dict(stage="compiling_network", epoch=0))
    provider = prepare_network_provider(inputs, contract, output.name, geometry_only=True)
    validate_host_lookup(inputs["engram_lookup"], inputs["engram_contracts"][1], provider.hash_batch())
    program = compiled if compiled is not None else NetworkProgram(output / "compiled")
    program.compile(provider, contract["model_config"], topology, inputs["engram_contracts"])
    provider = prepare_network_provider(inputs, contract, output.name)
    with ExitStack() as stack:
        resident = stack.enter_context(ResidentNetworkRuntime(assigned, topology))
        resident.register(program)
        weights = A3ResidentWeights(contract["checkpoint"], topology, contract["model_config"], resident)
        stack.callback(weights.close)
        weights.preload_once(progress)
        write_json(output / "weight-audit.json", dict(checkpoint_reads=weights.reader.consumed,
                                                      host_preparations=weights.reader.preparations))
        for family, values in (("raw", provider.arenas), ("shared", provider.shared)):
            for rank, cases in enumerate(values):
                for case, tensors in cases.items():
                    items = tensors.items() if isinstance(tensors, dict) else (("history", tensors),)
                    for name, value in items:
                        resident.pin(value, [rank], readonly=False, name=f"{family}/{rank}/{case}/{name}")
        session = GenerationSession(stack, resident=resident, weights=weights, provider=provider, program=program,
            contract=contract, engram_contracts=inputs["engram_contracts"], memory=CANNMemory(diagnostics=False),
            assigned=assigned, output=output, request_inputs=request_inputs)
        first = run_network_chunk(inputs, contract=contract, output=output, provider=provider, weights=weights,
            memory=session.memory, ownership=session.ownership, assigned=assigned,
            engram_contracts=session.engram_contracts, program=program, epoch=0, progress=progress, resident=resident)
        weights.assert_static_unchanged()
        completed = [first]
        chunks = (continuation,) if isinstance(continuation, Mapping) else continuation
        if chunks is not None:
            for chunk in chunks:
                completed.append(session.advance(chunk, output=output / f"continuation-chunk{session.epoch + 1}",
                                                 progress=progress))
        result = completed[-1]
        if len(completed) > 1:
            result = dict(result, completed_prefill_chunks=completed, complete_network_acceptance="not_run")
        if keep_runtime:
            session._stack = stack.pop_all()
            result = dict(result, session=session)
        return result


def run_network_chunk(
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
    phase="prefill",
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
        logical_rank_mapping=list(range(topology["ep"])),
        source_root=str(Path(__file__).resolve().parents[2]),
        source_files={},
        groups=dict(
            ep=[list(range(topology["ep"]))],
            tp=[list(range(dp * topology["tp"], dp * topology["tp"] + topology["tp"]))
                for dp in range(topology["dp"])],
            dp=[[dp * topology["tp"] + lane for dp in range(topology["dp"])]
                for lane in range(topology["tp"])],
        ),
        status="executing_not_compared",
        epoch=epoch,
        phase=phase,
        runtime_identity=id(resident),
        namespace=namespace,
        input_contract_sha256=contract["input_contract_sha256"],
        fresh_shared_initialization=contract["fresh_shared_initialization"],
        fresh_initialization_applied=epoch == 0,
    )
    loaded_source_files(runtime)
    write_json(runtime_path, runtime)
    reports = [dict(rank=rank, communication=[]) for rank in range(topology["ep"])]
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
                raise ValueError("Engram contract differs from actual captured hash constants")
        if (
            tuple(item["hash_constants"]["layer_ids"]) != (1, 14)
            or item["hash_constants"]["pad_id"] != captured_constants["pad_id"]
        ):
            raise ValueError("Engram contract differs from the original slot/pad layout")
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
    layer_started, layer_timings = {}, []

    def runtime_for_layer(layer):
        layer_started[layer] = perf_counter()
        if progress is not None:
            progress(dict(stage="executing_layer", epoch=epoch, layer=layer))
        actual = executor.for_layer(layer)
        actual["required_modes"] = contract["required_modes"]
        actual["engram"] = prepared_engram.apply
        return actual

    completed_layers = []

    def observe_layer(layer, result):
        for case, records in provider.batch_for_layer(layer)["cases"].items():
            for rank, record in enumerate(records):
                valid = record["SP_valid_rows"]
                if (not torch.isfinite(result["hidden"][rank][case][:valid]).all()
                        or not torch.isfinite(result["pre_mix"][rank][case][:valid]).all()):
                    raise RuntimeError(f"Non-finite {phase} state after layer {layer} on rank {rank}")
        provider.observe_layer(layer, result)
        completed_layers.append(layer)
        layer_timings.append(dict(layer=layer, weight_load_seconds=0.0,
                                  total_seconds=perf_counter() - layer_started[layer]))
        if progress is not None:
            progress(dict(stage="layer_complete", epoch=epoch, layer=layer))
        print(dict(stage="full40_layer_complete", layer=layer), flush=True)
        weights.release_current()

    try:
        result = run_network_inmemory(
            hidden,
            premix,
            batch_for_layer=provider.batch_for_layer,
            cache_for_layer=provider.cache_for_layer,
            weights_for_layer=weights.decode.layer if phase == "decode" else weights.for_layer,
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
        if any(not torch.isfinite(value).all() for row in output_result["logits"] for value in row.values()):
            raise RuntimeError(f"Non-finite {phase} vocabulary logits")
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
                outcome=f"executed_actual_full40_{phase}_pending_independent_comparison",
                reports=reports,
                compiled=executor.compiled,
                compilation=program.summary(),
                timings=dict(
                    layers=layer_timings,
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
        return dict(runtime, logits=output_result["logits"], compilation=program.summary(),
                    timings=dict(layers=layer_timings, weight_load_seconds=weights.load_seconds - initial_load_seconds,
                                 checkpoint_read_calls=weights.reader.read_calls - initial_read_calls))
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


def run_network_inmemory(
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



# Preserve the documented compilation API while both phases share its implementation.
PrefillProgram = NetworkProgram

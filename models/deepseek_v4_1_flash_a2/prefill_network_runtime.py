# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Wire original full40 Prefill providers, live state, actual HCCL and the layer39 output boundary."""

from collections.abc import Mapping
from contextlib import ExitStack
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
from time import perf_counter

import torch
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig

from .prefill_layer import run_prefill_network_inmemory
from .prefill_embedding import make_input_entries
from .prefill_layer_state import resolve_layer_role
from .prefill_network_cache import attach_network_cache_geometry
from .prefill_network_engram import prepare_network_engram
from .prefill_network_discrete_observer import (
    begin_product_observation, end_product_observation, product_engram_observation,
)
from .prefill_network_guard import NetworkStatePreservation
from .prefill_network_state import NetworkCacheProvider
from .prefill_network_input import run_input_boundary
from .prefill_output import make_output_entries
from .prefill_program import PrefillProgram, make_network_layer_entries
from .prefill_swa import prepare_product_operands
from .runtime_config import ExpertParallelConfig
from .runtime_utils import communicate, file_sha, loaded_source_files, tensor_sha, write_json
from .weights import CheckpointOperands


class NetworkWeights:
    """Retain one actual layer's non-expert host weights and load only real owned experts on demand."""

    def __init__(self, checkpoint, topology, model_config):
        self.reader = CheckpointOperands(checkpoint)
        self.layout = ExpertParallelConfig(topology["dp"], topology["tp"], topology["ep"], 384)
        self.model_config = model_config
        self.current_layer, self.current = None, None
        self.layer_receipts = {}
        self.load_seconds, self.load_calls = 0.0, 0

    def for_layer(self, layer_id):
        role = resolve_layer_role(layer_id)
        if self.current_layer != layer_id:
            begin = perf_counter()
            self.release_current()
            prefix = f"layers.{layer_id}."
            selected = [name for name in self.reader.index if name.startswith(prefix)
                        and ".ffn.experts." not in name and ".engram.embed." not in name]
            if not selected:
                raise ValueError("Missing original physical-layer checkpoint operands")
            raw = {name: self.reader.load(name) for name in selected}
            operands = prepare_product_operands(raw, layer_id, role=role)
            self.current_layer = layer_id
            self.current = dict(raw=raw, operands=operands, expert=self.expert)
            self.layer_receipts[layer_id] = dict(
                original_names=selected, loaded_non_expert_bytes=sum(value.numel() * value.element_size() for value in raw.values()),
                prepared_bytes=sum(value.numel() * value.element_size() for value in operands.values()),
                policy="One current layer on host; real expert payloads loaded for owned active experts",
                HBM_residency="Not established by host byte counts; measure actual execution",
            )
            self.load_seconds += perf_counter() - begin
            self.load_calls += 1
        return self.current

    def raw_for_layer(self, layer_id):
        return self.for_layer(layer_id)["raw"]

    def expert(self, rank, expert):
        if self.current_layer is None or not 0 <= rank < self.layout.ep:
            raise ValueError("An actual current layer and allocated expert rank are required")
        if expert is not None and self.layout.owner(expert) != rank:
            raise ValueError("Routed expert request does not belong to this real EP rank")
        prefix = f"layers.{self.current_layer}.ffn."
        prefix += "shared_experts" if expert is None else f"experts.{expert}"
        begin = perf_counter()
        result = self.reader.expert(prefix)
        self.load_seconds += perf_counter() - begin
        self.load_calls += 1
        return result

    def release_current(self):
        self.current = None
        self.current_layer = None
        gc.collect()


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
        if (names != entry.param_names or pure_outputs != entry.output_indices
                or bool(metadata["num_return_types"]) != entry.has_return
                or metadata["platform"] != entry.platform or metadata["backend_type"] != entry.backend_type.name):
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
                raise ValueError("Lowered output readback lacks an exact tensor return alias: " + logical["name"])
            effective, source = after, "same-slot-output-readback-and-validated-return-alias"
        elif before == after:
            effective, source = before, "explicit-logical-direction"
        else:
            raise ValueError("Unsupported logical/lowered JIT direction change: " + logical["name"])
        directions.append(effective)
        if effective != "In":
            proofs.append(dict(index=index, name=logical["name"], logical_direction=before,
                               lowered_direction=after, effective_direction=effective,
                               return_positions=alias_positions, source=source))
    live_checked = compiled.program is not None
    if live_checked:
        original, original_outputs, returns = extract_param_infos(compiled.program)
        records = [dict(name=item.name, direction=item.direction.name,
                        shape=None if item.shape is None else list(item.shape),
                        dtype=str(item.dtype), layout=item.layout) for item in original]
        if records != params or original_outputs != pure_outputs or len(returns) != metadata["num_return_types"]:
            raise ValueError("Saved JIT ABI differs from the original live Program")
        functions = list(compiled.program.functions.values())
        entries = [item for item in functions if item.func_type == ir.FunctionType.Orchestration]
        if len(entries) != 1:
            if not entries and len(functions) == 1:
                entries = functions
            else:
                raise ValueError("Network JIT requires one unambiguous live Program entry")
        traced = ir.returned_param_indices(entries[0], compiled.program)
        known = (len(traced) == len(returns) and all(type(index) is int and 0 <= index < len(params)
                 and params[index]["shape"] is not None for index in traced))
        live_aliases = list(traced) if known else None
        if live_aliases != aliases:
            raise ValueError("Saved return aliases differ from the original live Program")
    if file_sha(path) != digest:
        raise ValueError("Compiled parameter metadata changed during ABI validation")
    return dict(path=str(path), sha256=digest, schema=3, params=params, lowered_params=lowered,
                return_aliases=aliases, num_return_types=metadata["num_return_types"],
                effective_directions=directions, writable_proofs=proofs, live_program_aliases_checked=live_checked,
                parameter_mapping="same ordered name/shape/dtype/layout slots",
                validation="public CompiledProgram.from_dir and CompiledArtifact.from_dir; no runtime load")


def network_argument_protection(arguments, outputs, abi):
    """Protect true In tensors; audit-output names and selection grant no write permission."""
    params, directions = abi["params"], abi["effective_directions"]
    if len(arguments) != len(params) or len(directions) != len(params):
        raise ValueError("Require all actual JIT parameters including Out/InOut")

    def extent(value):
        if value.layout != torch.strided or any(stride < 0 for stride in value.stride()):
            raise ValueError("Unsupported tensor layout for exact readonly alias protection")
        if not value.numel():
            return None
        first = value.data_ptr()
        span = (sum((size - 1) * stride for size, stride in zip(value.shape, value.stride())) + 1)
        return str(value.device), first, first + span * value.element_size()

    def overlap(left, right):
        return (left is not None and right is not None and left[0] == right[0]
                and max(left[1], right[1]) < min(left[2], right[2]))

    readonly, writable = [], []
    for index, (param, value, direction) in enumerate(zip(params, arguments, directions)):
        if param["direction"] not in {"In", "Out", "InOut"} or direction not in {"In", "Out", "InOut"}:
            raise ValueError("Unknown JIT parameter direction")
        if param["shape"] is None:
            if isinstance(value, torch.Tensor) or direction != "In":
                raise ValueError("Unexpected scalar argument or writable scalar ABI")
            continue
        if not isinstance(value, torch.Tensor):
            raise ValueError("The network wrapper requires actual host tensor arguments")
        record = dict(index=index, name=param["name"], direction=direction,
                      logical_direction=param["direction"], lowered_direction=abi["lowered_params"][index]["direction"],
                      extent=extent(value))
        (readonly if direction == "In" else writable).append(record)
    for source in readonly:
        for target in writable:
            if overlap(source["extent"], target["extent"]):
                raise ValueError("Readonly/writable JIT parameters overlap: " + source["name"] + "/" + target["name"])
    for name, value in outputs.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError("Network audit outputs must be tensors")
        output_extent = extent(value)
        if output_extent is not None and not any(
                target["extent"] is not None and output_extent[0] == target["extent"][0]
                and target["extent"][1] <= output_extent[1] <= output_extent[2] <= target["extent"][2]
                for target in writable):
            raise ValueError("Audit output does not belong to a writable JIT parameter: " + name)
    return [(item["index"], tensor_sha(arguments[item["index"]])) for item in readonly], writable

class NetworkExecutor:
    """Compile exact existing per-layer entries and record invocations without retaining all tensor captures."""

    def __init__(self, output, config, topology, provider, ownership, memory, runtime_path, reports,
                 program, namespace="chunk0", diagnostics=True):
        self.output, self.config, self.topology = Path(output), config, topology
        self.provider, self.ownership, self.memory = provider, ownership, memory
        self.runtime_path, self.reports = runtime_path, reports
        self.namespace = namespace
        self.diagnostics = diagnostics
        self.program = program
        self.current_layer, self.current_entries = None, None
        self.compiled = {}
        self.entry_abis = {}

    def exchange(self, stage, values, collective, group, *, packets=False):
        if not stage or "/" in stage or ".." in stage:
            raise ValueError("HCCL stages require distinct safe chunk/layer identities")
        transport = values if packets else [{case: {"value": value} for case, value in row.items()} for row in values]
        result, receipts = communicate(stage, transport, self.output, self.runtime_path, collective=collective, group=group)
        for rank, receipt in enumerate(receipts):
            self.reports[rank]["communication"].append(receipt)
        if packets:
            return result
        return [{case: packet["value"] for case, packet in row.items()} for row in result]

    def precheck(self):
        world = self.topology["ep"]
        values = [{"precheck": torch.tensor([[rank]], dtype=torch.int32)} for rank in range(world)]
        for group in ("ep", "tp", "dp"):
            observed = self.exchange(self.namespace + "-network-startup-" + group, values, "all_gather", group)
            for rank in range(world):
                tp = self.topology["tp"]
                expected = list(range(world)) if group == "ep" else list(range(rank // tp * tp, rank // tp * tp + tp))
                if group == "dp":
                    expected = list(range(rank % tp, world, tp))
                if observed[rank]["precheck"].flatten().tolist() != expected:
                    raise RuntimeError("Actual network HCCL group precheck failed: " + group)

    def for_layer(self, layer_id):
        definitions, geometry, expert_capacity = make_network_layer_entries(
            self.provider, self.config, self.topology, layer_id,
        )
        batch = self.provider.batch_for_layer(layer_id)
        if self.current_layer != layer_id:
            self.current_entries = None
            directory = self.output / f"{self.namespace}-layer{layer_id}"
            directory.mkdir(exist_ok=False)
            self.current_entries = self.program.bind(f"layer{layer_id}", definitions)
            self.entry_abis = {name: network_parameter_abi(entry.compiled)
                               for name, entry in self.current_entries.items()}
            for name, entry in self.current_entries.items():
                abi = self.entry_abis[name]
                entry.artifacts[abi["path"]] = abi["sha256"]
            self.compiled[layer_id] = {name: entry.artifacts for name, entry in self.current_entries.items()}
            self.current_layer = layer_id

        def execute(name, rank, case, arguments, outputs, responsibility=None):
            entry = self.current_entries[name]
            config = RunConfig(arch=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD, device_id=rank)
            if not self.diagnostics:
                entry.compiled(*arguments, config=config)
                return
            abi = self.entry_abis[name]
            readonly, writable = network_argument_protection(arguments, outputs, abi)
            roles = name if responsibility is None else responsibility
            config = RunConfig(arch=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD, device_id=rank)
            self.memory.sample(rank, f"L{layer_id}/{name}/before")
            with self.memory.interval(rank, "execute", f"L{layer_id}/{name}"):
                entry.compiled(*arguments, config=config)
            self.memory.sample(rank, f"L{layer_id}/{name}/after")
            if any(tensor_sha(arguments[index]) != digest for index, digest in readonly):
                raise RuntimeError("Network entry modified a readonly argument: " + name)
            self.reports[rank]["kernels"].append(dict(
                layer=layer_id, name=name, case=case, responsibility=roles, artifacts=entry.artifacts,
                argument_shapes=[list(value.shape) if isinstance(value, torch.Tensor) else value for value in arguments],
                jit_parameter_abi=dict(path=abi["path"], sha256=abi["sha256"], schema=abi["schema"],
                                       directions=[item["direction"] for item in abi["params"]],
                                       lowered_directions=[item["direction"] for item in abi["lowered_params"]],
                                       effective_directions=abi["effective_directions"],
                                       return_aliases=abi["return_aliases"], num_return_types=abi["num_return_types"],
                                       writable_arguments=[item["index"] for item in writable],
                                       writable_proofs=abi["writable_proofs"],
                                       live_program_aliases_checked=abi["live_program_aliases_checked"]),
                readonly_exact=True, readonly_arguments=len(readonly), status="executed_not_independently_compared",
            ))

        return dict(
            topology=self.topology, ownership=self.ownership, enable_expert_parallel=True,
            capacity=geometry["capacity"], expert_capacity=expert_capacity, cache_blocks=geometry["cache_blocks"],
            requests=geometry["requests"], table_pages=geometry["table_pages"], memory=self.memory,
            execute=execute, execute_l2=execute, exchange=self.exchange, epoch=batch["epoch"], token_map_id=batch["token_map_id"],
            output_sentinel=-37.0, stage_namespace=f"{self.namespace}-layer{layer_id}", capture_nodes=False, capture_state=False,
        )


def load_product_backbone_input(paths, *, input_contract, provider, capacity, expected_files):
    """Load only admitted actual product input-component outputs, then preserve their original local SP mapping."""
    world = input_contract["topology"]["ep"]
    if len(paths) != world or set(expected_files) != {str(path) for path in paths}:
        raise ValueError("Every original rank needs an exact admitted product input output")
    hidden, premix = [], []
    for rank, path in enumerate(paths):
        if file_sha(path) != expected_files[str(path)]:
            raise ValueError("An admitted product backbone input changed")
        original = torch.load(path, map_location="cpu", weights_only=True)
        record = provider.batch_for_layer(0)["cases"]["fresh"][rank]
        physical, valid = record["SP_physical_rows"], record["SP_valid_rows"]
        if original["SP_physical_rows"] != physical or original["SP_valid_rows"] != valid:
            raise ValueError("Input component and actual network metadata disagree on CP/SP rows")
        if original["DP_rank"] != rank // input_contract["topology"]["tp"]:
            raise ValueError("Actual product input belongs to another DP request engine")
        actual_hidden, actual_pre = original["hidden"], original["pre_mix"]
        if (actual_hidden.dtype != torch.bfloat16 or tuple(actual_hidden.shape) != (physical, 4, 5120)
                or actual_pre.dtype != torch.float32 or tuple(actual_pre.shape) != (physical, 4)):
            raise ValueError("Actual initial HC product boundary has a different tensor ABI")
        padded_hidden = torch.zeros((capacity, 4, 5120), dtype=torch.bfloat16)
        padded_pre = torch.zeros((capacity, 4), dtype=torch.float32)
        padded_hidden[:physical].copy_(actual_hidden)
        padded_pre[:physical].copy_(actual_pre)
        hidden.append(dict(fresh=padded_hidden))
        premix.append(dict(fresh=padded_pre))
    return hidden, premix


def embed_network_input(input_contract, weights, executor, config, output):
    """Run the shared vocabulary lookup and initial HC directly into the backbone."""
    directory = Path(output)
    directory.mkdir(exist_ok=False)
    entries = executor.program.bind("embedding", make_input_entries(executor.provider.capacity, config,
                                                                   executor.topology["tp"]))

    def execute(name, rank, arguments, outputs):
        readonly = [(index, tensor_sha(value)) for index, value in enumerate(arguments)
                    if isinstance(value, torch.Tensor) and all(value is not out for out in outputs.values())]
        capture = directory / f"rank-{rank}-{name}.pt"
        entries[name].execute(arguments, outputs, rank, "network_input/" + name,
                              capture, executor.reports[rank], executor.memory)
        if any(tensor_sha(arguments[index]) != digest for index, digest in readonly):
            raise RuntimeError("Embedding changed a readonly token or weight operand")

    def exchange(stage, packets):
        return executor.exchange(executor.namespace + "-" + stage, packets, "all_gather", "tp", packets=True)

    begin = perf_counter()
    boundaries = run_input_boundary(input_contract, config=config, capacity=executor.provider.capacity,
                                    reader=weights.reader, execute=execute, exchange=exchange)
    write_json(directory / "execution.json", dict(status="executed_not_independently_compared",
               seconds=perf_counter() - begin, ranks=len(boundaries), layers_executed=[]))
    hidden, premix = [], []
    for rank, original in enumerate(boundaries):
        record = executor.provider.batch_for_layer(0)["cases"]["fresh"][rank]
        if any(original[name] != record[name] for name in ("SP_physical_rows", "SP_valid_rows")):
            raise ValueError("Embedding and cache metadata disagree on rank-local token ownership")
        local_hidden = torch.zeros((executor.provider.capacity, config["hc_mult"], config["hidden_size"]),
                                   dtype=torch.bfloat16)
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
    entries = executor.program.bind("output", make_output_entries(capacity, config, tp))
    norm = weights.reader.load("norm.weight")
    head = weights.reader.load("head.weight")
    if norm.dtype != torch.bfloat16 or head.dtype != torch.bfloat16 or tuple(head.shape) != (config["vocab_size"], 5120):
        raise ValueError("Actual complete output norm/head checkpoint ABI differs")
    output_report = dict(kernels=[])

    def execute(name, rank, case, arguments, outputs):
        if not executor.diagnostics:
            config = RunConfig(arch=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD, device_id=rank)
            entries[name].compiled(*arguments, config=config)
            return
        mutable = {value.untyped_storage().data_ptr() for value in outputs.values()}
        before = [(index, tensor_sha(value)) for index, value in enumerate(arguments)
                  if isinstance(value, torch.Tensor) and value.untyped_storage().data_ptr() not in mutable]
        capture = directory / f"rank-{rank}-{case}-{name}.pt"
        entries[name].execute(arguments, outputs, rank, "network_output/" + name, capture, output_report, executor.memory)
        if any(tensor_sha(arguments[index]) != digest for index, digest in before):
            raise RuntimeError("Output boundary changed an actual readonly layer39 tensor or weight")

    packets = [{} for _ in range(world)]
    for case, records in batch["cases"].items():
        for rank, record in enumerate(records):
            physical = record["SP_physical_rows"]
            collapsed = torch.full((capacity, 5120), -17, dtype=torch.bfloat16)
            execute("collapse", rank, case, [result["hidden"][rank][case], result["pre_mix"][rank][case], collapsed, physical],
                    dict(local_collapsed=collapsed))
            packets[rank][case] = collapsed
    gathered = executor.exchange(executor.namespace + "-output-SP-gather", packets, "all_gather", "tp")
    logits_packets = [{} for _ in range(world)]
    for case, records in batch["cases"].items():
        for rank, record in enumerate(records):
            physical, active = record["SP_physical_rows"], len(record["positions"])
            collapsed = torch.full((capacity, 5120), -17, dtype=torch.bfloat16)
            normalized = torch.full((capacity, 5120), -19, dtype=torch.bfloat16)
            local_logits = torch.full((capacity, config["vocab_size"] // tp), -23, dtype=torch.bfloat16)
            execute("sp_unpack", rank, case, [gathered[rank][case], collapsed, max(physical, 1), active], dict(collapsed=collapsed))
            execute("norm", rank, case, [collapsed, norm, normalized, active], dict(normalized=normalized))
            count = config["vocab_size"] // tp
            weight = head.narrow(0, rank % tp * count, count).T.contiguous()
            if tuple(weight.shape) != (5120, count):
                raise ValueError("Output head must use canonical KN vocabulary shard layout")
            execute("projection", rank, case, [normalized, weight, local_logits, active], dict(local_logits=local_logits))
            logits_packets[rank][case] = local_logits
    gathered_logits = executor.exchange(executor.namespace + "-output-vocabulary-gather", logits_packets, "all_gather", "tp")
    files, actual_logits = {}, [{} for _ in range(world)]
    for case, records in batch["cases"].items():
        for rank, record in enumerate(records):
            logits = torch.full((capacity, config["vocab_size"]), -29, dtype=torch.bfloat16)
            execute("vocab_unpack", rank, case, [gathered_logits[rank][case], logits, len(record["positions"])], dict(logits=logits))
            path = directory / f"rank-{rank}-{case}-network-logits.pt"
            torch.save(dict(logits=logits, active=len(record["positions"]),
                            positions=record["positions"], DP_rank=rank // tp), path)
            files[str(path)] = file_sha(path)
            actual_logits[rank][case] = logits[:len(record["positions"])].clone()
    write_json(directory / "execution.json", dict(
        outcome="executed_actual_layer39_output_boundary_not_compared", files=files,
        kernels=output_report["kernels"], native_reference="not_run", full40_accuracy="not_run",
    ))
    return dict(files=files, logits=actual_logits)


def _network_owner_snapshot(provider, storage):
    """Bind each persistent object's identity and exact own bytes across chunk admission."""
    result = {"provider": id(provider), "history": id(provider.history), "arrays": {}, "tables": {}}
    for family, values in (("raw", provider.arenas), ("shared", provider.shared), ("history", provider.history.caches)):
        for rank, cases in enumerate(values):
            for case, tensors in cases.items():
                items = tensors.items() if isinstance(tensors, dict) else (("history", tensors),)
                for name, value in items:
                    key = f"{family}/{rank}/{case}/{name}"
                    result["arrays"][key] = dict(object_id=id(value), storage=value.untyped_storage().data_ptr(),
                                                sha256=tensor_sha(value))
    for key, record in storage.items():
        value = record["array"]
        if value.mapping is None or value.live:
            raise ValueError("Retained Engram table must be open and have no live rank registration")
        result["tables"][str(key)] = dict(object_id=id(value), receipt=dict(value.receipt), host_pointer=value.host_pointer)
    return result


def prepare_network_provider(inputs, contract, allocation_id, *, geometry_only=False):
    """Allocate canonical cache owners and validate the complete original network ABI."""
    topology = contract["topology"]
    if topology != dict(dp=4, tp=4, ep=16, pp=1, pcp=1, dcp=1):
        raise ValueError("The current network runtime requires DP4/TP4/EP16/PP1/PCP1/DCP1")
    if not all(contract["required_modes"].values()) or contract["capacity"] != 16 or contract["scheduler_capacity"] != 8192:
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
        {"fresh": inputs["cache_ABI"]}, {"fresh": inputs["input_contract"]}, topology=topology,
        model_config=model_config, capacity=16, scheduler_capacity=8192,
        allocation_id=allocation_id, sources_by_case={"fresh": inputs["cache_ABI_sources"]},
        geometry_only=geometry_only,
    )
    attach_network_cache_geometry(provider)
    provider.require_network_geometry()
    for cases in provider.shared:
        cases["fresh"]["topk_indices"].fill_(-1)
    provider.begin_chunk(epoch=0, token_map_id=contract["input_contract_sha256"])
    return provider


def run_network_runtime(inputs, *, contract, output, devices, checkpoint=None, continuation=None, compiled=None, diagnostics=True):
    """Execute a fresh chunk and a lazy stream of continuations with shared live state.

    ``continuation`` is one chunk mapping or an iterable of chunk mappings.
    Each item supplies inputs, its contract and every rank's metadata sources.
    The iterable advances only after the preceding complete chunk returns.
    """
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
    provider = prepare_network_provider(inputs, contract, output.name, geometry_only=True)
    program = compiled if compiled is not None else PrefillProgram(output / "compiled")
    if compiled is not None and not compiled.ready:
        raise ValueError("The supplied Prefill program is incomplete")
    program.compile(provider, contract["model_config"], topology, inputs["engram_contracts"])
    provider = prepare_network_provider(inputs, contract, output.name)
    with ExitStack() as stack:
        weights = NetworkWeights(contract["checkpoint"], topology, contract["model_config"])
        stack.callback(weights.release_current)
        memory = CANNMemory(diagnostics=diagnostics)
        ownership = [weights.layout.ownership(rank) for rank in range(16)]
        table_storage = {} if continuation is not None else None
        table_stack = stack if continuation is not None else None
        first = _run_network_chunk(inputs, contract=contract, output=output, provider=provider,
            weights=weights, memory=memory, ownership=ownership, assigned=assigned,
            engram_contracts=inputs["engram_contracts"], program=program, epoch=0, table_storage=table_storage,
            table_stack=table_stack, diagnostics=diagnostics)
        if continuation is None:
            return first
        completed = [first]
        previous_contract = contract
        fixed = ("topology", "checkpoint", "model_config", "required_modes", "capacity", "scheduler_capacity")
        for epoch, chunk in enumerate(following_chunks, start=1):
            if not isinstance(chunk, Mapping) or not {"inputs", "contract", "metadata_sources"} <= chunk.keys():
                raise ValueError("Each continuation needs inputs, contract and metadata_sources")
            following, next_contract = chunk["inputs"], chunk["contract"]
            required = set(fixed) | {"input_contract_path", "input_contract_sha256"}
            if (not isinstance(following, Mapping) or "input_contract" not in following
                    or not isinstance(next_contract, Mapping) or not required <= next_contract.keys()):
                raise ValueError("Continuation needs prepared inputs and a complete network contract")
            if checkpoint is not None:
                next_contract = dict(next_contract, checkpoint=contract["checkpoint"])
            if any(next_contract[name] != contract[name] for name in fixed):
                raise ValueError("Continuation cannot change numerical/model/topology configuration")
            input_path = Path(next_contract["input_contract_path"])
            actual_input = following["input_contract"]
            if not isinstance(actual_input, Mapping) or not {"epoch", "predecessor_input_sha256"} <= actual_input.keys():
                raise ValueError("Continuation input must identify its epoch and predecessor")
            if (file_sha(input_path) != next_contract["input_contract_sha256"]
                    or json.loads(input_path.read_text()) != actual_input
                    or actual_input["predecessor_input_sha256"] != previous_contract["input_contract_sha256"]
                    or type(actual_input["epoch"]) is not int or actual_input["epoch"] != epoch):
                raise ValueError("Continuation needs the next epoch and exact preceding input identity")
            sources = chunk["metadata_sources"]
            if not isinstance(sources, (list, tuple)) or len(sources) != topology["ep"]:
                raise ValueError("Continuation needs real original-builder metadata for every rank")
            metadata, receipts = [], []
            for rank, record in enumerate(sources):
                if not isinstance(record, Mapping) or not {"rank", "path", "sha256"} <= record.keys():
                    raise ValueError("Each continuation metadata source needs rank, path and sha256")
                if (type(record["rank"]) is not int or record["rank"] != rank
                        or file_sha(record["path"]) != record["sha256"]):
                    raise ValueError("Actual continuation metadata source changed")
                metadata.append(torch.load(record["path"], map_location="cpu", weights_only=True))
                receipts.append(dict(rank=rank, metadata_sha256=record["sha256"]))
            following = dict(following, cache_ABI=metadata)
            before = _network_owner_snapshot(provider, table_storage)
            provider.begin_chunk(epoch=epoch, token_map_id=next_contract["input_contract_sha256"],
                input_by_case={"fresh": actual_input}, metadata_by_case={"fresh": metadata},
                metadata_sources_by_case={"fresh": receipts})
            after = _network_owner_snapshot(provider, table_storage)
            if before != after:
                raise RuntimeError("Chunk admission changed persistent raw/shared/history/table owners or bytes")
            program.compile(provider, next_contract["model_config"], topology, inputs["engram_contracts"])
            chunk_output = output / f"continuation-chunk{epoch}"
            chunk_output.mkdir(exist_ok=False)
            write_json(chunk_output / "own-state-continuity.json", dict(before=before, after=after,
                exact=True, previous_complete_output=True,
                scope="Own previous-final to current-initial; no cross-framework float byte gate"))
            result = _run_network_chunk(following, contract=next_contract, output=chunk_output,
                provider=provider, weights=weights, memory=memory, ownership=ownership, assigned=assigned,
                engram_contracts=inputs["engram_contracts"], program=program, epoch=epoch, table_storage=table_storage, table_stack=table_stack, diagnostics=diagnostics)
            completed.append(result)
            previous_contract = next_contract
        if len(completed) == 1:
            return first
        return dict(completed[-1], completed_prefill_chunks=completed, complete_network_acceptance="not_run")


def _run_network_chunk(inputs, *, contract, output, provider, weights, memory, ownership,
                       assigned, engram_contracts, program, epoch, table_storage, table_stack, diagnostics=True):
    """Execute embedding, the complete backbone and the final head with persistent owners."""
    begin = perf_counter()
    initial_load_seconds, initial_load_calls = weights.load_seconds, weights.load_calls
    initial_read_seconds, initial_read_calls = weights.reader.read_seconds, weights.reader.read_calls
    topology, model_config = contract["topology"], contract["model_config"]
    text = dict(model_config["text_config"], image_sentinel_base_id=model_config["image_token_id"])
    namespace = f"chunk{epoch}"
    state_guard = NetworkStatePreservation(
        provider, contract["state_regions"]["path"], contract["state_regions"]["sha256"],
        contract["input_contract_sha256"],
    ) if diagnostics else None
    runtime_path = output / "runtime_config.json"
    runtime = dict(
        topology=topology, assigned_devices=assigned, logical_rank_mapping=list(range(16)),
        source_root=str(Path(__file__).resolve().parents[2]), source_files={},
        groups=dict(ep=[list(range(16))], tp=[list(range(dp * 4, dp * 4 + 4)) for dp in range(4)],
                    dp=[[dp * 4 + lane for dp in range(4)] for lane in range(4)]),
        status="executing_not_compared", epoch=epoch, namespace=namespace,
        input_contract_sha256=contract["input_contract_sha256"],
        state_regions_sha256=contract["state_regions"]["sha256"], diagnostics=diagnostics,
        optional_intermediate_numerical_gate=False if not diagnostics else "external_collector",
        optional_tensor_hashes=diagnostics, optional_state_snapshots=diagnostics,
        optional_boundary_captures=diagnostics, detailed_memory_sampling=diagnostics,
        fresh_shared_initialization=contract["fresh_shared_initialization"],
        fresh_initialization_applied=epoch == 0,
    )
    loaded_source_files(runtime)
    write_json(runtime_path, runtime)
    if diagnostics:
        for rank in range(16):
            torch.save(dict(raw=provider.arenas[rank], shared=provider.shared[rank],
                            hash_history=provider.history.caches[rank], layout=provider.layouts[rank],
                            metadata=inputs["cache_ABI"][rank]["metadata"], groups=inputs["cache_ABI"][rank]["groups"]),
                       output / f"rank-{rank}-initial-state.pt")
    reports = [dict(rank=rank, kernels=[], communication=[]) for rank in range(16)]
    executor = NetworkExecutor(output, text, topology, provider, ownership, memory, runtime_path, reports,
                               program, namespace, diagnostics=diagnostics)
    executor.precheck()
    if "product_input_paths" in inputs:
        hidden, premix = load_product_backbone_input(
            inputs["product_input_paths"], input_contract=inputs["input_contract"], provider=provider,
            capacity=16, expected_files=inputs["product_input_files"],
        )
    else:
        hidden, premix = embed_network_input(inputs["input_contract"], weights, executor, text, output / "embedding")
    hash_batch = provider.hash_batch()
    common_engram = engram_contracts[1]
    captured_constants = provider.hash_constants(layer_ids=[1, 14], pad_id=text["engram_pad_token_id"])
    for item in engram_contracts.values():
        for name in ("token_map", "multipliers", "primes", "offsets"):
            if not torch.equal(item["hash_constants"][name], captured_constants[name]):
                raise ValueError("TP4 Engram contract differs from actual captured hash constants")
        if tuple(item["hash_constants"]["layer_ids"]) != (1, 14) or item["hash_constants"]["pad_id"] != captured_constants["pad_id"]:
            raise ValueError("TP4 Engram contract differs from the original slot/pad layout")
    def observe_engram(layer, rank, case, observation):
        directory = output / "engram-discrete"
        directory.mkdir(exist_ok=True)
        path = directory / f"rank-{rank}-{case}-layer{layer}-discrete.pt"
        if path.exists():
            raise ValueError("Repeated product Engram boundary observation")
        torch.save(product_engram_observation(layer, rank, case, observation), path)

    eng_runtime = dict(topology=topology, assigned_devices=assigned, capacity=16,
                       stage_namespace=namespace + "-engram", output=output / "engram", reports=reports,
                       memory=memory, exchange=executor.exchange, capture_nodes=False, capture_state=diagnostics,
                       diagnostics=diagnostics, observe_discrete=observe_engram if diagnostics else None,
                       compile_entries=program.bind)
    prepared_engram = prepare_network_engram(
        common_engram, hash_batch, provider.history, engram_contracts, weights.raw_for_layer, eng_runtime,
        storage=table_storage, storage_stack=table_stack, storage_namespace=provider.allocation_id,
    )
    for layer in (1, 14):
        prepared_engram.attach(layer, provider.state_handles_for_layer(layer))

    discrete_pending = {}

    def runtime_for_layer(layer):
        actual = executor.for_layer(layer)
        if diagnostics:
            state_guard.begin_layer(layer)
            if layer in discrete_pending:
                raise ValueError("Repeated product layer observation begin")
            discrete_pending[layer] = begin_product_observation(provider, layer)
        actual["required_modes"] = contract["required_modes"]
        actual["engram"] = prepared_engram.apply
        return actual

    completed_layers = []

    def observe_layer(layer, result):
        if diagnostics:
            state_guard.finish_layer(layer, output / f"{namespace}-layer{layer}" / "state-preservation.json")
        provider.observe_layer(layer, result)
        completed_layers.append(layer)
        directory = output / f"{namespace}-layer{layer}"
        if not diagnostics:
            print(dict(stage="full40_layer_complete", layer=layer, diagnostics=False), flush=True)
            write_json(directory / "terminal.json", dict(layer=layer, diagnostics=False, status="executed_not_compared"))
            if layer == 39:
                for rank in range(16):
                    torch.save(dict(hidden=result["hidden"][rank], pre_mix=result["pre_mix"][rank]),
                               output / f"rank-{rank}-actual-layer39-boundary.pt")
            weights.release_current()
            return
        discrete = end_product_observation(provider, layer, result, discrete_pending.pop(layer))
        for rank in range(16):
            memory.sample(rank, f"L{layer}/complete")
        write_json(directory / "terminal.json", dict(
            layer=layer, status="executed_not_independently_compared", actual_state_owners=provider.versions,
            actual_local_shapes=[{case: list(value.shape) for case, value in row.items()} for row in result["hidden"]],
            current_host_weight_accounting=weights.layer_receipts.get(layer), next_layer_input="actual_local_product_output",
        ))
        for rank in range(16):
            boundaries = {}
            for case, value in result["hidden"][rank].items():
                record = provider.batch_for_layer(layer)["cases"][case][rank]
                physical = record["SP_physical_rows"]
                boundaries[case] = dict(
                    layer=layer, hidden=value[:physical].clone(),
                    pre_mix=result["pre_mix"][rank][case][:physical].clone(),
                    padded_hidden=value, padded_pre_mix=result["pre_mix"][rank][case],
                    SP_physical_rows=physical, SP_valid_rows=record["SP_valid_rows"],
                    cp_token_range=record["cp_token_range"],
                )
            torch.save(boundaries, directory / f"rank-{rank}-boundary.pt")
            torch.save(discrete[rank], directory / f"rank-{rank}-discrete.pt")
        if layer == 39:
            for rank in range(16):
                torch.save(dict(hidden=result["hidden"][rank], pre_mix=result["pre_mix"][rank]),
                           output / f"rank-{rank}-actual-layer39-boundary.pt")
        weights.release_current()

    try:
        result = run_prefill_network_inmemory(
            hidden, premix, batch_for_layer=provider.batch_for_layer, cache_for_layer=provider.cache_for_layer,
            weights_for_layer=weights.for_layer, runtime_for_layer=runtime_for_layer, observe_layer=observe_layer,
        )
        output_result = run_output_boundary(
            result, batch=provider.batch_for_layer(39), topology=topology, config=text,
            weights=weights, executor=executor, output=output / "actual-layer39-output",
        )
        if diagnostics:
            for rank in range(16):
                torch.save(dict(raw=provider.arenas[rank], shared=provider.shared[rank], versions=provider.versions[rank],
                                hash_history=provider.history.caches[rank]), output / f"rank-{rank}-final-state.pt")
        loaded_source_files(runtime)
        runtime.update(status="executed_full40_not_compared", completed_layers=list(range(40)),
                       actual_Layer39_boundary=True, final_output_files=output_result["files"], formal_accuracy="not_run")
        write_json(runtime_path, runtime)
        write_json(output / "execution.json", dict(
            at=datetime.now(timezone.utc).isoformat(), outcome="executed_actual_full40_Prefill_pending_independent_comparison",
            reports=reports, compiled=executor.compiled, compilation=program.summary(),
            timings=dict(chunk_seconds=perf_counter() - begin,
                         weight_load_seconds=weights.load_seconds - initial_load_seconds,
                         weight_load_calls=weights.load_calls - initial_load_calls,
                         checkpoint_read_seconds=weights.reader.read_seconds - initial_read_seconds,
                         checkpoint_read_calls=weights.reader.read_calls - initial_read_calls),
            consumed_checkpoint=weights.reader.consumed,
            layer_host_weight_accounting=weights.layer_receipts, memory_samples=memory.samples,
            memory_intervals=memory.intervals, capacity_verified_by_execution=False,
            formal_accuracy="not_run", complete_network_acceptance="not_run",
        ))
        return dict(runtime, logits=output_result["logits"], compilation=program.summary())
    except BaseException as error:
        write_json(output / "failure.json", dict(
            at=datetime.now(timezone.utc).isoformat(), error_type=type(error).__name__, error=str(error),
            last_attempted_layer=executor.current_layer, completed_layers=completed_layers,
            cached_chunk_state="May be mutated; preserve and discard before any retry",
            complete_network_acceptance="not_run",
        ))
        raise

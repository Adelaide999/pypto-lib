# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare shared network Engram hashes and both lookup slots before the backbone."""

from contextlib import ExitStack
from pathlib import Path

import torch

from .engram_storage import SharedEngramArray
from .prefill_engram import make_prefill_engram_entries
from .prefill_engram_runtime import MappedLookup, engram_owner, load_rotation
from .prefill_layer_state import resolve_layer_role
from .runtime_config import configure_devices
from .runtime_utils import Executable


class NetworkEngramHistory:
    """Own one V1 first-SWA slot history per rank/request batch across chunks."""

    def __init__(self, bindings, initial_caches=None, *, geometry_only=False):
        self.bindings = bindings
        self.geometry_only = geometry_only
        self.caches = [{} for _ in bindings]
        self.prepared = set()
        rows = set()
        for rank, records in enumerate(bindings):
            for case, binding in records.items():
                if (binding["owner_layer"] != 1 or binding["block_size"] != 128
                        or not binding["bound"] or not binding["kv_binding_id"] or binding["kv_blocks"] <= 0):
                    raise ValueError("Network V1 history requires the bound physical1 SWA identity and 128-row pages")
                count = binding["kv_blocks"] * binding["block_size"]
                device = "meta" if geometry_only else "cpu"
                cache = (torch.zeros(count, dtype=torch.int32, device=device) if initial_caches is None
                         else initial_caches[rank][case])
                if cache.dtype != torch.int32 or cache.device.type != device or tuple(cache.shape) != (count,):
                    raise ValueError("Network history must retain the exact first-SWA INT32 slot extent")
                self.caches[rank][case] = cache
                rows.add(count)
        if len(rows) != 1:
            raise ValueError("Freeze a uniform bound first-SWA history extent for the current executable ABI")
        self.cache_rows = rows.pop()


def _pad(value, rows, fill=0):
    target = torch.full((rows, *value.shape[1:]), fill, dtype=value.dtype)
    target[:len(value)].copy_(value)
    return target


def _check_constants(common, other):
    if common["layer_ids"] != other["layer_ids"] or common["pad_id"] != other["pad_id"]:
        raise ValueError("Both table slots must share the same original hash layout")
    for name in ("token_map", "multipliers", "primes", "offsets"):
        if not torch.equal(common[name], other[name]):
            raise ValueError("Both table slots must share exact hash constants: " + name)


class PreparedNetworkEngram:
    """Retain chunk lookup inputs; project and gate each actual layer hidden later."""

    def __init__(self, batch, history, states, contexts, raw_for_layer, runtime):
        self.batch, self.history, self.states = batch, history, states
        self.contexts, self.raw_for_layer, self.runtime = contexts, raw_for_layer, runtime
        self.applied = set()

    def attach(self, layer_id, cache_state):
        """Bind actual prepared slot objects into the in-memory layer state handles."""
        if layer_id not in self.states or len(cache_state) != len(self.history.caches):
            raise ValueError("Attach one original Engram slot to the complete world")
        for rank, records in enumerate(self.states[layer_id]):
            for case, state in records.items():
                cache_state[rank][case]["network_engram"] = state

    def apply(self, role, batch, local_hidden, cache_state, *, exchange):
        """Use a prepared lookup with the current product hidden; do not hash again."""
        layer_id = role["physical_layer"]
        role = resolve_layer_role(layer_id, role)
        if layer_id not in self.states or layer_id in self.applied:
            raise ValueError("Consume each prepared original Engram slot once per network chunk")
        if (batch["epoch"], batch["token_map_id"]) != (self.batch["epoch"], self.batch["token_map_id"]):
            raise ValueError("Prepared Engram and current layer identify different chunks")
        context = self.contexts[layer_id]
        runtime, capacity = self.runtime, self.runtime["capacity"]
        raw = self.raw_for_layer(layer_id)
        prefix = context["prefix"]
        projection = (runtime["projection_for_layer"](layer_id) if "projection_for_layer" in runtime
                      else raw[prefix + ".wkv.weight"].T.contiguous())
        if projection.dtype != torch.bfloat16 or tuple(projection.shape) != (6144, 25600):
            raise ValueError("Require the actual complete BF16 Engram projection")
        result = [{} for _ in self.history.caches]
        for rank, records in enumerate(self.states[layer_id]):
            for case, prepared in records.items():
                original, source = batch["cases"][case][rank], self.batch["cases"][case][rank]
                if cache_state[rank][case].get("network_engram") is not prepared:
                    raise ValueError("The layer must use this preparation's actual slot object")
                for name in ("full_input_ids", "positions", "input_ids"):
                    if not torch.equal(original[name], source[name]):
                        raise ValueError("Prepared Engram token coordinates differ: " + name)
                if (original["SP_physical_rows"] != source["SP_physical_rows"]
                        or original["SP_valid_rows"] != source["SP_valid_rows"]
                        or tuple(original["cp_token_range"]) != tuple(source["cp_token_range"])):
                    raise ValueError("Prepared Engram SP/CP token mapping differs")
                physical = original["SP_physical_rows"]
                hidden_input = local_hidden[rank][case]
                if hidden_input.dtype != torch.bfloat16 or tuple(hidden_input.shape) != (capacity, 4, 5120):
                    raise ValueError("Engram gate requires this layer's local padded product hidden")
                projected = torch.zeros((capacity, 25600), dtype=torch.bfloat16)
                context["execute"]("engram_projection", rank, case,
                    [prepared["local"], projection, projected, physical], {"projected": projected}, "engram_projection")
                restored = torch.zeros((capacity, 4, 5120), dtype=torch.float32)
                probability = torch.zeros((capacity, 4), dtype=torch.float32)
                hidden = torch.zeros((capacity, 4, 5120), dtype=torch.bfloat16)
                arguments = [hidden_input, projected, raw[prefix + ".q_weight"], raw[prefix + ".k_weight"],
                             context["rotation"], prepared["local_mask"], restored, probability, hidden, physical]
                context["execute"]("engram_gate", rank, case, arguments,
                    dict(original=restored, probability=probability, hidden=hidden), "engram_gate")
                nodes = {}
                if runtime.get("capture_nodes", False):
                    total = len(source["full_input_ids"])
                    nodes = dict(engram_hash_ids=prepared["hashes"][:total].clone(),
                        engram_token_mask=prepared["mask"][:total].to(torch.bool),
                        engram_lookup_full=prepared["full"][:total].clone(),
                        engram_lookup_local=prepared["local"][:physical].clone(),
                        engram_projection=projected[:physical].clone(), engram_hidden_after=hidden[:physical].clone())
                result[rank][case] = dict(hidden=hidden, nodes=nodes,
                    hash_cache_before=prepared["hash_cache_before"], hash_cache_after=prepared["hash_cache_after"],
                    network_history_owner=1, network_hash_slot=role["engram_slot"])
        self.applied.add(layer_id)
        return result


def _hash_slots(slots):
    """Bind canonical page/offset coordinates to the hash kernel's physical row vector."""
    if slots.dtype not in (torch.int32, torch.int64):
        raise ValueError("Engram history slots require integer coordinates")
    if slots.ndim == 1:
        return slots.to(torch.int64)
    if slots.ndim != 2 or slots.shape[1] != 2:
        raise ValueError("Engram history slots require rows or page/offset pairs")
    values = []
    for page, offset in slots.tolist():
        if offset >= 128:
            raise ValueError("Engram history offset exceeds its 128-row page")
        values.append(page * 128 + offset if page >= 0 and offset >= 0 else -1)
    return torch.tensor(values, dtype=torch.int64)


def prepare_network_engram(contract, batch, history, layer_contracts, raw_for_layer, runtime, *,
                           storage=None, storage_stack=None, storage_namespace=None):
    """Hash the global chunk once and prepare both table lookups with existing PyPTO entries."""
    if getattr(history, "geometry_only", False):
        raise ValueError("Compile-only Engram geometry cannot be used for execution")
    if (storage is None) != (storage_stack is None):
        raise ValueError("Caller-owned table storage requires the same finite owner ExitStack")
    constants = contract["hash_constants"]
    if tuple(constants["layer_ids"]) != (1, 14) or set(layer_contracts) != {1, 14}:
        raise ValueError("Require the pinned original Engram slot0/1 ordering")
    topology = runtime["topology"]
    tp, world, capacity = topology["tp"], topology["ep"], runtime["capacity"]
    if (tp not in (1, 2, 4) or world != topology["dp"] * tp or not 2 <= world <= 16
            or any(topology[name] != 1 for name in ("pp", "pcp", "dcp"))):
        raise ValueError("Require the complete original local DP/TP/EP topology")
    if (contract["topology"] != topology or contract["capacity"] != capacity
            or not 0 < capacity <= 512 or capacity % 16):
        raise ValueError("Freeze the matching bounded network Engram execution ABI")
    configure_devices(runtime["assigned_devices"], world=world)
    if len(history.caches) != world or batch.get("engram_metadata_owner_layer") != 1:
        raise ValueError("Use the single model-level history and first-Engram SWA metadata")
    identity = (batch["epoch"], batch["token_map_id"])
    if identity in history.prepared:
        raise ValueError("This history has already attempted the chunk; preserve failure or restore before retry")
    if any(len(records) != world for records in batch["cases"].values()):
        raise ValueError("Prepare every actual rank, including empty DP replicas")
    namespace = runtime["stage_namespace"]
    if not namespace or "/" in namespace or ".." in namespace:
        raise ValueError("Require a private safe chunk namespace")
    storage_contract = dict(dtype="INT8", scale_dtype="FP32", group_size=32, cpu_offload=True,
                   dp_shared_memory=True, global_columns=24)
    if contract["engram_storage_contract"] != storage_contract:
        raise ValueError("The current network path requires full INT8/FP32 DP-shared host tables")
    max_blocks = 1
    hashing = [{} for _ in range(world)]
    for case, records in batch["cases"].items():
        for rank, original in enumerate(records):
            if case not in history.caches[rank]:
                raise ValueError("Missing actual first-SWA request history binding")
            total = len(original["full_input_ids"])
            metadata = original["metadata"]
            starts, blocks = metadata["query_start_loc"], metadata["block_table"]
            if (total > capacity or total != len(original["positions"])
                    or not 1 <= len(starts) <= capacity + 1):
                raise ValueError("Invalid bounded full-token/request coordinates")
            requests = len(starts) - 1
            active = not original.get("force_dummy", False) and requests > 0 and blocks.shape[0] > 0
            if active and (int(starts[0]) != 0 or int(starts[-1]) != total or blocks.shape[0] < requests):
                raise ValueError("First-SWA request metadata must cover the actual native token extent")
            if active and original.get("valid_token_count", total) != total:
                raise ValueError("Masked graph-bucket token padding requires a separate explicit hash contract")
            if original.get("token_indices") is not None:
                raise ValueError("Nonidentity hash token selection is not implemented by this bounded network entry")
            if blocks.shape[0] > capacity:
                raise ValueError("Too many first-SWA block-table rows")
            max_blocks = max(max_blocks, blocks.shape[1])
            hashing[rank][case] = bool(active)
    output = Path(runtime["output"])
    output.mkdir(parents=True, exist_ok=False)
    reports, memory = runtime["reports"], runtime["memory"]
    if len(reports) != world:
        raise ValueError("Keep one network execution report for every allocated rank")
    config = dict(contract["model_config"]["text_config"])
    config["image_token_id"] = contract["model_config"]["image_token_id"]
    config["image_pad_token_id"] = contract["model_config"].get("image_pad_token_id", config["image_token_id"] + 1)
    contexts, shared_entries = {}, {}
    for layer_id in constants["layer_ids"]:
        own_contract = layer_contracts[layer_id]
        _check_constants(constants, own_contract["hash_constants"])
        if (own_contract["checkpoint"] != contract["checkpoint"] or own_contract["topology"] != topology
                or own_contract["model_config"] != contract["model_config"] or own_contract["capacity"] != capacity
                or own_contract["engram_storage_contract"] != storage_contract):
            raise ValueError("Both slots require the same frozen checkpoint, topology and storage modes")
        prefix, slot = engram_owner(own_contract, layer_id, resolve_layer_role(layer_id))
        directory = output / f"layer-{layer_id}"
        (directory / "captures").mkdir(parents=True)
        entries = {}
        if "entries_for_stage" in runtime:
            entries = runtime["entries_for_stage"](f"engram{layer_id}")
            for entry in entries.values():
                entry.diagnostics = runtime.get("diagnostics", True)
        else:
            functions = make_prefill_engram_entries(capacity, constants, config, own_contract["engram_shards"],
                                                    history.cache_rows, max_blocks, tp)
            for name, function in functions.items():
                if name.startswith("engram_lookup_") or name not in shared_entries:
                    entry = Executable(function, directory / "compiled" / name, diagnostics=runtime.get("diagnostics", True))
                    if not name.startswith("engram_lookup_"):
                        shared_entries[name] = entry
                else:
                    entry = shared_entries[name]
                entries[name] = entry
        rotation, rotation_receipt = runtime.get("load_rotation", load_rotation)(contract["checkpoint"], 5120)

        def execute(name, rank, case, arguments, outputs, roles, *, entries=entries, directory=directory):
            if not runtime.get("diagnostics", True):
                runtime["execute_entry"](entries[name], rank, arguments)
                return
            capture = directory / "captures" / f"rank-{rank}-{case}-{name}-{len(reports[rank]['kernels'])}.pt"
            entries[name].execute(arguments, outputs, rank, roles, capture, reports[rank], memory)

        contexts[layer_id] = dict(prefix=prefix, slot=slot, entries=entries, execute=execute,
                                  directory=directory, rotation=rotation, rotation_receipt=rotation_receipt,
                                  shards=own_contract["engram_shards"])
    history.prepared.add(identity)
    hashes = [{} for _ in range(world)]
    for case, records in batch["cases"].items():
        for rank, original in enumerate(records):
            metadata = original["metadata"]
            total, cache = len(original["full_input_ids"]), history.caches[rank][case]
            before = cache.clone() if runtime.get("capture_state", False) else None
            ids = torch.full((capacity, 2, 24), -1, dtype=torch.int32)
            mask = torch.zeros(capacity, dtype=torch.int32)
            receipt = dict(owner_layer=1, kv_binding_id=history.bindings[rank][case]["kv_binding_id"],
                history_shape=list(cache.shape), history_dtype=str(cache.dtype), selected_slots={"1": 0, "14": 1},
                epoch=batch["epoch"], token_map_id=batch["token_map_id"], status="not_run", hash_executions=0)
            reports[rank].setdefault("network_engram_hash", {})[case] = receipt
            if hashing[rank][case]:
                starts = metadata["query_start_loc"]
                blocks = torch.full((capacity, max_blocks), -1, dtype=torch.int32)
                original_blocks = metadata["block_table"]
                blocks[:original_blocks.shape[0], :original_blocks.shape[1]].copy_(original_blocks)
                lookback = torch.full((capacity, 3), -1, dtype=torch.int64)
                if "lookback_token_ids" in original:
                    value = original["lookback_token_ids"]
                    lookback[:len(value)].copy_(value)
                arguments = [_pad(original["full_input_ids"], capacity), _pad(original["positions"], capacity),
                    _pad(starts, capacity + 1, total), _pad(_hash_slots(metadata["slot_mapping"]), capacity, -1), blocks, lookback,
                    constants["token_map"], constants["multipliers"], constants["primes"].reshape(-1, 24),
                    constants["offsets"], cache, ids, mask, total, len(starts) - 1]
                contexts[1]["execute"]("engram_hash", rank, case, arguments,
                                       dict(cache=cache, hashes=ids, mask=mask), "engram_hash")
                receipt.update(status="executed", hash_executions=1)
            else:
                receipt["status"] = "skipped_no_requests_or_dummy_history_untouched"
            hashes[rank][case] = dict(hashes=ids, mask=mask, hash_cache_before=before,
                hash_cache_after=cache.clone() if runtime.get("capture_state", False) else None,
                hashing=hashing[rank][case])
    states = {}
    for layer_id, context in contexts.items():
        packets = [{case: torch.zeros((capacity, 6144 // tp), dtype=torch.bfloat16)
                    for case in batch["cases"]} for _ in range(world)]
        with ExitStack() as stack:
            for tp_rank, shard in enumerate(context["shards"]):
                owner_namespace = namespace if storage is None else storage_namespace
                if not owner_namespace or "/" in owner_namespace or ".." in owner_namespace:
                    raise ValueError("Require a safe allocation-scoped table owner")
                owner = f"{owner_namespace}-layer{layer_id}-tp{tp_rank}"
                arrays = {}
                for kind, suffix, width, dtype in (("codes", ".embed.weight", 256, "I8"), ("scales", ".embed.scale", 8, "F32")):
                    key = (layer_id, tp_rank, kind)
                    identity = dict(checkpoint=contract["checkpoint"], tensor=context["prefix"] + suffix,
                        row_range=shard["row_range"], width=width, dtype=dtype, sha256=shard["host_payload_sha256"][kind])
                    if storage is not None and key in storage:
                        record = storage[key]
                        if record["identity"] != identity or record["array"].mapping is None or record["array"].live:
                            raise ValueError("Retained complete Engram table ownership or checkpoint changed")
                        arrays[kind] = record["array"]
                    else:
                        array = SharedEngramArray(identity["checkpoint"], identity["tensor"], identity["row_range"],
                            width, dtype, identity["sha256"], owner)
                        cleanup = stack if storage is None else storage_stack
                        cleanup.callback(array.close)
                        arrays[kind] = array
                        if storage is not None:
                            storage[key] = dict(identity=identity, array=array)
                codes, scales = arrays["codes"], arrays["scales"]
                ranks = list(range(tp_rank, world, tp))
                if ranks != shard["DP_sharing_ranks"]:
                    raise ValueError("Engram TP shard must be shared by all matching local DP peers")
                for rank in ranks:
                    reports[rank].setdefault("network_engram_storage", {})[str(layer_id)] = dict(
                        codes=codes.receipt, scales=scales.receipt, DP_sharing_ranks=ranks,
                        rotation=context["rotation_receipt"], shared_history_owner=1, slot=context["slot"])
                    if not any(hashing[rank].values()):
                        reports[rank]["network_engram_storage"][str(layer_id)]["registration"] = "not_run_empty_DP"
                        continue
                    name = f"engram_lookup_{tp_rank}"
                    lookup = runtime.get("mapped_lookup", MappedLookup)(context["entries"][name], codes, scales, rank, memory)
                    with lookup as mapped:
                        for case in batch["cases"]:
                            if hashing[rank][case]:
                                ids = hashes[rank][case]["hashes"][:, context["slot"], :].contiguous()
                                capture = context["directory"] / "captures" / f"rank-{rank}-{case}-{name}.pt"
                                mapped.execute(ids, packets[rank][case], len(batch["cases"][case][rank]["full_input_ids"]),
                                               capture, reports[rank])
                    reports[rank]["network_engram_storage"][str(layer_id)]["unregistered"] = True
            gathered = runtime["exchange"](f"{namespace}-layer{layer_id}-engram-columns", packets,
                                            "all_gather", "tp", packets=False)
        states[layer_id] = [{} for _ in range(world)]
        for case, records in batch["cases"].items():
            for rank, original in enumerate(records):
                total, physical = len(original["full_input_ids"]), original["SP_physical_rows"]
                full = torch.zeros((capacity, 6144), dtype=torch.bfloat16)
                local = torch.zeros((capacity, 6144), dtype=torch.bfloat16)
                local_mask = torch.zeros(capacity, dtype=torch.int32)
                arguments = [gathered[rank][case], hashes[rank][case]["mask"], full, local, local_mask,
                             physical, total, rank % tp]
                context["execute"]("engram_layout", rank, case, arguments,
                                   dict(full=full, local=local, local_mask=local_mask), "engram_token_shard")
                states[layer_id][rank][case] = dict(hashes[rank][case], local=local, local_mask=local_mask,
                    full=full if runtime.get("capture_nodes", False) else None,
                    shared_history_owner=1, physical_layer=layer_id, slot=context["slot"],
                    epoch=batch["epoch"], token_map_id=batch["token_map_id"])
                observe = runtime.get("observe_discrete")
                if observe is not None:
                    observe(layer_id, rank, case, dict(
                        total=total, physical=physical, valid=original["SP_valid_rows"],
                        hashing=hashes[rank][case]["hashing"], hash_ids=hashes[rank][case]["hashes"],
                        token_mask=hashes[rank][case]["mask"], lookup_full=full,
                        lookup_local=local, local_mask=local_mask,
                    ))
                reports[rank]["network_engram_storage"][str(layer_id)]["closed_after_all_DP_peers"] = storage is None
    return PreparedNetworkEngram(batch, history, states, contexts, raw_for_layer, runtime)

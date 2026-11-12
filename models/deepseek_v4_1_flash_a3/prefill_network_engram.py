# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bind host-prepared Engram rows to device projection and gating."""

from pathlib import Path

import torch

from .engram_hash import update_slot_history
from .engram import make_prefill_engram_entries
from .prefill_engram_runtime import engram_owner, load_rotation
from .prefill_layer_state import resolve_layer_role
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
                if (
                    binding["owner_layer"] != 1
                    or binding["block_size"] != 128
                    or not binding["bound"]
                    or not binding["kv_binding_id"]
                    or binding["kv_blocks"] <= 0
                ):
                    raise ValueError(
                        "Network V1 history requires the bound physical1 SWA identity and 128-row pages"
                    )
                count = binding["kv_blocks"] * binding["block_size"]
                device = "meta" if geometry_only else "cpu"
                cache = (
                    torch.zeros(count, dtype=torch.int32, device=device)
                    if initial_caches is None
                    else initial_caches[rank][case]
                )
                if (
                    cache.dtype != torch.int32
                    or cache.device.type != device
                    or tuple(cache.shape) != (count,)
                ):
                    raise ValueError("Network history must retain the exact first-SWA INT32 slot extent")
                self.caches[rank][case] = cache
                rows.add(count)
        if len(rows) != 1:
            raise ValueError("Freeze a uniform bound first-SWA history extent for the current executable ABI")
        self.cache_rows = rows.pop()


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
        projection = (
            runtime["projection_for_layer"](layer_id)
            if "projection_for_layer" in runtime
            else raw[prefix + ".wkv.weight"].T.contiguous()
        )
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
                if (
                    original["SP_physical_rows"] != source["SP_physical_rows"]
                    or original["SP_valid_rows"] != source["SP_valid_rows"]
                    or tuple(original["cp_token_range"]) != tuple(source["cp_token_range"])
                ):
                    raise ValueError("Prepared Engram SP/CP token mapping differs")
                physical = original["SP_physical_rows"]
                hidden_input = local_hidden[rank][case]
                if hidden_input.dtype != torch.bfloat16 or tuple(hidden_input.shape) != (capacity, 4, 5120):
                    raise ValueError("Engram gate requires this layer's local padded product hidden")
                projected = torch.zeros((capacity, 25600), dtype=torch.bfloat16)
                context["execute"](
                    "engram_projection",
                    rank,
                    case,
                    [prepared["local"], projection, projected, physical],
                    {"projected": projected},
                    "engram_projection",
                )
                restored = torch.zeros((capacity, 4, 5120), dtype=torch.float32)
                probability = torch.zeros((capacity, 4), dtype=torch.float32)
                hidden = torch.zeros((capacity, 4, 5120), dtype=torch.bfloat16)
                arguments = [
                    hidden_input,
                    projected,
                    raw[prefix + ".q_weight"],
                    raw[prefix + ".k_weight"],
                    context["rotation"],
                    prepared["local_mask"],
                    restored,
                    probability,
                    hidden,
                    physical,
                ]
                context["execute"](
                    "engram_gate",
                    rank,
                    case,
                    arguments,
                    dict(original=restored, probability=probability, hidden=hidden),
                    "engram_gate",
                )
                nodes = {}
                if runtime.get("capture_nodes", False):
                    total = len(source["full_input_ids"])
                    nodes = dict(
                        engram_hash_ids=prepared["hashes"][:total].clone(),
                        engram_token_mask=prepared["mask"][:total].to(torch.bool),
                        engram_lookup_full=prepared["full"][:total].clone(),
                        engram_lookup_local=prepared["local"][:physical].clone(),
                        engram_projection=projected[:physical].clone(),
                        engram_hidden_after=hidden[:physical].clone(),
                    )
                result[rank][case] = dict(
                    hidden=hidden,
                    nodes=nodes,
                    hash_cache_before=prepared["hash_cache_before"],
                    hash_cache_after=prepared["hash_cache_after"],
                    network_history_owner=1,
                    network_hash_slot=role["engram_slot"],
                )
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


def validate_host_lookup(lookup, contract, batch):
    """Check prepared row coordinates and tensor ABI before allocating device resources."""
    topology, capacity = contract["topology"], contract["capacity"]
    world = topology["ep"]
    identity = batch["epoch"], batch["token_map_id"]
    if (lookup["epoch"], lookup["token_map_id"]) != identity:
        raise ValueError("Host Engram rows identify a different chunk")
    if (
        lookup["topology"] != topology
        or lookup["capacity"] != capacity
        or Path(lookup["checkpoint"]).resolve() != Path(contract["checkpoint"]).resolve()
        or set(lookup["cases"]) != set(batch["cases"])
    ):
        raise ValueError("Host Engram rows do not match the forward ABI")
    _check_constants(contract["hash_constants"], lookup["hash_constants"])
    for case, records in batch["cases"].items():
        if len(records) != world or len(lookup["cases"][case]) != world:
            raise ValueError("Host rows must cover active and empty DP ranks")
        for rank, original in enumerate(records):
            ready = lookup["cases"][case][rank]
            for name in ("full_input_ids", "positions"):
                if not torch.equal(ready[name], original[name]):
                    raise ValueError("Host Engram token coordinates changed: " + name)
            if tuple(ready["cp_token_range"]) != tuple(original["cp_token_range"]):
                raise ValueError("Host Engram SP/TP partition changed")
            for layer in (1, 14):
                for name in ("full", "local"):
                    value = ready[name][layer]
                    if (
                        value.device.type != "cpu"
                        or value.dtype != torch.bfloat16
                        or tuple(value.shape) != (capacity, 6144)
                        or not value.is_contiguous()
                    ):
                        raise ValueError("Forward requires ready contiguous BF16 Engram rows")
            for name, shape in (
                ("hashes", (capacity, 2, 24)),
                ("mask", (capacity,)),
                ("local_mask", (capacity,)),
            ):
                value = ready[name]
                if value.device.type != "cpu" or value.dtype != torch.int32 or tuple(value.shape) != shape:
                    raise ValueError("Host Engram control ABI changed: " + name)


def prepare_network_engram(contract, batch, history, layer_contracts, raw_for_layer, runtime, *, lookup):
    """Consume ready host rows; the forward never opens or registers a full table."""
    topology, capacity = runtime["topology"], runtime["capacity"]
    world = topology["ep"]
    identity = batch["epoch"], batch["token_map_id"]
    if history.geometry_only or identity in history.prepared:
        raise ValueError("Use an execution history and consume each chunk once")
    if set(layer_contracts) != {1, 14} or batch.get("engram_metadata_owner_layer") != 1:
        raise ValueError("Require both original slots and the first-SWA history owner")
    validate_host_lookup(lookup, contract, batch)
    output = Path(runtime["output"])
    output.mkdir(parents=True, exist_ok=False)
    contexts, states = {}, {}
    shared_entries = {}
    for layer_id in (1, 14):
        own = layer_contracts[layer_id]
        _check_constants(contract["hash_constants"], own["hash_constants"])
        prefix, slot = engram_owner(own, layer_id, resolve_layer_role(layer_id))
        directory = output / f"layer-{layer_id}"
        (directory / "captures").mkdir(parents=True)
        if "entries_for_stage" in runtime:
            entries = runtime["entries_for_stage"](f"engram{layer_id}")
        else:
            model = dict(
                own["model_config"]["text_config"], image_token_id=own["model_config"]["image_token_id"]
            )
            definitions = make_prefill_engram_entries(
                capacity,
                own["hash_constants"],
                model,
                own["engram_shards"],
                history.cache_rows,
                1,
                topology["tp"],
            )
            for name in ("engram_projection", "engram_gate"):
                if name not in shared_entries:
                    shared_entries[name] = Executable(
                        definitions[name],
                        directory / "compiled" / name,
                        diagnostics=runtime.get("diagnostics", True),
                    )
            entries = shared_entries
        rotation, receipt = runtime.get("load_rotation", load_rotation)(contract["checkpoint"], 5120)

        def execute(name, rank, case, arguments, outputs, roles, *, entries=entries, directory=directory):
            if not runtime.get("diagnostics", True):
                runtime["execute_entry"](entries[name], rank, arguments)
                return
            capture = directory / "captures" / f"rank-{rank}-{case}-{name}.pt"
            entries[name].execute(
                arguments, outputs, rank, roles, capture, runtime["reports"][rank], runtime["memory"]
            )

        contexts[layer_id] = dict(
            prefix=prefix, slot=slot, execute=execute, rotation=rotation, rotation_receipt=receipt
        )
        states[layer_id] = [{} for _ in range(world)]
    history.prepared.add(identity)
    constants = contract["hash_constants"]
    for case, records in batch["cases"].items():
        for rank, original in enumerate(records):
            ready = lookup["cases"][case][rank]
            cache = history.caches[rank][case]
            before = cache.clone() if runtime.get("capture_state", False) else None
            total = len(original["full_input_ids"])
            update_slot_history(
                original["full_input_ids"],
                constants["token_map"],
                ready["mask"][:total] == 0,
                _hash_slots(original["metadata"]["slot_mapping"]),
                cache,
            )
            after = cache.clone() if runtime.get("capture_state", False) else None
            report = runtime["reports"][rank]
            report.setdefault("network_engram_hash", {})[case] = dict(
                owner_layer=1,
                epoch=identity[0],
                token_map_id=identity[1],
                status="host_prepared",
                device_hash_executions=0,
            )
            for layer in (1, 14):
                state = dict(
                    local=ready["local"][layer],
                    local_mask=ready["local_mask"],
                    full=ready["full"][layer],
                    hashes=ready["hashes"],
                    mask=ready["mask"],
                    hash_cache_before=before,
                    hash_cache_after=after,
                    shared_history_owner=1,
                    physical_layer=layer,
                    slot=contexts[layer]["slot"],
                    epoch=identity[0],
                    token_map_id=identity[1],
                )
                states[layer][rank][case] = state
                report.setdefault("network_engram_storage", {})[str(layer)] = dict(
                    lookup="host_prepared_BF16_rows",
                    table_device_registration=False,
                    rotation=contexts[layer]["rotation_receipt"],
                    shared_history_owner=1,
                    slot=state["slot"],
                )
                observe = runtime.get("observe_discrete")
                if observe is not None:
                    observe(
                        layer,
                        rank,
                        case,
                        dict(
                            total=total,
                            physical=original["SP_physical_rows"],
                            valid=original["SP_valid_rows"],
                            hashing=total > 0,
                            hash_ids=ready["hashes"],
                            token_mask=ready["mask"],
                            lookup_full=ready["full"][layer],
                            lookup_local=ready["local"][layer],
                            local_mask=ready["local_mask"],
                        ),
                    )
    return PreparedNetworkEngram(batch, history, states, contexts, raw_for_layer, runtime)

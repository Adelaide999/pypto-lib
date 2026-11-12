# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Network cache providers, exact preservation and product observation."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
import torch
from .cache import resolve_layer_role
from .prefill_compressed_runtime import physical_cache_views
from .prefill_network_engram import NetworkEngramHistory
from .runtime_utils import file_sha, write_json


NAMES = ("topk_indices", "topk_lengths", "candidates", "candidate_lengths")
SHAPES = {
    "topk_indices": (8192, 512),
    "topk_lengths": (8192, 1),
    "candidates": (8192, 1, 2048),
    "candidate_lengths": (8192, 1),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def snapshot(value):
    return value.detach().cpu().clone()


def digest(value, start):
    data = value[start:].detach().cpu().contiguous()
    header = dict(dtype=str(value.dtype), shape=list(value.shape), start=start)
    state = hashlib.sha256(json.dumps(header, sort_keys=True).encode() + b"\0")
    state.update(data.numpy().tobytes(order="C"))
    return state.hexdigest()


def begin_shared(shared, query_rows, *, index_write, candidate_write):
    require(0 <= query_rows <= 16, "Require the frozen bounded query prefix")
    result = dict(query_rows=query_rows, prefix_before={}, complements={})
    for name in NAMES:
        value = shared[name]
        require(
            tuple(value.shape) == SHAPES[name] and str(value.dtype) == "torch.int32",
            "Shared scheduler storage ABI differs: " + name,
        )
        writes = (name in ("topk_indices", "topk_lengths") and index_write) or (
            name == "candidates" and candidate_write
        )
        start = query_rows if writes else 0
        result["prefix_before"][name] = snapshot(value[:query_rows])
        result["complements"][name] = dict(
            start=start,
            end=8192,
            before_sha256=digest(value, start),
            policy="exact_own_before_after",
            result_dependent_mask=False,
        )
    return result


def end_shared(shared, before):
    result = dict(
        query_rows=before["query_rows"],
        prefix_before=before["prefix_before"],
        prefix_after={},
        complements={},
    )
    for name in NAMES:
        previous = before["complements"][name]
        after = digest(shared[name], previous["start"])
        result["prefix_after"][name] = snapshot(shared[name][: before["query_rows"]])
        result["complements"][name] = dict(
            previous, after_sha256=after, exact_equal=previous["before_sha256"] == after
        )
        require(result["complements"][name]["exact_equal"], "Changed readonly/inactive shared state: " + name)
    return result


def begin_product_observation(provider, layer_id):
    batch = provider.batch_for_layer(layer_id)
    pending = [{} for _ in provider.shared]
    for case, records in batch["cases"].items():
        for rank, original in enumerate(records):
            # Roles are recorded from actual result states after the existing layer.
            # The admitted original source plan fixes writer masks before execution.
            role = resolve_layer_role(layer_id)
            pending[rank][case] = dict(
                shared=begin_shared(
                    provider.shared[rank][case],
                    original["actual_local_query_rows"],
                    index_write=role["index_selection_write"],
                    candidate_write=role["candidate_write"],
                ),
                versions_before=deepcopy(provider.versions[rank][case]),
            )
    return pending


def end_product_observation(provider, layer_id, result, pending):
    batch = provider.batch_for_layer(layer_id)
    records = [{} for _ in provider.shared]
    for case, originals in batch["cases"].items():
        routing_rows = max(original["SP_physical_rows"] for original in originals)
        for rank, original in enumerate(originals):
            state = result["states"][rank][case]
            require(
                state["role"]["physical_layer"] == layer_id and state["routing_rows"] == routing_rows,
                "Actual product router/layer identity differs",
            )
            require(
                all(name in state for name in ("route_ids", "weights", "received_routes", "returned_routes")),
                "Existing product state omitted required actual route metadata",
            )
            records[rank][case] = dict(
                layer=layer_id,
                rank=rank,
                epoch=batch["epoch"],
                token_map_id=batch["token_map_id"],
                query_rows=original["actual_local_query_rows"],
                SP_physical_rows=state["physical"],
                SP_valid_rows=state["valid"],
                cp_token_range=list(original["cp_token_range"]),
                source=deepcopy(state["role"]),
                metadata_source=(
                    None
                    if state["role"]["mode"] == "swa"
                    else dict(
                        physical_layer_id=original["cache_metadata"]["physical_layer_id"],
                        source_physical_layer_id=original["cache_metadata"]["source_physical_layer_id"],
                    )
                ),
                versions_before=pending[rank][case]["versions_before"],
                versions_after=deepcopy(provider.versions[rank][case]),
                router=dict(
                    route_ids=snapshot(state["route_ids"][:routing_rows]),
                    route_weights=snapshot(state["weights"][:routing_rows]),
                    route_rows=routing_rows,
                    inactive_route_ids=snapshot(state["route_ids"][routing_rows:]),
                    inactive_route_weights=snapshot(state["weights"][routing_rows:]),
                    routing_input_token_ids=snapshot(state["token_ids"]),
                    hook="Existing result.states; PyPTO router outputs unchanged",
                    tensor_computation_added=False,
                ),
                received_routes=snapshot(state["received_routes"]),
                returned_routes=snapshot(state["returned_routes"]),
                shared=end_shared(provider.shared[rank][case], pending[rank][case]["shared"]),
            )
    return records


def product_engram_observation(layer_id, rank, case, observation):
    total, physical = observation["total"], observation["physical"]
    require(layer_id in (1, 14), "Require one original Engram lookup slot")
    return dict(
        layer=layer_id,
        rank=rank,
        case=case,
        total=total,
        physical=physical,
        valid=observation["valid"],
        hashing=observation["hashing"],
        hash_ids=snapshot(observation["hash_ids"][:total]),
        token_mask=snapshot(observation["token_mask"][:total]),
        lookup_full=snapshot(observation["lookup_full"][:total]),
        lookup_local=snapshot(observation["lookup_local"][:physical]),
        local_mask=snapshot(observation["local_mask"][:physical]),
        inactive=dict(
            hash_ids=snapshot(observation["hash_ids"][total:]),
            token_mask=snapshot(observation["token_mask"][total:]),
            lookup_full=snapshot(observation["lookup_full"][total:]),
            lookup_local=snapshot(observation["lookup_local"][physical:]),
            local_mask=snapshot(observation["local_mask"][physical:]),
        ),
        hook="Existing network Engram layout outputs before capture_nodes=False discards full lookup",
        tensor_computation_added=False,
    )


def build_state_regions(payloads, input_contract_sha256, scheduler_capacity=8192):
    """Derive exact write complements from chunk slots, ring rows and layer roles."""

    def complement(extent, intervals):
        cursor, result = 0, []
        for begin, end in sorted(intervals):
            if not 0 <= begin < end <= extent:
                raise ValueError("A prepared write interval escapes its original byte arena")
            if cursor < begin:
                result.append([cursor, begin])
            cursor = max(cursor, end)
        if cursor < extent:
            result.append([cursor, extent])
        return result

    ranks = {}
    for rank, payload in enumerate(payloads):
        layout, metadata = payload["layout"], payload["metadata"]
        extents = {key: value["shape"][0] for key, value in layout["raw"].items()}
        layers = []
        for layer in range(40):
            role = resolve_layer_role(layer)
            root = f"model.layers.{layer}.self_attn."
            writable = [root + "swa_cache"]
            if role["mode"] == "full":
                writable += [root + "long_kv_cache", root + "indexer.k_cache"]
                if role["ratio"] == 2:
                    writable.append(root + "compressor.state_cache")
            intervals = {key: [] for key in extents}
            for name in writable:
                controls = metadata[name]["global_metadata"]
                slots = controls["slot_mapping"].tolist()
                if name.endswith("compressor.state_cache"):
                    starts = payload["query_start_loc"].tolist()
                    table = controls["block_table"].tolist()
                    slots = [
                        [table[request][0], int(payload["positions"][token]) % 32]
                        for request, (begin, end) in enumerate(zip(starts, starts[1:]))
                        for token in range(begin, end)
                    ]
                descriptor = layout["views"][name]
                for plane in descriptor if isinstance(descriptor, list) else [descriptor]:
                    dtype = getattr(torch, plane["dtype"].split(".")[-1])
                    size = torch.empty((), dtype=dtype).element_size()
                    for page, row in slots:
                        if page < 0:
                            continue
                        offset = (
                            plane["storage_offset_bytes"]
                            + (page * plane["stride"][0] + row * plane["stride"][1]) * size
                        )
                        intervals[plane["storage_key"]].append([offset, offset + plane["shape"][-1] * size])
            active = metadata[root + "swa_cache"]["num_actual_tokens"]
            shared = {}
            for name in ("topk_indices", "topk_lengths", "candidates", "candidate_lengths"):
                writes = (
                    role["index_selection_write"] if name.startswith("topk_") else role["candidate_write"]
                )
                if name == "candidate_lengths":
                    writes = False
                shared[name] = [[active if writes else 0, scheduler_capacity]]
            layers.append(
                dict(
                    layer=layer,
                    raw_readonly_untouched_complement={
                        key: complement(extent, intervals[key]) for key, extent in extents.items()
                    },
                    shared_readonly_untouched_rows=shared,
                )
            )
        ranks[str(rank)] = dict(raw_extents=extents, layers=layers)
    return dict(
        complete_rank_metadata=len(payloads) == 16,
        result_dependent_masks=False,
        input_contract_sha256=input_contract_sha256,
        ranks=ranks,
    )


def interval_digest(value, intervals, identity):
    """Hash exact CPU row intervals and their identities in the frozen order."""
    if value.device.type != "cpu" or value.ndim == 0:
        raise ValueError("State preservation requires actual CPU staging with an explicit row axis")
    digest = hashlib.sha256()
    description = dict(
        identity=identity, shape=list(value.shape), dtype=str(value.dtype), intervals=intervals
    )
    header = json.dumps(description, sort_keys=True, separators=(",", ":")).encode()
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    previous = 0
    for begin, end in intervals:
        if not 0 <= previous <= begin < end <= len(value):
            raise ValueError("State preservation intervals overlap or escape actual storage")
        payload = value[begin:end].contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
        previous = end
    return digest.hexdigest()


class NetworkStatePreservation:
    """Retain compact exact complement checks from independently admitted all-rank metadata."""

    def __init__(self, provider, regions_path, regions_sha256, input_contract_sha256):
        path = Path(regions_path)
        if file_sha(path) != regions_sha256:
            raise ValueError("Frozen network state region identity changed")
        self.regions = json.loads(path.read_text())
        if (
            not self.regions["complete_rank_metadata"]
            or self.regions["result_dependent_masks"]
            or self.regions["input_contract_sha256"] != input_contract_sha256
        ):
            raise ValueError(
                "Require actual all-rank regions derived from the identical input before any outputs"
            )
        if set(self.regions["ranks"]) != {str(rank) for rank in range(provider.topology["ep"])}:
            raise ValueError("State preservation must cover every actual allocated rank")
        self.provider, self.regions_sha256 = provider, regions_sha256
        self.pending, self.completed = None, []
        for rank, cases in enumerate(provider.arenas):
            if set(cases) != {"fresh"}:
                raise ValueError("This state contract covers the admitted initial fresh network chunk")
            record = self.regions["ranks"][str(rank)]
            extents = {name: value.numel() for name, value in cases["fresh"].items()}
            if extents != record["raw_extents"]:
                raise ValueError("State regions have a different actual raw allocation")
            if [item["layer"] for item in record["layers"]] != list(range(40)):
                raise ValueError("State preservation lacks an original physical layer")

    def _observe(self, layer):
        result = {}
        for rank, cases in enumerate(self.provider.arenas):
            regions = self.regions["ranks"][str(rank)]["layers"][layer]
            for case, raw in cases.items():
                identity = dict(rank=rank, case=case, layer=layer, regions_sha256=self.regions_sha256)
                for name, value in raw.items():
                    intervals = regions["raw_readonly_untouched_complement"][name]
                    key = f"rank{rank}/{case}/raw/{name}"
                    result[key] = interval_digest(value, intervals, dict(identity, storage=key))
                for name, value in self.provider.shared[rank][case].items():
                    intervals = regions["shared_readonly_untouched_rows"][name]
                    key = f"rank{rank}/{case}/shared/{name}"
                    result[key] = interval_digest(value, intervals, dict(identity, storage=key))
                history = self.provider.history.caches[rank][case]
                key = f"rank{rank}/{case}/singleton_hash_history"
                result[key] = interval_digest(history, [[0, len(history)]], dict(identity, storage=key))
        return result

    def begin_layer(self, layer):
        """Observe every exact complement before this layer can write its own allowed rows."""
        if self.pending is not None or layer != len(self.completed) or not 0 <= layer < 40:
            raise ValueError("State observations require one complete original layer at a time")
        self.pending = (layer, self._observe(layer))

    def finish_layer(self, layer, output):
        """Publish the real before/after result and reject any readonly or untouched mutation."""
        if self.pending is None or self.pending[0] != layer:
            raise ValueError("A complete layer has no matching state observation")
        before, after = self.pending[1], self._observe(layer)
        changed = [key for key in before if after.get(key) != before[key]]
        receipt = dict(
            layer=layer,
            regions_sha256=self.regions_sha256,
            before=before,
            after=after,
            exact=not changed,
            changed=changed,
            scope="Exact raw/shared complements and post-preparation singleton history only",
        )
        write_json(Path(output), receipt)
        if changed:
            raise RuntimeError(
                "A complete network layer changed readonly or untouched state: " + str(changed)
            )
        self.completed.append(layer)
        self.pending = None
        return receipt


DTYPE_BYTES = {"torch.int8": 1, "torch.float16": 2, "torch.bfloat16": 2, "torch.float32": 4}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _cpu_tensor(value, dtype, shape, name):
    _require(isinstance(value, torch.Tensor), name + " must be an already-loaded tensor")
    _require(
        value.device.type == "cpu" and value.dtype == dtype and list(value.shape) == list(shape),
        name + " must retain its captured CPU dtype and shape",
    )
    return value


def _metadata_copy(value):
    if isinstance(value, torch.Tensor):
        _require(value.device.type == "cpu", "Captured metadata must be explicit CPU snapshots")
        return value.clone()
    if isinstance(value, dict):
        return {key: _metadata_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_copy(item) for item in value]
    _require(value is None or isinstance(value, (str, bool, int, float)), "Unexpected native metadata object")
    return value


def _descriptor_planes(descriptor):
    return descriptor if isinstance(descriptor, list) else [descriptor]


def _check_layout(layout):
    _require(set(layout) == {"raw", "views", "aliases"}, "Require captured raw, typed and alias descriptors")
    _require(set(layout["raw"]) == set(layout["aliases"]), "Raw and alias arena coverage differs")
    for key, descriptor in layout["raw"].items():
        _require(
            descriptor["storage_key"] == key and descriptor["dtype"] == "torch.int8",
            "Each fresh allocation must be the original byte arena",
        )
        _require(
            len(descriptor["shape"]) == 1
            and descriptor["stride"] == [1]
            and descriptor["storage_offset_bytes"] == 0,
            "Require complete normalized byte backings",
        )
        _require(
            descriptor["shape"][0] > 0 and descriptor["storage_bytes"] == descriptor["shape"][0],
            "Raw backing extent differs from the captured physical allocation",
        )
    names = set(layout["views"])
    _require(
        names == {name for names in layout["aliases"].values() for name in names},
        "Every original typed cache must retain an alias group",
    )
    for name, descriptor in layout["views"].items():
        for plane in _descriptor_planes(descriptor):
            key, shape, stride = plane["storage_key"], plane["shape"], plane["stride"]
            _require(
                key in layout["raw"] and name in layout["aliases"][key], "Typed plane lost its arena alias"
            )
            _require(
                plane["dtype"] in DTYPE_BYTES
                and len(shape) == len(stride)
                and all(size > 0 for size in shape)
                and all(step > 0 for step in stride),
                "Unsupported physical typed plane geometry",
            )
            size = DTYPE_BYTES[plane["dtype"]]
            offset = plane["storage_offset_bytes"]
            extent = offset + size * (1 + sum((dim - 1) * step for dim, step in zip(shape, stride)))
            _require(
                offset >= 0 and offset % size == 0 and extent <= layout["raw"][key]["shape"][0],
                "Typed plane escapes its captured raw allocation",
            )
    for layer in range(40):
        _require(f"model.layers.{layer}.self_attn.swa_cache" in names, "Missing an original per-layer SWA")
    for source in (2, 8, 14, 20):
        root = f"model.layers.{source}.self_attn."
        _require({root + "long_kv_cache", root + "indexer.k_cache"} <= names, "Missing a real source cache")
        if source != 20:
            _require(root + "compressor.state_cache" in names, "Missing a ratio2 source ring")
    return layout


def core_geometry_issues(layout, role):
    """Describe the existing core's unsupported mappings without changing native aliases."""
    role = resolve_layer_role(role["physical_layer"], role)
    own_name = f"model.layers.{role['physical_layer']}.self_attn.swa_cache"
    own = layout["views"][own_name]
    blocks = own["shape"][0]
    issues = []
    if own["dtype"] != "torch.bfloat16" or own["shape"] != [blocks, 128, 1, 512]:
        issues.append("own_SWA_is_not_the_existing_128row_BF16_ABI")
    if own["stride"] != [65536, 512, 512, 1] or own["storage_offset_bytes"] != 0:
        issues.append("own_SWA_requires_explicit_strided_page_load_store")
    if role["mode"] == "swa":
        return issues
    source_name = f"model.layers.{role['kv_source']}.self_attn.long_kv_cache"
    source = layout["views"][source_name]
    if source["storage_key"] != own["storage_key"]:
        issues.append("source_long_and_own_SWA_are_distinct_real_backings")
    if role["ratio"] == 2:
        if source["shape"] == [blocks, 64, 1, 512]:
            issues.append("native_long_64row_plane_requires_separate_source_read_and_publish_ABI")
        else:
            issues.append("ratio2_long_geometry_has_no_validated_full_network_adapter")
    elif role["mode"] == "full":
        if source != own:
            issues.append("ratio1_producer_requires_separate_long_and_own_SWA_page_views")
    else:
        if source["stride"] != own["stride"] or source["storage_offset_bytes"] != own["storage_offset_bytes"]:
            issues.append("consumer_source_and_own_SWA_have_different_physical_geometry")
        if not 0 < source["shape"][0] < blocks:
            issues.append("native_group_pages_are_not_the_standalone_readonly_source_prefix")
    return issues


class NetworkCacheProvider:
    """Allocate real network aliases once and provide bounded original-layer metadata."""

    def __init__(
        self,
        abi_by_case,
        input_by_case,
        *,
        topology,
        model_config,
        capacity,
        scheduler_capacity,
        allocation_id,
        sources_by_case,
        geometry_only=False,
    ):
        _require(
            topology["tp"] in (1, 2, 4)
            and topology["ep"] == topology["dp"] * topology["tp"]
            and 2 <= topology["ep"] <= 16,
            "Require the actual DP/TP/EP world",
        )
        _require(all(topology[name] == 1 for name in ("pp", "pcp", "dcp")), "Require PP1/PCP1/DCP1")
        _require(
            0 < capacity <= 512 and capacity % 16 == 0 and scheduler_capacity >= capacity,
            "Freeze the bounded executable and actual scheduler capacities",
        )
        _require(
            isinstance(allocation_id, str) and allocation_id and "/" not in allocation_id,
            "Require a fresh task-private product allocation identity",
        )
        _require(
            set(abi_by_case) == set(input_by_case) == set(sources_by_case) and abi_by_case,
            "Every case needs captured ABI, real inputs and source receipts",
        )
        text = model_config.get("text_config", model_config)
        _require(
            text["num_hidden_layers"] == 40
            and text["index_topk"] == 512
            and text["candidate_topk_blocks"] == 2048
            and text["candidate_block_size"] == 8
            and text["engram_layer_ids"] == [1, 14],
            "Require the pinned complete model selection layout",
        )
        self.topology, self.capacity = deepcopy(topology), capacity
        self.geometry_only = geometry_only
        allocation_device = "meta" if geometry_only else "cpu"
        self.scheduler_capacity, self.allocation_id = scheduler_capacity, allocation_id
        self.sources = deepcopy(sources_by_case)
        world = topology["ep"]
        self.arenas, self.views = [{} for _ in range(world)], [{} for _ in range(world)]
        self.layouts = [{} for _ in range(world)]
        self.shared = [{} for _ in range(world)]
        self.versions = [{} for _ in range(world)]
        self.bindings = [{} for _ in range(world)]
        self.geometry = [{} for _ in range(world)]
        self._states = {layer: [{} for _ in range(world)] for layer in range(40)}
        self._seen_chunks, self._chunk = set(), None
        self._completed_layers, self._batches = [], {}
        self._sequence_tokens, self._pending_sequence_tokens = {}, {}
        self._input_by_case = _metadata_copy(input_by_case)
        self._abi_by_case = {}
        for case, payloads in abi_by_case.items():
            _require(
                len(payloads) == len(self.sources[case]) == world, "Every actual rank needs an ABI receipt"
            )
            self._abi_by_case[case] = []
            for rank, payload in enumerate(payloads):
                source = self.sources[case][rank]
                _require(
                    source["rank"] == rank and source["payload_sha256"] and source["report_sha256"],
                    "Bind the actual rank to its captured payload and report identity",
                )
                layout = _check_layout(deepcopy(payload["layout"]))
                _require(
                    set(layout["views"]) == set(payload["metadata"]), "All typed views need original metadata"
                )
                self.layouts[rank][case] = layout
                raw = {
                    key: torch.zeros(descriptor["shape"][0], dtype=torch.int8, device=allocation_device)
                    for key, descriptor in layout["raw"].items()
                }
                self.arenas[rank][case] = raw
                self.views[rank][case] = physical_cache_views(raw, layout)
                complete = dict(
                    topk_indices=torch.empty(
                        (scheduler_capacity, 512), dtype=torch.int32, device=allocation_device
                    ),
                    candidates=torch.full(
                        (scheduler_capacity, 1, 2048), -1, dtype=torch.int32, device=allocation_device
                    ),
                    candidate_lengths=torch.zeros(
                        (scheduler_capacity, 1), dtype=torch.int32, device=allocation_device
                    ),
                    topk_lengths=torch.zeros(
                        (scheduler_capacity, 1), dtype=torch.int32, device=allocation_device
                    ),
                )
                self.shared[rank][case] = complete
                shared = {name: value[:capacity] for name, value in complete.items()}
                versions = self.versions[rank][case] = {}
                first = layout["views"]["model.layers.1.self_attn.swa_cache"]
                self.bindings[rank][case] = dict(
                    owner_layer=1,
                    block_size=128,
                    bound=True,
                    kv_binding_id=f"{allocation_id}:r{rank}:{case}:{first['storage_key']}:L1",
                    kv_blocks=first["shape"][0],
                )
                for layer in range(40):
                    role = resolve_layer_role(layer)
                    own = self.views[rank][case][f"model.layers.{layer}.self_attn.swa_cache"]
                    self._states[layer][rank][case] = dict(
                        raw=raw,
                        layout=layout,
                        shared=shared,
                        versions=versions,
                        cache=own,
                        typed_views=self.views[rank][case],
                        allocation_id=allocation_id,
                        own_swa_owner=layer,
                        shared_history_owner=1,
                    )
                    self.geometry[rank].setdefault(case, {})[layer] = core_geometry_issues(layout, role)
                self._abi_by_case[case].append(
                    {
                        name: _metadata_copy(payload[name])
                        for name in (
                            "metadata",
                            "groups",
                            "positions",
                            "lengths",
                            "query_start_loc",
                            "hash_constants",
                        )
                    }
                )
        self.history = NetworkEngramHistory(self.bindings, geometry_only=geometry_only)

    def _input_coordinates(self, contract, rank):
        _require(contract["topology"] == self.topology, "Real input and product topology differ")
        _require(len(contract["DP_engines"]) == self.topology["dp"], "Missing an actual DP request engine")
        dp_rank = rank // self.topology["tp"]
        engine = contract["DP_engines"][dp_rank]
        _require(engine["dp_rank"] == dp_rank, "DP engine ordering differs from the real rank map")
        requests = engine["requests"]
        ids, positions, lengths, starts, slots = [], [], [], [0], []
        for request in requests:
            tokens, pos = request["token_ids"], request["position_ids"]
            _require(
                len(tokens) == len(pos) > 0 and request["live_token_mask"] == [1] * len(tokens),
                "Masked buckets or zero-query requests need a separate actual hash contract",
            )
            context = request["initial_context_length"]
            _require(
                pos == list(range(context, context + len(tokens))),
                "Actual request positions are not contiguous",
            )
            _require(
                len(request["history_tokens"]) == context, "Input history length differs from the real prefix"
            )
            ids.extend(tokens)
            positions.extend(pos)
            lengths.append(context + len(tokens))
            starts.append(len(ids))
            slots.append(request["request_slot"])
        _require(
            engine["active"] == len(ids) <= self.capacity and len(slots) == len(set(slots)),
            "Actual tokens exceed the executable ABI or request slots are duplicated",
        )
        return ids, positions, lengths, starts, slots

    def _record(self, case, layer, rank, payload, contract, active_by_dp):
        role, tp = resolve_layer_role(layer), self.topology["tp"]
        ids, positions, lengths, starts, request_slots = self._input_coordinates(contract, rank)
        total, requests = len(ids), len(lengths)
        _require(
            torch.equal(
                _cpu_tensor(payload["positions"], torch.int64, [total], "positions"),
                torch.tensor(positions, dtype=torch.int64),
            ),
            "Actual captured positions differ from input",
        )
        _require(
            torch.equal(
                _cpu_tensor(payload["lengths"], torch.int32, [requests], "lengths"),
                torch.tensor(lengths, dtype=torch.int32),
            ),
            "Actual captured lengths differ from input",
        )
        _require(
            torch.equal(
                _cpu_tensor(payload["query_start_loc"], torch.int32, [requests + 1], "query starts"),
                torch.tensor(starts, dtype=torch.int32),
            ),
            "Actual captured query starts differ from input",
        )
        metadata = payload["metadata"]
        own_name = f"model.layers.{layer}.self_attn.swa_cache"
        own, full = metadata[own_name], metadata[own_name]["global_metadata"]
        span = tuple(own["cp_token_range"])
        _require(len(span) == 4 and all(type(value) is int for value in span), "Invalid actual CP range")
        start, end, per_rank, padded = span
        valid = max(0, min(end, total) - min(start, total))
        _require(
            0 <= start <= end
            and end - start == per_rank
            and per_rank * tp == padded
            and end <= padded
            and padded >= total
            and per_rank <= self.capacity,
            "Actual CP physical/valid geometry leaves the current executable ABI",
        )
        _require(
            (total != 0 or span == (0, 0, 0, 0)) and start == rank % tp * per_rank,
            "Empty DP or TP rank-to-token mapping differs from native metadata",
        )
        _require(
            own["num_actual_tokens"] == valid and full["num_actual_tokens"] == total,
            "Actual local/full native token counts disagree",
        )
        table = full["block_table"]
        _cpu_tensor(table, torch.int32, [requests, table.shape[1]], "own block table")
        _require(
            torch.equal(full["query_start_loc"], payload["query_start_loc"])
            and torch.equal(full["seq_lens"], payload["lengths"]),
            "Own native request controls differ",
        )
        expected_names = {own_name}
        if role["kv_source"] is not None:
            root = f"model.layers.{role['kv_source']}.self_attn."
            expected_names.update((root + "long_kv_cache", root + "indexer.k_cache"))
            if role["ratio"] == 2:
                expected_names.add(root + "compressor.state_cache")
        groups = []
        covered = set()
        for group in payload["groups"]:
            names = set(group["names"])
            if not names.intersection(expected_names):
                continue
            _require(
                group["reserved_untouched_block_id"] not in group["live_block_ids"],
                "A reserved untouched page became live",
            )
            covered.update(names.intersection(expected_names))
            groups.append(
                dict(
                    group_id=group["group_id"],
                    layer_names=list(group["names"]),
                    block_table=group["block_table"],
                    original_token_slots=group["slot_mapping"],
                    live_block_ids=list(group["live_block_ids"]),
                    reserved_untouched_block_id=group["reserved_untouched_block_id"],
                )
            )
        _require(covered == expected_names, "Original group metadata does not cover own and source caches")
        for name in expected_names:
            _require(
                tuple(metadata[name]["cp_token_range"]) == span,
                "Layer cache groups disagree on actual CP ranges",
            )
        full_ids = torch.tensor(ids, dtype=torch.int64)
        local_ids = torch.zeros(per_rank, dtype=torch.int64)
        local_ids[:valid].copy_(full_ids[start : start + valid])
        record = dict(
            physical_layer_id=layer,
            role=role,
            input_ids=local_ids,
            full_input_ids=full_ids,
            positions=payload["positions"],
            SP_physical_rows=per_rank,
            SP_valid_rows=valid,
            actual_local_query_rows=valid,
            cp_token_range=span,
            active_by_dp=active_by_dp,
            force_dummy=total == 0,
            valid_token_count=total,
            request_slots=request_slots,
            metadata=dict(
                query_start_loc=payload["query_start_loc"],
                seq_lens=payload["lengths"],
                block_table=table,
                slot_mapping=full["slot_mapping"],
            ),
            cache_layout=self.layouts[rank][case],
            cache_metadata=dict(
                physical_layer_id=layer,
                source_physical_layer_id=role["kv_source"],
                actual_metadata={name: metadata[name] for name in expected_names},
                actual_group_requests=groups,
            ),
        )
        record["request_token_map"] = [
            dict(
                request_slot=request_slots[index],
                token_begin=starts[index],
                token_end=starts[index + 1],
                initial_context_length=lengths[index] - (starts[index + 1] - starts[index]),
            )
            for index in range(requests)
        ]
        return record

    def begin_chunk(
        self, *, epoch, token_map_id, input_by_case=None, metadata_by_case=None, metadata_sources_by_case=None
    ):
        """Replace actual chunk controls and invalidate publications without zeroing persistent bytes."""
        _require(
            type(epoch) is int and epoch >= 0 and isinstance(token_map_id, str) and token_map_id,
            "Require an explicit chunk epoch and request/token identity",
        )
        identity = (epoch, token_map_id)
        _require(
            identity not in self._seen_chunks,
            "A failed or completed chunk cannot be silently replayed in place",
        )
        _require(
            self._chunk is None or self._completed_layers == list(range(40)),
            "Complete or explicitly recover the previous real network chunk before advancing",
        )
        inputs = self._input_by_case if input_by_case is None else _metadata_copy(input_by_case)
        payloads = self._abi_by_case
        if metadata_by_case is not None:
            _require(
                metadata_sources_by_case is not None,
                "Bind replacement actual metadata to new source receipts",
            )
            _require(
                set(metadata_by_case) == set(metadata_sources_by_case),
                "Replacement metadata source coverage differs",
            )
            payloads = {}
            names = ("metadata", "groups", "positions", "lengths", "query_start_loc")
            for case, records in metadata_by_case.items():
                _require(
                    len(records) == len(metadata_sources_by_case[case]) == self.topology["ep"],
                    "Replacement actual metadata needs one source receipt per rank",
                )
                payloads[case] = []
                for rank, payload in enumerate(records):
                    source = metadata_sources_by_case[case][rank]
                    _require(
                        source["rank"] == rank and source["metadata_sha256"],
                        "Unbound replacement metadata identity",
                    )
                    if "layout" in payload:
                        _require(
                            payload["layout"] == self.layouts[rank][case],
                            "A chunk cannot silently replace cache allocation",
                        )
                    payloads[case].append({name: _metadata_copy(payload[name]) for name in names})
        _require(
            set(inputs) == set(payloads) == set(self._abi_by_case),
            "Chunk cases cannot replace allocation ownership",
        )
        batches = {layer: dict(cases={}, epoch=epoch, token_map_id=token_map_id) for layer in range(40)}
        token_maps, pending_sequences = {}, {}
        for case, records in payloads.items():
            _require(
                len(records) == self.topology["ep"], "Every chunk must retain empty as well as active ranks"
            )
            active = [engine["active"] for engine in inputs[case]["DP_engines"]]
            token_maps[case] = []
            for dp_rank, engine in enumerate(inputs[case]["DP_engines"]):
                previous_slots = {
                    slot
                    for old_case, old_dp, slot in self._sequence_tokens
                    if old_case == case and old_dp == dp_rank
                }
                current_slots = {request["request_slot"] for request in engine["requests"]}
                _require(
                    previous_slots <= current_slots,
                    "Request eviction or page reuse needs explicit product-state recovery",
                )
                for request in engine["requests"]:
                    key = (case, dp_rank, request["request_slot"])
                    previous = self._sequence_tokens.get(key, [])
                    _require(
                        request["initial_context_length"] == len(previous)
                        and request["history_tokens"] == previous,
                        "Fresh or continued input history differs from completed product chunks",
                    )
                    pending_sequences[key] = previous + list(request["token_ids"])
            for layer in range(40):
                batches[layer]["cases"][case] = [
                    self._record(case, layer, rank, payload, inputs[case], active)
                    for rank, payload in enumerate(records)
                ]
            for rank, record in enumerate(batches[0]["cases"][case]):
                token_maps[case].append(
                    dict(
                        rank=rank,
                        cp=record["cp_token_range"],
                        request_map=record["request_token_map"],
                        full_token_ids=record["full_input_ids"].tolist(),
                        positions=record["positions"].tolist(),
                    )
                )
        digest = hashlib.sha256(
            json.dumps(token_maps, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for batch in batches.values():
            batch["token_mapping_sha256"] = digest
        for rank in range(self.topology["ep"]):
            for case in payloads:
                self.versions[rank][case].clear()
                for layer in range(40):
                    self._states[layer][rank][case].pop("network_engram", None)
        self._seen_chunks.add(identity)
        self._chunk, self._completed_layers, self._batches = identity, [], batches
        self._pending_sequence_tokens = pending_sequences
        self._metadata_sources = (
            deepcopy(metadata_sources_by_case) if metadata_sources_by_case is not None else self.sources
        )
        return dict(
            epoch=epoch,
            token_map_id=token_map_id,
            token_mapping_sha256=digest,
            selection_bytes_reset=False,
            cache_bytes_reset=False,
            history_bytes_reset=False,
        )

    def batch_for_layer(self, layer_id):
        """Provide actual own/source coordinates without fixture activation fields."""
        resolve_layer_role(layer_id)
        _require(self._chunk is not None, "Begin the actual network chunk before requesting a batch")
        return self._batches[layer_id]

    def hash_batch(self):
        """Provide the original first-SWA request mapping for one model-level hash preparation."""
        return dict(self.batch_for_layer(1), engram_metadata_owner_layer=1)

    def runtime_geometry(self):
        """Derive compile-time buffer extents from all actual rank/request metadata."""
        self.batch_for_layer(0)
        blocks = {
            descriptor["shape"][0]
            for cases in self.layouts
            for layout in cases.values()
            for value in layout["views"].values()
            for descriptor in _descriptor_planes(value)
        }
        _require(len(blocks) == 1, "The existing core requires one uniform physical page count")
        records = [record for rows in self._batches[0]["cases"].values() for record in rows]
        requests = max(1, max(len(record["metadata"]["seq_lens"]) for record in records))
        pages = max(
            group["block_table"].shape[1]
            for rows in self._batches.values()
            for rank_records in rows["cases"].values()
            for record in rank_records
            for group in record["cache_metadata"]["actual_group_requests"]
        )
        return dict(
            capacity=self.capacity,
            cache_blocks=blocks.pop(),
            requests=requests,
            table_pages=pages,
            scheduler_capacity=self.scheduler_capacity,
        )

    def state_handles_for_layer(self, layer_id):
        """Expose original typed cache handles for a future reviewed separate-buffer adapter."""
        resolve_layer_role(layer_id)
        return self._states[layer_id]

    def cache_for_layer(self, layer_id):
        """Reject incompatible core geometry before any layer can mutate product state."""
        _require(not self.geometry_only, "Compile-only cache geometry cannot be used for execution")
        role = resolve_layer_role(layer_id)
        self.batch_for_layer(layer_id)
        problems = [
            (rank, case, issues[layer_id])
            for rank, cases in enumerate(self.geometry)
            for case, issues in cases.items()
            if issues[layer_id]
        ]
        _require(not problems, f"Full-network layer{layer_id} requires reviewed cache geometry: {problems}")
        states = self._states[layer_id]
        if role["engram_slot"] is not None:
            for records in states:
                for state in records.values():
                    prepared = state.get("network_engram")
                    _require(
                        prepared is not None
                        and prepared["shared_history_owner"] == 1
                        and (prepared["epoch"], prepared["token_map_id"]) == self._chunk,
                        "Attach the current singleton model-level Engram preparation before this layer",
                    )
        return states

    def require_network_geometry(self):
        """Stop the whole forward before layer0 when any backbone cache mapping is unsupported."""
        missing = sorted(
            {
                layer
                for cases in self.geometry
                for issues in cases.values()
                for layer, reasons in issues.items()
                if reasons
            }
        )
        _require(
            not missing,
            "Complete network cache geometry is not supported for physical layers " + str(missing),
        )

    def observe_layer(self, layer_id, result):
        """Record completed core calls and preserve cross-layer allocation ownership."""
        _require(
            self._chunk is not None and layer_id == len(self._completed_layers),
            "Network completion must follow original physical order",
        )
        _require(
            {"hidden", "pre_mix", "states"} <= set(result),
            "The complete core did not return its required outputs",
        )
        role = resolve_layer_role(layer_id)
        for rank in range(self.topology["ep"]):
            for case in self._batches[layer_id]["cases"]:
                versions = self.versions[rank][case]
                for name, writes, owner in (
                    ("kv", role["kv_write"], role["kv_source"]),
                    ("index", role["index_selection_write"], role["index_source"]),
                    ("candidate", role["candidate_write"], role["candidate_source"]),
                ):
                    if writes:
                        _require(
                            versions.get(name) == (owner, *self._chunk),
                            "A completed writer did not publish its real version",
                        )
        self._completed_layers.append(layer_id)
        if layer_id == 39:
            self._sequence_tokens = self._pending_sequence_tokens

    def hash_constants(self, *, layer_ids, pad_id):
        """Bind captured constants to the separately verified original hash-layout fields."""
        _require(
            tuple(layer_ids) == (1, 14) and type(pad_id) is int,
            "Verify the missing original hash-layout fields",
        )
        baseline = None
        for records in self._abi_by_case.values():
            for payload in records:
                constants = payload["hash_constants"]
                _require(
                    set(constants) == {"token_map", "multipliers", "primes", "offsets"},
                    "Require the actual four captured hash constants",
                )
                if baseline is None:
                    baseline = constants
                else:
                    _require(
                        all(torch.equal(baseline[name], constants[name]) for name in baseline),
                        "Native hash constants differ across rank/case captures",
                    )
        return dict(baseline, layer_ids=list(layer_ids), pad_id=pad_id)

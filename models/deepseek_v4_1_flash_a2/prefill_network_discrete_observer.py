# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Observe existing PyPTO runtime state without adding tensor computation; draft."""

from copy import deepcopy

from .prefill_layer_state import resolve_layer_role
from .prefill_network_shared_observer import begin_shared, end_shared, require, snapshot


def begin_product_observation(provider, layer_id):
    batch = provider.batch_for_layer(layer_id)
    pending = [{} for _ in provider.shared]
    for case, records in batch["cases"].items():
        for rank, original in enumerate(records):
            # Roles are recorded from actual result states after the existing layer.
            # The admitted original source plan fixes writer masks before execution.
            role = resolve_layer_role(layer_id)
            pending[rank][case] = dict(shared=begin_shared(provider.shared[rank][case],
                original["actual_local_query_rows"], index_write=role["index_selection_write"],
                candidate_write=role["candidate_write"]), versions_before=deepcopy(provider.versions[rank][case]))
    return pending


def end_product_observation(provider, layer_id, result, pending):
    batch = provider.batch_for_layer(layer_id)
    records = [{} for _ in provider.shared]
    for case, originals in batch["cases"].items():
        routing_rows = max(original["SP_physical_rows"] for original in originals)
        for rank, original in enumerate(originals):
            state = result["states"][rank][case]
            require(state["role"]["physical_layer"] == layer_id and state["routing_rows"] == routing_rows,
                    "Actual product router/layer identity differs")
            require(all(name in state for name in ("route_ids", "weights", "received_routes", "returned_routes")),
                    "Existing product state omitted required actual route metadata")
            records[rank][case] = dict(layer=layer_id, rank=rank, epoch=batch["epoch"],
                token_map_id=batch["token_map_id"], query_rows=original["actual_local_query_rows"],
                SP_physical_rows=state["physical"], SP_valid_rows=state["valid"],
                cp_token_range=list(original["cp_token_range"]), source=deepcopy(state["role"]),
                metadata_source=(None if state["role"]["mode"] == "swa" else
                    dict(physical_layer_id=original["cache_metadata"]["physical_layer_id"],
                         source_physical_layer_id=original["cache_metadata"]["source_physical_layer_id"])),
                versions_before=pending[rank][case]["versions_before"],
                versions_after=deepcopy(provider.versions[rank][case]),
                router=dict(route_ids=snapshot(state["route_ids"][:routing_rows]),
                    route_weights=snapshot(state["weights"][:routing_rows]), route_rows=routing_rows,
                    inactive_route_ids=snapshot(state["route_ids"][routing_rows:]),
                    inactive_route_weights=snapshot(state["weights"][routing_rows:]),
                    routing_input_token_ids=snapshot(state["token_ids"]),
                    hook="Existing result.states; PyPTO router outputs unchanged", tensor_computation_added=False),
                received_routes=snapshot(state["received_routes"]), returned_routes=snapshot(state["returned_routes"]),
                shared=end_shared(provider.shared[rank][case], pending[rank][case]["shared"]))
    return records


def product_engram_observation(layer_id, rank, case, observation):
    total, physical = observation["total"], observation["physical"]
    require(layer_id in (1, 14), "Require one original Engram lookup slot")
    return dict(layer=layer_id, rank=rank, case=case, total=total, physical=physical,
        valid=observation["valid"], hashing=observation["hashing"],
        hash_ids=snapshot(observation["hash_ids"][:total]),
        token_mask=snapshot(observation["token_mask"][:total]),
        lookup_full=snapshot(observation["lookup_full"][:total]),
        lookup_local=snapshot(observation["lookup_local"][:physical]),
        local_mask=snapshot(observation["local_mask"][:physical]),
        inactive=dict(hash_ids=snapshot(observation["hash_ids"][total:]),
            token_mask=snapshot(observation["token_mask"][total:]),
            lookup_full=snapshot(observation["lookup_full"][total:]),
            lookup_local=snapshot(observation["lookup_local"][physical:]),
            local_mask=snapshot(observation["local_mask"][physical:])),
        hook="Existing network Engram layout outputs before capture_nodes=False discards full lookup",
        tensor_computation_added=False)

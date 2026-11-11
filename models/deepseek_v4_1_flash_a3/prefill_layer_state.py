# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host ownership and fixture adaptation for the shared in-memory Prefill core."""

from functools import lru_cache
import json
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _layer_plan():
    with Path(__file__).with_name("prefill_layer_plan.json").open(encoding="utf-8") as stream:
        plan = json.load(stream)
    roles = plan["layers"]
    if [role["physical_layer"] for role in roles] != list(range(40)):
        raise ValueError("Require the complete original physical-layer plan")
    return roles


def resolve_layer_role(layer_id, role=None):
    """Resolve original pure-data semantics without importing a reference model."""
    if type(layer_id) is not int or not 0 <= layer_id < 40:
        raise ValueError("Require an original physical layer in 0..39")
    expected = _layer_plan()[layer_id]
    if role is not None and any(role.get(name) != value for name, value in expected.items()):
        raise ValueError("Supplied role differs from the original physical-layer plan")
    return dict(expected)


def fixture_live_state(contract, role, epoch, token_map_id):
    """Initialize once from an independent fixture; never use this adapter in a chain."""
    world = contract["topology"]["ep"]
    live = [{} for _ in range(world)]
    for case, records in contract["cases"].items():
        for rank, original in enumerate(records):
            state = {"versions": {}}
            if role["mode"] == "swa":
                state["cache"] = original["cache"].clone()
            else:
                state["raw"] = {key: value.clone() for key, value in original["raw_cache_before"].items()}
                state["layout"] = original["cache_layout"]
                state["shared"] = {}
                for name, encoded in original["shared_state_initial"].items():
                    dtype = getattr(torch, encoded["dtype"].split(".")[-1])
                    value = torch.full(encoded["shape"], encoded["fill"], dtype=dtype)
                    value[:len(encoded["prefix"])].copy_(encoded["prefix"])
                    state["shared"][name] = value
                for name, owner in (("kv", role["kv_source"]), ("index", role["index_source"]),
                                    ("candidate", role["candidate_source"])):
                    if owner is not None:
                        state["versions"][name] = (owner, epoch, token_map_id)
            if role["engram_slot"] is not None:
                state["hash_cache"] = original["hash_cache"].clone()
                state["hash_owner"] = role["physical_layer"]
            live[rank][case] = state
    return live


def validate_live_state(role, original, live, epoch, token_map_id):
    """Reject stale selections and missing actual arenas before any layer mutation."""
    if role["engram_slot"] is not None:
        prepared = live.get("network_engram")
        if prepared is not None:
            if (prepared["shared_history_owner"] != 1 or prepared["physical_layer"] != role["physical_layer"]
                    or prepared["slot"] != role["engram_slot"] or prepared["epoch"] != epoch
                    or prepared["token_map_id"] != token_map_id):
                raise ValueError("The network layer requires its current slot from shared model-level history")
        elif "hash_cache" not in live or live.get("hash_owner") != role["physical_layer"]:
            raise ValueError("Engram requires its own persistent request hash-cache state")
    if role["mode"] == "swa":
        if "cache" not in live:
            raise ValueError("SWA requires its request-owned live cache")
        return
    if not all(name in live for name in ("raw", "layout", "shared", "versions")):
        raise ValueError("Compressed attention requires actual live backing, layout, selection and owner versions")
    metadata = original["cache_metadata"]
    if metadata.get("physical_layer_id", role["physical_layer"]) != role["physical_layer"]:
        raise ValueError("Cache metadata differs from the requested physical layer")
    if role["mode"] != "full":
        if metadata["source_physical_layer_id"] != role["kv_source"]:
            raise ValueError("Readonly cache metadata differs from the original KV source")
        required = [("kv", role["kv_source"])]
        if role["mode"] != "reindex":
            required.append(("index", role["index_source"]))
        if role["candidate_filter"]:
            required.append(("candidate", role["candidate_source"]))
        for name, owner in required:
            if tuple(live["versions"].get(name, ())) != (owner, epoch, token_map_id):
                raise ValueError("Missing or stale live " + name + " ownership for this token map")
    if role["candidate_filter"]:
        from .prefill_indexer_runtime import validate_live_refresh_candidates

        validate_live_refresh_candidates(original, live["shared"], role["kv_source"])


def publish_live_versions(role, live, epoch, token_map_id):
    """Publish owner versions only after the layer has completed its real computation."""
    for name, writes, owner in (("kv", role["kv_write"], role["kv_source"]),
                                ("index", role["index_selection_write"], role["index_source"]),
                                ("candidate", role["candidate_write"], role["candidate_source"])):
        if writes:
            live["versions"][name] = (owner, epoch, token_map_id)


def invalidate_live_versions(role, live):
    """Clear pending publications so a failed retry cannot retain a valid owner tag."""
    for name, writes in (("kv", role["kv_write"]), ("index", role["index_selection_write"]),
                         ("candidate", role["candidate_write"])):
        if writes:
            live["versions"].pop(name, None)


class LayerObservations(dict):
    """Retain optional diagnostic tensors without retaining every layer in a network."""

    def __init__(self, enabled):
        super().__init__()
        self.enabled = enabled

    def __setitem__(self, key, value):
        if self.enabled:
            super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        if self.enabled:
            super().update(*args, **kwargs)

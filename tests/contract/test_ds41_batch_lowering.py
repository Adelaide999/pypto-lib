# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check packed native controls, page leases and provider admission on the HOST."""

import json
from dataclasses import asdict

import pytest

torch = pytest.importorskip("torch")
pypto = pytest.importorskip("pypto")
if getattr(pypto, "__pypto_stub__", False):
    pytest.skip("Real PyPTO imports are required", allow_module_level=True)

from models.deepseek_v4_1_flash_a3.cache_pool import cache_capacity_from_hbm, cache_page_bytes
from models.deepseek_v4_1_flash_a3.config import FLASH
from models.deepseek_v4_1_flash_a3.metadata import BatchMetadata, RequestHandle, RequestQuery, RequestSlice
from models.deepseek_v4_1_flash_a3.prefill_prepare import PackedTokenChunkInputs, cache_templates
from models.deepseek_v4_1_flash_a3.state import NetworkCacheProvider


def make_owner(tmp_path, *, tp=4, context=1024, sequences=4):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = json.loads(json.dumps(dict(text_config=asdict(FLASH))))
    (checkpoint / "config.json").write_text(json.dumps(model))
    cache = dict(cache_ABI=cache_templates({"token_map": torch.arange(16)}), engram_contracts={})
    config = dict(model_config=model, topology=dict(dp=16 // tp, tp=tp, ep=16, pp=1, pcp=1, dcp=1),
                  runtime_limits=dict(max_num_seqs=sequences, max_seqs_per_dp=sequences, max_num_batched_tokens=64))
    return PackedTokenChunkInputs(checkpoint=checkpoint, cache=cache, config=config,
                                  output=tmp_path / "prepared", context_capacity=context)


def provider_for(owner, chunk):
    return NetworkCacheProvider({"fresh": chunk["cache"]["cache_ABI"]},
        {"fresh": chunk["inputs"]["input_contract"]}, topology=owner.topology,
        model_config=owner.config["model_config"], capacity=owner.config["capacity"], scheduler_capacity=8192,
        allocation_id="host-contract", sources_by_case={"fresh": chunk["cache"]["cache_ABI_sources"]},
        geometry_only=True)


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_native_controls_are_stable_with_request_omission_and_c2_odd_pairs(tmp_path, tp):
    owner = make_owner(tmp_path, tp=tp)
    first = owner.batch([RequestQuery("a", 0, (1, 2, 3), 5), RequestQuery("b", 0, (4,), 1)], phase="prefill")
    provider = provider_for(owner, first)
    provider.begin_chunk(epoch=0, token_map_id=first["contract"]["input_contract_sha256"])
    geometry = provider.runtime_geometry()
    record = provider.batch_for_layer(2)["cases"]["fresh"][0]
    assert record["metadata"]["query_start_loc"].tolist() == [0, 3, 4, 4, 4]
    assert record["metadata"]["seq_lens"].tolist() == [3, 1, 0, 0]
    compressed = next(value["global_metadata"] for name, value in
                      record["cache_metadata"]["actual_metadata"].items() if name.endswith("long_kv_cache"))
    assert compressed["cache_seq_lens"].tolist() == [1, 0, 0, 0]
    assert compressed["cmp_residual"].tolist() == [1, 1, 0, 0]
    assert compressed["slot_mapping"][:, 1].tolist() == [-1, 0, -1, -1]
    owner.commit()
    # Simulate completed provider publication; arithmetic is tested by operator goldens.
    provider._sequence_tokens = provider._pending_sequence_tokens
    provider._completed_layers = list(range(40))
    a = owner.ledger.owners["a"].handle
    second = owner.batch([RequestQuery("a", 0, (5, 6), 5, a.generation)], phase="prefill")
    provider.begin_chunk(epoch=1, token_map_id=second["contract"]["input_contract_sha256"],
        input_by_case={"fresh": second["inputs"]["input_contract"]},
        metadata_by_case={"fresh": second["cache"]["cache_ABI"]},
        metadata_sources_by_case={"fresh": [dict(rank=row["rank"], metadata_sha256=row["sha256"])
                                           for row in second["metadata_sources"]]})
    assert provider.runtime_geometry() == geometry
    assert ("fresh", 0, owner.ledger.owners["b"].handle.state_slot) in provider._pending_sequence_tokens
    assert second["inputs"]["input_contract"]["output_map"][0]["row"] == 1
    owner.close()


def test_page_exhaustion_rolls_back_all_admission_and_aliases_count_once(tmp_path):
    owner = make_owner(tmp_path)
    layout = owner.layouts[0]
    sizes = cache_page_bytes(layout)
    assert sizes == {"shared": 131072, "index": 131072}
    budget = cache_capacity_from_hbm(owner.layouts, usable_bytes=[10 * (sum(sizes.values()) + 512)] * 16,
        weight_bytes=[0] * 16, scratch_bytes=[0] * 16, communication_bytes=[0] * 16,
        runtime_reserve_bytes=[0] * 16)
    assert budget["page_capacity"] == 8
    free = [list(row) for row in owner.pool.free]
    owner.pool.free = [[] for _ in free]
    with pytest.raises(MemoryError, match="budget exhausted"):
        owner.batch([RequestQuery("a", 0, (1,), 1)], phase="prefill")
    assert not owner.ledger.owners and owner.ledger.pending is None
    assert not owner.pool.leases
    owner.close()


def test_long_prefix_payload_is_bounded_and_reused_pages_require_reset(tmp_path):
    owner = make_owner(tmp_path, context=32)
    for index in range(12):
        handle = owner.ledger.owners.get("a")
        query = RequestQuery("a", 0, (index,), 12, None if handle is None else handle.handle.generation)
        chunk = owner.batch([query], phase="prefill")
        request = chunk["inputs"]["input_contract"]["DP_engines"][0]["requests"][0]
        assert len(request["history_tokens"]) <= 8
        assert request["history_start"] + len(request["history_tokens"]) == index
        owner.commit()
    handle = owner.ledger.owners["a"].handle
    pages = owner.pool.pages(handle)
    resets = []
    owner.release(handle, lambda old, owned: resets.append((old, owned)))
    assert resets == [(handle, pages)]
    next_chunk = owner.batch([RequestQuery("c", 0, (3,), 1)], phase="prefill")
    new = next_chunk["batch_metadata"].requests[0].handle
    assert new.state_slot == handle.state_slot and new.generation == handle.generation + 1
    assert next_chunk["inputs"]["input_contract"]["retired_requests"][0]["generation"] == handle.generation
    owner.close()


def test_swa_page_reuse_retains_every_query_window_without_history_growth(tmp_path):
    owner = make_owner(tmp_path, context=1024)
    handle, pages = None, None
    for step in range(8):
        query = RequestQuery("a", 0, (1,) * 64, 1024, None if handle is None else handle.generation)
        chunk = owner.batch([query], phase="prefill")
        handle = chunk["batch_metadata"].requests[0].handle
        metadata = chunk["cache"]["cache_ABI"][0]["metadata"]["model.layers.1.self_attn.swa_cache"]
        table = metadata["global_metadata"]["block_table"][0]
        retained = range(max(0, step * 64 - 127) // 128, ((step + 1) * 64 + 127) // 128)
        assert len({int(table[index]) for index in retained}) == len(retained)
        owned = owner.pool.pages(handle)[1]
        if pages is not None:
            assert owned == pages
        pages = owned
        assert len(owned) == owner.pool.swa_pages
        owner.commit()
    owner.close()


def test_memory_transport_has_bounded_receipts_and_commit_checks_input_identity(tmp_path):
    owner = make_owner(tmp_path)
    owner.metadata_transport = "memory"
    first = owner.batch([RequestQuery("a", 0, (1,), 2)], phase="prefill")
    assert all(row["transport"] == "memory" and "path" not in row for row in first["metadata_sources"])
    assert not list(owner.output.rglob("*-metadata.pt"))
    with pytest.raises(ValueError, match="identify"):
        owner.commit("another-input")
    owner.commit(first["contract"]["input_contract_sha256"])
    handle = owner.ledger.owners["a"].handle
    second = owner.batch([RequestQuery("a", 0, (2,), 2, handle.generation)], phase="prefill")
    assert second["inputs"]["input_contract"]["cache_page_updates"] == []
    assert second["inputs"]["input_contract"]["predecessor_input_sha256"] == first["contract"]["input_contract_sha256"]
    owner.close()


def test_one_million_context_plans_wide_byte_addresses_with_bounded_swa_pages(tmp_path):
    owner = make_owner(tmp_path, context=1048576, sequences=1)
    row = RequestSlice(RequestHandle("address-check", 0, 0, 0), (1,), 1048575, 1048576, ())
    batch = BatchMetadata(0, "prefill", (row,), owner.plan, 16, 1)
    changes = owner.pool.reserve(batch)
    groups = owner.cache["cache_ABI"][0]["groups"]
    for group, pages in zip(groups, owner.pool.pages(row.handle)):
        if all(name.endswith("swa_cache") for name in group["names"]):
            assert len(pages) == owner.pool.swa_pages
        elif any(name.endswith("long_kv_cache") for name in group["names"]):
            assert len(pages) == 8192
    assert owner.layouts[0]["raw"]["shared"]["storage_bytes"] > 2**31
    owner.pool.rollback(changes)
    assert not owner.pool.leases
    owner.close()


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_complete_network_ir_fits_resident_program_slots(tmp_path, monkeypatch, tp):
    from models.deepseek_v4_1_flash_a3 import network
    from models.deepseek_v4_1_flash_a3.cache import attach_network_cache_geometry

    class HostEntry:
        def __init__(self, function, directory):
            self.artifacts = {}

    monkeypatch.setattr(network, "Executable", HostEntry)
    owner = make_owner(tmp_path, tp=tp, sequences=2)
    chunk = owner.batch([RequestQuery("a", 0, (1, 2, 3), 3)], phase="prefill")
    provider = provider_for(owner, chunk)
    attach_network_cache_geometry(provider)
    provider.begin_chunk(epoch=0, token_map_id=chunk["contract"]["input_contract_sha256"])
    model = dict(owner.config["model_config"], image_token_id=16)
    constants = dict(token_map=torch.arange(16), layer_ids=[1, 14], pad_id=2)
    shards = [dict(head_range=[rank * 24 // tp, (rank + 1) * 24 // tp]) for rank in range(tp)]
    contracts = {layer: dict(model_config=model, hash_constants=constants, engram_shards=shards)
                 for layer in (1, 14)}
    program = network.NetworkProgram(tmp_path / "compiled").compile(provider, model, owner.topology, contracts)
    assert program.ready and len(program.records) <= 64
    assert program.stages.keys() == {
        "embedding", "validation", "engram1", "engram14", "output", *(f"layer{layer}" for layer in range(40))
    }
    assert program.stages["validation"].keys() == {"finite"}
    assert all(program.stages[f"layer{layer}"] for layer in range(40))
    owner.close()

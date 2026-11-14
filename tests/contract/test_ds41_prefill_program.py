# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check program reuse and compile-time admission without compiling or initializing devices."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pypto = pytest.importorskip("pypto")
if getattr(pypto, "__pypto_stub__", False):
    pytest.skip("Real PyPTO IR is required", allow_module_level=True)

from models.deepseek_v4_1_flash_a3 import network as pool
from models.deepseek_v4_1_flash_a3.config import FLASH
from models.deepseek_v4_1_flash_a3.input_pack import make_vocab_embedding
from models.deepseek_v4_1_flash_a3.lm_head import make_output_entries


def test_chunk_reuse_does_not_rebuild_ir_and_rejects_changed_constants(tmp_path, monkeypatch):
    definition = make_vocab_embedding(16, 512, 8, 2)
    builds = []

    def definitions(*args):
        builds.append(args)
        return {"lookup": definition}

    def engram_definitions(*args):
        builds.append(args)
        return {"engram_projection": definition, "engram_gate": definition}

    class HostEntry:
        def __init__(self, function, directory):
            self.artifacts = {}

    monkeypatch.setattr(pool, "Executable", HostEntry)
    monkeypatch.setattr(pool, "make_input_entries", definitions)
    monkeypatch.setattr(pool, "make_network_layer_entries", lambda *args: (definitions(), {}, 16))
    monkeypatch.setattr(pool, "make_prefill_engram_entries", engram_definitions)
    monkeypatch.setattr(pool, "make_output_entries", definitions)
    # This contract probe replaces stage computations; real L2 composition is checked by the model entry.
    monkeypatch.setattr(pool, "make_network_l2_entries", lambda definitions, *args: definitions)
    provider = SimpleNamespace(
        layouts=[], runtime_geometry=lambda: dict(capacity=16), history=SimpleNamespace(cache_rows=128),
        hash_batch=lambda: dict(cases={"fresh": [dict(metadata=dict(block_table=torch.zeros(1, 1)))]}),
    )
    model, topology = dict(text_config={}, image_token_id=0), dict(tp=2)
    contracts = {layer: dict(model_config=dict(text_config={}), hash_constants={"ids": torch.tensor([1, 2])},
                             engram_shards=[]) for layer in (1, 14)}
    program = pool.PrefillProgram(tmp_path / "compiled").compile(provider, model, topology, contracts)
    first_builds = len(builds)
    program.compile(provider, model, topology, contracts)
    assert len(builds) == first_builds
    assert program.summary()["unique_entries"] == 1
    assert program.summary()["stage_bindings"] == 46
    assert program.entries_for_stage("layer39") is program.stages["layer39"]
    with pytest.raises(ValueError, match="does not cover"):
        program.entries_for_stage("missing")
    contracts[1]["hash_constants"]["ids"][0] = 9
    with pytest.raises(ValueError, match="constants changed"):
        program.compile(provider, model, topology, contracts)
    contracts[1]["hash_constants"]["ids"][0] = 1
    provider.layouts.append({"different_geometry": 1})
    with pytest.raises(ValueError, match="geometry"):
        program.compile(provider, model, topology, contracts)


def test_ir_hash_collision_does_not_alias_different_shapes(tmp_path, monkeypatch):
    class HostEntry:
        def __init__(self, function, directory):
            self.artifacts = {}

    monkeypatch.setattr(pool, "Executable", HostEntry)
    monkeypatch.setattr(pool.ir, "structural_hash", lambda value: 1)
    program = pool.PrefillProgram(tmp_path / "compiled")
    first = program.bind("a", {"lookup": make_vocab_embedding(16, 512, 8, 2)})["lookup"]
    same = program.bind("b", {"lookup": make_vocab_embedding(16, 512, 8, 2)})["lookup"]
    changed = program.bind("c", {"lookup": make_vocab_embedding(16, 1024, 8, 2)})["lookup"]
    assert first is same and first is not changed


def test_output_head_specializes_without_tensor_payloads():
    definitions = make_output_entries(16, asdict(FLASH), 4)
    entries = pool.make_network_l2_entries(definitions, 16, "output")
    for function in entries.values():
        assert function.specialize().functions

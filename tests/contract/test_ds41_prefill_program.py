# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check program reuse and compile-time admission without compiling or initializing devices."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pypto = pytest.importorskip("pypto")
if getattr(pypto, "__pypto_stub__", False):
    pytest.skip("Real PyPTO IR is required", allow_module_level=True)

from models.deepseek_v4_1_flash_a3 import prefill_program as pool
from models.deepseek_v4_1_flash_a3.prefill_embedding import make_vocab_embedding


def test_chunk_reuse_does_not_rebuild_ir_and_rejects_changed_constants(tmp_path, monkeypatch):
    definition = make_vocab_embedding(16, 512, 8, 2)
    builds = []

    def definitions(*args):
        builds.append(args)
        return {"lookup": definition}

    class HostEntry:
        def __init__(self, function, directory):
            self.artifacts = {}

    monkeypatch.setattr(pool, "Executable", HostEntry)
    monkeypatch.setattr(pool, "make_input_entries", definitions)
    monkeypatch.setattr(pool, "make_network_layer_entries", lambda *args: (definitions(), {}, 16))
    monkeypatch.setattr(pool, "make_prefill_engram_entries", definitions)
    monkeypatch.setattr(pool, "make_output_entries", definitions)
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
    assert program.summary()["stage_bindings"] == 44
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

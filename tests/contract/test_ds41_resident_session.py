# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise generation allocation, static-read and topology contracts without a device."""

import ctypes
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pypto = pytest.importorskip("pypto")
if getattr(pypto, "__pypto_stub__", False):
    pytest.skip("Real PyPTO device handles are required", allow_module_level=True)

from pypto.runtime import DeviceTensor
from safetensors.torch import save_file

from models.deepseek_v4_1_flash_a3.cache import decode_cache_views
from models.deepseek_v4_1_flash_a3.prefill_prepare import TokenChunkInputs, cache_templates
from models.deepseek_v4_1_flash_a3.prefill_resident import ResidentNetworkRuntime
from models.deepseek_v4_1_flash_a3.runtime_config import ExpertParallelConfig
from models.deepseek_v4_1_flash_a3.weights import ResidentCheckpointOperands


class HostWorker:
    def __init__(self):
        self.live, self.uploads, self.runs = {}, [], 0
        self.closed = False

    def alloc_tensor(self, shape, dtype, init=None):
        value = torch.zeros(shape, dtype=dtype) if init is None else init.clone()
        buffer = SimpleNamespace(base=value.data_ptr(), nbytes=value.numel() * value.element_size(),
                                 tensor=lambda *args: None)
        handle = DeviceTensor(value.data_ptr(), shape, dtype, buffer=buffer)
        self.live[handle.data_ptr] = value
        self.uploads.append(handle.nbytes)
        return handle

    def free_tensor(self, handle):
        del self.live[handle.data_ptr]

    def copy_to(self, destination, source, nbytes):
        ctypes.memmove(destination, source, nbytes)
        self.uploads.append(nbytes)

    def copy_from(self, destination, source, nbytes):
        ctypes.memmove(destination, source, nbytes)

    def run(self, program, *arguments):
        self.runs += 1

    def close(self):
        assert not self.live
        self.closed = True


def test_static_views_and_cache_aliases_keep_their_owner_and_reject_overruns():
    worker = HostWorker()
    with ResidentNetworkRuntime([0], dict(dp=1, tp=1, ep=1), worker_factory=lambda rank: worker) as runtime:
        value = torch.arange(32, dtype=torch.float32)
        runtime.pin(value, [0], readonly=False, name="raw/0/fresh/shared")
        source = runtime.handle(value[4:12], 0)
        aliases = decode_cache_views({"source": source}, {"decode": ("source", (2, 4))})
        view = aliases["decode"]
        assert view.data_ptr == source.data_ptr
        assert view.buffer is source.buffer
        assert view.byte_offset == 16
        with pytest.raises(ValueError, match="exceeds"):
            decode_cache_views({"source": source}, {"decode": ("source", (9,))})
        with pytest.raises(ValueError, match="exceeds"):
            decode_cache_views({"source": source}, {"decode": ("source", (-1,))})
    assert worker.closed


def test_dispatch_reuses_dynamic_buffers_and_never_reuploads_static_or_cache_views():
    worker = HostWorker()
    with ResidentNetworkRuntime([0], dict(dp=1, tp=1, ep=1), worker_factory=lambda rank: worker) as runtime:
        weight, cache = torch.ones(8), torch.zeros(16)
        runtime.pin(weight, [0], readonly=True, name="weight")
        runtime.pin(cache, [0], readonly=False, name="raw/0/fresh/shared")
        abi = dict(params=[dict(shape=[8]), dict(shape=[8]), dict(shape=[8])],
                   effective_directions=["In", "InOut", "InOut"])
        program = object()
        static_uploads = runtime.stats["device_weight_upload_calls"]
        cache_uploads = runtime.stats["cache_h2d_bytes"]
        for _ in range(3):
            runtime.execute(program, 0, [weight, cache[:8], torch.zeros(8)], abi)
        assert runtime.stats["device_weight_upload_calls"] == static_uploads
        assert runtime.stats["cache_h2d_bytes"] == cache_uploads
        assert runtime.stats["dynamic_allocations"] == 1
        assert worker.runs == 3
        weight.add_(1)
        with pytest.raises(ValueError, match="Immutable"):
            runtime.execute(program, 0, [weight, cache[:8], torch.zeros(8)], abi)
    assert worker.closed


def test_resident_checkpoint_reads_are_once_per_name_and_sealed_after_preload(tmp_path):
    save_file({"a": torch.ones(4), "b": torch.zeros(4)}, tmp_path / "model.safetensors")
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(json.dumps(
        dict(weight_map={name: "model.safetensors" for name in ("a", "b")})))
    reader = ResidentCheckpointOperands(tmp_path)
    reader.load("a")
    with pytest.raises(RuntimeError, match="already read"):
        reader.load("a")
    reader.seal()
    with pytest.raises(RuntimeError, match="disabled"):
        reader.load("b")
    assert reader.read_calls == 1
    assert reader.file_open_calls == 1
    assert reader._sources == {}
    assert reader.consumed["a"]["read_calls"] == 1


def test_dp4_tp4_ep16_expert_ownership_is_complete_and_disjoint():
    topology = ExpertParallelConfig(dp=4, tp=4, ep=16)
    owners = [topology.ownership(rank) for rank in range(16)]
    assert [expert for row in owners for expert, _ in row] == list(range(384))
    assert all(len(row) == 24 for row in owners)
    assert all(topology.owner(expert) == rank for rank, row in enumerate(owners) for expert, _ in row)


def test_token_feedback_preserves_reserved_pages_and_original_positions(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = dict(text_config=dict(vocab_size=128))
    (checkpoint / "config.json").write_text(json.dumps(model))
    constants = {"token_map": torch.arange(128)}
    cache = dict(cache_ABI=cache_templates(constants), engram_contracts={})
    config = dict(model_config=model, topology=dict(dp=4, tp=4, ep=16, pp=1, pcp=1, dcp=1))
    request = TokenChunkInputs(checkpoint=checkpoint, cache=cache, config=config,
                               output=tmp_path / "prepared", context_capacity=130)
    first = request.chunk([1, 2])
    second = request.chunk([3])
    assert first["inputs"]["input_contract"]["epoch"] == 0
    actual = second["inputs"]["input_contract"]
    assert actual["epoch"] == 1
    assert actual["predecessor_input_sha256"] == first["contract"]["input_contract_sha256"]
    assert actual["DP_engines"][0]["requests"][0]["position_ids"] == [2]
    assert actual["DP_engines"][0]["requests"][0]["history_tokens"] == [1, 2]
    for old, new in zip(first["cache"]["cache_ABI"], second["cache"]["cache_ABI"]):
        assert old["layout"] == new["layout"]
        assert all(torch.equal(a["block_table"], b["block_table"]) for a, b in zip(old["groups"], new["groups"]))
    request.close()
    with pytest.raises(ValueError, match="live session"):
        request.chunk([4])


def test_generation_preload_is_idempotent_and_both_phases_share_static_pointers(monkeypatch):
    from models.deepseek_v4_1_flash_a3 import resident_weights

    class PreparedWeights:
        def __init__(self, checkpoint, topology, model_config, runtime):
            self.runtime = runtime
            self.reader = SimpleNamespace(read_calls=0, file_open_calls=0, read_seconds=0.0,
                                          consumed={}, preparations={}, frozen=False)
            self.reader.seal = lambda: setattr(self.reader, "frozen", True)
            self.experts = {}
            self.load_seconds = 0.0
            self.value = torch.ones(4)

        def preload(self, progress=None):
            self.reader.read_calls += 1
            self.reader.consumed["weight"] = dict(read_calls=1)
            self.reader.preparations["weight"] = 1
            self.runtime.pin(self.value, range(16), readonly=True, name="weight")

        def close(self):
            pass

    monkeypatch.setattr(resident_weights, "ResidentNetworkWeights", PreparedWeights)
    workers = [HostWorker() for _ in range(16)]
    topology = dict(dp=4, tp=4, ep=16)
    with ResidentNetworkRuntime(list(range(16)), topology, worker_factory=workers.__getitem__) as runtime:
        runtime.load_rotation = lambda *args: (torch.zeros(32, 32), {})
        owner = resident_weights.A3ResidentWeights("checkpoint", topology, {}, runtime)
        owner.preload_once()
        counters = owner.counters()
        owner.preload_once()
        assert owner.counters() == counters
        assert owner.reader.read_calls == 2
        for rank in range(16):
            prefill, decode = owner.prefill_handles[rank], owner.decode.handles(rank)
            assert prefill.keys() == decode.keys()
            assert all(prefill[name] is decode[name] for name in prefill)
        with pytest.raises(RuntimeError, match="cannot read or upload"):
            owner.decode.preload(runtime)
        runtime.stats["device_weight_upload_calls"] += 1
        with pytest.raises(RuntimeError, match="changed"):
            owner.assert_static_unchanged()
        owner.close()
        assert not any(worker.closed for worker in workers)
    assert all(worker.closed for worker in workers)


def test_borrowed_decode_driver_validates_identity_and_leaves_session_open(tmp_path):
    from models.deepseek_v4_1_flash_a3.decode_network import DecodeNetwork, TOPOLOGY

    runtime = SimpleNamespace(devices=list(range(16)))
    weights = SimpleNamespace(initialization_timings={}, assert_static_unchanged=lambda: None)
    session = SimpleNamespace(closed=False, failed=False, contract=dict(topology=TOPOLOGY),
        runtime=runtime, weights=weights, cache_views=[], request_inputs=SimpleNamespace(config={}))
    driver = DecodeNetwork(session=session, runtime=runtime, resident_weights=weights,
                           cache_views=session.cache_views, output=tmp_path)
    with pytest.raises(ValueError, match="cannot replace"):
        DecodeNetwork(session=session, runtime=object(), output=tmp_path)
    with pytest.raises(ValueError, match="devices differ"):
        DecodeNetwork(session=session, devices=list(reversed(range(16))), output=tmp_path)
    driver.close()
    assert not session.closed
    with pytest.raises(RuntimeError, match="open initialized"):
        driver.step(1)


def test_missing_static_handle_is_rejected_before_a_dynamic_upload():
    worker = HostWorker()
    with ResidentNetworkRuntime([0], dict(dp=1, tp=1, ep=1), worker_factory=lambda rank: worker) as runtime:
        program = object()
        runtime.static_parameters[id(program)] = {"weight"}
        abi = dict(params=[dict(name="weight", shape=[8])], effective_directions=["In"])
        with pytest.raises(RuntimeError, match="missing its resident handle"):
            runtime.execute(program, 0, [torch.ones(8)], abi)
        assert worker.uploads == []
        assert worker.runs == 0


def test_exported_weight_and_mutable_cache_handles_dispatch_without_uploads():
    worker = HostWorker()
    with ResidentNetworkRuntime([0], dict(dp=1, tp=1, ep=1), worker_factory=lambda rank: worker) as runtime:
        weight, cache = torch.ones(8), torch.zeros(8)
        runtime.pin(weight, [0], readonly=True, name="weight")
        runtime.pin(cache, [0], readonly=False, name="raw/0/fresh/shared")
        arguments = [runtime.handle(weight, 0), runtime.handle(cache, 0)]
        abi = dict(params=[dict(name="weight", shape=[8]), dict(name="cache", shape=[8])],
                   effective_directions=["In", "InOut"])
        uploads = list(worker.uploads)
        runtime.execute(object(), 0, arguments, abi)
        assert worker.uploads == uploads
        assert runtime.stats["dynamic_allocations"] == 0
        abi["effective_directions"][0] = "InOut"
        with pytest.raises(ValueError, match="immutable resident weight"):
            runtime.execute(object(), 0, arguments, abi)


def test_preload_opens_each_shard_once_and_keeps_loaded_tensors_valid_after_seal(tmp_path):
    save_file({"a": torch.ones(4), "b": torch.zeros(4)}, tmp_path / "model.safetensors")
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(json.dumps(
        dict(weight_map={name: "model.safetensors" for name in ("a", "b")})))
    reader = ResidentCheckpointOperands(tmp_path)
    first, second = reader.load("a"), reader.load("b")
    assert reader.read_calls == 2
    assert reader.file_open_calls == 1
    reader.seal()
    assert torch.equal(first, torch.ones(4))
    assert torch.equal(second, torch.zeros(4))
    reader.close()

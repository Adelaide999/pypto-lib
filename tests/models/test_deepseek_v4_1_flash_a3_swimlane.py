# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU coverage of eight-step swimlane attribution and incomplete captures."""

import csv
import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def swimlane(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "models/deepseek_v4_1_flash_a3"))
    import swimlane

    return swimlane


def write_capture(directory, step, rank):
    directory.mkdir(parents=True)
    base = (step + 1) * 100 + rank
    # One mixed GEMM task with overlapping AIC/AIV blocks and one independent routed task.
    records = {"chip_swimlane_level": 1, "metadata": {"runtime": "tensormap_and_ringbuffer",
               "clock_freq_hz": 1000, "num_cores": 2, "core_types": ["aic", "aiv"]},
               "aicore_tasks": [[0, 2, 1, 1, base + 1, 0, 8], [1, 2, 1, 2, base + 3, 0, 8],
                                [1, 4, 2, base + 6, base + 10, 0, 8]]}
    deps = {"runtime": "tensormap_and_ringbuffer", "edges": [], "tasks": [
        {"task_id": "2", "kernel_ids": [0, 1, -1], "block_num": 1},
        {"task_id": "4", "kernel_ids": [-1, 2, -1], "block_num": 1},
        {"task_id": "7", "kernel_ids": [-1, -1, -1], "block_num": 1}]}
    names = {"callable_id_to_name": {"0": "proj_q_a_aic", "1": "proj_q_a_aiv", "2": "route_sort"}}
    for name, data in (("chip_swimlane_records", records), ("deps", deps), ("name_map", names)):
        (directory / f"{name}.json").write_text(json.dumps(data))
    (directory / "merged_swimlane_0.json").write_text('{"traceEvents": []}')


def test_operator_summary(tmp_path, swimlane):
    for step in range(8):
        for rank in range(16):
            write_capture(tmp_path / swimlane.step_name(step) / f"rank{rank}" / f"d{step * 2}", step, rank)

    devices = list(reversed(range(16)))
    report = swimlane.summarize_swimlanes(tmp_path, devices, owner=4)
    assert report["device"] == 11
    with (tmp_path / "operator_summary.csv").open() as stream:
        rows = {r["category"]: r for r in csv.DictReader(stream)}
    assert len(rows) == 10
    assert float(rows["Matmul / GEMM"]["prefill_task_ms"]) == 106
    assert float(rows["Matmul / GEMM"]["decode_mean_task_ms"]) == 506
    assert float(rows["Matmul / GEMM"]["prefill_core_ms"]) == 209
    assert float(rows["MoE routing"]["decode_mean_task_ms"]) == 4
    assert float(rows["Other"]["prefill_task_ms"]) == 0
    with (tmp_path / "per_rank.csv").open() as stream:
        ranks = list(csv.DictReader(stream))
    assert len(ranks) == 16 * 10
    gemm = next(r for r in ranks if r["rank"] == "15" and r["category"] == "Matmul / GEMM")
    assert float(gemm["decode-07_task_ms"]) == 817


@pytest.mark.parametrize("missing", ["chip_swimlane_records.json", "deps.json"])
def test_incomplete_capture_has_no_summary(tmp_path, missing, swimlane):
    directory = tmp_path / "prefill" / "rank0" / "d9"
    write_capture(directory, 0, 0)
    (directory / missing).unlink()
    with pytest.raises(ValueError, match="rank0"):
        swimlane.summarize_swimlanes(tmp_path, list(range(16)))
    assert not (tmp_path / "operator_summary.csv").exists()


def test_missing_block_on_mixed_task(tmp_path, swimlane):
    directory = tmp_path / "decode-03" / "rank2" / "d0"
    write_capture(directory, 3, 2)
    records = directory / "chip_swimlane_records.json"
    data = json.loads(records.read_text())
    data["aicore_tasks"].pop(1)
    records.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="block count mismatch.*decode-03/rank2"):
        swimlane.dispatch_totals(records, tmp_path, {})


def test_raw_step_reports_use_same_dispatch_graph(tmp_path, swimlane, monkeypatch):
    runner = pytest.importorskip("pypto.runtime.runner")
    converted = []

    def convert(program, dispatch, records, *, func_names, deps_json):
        assert records.parent == deps_json.parent == func_names.parent == dispatch
        converted.append(dispatch)

    monkeypatch.setattr(runner, "_generate_swimlane", convert)
    for mode in ("prefill", "decode"):
        program = tmp_path / ".programs" / mode / "next_levels" / f"l2_{mode}_fwd"
        program.mkdir(parents=True)
        (program / "kernel_config.py").write_text(
            'KERNELS = [{"func_id": 0, "name": "proj_q_a_aic"}, {"func_id": 1, "name": "proj_q_a_aiv"},'
            ' {"func_id": 2, "name": "route_sort"}]\n'
        )
    for step in range(8):
        dispatch = tmp_path / swimlane.step_name(step) / "rank0" / f"d{step}"
        write_capture(dispatch, step, 0)
        (dispatch / "name_map.json").unlink()
        (dispatch / "merged_swimlane_0.json").unlink()
    result = swimlane.summarize_swimlanes(tmp_path, [0], convert=True)
    assert len(converted) == 8
    assert result["categories"][swimlane.CATEGORIES.index("MoE routing")]["decode_mean_task_ms"] == 4


def test_classifier_precedence(tmp_path, swimlane):
    kernels = ("mhc_linear", "hc_post", "c1a_sparse_attn_aic", "proj_idx_q_aic", "routed_swiglu",
               "dispatch_scale_pack", "proj_q_a_aic", "dynamic_int8_quant", "moe_output_zero", "prefill_tp_wait")
    assert tuple(swimlane.kernel_category(name, tmp_path) for name in kernels) == swimlane.CATEGORIES


@pytest.mark.parametrize("profile", [False, True])
def test_generate_eight_tokens_with_one_dispatch_each(tmp_path, swimlane, monkeypatch, profile):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pypto.runtime.distributed_runner")
    from generation import DistributedWorker, SwimlaneWorker, RunConfig, DistributedConfig
    import replay

    (tmp_path / "tokenizer_config.json").write_text('{"eos_token":"eos"}')
    devices = list(range(16))
    config = RunConfig(distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    worker = object.__new__(SwimlaneWorker if profile else DistributedWorker)
    monkeypatch.setattr(worker, "_drain_dispatch_handles", lambda: None)
    submissions = []

    def submit(program, *args, config):
        native, _, two_pass = worker._prepare_dispatch_config(program, config)
        assert not two_pass
        assert native.enable_dep_gen == profile
        assert native.enable_chip_swimlane == int(profile)
        if profile:
            assert Path(native.output_prefix) == tmp_path / swimlane.step_name(len(submissions))
        submissions.append(program.mode)

    monkeypatch.setattr(worker, "run", submit)
    monkeypatch.setattr(worker, "copy_from", lambda dst, src, size, **kw: ctypes.memset(dst, 0, size))
    monkeypatch.setattr(replay, "prepare_resident_inputs", lambda *a, **kw: {"logit_rows": torch.zeros(1)})
    monkeypatch.setattr(replay, "prepare_decode_inputs", lambda inputs: inputs)
    monkeypatch.setattr(replay, "route_distribution", lambda *a: {})
    monkeypatch.setattr(replay, "M", SimpleNamespace(vocab_size=10))
    programs = [SimpleNamespace(mode=mode, platform="a2a3", output_dir=tmp_path / mode / "artifact",
                                _distributed_config=config.distributed_config) for mode in ("prompt", "decode")]
    ranks = [[{"logits": SimpleNamespace(data_ptr=0), "sampled_ids": SimpleNamespace(data_ptr=0)}
              for _ in devices] for _ in programs]
    counts = {name: torch.zeros(16) for name in ("attention_num_tokens", "moe_num_tokens")}
    report = {}
    replay.run_request(
        worker, programs, [{}, counts], [[], []], ranks,
        SimpleNamespace(tensor_layouts={}, checkpoint=SimpleNamespace(root=tmp_path)), None,
        SimpleNamespace(token_to_id=lambda token: 9, decode=lambda *a, **kw: "result"),
        {"dp_rank": 0, "input_ids": [1] * 8192, "request_id": "test"}, [None] * 16, None,
        config, SimpleNamespace(), report, profile=profile, profile_dir=tmp_path if profile else None,
    )
    assert submissions == ["prompt"] + ["decode"] * 7
    assert report["profiled"] == profile
    assert len(report["engine_seconds"]) == 8
    assert (tmp_path / "prefill").exists() == profile
    assert config.enable_chip_swimlane == 0 and not config.enable_dep_gen

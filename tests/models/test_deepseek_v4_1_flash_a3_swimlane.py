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


@pytest.mark.parametrize("missing", ["chip_swimlane_records.json", "deps.json", "merged_swimlane_0.json"])
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


def test_classifier_precedence(tmp_path, swimlane):
    kernels = ("mhc_linear", "hc_post", "c1a_sparse_attn_aic", "proj_idx_q_aic", "routed_swiglu",
               "dispatch_scale_pack", "proj_q_a_aic", "dynamic_int8_quant", "moe_output_zero", "prefill_tp_wait")
    assert tuple(swimlane.kernel_category(name, tmp_path) for name in kernels) == swimlane.CATEGORIES


def test_only_timing_pass_commits_resident_state(swimlane, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pypto.runtime.distributed_runner")
    from generation import DistributedWorker, ParamDirection, SwimlaneWorker
    from pypto.runtime import DeviceTensor, StackedDeviceTensor

    worker = object.__new__(SwimlaneWorker)
    values = [torch.tensor([10]), torch.tensor([20])]
    shards = [DeviceTensor(v.data_ptr(), v.shape, v.dtype) for v in values]
    tensors = {"state": StackedDeviceTensor(shards, (2, 1), (0, 1))}
    state = {"param_infos": [SimpleNamespace(name="state", direction=ParamDirection.InOut),
                             SimpleNamespace(name="weights", direction=ParamDirection.In)]}

    def dispatch(*args):
        for value in values:
            value.add_(1)

    def copy(dst, src, nbytes, *, worker_id):
        ctypes.memmove(dst, src, nbytes)

    monkeypatch.setattr(DistributedWorker, "_dispatch_prepared", dispatch)
    monkeypatch.setattr(worker, "copy_from", copy)
    monkeypatch.setattr(worker, "copy_to", copy)
    worker._dispatch_prepared(state, tensors, SimpleNamespace(enable_dep_gen=True, enable_chip_swimlane=0))
    assert [v.item() for v in values] == [10, 20]
    worker._dispatch_prepared(state, tensors, SimpleNamespace(enable_dep_gen=False, enable_chip_swimlane=1))
    assert [v.item() for v in values] == [11, 21]

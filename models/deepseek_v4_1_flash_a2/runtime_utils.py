# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host identity helpers for native operator subprocesses.

Device allocation belongs to the caller. These helpers do not reserve a device
or launch a persistent process.
"""

import hashlib
import os
from pathlib import Path
import json
import subprocess
import sys

import torch
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig


def file_sha(path):
    """Hash a source, runtime library, or staging artifact."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):
    """Hash actual contiguous CPU tensor bytes, including BF16 payloads."""
    import torch

    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def validate_device(settings):
    """Require one explicit physical device mapped to logical device zero."""
    device = settings["physical_device"]
    if type(device) is not int or device < 0:
        raise ValueError("physical_device must be a nonnegative integer")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(device):
        raise RuntimeError("ASCEND_RT_VISIBLE_DEVICES must select exactly physical_device")
    return os.getpid()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


class Executable:
    """Compile one operator definition and retain its device output captures."""

    def __init__(self, function, directory, *, diagnostics=True):
        self.diagnostics = diagnostics
        print({"stage": "compile", "operator": directory.name}, flush=True)
        config = RunConfig(arch=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                           save_kernels=True, save_kernels_dir=str(directory))
        reuse_manifest = os.environ.get("DS41_COMPILED_REUSE_MANIFEST")
        reuse = {}
        if not diagnostics and reuse_manifest:
            key = str(directory.relative_to(Path(os.environ["DS41_JOB_ROOT"]) / "product"))
            reuse = json.loads(Path(reuse_manifest).read_text()).get(key, {})
        if reuse:
            from pypto.ir import CompiledProgram

            metadata = Path(reuse["path"])
            if file_sha(metadata) != reuse["sha256"]:
                raise ValueError("Admitted unchanged compiled program metadata changed")
            self.compiled = CompiledProgram.from_dir(metadata.parent)
            directory = metadata.parent
            print(dict(stage="reuse_unchanged_compiled_program", directory=str(directory)), flush=True)
        else:
            self.compiled = function.compile(config=config)
        self.artifacts = {}
        for pattern in ("*.pto", "*.cpp", "*.so", "kernel_config.py"):
            self.artifacts.update({str(path): file_sha(path) for path in sorted(directory.rglob(pattern))})
        if not any(path.endswith(".pto") for path in self.artifacts):
            raise RuntimeError("PyPTO compilation produced no retained PTO artifact")
        print({"stage": "compiled", "operator": directory.name, "directory": str(directory)}, flush=True)

    def execute(self, arguments, outputs, rank, responsibility, capture, report, memory):
        if not self.diagnostics:
            config = RunConfig(arch=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD, device_id=rank)
            self.compiled(*arguments, config=config)
            return
        responsibilities = [responsibility] if isinstance(responsibility, str) else list(responsibility)
        responsibility = "/".join(responsibilities)
        config = RunConfig(arch=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD, device_id=rank)
        print({"stage": responsibility, "rank": rank, "capture": str(capture)}, flush=True)
        memory.sample(rank, responsibility + "/execute/before")
        with memory.interval(rank, "execute", responsibility):
            self.compiled(*arguments, config=config)
        memory.sample(rank, responsibility + "/execute/after")
        descriptions = [{"shape": list(arg.shape), "dtype": str(arg.dtype)}
                        if isinstance(arg, torch.Tensor) else arg for arg in arguments]
        self.record(descriptions, outputs, rank, responsibilities, capture, report)

    def record(self, arguments, outputs, rank, responsibilities, capture, report):
        """Retain actual outputs and artifacts after a completed PyPTO invocation."""
        if not self.diagnostics:
            return
        torch.save(outputs, capture)
        record = next((item for item in report["kernels"] if item["artifacts"] == self.artifacts), None)
        if record is None:
            record = {"responsibilities": [], "artifacts": self.artifacts, "outputs": {}, "invocations": []}
            report["kernels"].append(record)
        for role in responsibilities:
            if role not in record["responsibilities"]:
                record["responsibilities"].append(role)
        record["outputs"][str(capture)] = file_sha(capture)
        record["invocations"].append({"logical_rank": rank, "capture": str(capture), "arguments": arguments})


def communicate(stage, packets, output, runtime_path, *, collective=None, group=None):
    directory = output / stage
    directory.mkdir()
    if collective is not None:
        settings = json.loads(runtime_path.read_text())
        settings["transport"] = {"collective": collective, "group": group, "stage": stage}
        runtime_path = directory / "transport-config.json"
        write_json(runtime_path, settings)
    processes, logs = [], []
    try:
        for rank, packet in enumerate(packets):
            torch.save(packet, directory / f"rank-{rank}-inputs.pt")
        for rank in range(len(packets)):
            command = [sys.executable, "-u", "-m", "models.deepseek_v4_1_flash_a2.moe_transport",
                       "--directory", str(directory), "--runtime-config", str(runtime_path), "--rank", str(rank)]
            write_json(directory / f"rank-{rank}-command.json", command)
            log = (directory / f"rank-{rank}-transport.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        for rank, process in enumerate(processes):
            status = process.wait(timeout=600)
            if status:
                raise RuntimeError(f"HCCL byte transport rank {rank} exited {status}; inspect {directory}")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        for log in logs:
            log.close()
    results, receipts = [], []
    for rank in range(len(packets)):
        receipt = json.loads((directory / f"rank-{rank}-report.json").read_text())
        path = directory / f"rank-{rank}-outputs.pt"
        if receipt["outputs_sha256"] != file_sha(path):
            raise RuntimeError("HCCL received output identity changed")
        if receipt["inputs_sha256"] != file_sha(directory / f"rank-{rank}-inputs.pt"):
            raise RuntimeError("HCCL sent input identity changed")
        results.append(torch.load(path, map_location="cpu", weights_only=True))
        receipts.append(receipt)
    return results, receipts

def loaded_source_files(runtime):
    root = Path(runtime["source_root"])
    model = root / "models/deepseek_v4_1_flash_a2"
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path is not None:
            path = Path(path).resolve()
            if path.is_relative_to(model) and path.suffix == ".py":
                runtime["source_files"][str(path.relative_to(root))] = file_sha(path)

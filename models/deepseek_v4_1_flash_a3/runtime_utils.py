# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared compiler artifacts, input identities and runtime JSON helpers."""

import hashlib
import json
import sys
from pathlib import Path
import torch
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig


def file_sha(path):
    """Hash a source, runtime library, or staging artifact."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):
    """Hash actual contiguous CPU tensor bytes, including BF16 payloads."""

    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


class Executable:
    """Compile one production operator definition and retain its compiler artifacts."""

    def __init__(self, function, directory):
        print({"stage": "compile", "operator": directory.name}, flush=True)
        config = RunConfig(arch=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                           save_kernels=True, save_kernels_dir=str(directory))
        self.compiled = function.compile(config=config)
        self.artifacts = {}
        for pattern in ("*.pto", "*.cpp", "*.so", "kernel_config.py"):
            self.artifacts.update({str(path): file_sha(path) for path in sorted(directory.rglob(pattern))})
        if not any(path.endswith(".pto") for path in self.artifacts):
            raise RuntimeError("PyPTO compilation produced no retained PTO artifact")
        print({"stage": "compiled", "operator": directory.name, "directory": str(directory)}, flush=True)


def loaded_source_files(runtime):
    root = Path(runtime["source_root"])
    model = root / "models/deepseek_v4_1_flash_a3"
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path is not None:
            path = Path(path).resolve()
            if path.is_relative_to(model) and path.suffix == ".py":
                runtime["source_files"][str(path.relative_to(root))] = file_sha(path)

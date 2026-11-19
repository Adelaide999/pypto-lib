# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile the A3 prefill/decode ABI and validate artifact compatibility."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
from golden import TensorSpec, run
from pypto.ir import DistributedConfig

import config as C


TOPOLOGY = {"world": 16, "dp": 4, "tp": 4, "ep": 16, "local_experts": 24}
HISTORY_CAPACITY = 8200


def build_specs(function, *, mode, capacity):
    """Construct compile shapes without allocating the model's weight tensors."""
    dtype_map = {"FP32": torch.float32, "BF16": torch.bfloat16, "INT8": torch.int8,
                 "UINT8": torch.uint8, "INT32": torch.int32, "INT64": torch.int64}
    pages = (HISTORY_CAPACITY + 127) // 128 + 1
    dimensions = {
        "FWD_ORI_BLOCKS_DYN": 40 * pages, "FWD_CMP_BLOCKS_DYN": 4 * pages,
        "FWD_INDEX_BLOCKS_DYN": 4 * pages, "FWD_STATE_BLOCKS_DYN": 3,
        "FWD_C2A_TABLE_DYN": 33, "FWD_C1A_TABLE_DYN": 65,
        "PREFILL_HISTORY_PAGES": pages, "PREFILL_C2A_TABLE": 33, "PREFILL_C1A_TABLE": 65,
        "ROPE_ROWS_DYN": HISTORY_CAPACITY, "LOCAL_T_DYN": capacity, "T_DYN": capacity,
        "Q_START_DYN": 2, "STATE_BLOCKS_DYN": 1, "B_DYN": 1,
        "INDEX_POOL_BLOCKS_DYN": 1056,
    }

    def extent(dim):
        if isinstance(dim, int):
            return dim
        label = repr(dim)
        for marker, value in sorted(dimensions.items(), key=lambda item: len(item[0]), reverse=True):
            if marker in label:
                return value
        raise ValueError(f"unknown {mode} dimension {dim!r}")

    specs = []
    for name, parameter in inspect.signature(function._func).parameters.items():
        annotation = parameter.annotation
        dtype_name = str(annotation.dtype).split(".")[-1].upper().replace("BFLOAT16", "BF16").replace("FLOAT32", "FP32")
        if dtype_name not in dtype_map:
            raise ValueError(f"unsupported A3 dtype {annotation.dtype} for {name}")
        shape = [extent(dim) for dim in annotation.shape]
        specs.append(TensorSpec(name, shape, dtype_map[dtype_name]))
    return specs


def manifest(mode, capacity):
    """Version the shared weight, cache and quantization ABI."""
    root = Path(__file__).parent
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(root.glob("*.py"))
               if path.name in {"config.py", "driver.py"} or "@pl.jit" in path.read_text()}
    return {"format": "deepseek-v41-a3-artifact-v1", "platform": "a2a3", "mode": mode,
            "topology": TOPOLOGY, "capacity": capacity, "history_capacity": HISTORY_CAPACITY,
            "weight_layout": C.WEIGHT_LAYOUT_VERSION, "cache_layout": C.CACHE_LAYOUT_VERSION,
            "quantization": "W8A8_DYNAMIC", "sources": sources}


def load_manifest(directory, *, mode, capacity):
    """Reject artifacts from a different topology, dtype ABI or source snapshot."""
    path = Path(directory) / "a3_manifest.json"
    actual = json.loads(path.read_text())
    expected = manifest(mode, capacity)
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"artifact {path} has incompatible {key}")
    return actual


def main(function, *, mode, capacity):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3"], default="a2a3")
    parser.add_argument("-d", "--device", "--devices", dest="devices", default=",".join(map(str, range(16))))
    parser.add_argument("--tp", type=int, choices=[4], default=4)
    parser.add_argument("--dp", type=int, choices=[4], default=4)
    parser.add_argument("--ep", type=int, choices=[16], default=16)
    parser.add_argument("--experts-per-rank", type=int, choices=[24], default=24)
    parser.add_argument("--capacity", type=int, default=capacity)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--decode-runtime-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--input-ids-json", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--teacher-forced", action="store_true")
    parser.add_argument("--audit-caches", action="store_true")
    parser.add_argument("--save-logits", type=Path)
    parser.add_argument("--validate-layers", default="")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--validate-teacher-forced", action="store_true")
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--measured-requests", type=int, default=1)
    profiling = parser.add_mutually_exclusive_group()
    profiling.add_argument("--profile-last-decode", action="store_true")
    profiling.add_argument("--profile-all-steps", type=Path, metavar="DIR",
                           help="capture Level 1 swimlanes for prefill and all seven decode steps in a new directory")
    parser.add_argument("--execution-log", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.profile_all_steps is not None and (args.checkpoint is None or args.validation_only or args.compile_only):
        parser.error("--profile-all-steps requires full checkpoint generation")
    if args.warmup_requests < 0 or args.measured_requests < 1:
        parser.error("warmup requests must be nonnegative; measured requests must be positive")
    if args.validation_only and not args.validate_layers:
        parser.error("--validation-only requires --validate-layers")
    if (args.teacher_forced or args.validate_teacher_forced or args.validate_layers) and not args.reference_dir:
        parser.error("numerical validation requires --reference-dir")
    devices = [int(value) for value in args.devices.split(",")]
    if len(devices) != 16 or len(set(devices)) != 16:
        parser.error("exactly 16 distinct allocated devices are required")
    if args.capacity != capacity:
        parser.error("capacity must match import-time specialization")
    if args.checkpoint is not None:
        if mode != "prefill" or args.runtime_dir is None or args.decode_runtime_dir is None:
            parser.error("generation requires prefill and decode artifact directories")
        from generation import generate

        manifests = (load_manifest(args.runtime_dir, mode="prefill", capacity=8192),
                     load_manifest(args.decode_runtime_dir, mode="decode", capacity=C.DECODE_INPUT_MAX_TOKENS))
        report = generate(args, devices, manifests=manifests)
        if args.execution_log:
            from telemetry import rank_timings

            report["device_timings"] = rank_timings(args.execution_log, devices, dp_rank=report["dp_rank"])
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return
    if not args.compile_only:
        parser.error("use --compile-only, or provide --checkpoint and both compiled artifacts")
    snapshot = manifest(mode, capacity)
    result = run(fn=function, specs=build_specs(function, mode=mode, capacity=capacity),
                 config={"backend_type": "a2a3", "execution_mode": False,
                         "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0),
                         "dump_passes": args.dump_passes, "ring_dep_pool": 131072,
                         "ring_heap": (4 * 1024**3, 4 * 1024**3, 2 * 1024**3, 1024**3)},
                 compile_only=True)
    if not result.passed:
        raise SystemExit(result.error or 1)
    if manifest(mode, capacity) != snapshot:
        raise RuntimeError("model sources changed during compilation; rebuild the artifact")
    path = Path(result.work_dir) / "a3_manifest.json"
    path.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(f"A3_ARTIFACT {result.work_dir}", flush=True)

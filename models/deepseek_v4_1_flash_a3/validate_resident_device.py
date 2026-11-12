# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: manual
# ci: no-sim
# ci: devices=2
"""Exercise resident A3 dispatch and repeated HCCL on two explicitly allocated ranks."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from .prefill_network_runtime import network_parameter_abi
from .rmsnorm import make_prefill_norm
from .prefill_resident import ResidentNetworkRuntime
from .rmsnorm import golden_rms_norm
from .runtime_config import configure_devices
from .runtime_utils import Executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", "--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--compiled", type=Path)
    args = parser.parse_args()
    devices = configure_devices(args.devices.split(","), world=2)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.compiled is None:
        compiled = Executable(make_prefill_norm(16, 5120), args.output / "norm", diagnostics=False).compiled
    else:
        from pypto.ir import CompiledProgram

        compiled = CompiledProgram.from_dir(args.compiled)
    abi = network_parameter_abi(compiled)
    if args.compile_only:
        print(
            json.dumps(
                dict(status="compiled", device_initialization="not_run", compiled=str(compiled.output_dir))
            )
        )
        return
    torch.manual_seed(41)
    weight = torch.randn(5120, dtype=torch.bfloat16)
    topology = dict(dp=1, tp=2, ep=2)
    begin = perf_counter()
    with ResidentNetworkRuntime(devices, topology) as runtime:
        runtime.pin(weight, [0, 1], readonly=True, name="validation norm weight")
        for step in range(2):
            for rank in range(2):
                source = torch.randn(16, 5120, dtype=torch.bfloat16)
                output = torch.zeros_like(source)
                runtime.execute(compiled, rank, [source, weight, output, 16], abi)
                torch.testing.assert_close(output, golden_rms_norm(source, weight), rtol=0.05, atol=0.05)
            values = [
                {
                    "test": {
                        "value": torch.arange(8, dtype=torch.float32).reshape(2, 4) + rank * 10 + step * 100
                    }
                }
                for rank in range(2)
            ]
            gathered, _ = runtime.exchange(f"gather-{step}", values, "all_gather", "ep")
            expected = torch.cat([row["test"]["value"] for row in values])
            for row in gathered:
                torch.testing.assert_close(row["test"]["value"], expected, rtol=0, atol=0)
            swapped, _ = runtime.exchange(f"exchange-{step}", values, "all_to_all", "tp")
            for rank, row in enumerate(swapped):
                expected = torch.stack([source["test"]["value"][rank] for source in values])
                torch.testing.assert_close(row["test"]["value"], expected, rtol=0, atol=0)
            reduced, _ = runtime.exchange(f"reduce-{step}", values, "reduce_scatter", "tp")
            total = sum(source["test"]["value"] for source in values)
            for rank, row in enumerate(reduced):
                torch.testing.assert_close(row["test"]["value"], total[rank : rank + 1], rtol=0, atol=0)
        summary = runtime.summary()
        if summary["weight_bytes"] != [10240, 10240] or summary["worker_initializations"] != 2:
            raise RuntimeError("Resident worker/weight ownership was not preserved across steps")
        if len(runtime.channels) != 2 or summary["transport_calls"] != 6:
            raise RuntimeError("Persistent HCCL groups were not reused")
    report = dict(
        status="PASS",
        scope="Resident norm dispatch and three repeated two-rank collectives",
        devices=devices,
        seconds=perf_counter() - begin,
        runtime=summary,
        full_model_accuracy="not_run",
        full_model_speedup="not_measured",
    )
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

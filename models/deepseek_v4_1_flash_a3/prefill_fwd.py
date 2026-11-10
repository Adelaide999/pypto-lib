# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: manual
"""Compile and execute token embedding, all forty Flash layers and the final vocabulary head."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

import argparse
import json
from pathlib import Path


def compile_prefill(inputs, *, cache, config, output):
    """Compile every original network stage from canonical request/cache geometry.

    This uses metadata and storage-free meta tensors to derive exact shapes.
    It does not read checkpoint payloads, allocate cache backing storage,
    initialize CANN, execute kernels or establish accuracy.
    The returned program can be supplied to ``prefill_fwd`` in this process.
    """
    from .prefill_network_runtime import prepare_network_provider
    from .prefill_program import PrefillProgram

    operands = _merge_inputs(inputs, cache)
    directory = Path(output).resolve()
    provider = prepare_network_provider(operands, config, directory.name, geometry_only=True)
    return PrefillProgram(directory).compile(provider, config["model_config"], config["topology"],
                                             operands["engram_contracts"])


def _merge_inputs(inputs, cache):
    overlap = set(inputs) & set(cache)
    if overlap:
        raise ValueError("Input and cache configuration have duplicate fields: " + ", ".join(sorted(overlap)))
    return dict(inputs, **cache)


def prefill_fwd(inputs, *, weights, cache, config, devices, output, compiled=None, continuation=None):
    """Execute the existing network using explicit checkpoint and cache-state providers.

    ``inputs["input_contract"]`` contains the actual request tokens and metadata.
    Without ``product_input_paths``, embedding and initial HC execute inside
    this forward. Prepared backbone tensors remain supported explicitly.
    ``cache`` supplies the canonical all-rank cache ABI, source descriptors and
    Engram contracts. No historical task release or layer receipt is required.
    """
    from .runtime_config import configure_devices

    configure_devices(devices, world=config["topology"]["ep"])
    from .prefill_network_runtime import run_network_runtime

    return run_network_runtime(
        _merge_inputs(inputs, cache), contract=config, checkpoint=weights,
        devices=devices, output=output, compiled=compiled, continuation=continuation,
    )


# Preserve the published entry name for callers of the original package API.
run_prefill = prefill_fwd


def load_configuration(path):
    """Read paths relative to a JSON configuration without importing the device runtime."""
    path = Path(path).resolve()
    settings = json.loads(path.read_text(encoding="utf-8"))
    required = {"checkpoint", "inputs", "cache_state", "contract", "devices", "output"}
    missing = required - settings.keys()
    if missing:
        raise ValueError("Missing Prefill configuration fields: " + ", ".join(sorted(missing)))
    for name in ("checkpoint", "inputs", "cache_state", "contract", "output"):
        value = Path(settings[name])
        settings[name] = value if value.is_absolute() else (path.parent / value).resolve()
    devices = settings["devices"]
    if (not isinstance(devices, list) or len(devices) != 16 or
            any(type(value) is not int or value < 0 for value in devices) or len(set(devices)) != 16):
        raise ValueError("The current full-network configuration requires 16 distinct physical device IDs")
    return settings


def main():
    """Load an explicit full-network configuration and run the shared product path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("-p", "--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("--devices", help="Comma-separated physical device IDs; overrides the JSON mapping")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-config", action="store_true", help="Validate the configuration without device execution")
    parser.add_argument("--compile-only", action="store_true", help="Compile the complete network without NPU execution")
    args = parser.parse_args()
    settings = load_configuration(args.config)
    if args.devices is not None:
        settings["devices"] = [int(value) for value in args.devices.split(",")]
    if args.checkpoint is not None:
        settings["checkpoint"] = args.checkpoint.resolve()
    if args.output is not None:
        settings["output"] = args.output.resolve()
    # Bind device visibility before loading the compiler, Torch or CANN.
    from .runtime_config import configure_devices, validate_devices

    if args.check_config:
        validate_devices(settings["devices"], world=16)
        print(json.dumps({"configuration": "valid", "execution": "not_run", "platform": args.platform}))
        return
    if args.compile_only:
        validate_devices(settings["devices"], world=16)
    else:
        configure_devices(settings["devices"], world=16)
    import torch

    contract = json.loads(settings["contract"].read_text(encoding="utf-8"))
    inputs = torch.load(settings["inputs"], map_location="cpu", weights_only=True)
    cache = torch.load(settings["cache_state"], map_location="cpu", weights_only=True)
    if args.compile_only:
        program = compile_prefill(inputs, cache=cache, config=contract, output=settings["output"])
        print(json.dumps(program.summary()))
        return
    result = prefill_fwd(inputs, weights=settings["checkpoint"], cache=cache, config=contract,
                         devices=settings["devices"], output=settings["output"])
    print(json.dumps(dict(completed_layers=result["completed_layers"], final_output_files=result["final_output_files"],
                          numerical_acceptance=result["formal_accuracy"])))


if __name__ == "__main__":
    main()

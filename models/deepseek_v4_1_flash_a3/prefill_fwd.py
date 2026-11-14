# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Whole-network Prefill input preparation and public command."""

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"
from .network import NetworkProgram as PrefillProgram, _merge_inputs, prepare_network_provider, run_network_runtime


def compile_prefill(inputs, *, cache, config, output, weights=None):
    """Compile every original network stage from canonical request/cache geometry.

    This uses metadata and storage-free meta tensors to derive exact shapes.
    It does not read checkpoint payloads, allocate cache backing storage,
    initialize CANN, execute kernels or establish accuracy.
    The returned program can be supplied to ``prefill_fwd`` in this process.
    Raw text or token IDs require ``weights`` for checkpoint metadata and use
    the same full-prompt cache geometry as automatic execution.
    """

    if isinstance(inputs, (str, list, tuple)):
        from .prefill_prepare import prepare_token_chunks

        if weights is None:
            raise ValueError("Automatic compile-only inputs require an explicit checkpoint via weights")
        with TemporaryDirectory(prefix="ds41-prefill-compile-") as temporary:
            chunks = prepare_token_chunks(
                inputs,
                checkpoint=weights,
                cache=cache,
                config=config,
                output=Path(temporary) / "prepared",
                chunk_size=config.get("chunk_size", 16),
                dp_rank=config.get("dp_rank", 0),
            )
            try:
                first = next(chunks)
                return compile_prefill(
                    first["inputs"], cache=first["cache"], config=first["contract"], output=output
                )
            finally:
                chunks.close()
    operands = _merge_inputs(inputs, cache)
    directory = Path(output).resolve()
    provider = prepare_network_provider(operands, config, directory.name, geometry_only=True)
    return PrefillProgram(directory).compile(
        provider, config["model_config"], config["topology"], operands["engram_contracts"]
    )


def prefill_fwd(
    inputs,
    *,
    weights,
    cache,
    config,
    devices,
    output,
    compiled=None,
    continuation=None,
    progress=None,
    engram_lookup=None,
    keep_runtime=False,
):
    """Execute the existing network using explicit checkpoint and cache-state providers.

    ``inputs`` accepts raw completion text, 1..2048 token IDs, or prepared
    backbone tensors and their input contract. Automatic inputs use one DP
    engine and chunks of up to 16 tokens; config may set chunk_size/dp_rank.
    Embedding and initial HC execute inside this forward unless explicit
    prepared backbone tensors are supplied.
    ``cache`` supplies the canonical all-rank cache ABI, source descriptors and
    Engram contracts. No historical task release or layer receipt is required.
    ``continuation`` accepts one prepared chunk or a lazy iterable of chunks;
    each chunk follows the preceding input's hash and next epoch.
    ``engram_lookup`` contains host-gathered BF16 rows for both slots.
    Each prepared continuation supplies its own rows in inputs["engram_lookup"].
    The forward performs only device projection/gating on these ready rows.
    Ordinary execution preloads all compute weights and retains workers,
    caches and communication groups until the complete stream ends.
    """
    from .runtime_config import configure_devices

    chunks = None
    request_inputs = None
    transferred = False
    try:
        configure_devices(devices, world=config["topology"]["ep"])
        if isinstance(inputs, (str, list, tuple)):
            from .prefill_prepare import TokenChunkInputs, tokenize_prompt

            if continuation is not None or Path(output).exists():
                raise ValueError(
                    "Automatic Prefill needs a new output directory and owns its continuation stream"
                )
            tokens = tokenize_prompt(inputs, weights, config["model_config"]["text_config"]["vocab_size"])
            chunk_size = config.get("chunk_size", 16)
            if type(chunk_size) is not int or not 1 <= chunk_size <= 16:
                raise ValueError("Chunk size must be in 1..16")
            request_inputs = TokenChunkInputs(
                checkpoint=weights,
                cache=cache,
                config=config,
                output=Path(output) / "prepared",
                context_capacity=config.get("context_capacity", len(tokens)),
                prepare_engram=True,
            )
            chunks = (request_inputs.chunk(tokens[begin:begin + chunk_size], dp_rank=config.get("dp_rank", 0))
                      for begin in range(0, len(tokens), chunk_size))
            first = next(chunks)
            inputs, cache, config = first["inputs"], first["cache"], first["contract"]
            output, continuation = Path(output) / "network", chunks

        if engram_lookup is not None:
            if "engram_lookup" in inputs:
                raise ValueError("Pass host Engram rows once, as an argument or prepared input field")
            inputs = dict(inputs, engram_lookup=engram_lookup)
        result = run_network_runtime(
            _merge_inputs(inputs, cache),
            contract=config,
            checkpoint=weights,
            devices=devices,
            output=output,
            compiled=compiled,
            continuation=continuation,
            progress=progress,
            keep_runtime=keep_runtime,
            request_inputs=request_inputs,
        )
        transferred = keep_runtime and "session" in result
        return result
    finally:
        if chunks is not None:
            chunks.close()
        if request_inputs is not None and not transferred:
            request_inputs.close()


def load_configuration(path):
    """Read paths relative to a JSON configuration without importing the device runtime."""
    path = Path(path).resolve()
    settings = json.loads(path.read_text(encoding="utf-8"))
    automatic = set(settings) & {"prompt", "token_ids"}
    if len(automatic) > 1:
        raise ValueError("Choose either prompt or token_ids for automatic Prefill")
    if automatic:
        if "inputs" in settings:
            raise ValueError("Automatic Prefill cannot also supply prepared inputs")
        if "prompt" in settings and (not isinstance(settings["prompt"], str) or not settings["prompt"]):
            raise ValueError("prompt must contain raw completion text")
        if "token_ids" in settings and (
            not isinstance(settings["token_ids"], list)
            or not 1 <= len(settings["token_ids"]) <= 2048
            or any(type(token) is not int or token < 0 for token in settings["token_ids"])
        ):
            raise ValueError("token_ids must contain 1..2048 nonnegative integers")
        if settings.get("continuations"):
            raise ValueError("Automatic Prefill owns its continuation descriptors")
        for name, default, lower, upper in (("chunk_size", 16, 1, 16), ("dp_rank", 0, 0, 3)):
            value = settings.get(name, default)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in {lower}..{upper}")
            settings[name] = value
    required = {"checkpoint", "cache_state", "contract", "devices", "output"}
    if not automatic:
        required.add("inputs")
    missing = required - settings.keys()
    if missing:
        raise ValueError("Missing Prefill configuration fields: " + ", ".join(sorted(missing)))
    for name in ("checkpoint", "inputs", "cache_state", "contract", "output"):
        if name not in settings:
            continue
        value = Path(settings[name])
        settings[name] = value if value.is_absolute() else (path.parent / value).resolve()
    chunks = settings.get("continuations", [])
    if not isinstance(chunks, list):
        raise ValueError("continuations must be a list of prepared chunk descriptors")
    for chunk in chunks:
        if not isinstance(chunk, dict) or not {"inputs", "contract", "metadata_sources"} <= chunk.keys():
            raise ValueError("Each continuation needs inputs, contract and metadata_sources paths")
        for name in ("inputs", "contract", "metadata_sources"):
            if not isinstance(chunk[name], str) or not chunk[name]:
                raise ValueError("Continuation paths must be nonempty strings")
            value = Path(chunk[name])
            chunk[name] = value if value.is_absolute() else (path.parent / value).resolve()
    devices = settings["devices"]
    if (
        not isinstance(devices, list)
        or len(devices) != 16
        or any(type(value) is not int or value < 0 for value in devices)
        or len(set(devices)) != 16
    ):
        raise ValueError("The current full-network configuration requires 16 distinct physical device IDs")
    return settings


def load_continuations(chunks):
    """Load each prepared chunk and its rank metadata descriptors on demand."""
    import torch

    for chunk in chunks:
        metadata_path = Path(chunk["metadata_sources"])
        sources = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(sources, list) or len(sources) != 16:
            raise ValueError("Continuation metadata_sources must describe all 16 ranks")
        for rank, source in enumerate(sources):
            if not isinstance(source, dict) or not {"rank", "path", "sha256"} <= source.keys():
                raise ValueError("Each continuation metadata source needs rank, path and sha256")
            if type(source["rank"]) is not int or source["rank"] != rank:
                raise ValueError("Continuation metadata sources must retain rank order")
            value = Path(source["path"])
            source["path"] = str(value if value.is_absolute() else (metadata_path.parent / value).resolve())
        inputs = torch.load(chunk["inputs"], map_location="cpu", weights_only=True)
        contract = json.loads(Path(chunk["contract"]).read_text(encoding="utf-8"))
        yield dict(inputs=inputs, contract=contract, metadata_sources=sources)


def main():
    """Load an explicit full-network configuration and run the shared product path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("-p", "--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("--devices", help="Comma-separated physical device IDs; overrides the JSON mapping")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check-config", action="store_true", help="Validate the configuration without device execution"
    )
    parser.add_argument(
        "--compile-only", action="store_true", help="Compile the complete network without NPU execution"
    )
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
    if "prompt" in settings or "token_ids" in settings:
        inputs = settings["prompt"] if "prompt" in settings else settings["token_ids"]
        contract = dict(contract, chunk_size=settings["chunk_size"], dp_rank=settings["dp_rank"])
    else:
        inputs = torch.load(settings["inputs"], map_location="cpu", weights_only=True)
    cache = torch.load(settings["cache_state"], map_location="cpu", weights_only=True)
    if args.compile_only:
        program = compile_prefill(
            inputs, cache=cache, config=contract, output=settings["output"], weights=settings["checkpoint"]
        )
        print(json.dumps(program.summary()))
        return
    continuation = load_continuations(settings["continuations"]) if settings.get("continuations") else None
    if isinstance(inputs, dict):
        from .engram_host import HostEngram

        with HostEngram(settings["checkpoint"], cache["engram_contracts"], settings["output"].name) as host:
            ready = host.lookup(inputs, cache=cache, config=contract)
            source_chunks = continuation

            def prepared_continuations():
                for chunk in source_chunks:
                    payloads = [
                        torch.load(row["path"], map_location="cpu", weights_only=True)
                        for row in chunk["metadata_sources"]
                    ]
                    ready = host.lookup(
                        chunk["inputs"], cache=dict(cache, cache_ABI=payloads), config=chunk["contract"]
                    )
                    yield dict(chunk, inputs=dict(chunk["inputs"], engram_lookup=ready))

            result = prefill_fwd(
                inputs,
                weights=settings["checkpoint"],
                cache=cache,
                config=contract,
                devices=settings["devices"],
                output=settings["output"],
                engram_lookup=ready,
                continuation=prepared_continuations() if source_chunks is not None else None,
            )
    else:
        result = prefill_fwd(
            inputs,
            weights=settings["checkpoint"],
            cache=cache,
            config=contract,
            devices=settings["devices"],
            output=settings["output"],
        )
    print(
        json.dumps(
            dict(
                completed_layers=result["completed_layers"],
                final_output_files=result["final_output_files"],
                numerical_acceptance=result["formal_accuracy"],
            )
        )
    )


run_prefill = prefill_fwd


if __name__ == "__main__":
    main()

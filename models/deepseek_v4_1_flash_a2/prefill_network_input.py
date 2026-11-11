# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compose vocabulary lookup, TP exchange, PyPTO summation and initial SP/HC streams."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a2"

import argparse
import json
from pathlib import Path

import torch

from .prefill_embedding import make_input_entries
from .runtime_utils import Executable, communicate, file_sha, loaded_source_files, tensor_sha, write_json
from .weights import CheckpointOperands


def run_input_boundary(settings, *, config, capacity, reader, execute, exchange):
    """Embed the actual request tokens and return rank-local HC inputs in memory."""
    topology = settings["topology"]
    dp, tp, world = topology["dp"], topology["tp"], topology["ep"]
    hidden, vocabulary, hc_mult = config["hidden_size"], config["vocab_size"], config["hc_mult"]
    if world != dp * tp or len(settings["DP_engines"]) != dp:
        raise ValueError("Input request engines must match the complete network topology")
    if any(not 0 <= engine["active"] <= capacity for engine in settings["DP_engines"]):
        raise ValueError("Actual request tokens exceed the compiled Prefill capacity")
    weight = reader.load("embed.weight")
    if tuple(weight.shape) != (vocabulary, hidden) or weight.dtype != torch.bfloat16:
        raise ValueError("Actual checkpoint embedding shape or dtype differs")
    packets, rows_by_rank = [], []
    for rank in range(world):
        engine = settings["DP_engines"][rank // tp]
        tokens = [token for request in engine["requests"] for token in request["token_ids"]]
        masks = [value for request in engine["requests"] for value in request["live_token_mask"]]
        if len(tokens) != engine["active"] or len(masks) != len(tokens):
            raise ValueError("Request/token ownership differs from the actual prepared boundary")
        if any(token < 0 or token >= vocabulary for token in tokens):
            raise ValueError("Prepared token ID is outside the original vocabulary")
        ids, live = torch.zeros(capacity, dtype=torch.int64), torch.zeros(capacity, dtype=torch.int32)
        ids[:len(tokens)] = torch.tensor(tokens, dtype=torch.int64)
        live[:len(tokens)] = torch.tensor(masks, dtype=torch.int32)
        output = torch.zeros((capacity, hidden), dtype=torch.bfloat16)
        start, count = rank % tp * (vocabulary // tp), vocabulary // tp
        local_weight = weight.narrow(0, start, count).contiguous()
        execute("lookup", rank, [ids, live, local_weight, output, start, len(tokens)], dict(lookup=output))
        packets.append({"network-input": {"value": output}})
        rows_by_rank.append(len(tokens))
    gathered = exchange("input-vocabulary-TP-gather", packets)
    result = []
    for rank in range(world):
        active = rows_by_rank[rank]
        physical = max(1, (active + tp - 1) // tp)
        embedding = torch.zeros((capacity, hidden), dtype=torch.bfloat16)
        execute("reduce", rank, [gathered[rank]["network-input"]["value"], embedding, active], dict(embedding=embedding))
        streams = torch.zeros((capacity, hc_mult, hidden), dtype=torch.bfloat16)
        premix = torch.zeros((capacity, hc_mult), dtype=torch.float32)
        execute("initial_hc", rank, [embedding, streams, premix, rank % tp, physical, active],
                dict(hidden=streams, pre_mix=premix))
        valid = max(0, min(physical, active - rank % tp * physical))
        result.append(dict(hidden=streams[:(active + tp - 1) // tp].clone(),
            pre_mix=premix[:(active + tp - 1) // tp].clone(),
            SP_physical_rows=(active + tp - 1) // tp, SP_kernel_rows=physical, SP_valid_rows=valid, DP_rank=rank // tp, TP_rank=rank % tp,
            global_token_start=rank % tp * physical))
    return result


def run_input_stage(args):
    """Produce the actual backbone input boundary without invoking any model layer."""
    settings = json.loads(args.input_contract.read_text())
    topology = settings["topology"]
    dp, tp, world = topology["dp"], topology["tp"], topology["ep"]
    if (dp, tp, world) != (4, 4, 16) or any(topology[name] != 1 for name in ("pp", "pcp", "dcp")):
        raise ValueError("This prepared input boundary requires DP4TP4EP16PP1PCP1DCP1")
    checkpoint = Path(settings["checkpoint"])
    for name, expected in settings["checkpoint_sha256"].items():
        if file_sha(checkpoint / name) != expected:
            raise ValueError("Prepared checkpoint, tokenizer or quantization contract changed: " + name)
    config = json.loads((checkpoint / "config.json").read_text())["text_config"]
    capacity = 16
    if len(settings["DP_engines"]) != dp or any(not 0 <= row["active"] <= capacity for row in settings["DP_engines"]):
        raise ValueError("Prepared bounded real-token input geometry differs")
    from .runtime_config import configure_devices

    assigned = [] if args.compile_only else configure_devices(args.devices, world=world)
    args.output.mkdir(parents=True, exist_ok=False)
    definitions = make_input_entries(capacity, config, tp)
    entries = {name: Executable(function, args.output / "compiled" / name)
               for name, function in definitions.items()}
    runtime = dict(source_root=str(Path(__file__).resolve().parents[2]), source_files={},
        topology=topology, assigned_devices=[] if args.compile_only else assigned,
        logical_rank_mapping=list(range(world)), inputs_sha256=file_sha(args.input_contract),
        groups=dict(ep=[list(range(world))], tp=[list(range(group * tp, (group + 1) * tp)) for group in range(dp)],
                    dp=[[lane + group * tp for group in range(dp)] for lane in range(tp)]),
        component="input_embedding_and_initial_HC_only", model_layers_executed=[],
        native_reference="not_run", complete_network="not_run", installed=False, frozen=False)
    runtime_path = args.output / "runtime_config.json"
    loaded_source_files(runtime)
    write_json(runtime_path, runtime)
    if args.compile_only:
        write_json(args.output / "compilation.json", dict(entries={k: v.artifacts for k, v in entries.items()},
            runtime_config_sha256=file_sha(runtime_path), source_files=runtime["source_files"],
            NPU="not_run", native_reference="not_run", complete_network="not_run"))
        return

    from .moe_transport import CANNMemory

    torch.set_num_threads(2)
    memory = CANNMemory()
    reports = [dict(rank=rank, kernels=[], communication=[]) for rank in range(world)]
    packets = [{"startup": {"rank": torch.tensor([[rank]], dtype=torch.int32)}} for rank in range(world)]
    for group in ("ep", "tp", "dp"):
        values, receipts = communicate("input-startup-" + group, packets, args.output, runtime_path,
                                        collective="all_gather", group=group)
        for rank in range(world):
            expected = next(row for row in runtime["groups"][group] if rank in row)
            if values[rank]["startup"]["rank"].flatten().tolist() != expected:
                raise RuntimeError("Input-stage rank/group startup mismatch: " + group)
            reports[rank]["communication"].append(receipts[rank])
    reader = CheckpointOperands(checkpoint)
    captures = args.output / "captures"
    captures.mkdir()

    def execute(name, rank, arguments, outputs):
        readonly = [(index, tensor_sha(value)) for index, value in enumerate(arguments)
                    if isinstance(value, torch.Tensor) and all(value is not out for out in outputs.values())]
        entries[name].execute(arguments, outputs, rank, "network_input/" + name,
                              captures / f"rank-{rank}-{name}.pt", reports[rank], memory)
        if any(tensor_sha(arguments[index]) != digest for index, digest in readonly):
            raise RuntimeError("A readonly input-component tensor changed: " + name)
        reports[rank].setdefault("readonly_checks", []).append(dict(operator=name, passed=True, tensors=len(readonly)))

    def exchange(stage, packets):
        values, receipts = communicate(stage, packets, args.output, runtime_path, collective="all_gather", group="tp")
        for rank, receipt in enumerate(receipts):
            reports[rank]["communication"].append(receipt)
        return values

    outputs = run_input_boundary(settings, config=config, capacity=capacity, reader=reader,
                                 execute=execute, exchange=exchange)
    for rank, result in enumerate(outputs):
        torch.save(result, args.output / f"rank-{rank}-backbone-input.pt")
        write_json(args.output / f"rank-{rank}-report.json", reports[rank])
    loaded_source_files(runtime)
    write_json(runtime_path, runtime)
    write_json(args.output / "execution.json", dict(component_executed=True, ranks=world,
        source_files=runtime["source_files"], consumed_checkpoint=reader.consumed,
        memory_samples=memory.samples, native_reference="not_run", numerical_acceptance="not_run",
        complete_network="not_run", model_layers_executed=[]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=lambda value: value.split(","), required=True)
    parser.add_argument("--input-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    run_input_stage(parser.parse_args())

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compose PyPTO prefill MoE routing, EP dispatch, W8A8 experts, and token restoration."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

import argparse
import json
from pathlib import Path
import shlex
import sys

import torch
import pypto.language as pl

from .moe import make_moe_combine
from .moe import DISPATCH_MAP_PAD as MAP_PAD, make_ep_dispatch_pack, make_ep_dispatch_unpack, make_ep_return_pack
from .moe import validate_received_routes, validate_returned_routes
from .moe import make_moe_router
from .moe import make_shared_moe_w8a8
from .moe_transport import CANNMemory
from .runtime_config import ExpertParallelConfig, configure_devices
from .runtime_utils import Executable as _Executable, communicate as _communicate
from .runtime_utils import loaded_source_files as _loaded_source_files, write_json as _json
from .runtime_utils import file_sha, tensor_sha
from .weights import CheckpointOperands




def _prepare(args, contract):
    topology = contract["topology"]
    layout = ExpertParallelConfig(topology["dp"], topology["tp"], topology["ep"], contract["global_experts"])
    if any(topology[key] != 1 for key in ("pp", "pcp", "dcp")) or not contract["enable_expert_parallel"]:
        raise ValueError("Require PP1/PCP1/DCP1 with expert parallelism enabled")
    for rank in range(layout.ep):
        if layout.ownership(rank) != contract["ownership"][rank]:
            raise ValueError("Expert ownership differs from the input contract")
    devices = configure_devices(args.devices.split(","), world=layout.ep)
    root = Path(__file__).resolve().parents[2]
    runtime = {"topology": topology, "enable_expert_parallel": True, "assigned_devices": devices,
               "logical_rank_mapping": list(range(layout.ep)), "source_root": str(root), "source_files": {},
               "command": [sys.executable, *sys.argv], "inputs_sha256": file_sha(args.inputs),
               "checkpoint_config_sha256": file_sha(Path(contract["checkpoint"]) / "config.json"),
               "complete_layer_status": "not_run", "unexecuted_topologies": ["EP16/TP1", "EP16/TP2", "EP16/TP4"]}
    path = args.output / "runtime_config.json"
    _json(path, runtime)
    (args.output / "command.txt").write_text(shlex.join(runtime["command"]) + "\n")
    return layout, runtime, path




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", required=True, help="Comma-separated physical IDs in logical rank order")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    contract = torch.load(args.inputs, map_location="cpu", weights_only=True)
    torch.set_num_threads(4)
    layout, runtime, runtime_path = _prepare(args, contract)
    capacity, topk = contract["capacity"], contract["top_k"]
    root_config = json.loads((Path(contract["checkpoint"]) / "config.json").read_text())
    config = dict(root_config["text_config"], image_sentinel_base_id=root_config["image_token_id"])
    hidden = config["hidden_size"]
    if topk != config["num_experts_per_tok"] or layout.experts != config["n_routed_experts"]:
        raise ValueError("Checkpoint MoE dimensions differ from the input contract")
    captures, build = args.output / "captures", args.output / "compiled"
    captures.mkdir()
    build.mkdir()
    routers = {controlled: _Executable(make_moe_router(capacity, dict(config, controlled_routing=controlled)),
                                      build / ("router-controlled" if controlled else "router-natural"))
               for controlled in (False, True)}
    shared_compute = make_shared_moe_w8a8(capacity, config, fused_activation=True, fused_channel_dtype=pl.FP32)
    shared = _Executable(shared_compute, build / "shared")
    expert_capacity = ((capacity * layout.ep + 15) // 16) * 16
    routed = _Executable(make_shared_moe_w8a8(expert_capacity, config, fused_activation=True), build / "routed")
    dimensions = [capacity, hidden, layout.experts, topk, layout.ep]
    pack = _Executable(make_ep_dispatch_pack(*dimensions), build / "dispatch-pack")
    unpack = _Executable(make_ep_dispatch_unpack(*dimensions), build / "dispatch-unpack")
    return_pack = _Executable(make_ep_return_pack(*dimensions), build / "return-pack")
    combine = _Executable(make_moe_combine(capacity, hidden, topk, layout.ep), build / "combine")
    local_experts, routes = layout.experts // layout.ep, capacity * topk
    packet_rows, slab_rows = layout.ep * routes, local_experts * expert_capacity
    memory = CANNMemory()
    reports, readers, shared_operands, gate_operands = [], [], [], []
    states, dispatch_packets = [{} for _ in range(layout.ep)], [{} for _ in range(layout.ep)]
    try:
        for rank in range(layout.ep):
            memory.sample(rank, "checkpoint/load/before")
            reader = CheckpointOperands(contract["checkpoint"])
            readers.append(reader)
            prefix = f"layers.{contract['layer']}.ffn"
            gate_operands.append([reader.load(prefix + suffix).float()
                                  for suffix in (".gate.weight", ".gate.bias", ".gate.bias_vl")])
            shared_operands.append(reader.expert(prefix + ".shared_experts"))
            for expert, _slot in layout.ownership(rank):
                for stem in ("w1", "w2", "w3"):
                    for suffix in ("weight", "weight_scale", "weight_offset"):
                        reader.load(f"{prefix}.experts.{expert}.{stem}.{suffix}")
            memory.sample(rank, "checkpoint/load/after")
            reports.append({"rank": rank, "ownership": layout.ownership(rank), "consumed_tensors": reader.consumed,
                            "kernels": [], "command": runtime["command"], "communication": []})
            _json(args.output / f"rank-{rank}-report.json", reports[-1])
        for case, originals in contract["cases"].items():
            for rank, original in enumerate(originals):
                active = layout.active_rows(original, rank)
                physical = len(original["x"])
                if not 0 <= active <= physical <= capacity:
                    raise ValueError("Rank-local active/input rows exceed the declared capacity")
                x = torch.zeros((capacity, hidden), dtype=torch.bfloat16)
                input_ids = torch.zeros(capacity, dtype=torch.int64)
                x[:physical].copy_(original["x"])
                input_ids[:physical].copy_(original["input_ids"])
                logits = torch.zeros((capacity, layout.experts), dtype=torch.float32)
                route_ids = torch.zeros((capacity, topk), dtype=torch.int32)
                weights = torch.zeros((capacity, topk), dtype=torch.float32)
                control = original.get("route_override")
                if control is not None:
                    route_ids[:active].copy_(torch.tensor(control, dtype=torch.int32))
                router_arguments = [x, *gate_operands[rank], input_ids, logits, route_ids, weights, active]
                router_outputs = {"logits": logits, "route_ids": route_ids, "route_weights": weights,
                                  "readonly_x": x, "readonly_input_ids": input_ids}
                routers[control is not None].execute(router_arguments, router_outputs, rank, "router",
                                                    captures / f"rank-{rank}-{case}-router.pt", reports[rank], memory)
                shared_output = torch.zeros_like(x)
                shared.execute([x, *shared_operands[rank], shared_output, active], {"shared": shared_output},
                               rank, "shared", captures / f"rank-{rank}-{case}-shared.pt", reports[rank], memory)
                packet = {"routes": torch.full((packet_rows, 4), -1, dtype=torch.int64),
                          "hidden": torch.zeros((packet_rows, hidden), dtype=torch.bfloat16),
                          "counts": torch.zeros((layout.ep, MAP_PAD), dtype=torch.int32)}
                pack_arguments = [x, route_ids, packet["routes"], packet["hidden"], packet["counts"], active, rank]
                pack.execute(pack_arguments, packet, rank, "dispatch_pack",
                             captures / f"rank-{rank}-{case}-dispatch-pack.pt", reports[rank], memory)
                if not torch.equal(x[:physical], original["x"]) or not torch.equal(input_ids[:physical], original["input_ids"]):
                    raise RuntimeError("A PyPTO operator modified a readonly boundary input")
                dispatch_packets[rank][case] = packet
                states[rank][case] = {"logits": logits[:active].clone(), "route_ids": route_ids[:active].clone(),
                                      "route_weights": weights[:active].clone(), "shared": shared_output[:active].clone(),
                                      "readonly_x": x[:physical].clone(), "readonly_input_ids": input_ids[:physical].clone(),
                                      "padded_weights": weights, "padded_shared": shared_output}
        received, receipts = _communicate("dispatch", dispatch_packets, args.output, runtime_path)
        return_packets = [{} for _ in range(layout.ep)]
        for rank in range(layout.ep):
            reports[rank]["communication"].append(receipts[rank])
            for case, packet in received[rank].items():
                total = validate_received_routes(packet, rank, layout, routes)
                recv_routes = torch.full((packet_rows, 4), -1, dtype=torch.int64)
                recv_hidden = torch.zeros((packet_rows, hidden), dtype=torch.bfloat16)
                expert_x = torch.zeros((slab_rows, hidden), dtype=torch.bfloat16)
                counts = torch.zeros(local_experts, dtype=torch.int64)
                packet_map = torch.full((packet_rows, MAP_PAD), -1, dtype=torch.int32)
                received_total = torch.zeros(MAP_PAD, dtype=torch.int32)
                unpack_arguments = [packet["routes"], packet["hidden"], packet["counts"], recv_routes, recv_hidden,
                                    expert_x, counts, packet_map, received_total, rank]
                unpack_outputs = {"received_routes": recv_routes, "received_hidden": recv_hidden,
                                  "expert_counts": counts, "packet_map": packet_map, "received_total": received_total}
                unpack.execute(unpack_arguments, unpack_outputs, rank, "dispatch_unpack",
                               captures / f"rank-{rank}-{case}-dispatch-unpack.pt", reports[rank], memory)
                if int(received_total[0]) != total:
                    raise RuntimeError("PyPTO unpack did not account for every received route")
                state = states[rank][case]
                state.update(received_routes=recv_routes[:total].clone(), received_hidden=recv_hidden[:total].clone(),
                             local_expert_counts=counts)
                expert_y = torch.zeros_like(expert_x)
                for local, count in enumerate(counts.tolist()):
                    if not count:
                        continue
                    if not 0 < count <= expert_capacity:
                        raise ValueError("Received expert rows exceed the parameterized slab capacity")
                    expert = rank * local_experts + local
                    begin = local * expert_capacity
                    x_view, y_view = expert_x[begin:begin + expert_capacity], expert_y[begin:begin + expert_capacity]
                    memory.sample(rank, f"expert-{expert}/load/before")
                    operands = readers[rank].expert(f"layers.{contract['layer']}.ffn.experts.{expert}")
                    memory.sample(rank, f"expert-{expert}/load/after")
                    routed.execute([x_view, *operands, y_view, count], {"expert_output": y_view}, rank, "routed",
                                   captures / f"rank-{rank}-{case}-expert-{expert}.pt", reports[rank], memory)
                sent_hidden = torch.zeros_like(packet["hidden"])
                return_packet = {"routes": packet["routes"], "counts": packet["counts"], "hidden": sent_hidden}
                return_pack.execute([expert_y, packet["counts"], packet_map, sent_hidden], return_packet,
                                    rank, "dispatch_pack", captures / f"rank-{rank}-{case}-return-pack.pt", reports[rank], memory)
                return_packets[rank][case] = return_packet
        returned, receipts = _communicate("return", return_packets, args.output, runtime_path)
        for rank in range(layout.ep):
            reports[rank]["communication"].append(receipts[rank])
            for case, packet in returned[rank].items():
                state = states[rank][case]
                active = len(state["route_ids"])
                validate_returned_routes(packet, rank, routes, state["route_ids"], active)
                routed_output = torch.zeros((capacity, hidden), dtype=torch.bfloat16)
                output = torch.full_like(routed_output, contract["output_sentinel"])
                returned_routes = torch.full((routes, 4), -1, dtype=torch.int64)
                combine_arguments = [packet["routes"], packet["hidden"], packet["counts"], state.pop("padded_weights"),
                                     state.pop("padded_shared"), routed_output, output, returned_routes, active]
                combine_outputs = {"routed": routed_output, "output": output, "returned_routes": returned_routes}
                combine.execute(combine_arguments, combine_outputs, rank, "combine",
                                captures / f"rank-{rank}-{case}-combine.pt", reports[rank], memory)
                state.update(routed=routed_output[:active].clone(), output=output,
                             returned_routes=returned_routes[:active * topk].clone())
                torch.save(state, args.output / f"rank-{rank}-{case}.pt")
                print({"rank": rank, "case": case, "active": active, "output_sha256": tensor_sha(output)}, flush=True)
    finally:
        for rank, report in enumerate(reports):
            report["consumed_tensors"] = readers[rank].consumed
            report["memory"] = {"scope": "CANN whole-device HBM samples at load/execute boundaries; not a peak",
                                "samples": [sample for sample in memory.samples if sample["rank"] == rank],
                                "transport": [receipt["memory"] for receipt in report["communication"]]}
            _json(args.output / f"rank-{rank}-report.json", report)
        _loaded_source_files(runtime)
        _json(runtime_path, runtime)


if __name__ == "__main__":
    main()

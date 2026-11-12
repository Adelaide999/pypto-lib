# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate CPU table lookup, chunk history and ready-row forward admission."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from .engram_host import HostEngram, lookup_columns


def require(value, message):
    if not value:
        raise AssertionError(message)


def reject(call):
    try:
        call()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("Invalid host-row input was admitted")


def fixture(root, *, tp=2, dp=2):
    from safetensors.torch import save_file

    constants = dict(layer_ids=[1, 14], pad_id=2, token_map=torch.arange(64, dtype=torch.int32),
        multipliers=torch.tensor([[3, 5, 7, 11], [13, 17, 19, 23]], dtype=torch.int64),
        primes=torch.full((2, 24), 5, dtype=torch.int64),
        offsets=torch.arange(24, dtype=torch.int64).repeat(2, 1) * 5)
    topology = dict(dp=dp, tp=tp, ep=dp * tp, pp=1, pcp=1, dcp=1)
    model = dict(image_token_id=61, image_pad_token_id=62, text_config={})
    tensors, contracts, weights = {}, {}, {}
    for layer in (1, 14):
        codes = ((torch.arange(120 * 256).reshape(120, 256) + layer) % 255 - 127).to(torch.int8)
        scales = torch.linspace(0.001, 0.2, 120 * 8).reshape(120, 8)
        tensors[f"layers.{layer}.engram.embed.weight"] = codes
        tensors[f"layers.{layer}.engram.embed.scale"] = scales
        weights[layer] = codes, scales
        shards = []
        for rank in range(tp):
            begin, end = rank * 120 // tp, (rank + 1) * 120 // tp
            hashes = {kind: sha256(value[begin:end].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                      for kind, value in (("codes", codes), ("scales", scales))}
            shards.append(dict(head_range=[rank * 24 // tp, (rank + 1) * 24 // tp], row_range=[begin, end],
                               DP_sharing_ranks=list(range(rank, dp * tp, tp)), host_payload_sha256=hashes))
        contracts[layer] = dict(checkpoint=str(root), topology=topology, model_config=model,
            physical_layer_id=layer, hash_constants=constants, engram_shards=shards,
            engram_storage_contract=dict(dtype="INT8", scale_dtype="FP32", group_size=32,
                cpu_offload=True, dp_shared_memory=True, global_columns=24), capacity=16)
    save_file(tensors, root / "weights.safetensors")
    (root / "quant_model_weights.safetensors.index.json").write_text(json.dumps(
        dict(weight_map={name: "weights.safetensors" for name in tensors})))
    return contracts, weights


def request(tokens, history=(), slot=0):
    return dict(request_slot=slot, token_ids=list(tokens), history_tokens=list(history),
        position_ids=list(range(len(history), len(history) + len(tokens))),
        live_token_mask=[1] * len(tokens), initial_context_length=len(history))


def chunk(root, contracts, requests, epoch):
    topology = contracts[1]["topology"]
    count = sum(len(row["token_ids"]) for row in requests)
    tp = topology["tp"]
    engines = [dict(dp_rank=dp, active=count if dp == 0 else 0, requests=requests if dp == 0 else [])
               for dp in range(topology["dp"])]
    inputs = dict(input_contract=dict(topology=topology, epoch=epoch, DP_engines=engines))
    payloads = []
    for rank in range(topology["ep"]):
        rows = requests if rank < tp else []
        positions = [pos for row in rows for pos in row["position_ids"]]
        starts = [0]
        for row in rows:
            starts.append(starts[-1] + len(row["token_ids"]))
        physical = (count + tp - 1) // tp if rank < tp else 0
        begin = rank % tp * physical
        metadata = dict(query_start_loc=torch.tensor(starts, dtype=torch.int32),
            slot_mapping=torch.arange(len(positions), dtype=torch.int64),
            block_table=torch.zeros((len(rows), 1), dtype=torch.int32))
        payloads.append(dict(positions=torch.tensor(positions, dtype=torch.int64),
            query_start_loc=metadata["query_start_loc"], metadata={"model.layers.1.self_attn.swa_cache": dict(
                cp_token_range=[begin, begin + physical, physical, physical * tp], global_metadata=metadata)}))
    config = dict(topology=topology, capacity=16, checkpoint=str(root), input_contract_sha256=f"chunk-{epoch}")
    return inputs, dict(cache_ABI=payloads), config


def validate_host():
    with TemporaryDirectory(prefix="ds41-host-rows-") as temporary:
        root = Path(temporary)
        contracts, weights = fixture(root)
        first = chunk(root, contracts, [request([7, 61, 9])], 0)
        second = chunk(root, contracts, [request([10, 11], [7, 61, 9])], 1)
        whole = chunk(root, contracts, [request([7, 61, 9, 10, 11])], 2)
        with HostEngram(root, contracts, "host-validation") as host:
            prefix = host.lookup(first[0], cache=first[1], config=first[2])
            next_rows = host.lookup(second[0], cache=second[1], config=second[2])
            full = host.lookup(whole[0], cache=whole[1], config=whole[2])
            left, right = next_rows["cases"]["fresh"][0], full["cases"]["fresh"][0]
            require(torch.equal(left["hashes"][:2], right["hashes"][3:5]), "Chunked request history changed hashes")
            for layer, slot in ((1, 0), (14, 1)):
                codes, scales = weights[layer]
                expected = torch.zeros((5, 6144), dtype=torch.bfloat16)
                for token in range(5):
                    for head in range(24):
                        row = int(right["hashes"][token, slot, head])
                        for group in range(8):
                            begin = group * 32
                            expected[token, head * 256 + begin:head * 256 + begin + 32] = (
                                codes[row, begin:begin + 32].float() * scales[row, group]).to(torch.bfloat16)
                require(torch.equal(right["full"][layer][:5], expected), "CPU group32 decoding changed real row values")
                require(torch.equal(left["full"][layer][:2], expected[3:5]), "Chunked lookup changed selected entries")
                require(torch.equal(right["local"][layer][:3], expected[:3]), "TP lane0 changed the SP partition")
                lane1 = full["cases"]["fresh"][1]
                require(torch.equal(lane1["local"][layer][:2], expected[3:5]), "TP lane1 changed the SP partition")
                require(not torch.any(lane1["local"][layer][2:]), "Padded rows were not zero")
            require(prefix["cases"]["fresh"][0]["mask"][:3].tolist() == [1, 0, 1], "Image mask changed")
            require(all(not torch.any(row["mask"]) and not torch.any(row["local"][1])
                        for row in full["cases"]["fresh"][2:]), "Empty DP replicas were not preserved")
            codes = host._table(1, 1, "codes")
            scales = host._table(1, 1, "scales")
            invalid = lookup_columns(codes, scales, torch.tensor([[-1, 59, 120]], dtype=torch.int32))
            require(not torch.any(invalid), "Out-of-shard IDs did not produce zero rows")
            copied = right["full"][1].clone()
            paths = [array.path for array in host.arrays.values()]
            bad = deepcopy(first[0])
            bad["input_contract"]["DP_engines"][0]["requests"][0]["position_ids"][0] = 4
            reject(lambda: host.lookup(bad, cache=first[1], config=first[2]))
            multiple = chunk(root, contracts, [request([7, 8], [3, 4]), request([9], [10], slot=1)], 3)
            multi = host.lookup(multiple[0], cache=multiple[1], config=multiple[2])
            for index, req in enumerate(multiple[0]["input_contract"]["DP_engines"][0]["requests"]):
                single = chunk(root, contracts, [req], 4 + index)
                one = host.lookup(single[0], cache=single[1], config=single[2])
                begin, end = ((0, 2), (2, 3))[index]
                require(torch.equal(multi["cases"]["fresh"][0]["hashes"][begin:end],
                    one["cases"]["fresh"][0]["hashes"][:end - begin]), "Request boundary leaked another request's history")
            original_cache = first[1]
        require(not any(path.exists() for path in paths), "Host storage teardown leaked mappings")
        require(torch.equal(right["full"][1], copied), "Prepared rows alias closed table mappings")
        reject(lambda: host.lookup(first[0], cache=original_cache, config=first[2]))

        from . import prefill_network_engram as network

        records = full["cases"]["fresh"]
        batch_rows = []
        for rank, row in enumerate(records):
            total = len(row["full_input_ids"])
            physical = row["cp_token_range"][2]
            batch_rows.append(dict(full_input_ids=row["full_input_ids"], positions=row["positions"],
                cp_token_range=row["cp_token_range"], SP_physical_rows=physical,
                SP_valid_rows=min(physical, max(0, total - row["cp_token_range"][0])),
                metadata=dict(slot_mapping=torch.arange(total, dtype=torch.int64))))
        batch = dict(epoch=2, token_map_id="chunk-2", engram_metadata_owner_layer=1, cases=dict(fresh=batch_rows))
        bindings = [dict(fresh=dict(owner_layer=1, block_size=128, bound=True,
                                    kv_binding_id=f"rank-{rank}-L1", kv_blocks=1)) for rank in range(4)]
        history = network.NetworkEngramHistory(bindings)
        runtime = dict(topology=contracts[1]["topology"], capacity=16, output=root / "forward",
            entries_for_stage=lambda stage: {}, reports=[{} for _ in range(4)], diagnostics=False,
            capture_state=True, load_rotation=lambda *args: (torch.eye(32), {}))
        with patch.object(network, "Executable", side_effect=AssertionError("Unexpected compilation")):
            prepared = network.prepare_network_engram(contracts[1], batch, history, contracts,
                lambda layer: None, runtime, lookup=full)
            require(history.caches[0]["fresh"][:5].tolist() == [7, -1, 9, 10, 11], "Forward history publication changed")
            require(not torch.any(history.caches[2]["fresh"]), "Empty DP mutated history")
            require(prepared.states[1][0]["fresh"]["local"] is records[0]["local"][1],
                    "Forward did not consume caller-prepared rows")
            reject(lambda: network.prepare_network_engram(contracts[1], batch, history, contracts,
                lambda layer: None, runtime, lookup=full))
            changed = dict(full, epoch=3)
            reject(lambda: network.prepare_network_engram(contracts[1], batch,
                network.NetworkEngramHistory(bindings), contracts, lambda layer: None, runtime, lookup=changed))
            changed = deepcopy(full)
            changed["cases"]["fresh"][0]["local"][1] = changed["cases"]["fresh"][0]["local"][1].float()
            reject(lambda: network.validate_host_lookup(changed, contracts[1], batch))
        tp4_root = root / "tp4"
        tp4_root.mkdir()
        tp4_contracts, _ = fixture(tp4_root, tp=4, dp=4)
        small = chunk(tp4_root, tp4_contracts, [request([7, 8, 9])], 0)
        with HostEngram(tp4_root, tp4_contracts, "host-validation-tp4") as host:
            value = host.lookup(small[0], cache=small[1], config=small[2])
        require(len(value["cases"]["fresh"]) == 16, "TP4/DP4 did not preserve every rank")
        for rank, row in enumerate(value["cases"]["fresh"]):
            if rank < 3:
                require(torch.equal(row["local"][1][0], row["full"][1][rank]), "TP4 changed row ownership")
            else:
                require(not torch.any(row["local"][1]), "TP4 padding or empty DP rows were not zero")
        return dict(lookup="CPU INT8/FP32 group32 to BF16; exact row oracle", slots=2,
            requests="chunked history, separate requests and image masking", ranks="TP2/DP2 and TP4/DP4",
            forward="ready-row admission and history publication", device_execution="not_run")


def validate_device(device, *, compile_only=False):
    """Run ready CPU rows through the existing projection and gate kernels."""
    from golden import ScalarSpec, TensorSpec, run

    from .prefill_engram import prefill_engram_gate_golden_cases
    from .qkv_proj_rope import golden_bf16_projection, golden_prefix_compare, make_bf16_projection_staged

    with TemporaryDirectory(prefix="ds41-host-device-") as temporary:
        root = Path(temporary)
        contracts, _ = fixture(root)
        inputs, cache, config = chunk(root, contracts, [request([7, 61, 9])], 0)
        with HostEngram(root, contracts, "host-device-validation") as host:
            ready = host.lookup(inputs, cache=cache, config=config)
        source = ready["cases"]["fresh"][0]["local"][1]
        generator = torch.Generator().manual_seed(41)
        weight = torch.empty((6144, 25600), dtype=torch.bfloat16).uniform_(-0.002, 0.002, generator=generator)
        specs = [TensorSpec("source", [16, 6144], torch.bfloat16, init_value=source),
                 TensorSpec("weight", [6144, 25600], torch.bfloat16, init_value=weight),
                 TensorSpec("output", [16, 25600], torch.bfloat16, init_value=-12.5),
                 ScalarSpec("num_tokens", torch.int32, 2, compile_runtime=True)]
        result = run(make_bf16_projection_staged(6144, 25600, capacity=16), specs,
            golden_fn=golden_bf16_projection, compare_fn={"output": golden_prefix_compare(2)},
            config=dict(backend_type="a2a3", execution_mode=False, device_id=device), compile_only=compile_only)
        require(result.passed, f"Host-row projection: {result.error}")
        gate = next(case for case in prefill_engram_gate_golden_cases() if case["name"].endswith("-7"))
        gate.pop("name")
        result = run(**gate, config=dict(backend_type="a2a3", execution_mode=False, device_id=device), compile_only=compile_only)
        require(result.passed, f"Engram gate: {result.error}")
    return dict(device_execution="not_run" if compile_only else "PASS", compilation="PASS", device=device,
                projection="host-selected BF16 rows, full 6144x25600 weight",
                gate="existing original-basis gate; active and masked rows", full_network="not_run")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, help="Run projection/gate on an allocated device after CPU validation")
    parser.add_argument("--compile-only", action="store_true", help="Compile projection/gate without device execution")
    arguments = parser.parse_args()
    print(json.dumps(validate_host()))
    if arguments.device is not None or arguments.compile_only:
        print(json.dumps(validate_device(arguments.device or 0, compile_only=arguments.compile_only)))

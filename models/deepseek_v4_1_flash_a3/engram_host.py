# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare Engram rows on the host before a Prefill or token-continuation forward."""

from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path

import torch

from .engram_hash import HashLayout, hash_ids
from .engram_storage import MappedEngramArray, SharedEngramArray


def lookup_columns(codes, scales, rows):
    """Gather INT8/FP32 group32 rows on CPU and round the decoded entries to BF16."""
    values = codes.gather(rows).float().reshape(*rows.shape, 8, 32)
    factors = scales.gather(rows).unsqueeze(-1)
    return (values * factors).reshape(*rows.shape, 256).to(torch.bfloat16)


class HostEngram:
    """Retain both complete host tables and return only the current chunk's rows."""

    def __init__(self, checkpoint, contracts, namespace):
        if set(contracts) != {1, 14} or not namespace or "/" in namespace or ".." in namespace:
            raise ValueError("Require both original Engram slots and a private host-storage namespace")
        self.checkpoint = str(Path(checkpoint).resolve())
        self.contracts = deepcopy(contracts)
        self.namespace = namespace
        self.arrays = {}
        self._stack = ExitStack()
        self._closed = False
        self.constants = self.contracts[1]["hash_constants"]
        for layer, contract in self.contracts.items():
            if str(Path(contract["checkpoint"]).resolve()) != self.checkpoint:
                raise ValueError("Host lookup and forward must use the same checkpoint")
            if contract["topology"] != self.contracts[1]["topology"]:
                raise ValueError("Both host tables must use the same rank ownership")
            storage = dict(dtype="INT8", scale_dtype="FP32", group_size=32, cpu_offload=True,
                           dp_shared_memory=True, global_columns=24)
            if contract["engram_storage_contract"] != storage:
                raise ValueError("Host lookup requires the original Eco-Tech INT8/FP32 group32 tables")
            other = contract["hash_constants"]
            if (other["layer_ids"] != [1, 14] or other["pad_id"] != self.constants["pad_id"]
                    or any(not torch.equal(other[key], self.constants[key]) for key in
                           ("token_map", "multipliers", "primes", "offsets"))):
                raise ValueError("Both host tables must use the same original hash constants")
            tp = contract["topology"]["tp"]
            if len(contract["engram_shards"]) != tp or tp not in (1, 2, 4):
                raise ValueError("Require complete original TP1/2/4 table ownership")
            for rank, shard in enumerate(contract["engram_shards"]):
                if shard["head_range"] != [rank * 24 // tp, (rank + 1) * 24 // tp]:
                    raise ValueError("Host table shards must preserve original column order")
                if shard["DP_sharing_ranks"] != list(range(rank, contract["topology"]["ep"], tp)):
                    raise ValueError("Host tables must be shared by matching DP peers")
        self.layout = HashLayout(
            (1, 14), self.constants["multipliers"], self.constants["primes"].reshape(2, 24),
            self.constants["offsets"], 4, 8, int(self.constants["token_map"].max()) + 1,
            self.constants["pad_id"], len(self.constants["token_map"]),
        )

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    def close(self):
        """Release full host tables; already prepared row tensors remain independent."""
        if not self._closed:
            self._closed = True
            self._stack.close()
            self.arrays.clear()

    def _table(self, layer, rank, kind):
        if self._closed:
            raise RuntimeError("Host Engram is closed")
        key = layer, rank, kind
        if key not in self.arrays:
            suffix, width, dtype = {"codes": ("weight", 256, "I8"), "scales": ("scale", 8, "F32")}[kind]
            shard = self.contracts[layer]["engram_shards"][rank]
            if self.contracts[layer].get("host_storage") == "checkpoint_file_mapping":
                array = MappedEngramArray(self.checkpoint, f"layers.{layer}.engram.embed.{suffix}",
                    shard["row_range"], width, dtype)
            else:
                array = SharedEngramArray(self.checkpoint, f"layers.{layer}.engram.embed.{suffix}",
                    shard["row_range"], width, dtype, shard["host_payload_sha256"][kind],
                    f"{self.namespace}-L{layer}-tp{rank}-{kind}")
            self._stack.callback(array.close)
            self.arrays[key] = array
        return self.arrays[key]

    def lookup(self, inputs, *, cache, config):
        """Return ready rows for an explicit request, including single-token continuations.

        Hashes use the request's explicit last three history tokens. No device
        workers, registrations, descriptors or transfers are used here.
        """
        if self._closed:
            raise RuntimeError("Host Engram is closed")
        topology, capacity = config["topology"], config["capacity"]
        actual = inputs["input_contract"]
        if (topology != self.contracts[1]["topology"] or actual["topology"] != topology
                or str(Path(config["checkpoint"]).resolve()) != self.checkpoint):
            raise ValueError("Host rows must match the forward checkpoint and topology")
        tp, world = topology["tp"], topology["ep"]
        payloads = cache["cache_ABI"]
        if len(payloads) != world or len(actual["DP_engines"]) != topology["dp"]:
            raise ValueError("Host preparation needs the actual all-rank request metadata")
        records = []
        model = self.contracts[1]["model_config"]
        image_id = model["image_token_id"]
        image_pad_id = model.get("image_pad_token_id", image_id + 1)
        for dp, engine in enumerate(actual["DP_engines"]):
            tokens, positions, starts, lookback = [], [], [0], []
            for request in engine["requests"]:
                prefix = request["history_tokens"]
                current = request["token_ids"]
                if (not current or any(type(token) is not int or not 0 <= token < len(self.constants["token_map"])
                                       for token in [*prefix, *current])
                        or request["initial_context_length"] != len(prefix) or request["live_token_mask"] != [1] * len(current)
                        or request["position_ids"] != list(range(len(prefix), len(prefix) + len(current)))):
                    raise ValueError("Host lookup requires complete contiguous request coordinates")
                tokens.extend(current)
                positions.extend(request["position_ids"])
                starts.append(len(tokens))
                lookback.append((list(reversed(prefix[-3:])) + [-1] * 3)[:3])
            if engine["dp_rank"] != dp or engine["active"] != len(tokens) or len(tokens) > capacity:
                raise ValueError("Host input does not match the bounded DP engine")
            ids = torch.tensor(tokens, dtype=torch.int64)
            pos = torch.tensor(positions, dtype=torch.int64)
            if torch.any(ids < 0) or torch.any(ids >= len(self.constants["token_map"])):
                raise ValueError("Host lookup token IDs exceed the compressed token map")
            dead = (ids == image_id) | (ids == image_pad_id)
            hashes = torch.full((capacity, 2, 24), -1, dtype=torch.int32)
            mask = torch.zeros(capacity, dtype=torch.int32)
            if tokens:
                previous = torch.tensor(lookback, dtype=torch.int64)
                hashes[:len(tokens)] = hash_ids(ids, self.constants["token_map"], pos,
                    torch.tensor(starts, dtype=torch.int32), layout=self.layout, dead_mask=dead,
                    lookback_token_ids=previous, lookback_dead_mask=(previous == image_id) | (previous == image_pad_id))
                mask[:len(tokens)] = (~dead).to(torch.int32)
            full = {layer: torch.zeros((capacity, 6144), dtype=torch.bfloat16) for layer in (1, 14)}
            if tokens:
                for layer, slot in ((1, 0), (14, 1)):
                    for rank, shard in enumerate(self.contracts[layer]["engram_shards"]):
                        begin, end = shard["head_range"]
                        rows = hashes[:len(tokens), slot, begin:end]
                        decoded = lookup_columns(self._table(layer, rank, "codes"), self._table(layer, rank, "scales"), rows)
                        full[layer][:len(tokens), begin * 256:end * 256] = decoded.flatten(1)
            for lane in range(tp):
                payload = payloads[dp * tp + lane]
                native = payload["metadata"]["model.layers.1.self_attn.swa_cache"]
                metadata = native["global_metadata"]
                if (not torch.equal(payload["positions"], pos)
                        or payload["query_start_loc"].tolist() != starts
                        or metadata["query_start_loc"].tolist() != starts):
                    raise ValueError("Host row coordinates differ from the original cache metadata")
                span = tuple(native["cp_token_range"])
                begin, end, physical, padded = span
                if (begin != lane * physical or end != begin + physical or padded != physical * tp
                        or padded < len(tokens) or physical > capacity):
                    raise ValueError("Host rows must retain the original SP/TP token partition")
                valid = max(0, min(end, len(tokens)) - min(begin, len(tokens)))
                local = {layer: torch.zeros_like(value) for layer, value in full.items()}
                local_mask = torch.zeros_like(mask)
                local_mask[:valid] = mask[begin:begin + valid]
                for layer in full:
                    local[layer][:valid] = full[layer][begin:begin + valid]
                records.append(dict(full_input_ids=ids, positions=pos, cp_token_range=span,
                    hashes=hashes, mask=mask, full=full, local=local, local_mask=local_mask))
        return dict(epoch=actual.get("epoch", 0), token_map_id=config["input_contract_sha256"],
            checkpoint=self.checkpoint, topology=deepcopy(topology), capacity=capacity, cases={"fresh": records},
            storage={str(key): dict(array.receipt) for key, array in self.arrays.items()},
            hash_constants=self.constants)


def load_rotation(checkpoint, hidden_size):
    """Load and validate repeated block32 checkpoint rotation by byte identity."""
    from safetensors import safe_open

    from .runtime_utils import file_sha

    path = Path(checkpoint) / "optional/quarot.safetensors"
    if not path.is_file():
        raise FileNotFoundError("The physical A3 Engram requires its checkpoint rotation")
    with safe_open(path, framework="pt", device="cpu") as source:
        rotation = source.get_tensor("global_rotation")
    if tuple(rotation.shape) != (hidden_size, hidden_size):
        raise ValueError("Checkpoint rotation has a different full hidden dimension")
    block = rotation[:32, :32].contiguous()
    zero = torch.zeros_like(block)
    for row in range(0, hidden_size, 32):
        for col in range(0, hidden_size, 32):
            expected = block if row == col else zero
            if not torch.equal(rotation[row:row + 32, col:col + 32], expected):
                raise ValueError("Checkpoint rotation is not repeated block32")
    return block.float(), {"path": str(path), "sha256": file_sha(path)}

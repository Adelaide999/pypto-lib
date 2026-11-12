# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run complete shared-host Engram lookup and compose its physical Prefill observations."""

from contextlib import ExitStack
from pathlib import Path
import torch
from .engram import make_prefill_engram_entries
from .engram_host import SharedEngramArray, lookup_columns
from .runtime_utils import Executable, file_sha


def load_rotation(checkpoint, hidden_size):
    """Load and validate repeated block32 checkpoint rotation by byte identity."""
    from safetensors import safe_open

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
            if not torch.equal(rotation[row : row + 32, col : col + 32], expected):
                raise ValueError("Checkpoint rotation is not repeated block32")
    return block.float(), {"path": str(path), "sha256": file_sha(path)}


def engram_owner(contract, layer_id, role=None):
    """Resolve the physical checkpoint prefix and original hash slot."""
    layer_ids = tuple(contract["hash_constants"]["layer_ids"])
    if role is None:
        if (
            layer_id not in (1, 14)
            or contract.get("physical_layer_id") != layer_id
            or layer_id not in layer_ids
        ):
            raise ValueError("Require the actual physical Engram owner in the original hash layout")
    else:
        from .cache import resolve_layer_role

        role = resolve_layer_role(layer_id, role)
        if role["engram_slot"] is None or layer_id not in layer_ids:
            raise ValueError("Require the original Engram owner role")
    slot = layer_ids.index(layer_id)
    if contract.get("engram_slot", slot) != slot or (role is not None and role["engram_slot"] != slot):
        raise ValueError("Engram slot differs from the original physical-layer hash layout")
    return f"layers.{layer_id}.engram", slot


def run_prefill_engram(
    contract,
    raw,
    output,
    reports,
    memory,
    exchange,
    layer_id,
    *,
    role=None,
    local_hidden=None,
    live_hash_caches=None,
):
    """Execute outer-model Engram before the shared complete physical-layer computation."""
    capacity, tp = contract["capacity"], contract["topology"]["tp"]
    constants, shards = contract["hash_constants"], contract["engram_shards"]
    prefix, slot = engram_owner(contract, layer_id, role)
    if (local_hidden is None) != (live_hash_caches is None):
        raise ValueError("Live Engram requires both actual hidden and request hash-cache state")
    storage = contract["engram_storage_contract"]
    required = dict(
        dtype="INT8",
        scale_dtype="FP32",
        group_size=32,
        cpu_offload=True,
        dp_shared_memory=True,
        global_columns=24,
    )
    if storage != required:
        raise ValueError("Require original complete INT8/FP32 DP-shared Engram storage")
    originals = [original for records in contract["cases"].values() for original in records]
    cache_rows = (
        originals[0]["hash_cache"].numel()
        if live_hash_caches is None
        else next(iter(live_hash_caches[0].values())).numel()
    )
    max_blocks = max(original["metadata"]["block_table"].shape[1] for original in originals)
    config = dict(contract["model_config"]["text_config"])
    config["image_token_id"] = contract["model_config"]["image_token_id"]
    config["image_pad_token_id"] = contract["model_config"].get(
        "image_pad_token_id", config["image_token_id"] + 1
    )
    functions = make_prefill_engram_entries(capacity, constants, config, shards, cache_rows, max_blocks, tp)
    entries = {name: Executable(function, output / "compiled" / name) for name, function in functions.items()}
    rotation, rotation_receipt = load_rotation(contract["checkpoint"], 5120)
    projection = raw[prefix + ".wkv.weight"].T.contiguous()
    if projection.dtype != torch.bfloat16 or tuple(projection.shape) != (6144, 25600):
        raise ValueError("Engram requires its complete original BF16 projection")
    states = [{} for _ in reports]
    packets = [{} for _ in reports]

    def zeros(*shape, dtype=torch.bfloat16):
        return torch.zeros(shape, dtype=dtype)

    def pad(value, rows=capacity, fill=0):
        target = torch.full((rows, *value.shape[1:]), fill, dtype=value.dtype)
        target[: len(value)].copy_(value)
        return target

    def execute(name, rank, case, arguments, outputs, roles):
        path = output / "captures" / f"rank-{rank}-{case}-{name}.pt"
        entries[name].execute(arguments, outputs, rank, roles, path, reports[rank], memory)

    for case, records in contract["cases"].items():
        for rank, original in enumerate(records):
            total = len(original["full_input_ids"])
            source_cache = (
                original["hash_cache"] if live_hash_caches is None else live_hash_caches[rank][case]
            )
            if total != len(original["positions"]) or source_cache.numel() != cache_rows:
                raise ValueError("Engram token/position/cache geometry differs")
            before = source_cache.clone()
            cache = before.clone() if live_hash_caches is None else source_cache
            hashes = zeros(capacity, len(constants["layer_ids"]), 24, dtype=torch.int32)
            mask = zeros(capacity, dtype=torch.int32)
            metadata = original["metadata"]
            query = pad(metadata["query_start_loc"], capacity + 1, total)
            blocks = torch.full((capacity, max_blocks), -1, dtype=torch.int32)
            source_blocks = metadata["block_table"]
            blocks[: source_blocks.shape[0], : source_blocks.shape[1]].copy_(source_blocks)
            lookback = torch.full((capacity, 3), -1, dtype=torch.int64)
            if "lookback_token_ids" in original:
                source_lookback = original["lookback_token_ids"]
                lookback[: len(source_lookback)].copy_(source_lookback)
            arguments = [
                pad(original["full_input_ids"]),
                pad(original["positions"]),
                query,
                pad(metadata["slot_mapping"], fill=-1),
                blocks,
                lookback,
                constants["token_map"],
                constants["multipliers"],
                constants["primes"].reshape(-1, 24),
                constants["offsets"],
                cache,
                hashes,
                mask,
                total,
                len(metadata["query_start_loc"]) - 1,
            ]
            execute(
                "engram_hash",
                rank,
                case,
                arguments,
                dict(cache=cache, hashes=hashes, mask=mask),
                "engram_hash",
            )
            states[rank][case] = dict(
                hash_cache_before=before, hash_cache_after=cache, hashes=hashes, mask=mask, original=original
            )
    with ExitStack() as stack:
        for tp_rank, shard in enumerate(shards):
            codes = SharedEngramArray(
                contract["checkpoint"],
                prefix + ".embed.weight",
                shard["row_range"],
                256,
                "I8",
                shard["host_payload_sha256"]["codes"],
                output.name + f"-tp{tp_rank}",
            )
            stack.callback(codes.close)
            scales = SharedEngramArray(
                contract["checkpoint"],
                prefix + ".embed.scale",
                shard["row_range"],
                8,
                "F32",
                shard["host_payload_sha256"]["scales"],
                output.name + f"-tp{tp_rank}",
            )
            stack.callback(scales.close)
            ranks = list(range(tp_rank, len(reports), tp))
            if ranks != shard["DP_sharing_ranks"]:
                raise ValueError("Engram DP sharing differs from original rank ownership")
            for rank in ranks:
                reports[rank]["engram_storage"] = dict(
                    codes=codes.receipt,
                    scales=scales.receipt,
                    DP_sharing_ranks=ranks,
                    rotation=rotation_receipt,
                )
                for case, state in states[rank].items():
                    total = len(state["original"]["full_input_ids"])
                    lookup = zeros(capacity, 6144 // tp)
                    begin, end = shard["head_range"]
                    ids = state["hashes"][:total, slot, begin:end]
                    lookup[:total] = lookup_columns(codes, scales, ids).flatten(1)
                    packets[rank][case] = lookup
                reports[rank]["engram_storage"]["lookup"] = "host_INT8_FP32_group32_to_BF16"
        gathered = exchange("engram-column-gather", packets, "all_gather", "tp")
    for rank, records in enumerate(states):
        reports[rank]["engram_storage"]["closed_after_all_DP_peers"] = True
        for case, state in records.items():
            original = state["original"]
            total, physical = len(original["full_input_ids"]), original["SP_physical_rows"]
            full, local = zeros(capacity, 6144), zeros(capacity, 6144)
            local_mask = zeros(capacity, dtype=torch.int32)
            arguments = [
                gathered[rank][case],
                state["mask"],
                full,
                local,
                local_mask,
                physical,
                total,
                rank % tp,
            ]
            execute(
                "engram_layout",
                rank,
                case,
                arguments,
                dict(full=full, local=local, local_mask=local_mask),
                "engram_token_shard",
            )
            projected = zeros(capacity, 25600)
            execute(
                "engram_projection",
                rank,
                case,
                [local, projection, projected, physical],
                dict(projected=projected),
                "engram_projection",
            )
            restored = zeros(capacity, 4, 5120, dtype=torch.float32)
            probability, hidden = zeros(capacity, 4, dtype=torch.float32), zeros(capacity, 4, 5120)
            hidden_input = pad(original["hidden"]) if local_hidden is None else local_hidden[rank][case]
            if tuple(hidden_input.shape) != (capacity, 4, 5120) or hidden_input.dtype != torch.bfloat16:
                raise ValueError("Live Engram requires local padded BF16 hidden")
            arguments = [
                hidden_input,
                projected,
                raw[prefix + ".q_weight"],
                raw[prefix + ".k_weight"],
                rotation,
                local_mask,
                restored,
                probability,
                hidden,
                physical,
            ]
            execute(
                "engram_gate",
                rank,
                case,
                arguments,
                dict(original=restored, probability=probability, hidden=hidden),
                "engram_gate",
            )
            state["hidden"] = hidden
            state["nodes"] = dict(
                engram_hash_ids=state["hashes"][:total].clone(),
                engram_token_mask=state["mask"][:total].to(torch.bool),
                engram_lookup_full=full[:total].clone(),
                engram_lookup_local=local[:physical].clone(),
                engram_projection=projected[:physical].clone(),
                engram_hidden_after=hidden[:physical].clone(),
            )
    return states

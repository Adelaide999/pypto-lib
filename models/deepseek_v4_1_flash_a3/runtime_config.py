# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Runtime sizing controls for A2/A3 operator entry points."""

from dataclasses import dataclass


MAX_CONTEXT_TOKENS = 16384
PREFILL_CAPACITIES = (16, 128, 256, 512, 1024, 2048, 4096, 8192)
MAX_PREFILL_CHUNK_SIZE = PREFILL_CAPACITIES[-1]
SPMD_WORKER_LIMIT = 256


def prefill_execution_capacity(chunk_size):
    """Select the bounded Prefill shape while retaining the 16-row Decode ABI."""
    if type(chunk_size) is not int or not 1 <= chunk_size <= MAX_PREFILL_CHUNK_SIZE:
        raise ValueError(f"Chunk size must be in 1..{MAX_PREFILL_CHUNK_SIZE}")
    return next(capacity for capacity in PREFILL_CAPACITIES if chunk_size <= capacity)


def network_ring_heap(capacity, config, tp):
    """Size sparse-gather and mixed-kernel pipeline heaps for a network admission."""
    if capacity not in PREFILL_CAPACITIES:
        raise ValueError("Unsupported network execution capacity")
    validate_tp(tp)
    default = 256 * 1024 * 1024
    query_rows = (capacity + tp - 1) // tp
    selected_bytes = query_rows * (config["sliding_window"] + config["index_topk"]) * config["head_dim"] * 2
    hidden, hc = config["hidden_size"], config["hc_mult"]
    intermediate = ((config["moe_intermediate_size"] // tp + 255) // 256) * 256
    # Full-capacity shared-expert partials and unrolled quantized projections.
    shared_bytes = capacity * tp * (hidden * 7 + intermediate * 9 + 32)
    # HC preparation, inverse scale, rounded function and function accumulation.
    hc_width = hc * (hc + 2)
    hc_bytes = capacity * (hidden * hc * 4 + 32 + hc_width * 4) + hc_width * hidden * hc * 4
    outer_bytes = max(selected_bytes, shared_bytes, hc_bytes)
    # Index selection keeps one 16384-element FP32 pipeline per query row.
    pipe_bytes = capacity * 16384 * 4
    if max(outer_bytes, pipe_bytes) <= default:
        return None
    outer_heap = 1 << (outer_bytes + 1024 * 1024 - 1).bit_length() if outer_bytes > default else 0
    inner_heap = 1 << (pipe_bytes + 1024 * 1024 - 1).bit_length() if pipe_bytes > default else 0
    return (0, outer_heap, inner_heap, 0)


def validate_tp(tp):
    """Validate a caller-supplied tensor-parallel specialization."""
    if type(tp) is not int or tp not in (1, 2, 4):
        raise ValueError("Tensor parallelism must be 1, 2 or 4")
    return tp


@dataclass(frozen=True)
class ExpertParallelConfig:
    """Describe contiguous global expert ownership across DP-by-TP ranks."""

    dp: int
    tp: int
    ep: int
    experts: int = 384

    def __post_init__(self):
        if self.tp not in (1, 2, 4) or self.dp < 1 or self.ep != self.dp * self.tp:
            raise ValueError("Expert parallelism must span the DP-by-TP ranks with TP1/2/4")
        if self.experts % self.ep:
            raise ValueError("Global expert count must divide evenly across EP ranks")

    def ownership(self, rank):
        """Return global expert IDs paired with their local checkpoint slots."""
        if not 0 <= rank < self.ep:
            raise ValueError("Logical rank is outside the expert group")
        count = self.experts // self.ep
        return [[rank * count + slot, slot] for slot in range(count)]

    def owner(self, expert):
        """Resolve a global expert without changing its identity."""
        if not 0 <= expert < self.experts:
            raise ValueError("Global expert ID is outside the checkpoint")
        return expert // (self.experts // self.ep)

    def active_rows(self, controls, rank):
        """Read rank-local token counts for TP1 or an explicit TP2/4 partition."""
        if not 0 <= rank < self.ep:
            raise ValueError("Logical rank is outside the expert group")
        if "active_by_rank" in controls:
            counts = controls["active_by_rank"]
            if len(counts) != self.ep:
                raise ValueError("Rank-local token counts must cover the complete EP group")
            return int(counts[rank])
        if self.tp != 1:
            raise ValueError("TP2/4 require explicit active_by_rank counts and rank-local input rows")
        return int(controls["active_by_dp"][rank])


def validate_devices(devices, *, world):
    """Validate physical-to-logical device mapping without changing process state."""
    if not isinstance(devices, (list, tuple)) or any(type(value) not in (int, str) for value in devices):
        raise ValueError("Devices must be an explicit list of nonnegative physical IDs")
    assigned = [str(value) for value in devices]
    if len(assigned) != world or len(set(assigned)) != world or any(not value.isdecimal() for value in assigned):
        raise ValueError("Devices must contain one distinct nonnegative physical ID per rank")
    assigned = [str(int(value)) for value in assigned]
    if len(set(assigned)) != world:
        raise ValueError("Physical device aliases cannot identify different logical ranks")
    return assigned


def configure_devices(devices, *, world):
    """Bind explicit physical devices to logical ranks before runtime initialization."""
    import os

    assigned = validate_devices(devices, world=world)
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible is not None and visible != ",".join(assigned):
        raise ValueError("Explicit devices differ from the existing visible-device mapping")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(assigned)
    return assigned


__all__ = ["ExpertParallelConfig", "configure_devices", "validate_devices", "validate_tp"]

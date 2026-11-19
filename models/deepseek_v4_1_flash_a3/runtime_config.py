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
MAX_MODEL_CONTEXT_TOKENS = 1_048_576
PREFILL_CAPACITIES = (16, 128, 256, 512, 1024, 2048, 4096, 8192)
MAX_PREFILL_CHUNK_SIZE = PREFILL_CAPACITIES[-1]
PREFILL_MOE_SLAB_ROWS = (16, 64, 128)
SPMD_WORKER_LIMIT = 256


def prefill_execution_capacity(chunk_size):
    """Select the bounded Prefill shape while retaining the 16-row Decode ABI."""
    if type(chunk_size) is not int or not 1 <= chunk_size <= MAX_PREFILL_CHUNK_SIZE:
        raise ValueError(f"Chunk size must be in 1..{MAX_PREFILL_CHUNK_SIZE}")
    return next(capacity for capacity in PREFILL_CAPACITIES if chunk_size <= capacity)


def prefill_moe_slab_limit(capacity, requested=None):
    """Select the source rows per rank allowed in one expert exchange."""
    if requested is None:
        requested = 64 if capacity == MAX_PREFILL_CHUNK_SIZE else 16
    if type(requested) is not int or requested not in PREFILL_MOE_SLAB_ROWS:
        raise ValueError(f"Prefill MoE slab rows must be one of {PREFILL_MOE_SLAB_ROWS}")
    return min(requested, capacity)


def network_ring_heap(capacity, config, tp, *, routed_rows=0):
    """Size attention, shared and grouped expert workspaces for a network admission."""
    if capacity != 64 and capacity not in PREFILL_CAPACITIES:
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
    # Grouped expert input codes, gate/up INT32, activated codes, scales and scheduling maps.
    routed_bytes = routed_rows * (hidden + 9 * config["moe_intermediate_size"] + 32 + 8 + 4) + 64
    outer_bytes = max(selected_bytes, shared_bytes, hc_bytes, routed_bytes)
    # Index selection keeps one 16384-element FP32 pipeline per query row.
    # Grouped down uses two 16-by-64 FP32 pipeline tiles per active row tile and 256 output columns.
    routed_pipe_bytes = (routed_rows // 16) * (hidden // 256) * 2 * 16 * 64 * 4
    pipe_bytes = max(capacity * 16384 * 4, routed_pipe_bytes)
    if max(outer_bytes, pipe_bytes) <= default:
        return None
    outer_heap = 1 << (outer_bytes + 1024 * 1024 - 1).bit_length() if outer_bytes > default else 0
    inner_heap = 1 << (pipe_bytes + 1024 * 1024 - 1).bit_length() if pipe_bytes > default else 0
    return (0, outer_heap, inner_heap, 0)


@dataclass(frozen=True)
class RuntimeLimits:
    """Separate live requests, new query tokens, history and physical buckets."""

    max_num_seqs: int = 16
    max_num_batched_tokens: int = 16
    max_model_len: int = MAX_MODEL_CONTEXT_TOKENS
    max_seqs_per_dp: int = 16
    token_buckets: tuple = (16, 64, 256, 512, 1024, 2048)
    page_size: int = 128
    index_tile_rows: int = 2048

    def __post_init__(self):
        object.__setattr__(self, "token_buckets", tuple(self.token_buckets))
        values = (self.max_num_seqs, self.max_num_batched_tokens, self.max_model_len,
                  self.max_seqs_per_dp, self.page_size, self.index_tile_rows)
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("Runtime limits must be positive integers")
        if self.max_model_len > MAX_MODEL_CONTEXT_TOKENS or self.page_size != 128:
            raise ValueError(f"A3 uses 128-token pages and at most {MAX_MODEL_CONTEXT_TOKENS} original positions")
        if (not self.token_buckets or tuple(sorted(set(self.token_buckets))) != self.token_buckets
                or any(type(value) is not int or value < 16 or value % 16 for value in self.token_buckets)):
            raise ValueError("Token buckets must be increasing positive multiples of sixteen")
        if self.max_num_batched_tokens > self.token_buckets[-1]:
            raise ValueError("New-token budget exceeds the largest physical bucket")
        if self.index_tile_rows != 2048:
            raise ValueError("The A3 QLI scratch tile has 2048 rows, independent of history length")

    def token_capacity(self, tokens):
        """Select physical capacity without putting the current token count in a compile key."""
        if type(tokens) is not int or not 0 <= tokens <= self.max_num_batched_tokens:
            raise ValueError("Dispatch exceeds its independent new-token budget")
        return next(value for value in self.token_buckets if value >= max(1, tokens))


@dataclass(frozen=True)
class ParallelPlan:
    """Map one DP-by-TP world onto an expert group and explicit physical devices."""

    devices: tuple
    dp: int = 4
    tp: int = 4
    ep: int = 16
    experts: int = 384

    def __post_init__(self):
        if any(type(value) is not int for value in (self.dp, self.tp, self.ep, self.experts)):
            raise ValueError("Parallel dimensions must be integers")
        ExpertParallelConfig(self.dp, self.tp, self.ep, self.experts)
        if self.ep != 16:
            raise ValueError("The complete A3 network currently deploys one sixteen-rank EP world")
        assigned = tuple(int(value) for value in validate_devices(self.devices, world=self.ep))
        object.__setattr__(self, "devices", assigned)

    @classmethod
    def from_topology(cls, topology, devices=None):
        if any(topology.get(name, 1) != 1 for name in ("pp", "pcp", "dcp")):
            raise ValueError("A3 request ownership requires PP1/PCP1/DCP1")
        return cls(tuple(range(topology["ep"])) if devices is None else tuple(devices),
                   topology["dp"], topology["tp"], topology["ep"])

    @property
    def topology(self):
        return dict(dp=self.dp, tp=self.tp, ep=self.ep, pp=1, pcp=1, dcp=1)

    @property
    def tp_groups(self):
        return tuple(tuple(range(dp * self.tp, (dp + 1) * self.tp)) for dp in range(self.dp))

    @property
    def dp_groups(self):
        return tuple(tuple(range(lane, self.ep, self.tp)) for lane in range(self.tp))

    @property
    def ep_groups(self):
        return (tuple(range(self.ep)),)

    def rank(self, dp, tp=0):
        if type(dp) is not int or type(tp) is not int or not 0 <= dp < self.dp or not 0 <= tp < self.tp:
            raise ValueError("Request rank is outside the parallel plan")
        return dp * self.tp + tp

    def expert_ownership(self, rank):
        return ExpertParallelConfig(self.dp, self.tp, self.ep, self.experts).ownership(rank)

    def token_range(self, rank, tokens):
        if type(rank) is not int or not 0 <= rank < self.ep or type(tokens) is not int or tokens < 0:
            raise ValueError("Invalid rank-local query extent")
        rows = (tokens + self.tp - 1) // self.tp
        begin = rank % self.tp * rows
        return begin, begin + rows, rows, rows * self.tp

    def validate_model(self, text):
        for name in ("num_attention_heads", "o_groups", "vocab_size", "index_n_heads"):
            if name in text and (type(text[name]) is not int or text[name] % self.tp):
                raise ValueError(f"{name} cannot be sharded over the selected TP group")
        return self


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


__all__ = ["ExpertParallelConfig", "MAX_MODEL_CONTEXT_TOKENS", "ParallelPlan", "RuntimeLimits", "configure_devices",
           "validate_devices", "validate_tp"]

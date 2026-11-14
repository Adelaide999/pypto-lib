# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Lazy reader for the Aurora W8A8 Decode checkpoint layout.

The reader owns no model computation.  It resolves a tensor name through the
official safetensors index and loads only that tensor on the host, leaving
PyPTO to own all device computation and quantization boundaries.
"""


import json
from contextlib import ExitStack, contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator


class DecodeSafetensorsCheckpoint:
    """Resolve and lazily load tensors from an indexed Aurora export."""

    INDEX_NAME = "quant_model_weights.safetensors.index.json"

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {self.root}")
        index_path = self.root / self.INDEX_NAME
        if not index_path.is_file():
            raise FileNotFoundError(f"safetensors index is missing: {index_path}")
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"safetensors index has no weight_map: {index_path}")
        self.weight_map: dict[str, str] = {str(name): str(shard) for name, shard in weight_map.items()}
        self.config = self._load_json("config.json")
        self.quantization_description = self._load_json("quant_model_description.json")
        self._file_stack = None
        self._open_files = {}
        # These counters are deliberately owned by the reader rather than by
        # the network so tests and runtime reports can prove that residency is
        # a one-time operation.
        self.read_calls = 0
        self.read_seconds = 0.0
        self.read_counts: dict[str, int] = {}

    def _load_json(self, name: str) -> dict[str, Any]:
        path = self.root / name
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint metadata is missing: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"checkpoint metadata must be an object: {path}")
        return value

    def has(self, name: str) -> bool:
        return name in self.weight_map or (self.root / name).is_file()

    def shard_for(self, name: str) -> Path:
        try:
            shard = self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"tensor is absent from the safetensors index: {name}") from exc
        path = self.root / shard
        if not path.is_file():
            raise FileNotFoundError(f"indexed shard is missing: {path}")
        return path

    @contextmanager
    def cached_files(self):
        """Retain file handles during one layer's streaming weight conversion."""
        if self._file_stack is not None:
            raise RuntimeError("Checkpoint file-cache scopes cannot be nested")
        with ExitStack() as stack:
            self._file_stack = stack
            try:
                yield self
            finally:
                self._file_stack = None
                self._open_files.clear()

    @contextmanager
    def open_tensor(self, name: str) -> Iterator[Any]:
        """Yield one safetensors handle and the requested tensor name."""

        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Aurora Decode weights require the safetensors package") from exc
        path = self.shard_for(name)
        if self._file_stack is not None:
            if path not in self._open_files:
                self._open_files[path] = self._file_stack.enter_context(
                    safe_open(str(path), framework="pt", device="cpu")
                )
            yield self._open_files[path], name
            return
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if name not in handle.keys():
                raise KeyError(f"tensor {name!r} is not present in indexed shard {path}")
            yield handle, name

    def tensor(self, name: str) -> Any:
        """Load one host tensor and account for the checkpoint read."""

        begin = perf_counter()
        try:
            with self.open_tensor(name) as (handle, tensor_name):
                return handle.get_tensor(tensor_name)
        finally:
            self.read_calls += 1
            self.read_seconds += perf_counter() - begin
            self.read_counts[name] = self.read_counts.get(name, 0) + 1

    def require(self, names: list[str] | tuple[str, ...]) -> dict[str, Any]:
        return {name: self.tensor(name) for name in names}

    def layer_tensor_names(self, layer: int) -> tuple[str, ...]:
        if layer < 0:
            raise ValueError("layer must be non-negative")
        prefix = f"layers.{layer}."
        return tuple(sorted(name for name in self.weight_map if name.startswith(prefix)))

    def layer_shards(self, layer: int) -> tuple[Path, ...]:
        return tuple(sorted({self.shard_for(name) for name in self.layer_tensor_names(layer)}))

    def engram_tensor(self, layer: int, kind: str) -> Any:
        if layer not in (1, 14):
            raise ValueError("Aurora Engram tables are defined for layers 1 and 14")
        if kind not in ("weight", "scale"):
            raise ValueError("Engram kind must be 'weight' or 'scale'")
        path = self.root / f"engram_embed_{kind}_l{layer}.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"Engram tensor file is missing: {path}")
        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Aurora Engram weights require the safetensors package") from exc
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            names = list(handle.keys())
            if len(names) != 1:
                raise ValueError(f"expected one tensor in {path}, found {names}")
            return handle.get_tensor(names[0])


__all__ = ["DecodeSafetensorsCheckpoint"]


# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-only adapter from Aurora safetensors to the PyPTO Decode weight ABI.

The checkpoint stores logical row-major W8A8 matrices.  The routed-expert
kernel consumes NZ matrices and an aligned W2 reduction width, while the
shared-expert kernel consumes logical matrices and performs its own packing.
This module owns that representation change in the host weight-loading path.
It never executes model computation.
"""


from collections.abc import Iterator
from typing import Any

import torch

from .utils import pack_nz


HIDDEN = 5120
INTERMEDIATE = 2304
INTERMEDIATE_PADDED = 2560
EXPERTS = 384
LOCAL_EXPERTS = 192
TOP_K = 6


def _name(layer: int, suffix: str) -> str:
    if layer < 0:
        raise ValueError(f"layer must be non-negative, got {layer}")
    return f"layers.{layer}.ffn.{suffix}"


def _expert_name(layer: int, expert: int, suffix: str) -> str:
    if not 0 <= expert < EXPERTS:
        raise ValueError(f"expert must be in [0, {EXPERTS}), got {expert}")
    return _name(layer, f"experts.{expert}.{suffix}")


def _require(tensor: torch.Tensor, name: str, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    if tensor.dtype != dtype or tuple(tensor.shape) != shape:
        raise ValueError(
            f"{name} contract mismatch: got dtype={tensor.dtype}, shape={tuple(tensor.shape)}; "
            f"expected dtype={dtype}, shape={shape}"
        )
    return tensor.contiguous()


def _require_zero_offset(tensor: torch.Tensor, name: str, shape: tuple[int, ...]) -> None:
    offset = _require(tensor, name, torch.float32, shape)
    if bool(torch.count_nonzero(offset).item()):
        raise ValueError(
            f"{name} contains non-zero affine offsets; the current PyPTO W8A8 ABI "
            "supports the verified symmetric zero-offset export only"
        )


def _scale(tensor: torch.Tensor, name: str, channels: int) -> torch.Tensor:
    value = _require(tensor, name, torch.float32, (channels, 1))
    return value.reshape(channels).contiguous()


def _pad_w2(weight: torch.Tensor, name: str) -> torch.Tensor:
    logical = _require(weight, name, torch.int8, (HIDDEN, INTERMEDIATE))
    padded = torch.zeros((HIDDEN, INTERMEDIATE_PADDED), dtype=torch.int8)
    padded[:, :INTERMEDIATE].copy_(logical)
    return padded


def pack_routed_expert(
    checkpoint: DecodeSafetensorsCheckpoint,
    layer: int,
    expert: int,
) -> dict[str, torch.Tensor]:
    """Load and pack one routed expert for ``decode_ep2_rank``.

    W1/W3 are already in the kernel's logical ``[out, in]`` orientation.
    W2 is also ``[hidden, intermediate]``; only its reduction tail is padded
    from 2304 to 2560 for the A3 INT8 matmul tile contract.
    """

    w1 = _require(
        checkpoint.tensor(_expert_name(layer, expert, "w1.weight")),
        _expert_name(layer, expert, "w1.weight"),
        torch.int8,
        (INTERMEDIATE, HIDDEN),
    )
    w3 = _require(
        checkpoint.tensor(_expert_name(layer, expert, "w3.weight")),
        _expert_name(layer, expert, "w3.weight"),
        torch.int8,
        (INTERMEDIATE, HIDDEN),
    )
    w2_name = _expert_name(layer, expert, "w2.weight")
    w2 = _pad_w2(checkpoint.tensor(w2_name), w2_name)

    w1_scale_name = _expert_name(layer, expert, "w1.weight_scale")
    w3_scale_name = _expert_name(layer, expert, "w3.weight_scale")
    w2_scale_name = _expert_name(layer, expert, "w2.weight_scale")
    w1_offset_name = _expert_name(layer, expert, "w1.weight_offset")
    w3_offset_name = _expert_name(layer, expert, "w3.weight_offset")
    w2_offset_name = _expert_name(layer, expert, "w2.weight_offset")
    _require_zero_offset(checkpoint.tensor(w1_offset_name), w1_offset_name, (INTERMEDIATE, 1))
    _require_zero_offset(checkpoint.tensor(w3_offset_name), w3_offset_name, (INTERMEDIATE, 1))
    _require_zero_offset(checkpoint.tensor(w2_offset_name), w2_offset_name, (HIDDEN, 1))

    return {
        "w1": pack_nz(w1),
        "w1_scale": _scale(checkpoint.tensor(w1_scale_name), w1_scale_name, INTERMEDIATE),
        "w3": pack_nz(w3),
        "w3_scale": _scale(checkpoint.tensor(w3_scale_name), w3_scale_name, INTERMEDIATE),
        "w2": pack_nz(w2),
        "w2_scale": _scale(checkpoint.tensor(w2_scale_name), w2_scale_name, HIDDEN),
    }


def iter_routed_experts(
    checkpoint: DecodeSafetensorsCheckpoint,
    layer: int,
    rank: int,
    *,
    ep_size: int = 2,
) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
    """Yield one rank's owner experts without retaining a second full shard."""

    if ep_size not in (2, 16) or not 0 <= rank < ep_size:
        raise ValueError("Require a logical rank in an EP2 or EP16 group")
    count = EXPERTS // ep_size
    first = rank * count
    for expert in range(first, first + count):
        yield expert, pack_routed_expert(checkpoint, layer, expert)


def load_routed_shard(
    checkpoint: DecodeSafetensorsCheckpoint,
    layer: int,
    rank: int,
    *,
    ep_size: int = 2,
) -> dict[str, torch.Tensor]:
    """Materialize one rank's complete EP2 or EP16 shard in external NZ layout.

    The ``pl.NZ`` annotation is logical in the Python signature, but the
    distributed host ABI passes the compiler's physical five-dimensional
    view.  The first dimension is the local expert, followed by
    ``[C / C0, R / 16, 16, C0]``.  Keeping that shape here prevents the
    validation runner from silently treating packed bytes as row-major data.
    """

    result = {}
    for index, (_, packed) in enumerate(iter_routed_experts(checkpoint, layer, rank, ep_size=ep_size)):
        for name, value in packed.items():
            target = "routed_" + name
            if not name.endswith("_scale"):
                rows, cols = value.shape
                value = value.reshape(cols // 32, rows // 16, 16, 32)
            if target not in result:
                result[target] = torch.empty((EXPERTS // ep_size, *value.shape), dtype=value.dtype)
            result[target][index].copy_(value)
    return result


def load_router(checkpoint: DecodeSafetensorsCheckpoint, layer: int) -> dict[str, torch.Tensor]:
    """Load the replicated FP32 router operands for one physical layer."""

    gate_name = _name(layer, "gate.weight")
    text_name = _name(layer, "gate.bias")
    vision_name = _name(layer, "gate.bias_vl")
    gate = _require(checkpoint.tensor(gate_name), gate_name, torch.bfloat16, (EXPERTS, HIDDEN))
    text = _require(checkpoint.tensor(text_name), text_name, torch.float32, (EXPERTS,))
    vision = _require(checkpoint.tensor(vision_name), vision_name, torch.float32, (EXPERTS,))
    return {
        "gate_weight": gate.float().contiguous(),
        "text_bias": text,
        "vision_bias": vision,
    }


def load_shared_expert(checkpoint: DecodeSafetensorsCheckpoint, layer: int) -> dict[str, torch.Tensor]:
    """Load shared W8A8 weights in the logical ABI consumed by ``moe_shared``."""

    w1_name = _name(layer, "shared_experts.w1.weight")
    w3_name = _name(layer, "shared_experts.w3.weight")
    w2_name = _name(layer, "shared_experts.w2.weight")
    w1 = _require(checkpoint.tensor(w1_name), w1_name, torch.int8, (INTERMEDIATE, HIDDEN))
    w3 = _require(checkpoint.tensor(w3_name), w3_name, torch.int8, (INTERMEDIATE, HIDDEN))
    w2 = _require(checkpoint.tensor(w2_name), w2_name, torch.int8, (HIDDEN, INTERMEDIATE))
    _require_zero_offset(
        checkpoint.tensor(_name(layer, "shared_experts.w1.weight_offset")),
        _name(layer, "shared_experts.w1.weight_offset"),
        (INTERMEDIATE, 1),
    )
    _require_zero_offset(
        checkpoint.tensor(_name(layer, "shared_experts.w3.weight_offset")),
        _name(layer, "shared_experts.w3.weight_offset"),
        (INTERMEDIATE, 1),
    )
    _require_zero_offset(
        checkpoint.tensor(_name(layer, "shared_experts.w2.weight_offset")),
        _name(layer, "shared_experts.w2.weight_offset"),
        (HIDDEN, 1),
    )
    w1_scale_name = _name(layer, "shared_experts.w1.weight_scale")
    w3_scale_name = _name(layer, "shared_experts.w3.weight_scale")
    w2_scale_name = _name(layer, "shared_experts.w2.weight_scale")
    return {
        "shared_gate_up": torch.cat((w1.transpose(0, 1), w3.transpose(0, 1)), dim=1).contiguous(),
        "shared_gate_up_scale": torch.cat(
            (
                _scale(checkpoint.tensor(w1_scale_name), w1_scale_name, INTERMEDIATE),
                _scale(checkpoint.tensor(w3_scale_name), w3_scale_name, INTERMEDIATE),
            )
        ).contiguous(),
        "shared_down": w2.transpose(0, 1).contiguous(),
        "shared_down_scale": _scale(checkpoint.tensor(w2_scale_name), w2_scale_name, HIDDEN),
    }


def load_ep2_rank_weights(
    checkpoint: DecodeSafetensorsCheckpoint,
    layer: int,
    rank: int,
) -> dict[str, torch.Tensor]:
    """Materialize all real checkpoint operands required by one EP2 rank."""

    result: dict[str, torch.Tensor] = {}
    result.update(load_router(checkpoint, layer))
    result.update(load_routed_shard(checkpoint, layer, rank))
    result.update(load_shared_expert(checkpoint, layer))
    return result


def abi_summary() -> dict[str, Any]:
    """Return the fixed host-to-PyPTO layout without reading checkpoint data."""

    return {
        "source": "Aurora safetensors W8A8_DYNAMIC",
        "offset_policy": "reject_nonzero",
        "routed": {
            "w1_w3_logical": [INTERMEDIATE, HIDDEN],
            "w1_w3_device": [LOCAL_EXPERTS, HIDDEN // 32, INTERMEDIATE // 16, 16, 32],
            "w2_logical": [HIDDEN, INTERMEDIATE],
            "w2_device": [LOCAL_EXPERTS, INTERMEDIATE_PADDED // 32, HIDDEN // 16, 16, 32],
            "scale_w1_w3": [LOCAL_EXPERTS, INTERMEDIATE],
            "scale_w2": [LOCAL_EXPERTS, HIDDEN],
            "layout": "FRACTAL_NZ for all routed matrices; EP host shape flattens to [EP*LOCAL_EXPERTS, ...]",
        },
        "shared": {
            "gate_up": [HIDDEN, INTERMEDIATE * 2],
            "down": [INTERMEDIATE, HIDDEN],
            "layout": "logical row-major; moe_shared packs at kernel entry",
        },
        "router": {"gate_weight": [EXPERTS, HIDDEN], "bias": [EXPERTS]},
    }


__all__ = [
    "abi_summary",
    "iter_routed_experts",
    "load_ep2_rank_weights",
    "load_router",
    "load_routed_shard",
    "load_shared_expert",
    "pack_routed_expert",
]


# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2 host-side Engram hash/history contract.

This module owns the part of Engram that runs before the PyPTO BF16 table
lookup: compressed token ids, request lookback precedence, rolling hash and
bounded host-table gathers.  The arithmetic mirrors vLLM's
``vllm.models.deepseek_v41.common.engram`` and the Ascend
``_hash_ids_kernel``.  It deliberately does not pretend to implement the
384,006,168-row host UVM table or a cross-rank collective; those remain
explicit integration boundaries.
"""


from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


DEAD_ID = -1


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % p == 0:
            return value == p
    d, r = value - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, value)
        if x in (1, value - 1):
            continue
        for _ in range(r - 1):
            x = x * x % value
            if x == value - 1:
                break
        else:
            return False
    return True


def _next_prime(start: int, seen: set[int]) -> int:
    candidate = start + 1
    while candidate in seen or not _is_prime(candidate):
        candidate += 1
    return candidate


@dataclass(frozen=True)
class HashLayout:
    """Frozen V4.1 hash parameters for the configured Engram layers."""

    layer_ids: tuple[int, ...]
    multipliers: torch.Tensor  # [layers, max_ngram]
    primes: torch.Tensor  # [layers, (max_ngram - 1) * heads]
    offsets: torch.Tensor  # [layers, (max_ngram - 1) * heads]
    max_ngram_size: int
    n_heads: int
    compressed_vocab_size: int
    pad_token_id: int
    vocab_size: int

    @property
    def lookback_depth(self) -> int:
        return self.max_ngram_size - 1

    @property
    def n_hash_cols(self) -> int:
        return self.lookback_depth * self.n_heads


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Build the V4.1 normalized token map used by the hash state.

    The tokenizer object is deliberately supplied by the host runner.  This
    keeps tokenizer I/O out of PyPTO graphs while preserving the exact
    normalization contract (including one-space and partial UTF-8 tokens).
    """
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def make_layout(
    layer_ids: Sequence[int] = (1, 14),
    *,
    max_ngram_size: int = 4,
    n_heads: int = 8,
    compressed_vocab_size: int = 99092,
    vocab_size: int = 16_000_000,
    pad_token_id: int = 2,
) -> HashLayout:
    """Construct the same deterministic layout as upstream vLLM."""
    layers = tuple(int(x) for x in layer_ids)
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // compressed_vocab_size) // 2)
    multipliers = []
    primes = []
    offsets = []
    seen: set[int] = set()
    for layer_id in layers:
        rng = np.random.default_rng(10007 * layer_id)
        values = rng.integers(0, bound, size=(max_ngram_size,), dtype=np.int64)
        multipliers.append(torch.from_numpy(values * 2 + 1))
        layer_primes: list[int] = []
        for _ in range(max_ngram_size - 1):
            for _ in range(n_heads):
                prime = _next_prime(vocab_size - 1, seen)
                seen.add(prime)
                layer_primes.append(prime)
        primes.append(torch.tensor(layer_primes, dtype=torch.int64))
        running = 0
        layer_offsets = []
        for prime in layer_primes:
            layer_offsets.append(running)
            running += prime
        offsets.append(torch.tensor(layer_offsets, dtype=torch.int64))
    return HashLayout(
        layers,
        torch.stack(multipliers),
        torch.stack(primes),
        torch.stack(offsets),
        max_ngram_size,
        n_heads,
        compressed_vocab_size,
        pad_token_id,
        vocab_size,
    )


def update_slot_history(
    input_ids: torch.Tensor,
    token_map: torch.Tensor,
    dead_mask: torch.Tensor,
    slot_mapping: torch.Tensor,
    hash_cache: torch.Tensor,
) -> None:
    """Publish compressed ids to physical slots, matching Ascend's writer."""
    if input_ids.ndim != 1 or token_map.ndim != 1 or dead_mask.ndim != 1:
        raise ValueError("input_ids, token_map and dead_mask must be vectors")
    if slot_mapping.shape != input_ids.shape:
        raise ValueError("slot_mapping shape mismatch")
    valid = (slot_mapping >= 0) & (slot_mapping < hash_cache.numel())
    values = token_map[input_ids.clamp(0, token_map.numel() - 1).long()].to(torch.int32)
    values = torch.where(dead_mask.bool(), torch.full_like(values, DEAD_ID), values)
    hash_cache[slot_mapping[valid].long()] = values[valid]


def hash_ids(
    input_ids: torch.Tensor,
    token_map: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    layout: HashLayout,
    dead_mask: torch.Tensor | None = None,
    lookback_token_ids: torch.Tensor | None = None,
    lookback_dead_mask: torch.Tensor | None = None,
    hash_cache: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    cache_block_size: int = 128,
) -> torch.Tensor:
    """Generate ``[tokens, layers, 24]`` hash ids with vLLM precedence.

    Current chunk tokens win over explicit request lookback, which wins over
    the physical slot cache.  A dead token or a missing/negative history
    position blocks the rolling hash and emits ``pad_token_id`` for all later
    n-gram sizes at that position.
    """
    input_ids = input_ids.to(torch.int64).flatten()
    token_map = token_map.to(torch.int64).flatten()
    positions = positions.to(torch.int64).flatten()
    query_start_loc = query_start_loc.to(torch.int64).flatten()
    n_tokens = int(input_ids.numel())
    if positions.numel() != n_tokens or dead_mask is not None and dead_mask.numel() != n_tokens:
        raise ValueError("input/position/dead-mask lengths differ")
    dead = torch.zeros(n_tokens, dtype=torch.bool) if dead_mask is None else dead_mask.bool().flatten()
    if query_start_loc.numel() < 2 or int(query_start_loc[-1]) != n_tokens:
        raise ValueError("query_start_loc must bound every input token")
    lookback_depth = layout.lookback_depth
    if lookback_token_ids is None:
        lookback_token_ids = torch.full((query_start_loc.numel() - 1, lookback_depth), -1, dtype=torch.int64)
    else:
        lookback_token_ids = lookback_token_ids.to(torch.int64)
    if lookback_dead_mask is None:
        lookback_dead_mask = torch.zeros_like(lookback_token_ids, dtype=torch.bool)
    else:
        lookback_dead_mask = lookback_dead_mask.bool()
    if lookback_token_ids.shape != lookback_dead_mask.shape:
        raise ValueError("lookback token and mask shapes differ")
    output = torch.full((n_tokens, len(layout.layer_ids), layout.n_hash_cols), layout.pad_token_id, dtype=torch.int32)
    if n_tokens == 0:
        return output
    req_ends = query_start_loc[1:]
    for token in range(n_tokens):
        # The native binary search advances when token >= end, i.e. upper
        # bound.  A token exactly at a request boundary belongs to the next
        # request rather than the preceding chunk.
        req = int(torch.searchsorted(req_ends, torch.tensor(token), right=True))
        req = min(req, query_start_loc.numel() - 2)
        chunk_start = int(positions[int(query_start_loc[req])])
        rolling = [0] * len(layout.layer_ids)
        blocked = False
        for shift in range(layout.max_ngram_size):
            lookback = int(positions[token]) - shift
            in_batch = lookback >= chunk_start
            batch_idx = max(token - shift, 0)
            source: int
            source_dead = False
            if in_batch:
                source = int(token_map[int(input_ids[batch_idx])])
                source_dead = bool(dead[batch_idx])
            else:
                col = chunk_start - 1 - lookback
                in_window = 0 <= col < lookback_depth
                known = in_window and int(lookback_token_ids[req, col]) >= 0
                if known:
                    source_token = int(lookback_token_ids[req, col])
                    source = int(token_map[source_token])
                    source_dead = bool(lookback_dead_mask[req, col])
                elif hash_cache is not None and block_table is not None:
                    clamped = min(max(lookback, 0), int(block_table.shape[1]) * cache_block_size - 1)
                    block = int(block_table[min(req, block_table.shape[0] - 1), clamped // cache_block_size])
                    slot = min(max(block * cache_block_size + clamped % cache_block_size, 0), hash_cache.numel() - 1)
                    source = int(hash_cache[slot])
                else:
                    source = layout.pad_token_id
            blocked = blocked or lookback < 0 or source_dead or source == DEAD_ID
            value = layout.pad_token_id if blocked else source
            for li in range(len(layout.layer_ids)):
                rolling[li] ^= int(value) * int(layout.multipliers[li, shift])
                if shift:
                    base = (shift - 1) * layout.n_heads
                    for head in range(layout.n_heads):
                        col = base + head
                        output[token, li, col] = int(rolling[li] % int(layout.primes[li, col]) + int(layout.offsets[li, col]))
    return output


def bounded_host_gather(
    table: torch.Tensor,
    row_ids: torch.Tensor,
    *,
    full_rows: int,
    chunk_rows: int = 32,
) -> torch.Tensor:
    """Gather a bounded host table window with explicit range checks.

    ``table`` is a resident or mmap-backed host tensor.  The function records
    the full checkpoint row count separately so callers cannot accidentally
    treat a 4,096-row audit window as the 384,006,168-row production table.
    """
    if table.ndim != 2 or row_ids.ndim != 2:
        raise ValueError("table and row_ids must be rank-2")
    if full_rows < table.shape[0] or row_ids.numel() and (int(row_ids.min()) < 0 or int(row_ids.max()) >= full_rows):
        raise ValueError("row id outside declared full table range")
    if int(row_ids.max()) >= table.shape[0]:
        raise LookupOutsideResidentWindow(int(row_ids.max()), int(table.shape[0]), full_rows)
    pieces = []
    for start in range(0, row_ids.shape[0], chunk_rows):
        pieces.append(table.index_select(0, row_ids[start : start + chunk_rows].reshape(-1).long()).reshape(-1, row_ids.shape[1], table.shape[1]))
    return torch.cat(pieces, dim=0) if pieces else table.new_empty((0, row_ids.shape[1], table.shape[1]))


class LookupOutsideResidentWindow(IndexError):
    """A valid full-table row was not present in the bounded audit window."""

    def __init__(self, row: int, resident_rows: int, full_rows: int):
        super().__init__(f"row {row} needs host offload (resident={resident_rows}, full={full_rows})")
        self.row, self.resident_rows, self.full_rows = row, resident_rows, full_rows


__all__ = [
    "DEAD_ID",
    "HashLayout",
    "LookupOutsideResidentWindow",
    "build_compressed_token_map",
    "make_layout",
    "hash_ids",
    "update_slot_history",
    "bounded_host_gather",
]


# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Stream physical-layer checkpoint operands and sparse Engram rows for Decode."""

from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import torch
from safetensors import safe_open

from .config import FLASH
from .engram_host import load_rotation
from .utils import unpack_blocked_weight
from .weights import prepare_product_operands


class DecodeNetworkWeights:
    """Load Decode weights once and retain their device-side representation."""

    _RENAME = dict(attn_norm="attn_norm_weight", q_norm="q_norm_weight", kv_norm="kv_norm_weight",
                   hc_function="attn_hc_function", hc_scale="attn_hc_scale", hc_base="attn_hc_base",
                   index_weights="index_weights_proj", index_wqb="index_wq_b",
                   index_wqb_scale="index_wq_b_scale")
    _STATIC_NAMES = frozenset({
        "attn_hc_function", "attn_hc_scale", "attn_hc_base", "attn_norm_weight",
        "wq_a", "wq_a_scale", "q_norm_weight", "wq_b", "wq_b_scale", "wkv", "wkv_scale",
        "kv_norm_weight", "wo_a", "wo_b", "sink", "ffn_hc_function", "ffn_hc_scale",
        "ffn_hc_base", "ffn_norm_weight", "gate_weight", "text_bias", "vision_bias",
        "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale", "routed_w2",
        "routed_w2_scale", "shared_gate_up", "shared_gate_up_scale", "shared_down",
        "shared_down_scale", "index_wq_b", "index_wq_b_scale", "index_weights_proj",
        "compressor_wkv", "compressor_wgate", "compressor_norm", "index_wk", "index_norm",
    })

    def __init__(self, checkpoint, world):
        self.reader = DecodeSafetensorsCheckpoint(checkpoint)
        self.world = world
        self._stack = ExitStack()
        self._slices = {}
        self.token_map = None
        self._resident_layers = {}
        self._engram = {}
        self._boundary = None
        self.initialized = False
        self.initialization_timings = {}
        self.device_upload_calls = 0
        self._routed_prepare_seconds = 0.0
        text = self.reader.config["text_config"]
        for name in ("hidden_size", "vocab_size", "num_hidden_layers", "n_routed_experts",
                     "num_experts_per_tok", "num_attention_heads", "head_dim", "hc_mult"):
            if text[name] != getattr(FLASH, name):
                raise ValueError("Checkpoint model geometry differs: " + name)
        if tuple(text["compress_ratios"][:40]) != FLASH.compress_ratios:
            raise ValueError("Checkpoint backbone compression schedule differs")
        self.image_ids = (self.reader.config["image_token_id"],
                          self.reader.config.get("image_pad_token_id", self.reader.config["image_token_id"] + 1))

    def close(self):
        self._stack.close()
        self._slices.clear()
        self._resident_layers.clear()
        self._engram.clear()
        self._boundary = None
        self.initialized = False

    @staticmethod
    def _stacked(runtime, value, world):
        """Upload one local copy per rank and bind the full distributed shape."""
        from pypto.runtime import StackedDeviceTensor

        value = value.contiguous()
        shards = []
        try:
            for rank in range(world):
                shards.append(runtime.alloc_tensor(value.shape, value.dtype, init=value, worker_id=rank))
        except BaseException:
            for rank, shard in enumerate(shards):
                try:
                    runtime.free_tensor(shard, worker_id=rank)
                except BaseException:
                    pass
            raise
        return StackedDeviceTensor(shards, (world, *value.shape), tuple(range(world)))

    def _upload_layer(self, runtime, values):
        self.device_upload_calls += self.world * len(values)
        return {name: self._stacked(runtime, value, self.world) for name, value in values.items()}

    def _upload_routed(self, runtime, layer):
        """Stream owned experts without materializing all EP shards at once."""
        from pypto.runtime import StackedDeviceTensor

        result = {}
        local = 384 // self.world
        for rank in range(self.world):
            expert_index = 0
            experts = iter_routed_experts(self.reader, layer, rank, ep_size=self.world)
            while True:
                prepare_begin = perf_counter()
                try:
                    _, packed = next(experts)
                except StopIteration:
                    break
                self._routed_prepare_seconds += perf_counter() - prepare_begin
                for name, value in packed.items():
                    value = value.contiguous()
                    target = "routed_" + name
                    shards = result.setdefault(target, [None] * self.world)
                    if shards[rank] is None:
                        shards[rank] = runtime.alloc_tensor(
                            (local, *value.shape), value.dtype, init=None, worker_id=rank
                        )
                    runtime.copy_to(
                        shards[rank].data_ptr, value.data_ptr(), value.numel() * value.element_size(),
                        dst_offset=expert_index * value.numel() * value.element_size(), worker_id=rank,
                    )
                    self.device_upload_calls += 1
                expert_index += 1
            if expert_index != local:
                raise ValueError(f"L{layer} rank {rank}: expected {local} owned experts, got {expert_index}")
        return {
            name: StackedDeviceTensor(shards, (self.world, *shards[0].shape), tuple(range(self.world)))
            for name, shards in result.items()
        }

    @staticmethod
    def _decode_operands(raw, layer):
        """Adapt target Prefill-packed INT8 projections to Decode's canonical ABI."""
        prepared = prepare_product_operands(raw, layer)
        # Keep the Decode ABI explicit for full/reindex layers.  The shared
        # product preparer intentionally follows the Prefill role table and
        # may omit readonly indexer/compressor tensors on a reuse role, while
        # the Decode entry still declares the source-layer parameters.
        prefix = f"layers.{layer}.attn."
        fallback = {
            "index_wk": prefix + "indexer.wk.weight",
            "index_norm": prefix + "indexer.k_norm.weight",
            "index_weights": prefix + "indexer.weights_proj.weight",
            "compressor_wkv": prefix + "compressor.wkv.weight",
            "compressor_wgate": prefix + "compressor.wgate.weight",
            "compressor_norm": prefix + "compressor.norm.weight",
        }
        for name, source in fallback.items():
            if name not in prepared and source in raw:
                value = raw[source]
                if name in ("index_norm", "compressor_norm"):
                    prepared[name] = value.contiguous()
                else:
                    prepared[name] = value.T.contiguous()
        for name, shape in dict(
            wq_a=(5120, 1280),
            wq_b=(1280, 32768),
            wkv=(5120, 512),
            index_wqb=(1280, 4096),
        ).items():
            if name in prepared:
                prepared[name] = unpack_blocked_weight(prepared[name], *shape)
        return prepared

    def preload(self, runtime):
        """Prepare and upload the complete Decode weight bank exactly once."""
        if self.initialized:
            return dict(self.initialization_timings)
        begin = perf_counter()
        read_before = self.reader.read_calls
        read_seconds_before = self.reader.read_seconds
        host_prepare = 0.0
        upload_seconds = 0.0
        try:
            for layer in range(FLASH.num_hidden_layers):
                prefix = f"layers.{layer}."
                names = [name for name in self.reader.layer_tensor_names(layer)
                         if ".ffn.experts." not in name and ".engram.embed." not in name]
                with self.reader.cached_files():
                    raw = self.reader.require(names)
                    prepare_begin = perf_counter()
                    prepared = self._decode_operands(raw, layer)
                    result = {self._RENAME.get(name, name): value for name, value in prepared.items()
                              if name != "wo_a_grouped_prefill"}
                    for name, suffix in dict(ffn_hc_function="hc_ffn_fn", ffn_hc_scale="hc_ffn_scale",
                                             ffn_hc_base="hc_ffn_base", ffn_norm_weight="ffn_norm.weight").items():
                        result[name] = raw[prefix + suffix].contiguous()
                    result.update(load_router(self.reader, layer))
                    result.update(load_shared_expert(self.reader, layer))
                    host_prepare += perf_counter() - prepare_begin
                    upload_begin = perf_counter()
                    resident = self._upload_layer(runtime, result)
                    resident.update(self._upload_routed(runtime, layer))
                    routed_prepare = self._routed_prepare_seconds
                    upload_seconds += perf_counter() - upload_begin - routed_prepare
                    host_prepare += routed_prepare
                    self._routed_prepare_seconds = 0.0
                    if layer in FLASH.engram_layer_ids:
                        engram_prepare_begin = perf_counter()
                        q = raw[prefix + "engram.q_weight"].float()
                        k = raw[prefix + "engram.k_weight"].float()
                        self._engram[layer] = dict(
                            wkv=raw[prefix + "engram.wkv.weight"].T.contiguous(),
                            channel_weight=(q * k).reshape(4, 5120).contiguous(),
                            rotation=load_rotation(self.reader.root, FLASH.hidden_size)[0],
                        )
                        host_prepare += perf_counter() - engram_prepare_begin
                    self._resident_layers[layer] = resident
                del raw, prepared, result

            boundary_prepare_begin = perf_counter()
            boundary_read_seconds = self.reader.read_seconds
            self._boundary = (self.reader.tensor("embed.weight"), self.reader.tensor("norm.weight"),
                              self.reader.tensor("head.weight").T.contiguous())
            host_prepare += (
                perf_counter() - boundary_prepare_begin - (self.reader.read_seconds - boundary_read_seconds)
            )
            self.initialized = True
        except BaseException:
            self._resident_layers.clear()
            self._engram.clear()
            self._boundary = None
            raise
        self.initialization_timings = dict(
            checkpoint_read_seconds=self.reader.read_seconds - read_seconds_before,
            host_weight_preparation_seconds=host_prepare,
            device_weight_upload_seconds=upload_seconds,
            resident_weight_initialization_total_seconds=perf_counter() - begin,
            checkpoint_read_calls=self.reader.read_calls - read_before,
            device_weight_upload_calls=self.device_upload_calls,
        )
        return dict(self.initialization_timings)

    def layer(self, layer):
        """Return one resident layer, retaining lazy loading for compatibility."""
        if self.initialized:
            try:
                return self._resident_layers[layer]
            except KeyError as exc:
                raise ValueError(f"resident layer {layer} is not initialized") from exc
        with self.reader.cached_files():
            prefix = f"layers.{layer}."
            names = [name for name in self.reader.layer_tensor_names(layer)
                     if ".ffn.experts." not in name and ".engram.embed." not in name]
            raw = self.reader.require(names)
            prepared = self._decode_operands(raw, layer)
            rename = dict(attn_norm="attn_norm_weight", q_norm="q_norm_weight", kv_norm="kv_norm_weight",
                          hc_function="attn_hc_function", hc_scale="attn_hc_scale", hc_base="attn_hc_base",
                          index_weights="index_weights_proj", index_wqb="index_wq_b",
                          index_wqb_scale="index_wq_b_scale")
            result = {rename.get(name, name): value for name, value in prepared.items()
                      if name != "wo_a_grouped_prefill"}
            for name, suffix in dict(ffn_hc_function="hc_ffn_fn", ffn_hc_scale="hc_ffn_scale",
                                     ffn_hc_base="hc_ffn_base", ffn_norm_weight="ffn_norm.weight").items():
                result[name] = raw[prefix + suffix].contiguous()
            result.update(load_router(self.reader, layer))
            result.update(load_shared_expert(self.reader, layer))
            result = {name: value.unsqueeze(0).expand(self.world, *value.shape).contiguous()
                      for name, value in result.items()}
            for rank in range(self.world):
                for expert, packed in iter_routed_experts(self.reader, layer, rank, ep_size=self.world):
                    local = expert % (384 // self.world)
                    for name, value in packed.items():
                        target = "routed_" + name
                        if target not in result:
                            result[target] = torch.empty(
                                (self.world, 384 // self.world, *value.shape), dtype=value.dtype
                            )
                        result[target][rank, local].copy_(value)
            if layer in FLASH.engram_layer_ids:
                # The repeated block32 rotation is shared with the Prefill checkpoint ABI.
                result["engram_rotation"] = load_rotation(self.reader.root, FLASH.hidden_size)[0]
                result["engram_wkv"] = raw[prefix + "engram.wkv.weight"].T.contiguous()
                q = raw[prefix + "engram.q_weight"].float()
                k = raw[prefix + "engram.k_weight"].float()
                result["engram_channel_weight"] = (q * k).reshape(4, 5120).contiguous()
            return result

    def embedding(self):
        if self._boundary is not None:
            return self._boundary[0]
        return self.reader.tensor("embed.weight")

    def head(self):
        if self._boundary is not None:
            return self._boundary[1], self._boundary[2]
        return self.reader.tensor("norm.weight"), self.reader.tensor("head.weight").T.contiguous()

    def engram(self, layer):
        if layer not in self._engram:
            raise ValueError(f"Engram weights for layer {layer} are not resident")
        return self._engram[layer]

    @property
    def static_names(self):
        return self._STATIC_NAMES

    @property
    def reader_stats(self):
        return dict(read_calls=self.reader.read_calls, read_seconds=self.reader.read_seconds,
                    read_counts=dict(self.reader.read_counts))

    def release_host_staging(self):
        """Drop checkpoint-sized host arrays after IO workers upload boundaries."""
        self._boundary = None
        self._engram.clear()

    def _table_slice(self, layer, kind):
        key = layer, kind
        if key not in self._slices:
            name = f"layers.{layer}.engram.embed.{kind}"
            path = (self.reader.shard_for(name) if name in self.reader.weight_map else
                    Path(self.reader.root) / f"engram_embed_{kind}_l{layer}.safetensors")
            handle = self._stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            keys = list(handle.keys())
            if name not in keys:
                if len(keys) != 1:
                    raise ValueError("Engram table file has ambiguous tensor names")
                name = keys[0]
            self._slices[key] = handle.get_slice(name)
        return self._slices[key]

    def engram_rows(self, tokens, histories):
        """Hash each request with its last three tokens and gather only 24 rows per slot."""
        if self.token_map is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.reader.root, local_files_only=True)
            mapping, compressed = build_compressed_token_map(tokenizer)
            if compressed != FLASH.engram_compressed_vocab_size:
                raise ValueError("Checkpoint tokenizer compressed vocabulary differs")
            self.token_map = torch.tensor(mapping, dtype=torch.int64)
            self.layout = make_layout()
        rows = {layer: torch.zeros((self.world, 16, 6144), dtype=torch.bfloat16) for layer in (1, 14)}
        masks = torch.zeros((self.world, 16), dtype=torch.int32)
        for rank, (token, history) in enumerate(zip(tokens, histories, strict=True)):
            previous = torch.tensor([(list(reversed(history[-3:])) + [-1] * 3)[:3]], dtype=torch.int64)
            dead = token in self.image_ids
            hashes = hash_ids(torch.tensor([token]), self.token_map, torch.tensor([len(history)]),
                              torch.tensor([0, 1], dtype=torch.int32), layout=self.layout,
                              dead_mask=torch.tensor([dead]), lookback_token_ids=previous,
                              lookback_dead_mask=(previous == self.image_ids[0]) | (previous == self.image_ids[1]))
            masks[rank, 0] = int(not dead)
            for slot, layer in enumerate((1, 14)):
                codes = self._table_slice(layer, "weight")
                scales = self._table_slice(layer, "scale")
                for col, row in enumerate(hashes[0, slot].tolist()):
                    code = codes[row:row + 1].float().reshape(8, 32)
                    scale = scales[row:row + 1].float().reshape(8, 1)
                    rows[layer][rank, 0, col * 256:(col + 1) * 256] = (code * scale).flatten().to(torch.bfloat16)
        return rows, masks

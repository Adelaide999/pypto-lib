# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Checkpoint readers, phase weight layouts and resident operand ownership."""

import argparse
import hashlib
import json
import re
import struct
from collections import Counter
from collections.abc import Iterator as decode_Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Mapping
import torch
from .cache import resolve_layer_role
from .gemm import prepare_ecotech_dynamic_weight
from .runtime_config import ExpertParallelConfig
from .runtime_utils import tensor_sha
from .utils import pack_nz


_DTYPES = {
    "I8": torch.int8,
    "U8": torch.uint8,
    "I32": torch.int32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
}


def _shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"metadata shape must be a list, got {type(value).__name__}")
    result = tuple(int(dim) for dim in value)
    if any(dim < 0 for dim in result):
        raise ValueError(f"metadata shape contains a negative dimension: {result}")
    return result


class RawCheckpoint:
    """Read complete tensors from a raw checkpoint without copying by default."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {self.root}")
        self._metadata: dict[str, dict[str, Any]] = {}

    def _base_name(self, name: str) -> str:
        clean = name.removesuffix(".bin").removesuffix(".bin.json")
        if not clean:
            raise ValueError("checkpoint tensor name is empty")
        return clean

    def metadata(self, name: str) -> dict[str, Any]:
        base = self._base_name(name)
        if base not in self._metadata:
            metadata_path = self.root / f"{base}.bin.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"checkpoint metadata is missing: {metadata_path}")
            payload = json.loads(metadata_path.read_text())
            if not isinstance(payload, dict):
                raise TypeError(f"checkpoint metadata must be an object: {metadata_path}")
            self._metadata[base] = payload
        return self._metadata[base]

    def tensor(self, name: str, *, clone: bool = False) -> torch.Tensor:
        base = self._base_name(name)
        metadata = self.metadata(base)
        if metadata.get("outcome") not in (None, "complete"):
            raise ValueError(f"checkpoint tensor is not complete: {base} ({metadata.get('outcome')})")
        dtype_name = str(metadata.get("dtype", "")).upper()
        if dtype_name not in _DTYPES:
            raise TypeError(f"unsupported checkpoint dtype for {base}: {dtype_name!r}")
        dtype = _DTYPES[dtype_name]
        shape = _shape(metadata.get("shape"))
        numel = 1
        for dim in shape:
            numel *= dim
        path = self.root / f"{base}.bin"
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint tensor is missing: {path}")
        expected_bytes = numel * torch.empty((), dtype=dtype).element_size()
        actual_bytes = path.stat().st_size
        if actual_bytes < expected_bytes:
            raise ValueError(f"{base} is truncated: {actual_bytes} < {expected_bytes} bytes")
        declared_bytes = metadata.get("bytes")
        if declared_bytes is not None and int(declared_bytes) != expected_bytes:
            raise ValueError(
                f"{base} metadata bytes do not match shape/dtype: {declared_bytes} != {expected_bytes}"
            )
        value = torch.from_file(str(path), shared=False, size=numel, dtype=dtype).reshape(shape)
        return value.clone() if clone else value

    def require(self, names: list[str]) -> dict[str, torch.Tensor]:
        return {name: self.tensor(name) for name in names}


def flatten_scale(value: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize a checkpoint ``[N, 1]`` channel scale to ``[N]``."""

    if value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 1:
        raise ValueError(f"{name} must be a vector or [N, 1], got {tuple(value.shape)}")
    return value


def load_attention_weights(checkpoint: RawCheckpoint, layer: int) -> dict[str, torch.Tensor]:
    """Load A2/A3 attention tensors for any migrated decoder layer.

    Entries ending in ``_raw`` retain checkpoint output-major layout.  Entries
    ending in ``_canonical`` are transposed to the ``[K, N]`` layout consumed
    by the standalone PyPTO W8A8 helpers.
    """

    if layer < 0:
        raise ValueError("layer must be non-negative")
    prefix = f"layers.{layer}.attn"
    names = [
        f"{prefix}.wq_a.weight",
        f"{prefix}.wq_a.weight_scale",
        f"{prefix}.wq_a.weight_offset",
        f"{prefix}.wq_b.weight",
        f"{prefix}.wq_b.weight_scale",
        f"{prefix}.wq_b.weight_offset",
        f"{prefix}.wkv.weight",
        f"{prefix}.wkv.weight_scale",
        f"{prefix}.wkv.weight_offset",
        f"{prefix}.q_norm.weight",
        f"{prefix}.kv_norm.weight",
        f"{prefix}.wo_a.weight",
        f"{prefix}.wo_b.weight",
        f"{prefix}.attn_sink",
    ]
    optional_names = [
        f"{prefix}.compressor.wkv.weight",
        f"{prefix}.compressor.wgate.weight",
        f"{prefix}.compressor.norm.weight",
        f"{prefix}.indexer.wk.weight",
        f"{prefix}.indexer.k_norm.weight",
        f"{prefix}.indexer.wq_b.weight",
        f"{prefix}.indexer.wq_b.weight_scale",
        f"{prefix}.indexer.wq_b.weight_offset",
        f"{prefix}.indexer.weights_proj.weight",
    ]
    names.extend(name for name in optional_names if (checkpoint.root / f"{name}.bin.json").is_file())
    raw = checkpoint.require(names)
    result: dict[str, torch.Tensor] = dict(raw)
    for name in ("wq_a", "wq_b", "wkv"):
        result[f"{name}_canonical"] = raw[f"{prefix}.{name}.weight"].transpose(0, 1).contiguous()
        result[f"{name}_scale"] = flatten_scale(
            raw[f"{prefix}.{name}.weight_scale"], f"{prefix}.{name}.weight_scale"
        )
    result["q_norm"] = raw[f"{prefix}.q_norm.weight"]
    result["kv_norm"] = raw[f"{prefix}.kv_norm.weight"]
    result["wo_a_raw"] = raw[f"{prefix}.wo_a.weight"]
    result["wo_b_raw"] = raw[f"{prefix}.wo_b.weight"]
    result["attn_sink"] = raw[f"{prefix}.attn_sink"]
    # The vLLM A2/A3 W8A8 path is symmetric.  Keep this check next to the
    # conversion so a per-channel asymmetric checkpoint cannot silently be
    # interpreted as a different operator ABI.
    for name in ("wq_a", "wq_b", "wkv"):
        offset = flatten_scale(raw[f"{prefix}.{name}.weight_offset"], f"{prefix}.{name}.weight_offset")
        if not torch.equal(offset, torch.zeros_like(offset)):
            raise ValueError(
                f"{prefix}.{name} has non-zero weight offsets; A2/A3 W8A8 requires symmetric weights"
            )
        result[f"{name}_offset"] = offset

    # Wo-A is stored output-major as [groups * rank, group_input], while the
    # A2/A3 grouped kernel consumes [groups, group_input, rank].  Wo-B is
    # stored [hidden, groups * rank] and is transposed to [K, N].
    wo_a = raw[f"{prefix}.wo_a.weight"]
    wo_b = raw[f"{prefix}.wo_b.weight"]
    if wo_a.ndim != 2 or wo_b.ndim != 2:
        raise ValueError(f"{prefix} output projections must be matrices")
    if wo_a.shape[0] % 8:
        raise ValueError(f"{prefix}.wo_a first dimension must be divisible by 8")
    groups = 8
    rank = wo_a.shape[0] // groups
    result["wo_a"] = wo_a.reshape(groups, rank, wo_a.shape[1]).transpose(1, 2).contiguous()
    result["wo_b"] = wo_b.transpose(0, 1).contiguous()

    # The source layers carry the ratio-2 compressor/indexer tensors.  The
    # ratio-1 source (L20) uses the same names but only has compressor.wkv.
    compressor_wkv_name = f"{prefix}.compressor.wkv.weight"
    if compressor_wkv_name in raw:
        result["compressor_wkv"] = raw[compressor_wkv_name]
        result["compressor_wkv_canonical"] = raw[compressor_wkv_name].transpose(0, 1).contiguous()
        result["compressor_norm"] = raw[f"{prefix}.compressor.norm.weight"]
    index_prefix = f"{prefix}.indexer"
    if (checkpoint.root / f"{index_prefix}.wk.weight.bin.json").is_file():
        result["index_wk"] = raw[f"{index_prefix}.wk.weight"].transpose(0, 1).contiguous()
        result["index_k_norm"] = raw[f"{index_prefix}.k_norm.weight"]
    if (checkpoint.root / f"{index_prefix}.weights_proj.weight.bin.json").is_file():
        result["index_weights_proj"] = raw[f"{index_prefix}.weights_proj.weight"].transpose(0, 1).contiguous()
    if (checkpoint.root / f"{index_prefix}.wq_b.weight.bin.json").is_file():
        result["index_wq_b"] = raw[f"{index_prefix}.wq_b.weight"].transpose(0, 1).contiguous()
        result["index_wq_b_scale"] = flatten_scale(
            raw[f"{index_prefix}.wq_b.weight_scale"], f"{index_prefix}.wq_b.weight_scale"
        )
        index_offset = flatten_scale(
            raw[f"{index_prefix}.wq_b.weight_offset"], f"{index_prefix}.wq_b.weight_offset"
        )
        if not torch.equal(index_offset, torch.zeros_like(index_offset)):
            raise ValueError(f"{index_prefix}.wq_b has non-zero weight offsets")

    if f"{prefix}.compressor.wgate.weight" in raw:
        result["compressor_wgate"] = raw[f"{prefix}.compressor.wgate.weight"]
        result["compressor_wgate_canonical"] = (
            raw[f"{prefix}.compressor.wgate.weight"].transpose(0, 1).contiguous()
        )
    if "index_wk" in result:
        result["index_wk_raw"] = raw[f"{prefix}.indexer.wk.weight"]
        result["index_wk_canonical"] = result["index_wk"]
        result["index_wq_b_canonical"] = result["index_wq_b"]
        result["index_weights_proj_raw"] = raw[f"{prefix}.indexer.weights_proj.weight"]
    # Keep the original raw tensors for callers that need to inspect the
    # checkpoint layout, while exposing canonical tensors for A2/A3 kernels.
    result["wo_a_raw"] = wo_a
    result["wo_b_raw"] = wo_b
    return result


load_decoder_layer_weights = load_attention_weights


class CheckpointOperands:
    """Read original sharded safetensors and record each consumed operand."""

    def __init__(self, root, *, diagnostics=True):
        import json

        self.root = Path(root).resolve()
        index = self.root / "quant_model_weights.safetensors.index.json"
        self.index = json.loads(index.read_text())["weight_map"]
        self.consumed = {}
        self.diagnostics = diagnostics
        self.read_seconds, self.read_calls = 0.0, 0

    def load(self, name):
        """Load complete logical checkpoint bytes before any layout or dtype conversion."""
        from safetensors import safe_open

        begin = perf_counter()
        path = self.root / self.index[name]
        with safe_open(path, framework="pt", device="cpu") as source:
            value = source.get_tensor(name)
        self.consumed[name] = {
            "file": path.name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "selected_rows": None,
        }
        if self.diagnostics:
            self.consumed[name]["sha256"] = tensor_sha(value)
        self.read_seconds += perf_counter() - begin
        self.read_calls += 1
        return value

    def expert(self, prefix):
        """Return canonical gate/up and down operands with the true channel scales."""
        operands = {}
        for stem in ("w1", "w2", "w3"):
            for suffix in ("weight", "weight_scale", "weight_offset"):
                operands[stem + "." + suffix] = self.load(prefix + "." + stem + "." + suffix)
            if any(value != 0 for value in operands[stem + ".weight_offset"].reshape(-1).tolist()):
                raise ValueError("W8A8 expert operands must have symmetric zero offsets")
        w1, w3 = operands["w1.weight"], operands["w3.weight"]
        gate_up = torch.cat((w1, w3), dim=0).transpose(0, 1).contiguous()
        gate_scale = torch.cat((operands["w1.weight_scale"], operands["w3.weight_scale"])).flatten().float()
        down = operands["w2.weight"].transpose(0, 1).contiguous()
        down_scale = operands["w2.weight_scale"].flatten().float()
        if gate_up.dtype != torch.int8 or down.dtype != torch.int8:
            raise TypeError("Expert checkpoint weights must remain INT8")
        return gate_up, gate_scale, down, down_scale


def load_tensor_bundle(
    path: str | Path, required: Mapping[str, tuple[torch.dtype, tuple[int, ...]]] | None = None
):
    """Load a ``torch.save`` bundle and validate the A2/A3 tensor contract.

    The prepared layer-0 and layer-2 assets under the checkpoint workspace are
    already converted to contiguous ``[K, N]`` INT8 weights with FP32 channel
    scales.  This loader deliberately refuses silent dtype or shape changes.
    """

    bundle_path = Path(path)
    payload = torch.load(bundle_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"expected a tensor dictionary in {bundle_path}, got {type(payload).__name__}")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in payload.items()):
        raise TypeError(f"all entries in {bundle_path} must be named torch.Tensor values")
    if required is not None:
        missing = sorted(set(required) - set(payload))
        if missing:
            raise KeyError(f"missing tensors in {bundle_path}: {missing}")
        for name, (dtype, shape) in required.items():
            tensor = payload[name]
            if tensor.dtype != dtype or tuple(tensor.shape) != tuple(shape):
                raise ValueError(
                    f"{name} contract mismatch: got dtype={tensor.dtype}, shape={tuple(tensor.shape)}; "
                    f"expected dtype={dtype}, shape={shape}"
                )
    return payload


def load_layer0_qkv(path: str | Path):
    """Load the prepared layer-0 QKV bundle used by the first decoder step."""

    return load_tensor_bundle(
        path,
        {
            "wq_a": (torch.int8, (5120, 1280)),
            "wq_a_scale": (torch.float32, (1280,)),
            "wq_b": (torch.int8, (1280, 32768)),
            "wq_b_scale": (torch.float32, (32768,)),
            "wkv": (torch.int8, (5120, 512)),
            "wkv_scale": (torch.float32, (512,)),
            "q_norm_weight": (torch.bfloat16, (1280,)),
            "kv_norm_weight": (torch.bfloat16, (512,)),
        },
    )


def load_layer0_shared_expert(path: str | Path):
    """Load the prepared layer-0 shared W8A8 expert bundle."""

    return load_tensor_bundle(
        path,
        {
            "gate_up_weight": (torch.int8, (5120, 4608)),
            "gate_up_scale": (torch.float32, (4608,)),
            "down_weight": (torch.int8, (2304, 5120)),
            "down_scale": (torch.float32, (5120,)),
        },
    )


def load_layer0_output(path: str | Path):
    """Load the prepared layer-0 grouped output projection bundle."""

    return load_tensor_bundle(
        path,
        {
            "wo_a": (torch.bfloat16, (8, 4096, 1024)),
            "wo_b": (torch.int8, (8192, 5120)),
            "wo_b_scale": (torch.float32, (5120,)),
        },
    )


def load_layer0_router(path: str | Path):
    """Load router tensors from the converted layer-0 MoE safetensors."""

    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - only hit outside pto
        raise RuntimeError("router safetensors require the safetensors package") from exc
    required = {
        "gate.weight": (torch.bfloat16, (384, 5120)),
        "gate.bias": (torch.float32, (384,)),
    }
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        result = {}
        for name, (dtype, shape) in required.items():
            if name not in handle.keys():
                raise KeyError(f"missing tensors in {path}: {name}")
            tensor = handle.get_tensor(name)
            if tensor.dtype != dtype or tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{name} contract mismatch: got dtype={tensor.dtype}, shape={tuple(tensor.shape)}; "
                    f"expected dtype={dtype}, shape={shape}"
                )
            result[name] = tensor
        if "gate.bias_vl" in handle.keys():
            vision_bias = handle.get_tensor("gate.bias_vl")
            if vision_bias.dtype != torch.float32 or tuple(vision_bias.shape) != (384,):
                raise ValueError(
                    f"gate.bias_vl contract mismatch: got dtype={vision_bias.dtype}, "
                    f"shape={tuple(vision_bias.shape)}; expected dtype={torch.float32}, shape=(384,)"
                )
            result["gate.bias_vl"] = vision_bias
    return result


@contextmanager
def open_layer0_moe(path: str | Path):
    """Open the converted layer-0 MoE and expose a lazy expert provider.

    The routed checkpoint is about 13 GiB.  Keeping the ``safe_open`` handle
    alive lets a validator load only the six tensors for experts selected by
    the router while retaining the exact INT8/FP32 tensors used by A2/A3.
    """

    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - only hit outside pto
        raise RuntimeError("MoE safetensors require the safetensors package") from exc

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        required_router = {
            "gate.weight": (torch.bfloat16, (384, 5120)),
            "gate.bias": (torch.float32, (384,)),
        }
        router = {}
        for name, (dtype, shape) in required_router.items():
            if name not in handle.keys():
                raise KeyError(f"missing tensors in {path}: {name}")
            tensor = handle.get_tensor(name)
            if tensor.dtype != dtype or tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{name} contract mismatch: got dtype={tensor.dtype}, shape={tuple(tensor.shape)}; "
                    f"expected dtype={dtype}, shape={shape}"
                )
            router[name] = tensor
        if "gate.bias_vl" in handle.keys():
            vision_bias = handle.get_tensor("gate.bias_vl")
            if vision_bias.dtype != torch.float32 or tuple(vision_bias.shape) != (384,):
                raise ValueError("gate.bias_vl must be FP32[384]")
            router["gate.bias_vl"] = vision_bias

        def expert_provider(expert_id: int) -> dict[str, torch.Tensor]:
            expert = int(expert_id)
            if not 0 <= expert < 384:
                raise IndexError(f"expert id must be in [0, 384), got {expert}")
            names = {
                "w1": f"experts.{expert}.w1.weight",
                "w1_scale": f"experts.{expert}.w1.scale",
                "w2": f"experts.{expert}.w2.weight",
                "w2_scale": f"experts.{expert}.w2.scale",
                "w3": f"experts.{expert}.w3.weight",
                "w3_scale": f"experts.{expert}.w3.scale",
            }
            result = {}
            expected = {
                "w1": (torch.int8, (5120, 2304)),
                "w1_scale": (torch.float32, (2304,)),
                "w2": (torch.int8, (2304, 5120)),
                "w2_scale": (torch.float32, (5120,)),
                "w3": (torch.int8, (5120, 2304)),
                "w3_scale": (torch.float32, (2304,)),
            }
            for key, name in names.items():
                if name not in handle.keys():
                    raise KeyError(f"missing tensors in {path}: {name}")
                tensor = handle.get_tensor(name)
                dtype, shape = expected[key]
                if tensor.dtype != dtype or tuple(tensor.shape) != shape:
                    raise ValueError(
                        f"{name} contract mismatch: got dtype={tensor.dtype}, shape={tuple(tensor.shape)}; "
                        f"expected dtype={dtype}, shape={shape}"
                    )
                result[key] = tensor
            return result

        yield router, expert_provider


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
    def open_tensor(self, name: str) -> Iterator[Any]:
        """Yield one safetensors handle and the requested tensor name."""

        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Aurora Decode weights require the safetensors package") from exc
        path = self.shard_for(name)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if name not in handle.keys():
                raise KeyError(f"tensor {name!r} is not present in indexed shard {path}")
            yield handle, name

    def tensor(self, name: str) -> Any:
        """Load one host tensor; no tensor is retained by this reader."""

        with self.open_tensor(name) as (handle, tensor_name):
            return handle.get_tensor(tensor_name)

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


HIDDEN = 5120
INTERMEDIATE = 2304
INTERMEDIATE_PADDED = 2560
EXPERTS = 384
DECODE_EP_SIZE = 16
LOCAL_EXPERTS = EXPERTS // DECODE_EP_SIZE
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
    """Load and pack one routed expert in the Decode FRACTAL_NZ layout.

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
) -> decode_Iterator[tuple[int, dict[str, torch.Tensor]]]:
    """Yield one EP16 owner's experts without retaining a second full shard."""

    if not 0 <= rank < DECODE_EP_SIZE:
        raise ValueError(f"Decode rank must be in [0, {DECODE_EP_SIZE}), got {rank}")
    first = rank * LOCAL_EXPERTS
    for expert in range(first, first + LOCAL_EXPERTS):
        yield expert, pack_routed_expert(checkpoint, layer, expert)


def load_routed_shard(
    checkpoint: DecodeSafetensorsCheckpoint,
    layer: int,
    rank: int,
) -> dict[str, torch.Tensor]:
    """Materialize one complete 24-expert EP16 shard in external NZ layout.

    The ``pl.NZ`` annotation is logical in the Python signature, but the
    distributed host ABI passes the compiler's physical five-dimensional
    view.  The first dimension is the local expert, followed by
    ``[C / C0, R / 16, 16, C0]``.  Keeping that shape here prevents the
    validation runner from silently treating packed bytes as row-major data.
    """

    entries = [packed for _, packed in iter_routed_experts(checkpoint, layer, rank)]

    def stack_nz(name: str, rows: int, cols: int) -> torch.Tensor:
        value = torch.stack([entry[name] for entry in entries])
        return value.reshape(LOCAL_EXPERTS, cols // 32, rows // 16, 16, 32).contiguous()

    return {
        "routed_w1": stack_nz("w1", INTERMEDIATE, HIDDEN),
        "routed_w1_scale": torch.stack([entry["w1_scale"] for entry in entries]),
        "routed_w3": stack_nz("w3", INTERMEDIATE, HIDDEN),
        "routed_w3_scale": torch.stack([entry["w3_scale"] for entry in entries]),
        "routed_w2": stack_nz("w2", HIDDEN, INTERMEDIATE_PADDED),
        "routed_w2_scale": torch.stack([entry["w2_scale"] for entry in entries]),
    }


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


def load_decode_rank_weights(
    checkpoint: DecodeSafetensorsCheckpoint,
    layer: int,
    rank: int,
) -> dict[str, torch.Tensor]:
    """Load an EP16 MoE shard; attention and cache bindings are separate."""

    result: dict[str, torch.Tensor] = {}
    result.update(load_router(checkpoint, layer))
    result.update(load_routed_shard(checkpoint, layer, rank))
    result.update(load_shared_expert(checkpoint, layer))
    return result


def abi_summary() -> dict[str, Any]:
    """Return the fixed host-to-PyPTO layout without reading checkpoint data."""

    return {
        "source": "Aurora safetensors W8A8_DYNAMIC",
        "expert_parallel": DECODE_EP_SIZE,
        "local_experts": LOCAL_EXPERTS,
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


TARGET_LAYERS = (0, 1, 2, 3, 14, 15, 19, 20, 21, 24, 25)
METADATA_FILES = (
    "config.json",
    "quant_model_weights.safetensors.index.json",
    "quant_model_description.json",
    "aurora_export_manifest.json",
)
SHARD_PATTERN = re.compile(r"quant_model_weights-(\d{5})-of-(\d{5})\.safetensors$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safetensors_header(path: Path) -> tuple[int, str, dict[str, Any]]:
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"truncated safetensors length prefix: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        header = stream.read(header_length)
        if len(header) != header_length:
            raise ValueError(f"truncated safetensors header: {path}")
    try:
        payload = json.loads(header)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid safetensors header: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"safetensors header must be an object: {path}")
    return header_length, hashlib.sha256(raw_length + header).hexdigest(), payload


def _config_summary(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config", {})
    keys = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "q_lora_rank",
        "o_lora_rank",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "compress_ratios",
        "kv_source_layer_ids",
        "index_source_layer_ids",
        "index_n_heads",
        "index_head_dim",
        "index_topk",
        "candidate_source_layer_id",
        "engram_layer_ids",
        "engram_num_embeddings",
        "engram_n_heads",
        "engram_head_dim",
    )
    return {key: text.get(key) for key in keys}


def inspect_checkpoint(root: str | Path) -> dict[str, Any]:
    """Return a metadata and header manifest for one Aurora export."""

    checkpoint_root = Path(root).expanduser().resolve()
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint_root}")
    missing = [name for name in METADATA_FILES if not (checkpoint_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint metadata is missing: {', '.join(missing)}")

    config = _read_json(checkpoint_root / "config.json")
    index = _read_json(checkpoint_root / "quant_model_weights.safetensors.index.json")
    description = _read_json(checkpoint_root / "quant_model_description.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise TypeError("safetensors index does not contain an object weight_map")

    shard_paths: dict[str, Path] = {}
    for path in sorted(checkpoint_root.glob("quant_model_weights-*-of-*.safetensors")):
        match = SHARD_PATTERN.fullmatch(path.name)
        if match:
            shard_paths[path.name] = path
    expected_shards = {
        name for name in weight_map.values() if isinstance(name, str) and SHARD_PATTERN.fullmatch(name)
    }
    declared_totals = {int(SHARD_PATTERN.fullmatch(name).group(2)) for name in expected_shards}
    if len(declared_totals) != 1:
        raise ValueError(f"inconsistent shard totals in index: {sorted(declared_totals)}")
    declared_shard_total = declared_totals.pop()
    if not expected_shards:
        raise ValueError("safetensors index contains no quantized weight shards")
    missing_shards = sorted(expected_shards - shard_paths.keys())

    headers: dict[str, dict[str, Any]] = {}
    for name, path in shard_paths.items():
        headers[name] = {
            "bytes": path.stat().st_size,
            "header_scanned": False,
        }

    target_layer_files: dict[str, list[str]] = {}
    target_tensor_samples: dict[str, dict[str, Any]] = {}
    scanned_headers: dict[str, dict[str, Any]] = {}
    for layer in TARGET_LAYERS:
        prefix = f"layers.{layer}."
        names = sorted(name for name in weight_map if name.startswith(prefix))
        files = {weight_map[name] for name in names}
        target_layer_files[str(layer)] = sorted(files)
        sample_names = [
            name
            for name in (
                f"layers.{layer}.attn.wq_a.weight",
                f"layers.{layer}.attn.wq_a.weight_scale",
                f"layers.{layer}.attn.wq_b.weight",
                f"layers.{layer}.attn.wkv.weight",
                f"layers.{layer}.attn.wo_a.weight",
                f"layers.{layer}.attn.wo_b.weight",
                f"layers.{layer}.attn.indexer.wk.weight",
                f"layers.{layer}.attn.indexer.wq_b.weight",
                f"layers.{layer}.ffn.experts.0.w1.weight",
            )
            if name in weight_map
        ]
        samples: dict[str, Any] = {}
        for name in sample_names:
            shard = weight_map[name]
            if shard not in scanned_headers:
                header_bytes, header_sha256, header = _safetensors_header(checkpoint_root / shard)
                tensors = {key: value for key, value in header.items() if key != "__metadata__"}
                scanned_headers[shard] = header
                headers[shard].update(
                    {
                        "header_bytes": header_bytes,
                        "header_sha256": header_sha256,
                        "tensor_count": len(tensors),
                        "header_scanned": True,
                    }
                )
            header = scanned_headers[shard]
            tensor = header[name]
            samples[name] = {
                "shard": shard,
                "dtype": tensor.get("dtype"),
                "shape": tensor.get("shape"),
            }
        target_tensor_samples[str(layer)] = samples

    quant_modes = Counter(value for value in description.values() if isinstance(value, str))
    engram_files = sorted(
        name
        for name in (
            "engram_embed_weight_l1.safetensors",
            "engram_embed_scale_l1.safetensors",
            "engram_embed_weight_l14.safetensors",
            "engram_embed_scale_l14.safetensors",
        )
        if (checkpoint_root / name).is_file()
    )
    engram_records = []
    for name in engram_files:
        path = checkpoint_root / name
        header_bytes, header_sha256, header = _safetensors_header(path)
        engram_records.append(
            {
                "file": name,
                "bytes": path.stat().st_size,
                "header_bytes": header_bytes,
                "header_sha256": header_sha256,
                "tensors": {
                    key: {"dtype": value.get("dtype"), "shape": value.get("shape")}
                    for key, value in header.items()
                    if key != "__metadata__"
                },
            }
        )
    return {
        "schema_version": 1,
        "status": "metadata_verified" if not missing_shards else "incomplete",
        "root": str(checkpoint_root),
        "payload_hashing": "omitted; safetensors payloads are immutable external assets",
        "metadata": {
            name: {
                "bytes": (checkpoint_root / name).stat().st_size,
                "sha256": _sha256(checkpoint_root / name),
            }
            for name in METADATA_FILES
        },
        "config": _config_summary(config),
        "quantization": {
            "config": config.get("quantization_config"),
            "description_modes": dict(sorted(quant_modes.items())),
        },
        "index": {
            "weight_map_entries": len(weight_map),
            "total_size": index.get("metadata", {}).get("total_size"),
            "weight_shards": len(expected_shards),
            "declared_shard_total": declared_shard_total,
            "present_shards": len(shard_paths),
            "missing_shards": missing_shards,
        },
        "shards": headers,
        "engram_files": engram_records,
        "target_layer_files": target_layer_files,
        "target_tensor_samples": target_tensor_samples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = inspect_checkpoint(args.root)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["status"] == "metadata_verified" else 1


class ResidentCheckpointOperands(CheckpointOperands):
    """Cache model boundary weights while streaming expert source tensors once."""

    def __init__(self, root):
        super().__init__(root, diagnostics=False)
        self.boundaries = {}

    def load(self, name):
        if name in self.boundaries:
            return self.boundaries[name]
        value = super().load(name)
        if name in ("embed.weight", "norm.weight", "head.weight"):
            self.boundaries[name] = value
        return value


class ResidentNetworkWeights:
    """Retain complete compute weights, with routed experts sharded by their actual EP owner."""

    def __init__(self, checkpoint, topology, model_config, runtime):
        self.reader = ResidentCheckpointOperands(checkpoint)
        self.layout = ExpertParallelConfig(topology["dp"], topology["tp"], topology["ep"], 384)
        self.model_config, self.runtime = model_config, runtime
        self.layers, self.experts, self.heads = {}, {}, {}
        self.layer_receipts = {}
        self.load_seconds, self.load_calls = 0.0, 0
        self.ready = False

    def preload(self, progress=None):
        """Stream each checkpoint operand into its persistent compute owner before token execution."""
        if self.ready or self.layers:
            raise ValueError("Resident weight preload is a single generation-scoped operation")
        begin = perf_counter()
        world, tp = self.layout.ep, self.layout.tp
        ranks = range(world)
        for layer in range(40):
            if progress is not None:
                progress(dict(stage="loading_resident_weights", layer=layer, epoch=0))
            prefix = f"layers.{layer}."
            names = [
                name
                for name in self.reader.index
                if name.startswith(prefix) and ".ffn.experts." not in name and ".engram.embed." not in name
            ]
            raw = {name: self.reader.load(name) for name in names}
            operands = prepare_product_operands(raw, layer, role=resolve_layer_role(layer))
            raw[prefix + "ffn.gate.weight"] = raw[prefix + "ffn.gate.weight"].float()
            for name in (
                "hc_ffn_fn",
                "hc_ffn_scale",
                "hc_ffn_base",
                "ffn_norm.weight",
                "ffn.gate.weight",
                "ffn.gate.bias",
                "ffn.gate.bias_vl",
            ):
                self.runtime.pin(raw[prefix + name], ranks, readonly=True, name=prefix + name)
            for name, value in operands.items():
                if name in ("wo_a", "wo_a_grouped_prefill", "wo_b"):
                    continue
                self.runtime.pin(value, ranks, readonly=True, name=prefix + name)
            for rank in ranks:
                lane = rank % tp
                groups, width = 8 // tp, 8192 // tp
                for name, value in (
                    ("wo_a", operands["wo_a_grouped_prefill"][lane * groups : (lane + 1) * groups]),
                    ("wo_b", operands["wo_b"][lane * width : (lane + 1) * width]),
                ):
                    self.runtime.pin(value, [rank], readonly=True, name=prefix + name)
            role = resolve_layer_role(layer)
            if role["engram_slot"] is not None:
                engram = prefix + "engram"
                projection = raw[engram + ".wkv.weight"].T.contiguous()
                operands["engram_projection"] = projection
                self.runtime.pin(projection, ranks, readonly=True, name=engram + ".wkv.kn")
                for name in ("q_weight", "k_weight"):
                    self.runtime.pin(raw[engram + "." + name], ranks, readonly=True, name=engram + "." + name)
            self.layers[layer] = dict(
                raw=raw,
                operands=operands,
                expert=lambda rank, expert, layer=layer: self.expert(layer, rank, expert),
            )
            shared = self.reader.expert(prefix + "ffn.shared_experts")
            for rank in ranks:
                self.experts[layer, rank, None] = tuple(self.runtime.weight(value, rank) for value in shared)
            del shared
            for rank in ranks:
                for expert, _ in self.layout.ownership(rank):
                    values = self.reader.expert(prefix + f"ffn.experts.{expert}")
                    self.experts[layer, rank, expert] = tuple(
                        self.runtime.weight(value, rank) for value in values
                    )
            self.layer_receipts[layer] = dict(
                original_names=names,
                policy="generation-scoped resident compute weights",
                routed_experts=384,
                shared_replicas=world,
            )
        embedding, norm, head = (
            self.reader.load(name) for name in ("embed.weight", "norm.weight", "head.weight")
        )
        rows = embedding.shape[0] // tp
        for rank in ranks:
            lane = rank % tp
            self.runtime.pin(
                embedding.narrow(0, lane * rows, rows), [rank], readonly=True, name="embed.weight"
            )
            self.runtime.pin(norm, [rank], readonly=True, name="norm.weight")
            if lane not in self.heads:
                self.heads[lane] = head.narrow(0, lane * rows, rows).T.contiguous()
            self.runtime.pin(self.heads[lane], [rank], readonly=True, name="head.weight.kn")
        self.ready = True
        self.load_seconds = perf_counter() - begin
        self.load_calls = self.reader.read_calls
        self.runtime.stats.update(
            weight_preload_seconds=self.load_seconds,
            checkpoint_preload_read_calls=self.reader.read_calls,
            checkpoint_preload_read_seconds=self.reader.read_seconds,
        )

    def for_layer(self, layer):
        if not self.ready or layer not in self.layers:
            raise ValueError("Every resident layer must be loaded before execution")
        return self.layers[layer]

    def raw_for_layer(self, layer):
        return self.for_layer(layer)["raw"]

    def projection_for_layer(self, layer):
        return self.for_layer(layer)["operands"]["engram_projection"]

    def expert(self, layer, rank, expert):
        if expert is not None and self.layout.owner(expert) != rank:
            raise ValueError("Resident expert belongs to another EP rank")
        return self.experts[layer, rank, expert]

    def head_for_rank(self, rank):
        return self.heads[rank % self.layout.tp]

    def release_current(self):
        """Preserve generation-scoped layers across all layer and token boundaries."""

    def close(self):
        self.layers.clear()
        self.experts.clear()
        self.heads.clear()
        self.reader.boundaries.clear()


def prepare_product_operands(raw, layer_id=0, *, role=None):
    """Adapt CPU source weights to complete KN and grouped Prefill layouts."""
    from .cache import resolve_layer_role

    role = resolve_layer_role(layer_id, role)
    layer = f"layers.{layer_id}"
    prepared = {}
    for projection in ("wq_a", "wq_b", "wkv"):
        prefix = f"{layer}.attn.{projection}."
        weight, scale = prepare_ecotech_dynamic_weight(
            raw[prefix + "weight"],
            raw[prefix + "weight_scale"],
            raw[prefix + "weight_offset"],
        )
        prepared[projection] = weight
        prepared[projection + "_scale"] = scale
    wo_a, wo_b = raw[f"{layer}.attn.wo_a.weight"], raw[f"{layer}.attn.wo_b.weight"]
    if wo_a.dtype != torch.bfloat16 or wo_b.dtype != torch.bfloat16:
        raise ValueError("Output projections require original BF16 weights")
    if tuple(wo_a.shape) != (8192, 4096) or tuple(wo_b.shape) != (5120, 8192):
        raise ValueError("Output projection geometry differs")
    prepared["wo_a"] = wo_a.reshape(8, 1024, 4096).transpose(1, 2).contiguous()
    prepared["wo_a_grouped_prefill"] = wo_a.reshape(8, 1024, 4096).contiguous()
    prepared["wo_b"] = wo_b.T.contiguous()
    names = dict(
        attn_norm=f"{layer}.attn_norm.weight",
        q_norm=f"{layer}.attn.q_norm.weight",
        kv_norm=f"{layer}.attn.kv_norm.weight",
        sink=f"{layer}.attn.attn_sink",
        hc_function=f"{layer}.hc_attn_fn",
        hc_scale=f"{layer}.hc_attn_scale",
        hc_base=f"{layer}.hc_attn_base",
    )
    for target, source in names.items():
        prepared[target] = raw[source].contiguous()
    if role["mode"] == "full":
        compressor = f"{layer}.attn.compressor"
        indexer = f"{layer}.attn.indexer"
        prepared["compressor_wkv"] = raw[compressor + ".wkv.weight"].T.contiguous()
        if role["ratio"] == 2:
            prepared["compressor_wgate"] = raw[compressor + ".wgate.weight"].T.contiguous()
        prepared["compressor_norm"] = raw[compressor + ".norm.weight"].contiguous()
        prepared["index_wk"] = raw[indexer + ".wk.weight"].T.contiguous()
        prepared["index_norm"] = raw[indexer + ".k_norm.weight"].contiguous()
        prepared["index_weights"] = raw[indexer + ".weights_proj.weight"].T.contiguous()
        prepared["index_wqb"], prepared["index_wqb_scale"] = prepare_ecotech_dynamic_weight(
            raw[indexer + ".wq_b.weight"],
            raw[indexer + ".wq_b.weight_scale"],
            raw[indexer + ".wq_b.weight_offset"],
        )
    if role["mode"] == "reindex":
        indexer = f"{layer}.attn.indexer"
        prepared["index_weights"] = raw[indexer + ".weights_proj.weight"].T.contiguous()
        prepared["index_wqb"], prepared["index_wqb_scale"] = prepare_ecotech_dynamic_weight(
            raw[indexer + ".wq_b.weight"],
            raw[indexer + ".wq_b.weight_scale"],
            raw[indexer + ".wq_b.weight_offset"],
        )
    if any(value.device.type != "cpu" for value in prepared.values()):
        raise ValueError("Operand preparation requires CPU tensors")
    return prepared


if __name__ == "__main__":
    raise SystemExit(main())

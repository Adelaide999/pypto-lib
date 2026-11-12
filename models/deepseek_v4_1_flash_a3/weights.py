# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Checkpoint readers and explicit Prefill and EP2 Decode weight layouts."""

from __future__ import annotations


import json
from pathlib import Path
from typing import Any

import torch


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
from time import perf_counter
from typing import Mapping
from contextlib import contextmanager

from .runtime_utils import tensor_sha


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


from typing import Iterator


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


from collections.abc import Iterator as decode_Iterator
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
) -> decode_Iterator[tuple[int, dict[str, torch.Tensor]]]:
    """Yield the 192 owner experts without retaining a second full shard."""

    if rank not in (0, 1):
        raise ValueError(f"EP2 rank must be 0 or 1, got {rank}")
    first = rank * LOCAL_EXPERTS
    for expert in range(first, first + LOCAL_EXPERTS):
        yield expert, pack_routed_expert(checkpoint, layer, expert)


def load_routed_shard(
    checkpoint: DecodeSafetensorsCheckpoint,
    layer: int,
    rank: int,
) -> dict[str, torch.Tensor]:
    """Materialize one complete 192-expert EP2 shard in external NZ layout.

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

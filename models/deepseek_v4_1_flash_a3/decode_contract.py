# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode shape, candidate and asset contracts."""

from __future__ import annotations
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class DecodeGlobalShape:
    """Physical model dimensions read from the validated text configuration."""

    hidden_size: int = 5120
    attention_heads: int = 64
    kv_heads: int = 1
    head_dim: int = 512
    q_lora_rank: int = 1280
    output_groups: int = 8
    output_lora_rank: int = 1024
    moe_intermediate: int = 2304
    routed_experts: int = 384
    top_k: int = 6
    hc_mult: int = 4
    cache_block_size: int = 128


@dataclass(frozen=True)
class DecodeRankShape:
    """Rank-local dimensions plus the axis ownership that must be captured."""

    dp: int
    tp: int
    ep: int
    q_heads_local: int
    query_width_local: int
    kv_heads_local: int
    kv_head_replicated: bool
    routed_experts_local: int
    expert_shard_axis: str
    q_shard_axis: str
    kv_shard_axis: str
    cache_layout: str
    cache_source_id: str
    dispatch_contract: str
    output_reduce_contract: str
    runtime_shape_status: str = "not_run"


GLOBAL = DecodeGlobalShape()


def rank_shape(dp: int, tp: int, ep: int) -> DecodeRankShape:
    """Return the conservative local shape for one supported topology."""

    if ep != 16:
        raise ValueError("A3 Decode shape contracts require EP16")
    if min(dp, tp, ep) <= 0 or dp * tp != ep:
        raise ValueError(f"invalid Decode topology DP={dp}, TP={tp}, EP={ep}")
    if GLOBAL.attention_heads % tp or GLOBAL.routed_experts % ep:
        raise ValueError("attention heads and routed experts must divide the requested TP/EP")
    return DecodeRankShape(
        dp=dp,
        tp=tp,
        ep=ep,
        q_heads_local=GLOBAL.attention_heads // tp,
        query_width_local=(GLOBAL.attention_heads // tp) * GLOBAL.head_dim,
        kv_heads_local=GLOBAL.kv_heads,
        kv_head_replicated=tp > 1,
        routed_experts_local=GLOBAL.routed_experts // ep,
        expert_shard_axis="expert_id over EP",
        q_shard_axis="query_head over TP",
        kv_shard_axis="replicated KV head; runtime capture required for any split",
        cache_layout="[page, row, kv_head, head_dim]",
        cache_source_id="physical source layer plus request slot",
        dispatch_contract="all-to-all EP dispatch, local expert, all-to-all combine",
        output_reduce_contract="TP grouped output reduction followed by shared/routed sum",
    )


def shape_matrix() -> dict[str, dict[str, object]]:
    """Return the required TP1/TP2/TP4 compatibility shape matrix."""

    rows: dict[str, dict[str, object]] = {}
    for name, dp, tp in (("DP16_TP1_EP16", 16, 1), ("DP8_TP2_EP16", 8, 2), ("DP4_TP4_EP16", 4, 4)):
        rows[name] = asdict(rank_shape(dp, tp, 16))
    return rows


SEARCH_K = 2
BASELINE_ID = "addb655f7b4f4a4ce56af875724a04d79d3fe325"
INPUT_CONTRACT_ID = "decode-contract-v1"
TOPOLOGY_CONTRACT_ID = "dp8-tp2-ep16-p1-pcp1-dcp1"
TOLERANCE_CONTRACT_ID = "rtol-0.05-atol-0.05-discrete-exact"
LAYER_CONTRACT_ID = "physical-layer-v1"
QUANTIZATION_CONTRACT_ID = "w8a8-int8-engram-block128-v1"
STATE_CONTRACT_ID = "decode-token-cache-request-state-v1"


@dataclass(frozen=True)
class DecodeCandidate:
    """One executable PyPTO route evaluated under the shared Decode contract."""

    name: str
    entrypoint: str
    factory: str
    cache_layout: str
    page_size: int
    packing: str
    scale_dtype: str
    status: str
    baseline: str = BASELINE_ID
    input_contract: str = INPUT_CONTRACT_ID
    topology_contract: str = TOPOLOGY_CONTRACT_ID
    tolerance_contract: str = TOLERANCE_CONTRACT_ID
    layer_contract: str = LAYER_CONTRACT_ID
    quantization_contract: str = QUANTIZATION_CONTRACT_ID
    state_contract: str = STATE_CONTRACT_ID
    product_scope: str = "bounded_swa_only"
    formal_status: str = "not_run"

    def contract_identity(self) -> tuple[object, ...]:
        """Return the fields that must remain equal across candidates."""

        return (
            self.baseline,
            self.input_contract,
            self.topology_contract,
            self.tolerance_contract,
            self.layer_contract,
            self.quantization_contract,
            self.state_contract,
            self.page_size,
            self.scale_dtype,
        )


CANDIDATES = (
    DecodeCandidate(
        name="A",
        entrypoint="models.deepseek_v4_1_flash_a3.decode_fwd",
        factory="make_decode_step",
        cache_layout="[block, row, kv_head, head_dim]",
        page_size=128,
        packing="canonical_KN_int8",
        scale_dtype="fp32_channel",
        status="runnable_bounded_swa",
        product_scope="bounded_swa_only",
    ),
    DecodeCandidate(
        name="B",
        entrypoint="models.deepseek_v4_1_flash_a3.decode_replay",
        factory="make_decode_replay(steps=1)",
        cache_layout="[block, row, kv_head, head_dim] padded_state_envelope",
        page_size=128,
        packing="canonical_KN_int8_padded_state",
        scale_dtype="fp32_channel",
        status="runnable_bounded_swa",
        product_scope="bounded_swa_only",
    ),
)


def validate_candidates() -> tuple[DecodeCandidate, ...]:
    """Validate Search K=2 and the shared baseline contract."""

    if len(CANDIDATES) != SEARCH_K:
        raise ValueError(f"Search K={SEARCH_K} requires exactly {SEARCH_K} candidates")
    if len({candidate.name for candidate in CANDIDATES}) != SEARCH_K:
        raise ValueError("Decode candidate names must be unique")
    identities = {candidate.contract_identity() for candidate in CANDIDATES}
    if len(identities) != 1:
        raise ValueError(
            "candidates must share baseline, layer, input, state, topology, quantization, page and scale contracts"
        )
    for candidate in CANDIDATES:
        if not candidate.entrypoint.startswith("models.deepseek_v4_1_flash_a3.decode_"):
            raise ValueError(f"candidate {candidate.name} is outside the Decode package")
        if candidate.status != "runnable_bounded_swa":
            raise ValueError(f"candidate {candidate.name} is not an executable PyPTO route")
        if candidate.product_scope != "bounded_swa_only":
            raise ValueError(f"candidate {candidate.name} has an unsupported product scope")
        if candidate.formal_status != "not_run":
            raise ValueError(f"candidate {candidate.name} cannot be promoted before a formal NPU run")
    return CANDIDATES


SESSION = "deepseek-v4.1-flash-a3-decode"
REQUIRED_FILES = ("checkpoint.json", "decode_input.json", "history_cache.npz", "reference_capture.json")
REQUIRED_INPUT_FIELDS = (
    "checkpoint",
    "layer_id",
    "decode_token",
    "history_token_length",
    "position_ids",
    "initial_kv_cache",
    "block_table",
    "slot_mapping",
    "valid_lengths",
    "mask",
    "active_finished_request_state",
    "engram_rows",
    "cache_source_ids",
    "ratio1_ratio2_state",
)
FULL_CACHE_CRC_LIMIT = 1 << 30


def _read_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"{path.name}: invalid JSON ({exc})")
        return None
    if not isinstance(value, dict):
        errors.append(f"{path.name}: top-level value must be an object")
        return None
    return value


def _field(document: Mapping[str, Any], name: str) -> Any:
    """Resolve a required field from the top-level or an input-state mapping."""

    if name in document:
        return document[name]
    for container_name in ("inputs", "input_state", "state"):
        container = document.get(container_name)
        if isinstance(container, Mapping) and name in container:
            return container[name]
    return None


def _mapping_matches(actual: Any, expected: Mapping[str, Any]) -> bool:
    if not isinstance(actual, Mapping):
        return False
    return all(actual.get(key) == value for key, value in expected.items())


def _reference_flags(document: Mapping[str, Any]) -> Mapping[str, Any]:
    comparison = document.get("comparison")
    return comparison if isinstance(comparison, Mapping) else document


def _validate_layer_state_capture(
    layer: int | None,
    decode_input: Mapping[str, Any] | None,
    reference: Mapping[str, Any] | None,
    errors: list[str],
) -> None:
    """Require physical cache planes beyond the generic MoE/layer capture."""

    if layer is None or decode_input is None or reference is None:
        return
    ratio_state = decode_input.get("ratio1_ratio2_state")
    if not isinstance(ratio_state, Mapping):
        return
    ratio = ratio_state.get("layer_ratio")
    engram_required = ratio_state.get("engram_required") is True
    if ratio == 0:
        if engram_required:
            rows = decode_input.get("engram_rows")
            if not isinstance(rows, Mapping) or rows.get("captured") is not True:
                errors.append("decode_input.json: Engram rows are required and must be captured")
        return
    if ratio not in (1, 2):
        errors.append("decode_input.json: unsupported layer compression ratio")
        return

    required = ["compressed_kv_cache", "index_cache", "index_scale"]
    if ratio == 1:
        required.append("candidate_pool")
    state = reference.get("layer_state_capture")
    if not isinstance(state, Mapping):
        errors.append("reference_capture.json: layer_state_capture is required for C1A/C2A")
        return
    if state.get("complete") is not True:
        errors.append("reference_capture.json: layer_state_capture.complete must be true")
    for name in required:
        value = state.get(name)
        if not isinstance(value, Mapping) or value.get("captured") is not True:
            errors.append(f"reference_capture.json: layer_state_capture.{name} is not captured")


def _check_history_cache(path: Path, errors: list[str]) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if not any(name.endswith(".npy") for name in names):
                errors.append("history_cache.npz: no NumPy array members")
                return
            logical_bytes = sum(info.file_size for info in archive.infolist())
            if logical_bytes <= FULL_CACHE_CRC_LIMIT:
                if archive.testzip() is not None:
                    errors.append("history_cache.npz: ZIP member checksum failed")
                return
            # A full CRC pass would expand the reference cache's tens of GiB
            # of logical zero pages. Validate the NPY magic on the members used
            # by the Decode runner; the runner then streams and checks the
            # exact cache-page prefixes consumed by the product.
            required = ("before_cache_0.npy", "after_cache_0.npy")
            for member in required:
                matches = [name for name in names if name.endswith(member)]
                if not matches:
                    errors.append(f"history_cache.npz: missing {member}")
                    continue
                with archive.open(matches[0]) as stream:
                    if stream.read(6) != b"\x93NUMPY":
                        errors.append(f"history_cache.npz: invalid NPY header for {matches[0]}")
    except (OSError, zipfile.BadZipFile) as exc:
        errors.append(f"history_cache.npz: invalid NPZ container ({exc})")


def validate_assets(
    asset_root: str | Path | None,
    topology: Mapping[str, Any],
    *,
    checkpoint_root: str | Path | None = None,
    expected_reference_commit: str | None = None,
) -> dict[str, Any]:
    """Return a structural validation report for one fixed Decode case."""

    errors: list[str] = []
    root = Path(asset_root).expanduser().resolve() if asset_root else None
    paths = {name: str(root / name) if root else "not_configured" for name in REQUIRED_FILES}
    present = {name: bool(root and (root / name).is_file()) for name in REQUIRED_FILES}
    if root is None:
        errors.append("Decode asset root is not configured")
        return {"status": "not_ready", "paths": paths, "present": present, "errors": errors}

    for name, exists in present.items():
        if not exists:
            errors.append(f"{name}: file is missing")
    if errors:
        return {"status": "not_ready", "paths": paths, "present": present, "errors": errors}

    checkpoint = _read_json(root / "checkpoint.json", errors)
    decode_input = _read_json(root / "decode_input.json", errors)
    reference = _read_json(root / "reference_capture.json", errors)
    if checkpoint is not None:
        configured_root = checkpoint.get("checkpoint_root")
        if not isinstance(configured_root, str) or not configured_root:
            errors.append("checkpoint.json: checkpoint_root is required")
        elif (
            checkpoint_root
            and Path(configured_root).expanduser().resolve() != Path(checkpoint_root).expanduser().resolve()
        ):
            errors.append("checkpoint.json: checkpoint_root differs from DEEPSEEK_V41_CHECKPOINT")
        if checkpoint.get("formal_accuracy") not in ("not_run", "metadata_verified"):
            errors.append("checkpoint.json: formal_accuracy must remain not_run or metadata_verified")

    input_layer: int | None = None
    if decode_input is not None:
        if decode_input.get("session", SESSION) != SESSION:
            errors.append("decode_input.json: session does not match the independent Decode session")
        raw_layer = decode_input.get("layer_id")
        if not isinstance(raw_layer, int) or isinstance(raw_layer, bool) or raw_layer < 0:
            errors.append("decode_input.json: layer_id must be a non-negative integer")
        else:
            input_layer = raw_layer
        if not _mapping_matches(decode_input.get("topology"), topology):
            errors.append("decode_input.json: topology does not match the queued gate")
        for name in REQUIRED_INPUT_FIELDS:
            if _field(decode_input, name) is None:
                errors.append(f"decode_input.json: required field {name!r} is missing")
        cache_ref = _field(decode_input, "initial_kv_cache")
        if isinstance(cache_ref, str) and Path(cache_ref).name != "history_cache.npz":
            errors.append("decode_input.json: initial_kv_cache must reference history_cache.npz")

    _check_history_cache(root / "history_cache.npz", errors)

    if reference is not None:
        if reference.get("session", SESSION) != SESSION:
            errors.append("reference_capture.json: session does not match the independent Decode session")
        if input_layer is not None and reference.get("layer_id") != input_layer:
            errors.append("reference_capture.json: layer_id differs from decode_input.json")
        if not _mapping_matches(reference.get("topology"), topology):
            errors.append("reference_capture.json: topology does not match the queued gate")
        flags = _reference_flags(reference)
        for name in ("same_inputs", "same_layer", "same_topology", "same_state_boundary"):
            if flags.get(name) is not True:
                errors.append(f"reference_capture.json: {name} must be true")
        if reference.get("status") not in ("pass", "captured", "complete"):
            errors.append("reference_capture.json: status must identify a completed capture")
        if expected_reference_commit and reference.get("commit") != expected_reference_commit:
            errors.append("reference_capture.json: vLLM-Ascend commit does not match the pinned reference")
        runtime = reference.get("runtime_identity", reference.get("runtime"))
        if not isinstance(runtime, Mapping):
            errors.append("reference_capture.json: runtime_identity is required")
        if reference.get("product_imported") is True:
            errors.append("reference_capture.json: product process must not import the reference stack")

    _validate_layer_state_capture(input_layer, decode_input, reference, errors)

    return {
        "status": "pass" if not errors else "not_ready",
        "paths": paths,
        "present": present,
        "layer_id": input_layer,
        "errors": errors,
    }

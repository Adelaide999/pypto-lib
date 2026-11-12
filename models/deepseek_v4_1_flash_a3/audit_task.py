# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Audit historical task admission separately from model execution."""

import hashlib
import json
from pathlib import Path


FORMAL_LAYERS = (0, 1, 2, 3, 14, 15, 19, 20, 21, 24, 25)


def file_sha(path):
    """Hash one explicitly supplied historical audit receipt."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def audit_task_release(release, *, continuation=None):
    """Verify the original eleven-layer release receipts for a task audit."""
    if release.get("scope") != "actual_full40_network_prefill" or not release.get("eleven_formal_layers_verified"):
        raise ValueError("Main must independently release full40 after all eleven formal single-layer gates")
    if continuation is not None and not release.get("complete_engram_retained_capacity_verified"):
        raise ValueError("Measure retained complete Engram slots and full host/workspace capacity before history execution")
    expected = {str(layer) for layer in FORMAL_LAYERS}
    if set(release.get("formal_layer_receipts", {})) != expected:
        raise ValueError("The complete independently audited eleven-layer delivery receipt mapping is missing")
    for receipt in release["formal_layer_receipts"].values():
        if file_sha(receipt["path"]) != receipt["sha256"]:
            raise ValueError("A Main-admitted actual formal layer delivery receipt changed")
    if not release.get("actual_capacity_and_sharding_verified") or not release.get("independent_full40_reference_ready"):
        raise ValueError("Actual capacity/sharding and independent same-input full40 reference must be ready before execution")
    return {"historical_release": "verified", "model_execution": "not_run"}


def audit_task_continuation(continuation, *, input_contract_sha256, execution, allocation_id):
    """Check historical chunk admission against a completed execution and retained allocation."""
    admission = continuation.get("chunk0_completion_admission")
    if not isinstance(admission, dict) or admission.get("reviewed_and_frozen") is not True:
        raise ValueError("A completed chunk0 comparison and historical Main admission are required for this audit")
    for name in ("comparison", "current_admission", "owner_state"):
        record = admission[name]
        if file_sha(record["path"]) != record["sha256"]:
            raise ValueError("Completed chunk0 admission evidence changed: " + name)
    comparison = json.loads(Path(admission["comparison"]["path"]).read_text(encoding="utf-8"))
    owner_state = json.loads(Path(admission["owner_state"]["path"]).read_text(encoding="utf-8"))
    execution_sha = file_sha(execution)
    if (comparison.get("outcome") != "valid" or comparison.get("input_contract_sha256") != input_contract_sha256
            or comparison.get("actual_product_execution_sha256") != execution_sha
            or owner_state.get("actual_product_execution_sha256") != execution_sha
            or owner_state.get("retained_allocation_id") != allocation_id
            or owner_state.get("own_state_preserved") is not True):
        raise ValueError("Historical admission does not match the completed chunk and retained owner state")
    return {"historical_continuation": "verified", "model_execution": "not_run"}

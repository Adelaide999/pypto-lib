# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate relocated source identities and bounded HOST controls without device imports."""

import ast
import hashlib
import json
from pathlib import Path

from .config import FLASH
from .runtime_config import ExpertParallelConfig
from . import smla_scheduler, swa_scheduler


ROOT = Path(__file__).resolve().parent


def require(condition, message):
    """Fail every invalid subcase rather than reporting a partial pass."""
    if not condition:
        raise RuntimeError(message)


class _ComputeIdentity(ast.NodeTransformer):
    """Exclude documentation and retain all source computation and ABI expressions."""

    def generic_visit(self, node):
        node = super().generic_visit(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body = node.body[1:]
        if hasattr(node, "type_params") and not node.type_params:
            del node.type_params
        return node


def _definitions(tree):
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            value = ast.dump(_ComputeIdentity().visit(node), include_attributes=False)
            result[node.name] = hashlib.sha256(value.encode()).hexdigest()
    return result


def check_sources():
    """Resolve the complete local import DAG and check every original definition."""
    manifest = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8"))
    modules = {path.stem: path for path in ROOT.glob("*.py")}
    graph, definitions = {}, 0
    for name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        graph[name] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                require(node.level == 1, "Model imports must stay in this flat package")
                imported = [node.module.split(".")[0]] if node.module else [item.name for item in node.names]
                for dependency in imported:
                    require(dependency in modules, "Missing local module: " + dependency)
                    graph[name].add(dependency)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [node.module or ""] if isinstance(node, ast.ImportFrom) else [item.name for item in node.names]
                require(not any(value.startswith("models.") for value in names), "Cross-model coupling is prohibited")
    visited, active = set(), set()

    def visit(name):
        require(name not in active, "Cyclic local import: " + name)
        if name in visited:
            return
        active.add(name)
        for dependency in graph[name]:
            visit(dependency)
        active.remove(name)
        visited.add(name)

    for name in graph:
        visit(name)
    for record in manifest["files"]:
        path = ROOT / record["target"]
        actual = _definitions(ast.parse(path.read_text(encoding="utf-8")))
        require(actual == record["definition_sha256"], "Changed numerical definition: " + path.name)
        definitions += len(actual)
    require(set(modules) == {Path(row["target"]).stem for row in manifest["files"]} | {"validate"},
            "Manifest and packaged source closure differ")
    require(not (ROOT / "grouped_output.py").exists() and (ROOT / "prefill_grouped_output.py").exists(),
            "The dedicated grouped Prefill kernel needs its filename prefix")
    return dict(modules=len(modules), preserved_source_modules=len(manifest["files"]), preserved_definitions=definitions)


def check_layer_plan():
    """Compare every original physical-layer role with the released model configuration."""
    plan = json.loads((ROOT / "prefill_layer_plan.json").read_text(encoding="utf-8"))["layers"]
    require(len(plan) == FLASH.num_hidden_layers == 40, "All original physical layers are required")
    for layer, role in enumerate(plan):
        expected = FLASH.layer_config(layer)
        require(role["physical_layer"] == layer and role["mode"] == expected.mode.value,
                "Layer numbering or attention mode changed")
        require(role["ratio"] == expected.compression_ratio, "Compression ratio changed")
        require(role["kv_source"] == expected.kv_source_layer_id, "KV ownership changed")
        require(role["index_source"] == expected.index_source_layer_id, "Index ownership changed")
        expected_slot = FLASH.engram_layer_ids.index(layer) if layer in FLASH.engram_layer_ids else None
        require(role["engram_slot"] == expected_slot, "Engram slot ownership changed")
    return dict(original_layers=len(plan), engram_layers=list(FLASH.engram_layer_ids))


def _reject(call, message):
    try:
        call()
    except (ValueError, RuntimeError):
        return
    raise RuntimeError(message)


def check_controls():
    """Exercise exact EP ownership, empty ranks, causal page mappings and invalid controls."""
    topologies = [(4, 4, 16), (2, 2, 4), (16, 1, 16), (8, 2, 16)]
    for dp, tp, ep in topologies:
        layout = ExpertParallelConfig(dp, tp, ep)
        owned = [expert for rank in range(ep) for expert, slot in layout.ownership(rank)]
        require(owned == list(range(384)), "Global expert coverage changed")
        for rank in range(ep):
            require(all(layout.owner(expert) == rank for expert, slot in layout.ownership(rank)),
                    "Owner and local expert slot disagree")
            require(layout.active_rows({"active_by_rank": [0] * ep}, rank) == 0, "Empty rank became active")
        _reject(lambda: layout.ownership(ep), "Out-of-world expert rank was accepted")
        _reject(lambda: layout.owner(384), "Out-of-range expert was accepted")
        if tp != 1:
            _reject(lambda: layout.active_rows({"active_by_dp": [1] * dp}, 0),
                    "TP partitions must not invent local token counts")
    _reject(lambda: ExpertParallelConfig(4, 4, 8), "Inconsistent EP world was accepted")
    pages = [list(range(request * 65, (request + 1) * 65)) for request in range(16)]
    schedule = swa_scheduler.build([8] * 16, [8192] * 16, pages, cache_blocks=1040, capacity=128)
    require(schedule["active"] == 128 and schedule["query_start_loc"] == list(range(0, 129, 8)),
            "Multiple requests lost their packed row partitions")
    for row, position in enumerate(schedule["positions"]):
        request = row // 8
        logical = list(range(position - 127, position + 1))
        physical = [pages[request][token // 128] * 128 + token % 128 for token in logical]
        require(schedule["logical_window_indices"][row] == logical, "Noncausal SWA window")
        require(schedule["physical_window_indices"][row] == physical, "Cross-request physical page alias")
    fresh = swa_scheduler.build([2, 1], [0, 0], [[1], [0]], cache_blocks=2, capacity=16)
    require(fresh["flat_slot_mapping"] == [128, 129, 0], "Fresh request page permutation changed")
    _reject(lambda: swa_scheduler.build([1, 1], [0, 0], [[0], [0]], cache_blocks=1, capacity=16),
            "Aliased live request pages were accepted")
    _reject(lambda: swa_scheduler.build([True], [0], [[0]], cache_blocks=1, capacity=16),
            "Boolean scheduler control was accepted as an integer")
    selected = [[0] + [-1] * 511]
    compressed = smla_scheduler.from_controls([0, 1], [1], [[0]], [[1]], selected,
                                              ori_blocks=1, cmp_blocks=2, capacity=16)
    require(compressed["physical_compressed_indices"][0][0] == 128, "Compressed selection lost its source page")
    _reject(lambda: smla_scheduler.from_controls([0, 1], [1], [[0]], [[1]], [[1] + [-1] * 511],
                                               ori_blocks=1, cmp_blocks=2, capacity=16),
            "A future compressed token was accepted")
    return dict(expert_topologies=len(topologies), experts=384, scheduling_requests=16, scheduling_active_rows=128)


def main():
    """Publish a bounded HOST result without claiming numerical device acceptance."""
    result = dict(source=check_sources(), layer_plan=check_layer_plan(), controls=check_controls(),
                  device_initialization="not_run", compilation="not_run", NPU_numerics="not_run")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

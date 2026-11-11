# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Audit original export provenance or the explicitly recorded maintenance baseline."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a2"

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class ComputeIdentity(ast.NodeTransformer):
    """Exclude documentation from recorded definition identities."""

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


def definition_hashes(path):
    """Hash maintained top-level definitions for an opt-in source identity audit."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definitions = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            normalized = ast.dump(ComputeIdentity().visit(node), include_attributes=False)
            definitions[node.name] = hashlib.sha256(normalized.encode()).hexdigest()
    return definitions


def audit_original(root):
    """Verify an unmodified exported source checkout against its original manifest."""
    manifest = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = Path(root) / record["source"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["source_sha256"]:
            raise ValueError("Original exported source changed: " + record["source"])
    return {"source_commit": manifest["source_commit"], "original_files": len(manifest["files"])}


def audit_maintained():
    """Compare code against the reviewed post-refactor source baseline, separately from correctness."""
    baseline = json.loads((ROOT / "maintenance_baseline.json").read_text(encoding="utf-8"))
    original_sha = hashlib.sha256((ROOT / "source_manifest.json").read_bytes()).hexdigest()
    if original_sha != baseline["original_manifest_sha256"]:
        raise ValueError("The original export manifest differs from the maintenance audit baseline")
    paths = {path.name: path for path in ROOT.glob("*.py")}
    if paths.keys() != baseline["definitions"].keys():
        raise ValueError("Maintained module set differs from the explicit audit baseline")
    for name, path in paths.items():
        if definition_hashes(path) != baseline["definitions"][name]:
            raise ValueError("Maintained source identity changed: " + name)
    return {"maintained_modules": len(paths), "source_identity": "matched", "numerical_validation": "not_run"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, help="Unmodified exported checkout; defaults to maintained identity audit")
    args = parser.parse_args()
    print(json.dumps(audit_original(args.original_root) if args.original_root else audit_maintained(), sort_keys=True))


if __name__ == "__main__":
    main()

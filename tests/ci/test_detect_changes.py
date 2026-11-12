# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise CI dependency selection without importing model or device runtimes."""

import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("ci_detect_changes", REPO / ".github/scripts/detect_changes.py")
detect = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(detect)

ENTRY = '''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--platform")
if __name__ == "__main__":
    parser.parse_args()
'''


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "platforms.json"
    config.write_text(json.dumps({"models/": ["a2a3"], "examples/": ["a2a3", "a5"]}), encoding="utf-8")
    monkeypatch.setattr(detect, "PLATFORM_CONFIG", config)

    def write(name, source="VALUE = 1\n"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        return name

    return write


@pytest.mark.parametrize("statement", [
    "import leaf",
    "from leaf import VALUE",
    "import models.pkg.leaf",
    "from models.pkg.leaf import VALUE",
    "from models.pkg import leaf as operand",
    "from .leaf import VALUE",
    "from .leaf import *",
    "from . import leaf as operand",
    "def build():\n    from .leaf import VALUE",
])
def test_import_spellings_select_the_dependent_entry(source_tree, statement):
    leaf = source_tree("models/pkg/leaf.py")
    entry = source_tree("models/pkg/entry.py", statement + "\n" + ENTRY)
    source_tree("models/pkg/unrelated.py", ENTRY)
    assert detect.select_runnable([leaf]) == [entry]


def test_relative_parent_and_multihop_imports_select_transitive_entry(source_tree):
    leaf = source_tree("models/pkg/leaf.py")
    source_tree("models/pkg/nested/helper.py", "from ..leaf import VALUE\n")
    entry = source_tree("models/pkg/nested/entry.py", "from . import helper\n" + ENTRY)
    assert detect.select_runnable([leaf]) == [entry]


def test_explicit_relative_import_is_not_shadowed_by_root_module(source_tree):
    source_tree("golden.py")
    leaf = source_tree("models/pkg/golden.py")
    relative = source_tree("models/pkg/relative.py", "from .golden import VALUE\n" + ENTRY)
    source_tree("models/pkg/absolute.py", "from golden import VALUE\n" + ENTRY)
    assert detect.select_runnable([leaf]) == [relative]


def test_package_initializers_and_imported_submodules_are_dependencies(source_tree):
    package = source_tree("models/pkg/child/__init__.py")
    leaf = source_tree("models/pkg/child/leaf.py")
    entry = source_tree("models/pkg/entry.py", "from .child import leaf\n" + ENTRY)
    rooted = source_tree("models/pkg/rooted.py", "from models.pkg.child import leaf\n" + ENTRY)
    assert detect.select_runnable([package]) == [entry, rooted]
    assert detect.select_runnable([leaf]) == [entry, rooted]


def test_relative_import_inside_initializer_stays_in_its_package(source_tree):
    leaf = source_tree("models/pkg/leaf.py")
    source_tree("models/pkg/__init__.py", "from .leaf import VALUE\n")
    entry = source_tree("models/pkg/entry.py", "from . import VALUE\n" + ENTRY)
    assert detect.select_runnable([leaf]) == [entry]


def test_invalid_relative_import_does_not_fall_back_to_a_sibling(source_tree):
    leaf = source_tree("models/pkg/leaf.py")
    source_tree("models/pkg/entry.py", "from ...leaf import VALUE\n" + ENTRY)
    assert detect.select_runnable([leaf]) == []


def test_relative_import_closure_keeps_platform_and_simulator_filters(source_tree):
    leaf = source_tree("models/pkg/leaf.py")
    entry = source_tree("models/pkg/entry.py", "# ci: no-sim\nfrom .leaf import VALUE\n" + ENTRY)
    assert detect.select_runnable([leaf], "a2a3") == [entry]
    assert detect.select_runnable([leaf], "a2a3sim") == []
    assert detect.select_runnable([leaf], "a5") == []


def test_source_paths_match_git_posix_paths_on_every_host(source_tree):
    leaf = source_tree("models/pkg/leaf.py")
    assert list(detect._iter_source_files()) == [leaf]


def test_manual_driver_is_excluded_but_remains_in_the_dependency_graph(source_tree):
    leaf = source_tree("models/pkg/leaf.py")
    driver = source_tree("models/pkg/driver.py", "# ci: manual\nfrom .leaf import VALUE\n" + ENTRY)
    entry = source_tree("models/pkg/entry.py", "from .driver import VALUE\n" + ENTRY)
    for platform in ("a2a3", "a2a3sim"):
        assert detect.select_runnable([leaf], platform) == [entry]
        assert detect.select_runnable([driver], platform) == [entry]
        assert detect.all_entries(platform) == [entry]


def test_a3_embedding_entry_has_a2a3_platform_ownership(monkeypatch):
    monkeypatch.chdir(REPO)
    entry = "models/deepseek_v4_1_flash_a3/prefill_embedding.py"
    assert detect.platforms_for(entry) == ["a2a3"]
    assert entry in detect.select_runnable([entry], "a2a3")
    assert detect.select_runnable([entry], "a5") == []


def test_a3_shared_config_change_selects_downstream_routed_expert(monkeypatch):
    monkeypatch.chdir(REPO)
    leaf = "models/deepseek_v4_1_flash_a3/config.py"
    entry = "models/deepseek_v4_1_flash_a3/expert_routed.py"
    assert entry in detect.select_runnable([leaf], "a2a3")
    assert detect.select_runnable([leaf], "a5") == []

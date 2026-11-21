# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the resident DP4/TP4/EP16 Prefill and Decode programs."""

import argparse
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--staged-network", action="store_true")
    selector.add_argument("--full-graph", action="store_true")
    selector.add_argument("--graph-dir")
    selector.add_argument("--logits-mode")
    selector.add_argument("--prefill-moe-slab-rows", type=int)
    selected, other = selector.parse_known_args(arguments)
    legacy_options = {"--max-num-batched-tokens", "--hbm-budget",
                      "--enable-chip-swimlane", "--profile-phase", "--profile-output", "--assets"}
    staged = (selected.staged_network or selected.logits_mode == "all" or selected.prefill_moe_slab_rows == 16
              or any(arg.split("=", 1)[0] in legacy_options for arg in other))
    if staged:
        from .staged_generate import main as staged_main

        if selected.full_graph or selected.graph_dir is not None:
            raise SystemExit("Native graph options cannot be used with staged-network options")
        return staged_main([arg for arg in arguments if arg != "--staged-network"])
    from .driver import main as run

    return run(None, mode="prefill", capacity=None, argv=arguments)


if __name__ == "__main__":
    main()

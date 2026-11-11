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
import sys


def _parse_parallel_size(name: str, default: int) -> int:
    """Read one integer parallel-size option without changing model config."""

    flag = f"--{name}"
    for index, argument in enumerate(sys.argv):
        if argument == flag and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if argument.startswith(f"{flag}="):
            return int(argument.split("=", 1)[1])
    return default


TP_SIZE = _parse_parallel_size("tp", 1)


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


__all__ = ["ExpertParallelConfig", "TP_SIZE"]

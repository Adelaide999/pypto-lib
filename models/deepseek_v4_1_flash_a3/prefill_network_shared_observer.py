# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Small shared-state snapshots and exact preservation witnesses; draft only."""

import hashlib
import json


NAMES = ("topk_indices", "topk_lengths", "candidates", "candidate_lengths")
SHAPES = {"topk_indices": (8192, 512), "topk_lengths": (8192, 1),
          "candidates": (8192, 1, 2048), "candidate_lengths": (8192, 1)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def snapshot(value):
    return value.detach().cpu().clone()


def digest(value, start):
    data = value[start:].detach().cpu().contiguous()
    header = dict(dtype=str(value.dtype), shape=list(value.shape), start=start)
    state = hashlib.sha256(json.dumps(header, sort_keys=True).encode() + b"\0")
    state.update(data.numpy().tobytes(order="C"))
    return state.hexdigest()


def begin_shared(shared, query_rows, *, index_write, candidate_write):
    require(0 <= query_rows <= 16, "Require the frozen bounded query prefix")
    result = dict(query_rows=query_rows, prefix_before={}, complements={})
    for name in NAMES:
        value = shared[name]
        require(tuple(value.shape) == SHAPES[name] and str(value.dtype) == "torch.int32",
                "Shared scheduler storage ABI differs: " + name)
        writes = ((name in ("topk_indices", "topk_lengths") and index_write)
                  or (name == "candidates" and candidate_write))
        start = query_rows if writes else 0
        result["prefix_before"][name] = snapshot(value[:query_rows])
        result["complements"][name] = dict(start=start, end=8192,
            before_sha256=digest(value, start), policy="exact_own_before_after", result_dependent_mask=False)
    return result


def end_shared(shared, before):
    result = dict(query_rows=before["query_rows"], prefix_before=before["prefix_before"],
                  prefix_after={}, complements={})
    for name in NAMES:
        previous = before["complements"][name]
        after = digest(shared[name], previous["start"])
        result["prefix_after"][name] = snapshot(shared[name][:before["query_rows"]])
        result["complements"][name] = dict(previous, after_sha256=after,
            exact_equal=previous["before_sha256"] == after)
        require(result["complements"][name]["exact_equal"], "Changed readonly/inactive shared state: " + name)
    return result

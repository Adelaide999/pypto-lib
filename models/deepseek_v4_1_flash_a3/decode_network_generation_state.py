# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate the CPU state exchanged between complete Prefill and Decode phases."""

import torch


STATE_SCHEMA = "deepseek-v4.1-flash-generation-state-v1"


def validate_generation_state(packet):
    """Reject incomplete owners, invalid coordinates and unsupported request layouts."""
    topology = packet["topology"]
    world = topology["ep"]
    if (packet["schema"] != STATE_SCHEMA or topology != dict(dp=world, tp=1, ep=world, pp=1, pcp=1, dcp=1)
            or world not in (2, 16) or packet["completed_layers"] != list(range(40))
            or set(packet["sources"]) != {2, 8, 14, 20}):
        raise ValueError("State requires complete source ownership and a shared DP/EP topology")
    if any(len(packet[name]) != world for name in ("histories", "layers", "candidates", "engram_history")):
        raise ValueError("State must cover every logical rank")

    def tensor(value, shape, dtype):
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or list(value.shape) != list(shape) or value.dtype != dtype):
            raise ValueError(f"State tensor must have shape {shape} and dtype {dtype}")

    for rank, history in enumerate(packet["histories"]):
        count = len(history)
        if not 1 <= count <= 256 or any(type(token) is not int or not 0 <= token < 129280 for token in history):
            raise ValueError("State history exceeds the bounded Decode request ABI")
        if len(packet["layers"][rank]) != 40:
            raise ValueError("State must contain all forty independent SWA owners")
        for layer in packet["layers"][rank]:
            tensor(layer["swa"], (count, 512), torch.bfloat16)
            if "selection" in layer:
                tensor(layer["selection"], (512,), torch.int32)
        tensor(packet["engram_history"][rank], (count,), torch.int32)
        for owner, rows in packet["sources"].items():
            if len(rows) != world or rows[rank]["ratio"] != (1 if owner == 20 else 2):
                raise ValueError("State source rank or compression ratio differs")
            source = rows[rank]
            visible = count // source["ratio"]
            tensor(source["kv"], (visible, 512), torch.bfloat16)
            tensor(source["codes"], (visible, 128), torch.int8)
            tensor(source["scales"], (visible,), torch.float16)
            if source["ratio"] == 2:
                tensor(source["ring"], (32, 1024), torch.float32)
        candidate = packet["candidates"][rank]
        if not 0 <= candidate["count"] <= 64 or not 0 <= candidate["topk_length"] <= 512:
            raise ValueError("State candidate pool exceeds the bounded Decode capacity")
        tensor(candidate["ids"], (candidate["count"],), torch.int32)
    return packet

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compose vocabulary lookup, TP exchange, PyPTO summation and initial SP/HC streams."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"
import torch
from .prefill_layer import region_arguments


def run_input_boundary(settings, *, config, capacity, reader, execute, exchange, execute_l2=None):
    """Embed the actual request tokens and return rank-local HC inputs in memory."""
    topology = settings["topology"]
    dp, tp, world = topology["dp"], topology["tp"], topology["ep"]
    hidden, vocabulary, hc_mult = config["hidden_size"], config["vocab_size"], config["hc_mult"]
    if world != dp * tp or len(settings["DP_engines"]) != dp:
        raise ValueError("Input request engines must match the complete network topology")
    if any(not 0 <= engine["active"] <= capacity for engine in settings["DP_engines"]):
        raise ValueError("Actual request tokens exceed the compiled Prefill capacity")
    weight = reader.load("embed.weight")
    if tuple(weight.shape) != (vocabulary, hidden) or weight.dtype != torch.bfloat16:
        raise ValueError("Actual checkpoint embedding shape or dtype differs")
    packets, rows_by_rank = [], []
    for rank in range(world):
        engine = settings["DP_engines"][rank // tp]
        tokens = [token for request in engine["requests"] for token in request["token_ids"]]
        masks = [value for request in engine["requests"] for value in request["live_token_mask"]]
        if len(tokens) != engine["active"] or len(masks) != len(tokens):
            raise ValueError("Request/token ownership differs from the actual prepared boundary")
        if any(token < 0 or token >= vocabulary for token in tokens):
            raise ValueError("Prepared token ID is outside the original vocabulary")
        ids, live = torch.zeros(capacity, dtype=torch.int64), torch.zeros(capacity, dtype=torch.int32)
        ids[:len(tokens)] = torch.tensor(tokens, dtype=torch.int64)
        live[:len(tokens)] = torch.tensor(masks, dtype=torch.int32)
        output = torch.zeros((capacity, hidden), dtype=torch.bfloat16)
        start, count = rank % tp * (vocabulary // tp), vocabulary // tp
        local_weight = weight.narrow(0, start, count).contiguous()
        execute("lookup", rank, [ids, live, local_weight, output, start, len(tokens)], dict(lookup=output))
        packets.append({"network-input": {"value": output}})
        rows_by_rank.append(len(tokens))
    gathered = exchange("input-vocabulary-TP-gather", packets)
    result = []
    for rank in range(world):
        active = rows_by_rank[rank]
        physical = max(1, (active + tp - 1) // tp)
        embedding = torch.zeros((capacity, hidden), dtype=torch.bfloat16)
        streams = torch.zeros((capacity, hc_mult, hidden), dtype=torch.bfloat16)
        premix = torch.zeros((capacity, hc_mult), dtype=torch.float32)
        operations = (
            ("reduce", [gathered[rank]["network-input"]["value"], embedding, active], dict(embedding=embedding), "input"),
            ("initial_hc", [embedding, streams, premix, rank % tp, physical, active],
             dict(hidden=streams, pre_mix=premix), "input"),
        )
        if execute_l2 is None:
            for name, arguments, outputs, _ in operations:
                execute(name, rank, arguments, outputs)
        else:
            arguments, outputs, _ = region_arguments("input_l2", operations)
            execute_l2("input_l2", rank, arguments, outputs)
        valid = max(0, min(physical, active - rank % tp * physical))
        result.append(dict(hidden=streams[:(active + tp - 1) // tp].clone(),
            pre_mix=premix[:(active + tp - 1) // tp].clone(),
            SP_physical_rows=(active + tp - 1) // tp, SP_kernel_rows=physical, SP_valid_rows=valid, DP_rank=rank // tp, TP_rank=rank % tp,
            global_token_start=rank % tp * physical))
    return result

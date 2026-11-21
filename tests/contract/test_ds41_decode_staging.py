# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check that single-token staging preserves the resident Decode ABI."""

from types import SimpleNamespace

import pytest
import torch

from models.deepseek_v4_1_flash_a3.host import prepare_prefill_inputs, prepare_resident_inputs
from models.deepseek_v4_1_flash_a3.operators.config import FLASH
from models.deepseek_v4_1_flash_a3.operators.engram import ENGRAM_K
from models.deepseek_v4_1_flash_a3.operators.utils import DECODE_CACHE_NAMES, prepare_decode_inputs
from models.deepseek_v4_1_flash_a3.replay import DecodeInputs


@pytest.mark.parametrize("capacity", [16, 80])
def test_single_token_staging_matches_all_three_tp_images(capacity):
    def embeddings(ids):
        return ids[:, None].expand(-1, FLASH.hidden_size).to(torch.bfloat16)

    def lookup(ids, rows, position, *, request_id):
        result = torch.zeros(2, rows, ENGRAM_K, dtype=torch.bfloat16)
        result[:, 0].fill_(int(ids[0]) + position + request_id)
        return result

    weights = SimpleNamespace(lookup_rows=embeddings)
    engram = SimpleNamespace(lookup=lookup)
    empty = prepare_prefill_inputs(embeddings(torch.empty(0)), capacity, max_seq_len=130)
    cache = {name: empty[name] for name in DECODE_CACHE_NAMES.values()}
    rope = empty["freqs_cos"], empty["freqs_sin"]
    idle = {key: value for key, value in prepare_decode_inputs(empty).items()
            if key not in ("freqs_cos", "freqs_sin")}
    full = DecodeInputs(idle)
    single = DecodeInputs(idle)
    for position, owner in ((0, 0), (127, 12), (128, 12), (129, 4)):
        token = position + 3
        active = prepare_resident_inputs(
            weights, [token], capacity, start_pos=position, max_seq_len=130,
            cache_state=cache, rope_tables=rope, engram=engram, request_id=owner,
        )
        lowered = prepare_decode_inputs(active)
        full.stage({False: idle, True: {name: lowered[name] for name in idle}}, "logit_row_indices")
        single.stage_token(weights, engram, token, position, owner)
        assert torch.equal(single.images, full.images)

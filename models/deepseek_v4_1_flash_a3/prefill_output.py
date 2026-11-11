# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compose shared HC collapse, final RMSNorm and vocabulary projection."""

import pypto.language as pl

from .hc import make_hc_collapse_inline
from .prefill_cp import make_sp_gather_unpack
from .prefill_norm import make_prefill_norm
from .rmsnorm import D, EPS
from .gemm import make_bf16_linear


def make_output_entries(capacity, config, tp):
    """Build bounded output chunks with a vocabulary-axis TP shard."""
    if tp not in (1, 2, 4) or capacity <= 0 or capacity % 16:
        raise ValueError("Require TP1/2/4 and complete 16-row output chunks")
    hidden, vocabulary = config["hidden_size"], config["vocab_size"]
    if hidden != D or config["rms_norm_eps"] != EPS:
        raise ValueError("Final normalization must match the actual shared model configuration")
    if vocabulary % tp or vocabulary // tp % 64:
        raise ValueError("Freeze the actual vocabulary padding before output construction")
    return dict(
        collapse=pl.jit(make_hc_collapse_inline(capacity, config)._func),
        norm=make_prefill_norm(capacity, hidden),
        projection=make_bf16_linear(capacity, hidden, vocabulary // tp),
        sp_unpack=make_sp_gather_unpack(capacity, hidden, tp),
        vocab_unpack=make_output_vocab_unpack(capacity, vocabulary, tp),
    )


def make_output_vocab_unpack(capacity, vocabulary, tp):
    """Restore TP vocabulary packets to original token and vocabulary order."""
    local_width = vocabulary // tp
    if vocabulary % tp or local_width % 64:
        raise ValueError("Require complete 64-column vocabulary blocks per TP shard")

    @pl.jit
    def vocab_unpack(
        packet: pl.Tensor[[tp * capacity, local_width], pl.BF16],
        output: pl.InOut[pl.Tensor[[capacity, vocabulary], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(active * (vocabulary // 64), name_hint="prefill_output_vocab_unpack"):
            token = block // (vocabulary // 64)
            column = block % (vocabulary // 64) * 64
            peer = column // local_width
            source = peer * capacity + token
            local_column = column % local_width
            value = packet[source:source + 1, local_column:local_column + 64]
            output[token:token + 1, column:column + 64] = value
        return output

    return vocab_unpack

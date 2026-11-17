# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Final HC collapse, normalization, vocabulary projection and output unpacking."""

import pypto.language as pl
from .gemm import make_bf16_linear
from .hc import make_hc_collapse_inline
from .input_pack import make_sp_gather_unpack
from .rmsnorm import D, EPS, make_prefill_norm


def make_output_entries(capacity, config, tp, *, sample_rows=False):
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
        sp_unpack=(make_sample_gather_unpack(capacity, hidden, tp) if sample_rows
                   else make_sp_gather_unpack(capacity, hidden, tp)),
        vocab_unpack=make_output_vocab_unpack(capacity, vocabulary, tp),
    )


def make_sample_gather_unpack(capacity, hidden, tp):
    """Select final prompt/decode rows directly from the gathered SP packets."""

    @pl.jit
    def unpack(
        packet: pl.Tensor[[tp * capacity, hidden], pl.BF16],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        rows: pl.Tensor[[capacity], pl.INT32],
        per_rank: pl.Scalar[pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(active * (hidden // 64), name_hint="sample_output_gather"):
            token = block // (hidden // 64)
            column = block % (hidden // 64) * 64
            logical = pl.read(rows, [token])
            source = pl.cast(logical // per_rank * capacity + logical % per_rank, pl.INDEX)
            output[token:token + 1, column:column + 64] = packet[source:source + 1, column:column + 64]
        return output

    return unpack


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
            value = packet[source : source + 1, local_column : local_column + 64]
            output[token : token + 1, column : column + 64] = value
        return output

    return vocab_unpack


def golden_sample_gather(args):
    for token in range(int(args["active"])):
        logical, physical = int(args["rows"][token]), int(args["per_rank"])
        source = logical // physical * args["output"].shape[0] + logical % physical
        args["output"][token].copy_(args["packet"][source])


def sample_gather_golden_cases():
    """Cover TP packet boundaries, omitted rows, empty outputs and untouched tails."""
    import torch
    from golden import ScalarSpec, TensorSpec
    from golden.validation import ratio_allclose

    capacity, hidden = 16, 128
    for tp in (1, 2, 4):
        packet = (torch.arange(tp * capacity * hidden).reshape(tp * capacity, hidden) % 251).bfloat16()
        physical = (13 + tp - 1) // tp
        for active in (0, 1, 3):
            rows = torch.zeros(capacity, dtype=torch.int32)
            rows[:3] = torch.tensor([12, 0, 7], dtype=torch.int32)
            yield dict(name=f"samples-tp{tp}-active{active}",
                fn=make_sample_gather_unpack(capacity, hidden, tp),
                specs=[TensorSpec("packet", list(packet.shape), packet.dtype, init_value=packet),
                       TensorSpec("output", [capacity, hidden], torch.bfloat16, init_value=-12.5),
                       TensorSpec("rows", [capacity], torch.int32, init_value=rows),
                       ScalarSpec("per_rank", torch.int32, physical, compile_runtime=True),
                       ScalarSpec("active", torch.int32, active, compile_runtime=True)],
                golden_fn=golden_sample_gather,
                compare_fn={"output": ratio_allclose(rtol=0.0, atol=0.0, max_error_ratio=0.0)})


def main():
    import argparse
    from golden import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--sim", action="store_true")
    args = parser.parse_args()
    for case in sample_gather_golden_cases():
        name = case.pop("name")
        result = run(**case, compile_only=args.compile_only,
                     config=dict(backend_type=args.platform, execution_mode=args.sim, device_id=args.device))
        if not result.passed:
            raise RuntimeError(f"Sample gather failed: {name}: {result.error}")


if __name__ == "__main__":
    main()

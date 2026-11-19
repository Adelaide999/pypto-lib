# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Final HC collapse, normalization, vocabulary projection and output unpacking."""

from dataclasses import asdict

import pypto.language as pl
from .runtime_config import SPMD_WORKER_LIMIT

from .config import FLASH
from .gemm import make_bf16_linear, make_decode_bf16_linear
from .hc import make_hc_collapse_inline
from .input_pack import make_sp_gather_unpack
from .rmsnorm import D, EPS, NORM_T_TILE, make_prefill_norm, rms_norm_tiled


def make_output_entries(capacity, config, tp, *, logits_mode="all", sample_rows=False):
    """Build bounded output chunks with a vocabulary-axis TP shard."""
    if tp not in (1, 2, 4) or capacity <= 0 or capacity % 16:
        raise ValueError("Require TP1/2/4 and complete 16-row output chunks")
    hidden, vocabulary = config["hidden_size"], config["vocab_size"]
    if hidden != D or config["rms_norm_eps"] != EPS:
        raise ValueError("Final normalization must match the actual shared model configuration")
    if vocabulary % tp or vocabulary // tp % 64:
        raise ValueError("Freeze the actual vocabulary padding before output construction")
    if logits_mode not in ("all", "last"):
        raise ValueError("Output logits mode must be all or last")
    last_row = logits_mode == "last" and not sample_rows
    output_capacity = 16 if last_row else capacity
    return dict(
        collapse=pl.jit(make_hc_collapse_inline(capacity, config)._func),
        norm=make_prefill_norm(output_capacity, hidden),
        projection=make_bf16_linear(output_capacity, hidden, vocabulary // tp),
        sp_unpack=(make_sample_gather_unpack(capacity, hidden, tp) if sample_rows
                   else make_sp_gather_unpack(capacity, hidden, tp, last_row=last_row)),
        vocab_unpack=make_output_vocab_unpack(output_capacity, vocabulary, tp),
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
        block_count = active * (hidden // 64)
        for worker in pl.spmd(pl.min(block_count, SPMD_WORKER_LIMIT), name_hint="sample_output_gather"):
            for block in pl.range(worker, block_count, SPMD_WORKER_LIMIT):
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
        block_count = (active + 15) // 16 * (vocabulary // 64)
        for block_worker in pl.spmd(pl.min(block_count, SPMD_WORKER_LIMIT), name_hint='prefill_output_vocab_unpack'):
            for block in pl.range(block_worker, block_count, SPMD_WORKER_LIMIT):
                token = block // (vocabulary // 64) * 16
                column = block % (vocabulary // 64) * 64
                peer = column // local_width
                source = peer * capacity + token
                local_column = column % local_width
                rows = pl.min(16, active - token)
                value = packet[source:source + 16, local_column:local_column + 64]
                output[token:token + 16, column:column + 64] = pl.set_validshape(value, rows, 64)
        return output

    return vocab_unpack


def make_decode_head(capacity=16):
    """Compose the existing output operators after the final physical layer."""
    hidden, hc, vocabulary = FLASH.hidden_size, FLASH.hc_mult, FLASH.vocab_size
    collapse = make_hc_collapse_inline(capacity, asdict(FLASH))
    project = pl.jit.inline(make_decode_bf16_linear(capacity, hidden, vocabulary)._func)

    @pl.jit
    def decode_head(
        streams: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        premix: pl.Tensor[[capacity, hc], pl.FP32],
        norm_weight: pl.Tensor[[hidden], pl.BF16],
        head_weight: pl.Tensor[[hidden, vocabulary], pl.BF16],
        logits: pl.InOut[pl.Tensor[[capacity, vocabulary], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        collapsed = pl.create_tensor([capacity, hidden], dtype=pl.BF16)
        collapse(streams, premix, collapsed, active)
        normalized = pl.create_tensor([capacity, hidden], dtype=pl.BF16)
        for block in pl.spmd((active + NORM_T_TILE - 1) // NORM_T_TILE, name_hint="decode_final_norm"):
            rms_norm_tiled(collapsed, norm_weight, normalized, block, active)
        project(normalized, head_weight, logits, active)
        return logits

    return decode_head


def golden_sample_gather(args):
    for token in range(int(args["active"])):
        logical, physical = int(args["rows"][token]), int(args["per_rank"])
        source = logical // physical * args["output"].shape[0] + logical % physical
        args["output"][token].copy_(args["packet"][source])


def sample_gather_golden_cases():
    """Cover TP packet boundaries, empty outputs, untouched tails and worker reuse."""
    import torch
    from golden import ScalarSpec, TensorSpec
    from golden.validation import ratio_allclose

    capacity = 16
    shapes = [(tp, 128, active) for tp in (1, 2, 4) for active in (0, 1, 3)]
    shapes.append((4, FLASH.hidden_size, 4))
    for tp, hidden, active in shapes:
        packet = (torch.arange(tp * capacity * hidden).reshape(tp * capacity, hidden) % 251).bfloat16()
        physical = (13 + tp - 1) // tp
        rows = torch.zeros(capacity, dtype=torch.int32)
        rows[:4] = torch.tensor([12, 0, 7, 4], dtype=torch.int32)
        yield dict(
            name=f"samples-tp{tp}-active{active}",
            fn=make_sample_gather_unpack(capacity, hidden, tp),
            specs=[TensorSpec("packet", list(packet.shape), packet.dtype, init_value=packet),
                   TensorSpec("output", [capacity, hidden], torch.bfloat16, init_value=-12.5),
                   TensorSpec("rows", [capacity], torch.int32, init_value=rows),
                   ScalarSpec("per_rank", torch.int32, physical, compile_runtime=True),
                   ScalarSpec("active", torch.int32, active, compile_runtime=True)],
            golden_fn=golden_sample_gather,
            compare_fn={"output": ratio_allclose(rtol=0.0, atol=0.0, max_error_ratio=0.0)},
        )


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

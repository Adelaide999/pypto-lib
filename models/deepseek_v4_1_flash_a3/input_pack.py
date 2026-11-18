# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Token embedding, input packing and initial HC computations."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"
import pypto.language as pl
from .runtime_config import SPMD_WORKER_LIMIT


# tiling
D_TILE = 512


def make_vocab_embedding(capacity, hidden, vocabulary, tp):
    """Copy owned vocabulary rows and zero other active rows before TP reduction."""
    if capacity < 1 or hidden % D_TILE or tp not in (1, 2, 4) or vocabulary % tp:
        raise ValueError("Require aligned hidden width and complete TP1/2/4 vocabulary shards")
    local_rows = vocabulary // tp

    @pl.jit
    def embedding(
        token_ids: pl.Tensor[[capacity], pl.INT64],
        live: pl.Tensor[[capacity], pl.INT32],
        weight: pl.Tensor[[local_rows, hidden], pl.BF16],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        vocab_start: pl.Scalar[pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        block_count = active * (hidden // D_TILE)
        for block_worker in pl.spmd(pl.min(block_count, SPMD_WORKER_LIMIT), name_hint='prefill_vocab_embedding'):
            for block in pl.range(block_worker, block_count, SPMD_WORKER_LIMIT):
                token = block // (hidden // D_TILE)
                column = block % (hidden // D_TILE) * D_TILE
                global_id = pl.cast(pl.read(token_ids, [token]), pl.INT32)
                local_id = global_id - vocab_start
                tile = pl.full([1, D_TILE], dtype=pl.BF16, value=0.0)
                if pl.read(live, [token]) > 0:
                    if local_id >= 0:
                        if local_id < local_rows:
                            row = pl.cast(local_id, pl.INDEX)
                            tile = weight[row : row + 1, column : column + D_TILE]
                output[token : token + 1, column : column + D_TILE] = tile
        return output

    return embedding


def make_initial_hc(capacity, hidden, hc_mult):
    """Shard tokens, replicate BF16 hidden lanes and set the initial one-hot pre-mix."""
    if capacity < 1 or hidden % D_TILE or hc_mult < 1:
        raise ValueError("Require aligned hidden width and positive HC width")

    @pl.jit
    def initial_hc(
        embedding: pl.Tensor[[capacity, hidden], pl.BF16],
        streams: pl.InOut[pl.Tensor[[capacity, hc_mult, hidden], pl.BF16]],
        premix: pl.InOut[pl.Tensor[[capacity, hc_mult], pl.FP32]],
        tp_rank: pl.Scalar[pl.INT32],
        physical: pl.Scalar[pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        flattened = pl.reshape(streams, [capacity, hc_mult * hidden])
        block_count = physical * hc_mult * (hidden // D_TILE)
        for block_worker in pl.spmd(pl.min(block_count, SPMD_WORKER_LIMIT), name_hint='prefill_initial_hc'):
            for block in pl.range(block_worker, block_count, SPMD_WORKER_LIMIT):
                column = block % (hidden // D_TILE) * D_TILE
                lane = block // (hidden // D_TILE) % hc_mult
                token = block // (hidden // D_TILE) // hc_mult
                source = tp_rank * physical + token
                tile = pl.full([1, D_TILE], dtype=pl.BF16, value=0.0)
                if source < active:
                    tile = embedding[source : source + 1, column : column + D_TILE]
                offset = lane * hidden + column
                flattened[token : token + 1, offset : offset + D_TILE] = tile
        for instance in pl.spmd(1, name_hint="prefill_initial_premix"):
            for token in pl.range(physical):
                for lane in pl.unroll(hc_mult):
                    value = pl.cast(0.0, pl.FP32)
                    if lane == 0:
                        value = pl.cast(1.0, pl.FP32)
                    pl.write(premix, [token, lane], value)
        return streams, premix

    return initial_hc


def make_vocab_reduce(capacity, hidden, tp):
    """Sum the gathered vocabulary owners into complete token embeddings."""
    if capacity < 1 or hidden % D_TILE or tp not in (1, 2, 4):
        raise ValueError("Require aligned hidden width and TP1/2/4")

    @pl.jit
    def reduce_embedding(
        gathered: pl.Tensor[[tp * capacity, hidden], pl.BF16],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        block_count = active * (hidden // D_TILE)
        for block_worker in pl.spmd(pl.min(block_count, SPMD_WORKER_LIMIT), name_hint='prefill_vocab_reduce'):
            for block in pl.range(block_worker, block_count, SPMD_WORKER_LIMIT):
                token = block // (hidden // D_TILE)
                column = block % (hidden // D_TILE) * D_TILE
                total = pl.full([1, D_TILE], dtype=pl.FP32, value=0.0)
                for peer in pl.unroll(tp):
                    row = peer * capacity + token
                    shard = gathered[row : row + 1, column : column + D_TILE]
                    total = pl.add(total, pl.cast(shard, pl.FP32))
                output[token : token + 1, column : column + D_TILE] = pl.cast(total, pl.BF16, mode="rint")
        return output

    return reduce_embedding


def make_input_entries(capacity, config, tp):
    """Build vocabulary-parallel lookup and initial local HC streams."""
    hidden, vocabulary = config["hidden_size"], config["vocab_size"]
    return dict(
        lookup=make_vocab_embedding(capacity, hidden, vocabulary, tp),
        reduce=make_vocab_reduce(capacity, hidden, tp),
        initial_hc=make_initial_hc(capacity, hidden, config["hc_mult"]),
    )


def golden_vocab_embedding(tensors):
    """Compute owned rows independently and preserve the inactive output suffix."""
    output, weight = tensors["output"], tensors["weight"]
    for token in range(int(tensors["active"])):
        output[token].zero_()
        row = int(tensors["token_ids"][token]) - int(tensors["vocab_start"])
        if int(tensors["live"][token]) > 0 and 0 <= row < weight.shape[0]:
            output[token].copy_(weight[row])


def golden_initial_hc(tensors):
    """Compute the native token shard and initial HC identity mix independently."""
    streams, premix = tensors["streams"], tensors["premix"]
    physical, active = int(tensors["physical"]), int(tensors["active"])
    for token in range(physical):
        streams[token].zero_()
        source = int(tensors["tp_rank"]) * physical + token
        if source < active:
            for lane in range(streams.shape[1]):
                streams[token, lane].copy_(tensors["embedding"][source])
        premix[token].zero_()
        premix[token, 0] = 1.0


def golden_vocab_reduce(tensors):
    """Sum vocabulary ownership independently and preserve the inactive suffix."""
    output = tensors["output"]
    capacity, hidden = output.shape
    tp = len(tensors["gathered"]) // capacity
    active = int(tensors["active"])
    values = tensors["gathered"].reshape(tp, capacity, hidden)[:, :active].float().sum(dim=0)
    output[:active].copy_(values.to(output.dtype))


def embedding_golden_cases(checkpoint=None):
    """Cover shard edges, masks, empty work and SP padding with exact outputs."""
    from pathlib import Path
    import json
    import torch
    from golden import ScalarSpec, TensorSpec

    capacity, hidden, hc_mult, vocabulary = 16, 5120, 4, 128
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        config = json.loads((checkpoint / "config.json").read_text())["text_config"]
        hidden, hc_mult, vocabulary = config["hidden_size"], config["hc_mult"], config["vocab_size"]

    def weight_shard(tp, rank):
        rows, start = vocabulary // tp, rank * (vocabulary // tp)
        if checkpoint is None:
            values = torch.arange(rows * hidden).reshape(rows, hidden) % 257 - 128
            return (values.float() / 64.0).to(torch.bfloat16)
        from safetensors import safe_open

        name = "embed.weight"
        index = json.loads((checkpoint / "quant_model_weights.safetensors.index.json").read_text())
        with safe_open(checkpoint / index["weight_map"][name], framework="pt", device="cpu") as source:
            return source.get_slice(name)[start : start + rows].contiguous()

    for tp, rank, active in ((1, 0, 0), (1, 0, 16), (2, 1, 7), (4, 0, 1), (4, 3, 16)):
        rows, start = vocabulary // tp, rank * (vocabulary // tp)
        choices = torch.tensor([start - 1, start, start + rows - 1, start + rows, 0], dtype=torch.int64)
        ids = choices[torch.arange(capacity) % choices.numel()]
        live = torch.ones(capacity, dtype=torch.int32)
        live[6] = 0
        specs = [
            TensorSpec("token_ids", [capacity], torch.int64, init_value=ids),
            TensorSpec("live", [capacity], torch.int32, init_value=live),
            TensorSpec(
                "weight", [rows, hidden], torch.bfloat16, init_value=lambda p=tp, r=rank: weight_shard(p, r)
            ),
            TensorSpec("output", [capacity, hidden], torch.bfloat16, init_value=-12.5),
            ScalarSpec("vocab_start", torch.int32, start, compile_runtime=True),
            ScalarSpec("active", torch.int32, active, compile_runtime=True),
        ]
        yield dict(
            name=f"embedding-tp{tp}-rank{rank}-active{active}",
            fn=make_vocab_embedding(capacity, hidden, vocabulary, tp),
            specs=specs,
            golden_fn=golden_vocab_embedding,
        )

    for tp, active in ((1, 16), (2, 7), (4, 1), (4, 0)):

        def gathered_fixture(p=tp):
            value = torch.zeros(p, capacity, hidden, dtype=torch.bfloat16)
            columns = (torch.arange(hidden) % 257 - 128).float() / 64.0
            for token in range(capacity):
                value[token % p, token].copy_(columns.to(torch.bfloat16))
            return value.reshape(p * capacity, hidden)

        specs = [
            TensorSpec("gathered", [tp * capacity, hidden], torch.bfloat16, init_value=gathered_fixture),
            TensorSpec("output", [capacity, hidden], torch.bfloat16, init_value=-12.5),
            ScalarSpec("active", torch.int32, active, compile_runtime=True),
        ]
        yield dict(
            name=f"embedding-reduce-tp{tp}-active{active}",
            fn=make_vocab_reduce(capacity, hidden, tp),
            specs=specs,
            golden_fn=golden_vocab_reduce,
        )

    def embedding_fixture():
        return (torch.arange(capacity * hidden).reshape(capacity, hidden) % 31).to(torch.bfloat16)

    for tp, rank, active in ((1, 0, 16), (2, 1, 7), (4, 3, 1), (4, 0, 0)):
        physical = max(1, (active + tp - 1) // tp)
        specs = [
            TensorSpec("embedding", [capacity, hidden], torch.bfloat16, init_value=embedding_fixture),
            TensorSpec("streams", [capacity, hc_mult, hidden], torch.bfloat16, init_value=-12.5),
            TensorSpec("premix", [capacity, hc_mult], torch.float32, init_value=-17.0),
            ScalarSpec("tp_rank", torch.int32, rank, compile_runtime=True),
            ScalarSpec("physical", torch.int32, physical, compile_runtime=True),
            ScalarSpec("active", torch.int32, active, compile_runtime=True),
        ]
        yield dict(
            name=f"initial-hc-tp{tp}-rank{rank}-active{active}",
            fn=make_initial_hc(capacity, hidden, hc_mult),
            specs=specs,
            golden_fn=golden_initial_hc,
        )


def validate(*, device=0, compile_only=False, checkpoint=None, platform="a2a3", sim=False, output_unpack=False):
    """Run exact embedding and HC initialization boundaries through Golden Harness."""
    from golden import run

    records = []
    cases = sp_unpack_golden_cases() if output_unpack else embedding_golden_cases(checkpoint)
    for case in cases:
        name = case.pop("name")
        result = run(
            **case,
            config={"backend_type": platform, "execution_mode": sim, "device_id": device},
            compile_only=compile_only,
            rtol=0.0,
            atol=0.0,
        )
        records.append(
            dict(name=name, passed=result.passed, work_dir=str(result.work_dir), error=str(result.error))
        )
        if not result.passed:
            raise RuntimeError(f"{name}: {result.error}")
    return records


def make_sp_masked_zero(capacity, width):
    """Zero masked physical SP rows and preserve valid and inactive rows."""

    @pl.jit
    def masked_zero(
        value: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        valid: pl.Scalar[pl.INT32],
        physical: pl.Scalar[pl.INT32],
    ):
        block_count = (physical - valid) * (width // D_TILE)
        for block_worker in pl.spmd(pl.min(block_count, SPMD_WORKER_LIMIT), name_hint='prefill_sp_masked_zero'):
            for block in pl.range(block_worker, block_count, SPMD_WORKER_LIMIT):
                token = valid + block // (width // D_TILE)
                column = block % (width // D_TILE) * D_TILE
                tile = pl.full([1, D_TILE], dtype=pl.BF16, value=0.0)
                value[token : token + 1, column : column + D_TILE] = tile
        return value

    return masked_zero


def make_head_pack(capacity, heads, head_dim, tp):
    """Pack all local CP heads into fixed equal peer buckets."""
    local_width = heads // tp * head_dim

    @pl.jit
    def head_pack(
        value: pl.Tensor[[capacity, heads * head_dim], pl.BF16],
        packed: pl.InOut[pl.Tensor[[tp * capacity, local_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        row_count = tp * capacity
        for row_worker in pl.spmd(pl.min(row_count, SPMD_WORKER_LIMIT), name_hint='prefill_cp_head_pack'):
            for row in pl.range(row_worker, row_count, SPMD_WORKER_LIMIT):
                peer = row // capacity
                token = row % capacity
                if token < active:
                    source = peer * local_width
                    packed[row : row + 1, 0 : local_width] = value[token : token + 1, source : source + local_width]
                else:
                    packed[row : row + 1, 0 : local_width] = pl.full([1, local_width], dtype=pl.BF16, value=0.0)
        return packed

    return head_pack


def make_sp_gather_unpack(capacity, width, tp, dtype=pl.BF16, tile=D_TILE, *, last_row=False):
    """Compact fixed peer buckets into all token rows or the last active row."""
    output_capacity = 16 if last_row else capacity
    scalar_rows = width < 8
    while tile < 4096 and width % (2 * tile) == 0:
        tile *= 2

    @pl.jit
    def gather_unpack(
        packet: pl.Tensor[[tp * capacity, width], dtype],
        output: pl.InOut[pl.Tensor[[output_capacity, width], dtype]],
        physical: pl.Scalar[pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        rows = active
        token_offset = pl.cast(0, pl.INT32)
        if last_row:
            rows = pl.cast(pl.min(active, 1), pl.INT32)
            token_offset = pl.cast(active - 1, pl.INT32)
        if scalar_rows:
            for _block in pl.spmd(1, name_hint="prefill_sp_state_unpack"):
                for token in pl.range(rows):
                    input_token = token + token_offset
                    source = input_token // physical * capacity + input_token % physical
                    for column in pl.range(width):
                        state_value = pl.read(packet, [source, column])
                        pl.write(output, [token, column], state_value)
        else:
            block_count = rows * (width // tile)
            for block_worker in pl.spmd(pl.min(block_count, SPMD_WORKER_LIMIT), name_hint='prefill_sp_gather_unpack'):
                for block in pl.range(block_worker, block_count, SPMD_WORKER_LIMIT):
                    token = block // (width // tile)
                    column = block % (width // tile) * tile
                    input_token = token + token_offset
                    source = input_token // physical * capacity + input_token % physical
                    vector_value = packet[source : source + 1, column : column + tile]
                    output[token : token + 1, column : column + tile] = vector_value
        return output

    return gather_unpack


def make_sp_reduce_pack(capacity, hidden, tp):
    """Pad full Wo-B partial sums into equal reduce-scatter peer buckets."""

    @pl.jit
    def reduce_pack(
        value: pl.Tensor[[capacity, hidden], pl.BF16],
        packed: pl.InOut[pl.Tensor[[tp * capacity, hidden], pl.BF16]],
        physical: pl.Scalar[pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        row_count = tp * capacity
        for row_worker in pl.spmd(pl.min(row_count, SPMD_WORKER_LIMIT), name_hint='prefill_sp_reduce_pack'):
            for row in pl.range(row_worker, row_count, SPMD_WORKER_LIMIT):
                peer = row // capacity
                local = row % capacity
                source = peer * physical + local
                valid = pl.min(physical, pl.max(active - peer * physical, 0))
                if local < valid:
                    packed[row : row + 1, 0 : hidden] = value[source : source + 1, 0 : hidden]
                else:
                    packed[row : row + 1, 0 : hidden] = pl.full([1, hidden], dtype=pl.BF16, value=0.0)
        return packed

    return reduce_pack


def golden_sp_gather_unpack(tensors, *, tp, last_row):
    """Concatenate active SP shards and select the requested output rows."""
    import torch

    active, physical = int(tensors["active"]), int(tensors["physical"])
    if not active:
        return
    packet = tensors["packet"].reshape(tp, -1, tensors["output"].shape[1])
    tokens = torch.cat([shard[:physical] for shard in packet])[:active]
    selected = tokens[-1:] if last_row else tokens
    tensors["output"][:len(selected)].copy_(selected)


def sp_unpack_golden_cases():
    """Cover full and last-row output across empty, partial and complete TP shards."""
    from functools import partial
    import torch
    from golden import ScalarSpec, TensorSpec

    for capacity, tp, active in ((16, 1, 0), (16, 1, 1), (16, 2, 7), (16, 4, 1),
                                 (16, 4, 3), (16, 4, 16), (1024, 4, 1023), (1024, 4, 1024)):
        width = 5120
        physical = max(1, (active + tp - 1) // tp)
        for last_row in (False, True):
            output_rows = 16 if last_row else capacity
            specs = [
                TensorSpec("packet", [tp * capacity, width], torch.bfloat16,
                           init_value=lambda n=tp * capacity: (torch.arange(n * width).reshape(n, width) % 127)),
                TensorSpec("output", [output_rows, width], torch.bfloat16, init_value=-17),
                ScalarSpec("physical", torch.int32, physical, compile_runtime=True),
                ScalarSpec("active", torch.int32, active, compile_runtime=True),
            ]
            yield dict(name=f"sp-unpack-tp{tp}-capacity{capacity}-active{active}-last{last_row}",
                       fn=make_sp_gather_unpack(capacity, width, tp, last_row=last_row), specs=specs,
                       golden_fn=partial(golden_sp_gather_unpack, tp=tp, last_row=last_row))


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--output-unpack", action="store_true", help="Validate all-row and last-row SP output")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    records = validate(
        device=args.device,
        compile_only=args.compile_only,
        checkpoint=args.checkpoint,
        platform=args.platform,
        sim=args.sim,
        output_unpack=args.output_unpack,
    )
    if args.report is not None:
        args.report.write_text(
            json.dumps(dict(compile_only=args.compile_only, cases=records), indent=2) + "\n"
        )

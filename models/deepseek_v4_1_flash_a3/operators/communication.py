# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=16
# ci: no-sim
"""Validate EP16 peer transfers and four independent TP4 output reductions on A3."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "models.deepseek_v4_1_flash_a3.operators"

import argparse
import json

import torch
import pypto.language as pl
import pypto.language.distributed as pld
from golden import TensorSpec, ratio_allclose, run
from pypto.ir import DistributedConfig

from .config import FLASH as M, EP_SIZE, TP_SIZE, PREFILL_MAX_TOKENS
from .o_proj import prefill_tp_output_all_reduce, retire_prefill_signals


D = M.hidden_size
ROWS = 16
ROUNDS = 3
TP_ROUNDS = 4


@pl.jit(auto_scope=False)
def probe_rank(
    marks: pl.Tensor[[ROUNDS, 16], pl.INT32],
    partial: pl.Tensor[[ROWS, D], pl.BF16],
    poison: pl.Tensor[[ROWS, D], pl.BF16],
    received: pl.Out[pl.Tensor[[ROUNDS, EP_SIZE, 16], pl.INT32]],
    reduced: pl.Out[pl.Tensor[[TP_ROUNDS, ROWS, D], pl.BF16]],
    payload: pld.DistributedTensor[[EP_SIZE, 16], pl.INT32],
    ready: pld.DistributedTensor[[EP_SIZE, 16], pl.INT32],
    consumed: pld.DistributedTensor[[EP_SIZE, 16], pl.INT32],
    tp_payload: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    tp_ready: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
):
    finished = pl.array.create(1, pl.TASK_ID)
    finished[0] = pl.system.task_dummy(deps=[])
    for epoch in pl.range(1, ROUNDS + 1):
        with pl.spmd(EP_SIZE, name_hint="probe_ep_publish", deps=[finished[0]]) as pushed:
            send_peer = pl.tile.get_block_idx()
            if pl.read(marks, [epoch - 1, 0]) > 0:
                pld.tensor.put(dst=payload, peer=send_peer, src=marks,
                               dst_offsets=[rank, 0], src_offsets=[epoch - 1, 0], shape=[1, 16])
            pld.system.notify(ready, peer=send_peer, offsets=[rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="probe_ep_wait", deps=[pushed]) as arrived:
            for ready_peer in pl.range(EP_SIZE):
                pld.system.defer_wait(ready, offsets=[ready_peer, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
        with pl.spmd(EP_SIZE, name_hint="probe_ep_read", deps=[arrived]) as read:
            receive_peer = pl.tile.get_block_idx()
            value = pl.load(payload, [receive_peer, 0], [1, 16])
            pl.store(pl.reshape(value, [1, 1, 16]), [epoch - 1, receive_peer, 0], received)
            pld.system.notify(consumed, peer=receive_peer, offsets=[rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="probe_ep_release", deps=[read]) as released:
            for consumed_peer in pl.range(EP_SIZE):
                pld.system.defer_wait(consumed, offsets=[consumed_peer, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
        finished[0] = released
    result = pl.create_tensor([ROWS, D], dtype=pl.BF16)
    for epoch in pl.range(1, TP_ROUNDS + 1):
        with pl.spmd(ROWS, name_hint="probe_tp_zero", deps=[finished[0]]):
            row = pl.tile.get_block_idx()
            seeded = pl.load(poison, [row, 0], [1, D])
            pl.store(seeded, [row, 0], result)
        active = pl.cast(0, pl.INT32)
        if rank // TP_SIZE == (epoch - 1) % 2 * 3:
            active = pl.cast(ROWS, pl.INT32)
            if epoch > 2:
                active = pl.cast(1, pl.INT32)
        prefill_tp_output_all_reduce(partial, tp_payload, tp_ready, result,
                                    rank // TP_SIZE * TP_SIZE, rank % TP_SIZE, active, epoch, owner_rows=False)
        with pl.spmd(ROWS, name_hint="probe_tp_save") as saved:
            saved_row = pl.tile.get_block_idx()
            value = pl.load(result, [saved_row, 0], [1, D])
            pl.store(pl.reshape(value, [1, 1, D]), [epoch - 1, saved_row, 0], reduced)
        finished[0] = saved
    retire_prefill_signals(reduced, tp_ready, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE)
    return received, reduced


@pl.jit.host
def probe(
    marks: pl.Tensor[[EP_SIZE, ROUNDS, 16], pl.INT32],
    partial: pl.Tensor[[EP_SIZE, ROWS, D], pl.BF16],
    poison: pl.Tensor[[EP_SIZE, ROWS, D], pl.BF16],
    received: pl.Out[pl.Tensor[[EP_SIZE, ROUNDS, EP_SIZE, 16], pl.INT32]],
    reduced: pl.Out[pl.Tensor[[EP_SIZE, TP_ROUNDS, ROWS, D], pl.BF16]],
):
    payload_buffer = pld.alloc_window_buffer([EP_SIZE, 16], dtype=pl.INT32)
    ready_buffer = pld.alloc_window_buffer([EP_SIZE, 16], dtype=pl.INT32)
    consumed_buffer = pld.alloc_window_buffer([EP_SIZE, 16], dtype=pl.INT32)
    tp_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    tp_ready_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(EP_SIZE):
        payload = pld.window(payload_buffer, [EP_SIZE, 16], dtype=pl.INT32)
        ready = pld.window(ready_buffer, [EP_SIZE, 16], dtype=pl.INT32)
        consumed = pld.window(consumed_buffer, [EP_SIZE, 16], dtype=pl.INT32)
        tp_payload = pld.window(tp_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        tp_ready = pld.window(tp_ready_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        probe_rank(marks[rank], partial[rank], poison[rank], received[rank], reduced[rank],
                   payload, ready, consumed, tp_payload, tp_ready, rank, device=rank)


def golden_probe(tensors):
    retained = tensors["marks"][:, 0].clone()
    for epoch in range(ROUNDS):
        active = tensors["marks"][:, epoch, 0] > 0
        retained[active] = tensors["marks"][active, epoch]
        tensors["received"][:, epoch] = retained.unsqueeze(0)
    tensors["reduced"].zero_()
    for epoch, group in enumerate((0, 3, 0, 3)):
        start = group * TP_SIZE
        value = tensors["partial"][start:start + TP_SIZE].float().sum(0).bfloat16()
        rows = ROWS if epoch < 2 else 1
        tensors["reduced"][start:start + TP_SIZE, epoch, :rows] = value[:rows]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3"], default="a2a3")
    parser.add_argument("-d", "--devices", default=",".join(map(str, range(16))))
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--enable-dep-gen", action="store_true")
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    devices = [int(value) for value in args.devices.split(",")]
    if len(devices) != 16 or len(set(devices)) != 16:
        parser.error("exactly sixteen distinct allocated devices are required")
    marks = torch.arange(16 * ROUNDS * 16, dtype=torch.int32).reshape(16, ROUNDS, 16) + 1
    marks[1:, 1, 0] = 0
    partial = torch.arange(1, 17).bfloat16().reshape(16, 1, 1).expand(16, ROWS, D).contiguous()
    specs = [
        TensorSpec("marks", list(marks.shape), torch.int32, init_value=marks),
        TensorSpec("partial", list(partial.shape), torch.bfloat16, init_value=partial),
        TensorSpec("poison", list(partial.shape), torch.bfloat16, init_value=torch.full_like(partial, torch.nan)),
        TensorSpec("received", [16, ROUNDS, 16, 16], torch.int32),
        TensorSpec("reduced", [16, TP_ROUNDS, ROWS, D], torch.bfloat16),
    ]
    result = run(
        fn=probe,
        specs=specs,
        golden_fn=golden_probe,
        compare_fn={name: ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
                    for name in ("received", "reduced")},
        config={"backend_type": args.platform, "execution_mode": False,
                "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0),
                "enable_dep_gen": args.enable_dep_gen},
        runtime_dir=args.runtime_dir,
        compile_only=args.compile_only,
    )
    if args.report:
        report = {"passed": result.passed, "compile_only": args.compile_only, "devices": devices,
                  "ep_size": EP_SIZE, "tp_size": TP_SIZE, "rounds": ROUNDS, "active_dp_groups": [0, 3],
                  "tp_active_rows": [ROWS, ROWS, 1, 1], "nan_padding_seed": True,
                  "build": str(result.work_dir), "error": str(result.error) if result.error else None}
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    if not result.passed:
        raise SystemExit(result.error or 1)

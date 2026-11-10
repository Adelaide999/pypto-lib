# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Restore dispatch order, reduce FP32 weighted routes, and add the shared result."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

import pypto.language as pl

TOPK = 6
D = 5120


def make_moe_combine(capacity, hidden=D, topk=TOPK, world=2):
    """Restore returned token/top-k rows, weight experts, and add the shared result."""
    routes = capacity * topk
    packet_rows = world * routes

    @pl.jit
    def moe_combine(
        packet_routes: pl.Tensor[[packet_rows, 4], pl.INT64],
        packet_hidden: pl.Tensor[[packet_rows, hidden], pl.BF16],
        packet_counts: pl.Tensor[[world, 8], pl.INT32],
        route_weights: pl.Tensor[[capacity, topk], pl.FP32],
        shared: pl.Tensor[[capacity, hidden], pl.BF16],
        routed: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        returned_routes: pl.InOut[pl.Tensor[[routes, 4], pl.INT64]],
        active: pl.Scalar[pl.INT32],
    ):
        route_map = pl.create_tensor([routes, 8], dtype=pl.INT32)
        for _block in pl.spmd(1, name_hint="moe_ep_restore_metadata"):
            for original in pl.range(routes):
                pl.write(route_map, [original, 0], pl.cast(-1, pl.INT32))
            for peer in pl.range(world):
                count = pl.cast(pl.read(packet_counts, [peer, 0]), pl.INDEX)
                for row in pl.range(count):
                    metadata_packet = peer * routes + row
                    source_token = pl.cast(pl.read(packet_routes, [metadata_packet, 1]), pl.INDEX)
                    source_slot = pl.cast(pl.read(packet_routes, [metadata_packet, 2]), pl.INDEX)
                    original = source_token * topk + source_slot
                    pl.write(route_map, [original, 0], pl.cast(metadata_packet, pl.INT32))
                    for field in pl.range(4):
                        field_value = pl.read(packet_routes, [metadata_packet, field])
                        pl.write(returned_routes, [original, field], field_value)

        for token in pl.spmd(active, name_hint="moe_ep_source_combine"):
            acc = pl.full([1, hidden], dtype=pl.FP32, value=0.0)
            for slot in pl.range(topk):
                route_row = token * topk + slot
                packet = pl.cast(pl.read(route_map, [route_row, 0]), pl.INDEX)
                if packet >= 0:
                    value = pl.cast(packet_hidden[packet:packet + 1, :], pl.FP32)
                    weight = pl.read(route_weights, [token, slot])
                    acc = pl.add(acc, pl.mul(value, weight))
            narrowed = pl.cast(acc, pl.BF16, mode="rint")
            routed[token:token + 1, :] = narrowed
            shared_row = pl.cast(shared[token:token + 1, :], pl.FP32)
            total = pl.add(pl.cast(narrowed, pl.FP32), shared_row)
            output[token:token + 1, :] = pl.cast(total, pl.BF16, mode="rint")
        return routed, output, returned_routes

    return moe_combine


def golden_moe_combine(tensors):
    """Evaluate the same ordered FP32 route reduction for standalone validation."""
    import torch

    active = int(tensors["active"])
    topk = tensors["route_weights"].shape[1]
    routes = tensors["route_weights"].shape[0] * topk
    mapping = {}
    for peer in range(tensors["packet_counts"].shape[0]):
        for row in range(int(tensors["packet_counts"][peer, 0])):
            packet = peer * routes + row
            token, slot = (int(tensors["packet_routes"][packet, field]) for field in (1, 2))
            original = token * topk + slot
            mapping[original] = packet
            tensors["returned_routes"][original] = tensors["packet_routes"][packet]
    for token in range(active):
        acc = torch.zeros_like(tensors["shared"][token], dtype=torch.float32)
        for slot in range(topk):
            row = token * topk + slot
            if row in mapping:
                value = tensors["packet_hidden"][mapping[row]].float()
                acc += value * tensors["route_weights"][token, slot]
        routed = acc.to(torch.bfloat16)
        tensors["routed"][token] = routed
        tensors["output"][token] = (routed.float() + tensors["shared"][token].float()).to(torch.bfloat16)


def validate_moe_combine(*, device=0, compile_only=False, platform="a2a3", sim=False):
    """Validate ordered route weighting and inactive sentinel preservation."""
    import torch
    from golden import ScalarSpec, TensorSpec, run

    capacity, hidden, topk, world = 16, 256, 6, 2
    for active, valid in ((0, 0), (1, 0), (1, 1), (7, 1), (7, 7), (16, 7), (16, 16)):
        routes = capacity * topk
        metadata = torch.full((world * routes, 4), -1, dtype=torch.int64)
        counts = torch.zeros((world, 8), dtype=torch.int32)
        for token in range(valid):
            for slot in range(topk):
                peer = slot % world
                row = int(counts[peer, 0])
                metadata[peer * routes + row] = torch.tensor([0, token, slot, peer * 192 + slot])
                counts[peer, 0] += 1
        specs = [
            TensorSpec("packet_routes", [world * routes, 4], torch.int64, init_value=metadata),
            TensorSpec("packet_hidden", [world * routes, hidden], torch.bfloat16, init_value=torch.randn),
            TensorSpec("packet_counts", [world, 8], torch.int32, init_value=counts),
            TensorSpec("route_weights", [capacity, topk], torch.float32, init_value=torch.rand),
            TensorSpec("shared", [capacity, hidden], torch.bfloat16, init_value=torch.randn),
            TensorSpec("routed", [capacity, hidden], torch.bfloat16, init_value=-12.5),
            TensorSpec("output", [capacity, hidden], torch.bfloat16, init_value=-12.5),
            TensorSpec("returned_routes", [routes, 4], torch.int64, init_value=-1),
            ScalarSpec("active", torch.int32, active),
        ]
        result = run(
                     compile_only=compile_only,
            fn=make_moe_combine(capacity, hidden, topk),
            specs=specs,
            golden_fn=golden_moe_combine,
            config={"backend_type": platform, "execution_mode": sim, "device_id": device},
            rtol=0.05,
            atol=0.05,
        )
        if not result.passed:
            raise RuntimeError(f"Combine validation failed: active={active}, valid={valid}: {result.error}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    validate_moe_combine(device=args.device, compile_only=args.compile_only, platform=args.platform, sim=args.sim)

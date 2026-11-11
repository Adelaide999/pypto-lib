# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pack EP route payloads, unpack owned expert slabs, and stage returned results."""

import pypto.language as pl


# model config
EXPERTS = 384
TOPK = 6
D = 5120
RECV_MAX = 64

# tiling
D_TILE = 512
MAP_PAD = 8


def validate_received_routes(packet, rank, layout, routes, valid_by_rank=None):
    """Check transported source, expert ownership and bounded bucket metadata."""
    counts = [int(packet["counts"][peer, 0]) for peer in range(layout.ep)]
    if any(not 0 <= count <= routes for count in counts):
        raise ValueError("HCCL received bucket count exceeds its fixed capacity")
    for peer, count in enumerate(counts):
        for row in range(count):
            source, token, slot, expert = packet["routes"][peer * routes + row].tolist()
            if source != peer or layout.owner(expert) != rank or token < 0 or slot < 0:
                raise RuntimeError("HCCL delivered inconsistent route metadata")
            if valid_by_rank is not None and token >= valid_by_rank[peer]:
                raise RuntimeError("HCCL dispatched a masked source row")
    return sum(counts)


def validate_returned_routes(packet, rank, routes, route_ids, valid):
    """Require one returned result for every valid source token and top-k slot."""
    seen = set()
    topk = route_ids.shape[1]
    for peer in range(packet["counts"].shape[0]):
        count = int(packet["counts"][peer, 0])
        if not 0 <= count <= routes:
            raise ValueError("Returned bucket count exceeds capacity")
        for row in range(count):
            source, token, slot, expert = packet["routes"][peer * routes + row].tolist()
            if source != rank or not 0 <= token < valid or not 0 <= slot < topk:
                raise RuntimeError("Returned source token or top-k slot is invalid")
            if (token, slot) in seen or expert != int(route_ids[token, slot]):
                raise RuntimeError("Duplicate or inconsistent returned expert route")
            seen.add((token, slot))
    if len(seen) != valid * topk:
        raise RuntimeError("HCCL did not return every valid top-k route")


def make_ep_dispatch_pack(capacity, hidden, experts, topk, world):
    """Pack source token/top-k rows into contiguous destination buckets."""
    routes = capacity * topk
    local_experts = experts // world
    packet_rows = world * routes

    @pl.jit
    def ep_dispatch_pack(
        x: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        route_ids: pl.Tensor[[capacity, topk], pl.INT32],
        sent_routes: pl.InOut[pl.Tensor[[packet_rows, 4], pl.INT64]],
        sent_hidden: pl.InOut[pl.Tensor[[packet_rows, hidden], pl.BF16]],
        sent_counts: pl.InOut[pl.Tensor[[world, MAP_PAD], pl.INT32]],
        active: pl.Scalar[pl.INT32],
        source_rank: pl.Scalar[pl.INT32],
    ):
        route_map = pl.create_tensor([routes, MAP_PAD], dtype=pl.INT32)
        for _block in pl.spmd(1, name_hint="moe_ep_pack_metadata"):
            for peer in pl.range(world):
                bucket_count = pl.cast(0, pl.INDEX)
                for token in pl.range(active):
                    for slot in pl.range(topk):
                        expert = pl.read(route_ids, [token, slot])
                        if expert // local_experts == peer:
                            packet_index = peer * routes + bucket_count
                            pl.write(sent_routes, [packet_index, 0], pl.cast(source_rank, pl.INT64))
                            pl.write(sent_routes, [packet_index, 1], pl.cast(token, pl.INT64))
                            pl.write(sent_routes, [packet_index, 2], pl.cast(slot, pl.INT64))
                            pl.write(sent_routes, [packet_index, 3], pl.cast(expert, pl.INT64))
                            original_route = token * topk + slot
                            pl.write(route_map, [original_route, 0], pl.cast(packet_index, pl.INT32))
                            bucket_count = bucket_count + 1
                pl.write(sent_counts, [peer, 0], pl.cast(bucket_count, pl.INT32))

        for block in pl.spmd(active * topk * (hidden // D_TILE), name_hint="moe_ep_pack_hidden"):
            route_index = block // (hidden // D_TILE)
            column = block % (hidden // D_TILE) * D_TILE
            source_token = route_index // topk
            target_index = pl.cast(pl.read(route_map, [route_index, 0]), pl.INDEX)
            source_row = x[source_token:source_token + 1, column:column + D_TILE]
            sent_hidden[target_index:target_index + 1, column:column + D_TILE] = source_row
        return x, sent_routes, sent_hidden, sent_counts

    return ep_dispatch_pack


def make_ep_dispatch_unpack(capacity, hidden, experts, topk, world, expert_capacity=None):
    """Unpack received buckets into owned expert slabs and compact audit payloads."""
    routes = capacity * topk
    local_experts = experts // world
    packet_rows = world * routes
    if expert_capacity is None:
        expert_capacity = ((capacity * world + 15) // 16) * 16
    slab_rows = local_experts * expert_capacity

    @pl.jit
    def ep_dispatch_unpack(
        packet_routes: pl.Tensor[[packet_rows, 4], pl.INT64],
        packet_hidden: pl.InOut[pl.Tensor[[packet_rows, hidden], pl.BF16]],
        packet_counts: pl.Tensor[[world, MAP_PAD], pl.INT32],
        recv_routes: pl.InOut[pl.Tensor[[packet_rows, 4], pl.INT64]],
        recv_hidden: pl.InOut[pl.Tensor[[packet_rows, hidden], pl.BF16]],
        expert_x: pl.InOut[pl.Tensor[[slab_rows, hidden], pl.BF16]],
        expert_counts: pl.InOut[pl.Tensor[[local_experts], pl.INT64]],
        packet_map: pl.InOut[pl.Tensor[[packet_rows, MAP_PAD], pl.INT32]],
        received_total: pl.InOut[pl.Tensor[[MAP_PAD], pl.INT32]],
        owner_rank: pl.Scalar[pl.INT32],
    ):
        for _block in pl.spmd(1, name_hint="moe_ep_unpack_metadata"):
            compact_begin = pl.cast(0, pl.INDEX)
            for local in pl.range(local_experts):
                expert = owner_rank * local_experts + local
                owned_count = pl.cast(0, pl.INDEX)
                for peer in pl.range(world):
                    peer_count = pl.cast(pl.read(packet_counts, [peer, 0]), pl.INDEX)
                    for row in pl.range(peer_count):
                        received_packet = peer * routes + row
                        global_expert = pl.read(packet_routes, [received_packet, 3])
                        if global_expert == expert:
                            compact_row = compact_begin + owned_count
                            slab_row = local * expert_capacity + owned_count
                            pl.write(packet_map, [received_packet, 0], pl.cast(slab_row, pl.INT32))
                            pl.write(packet_map, [received_packet, 1], pl.cast(compact_row, pl.INT32))
                            for field in pl.range(4):
                                field_value = pl.read(packet_routes, [received_packet, field])
                                pl.write(recv_routes, [compact_row, field], field_value)
                            owned_count = owned_count + 1
                pl.write(expert_counts, [local], pl.cast(owned_count, pl.INT64))
                compact_begin = compact_begin + owned_count
            pl.write(received_total, [0], pl.cast(compact_begin, pl.INT32))

        for block in pl.spmd(packet_rows * (hidden // D_TILE), name_hint="moe_ep_unpack_hidden"):
            packet = block // (hidden // D_TILE)
            column = block % (hidden // D_TILE) * D_TILE
            source_peer = packet // routes
            bucket_rows = pl.read(packet_counts, [source_peer, 0])
            if packet % routes < bucket_rows:
                target = pl.cast(pl.read(packet_map, [packet, 0]), pl.INDEX)
                compact = pl.cast(pl.read(packet_map, [packet, 1]), pl.INDEX)
                value = packet_hidden[packet:packet + 1, column:column + D_TILE]
                expert_x[target:target + 1, column:column + D_TILE] = value
                recv_hidden[compact:compact + 1, column:column + D_TILE] = value
        return packet_hidden, recv_routes, recv_hidden, expert_x, expert_counts, packet_map, received_total

    return ep_dispatch_unpack


def make_ep_return_pack(capacity, hidden, experts, topk, world, expert_capacity=None):
    """Gather owned expert results into each original source's return bucket."""
    routes = capacity * topk
    packet_rows = world * routes
    if expert_capacity is None:
        expert_capacity = ((capacity * world + 15) // 16) * 16
    slab_rows = (experts // world) * expert_capacity

    @pl.jit
    def ep_return_pack(
        expert_y: pl.Tensor[[slab_rows, hidden], pl.BF16],
        packet_counts: pl.Tensor[[world, MAP_PAD], pl.INT32],
        packet_map: pl.Tensor[[packet_rows, MAP_PAD], pl.INT32],
        sent_hidden: pl.InOut[pl.Tensor[[packet_rows, hidden], pl.BF16]],
    ):
        for block in pl.spmd(packet_rows * (hidden // D_TILE), name_hint="moe_ep_return_pack"):
            packet = block // (hidden // D_TILE)
            column = block % (hidden // D_TILE) * D_TILE
            count = pl.read(packet_counts, [packet // routes, 0])
            if packet % routes < count:
                source = pl.cast(pl.read(packet_map, [packet, 0]), pl.INDEX)
                expert_row = expert_y[source:source + 1, column:column + D_TILE]
                sent_hidden[packet:packet + 1, column:column + D_TILE] = expert_row
        return sent_hidden

    return ep_return_pack


def golden_ep_dispatch_pack(tensors):
    """Pack destination buckets for standalone Golden Harness validation."""
    world = tensors["sent_counts"].shape[0]
    capacity, topk = tensors["route_ids"].shape
    local_experts = 384 // world
    routes = capacity * topk
    source = int(tensors["source_rank"])
    for peer in range(world):
        count = 0
        for token in range(int(tensors["active"])):
            for slot in range(topk):
                expert = int(tensors["route_ids"][token, slot])
                if expert // local_experts == peer:
                    target = peer * routes + count
                    for field, value in enumerate((source, token, slot, expert)):
                        tensors["sent_routes"][target, field] = value
                    tensors["sent_hidden"][target] = tensors["x"][token]
                    count += 1
        tensors["sent_counts"][peer, 0] = count


def golden_ep_dispatch_unpack(tensors):
    """Unpack received model rows for standalone Golden Harness validation."""
    world = tensors["packet_counts"].shape[0]
    packet_rows = tensors["packet_routes"].shape[0]
    routes = packet_rows // world
    local_experts = tensors["expert_counts"].shape[0]
    expert_capacity = tensors["expert_x"].shape[0] // local_experts
    begin = 0
    for local in range(local_experts):
        expert = int(tensors["owner_rank"]) * local_experts + local
        count = 0
        for peer in range(world):
            for row in range(int(tensors["packet_counts"][peer, 0])):
                packet = peer * routes + row
                if int(tensors["packet_routes"][packet, 3]) == expert:
                    target, compact = local * expert_capacity + count, begin + count
                    tensors["packet_map"][packet, 0] = target
                    tensors["packet_map"][packet, 1] = compact
                    tensors["recv_routes"][compact] = tensors["packet_routes"][packet]
                    tensors["recv_hidden"][compact] = tensors["packet_hidden"][packet]
                    tensors["expert_x"][target] = tensors["packet_hidden"][packet]
                    count += 1
        tensors["expert_counts"][local] = count
        begin += count
    tensors["received_total"][0] = begin


def golden_ep_return_pack(tensors):
    """Gather expert slabs for standalone Golden Harness validation."""
    world = tensors["packet_counts"].shape[0]
    routes = tensors["sent_hidden"].shape[0] // world
    for peer in range(world):
        for row in range(int(tensors["packet_counts"][peer, 0])):
            packet = peer * routes + row
            source = int(tensors["packet_map"][packet, 0])
            tensors["sent_hidden"][packet] = tensors["expert_y"][source]


def validate_ep_dispatch(*, device=0):
    """Validate packing, received expert layout, empty buckets, and return gathering."""
    import torch
    from golden import ScalarSpec, TensorSpec, run

    capacity, hidden, experts, topk = 16, 512, 384, 6
    for world in (2, 16):
        routes, local_experts = capacity * topk, experts // world
        packet_rows = world * routes
        expert_capacity = ((capacity * world + 15) // 16) * 16
        slab_rows = local_experts * expert_capacity
        for active in (0, 1, 7, 16):
            ids = torch.tensor([0, 1, 2, experts - 3, experts - 2, experts - 1], dtype=torch.int32)
            ids = ids.expand(capacity, -1).clone()
            pack_specs = [
                TensorSpec("x", [capacity, hidden], torch.bfloat16, init_value=torch.randn),
                TensorSpec("route_ids", [capacity, topk], torch.int32, init_value=ids),
                TensorSpec("sent_routes", [packet_rows, 4], torch.int64, init_value=-1),
                TensorSpec("sent_hidden", [packet_rows, hidden], torch.bfloat16, init_value=0.0),
                TensorSpec("sent_counts", [world, MAP_PAD], torch.int32, init_value=0),
                ScalarSpec("active", torch.int32, active),
                ScalarSpec("source_rank", torch.int32, 0),
            ]
            result = run(
                fn=make_ep_dispatch_pack(capacity, hidden, experts, topk, world),
                specs=pack_specs,
                golden_fn=golden_ep_dispatch_pack,
                config={"platform": "a2a3", "device_id": device},
                rtol=0.0,
                atol=0.0,
            )
            if not result.passed:
                raise RuntimeError(f"EP pack validation failed: EP{world}, active={active}: {result.error}")
            metadata = torch.full((packet_rows, 4), -1, dtype=torch.int64)
            counts = torch.zeros((world, MAP_PAD), dtype=torch.int32)
            for peer in range(world):
                counts[peer, 0] = active
                for token in range(active):
                    metadata[peer * routes + token] = torch.tensor([peer, token, 0, token % local_experts])
            payload = torch.randn(packet_rows, hidden, dtype=torch.bfloat16)
            unpack_specs = [
                TensorSpec("packet_routes", [packet_rows, 4], torch.int64, init_value=metadata),
                TensorSpec("packet_hidden", [packet_rows, hidden], torch.bfloat16, init_value=payload),
                TensorSpec("packet_counts", [world, MAP_PAD], torch.int32, init_value=counts),
                TensorSpec("recv_routes", [packet_rows, 4], torch.int64, init_value=-1),
                TensorSpec("recv_hidden", [packet_rows, hidden], torch.bfloat16, init_value=0.0),
                TensorSpec("expert_x", [slab_rows, hidden], torch.bfloat16, init_value=0.0),
                TensorSpec("expert_counts", [local_experts], torch.int64, init_value=0),
                TensorSpec("packet_map", [packet_rows, MAP_PAD], torch.int32, init_value=-1),
                TensorSpec("received_total", [MAP_PAD], torch.int32, init_value=0),
                ScalarSpec("owner_rank", torch.int32, 0),
            ]
            result = run(
                fn=make_ep_dispatch_unpack(capacity, hidden, experts, topk, world),
                specs=unpack_specs,
                golden_fn=golden_ep_dispatch_unpack,
                config={"platform": "a2a3", "device_id": device},
                rtol=0.0,
                atol=0.0,
            )
            if not result.passed:
                raise RuntimeError(f"EP unpack validation failed: EP{world}, active={active}: {result.error}")
            mapping = torch.full((packet_rows, MAP_PAD), -1, dtype=torch.int32)
            for peer in range(world):
                for token in range(active):
                    target = token % local_experts * expert_capacity + peer
                    mapping[peer * routes + token, 0] = target
            return_specs = [
                TensorSpec("expert_y", [slab_rows, hidden], torch.bfloat16, init_value=torch.randn),
                TensorSpec("packet_counts", [world, MAP_PAD], torch.int32, init_value=counts),
                TensorSpec("packet_map", [packet_rows, MAP_PAD], torch.int32, init_value=mapping),
                TensorSpec("sent_hidden", [packet_rows, hidden], torch.bfloat16, init_value=0.0),
            ]
            result = run(
                fn=make_ep_return_pack(capacity, hidden, experts, topk, world),
                specs=return_specs,
                golden_fn=golden_ep_return_pack,
                config={"platform": "a2a3", "device_id": device},
                rtol=0.0,
                atol=0.0,
            )
            if not result.passed:
                raise RuntimeError(f"EP return validation failed: EP{world}, active={active}: {result.error}")


def make_dispatch_layout(tokens):
    """Map native count-mode dispatch into the full384 expert ABI."""
    routes = tokens * TOPK

    @pl.jit
    def dispatch_layout(
        payload: pl.Tensor[[routes, D], pl.INT8],
        scales: pl.Tensor[[routes], pl.FP32],
        counts: pl.Tensor[[EXPERTS], pl.INT64],
        expanded: pl.Tensor[[routes], pl.INT32],
        weights: pl.Tensor[[tokens, TOPK], pl.FP32],
        recv_x: pl.InOut[pl.Tensor[[EXPERTS, RECV_MAX, D], pl.INT8]],
        recv_scale: pl.InOut[pl.Tensor[[EXPERTS, RECV_MAX], pl.FP32]],
        recv_weights: pl.InOut[pl.Tensor[[EXPERTS, RECV_MAX], pl.FP32]],
        recv_count: pl.InOut[pl.Tensor[[EXPERTS, 1], pl.INT32]],
        route_map: pl.InOut[pl.Tensor[[routes, MAP_PAD], pl.INT32]],
    ):
        # Serialize narrow scalar stores that can share a cache line.
        for _block in pl.spmd(1, name_hint="a2_moe_dispatch_metadata"):
            begin = pl.cast(0, pl.INT32)
            for expert in pl.range(EXPERTS):
                count = pl.cast(pl.read(counts, [expert]), pl.INT32)
                pl.write(recv_count, [expert, 0], count)
                for original in pl.range(routes):
                    dispatched = pl.read(expanded, [original])
                    if dispatched >= begin and dispatched < begin + count:
                        row = dispatched - begin
                        route_weight = pl.read(weights, [original // TOPK, original % TOPK])
                        token_scale = pl.read(scales, [dispatched])
                        pl.write(recv_scale, [expert, row], token_scale)
                        pl.write(recv_weights, [expert, row], route_weight)
                        target = pl.cast(expert * RECV_MAX, pl.INT32) + row
                        pl.write(route_map, [original, 0], target)
                begin = begin + count

        recv_flat = pl.reshape(recv_x, [EXPERTS * RECV_MAX, D])
        for block in pl.spmd(routes * (D // D_TILE), name_hint="a2_moe_dispatch_payload"):
            original = block // (D // D_TILE)
            column = block % (D // D_TILE) * D_TILE
            dispatched = pl.read(expanded, [original])
            target = pl.read(route_map, [original, 0])
            value = payload[dispatched:dispatched + 1, column:column + D_TILE]
            recv_flat[target:target + 1, column:column + D_TILE] = value
        return recv_x, recv_scale, recv_weights, recv_count, route_map

    return dispatch_layout


def make_route_restore(tokens):
    """Gather every actual weighted expert result in original token/top6 order."""
    routes = tokens * TOPK

    @pl.jit
    def route_restore(
        recv_y: pl.Tensor[[EXPERTS, RECV_MAX, D], pl.FP32],
        route_map: pl.Tensor[[routes, MAP_PAD], pl.INT32],
        dispatch_y: pl.Out[pl.Tensor[[routes, D], pl.FP32]],
    ):
        recv_flat = pl.reshape(recv_y, [EXPERTS * RECV_MAX, D])
        for block in pl.spmd(routes * (D // D_TILE), name_hint="a2_moe_route_restore"):
            original = block // (D // D_TILE)
            column = block % (D // D_TILE) * D_TILE
            source = pl.read(route_map, [original, 0])
            value = recv_flat[source:source + 1, column:column + D_TILE]
            dispatch_y[original:original + 1, column:column + D_TILE] = value
        return dispatch_y

    return route_restore


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--device", type=int, default=0)
    validate_ep_dispatch(device=parser.parse_args().device)

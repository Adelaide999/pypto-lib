# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: manual
"""Compose physical Prefill layers with CP/SP attention, Engram, EP MoE and delayed HC."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

from pathlib import Path
import sys

import torch
import pypto.language as pl


def validate_prefill_ownership(layout, ownership):
    """Validate every native expert and local slot and return JSON-compatible pairs."""
    if not isinstance(ownership, (list, tuple)) or len(ownership) != layout.ep:
        raise ValueError("Expert ownership must cover the complete EP group")
    canonical = []
    for rank, pairs in enumerate(ownership):
        if not isinstance(pairs, (list, tuple)) or len(pairs) != layout.experts // layout.ep:
            raise ValueError("Expert ownership must contain every local expert")
        actual = []
        for pair in pairs:
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or any(type(value) is not int for value in pair)
            ):
                raise ValueError("Expert ownership requires integer expert and local-slot pairs")
            actual.append(list(pair))
        if actual != layout.ownership(rank):
            raise ValueError("Expert ownership differs from the native checkpoint contract")
        canonical.append(actual)
    return canonical


def make_prefill_layer_entries(
    capacity,
    config,
    expert_capacity,
    layer_id=0,
    cache_blocks=3,
    requests=1,
    table_pages=2,
    *,
    ep,
    tp=1,
    cache_layout=None,
    role=None,
    network_cache=False,
):
    """Construct shared physical-layer attention, HC and MoE computations."""
    from .hc import make_hc_post_inline, make_hc_pre_inline
    from .moe import make_moe_combine
    from .moe import make_ep_dispatch_pack, make_ep_dispatch_unpack, make_ep_return_pack
    from .moe import make_moe_router
    from .moe import make_shared_moe_w8a8
    from .norm_rope import make_rope_tables
    from .prefill_cp import make_head_pack, make_sp_gather_unpack, make_sp_masked_zero, make_sp_reduce_pack
    from .rmsnorm import make_prefill_norm_cast
    from .prefill_layer_state import resolve_layer_role
    from .prefill_swa import make_builders
    from .runtime_config import validate_tp

    role = resolve_layer_role(layer_id, role)
    tp = validate_tp(tp)
    if tp not in (1, 2, 4) or capacity % 16 or not 2 <= ep <= 16 or ep % tp or 384 % ep:
        raise ValueError("Require TP1/2/4 and an aligned complete Prefill layer")
    entries = make_builders(capacity, cache_blocks=cache_blocks, tp=tp)
    del entries["swa"]
    entries.update(
        hc_pre=pl.jit(make_hc_pre_inline(capacity, config)._func),
        hc_post=pl.jit(make_hc_post_inline(capacity, config)._func),
        norm_cast=make_prefill_norm_cast(capacity, 5120),
        rope_tables=make_rope_tables(capacity, 64, config["rope_theta"]),
        head_pack=make_head_pack(capacity, 64, 512, tp),
        head_unpack=make_sp_gather_unpack(capacity, 32768 // tp, tp),
        sp_unpack=make_sp_gather_unpack(capacity, 5120, tp),
        reduce_pack=make_sp_reduce_pack(capacity, 5120, tp),
        final_unpack=make_sp_gather_unpack(capacity, 20480, tp),
        pre_unpack=make_sp_gather_unpack(capacity, 4, tp, pl.FP32, 4),
        router=make_moe_router(capacity, config, pl.FP32),
        shared=make_shared_moe_w8a8(capacity, config, fused_activation=True, fused_channel_dtype=pl.FP32),
        routed=make_shared_moe_w8a8(expert_capacity, config, fused_activation=True),
        dispatch_pack=make_ep_dispatch_pack(capacity, 5120, 384, 6, ep),
        dispatch_unpack=make_ep_dispatch_unpack(capacity, 5120, 384, 6, ep, expert_capacity),
        return_pack=make_ep_return_pack(capacity, 5120, 384, 6, ep, expert_capacity),
        combine=make_moe_combine(capacity, 5120, 6, ep),
        mask_moe=make_sp_masked_zero(capacity, 5120),
        mask_output=make_sp_masked_zero(capacity, 20480),
    )
    if role["mode"] == "full":
        from .prefill_c2a import make_physical_compressed_entries

        ratio = role["ratio"]
        compressed = make_physical_compressed_entries(
            capacity,
            config,
            cache_blocks,
            requests,
            table_pages,
            compress_ratio=ratio,
            cache_layout=cache_layout,
            layer_id=layer_id,
            network_cache=network_cache,
        )
        entries = dict(
            compressed, **{name: entry for name, entry in entries.items() if name not in compressed}
        )
    if role["mode"] in ("reuse", "reindex"):
        from .prefill_compressed_reuse import make_physical_consumer_entries

        consumer = make_physical_consumer_entries(
            capacity,
            config,
            cache_blocks,
            requests,
            table_pages,
            compress_ratio=role["ratio"],
            cache_layout=cache_layout,
            layer_id=layer_id,
            role=role,
            network_cache=network_cache,
        )
        entries = dict(consumer, **{name: entry for name, entry in entries.items() if name not in consumer})
    return entries


def run_prefill_layer_inmemory(role, batch, local_hidden, local_pre, cache_state, weight_provider, runtime):
    """Execute the shared complete layer on caller-owned local activations and live state.

    This host scheduler invokes the existing PyPTO kernels and actual multi-rank
    exchange callback. It does not load a fixture, read a reference, or restore
    next-layer inputs from files. Buffers are local SP shards, not gathered output.
    """
    from .moe import DISPATCH_MAP_PAD as MAP_PAD, validate_received_routes, validate_returned_routes
    from .prefill_l2 import execute_region
    from .prefill_compressed_runtime import run_compressed_attention
    from .prefill_consumer_runtime import run_consumer_attention
    from .prefill_layer_state import LayerObservations, invalidate_live_versions, publish_live_versions
    from .prefill_layer_state import resolve_layer_role, validate_live_state
    from .runtime_config import ExpertParallelConfig
    from .swa_scheduler import from_controls

    layer_id = role["physical_layer"]
    role = resolve_layer_role(layer_id, role)
    topology = runtime["topology"]
    layout = ExpertParallelConfig(topology["dp"], topology["tp"], topology["ep"], 384)
    tp, world = layout.tp, layout.ep
    if tp not in (1, 2, 4) or not 2 <= world <= 16:
        raise ValueError("Require matching TP1/2/4 with real multi-rank EP before building the layer")
    if any(topology[name] != 1 for name in ("pp", "pcp", "dcp")):
        raise ValueError("Require PP1/PCP1/DCP1")
    if not runtime["enable_expert_parallel"] or not all(runtime["required_modes"].values()):
        raise ValueError("Original parallel and model modes must remain enabled")
    validate_prefill_ownership(layout, runtime["ownership"])
    capacity, expert_capacity = runtime["capacity"], runtime["expert_capacity"]
    if not 0 < capacity <= 512 or capacity % 16 or expert_capacity <= 0 or expert_capacity % 16:
        raise ValueError("The current complete core requires bounded positive 16-row capacities")
    if any(len(records) != world for records in batch["cases"].values()):
        raise ValueError("Every batch must cover the actual complete world")
    if any(len(values) != world for values in (local_hidden, local_pre, cache_state)):
        raise ValueError("Local activation and state lists must cover the complete world")
    epoch, token_map_id = batch["epoch"], batch["token_map_id"]
    if epoch != runtime["epoch"] or token_map_id != runtime["token_map_id"]:
        raise ValueError("Layer runtime and batch must identify the same chunk and token map")
    for case, records in batch["cases"].items():
        for rank, original in enumerate(records):
            physical, valid = original["SP_physical_rows"], original["SP_valid_rows"]
            if not 0 <= valid <= physical <= capacity or len(original["positions"]) > capacity:
                raise ValueError("Invalid actual physical/valid row bounds")
            hidden, pre = local_hidden[rank][case], local_pre[rank][case]
            if tuple(hidden.shape) != (capacity, 4, 5120) or hidden.dtype != torch.bfloat16:
                raise ValueError("Require local padded BF16 hidden [capacity,4,5120]")
            if tuple(pre.shape) != (capacity, 4) or pre.dtype != torch.float32:
                raise ValueError("Require local delayed FP32 HC pre [capacity,4]")
            if hidden.device.type != "cpu" or pre.device.type != "cpu":
                raise ValueError("The current PyPTO executor uses explicit CPU staging")
            validate_live_state(role, original, cache_state[rank][case], epoch, token_map_id)
    raw, operands, expert_weights = (
        weight_provider["raw"],
        weight_provider["operands"],
        weight_provider["expert"],
    )
    prefix = f"layers.{layer_id}"
    cache_pages = runtime["cache_blocks"]
    requests, table_pages = runtime["requests"], runtime["table_pages"]
    memory, execute = runtime["memory"], runtime["execute"]
    namespace = runtime.get("stage_namespace", "")

    def exchange(stage, values, collective, group, *, packets=False):
        stage = namespace + "-" + stage if namespace else stage
        return runtime["exchange"](stage, values, collective, group, packets=packets)

    def zeros(*shape, dtype=torch.bfloat16):
        return torch.zeros(shape, dtype=dtype)

    def pad(value):
        target = zeros(capacity, *value.shape[1:], dtype=value.dtype)
        target[: len(value)].copy_(value)
        return target

    local_experts, head_width, output_width = 384 // world, 32768 // tp, 8192 // tp
    local_groups = 8 // tp
    capture_nodes = runtime.get("capture_nodes", False)
    capture_state = runtime.get("capture_state", False)
    states = [{} for _ in range(world)]
    for rank in range(world):
        for case in batch["cases"]:
            invalidate_live_versions(role, cache_state[rank][case])
    engram_states = None
    if role["engram_slot"] is not None:
        if not callable(runtime.get("engram")):
            raise ValueError(
                "Engram requires a real PyPTO callback with current hidden and request hash state"
            )
        engram_states = runtime["engram"](role, batch, local_hidden, cache_state, exchange=exchange)
    # SP physical observations and gathered normalized attention inputs.
    norm_packets = [{} for _ in range(world)]
    for case, originals in batch["cases"].items():
        for rank, original in enumerate(originals):
            physical, valid = original["SP_physical_rows"], original["SP_valid_rows"]
            total = len(original["positions"])
            if not 0 <= valid <= physical <= capacity or total > capacity:
                raise ValueError("Invalid frozen physical/valid row bounds")
            readonly_hidden = local_hidden[rank][case]
            hidden, incoming = readonly_hidden, local_pre[rank][case]
            engram = None if engram_states is None else engram_states[rank][case]
            if engram is not None:
                hidden = engram["hidden"]
            pre, post = zeros(capacity, 4, dtype=torch.float32), zeros(capacity, 4, dtype=torch.float32)
            comb = zeros(capacity, 4, 4, dtype=torch.float32)
            collapsed, normed = zeros(capacity, 5120), zeros(capacity, 5120)
            outputs = dict(pre=pre, post=post, comb=comb, collapsed=collapsed, normed=normed)
            arguments = [
                hidden,
                operands["hc_function"],
                operands["hc_scale"],
                operands["hc_base"],
                incoming,
                operands["attn_norm"],
                pre,
                post,
                comb,
                collapsed,
                normed,
                physical,
            ]
            execute("collapse_norm", rank, case, arguments, outputs, ("hc_pre", "attention_norm"))
            nodes = {} if capture_nodes else LayerObservations(False)
            if capture_nodes:
                nodes.update(
                    {
                        "hc_pre0.collapsed": collapsed[:physical].clone(),
                        "hc_pre0.post": post[:physical].clone(),
                        "hc_pre0.comb": comb[:physical].clone(),
                        "hc_pre0.next_pre": pre[:physical].clone(),
                        "attention_norm": normed[:physical].clone(),
                    }
                )
            if engram is not None:
                if capture_nodes:
                    nodes.update(engram["nodes"])
            states[rank][case] = dict(
                original=original,
                physical=physical,
                valid=valid,
                total=total,
                readonly_hidden=readonly_hidden,
                engram=engram,
                hidden=hidden,
                incoming=incoming,
                pre=pre,
                post=post,
                comb=comb,
                nodes=nodes,
                role=role,
                live=cache_state[rank][case],
            )
            state = states[rank][case]
            live = state["live"]
            if role["mode"] != "swa":
                state["raw_cache_before"] = (
                    {name: value.clone() for name, value in live["raw"].items()} if capture_state else None
                )
                state["shared_state_before"] = (
                    {name: value.clone() for name, value in live["shared"].items()} if capture_state else None
                )
            norm_packets[rank][case] = normed
    gathered = exchange("attention-input-gather", norm_packets, "all_gather", "tp")
    head_packets = [{} for _ in range(world)]
    for case, originals in batch["cases"].items():
        for rank, original in enumerate(originals):
            state = states[rank][case]
            physical, total, nodes = state["physical"], state["total"], state["nodes"]
            full_norm = zeros(capacity, 5120)
            execute(
                "sp_unpack",
                rank,
                case,
                [gathered[rank][case], full_norm, physical, total],
                {"normed": full_norm},
                "attention_norm",
            )
            start, end, per_rank, _padded = original["cp_token_range"]
            query_rows = max(0, min(end, total) - min(start, total))
            q_input = pad(full_norm[start : start + query_rows])
            positions = pad(original["positions"])
            cosine, sine = zeros(capacity, 32, dtype=torch.float32), zeros(capacity, 32, dtype=torch.float32)
            execute(
                "rope_tables",
                rank,
                case,
                [positions, cosine, sine, total],
                dict(cosine=cosine, sine=sine),
                "rope",
            )
            q_cos, q_sin = pad(cosine[start : start + query_rows]), pad(sine[start : start + query_rows])
            wqa, latent, qb, query = (
                zeros(capacity, 1280),
                zeros(capacity, 1280),
                zeros(capacity, 32768),
                zeros(capacity, 32768),
            )
            kv_raw, kv_norm, kv_rope = zeros(capacity, 512), zeros(capacity, 512), zeros(capacity, 512)
            execute_region(
                runtime,
                "qkv_l2",
                rank,
                case,
                [
                    (
                        "wq_a",
                        [q_input, operands["wq_a"], operands["wq_a_scale"], wqa, query_rows],
                        {"wqa": wqa},
                        "qkv",
                    ),
                    ("q_norm", [wqa, operands["q_norm"], latent, query_rows], {"latent": latent}, "qkv"),
                    (
                        "wq_b",
                        [latent, operands["wq_b"], operands["wq_b_scale"], qb, query_rows],
                        {"qb": qb},
                        "qkv",
                    ),
                    ("q_rope", [qb, q_cos, q_sin, query, query_rows], {"query": query}, "rope"),
                    (
                        "wkv",
                        [full_norm, operands["wkv"], operands["wkv_scale"], kv_raw, total],
                        {"kv_raw": kv_raw},
                        "qkv",
                    ),
                    ("kv_norm", [kv_raw, operands["kv_norm"], kv_norm, total], {"kv_norm": kv_norm}, "qkv"),
                    ("kv_rope", [kv_norm, cosine, sine, kv_rope, total], {"kv": kv_rope}, "rope"),
                ],
            )
            state["kv_rope"] = kv_rope
            if role["mode"] == "full":
                context, cache = run_compressed_attention(
                    state,
                    full_norm,
                    q_input,
                    latent,
                    query,
                    cosine,
                    sine,
                    operands,
                    execute,
                    rank,
                    case,
                    capacity,
                    cache_pages,
                    requests,
                    table_pages,
                    layer_id,
                )
                inverse = zeros(capacity, 32768)
            elif role["mode"] in ("reuse", "reindex"):
                context, cache = run_consumer_attention(
                    state,
                    query,
                    operands,
                    execute,
                    rank,
                    case,
                    capacity,
                    cache_pages,
                    requests,
                    table_pages,
                    layer_id,
                    q_input=q_input,
                    latent=latent,
                    cosine=cosine,
                    sine=sine,
                )
                inverse = zeros(capacity, 32768)
            else:
                controls = original["metadata"]
                schedule = from_controls(
                    controls["query_start_loc"].tolist(),
                    controls["seq_lens"].tolist(),
                    controls["block_table"].tolist(),
                    cache_blocks=cache_pages,
                    capacity=capacity,
                )
                slots = torch.full((capacity, 2), -1, dtype=torch.int64)
                slot_rows = torch.tensor(schedule["slot_mapping"], dtype=torch.int64).reshape(-1, 2)
                slots[:total].copy_(slot_rows)
                indices = torch.full((capacity, 128), -1, dtype=torch.int32)
                local_indices = schedule["physical_window_indices"][start : start + query_rows]
                if query_rows:
                    indices[:query_rows].copy_(torch.tensor(local_indices, dtype=torch.int32))
                cache = state["live"]["cache"]
                execute(
                    "cache_publish",
                    rank,
                    case,
                    [kv_rope, slots, cache, total],
                    {"cache": cache},
                    "cache_publish",
                )
                context, inverse = zeros(capacity, 32768), zeros(capacity, 32768)
                execute(
                    "swa_read",
                    rank,
                    case,
                    [query, indices, operands["sink"], cache, context, query_rows],
                    {"context": context},
                    "swa",
                )
            packet = zeros(tp * capacity, head_width)
            execute_region(
                runtime,
                "attention_pack_l2",
                rank,
                case,
                [
                    (
                        "inverse_rope",
                        [context, q_cos, q_sin, inverse, query_rows],
                        {"inverse": inverse},
                        "rope",
                    ),
                    ("head_pack", [inverse, packet, query_rows], {"packet": packet}, "output_projection"),
                ],
            )
            if capture_nodes:
                nodes.update(
                    q_norm=latent[:query_rows].clone(),
                    query_latent=latent[:query_rows].clone(),
                    query_rope=query[:query_rows].reshape(query_rows, 64, 512).clone(),
                    attention_context=context[:query_rows].reshape(query_rows, 64, 512).clone(),
                )
                nodes.update(
                    wkv=kv_raw[:total].clone(),
                    kv_norm=kv_norm[:total].clone(),
                    attention_context_inverse_rope=inverse[:query_rows].reshape(query_rows, 64, 512).clone(),
                )
            state.update(cache=cache, per_rank=per_rank, positions=positions)
            head_packets[rank][case] = packet
    exchanged = exchange("attention-head-exchange", head_packets, "all_to_all", "tp")
    reduction = [{} for _ in range(world)]
    for case in batch["cases"]:
        for rank in range(world):
            state, tp_rank = states[rank][case], rank % tp
            total, physical, nodes = state["total"], state["physical"], state["nodes"]
            heads, woa, wob = (
                zeros(capacity, head_width),
                zeros(capacity, output_width),
                zeros(capacity, 5120),
            )
            weight_a = operands["wo_a_grouped_prefill"][
                tp_rank * local_groups : (tp_rank + 1) * local_groups
            ].contiguous()
            weight_b = operands["wo_b"][tp_rank * output_width : (tp_rank + 1) * output_width].contiguous()
            packed = zeros(tp * capacity, 5120)
            execute_region(
                runtime,
                "output_projection_l2",
                rank,
                case,
                [
                    (
                        "head_unpack",
                        [exchanged[rank][case], heads, state["per_rank"], total],
                        {"heads": heads},
                        "output_projection",
                    ),
                    ("grouped_wo_a", [heads, weight_a, woa, total], {"wo_a": woa}, "output_projection"),
                    ("wo_b", [woa, weight_b, wob, total], {"wo_b": wob}, "output_projection"),
                    ("reduce_pack", [wob, packed, physical, total], {"packed": packed}, "output_projection"),
                ],
            )
            if capture_nodes:
                nodes.update(
                    attention_heads_after_exchange=heads[:total].reshape(total, 64 // tp, 512).clone(),
                    wo_a=woa[:total].clone(),
                    wo_b=wob[:total].clone(),
                    attention_output=wob[:total].clone(),
                )
            reduction[rank][case] = packed
    scattered = exchange("attention-output-scatter", reduction, "reduce_scatter", "tp")
    dispatch_packets = [{} for _ in range(world)]
    routes, packet_rows = capacity * 6, world * capacity * 6
    for case in batch["cases"]:
        routing_rows = max(original["SP_physical_rows"] for original in batch["cases"][case])
        for rank in range(world):
            state = states[rank][case]
            physical, nodes = state["physical"], state["nodes"]
            changed = zeros(capacity, 4, 5120)
            pre, post = zeros(capacity, 4, dtype=torch.float32), zeros(capacity, 4, dtype=torch.float32)
            comb, collapsed = zeros(capacity, 4, 4, dtype=torch.float32), zeros(capacity, 5120)
            hc_args = [
                changed,
                raw[f"{prefix}.hc_ffn_fn"],
                raw[f"{prefix}.hc_ffn_scale"],
                raw[f"{prefix}.hc_ffn_base"],
                state["pre"],
                pre,
                post,
                comb,
                collapsed,
                physical,
            ]
            normed, router_input = zeros(capacity, 5120), zeros(capacity, 5120, dtype=torch.float32)
            logits, weights = (
                zeros(capacity, 384, dtype=torch.float32),
                zeros(capacity, 6, dtype=torch.float32),
            )
            ids = torch.zeros((capacity, 6), dtype=torch.int32)
            gate = raw[f"{prefix}.ffn.gate.weight"].float()
            token_ids = pad(state["original"]["input_ids"])
            router_args = [
                router_input,
                gate,
                raw[f"{prefix}.ffn.gate.bias"],
                raw[f"{prefix}.ffn.gate.bias_vl"],
                token_ids,
                logits,
                ids,
                weights,
                routing_rows,
            ]
            shared = zeros(capacity, 5120)
            with memory.interval(rank, "load", "shared_expert"):
                shared_operands = expert_weights(rank, None)
            packet = {
                "routes": torch.full((packet_rows, 4), -1, dtype=torch.int64),
                "hidden": zeros(packet_rows, 5120),
                "counts": torch.zeros((world, MAP_PAD), dtype=torch.int32),
            }
            args_pack = [
                normed,
                ids,
                packet["routes"],
                packet["hidden"],
                packet["counts"],
                state["valid"],
                rank,
            ]
            execute_region(
                runtime,
                "ffn_dispatch_l2",
                rank,
                case,
                [
                    (
                        "hc_post",
                        [
                            scattered[rank][case],
                            state["hidden"],
                            state["post"],
                            state["comb"],
                            changed,
                            physical,
                        ],
                        {"changed": changed},
                        "hc_post",
                    ),
                    ("hc_pre", hc_args, dict(pre=pre, post=post, comb=comb, collapsed=collapsed), "hc_pre"),
                    (
                        "norm_cast",
                        [collapsed, raw[f"{prefix}.ffn_norm.weight"], normed, router_input, physical],
                        dict(normed=normed, router_input=router_input),
                        "ffn_norm",
                    ),
                    (
                        "router",
                        router_args,
                        dict(logits=logits, route_ids=ids, route_weights=weights),
                        "router",
                    ),
                    ("shared", [normed, *shared_operands, shared, physical], {"shared": shared}, "shared"),
                    ("dispatch_pack", args_pack, packet, "dispatch_pack"),
                ],
            )
            if capture_nodes:
                nodes.update(
                    {
                        "hc_post0": changed[:physical].clone(),
                        "hc_pre1.collapsed": collapsed[:physical].clone(),
                        "hc_pre1.post": post[:physical].clone(),
                        "hc_pre1.comb": comb[:physical].clone(),
                        "hc_pre1.next_pre": pre[:physical].clone(),
                        "ffn_norm_bf16": normed[:physical].clone(),
                        "ffn_router_input_fp32": router_input[:physical].clone(),
                        "logits": logits[:physical].clone(),
                        "route_ids": ids[:routing_rows].clone(),
                        "route_weights": weights[:routing_rows].clone(),
                        "shared": shared[:physical].clone(),
                    }
                )
            state.update(
                changed=changed,
                ffn_pre=pre,
                ffn_post=post,
                ffn_comb=comb,
                weights=weights,
                route_ids=ids,
                shared=shared,
                routing_rows=routing_rows,
                token_ids=token_ids,
            )
            dispatch_packets[rank][case] = packet
    received = exchange("expert-dispatch", dispatch_packets, "all_to_all", "ep", packets=True)
    returns = [{} for _ in range(world)]
    for rank in range(world):
        for case, packet in received[rank].items():
            state = states[rank][case]
            valid_by_rank = [states[source][case]["valid"] for source in range(world)]
            expected_total = validate_received_routes(packet, rank, layout, routes, valid_by_rank)
            recv_routes = torch.full((packet_rows, 4), -1, dtype=torch.int64)
            recv_hidden, expert_x = zeros(packet_rows, 5120), zeros(local_experts * expert_capacity, 5120)
            counts = torch.zeros(local_experts, dtype=torch.int64)
            packet_map = torch.full((packet_rows, MAP_PAD), -1, dtype=torch.int32)
            received_total = torch.zeros(MAP_PAD, dtype=torch.int32)
            unpack_args = [
                packet["routes"],
                packet["hidden"],
                packet["counts"],
                recv_routes,
                recv_hidden,
                expert_x,
                counts,
                packet_map,
                received_total,
                rank,
            ]
            unpack_out = dict(
                received_routes=recv_routes,
                received_hidden=recv_hidden,
                expert_counts=counts,
                packet_map=packet_map,
                received_total=received_total,
            )
            execute("dispatch_unpack", rank, case, unpack_args, unpack_out, "dispatch_unpack")
            total = int(received_total[0])
            if total != expected_total:
                raise RuntimeError("Dispatch unpack lost received routes")
            expert_y = zeros(local_experts * expert_capacity, 5120)
            for local, count in enumerate(counts.tolist()):
                if not 0 <= count <= expert_capacity:
                    raise RuntimeError("Expert slab exceeded its input-derived bound")
                if count:
                    expert, begin = rank * local_experts + local, local * expert_capacity
                    with memory.interval(rank, "load", f"expert-{expert}"):
                        expert_operands = expert_weights(rank, expert)
                    x = expert_x[begin : begin + expert_capacity]
                    y = expert_y[begin : begin + expert_capacity]
                    execute(
                        "routed",
                        rank,
                        case + f"-expert-{expert}",
                        [x, *expert_operands, y, count],
                        {"expert_output": y},
                        "routed",
                    )
            sent = zeros(packet_rows, 5120)
            execute(
                "return_pack",
                rank,
                case,
                [expert_y, packet["counts"], packet_map, sent],
                {"hidden": sent},
                "dispatch_pack",
            )
            returns[rank][case] = dict(routes=packet["routes"], counts=packet["counts"], hidden=sent)
            if capture_nodes:
                state.update(
                    received_routes=recv_routes[:total].clone(), received_hidden=recv_hidden[:total].clone()
                )
                state["nodes"]["local_expert_counts"] = counts.to(torch.int32)
    returned = exchange("expert-return", returns, "all_to_all", "ep", packets=True)
    final_hidden, final_pre = [{} for _ in range(world)], [{} for _ in range(world)]
    for rank in range(world):
        for case, packet in returned[rank].items():
            state = states[rank][case]
            validate_returned_routes(packet, rank, routes, state["route_ids"], state["valid"])
            routed, combined = zeros(capacity, 5120), zeros(capacity, 5120)
            returned_routes = torch.full((routes, 4), -1, dtype=torch.int64)
            combine_args = [
                packet["routes"],
                packet["hidden"],
                packet["counts"],
                state["weights"],
                state["shared"],
                routed,
                combined,
                returned_routes,
                state["physical"],
            ]
            valid, physical = state["valid"], state["physical"]
            output = torch.full((capacity, 4, 5120), runtime["output_sentinel"], dtype=torch.bfloat16)
            execute_region(
                runtime,
                "moe_finish_l2",
                rank,
                case,
                [
                    (
                        "combine",
                        combine_args,
                        dict(routed=routed, combined=combined, returned_routes=returned_routes),
                        "combine",
                    ),
                    ("mask_moe", [combined, valid, physical], {"combined": combined}, "combine"),
                    (
                        "hc_post",
                        [combined, state["changed"], state["ffn_post"], state["ffn_comb"], output, physical],
                        {"hidden": output},
                        "hc_post",
                        case + "-ffn",
                    ),
                ],
            )
            output_flat = output.reshape(capacity, 20480)
            execute("mask_output", rank, case, [output_flat, valid, physical], {"hidden": output}, "hc_post")
            routed_node = routed[: state["routing_rows"]].clone() if capture_nodes else None
            if capture_nodes:
                state["nodes"].update(
                    routed_native=routed_node,
                    moe_output=combined[:physical].clone(),
                    hc_post1=output[:physical].clone(),
                )
            state.update(padded_hidden=output)
            if capture_nodes:
                state["returned_routes"] = returned_routes[: state["valid"] * 6].clone()
            final_hidden[rank][case] = output
            final_pre[rank][case] = state["ffn_pre"]
    for rank in range(world):
        for case in batch["cases"]:
            states[rank][case]["nodes"] = dict(states[rank][case]["nodes"])
            publish_live_versions(role, cache_state[rank][case], epoch, token_map_id)
    return {"hidden": final_hidden, "pre_mix": final_pre, "states": states}


def run_prefill_network_inmemory(
    local_hidden,
    local_pre,
    *,
    batch_for_layer,
    cache_for_layer,
    weights_for_layer,
    runtime_for_layer,
    observe_layer=None,
):
    """Chain all original backbone roles using actual local product activations.

    Providers own allocation, weights, chunk metadata, queue-mapped runtime and
    persistent shared aliases. Final delayed collapse/norm/head is a separate
    caller-owned output stage, not a substitute for the layer39 boundary.
    """
    from .prefill_layer_state import resolve_layer_role

    namespaces = set()
    token_identity = None
    for layer_id in range(40):
        role = resolve_layer_role(layer_id)
        batch, runtime = batch_for_layer(layer_id), runtime_for_layer(layer_id)
        identity = (batch["epoch"], batch["token_map_id"])
        if token_identity is None:
            token_identity = identity
        if identity != token_identity:
            raise ValueError("A network chunk must preserve its request/token ownership through all layers")
        namespace = runtime.get("stage_namespace")
        if not namespace or namespace in namespaces or "/" in namespace or ".." in namespace:
            raise ValueError("Network layers require distinct safe chunk/layer stage namespaces")
        namespaces.add(namespace)
        result = run_prefill_layer_inmemory(
            role,
            batch,
            local_hidden,
            local_pre,
            cache_for_layer(layer_id),
            weights_for_layer(layer_id),
            runtime,
        )
        local_hidden, local_pre = result["hidden"], result["pre_mix"]
        if observe_layer is not None:
            observe_layer(layer_id, result)
        del result
    return {"hidden": local_hidden, "pre_mix": local_pre}


def run_prefill_layer(args, layer_id=0):
    """Execute one physical layer with CP/SP attention, Engram and multi-rank EP MoE."""
    from .moe_transport import CANNMemory
    from .prefill_layer_state import fixture_live_state, resolve_layer_role
    from .prefill_swa import prepare_product_operands
    from .runtime_config import ExpertParallelConfig
    from .runtime_utils import Executable, communicate, file_sha, loaded_source_files, write_json
    from .weights import CheckpointOperands

    contract = torch.load(args.inputs, map_location="cpu", weights_only=True)
    boundary_layer = contract.get("physical_layer_id", contract.get("layer", layer_id))
    if layer_id not in (0, 1, 2, 3, 14, 15, 19, 20, 21, 24, 25) or boundary_layer != layer_id:
        raise ValueError("Require the requested physical layer input boundary")
    if layer_id == 14:
        if contract.get("own_cache_sources") != dict(kv=14, index=14, hash_slot=1) or contract.get(
            "product_chain"
        ):
            raise ValueError("Physical14 requires its own caches and independent slot1 input boundary")
    if layer_id in (15, 19):
        if contract.get("source_physical_layer_id") != 14 or contract.get("product_chain"):
            raise ValueError("Require an independent physical consumer with explicit readonly source14")
    if layer_id == 21 and (contract.get("source_physical_layer_id") != 20 or contract.get("product_chain")):
        raise ValueError("Physical21 requires its own layer and explicit readonly source20")
    if layer_id == 25 and (
        contract.get("source_physical_layer_id") != 20
        or contract.get("incoming_source_physical_layer_id") != 24
        or contract.get("index_source_physical_layer_id") != 24
        or contract.get("candidate_source_physical_layer_id") != 20
        or contract.get("product_chain")
    ):
        raise ValueError("Physical25 requires readonly KV/candidates20 and incoming/selection24")
    if layer_id == 20 and (
        contract.get("own_cache_sources") != dict(kv=20, index=20, candidate=20)
        or contract.get("product_chain")
    ):
        raise ValueError("Physical20 requires its own ratio1 KV/index/candidate state")
    if layer_id == 24:
        from .prefill_indexer_runtime import validate_refresh_candidates

        if contract.get("product_chain") or contract.get("candidate_source_physical_layer_id") != 20:
            raise ValueError("Physical24 refresh requires an independent readonly20 candidate boundary")
        validate_refresh_candidates(contract)
    prefix = f"layers.{layer_id}"
    topology = contract["topology"]
    layout = ExpertParallelConfig(topology["dp"], topology["tp"], topology["ep"], 384)
    tp, world = layout.tp, layout.ep
    if any(topology[name] != 1 for name in ("pp", "pcp", "dcp")):
        raise ValueError("Require PP1/PCP1/DCP1")
    if args.tp != tp or not 2 <= world <= 16:
        raise ValueError("Require matching TP1/2/4 and a real local multi-rank EP topology")
    if any(len(records) != world for records in contract["cases"].values()):
        raise ValueError("Every case must cover the actual complete world")
    if not contract["enable_expert_parallel"] or not all(contract["required_modes"].values()):
        raise ValueError("Every native target mode must remain enabled")
    cache_pages = {2: 12, 3: 16, 14: 12, 15: 16, 19: 16, 20: 12, 21: 16, 24: 16, 25: 16}.get(layer_id, 3)
    if contract["global_experts"] != 384 or contract["top_k"] != 6 or contract["cache_pages"] != cache_pages:
        raise ValueError("Original expert, top-k and cache geometry is required")
    ownership = validate_prefill_ownership(layout, contract["ownership"])
    from .runtime_config import configure_devices

    assigned = configure_devices(args.devices, world=world)
    if (
        len(assigned) != world
        or len(set(assigned)) != world
        or any(not device.isdecimal() for device in assigned)
    ):
        raise ValueError("Require the complete distinct queue-assigned device set for this topology")
    torch.set_num_threads(4)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    captures, build = args.output / "captures", args.output / "compiled"
    captures.mkdir()
    build.mkdir()
    runtime = {
        "topology": topology,
        "assigned_devices": assigned,
        "logical_rank_mapping": list(range(world)),
        "enable_expert_parallel": True,
        "required_modes": contract["required_modes"],
        "groups": {
            "ep": [list(range(world))],
            "tp": [list(range(r, r + tp)) for r in range(0, world, tp)],
            "dp": [list(range(r, world, tp)) for r in range(tp)],
        },
        "source_root": str(Path(__file__).resolve().parents[2]),
        "source_files": {},
        "command": [sys.executable, *sys.argv],
        "inputs_sha256": file_sha(args.inputs),
        "checkpoint_config_sha256": file_sha(Path(contract["checkpoint"]) / "config.json"),
        "complete_layer_status": "not_run",
        "physical_layer_id": layer_id,
    }
    runtime_path = args.output / "runtime_config.json"
    write_json(runtime_path, runtime)
    reports = [
        {
            "rank": rank,
            "ownership": ownership[rank],
            "kernels": [],
            "communication": [],
            "consumed_tensors": {},
            "command": runtime["command"],
        }
        for rank in range(world)
    ]
    memory = CANNMemory()

    def exchange(stage, values, collective, group, *, packets=False):
        transport = (
            values
            if packets
            else [{case: {"value": value} for case, value in rank_values.items()} for rank_values in values]
        )
        results, receipts = communicate(
            stage, transport, args.output, runtime_path, collective=collective, group=group
        )
        for rank, receipt in enumerate(receipts):
            reports[rank]["communication"].append(receipt)
        if packets:
            return results
        return [{case: packet["value"] for case, packet in rank_values.items()} for rank_values in results]

    # Actual product group prechecks precede all model computation.
    precheck = [{"rank": torch.tensor([[rank]], dtype=torch.int32)} for rank in range(world)]
    for group in ("ep", "tp", "dp"):
        observed = exchange("startup-" + group, precheck, "all_gather", group)
        for rank in range(world):
            ranks = next(ranks for ranks in runtime["groups"][group] if rank in ranks)
            if observed[rank]["rank"].flatten().tolist() != ranks:
                raise RuntimeError("Product HCCL group precheck failed: " + group)
    capacity = contract["capacity"]
    model_config = contract["model_config"]
    config = dict(model_config["text_config"], image_sentinel_base_id=model_config["image_token_id"])
    max_tokens = max(
        sum(original["SP_valid_rows"] for original in originals) for originals in contract["cases"].values()
    )
    expert_capacity = (max_tokens + 15) // 16 * 16
    requests, table_pages = 1, 2
    if layer_id in (2, 14, 20):
        from .prefill_compressed_runtime import compressed_geometry

        cache_pages, requests, table_pages = compressed_geometry(contract)
    if layer_id in (3, 15, 19, 21, 24, 25):
        from .prefill_compressed_runtime import compressed_geometry

        cache_pages, requests, table_pages = compressed_geometry(contract)
    functions = make_prefill_layer_entries(
        capacity,
        config,
        expert_capacity,
        layer_id,
        cache_pages,
        requests,
        table_pages,
        ep=world,
        cache_layout=next(iter(contract["cases"].values()))[0].get("cache_layout"),
        tp=tp,
    )
    entries = {name: Executable(function, build / name) for name, function in functions.items()}

    def execute(name, rank, case, arguments, outputs, responsibility=None):
        roles = name if responsibility is None else responsibility
        path = captures / f"rank-{rank}-{case}-{name}-{len(reports[rank]['kernels'])}.pt"
        entries[name].execute(arguments, outputs, rank, roles, path, reports[rank], memory)

    def zeros(*shape, dtype=torch.bfloat16):
        return torch.zeros(shape, dtype=dtype)

    def pad(value):
        target = zeros(capacity, *value.shape[1:], dtype=value.dtype)
        target[: len(value)].copy_(value)
        return target

    readers = [CheckpointOperands(contract["checkpoint"]) for _ in range(world)]
    shared_reader = CheckpointOperands(contract["checkpoint"])
    try:
        with memory.interval(0, "load", "shared_checkpoint"):
            raw = {
                name: shared_reader.load(name)
                for name in shared_reader.index
                if name.startswith(prefix + ".")
                and ".ffn.experts." not in name
                and ".engram.embed." not in name
            }
            operands = prepare_product_operands(raw, layer_id)
        for rank, reader in enumerate(readers):
            reader.consumed.update(shared_reader.consumed)
            with memory.interval(rank, "load", "owned_checkpoint"):
                for expert, _slot in layout.ownership(rank):
                    for stem in ("w1", "w2", "w3"):
                        for suffix in ("weight", "weight_scale", "weight_offset"):
                            reader.load(f"{prefix}.ffn.experts.{expert}.{stem}.{suffix}")
            memory.sample(rank, "checkpoint/load/after")
        role = resolve_layer_role(layer_id)
        epoch, token_map_id = "independent-fixture", runtime["inputs_sha256"]
        batch = {"cases": contract["cases"], "epoch": epoch, "token_map_id": token_map_id}
        live = fixture_live_state(contract, role, epoch, token_map_id)
        hidden = [
            {case: pad(records[rank]["hidden"]) for case, records in contract["cases"].items()}
            for rank in range(world)
        ]
        pre = [
            {case: pad(records[rank]["pre_mix"]) for case, records in contract["cases"].items()}
            for rank in range(world)
        ]

        def expert_weights(rank, expert):
            name = prefix + ".ffn.shared_experts" if expert is None else f"{prefix}.ffn.experts.{expert}"
            return readers[rank].expert(name)

        def engram(current_role, current_batch, current_hidden, current_live, *, exchange):
            from .prefill_engram_runtime import run_prefill_engram

            hash_caches = [
                {case: state["hash_cache"] for case, state in records.items()} for records in current_live
            ]
            return run_prefill_engram(
                contract,
                raw,
                args.output,
                reports,
                memory,
                exchange,
                layer_id,
                role=current_role,
                local_hidden=current_hidden,
                live_hash_caches=hash_caches,
            )

        core_runtime = dict(
            topology=topology,
            ownership=ownership,
            enable_expert_parallel=True,
            required_modes=contract["required_modes"],
            capacity=capacity,
            expert_capacity=expert_capacity,
            cache_blocks=cache_pages,
            requests=requests,
            table_pages=table_pages,
            memory=memory,
            execute=execute,
            exchange=exchange,
            epoch=epoch,
            token_map_id=token_map_id,
            output_sentinel=contract["output_sentinel"],
            capture_nodes=True,
            capture_state=True,
            engram=engram,
        )
        computed = run_prefill_layer_inmemory(
            role,
            batch,
            hidden,
            pre,
            live,
            {"raw": raw, "operands": operands, "expert": expert_weights},
            core_runtime,
        )
        states = computed["states"]
        final_hidden = [
            {case: value.reshape(capacity, 20480) for case, value in records.items()}
            for records in computed["hidden"]
        ]
        final_pre = computed["pre_mix"]
        full_hidden = exchange("final-hidden-gather", final_hidden, "all_gather", "tp")
        full_pre = exchange("final-hc-gather", final_pre, "all_gather", "tp")
        for rank in range(world):
            for case, state in states[rank].items():
                physical, total, original = state["physical"], state["total"], state["original"]
                hidden, pre = zeros(capacity, 20480), zeros(capacity, 4, dtype=torch.float32)
                execute(
                    "final_unpack",
                    rank,
                    case,
                    [full_hidden[rank][case], hidden, physical, total],
                    {"hidden": hidden},
                    "hc_post",
                )
                execute(
                    "pre_unpack",
                    rank,
                    case,
                    [full_pre[rank][case], pre, physical, total],
                    {"pre": pre},
                    "hc_pre",
                )
                padded_pre = torch.full((capacity, 4), contract["output_sentinel"], dtype=torch.float32)
                padded_pre[:physical].copy_(state["ffn_pre"][:physical])
                result = dict(
                    output_hidden=hidden[:total].reshape(total, 4, 5120).clone(),
                    next_pre=pre[:total].clone(),
                    local_output_hidden=state["padded_hidden"][:physical].clone(),
                    local_next_pre=state["ffn_pre"][:physical].clone(),
                    padded_hidden=state["padded_hidden"],
                    padded_next_pre=padded_pre,
                    cache_after=state["cache"],
                    nodes=state["nodes"],
                    received_routes=state["received_routes"],
                    received_hidden=state["received_hidden"],
                    returned_routes=state["returned_routes"],
                    metadata=original["metadata"],
                    cp_token_range=original["cp_token_range"],
                    cache_mutable_slots=original.get("cache_mutable_slots", []),
                )
                if layer_id in (2, 3, 14, 15, 19, 20, 21, 24, 25):
                    result["cache_after"] = state["cache"].clone()
                    result.update(
                        {
                            name: state[name]
                            for name in (
                                "raw_cache_before",
                                "raw_cache_after",
                                "shared_state_before",
                                "shared_state_after",
                            )
                        }
                    )
                    result.update(
                        {name: original[name] for name in ("active_by_dp", "cache_layout", "cache_metadata")}
                    )
                if state["engram"] is not None:
                    result.update(
                        hash_cache_before=state["engram"]["hash_cache_before"],
                        hash_cache_after=state["engram"]["hash_cache_after"],
                    )
                readonly = dict(
                    hidden=state["readonly_hidden"][:physical],
                    pre_mix=state["incoming"][:physical],
                    input_ids=state["token_ids"][:physical],
                    positions=state["positions"][:total],
                    full_input_ids=original["full_input_ids"],
                )
                for key, value in readonly.items():
                    if not torch.equal(value, original[key]):
                        raise RuntimeError("Model compute modified readonly " + key)
                    result["readonly_" + key] = value.clone()
                torch.save(result, args.output / f"rank-{rank}-{case}.pt")
        runtime["complete_layer_status"] = "executed_short_gate"
    finally:
        for rank, report in enumerate(reports):
            report["consumed_tensors"] = readers[rank].consumed
            transport = [receipt["memory"] for receipt in report["communication"]]
            intervals = [item for item in memory.intervals if item["rank"] == rank]
            intervals += [item for receipt in transport for item in receipt.get("intervals", [])]
            report["memory"] = {
                "scope": "Stage samples; qualified peaks are computed by the frozen observer",
                "samples": [item for item in memory.samples if item["rank"] == rank],
                "transport": transport,
                "intervals": intervals,
            }
            write_json(args.output / f"rank-{rank}-report.json", report)
        loaded_source_files(runtime)
        write_json(runtime_path, runtime)


def compile_prefill_layer(args, layer_id=0):
    """Compile the complete physical-layer entries through Golden without a device."""
    from golden import normalize_config

    from .engram import make_prefill_engram_entries
    from .runtime_utils import file_sha, loaded_source_files, write_json

    contract = torch.load(args.inputs, map_location="cpu", weights_only=True)
    boundary_layer = contract.get("physical_layer_id", contract.get("layer", layer_id))
    if boundary_layer != layer_id:
        raise ValueError("Compile the same physical boundary as the product")
    config = dict(contract["model_config"]["text_config"])
    image_id = contract["model_config"]["image_token_id"]
    config.update(image_sentinel_base_id=image_id, image_token_id=image_id)
    config["image_pad_token_id"] = contract["model_config"].get("image_pad_token_id", image_id + 1)
    capacity = contract["capacity"]
    originals = [original for records in contract["cases"].values() for original in records]
    max_tokens = max(
        sum(original["SP_valid_rows"] for original in records) for records in contract["cases"].values()
    )
    expert_capacity = (max_tokens + 15) // 16 * 16
    cache_pages, requests, table_pages = contract["cache_pages"], 1, 2
    if layer_id in (2, 3, 14, 15, 19, 20, 21, 24, 25):
        from .prefill_compressed_runtime import compressed_geometry

        cache_pages, requests, table_pages = compressed_geometry(contract)
    entries = make_prefill_layer_entries(
        capacity,
        config,
        expert_capacity,
        layer_id,
        cache_pages,
        requests,
        table_pages,
        ep=contract["topology"]["ep"],
        cache_layout=originals[0].get("cache_layout"),
        tp=args.tp,
    )
    if layer_id in (1, 14):
        from .prefill_engram_runtime import engram_owner

        engram_owner(contract, layer_id)
        cache_rows = originals[0]["hash_cache"].numel()
        max_blocks = max(original["metadata"]["block_table"].shape[1] for original in originals)
        engram_args = (
            capacity,
            contract["hash_constants"],
            config,
            contract["engram_shards"],
            cache_rows,
            max_blocks,
        )
        entries.update(make_prefill_engram_entries(*engram_args, tp=args.tp))
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "physical_layer_id": layer_id,
        "device_execution": "not_run",
        "entries": {},
        "source_root": str(Path(__file__).resolve().parents[2]),
        "source_files": {},
        "inputs_sha256": file_sha(args.inputs),
    }
    try:
        for name, function in entries.items():
            directory = args.output / name
            config = normalize_config({"platform": "a2a3", "save_kernels_dir": str(directory)})
            function.compile(config=config)
            artifacts = {
                str(path): file_sha(path)
                for path in sorted(directory.rglob("*"))
                if path.is_file() and path.suffix in (".pto", ".cpp", ".so")
            }
            if not any(path.endswith(".pto") for path in artifacts):
                raise RuntimeError("No retained PTO for " + name)
            report["entries"][name] = {"compilation": "passed", "artifacts": artifacts}
            print({"stage": "compile_only", "entry": name, "status": "passed"}, flush=True)
    finally:
        loaded_source_files(report)
        write_json(args.output / "compilation.json", report)


def main():
    """Select and execute a complete physical-layer boundary or compile its entries."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument(
        "--layer-id", type=int, choices=(0, 1, 2, 3, 14, 15, 19, 20, 21, 24, 25), required=True
    )
    parser.add_argument("--tp", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--devices", type=lambda value: value.split(","), required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.compile_only:
        compile_prefill_layer(args, layer_id=args.layer_id)
    else:
        run_prefill_layer(args, layer_id=args.layer_id)


if __name__ == "__main__":
    main()

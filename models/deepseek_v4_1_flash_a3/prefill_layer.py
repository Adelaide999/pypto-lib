# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Complete-layer composition and L2 regions over device activation descriptors."""

import keyword
import linecache
import sys
from hashlib import sha256
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"
import pypto.language as pl
import torch
from pypto import ir, passes
from pypto.ir.compiled_program import extract_param_infos
from .utils import allocate_runtime_tensor, copy_runtime_tensor, read_runtime_control
from .runtime_config import MAX_PREFILL_CHUNK_SIZE


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
    moe_slab_rows=None,
):
    """Construct shared physical-layer attention, HC and MoE computations."""
    from .hc import make_hc_post_inline, make_hc_pre_inline
    from .prefill_moe import make_moe_combine_from_dispatch
    from .prefill_moe import make_ep_dispatch_pack, make_ep_dispatch_unpack, make_ep_return_pack
    from .prefill_moe import make_moe_router
    from .prefill_moe import make_shared_moe_w8a8
    from .grouped_moe import make_grouped_routed_experts
    from .rope import make_rope_tables
    from .input_pack import make_head_pack, make_sp_gather_unpack, make_sp_masked_zero, make_sp_reduce_pack
    from .rmsnorm import make_prefill_norm_cast
    from .cache import resolve_layer_role
    from .prefill_swa import make_builders
    from .runtime_config import validate_tp

    role = resolve_layer_role(layer_id, role)
    tp = validate_tp(tp)
    if tp not in (1, 2, 4) or capacity % 16 or not 2 <= ep <= 16 or ep % tp or 384 % ep:
        raise ValueError("Require TP1/2/4 and an aligned complete Prefill layer")
    slab = capacity if moe_slab_rows is None else moe_slab_rows
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
        router=make_moe_router(capacity, config, pl.FP32),
        shared=make_shared_moe_w8a8(
            capacity, config, fused_activation=True, fused_channel_dtype=pl.FP32, partitions=tp,
        ),
        routed=make_grouped_routed_experts(expert_capacity, config, 384 // ep),
        dispatch_pack=make_ep_dispatch_pack(slab, 5120, 384, 6, ep),
        dispatch_unpack=make_ep_dispatch_unpack(slab, 5120, 384, 6, ep, expert_capacity),
        return_pack=make_ep_return_pack(slab, 5120, 384, 6, ep, expert_capacity),
        combine=make_moe_combine_from_dispatch(slab, 5120, 6, ep),
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
    from .prefill_moe import DISPATCH_MAP_PAD as MAP_PAD, validate_received_routes, validate_returned_routes
    from .prefill_moe import DISPATCH_SOURCE_MAP_PAD
    from .prefill_compressed_runtime import run_compressed_attention
    from .prefill_consumer_runtime import run_consumer_attention
    from .cache import invalidate_live_versions, publish_live_versions
    from .cache import resolve_layer_role, validate_live_state
    from .runtime_config import ExpertParallelConfig
    from .utils import window_schedule_from_controls as from_controls

    layer_id = role["physical_layer"]
    role = resolve_layer_role(layer_id, role)
    topology = runtime["topology"]
    layout = ExpertParallelConfig(topology["dp"], topology["tp"], topology["ep"], 384)
    tp, world = layout.tp, layout.ep
    map_ranks = runtime.get("map_ranks", lambda function: [function(rank) for rank in range(world)])
    if tp not in (1, 2, 4) or not 2 <= world <= 16:
        raise ValueError("Require matching TP1/2/4 with real multi-rank EP before building the layer")
    if any(topology[name] != 1 for name in ("pp", "pcp", "dcp")):
        raise ValueError("Require PP1/PCP1/DCP1")
    if not runtime["enable_expert_parallel"] or not all(runtime["required_modes"].values()):
        raise ValueError("Original parallel and model modes must remain enabled")
    validate_prefill_ownership(layout, runtime["ownership"])
    capacity, expert_capacity = runtime["capacity"], runtime["expert_capacity"]
    if not 0 < capacity <= MAX_PREFILL_CHUNK_SIZE or capacity % 16 or expert_capacity <= 0 or expert_capacity % 16:
        raise ValueError("The current complete core requires bounded positive 16-row capacities")
    if any(len(records) != world for records in batch["cases"].values()):
        raise ValueError("Every batch must cover the actual complete world")
    if any(len(values) != world for values in (local_hidden, local_pre, cache_state)):
        raise ValueError("Local activation and state lists must cover the complete world")
    epoch, token_map_id = batch["epoch"], batch["token_map_id"]
    if epoch != runtime["epoch"] or token_map_id != runtime["token_map_id"]:
        raise ValueError("Layer runtime and batch must identify the same chunk and token map")
    for case, records in batch["cases"].items():
        def validate_rank_state(rank):
            original = records[rank]
            physical, valid = original["SP_physical_rows"], original["SP_valid_rows"]
            if not 0 <= valid <= physical <= capacity or len(original["positions"]) > capacity:
                raise ValueError("Invalid actual physical/valid row bounds")
            hidden, pre = local_hidden[rank][case], local_pre[rank][case]
            if tuple(hidden.shape) != (capacity, 4, 5120) or hidden.dtype != torch.bfloat16:
                raise ValueError("Require local padded BF16 hidden [capacity,4,5120]")
            if tuple(pre.shape) != (capacity, 4) or pre.dtype != torch.float32:
                raise ValueError("Require local delayed FP32 HC pre [capacity,4]")
            if hidden.device.type != "cpu" or pre.device.type != "cpu":
                raise ValueError("Layer descriptors require CPU shape and storage metadata")
            live = cache_state[rank][case]
            checked_live = live
            if role["candidate_filter"]:
                active = original["actual_local_query_rows"]
                candidates = read_runtime_control(runtime, live["shared"]["candidates"][:active], rank)
                checked_live = dict(live, shared=dict(live["shared"], candidates=candidates))
            validate_live_state(role, original, checked_live, epoch, token_map_id)
        map_ranks(validate_rank_state)
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

    def zeros(rank, *shape, dtype=torch.bfloat16):
        return allocate_runtime_tensor(runtime, shape, dtype, rank)

    def pad(value, rank):
        target = torch.zeros((capacity, *value.shape[1:]), dtype=value.dtype)
        copy_runtime_tensor(runtime, target[:len(value)], value, rank)
        return target

    local_experts, head_width, output_width = 384 // world, 32768 // tp, 8192 // tp
    local_groups = 8 // tp
    states = [{} for _ in range(world)]
    def invalidate_rank_state(rank):
        for case in batch["cases"]:
            invalidate_live_versions(role, cache_state[rank][case])
    map_ranks(invalidate_rank_state)
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
        def normalize_rank(rank):
            original = originals[rank]
            physical, valid = original["SP_physical_rows"], original["SP_valid_rows"]
            total = len(original["positions"])
            if not 0 <= valid <= physical <= capacity or total > capacity:
                raise ValueError("Invalid frozen physical/valid row bounds")
            readonly_hidden = local_hidden[rank][case]
            hidden, incoming = readonly_hidden, local_pre[rank][case]
            engram = None if engram_states is None else engram_states[rank][case]
            if engram is not None:
                hidden = engram["hidden"]
            pre, post = zeros(rank, capacity, 4, dtype=torch.float32), zeros(rank, capacity, 4, dtype=torch.float32)
            comb = zeros(rank, capacity, 4, 4, dtype=torch.float32)
            collapsed, normed = zeros(rank, capacity, 5120), zeros(rank, capacity, 5120)
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
            if total:
                execute("collapse_norm", rank, case, arguments, outputs, ("hc_pre", "attention_norm"))
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
                role=role,
                live=cache_state[rank][case],
                runtime=runtime,
            )
            norm_packets[rank][case] = normed
        map_ranks(normalize_rank)
    gathered = exchange("attention-input-gather", norm_packets, "all_gather", "tp")
    head_packets = [{} for _ in range(world)]
    for case, originals in batch["cases"].items():
        def attend_rank(rank):
            original = originals[rank]
            state = states[rank][case]
            physical, total = state["physical"], state["total"]
            if not total:
                head_packets[rank][case] = zeros(rank, tp * capacity, head_width)
                return
            full_norm = zeros(rank, capacity, 5120)
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
            q_input = pad(full_norm[start : start + query_rows], rank)
            positions = pad(original["positions"], rank)
            cosine, sine = zeros(rank, capacity, 32, dtype=torch.float32), zeros(rank, capacity, 32, dtype=torch.float32)
            execute(
                "rope_tables",
                rank,
                case,
                [positions, cosine, sine, total],
                dict(cosine=cosine, sine=sine),
                "rope",
            )
            q_cos = pad(cosine[start:start + query_rows], rank)
            q_sin = pad(sine[start:start + query_rows], rank)
            wqa, latent, qb, query = (
                zeros(rank, capacity, 1280),
                zeros(rank, capacity, 1280),
                zeros(rank, capacity, 32768),
                zeros(rank, capacity, 32768),
            )
            kv_raw, kv_norm, kv_rope = zeros(rank, capacity, 512), zeros(rank, capacity, 512), zeros(rank, capacity, 512)
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
                inverse = zeros(rank, capacity, 32768)
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
                inverse = zeros(rank, capacity, 32768)
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
                context, inverse = zeros(rank, capacity, 32768), zeros(rank, capacity, 32768)
                execute(
                    "swa_read",
                    rank,
                    case,
                    [query, indices, operands["sink"], cache, context, query_rows],
                    {"context": context},
                    "swa",
                )
            # Head-pack writes every peer bucket, including the inactive rows.
            packet = allocate_runtime_tensor(runtime, (tp * capacity, head_width), torch.bfloat16, rank, fill=None)
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
            state.update(cache=cache, per_rank=per_rank, positions=positions)
            head_packets[rank][case] = packet
        map_ranks(attend_rank)
    exchanged = exchange("attention-head-exchange", head_packets, "all_to_all", "tp")
    reduction = [{} for _ in range(world)]
    for case in batch["cases"]:
        def project_rank(rank):
            state, tp_rank = states[rank][case], rank % tp
            total, physical = state["total"], state["physical"]
            if not total:
                reduction[rank][case] = zeros(rank, tp * capacity, 5120)
                return
            heads, woa, wob = (
                zeros(rank, capacity, head_width),
                zeros(rank, capacity, output_width),
                zeros(rank, capacity, 5120),
            )
            weight_a = operands["wo_a_grouped_prefill"][
                tp_rank * local_groups : (tp_rank + 1) * local_groups
            ].contiguous()
            weight_b = operands["wo_b"][tp_rank * output_width : (tp_rank + 1) * output_width].contiguous()
            # Reduce-pack writes every peer bucket before the collective.
            packed = allocate_runtime_tensor(runtime, (tp * capacity, 5120), torch.bfloat16, rank, fill=None)
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
            reduction[rank][case] = packed
        map_ranks(project_rank)
    reduced = exchange("attention-output-reduce", reduction, "all_reduce", "tp")
    scattered = [{} for _ in range(world)]
    def select_rank_output(rank):
        begin = (rank % tp) * capacity
        for case, value in reduced[rank].items():
            scattered[rank][case] = value[begin : begin + capacity].contiguous()
    map_ranks(select_rank_output)
    slab = runtime.get("moe_slab_rows", capacity)
    routes, packet_rows = slab * 6, world * slab * 6
    for case in batch["cases"]:
        routing_rows = max(original["SP_physical_rows"] for original in batch["cases"][case])
        def route_rank(rank):
            state = states[rank][case]
            physical = state["physical"]
            buffers = runtime.get("layer_buffers")
            if buffers is None:
                output = torch.full((capacity, 4, 5120), runtime["output_sentinel"], dtype=torch.bfloat16)
                pre = zeros(rank, capacity, 4, dtype=torch.float32)
            else:
                output, pre = buffers(rank, case)
            if not state["valid"]:
                if physical:
                    empty = zeros(rank, physical, 4, 5120)
                    copy_runtime_tensor(runtime, output[:physical], empty, rank)
                state.update(output_hidden=output, ffn_pre=zeros(rank, capacity, 4, dtype=torch.float32))
                return
            changed = zeros(rank, capacity, 4, 5120)
            post = zeros(rank, capacity, 4, dtype=torch.float32)
            comb, collapsed = zeros(rank, capacity, 4, 4, dtype=torch.float32), zeros(rank, capacity, 5120)
            normed, router_input = zeros(rank, capacity, 5120), zeros(rank, capacity, 5120, dtype=torch.float32)
            logits = zeros(rank, capacity, 384, dtype=torch.float32)
            weights = zeros(rank, capacity, 6, dtype=torch.float32)
            ids = zeros(rank, capacity, 6, dtype=torch.int32)
            token_ids = pad(state["original"]["input_ids"], rank)
            gate = raw[f"{prefix}.ffn.gate.weight"].float()
            execute_region(runtime, "ffn_dispatch_l2", rank, case, [
                ("hc_post", [scattered[rank][case], state["hidden"], state["post"], state["comb"],
                             changed, physical], {"changed": changed}, "hc_post"),
                ("hc_pre", [changed, raw[f"{prefix}.hc_ffn_fn"], raw[f"{prefix}.hc_ffn_scale"],
                            raw[f"{prefix}.hc_ffn_base"], state["pre"], pre, post, comb, collapsed, physical],
                 dict(pre=pre, post=post, comb=comb, collapsed=collapsed), "hc_pre"),
                ("norm_cast", [collapsed, raw[f"{prefix}.ffn_norm.weight"], normed, router_input, physical],
                 dict(normed=normed, router_input=router_input), "ffn_norm"),
                ("router", [router_input, gate, raw[f"{prefix}.ffn.gate.bias"], raw[f"{prefix}.ffn.gate.bias_vl"],
                            token_ids, logits, ids, weights, routing_rows],
                 dict(logits=logits, route_ids=ids, route_weights=weights), "router"),
            ])
            host_route_ids = read_runtime_control(runtime, ids[:state["valid"]], rank)
            state.update(changed=changed, ffn_pre=pre, ffn_post=post, ffn_comb=comb, weights=weights,
                         route_ids=ids, host_route_ids=host_route_ids, normed=normed,
                         shared=zeros(rank, capacity, 5120), output_hidden=output,
                         combined=zeros(rank, capacity, 5120))
        map_ranks(route_rank)
    slab_work = [{} for _ in range(world)]
    def allocate_rank_slab(rank):
        for case in batch["cases"]:
            slab_work[rank][case] = dict(
                routes=allocate_runtime_tensor(runtime, (packet_rows, 4), torch.int64, rank, fill=-1),
                hidden=zeros(rank, packet_rows, 5120),
                counts=zeros(rank, world, MAP_PAD, dtype=torch.int32),
                source_map=zeros(rank, routes, DISPATCH_SOURCE_MAP_PAD, dtype=torch.int32),
                recv_routes=allocate_runtime_tensor(runtime, (packet_rows, 4), torch.int64, rank, fill=-1),
                recv_hidden=zeros(rank, packet_rows, 5120),
                expert_x=zeros(rank, local_experts * expert_capacity, 5120),
                expert_y=zeros(rank, local_experts * expert_capacity, 5120),
                expert_counts=zeros(rank, local_experts, dtype=torch.int64),
                packet_map=allocate_runtime_tensor(runtime, (packet_rows, MAP_PAD), torch.int32, rank, fill=-1),
                total=zeros(rank, MAP_PAD, dtype=torch.int32), sent=zeros(rank, packet_rows, 5120), routed=zeros(rank, slab, 5120),
                returned_routes=allocate_runtime_tensor(runtime, (routes, 4), torch.int64, rank, fill=-1),
            )
    map_ranks(allocate_rank_slab)
    max_physical = max(record["SP_physical_rows"] for records in batch["cases"].values() for record in records)
    rounds = max(1, (max_physical + slab - 1) // slab)
    for round_id in range(rounds):
        base = round_id * slab
        valid_by_case = {
            case: [max(0, min(slab, state[case]["valid"] - base)) for state in states]
            for case in batch["cases"]
        }
        dispatch_packets = [{} for _ in range(world)]
        def pack_rank_slab(rank):
            for case in batch["cases"]:
                state, work = states[rank][case], slab_work[rank][case]
                packet = {name: work[name] for name in ("routes", "hidden", "counts")}
                valid = valid_by_case[case][rank]
                if state["valid"]:
                    arguments = [state["normed"][base:base + slab], state["route_ids"][base:base + slab],
                                 packet["routes"], packet["hidden"], packet["counts"], work["source_map"], valid, rank]
                    execute("dispatch_pack", rank, case, arguments, dict(packet, source_map=work["source_map"]),
                            "dispatch_pack")
                dispatch_packets[rank][case] = packet
        map_ranks(pack_rank_slab)
        pending = None
        if round_id == 0 and "exchange_async" in runtime:
            stage = namespace + f"-expert-dispatch-r{round_id}"
            pending = runtime["exchange_async"](stage, dispatch_packets, "all_to_all", "ep", packets=True)
        else:
            received = exchange(f"expert-dispatch-r{round_id}", dispatch_packets, "all_to_all", "ep", packets=True)
        if round_id == 0:
            try:
                def execute_rank_shared(rank):
                    for case in batch["cases"]:
                        state = states[rank][case]
                        if not state["valid"]:
                            continue
                        shared_operands = expert_weights(rank, None)
                        arguments = [state["normed"], *shared_operands, state["shared"], state["physical"]]
                        execute("shared", rank, case, arguments, {"shared": state["shared"]}, "shared")
                map_ranks(execute_rank_shared)
            finally:
                if pending is not None:
                    received = pending.result()
        returns = [{} for _ in range(world)]
        def execute_rank_experts(rank):
            for case, packet in received[rank].items():
                work = slab_work[rank][case]
                controls = {name: read_runtime_control(runtime, packet[name], rank) for name in ("routes", "counts")}
                expected_total = validate_received_routes(
                    controls, rank, layout, routes, valid_by_case[case], expert_capacity=expert_capacity,
                )
                returns[rank][case] = dict(routes=packet["routes"], hidden=work["sent"], counts=packet["counts"])
                if expected_total == 0:
                    continue
                unpack_args = [packet["routes"], packet["hidden"], packet["counts"], work["recv_routes"],
                               work["recv_hidden"], work["expert_x"], work["expert_counts"], work["packet_map"],
                               work["total"], rank]
                with memory.interval(rank, "load", "grouped-experts"):
                    if "grouped_expert" in weight_provider:
                        expert_operands = weight_provider["grouped_expert"](rank)
                    else:
                        local_values = [expert_weights(rank, rank * local_experts + local)
                                        for local in range(local_experts)]
                        expert_operands = tuple(torch.cat([values[i] for values in local_values]) for i in range(4))
                expert_args = [work["expert_x"], *expert_operands, work["expert_counts"], work["expert_y"]]
                return_args = [work["expert_y"], packet["counts"], work["packet_map"], work["sent"]]
                execute_region(runtime, "moe_experts_l2", rank, case, [
                    ("dispatch_unpack", unpack_args, {}, "dispatch_unpack"),
                    ("routed", expert_args, {"expert_output": work["expert_y"]}, "routed"),
                    ("return_pack", return_args, {"return_hidden": work["sent"]}, "return_pack"),
                ])
                if int(read_runtime_control(runtime, work["total"], rank)[0]) != expected_total:
                    raise RuntimeError("Dispatch unpack lost received routes")
                counts = read_runtime_control(runtime, work["expert_counts"], rank)
                if bool(((counts < 0) | (counts > expert_capacity)).any()):
                    raise RuntimeError("Expert slab exceeded its proven per-round bound")
        map_ranks(execute_rank_experts)
        returned = exchange(f"expert-return-r{round_id}", returns, "all_to_all", "ep", packets=True)
        def combine_rank_slab(rank):
            for case, packet in returned[rank].items():
                state, work = states[rank][case], slab_work[rank][case]
                valid = valid_by_case[case][rank]
                physical = max(0, min(slab, state["physical"] - base))
                if valid == 0:
                    counts = read_runtime_control(runtime, packet["counts"], rank)
                    if bool((counts[:, 0] != 0).any()):
                        raise RuntimeError("HCCL returned routes to an empty source slab")
                    continue
                controls = {name: read_runtime_control(runtime, packet[name], rank) for name in ("routes", "counts")}
                route_ids = state["host_route_ids"][base:base + valid]
                validate_returned_routes(controls, rank, routes, route_ids, valid)
                # The return exchange preserves row order within each dispatch bucket.
                arguments = [packet["routes"], packet["hidden"], work["source_map"],
                             state["weights"][base:base + slab], state["shared"][base:base + slab], work["routed"],
                             state["combined"][base:base + slab], work["returned_routes"], physical, valid]
                execute("combine", rank, case, arguments, {}, "combine")
        map_ranks(combine_rank_slab)
        if runtime.get("release_tensors") is not None:
            runtime["release_tensors"]([
                (rank, value)
                for packets in (received, returned)
                for rank, cases in enumerate(packets)
                for packet in cases.values()
                for value in packet.values()
            ])
    final_hidden, final_pre = [{} for _ in range(world)], [{} for _ in range(world)]
    def finish_rank_moe(rank):
        for case in batch["cases"]:
            state = states[rank][case]
            if not state["valid"]:
                final_hidden[rank][case], final_pre[rank][case] = state["output_hidden"], state["ffn_pre"]
                state["padded_hidden"] = state["output_hidden"]
                continue
            valid, physical = state["valid"], state["physical"]
            combined, output = state["combined"], state["output_hidden"]
            execute_region(runtime, "moe_finish_l2", rank, case, [
                ("mask_moe", [combined, valid, physical], {"combined": combined}, "combine"),
                ("hc_post", [combined, state["changed"], state["ffn_post"], state["ffn_comb"], output, physical],
                 {"hidden": output}, "hc_post", case + "-ffn"),
            ])
            output_flat = output.reshape(capacity, 20480)
            execute("mask_output", rank, case, [output_flat, valid, physical], {"hidden": output}, "hc_post")
            state.update(padded_hidden=output)
            final_hidden[rank][case], final_pre[rank][case] = output, state["ffn_pre"]
    map_ranks(finish_rank_moe)
    def publish_rank_state(rank):
        for case in batch["cases"]:
            publish_live_versions(role, cache_state[rank][case], epoch, token_map_id)
    map_ranks(publish_rank_state)
    return {"hidden": final_hidden, "pre_mix": final_pre, "states": states}


# Each binding names one external tensor or scalar, shared by all uses in a region.
REGIONS = {
    "input_l2": (
        ("reduce", ("gathered", "embedding", "active")),
        ("initial_hc", ("embedding", "streams", "premix", "tp_rank", "physical", "active")),
    ),
    "head_l2": (
        ("sp_unpack", ("gathered", "collapsed", "physical", "input_active")),
        ("norm", ("collapsed", "norm_weight", "normalized", "active")),
        ("projection", ("normalized", "weight", "local_logits", "active")),
    ),
    "qkv_l2": (
        ("wq_a", ("q_input", "wq_a", "wq_a_scale", "wqa", "query_rows")),
        ("q_norm", ("wqa", "q_norm", "latent", "query_rows")),
        ("wq_b", ("latent", "wq_b", "wq_b_scale", "qb", "query_rows")),
        ("q_rope", ("qb", "q_cos", "q_sin", "query", "query_rows")),
        ("wkv", ("full_norm", "wkv", "wkv_scale", "kv_raw", "total")),
        ("kv_norm", ("kv_raw", "kv_norm", "kv_normalized", "total")),
        ("kv_rope", ("kv_normalized", "cosine", "sine", "kv_rope", "total")),
    ),
    "attention_pack_l2": (
        ("inverse_rope", ("context", "q_cos", "q_sin", "inverse", "query_rows")),
        ("head_pack", ("inverse", "packet", "query_rows")),
    ),
    "output_projection_l2": (
        ("head_unpack", ("exchanged", "heads", "per_rank", "total")),
        ("grouped_wo_a", ("heads", "weight_a", "woa", "total")),
        ("wo_b", ("woa", "weight_b", "wob", "total")),
        ("reduce_pack", ("wob", "packed", "physical", "total")),
    ),
    "ffn_dispatch_l2": (
        ("hc_post", ("scattered", "hidden", "attn_post", "attn_comb", "changed", "physical")),
        (
            "hc_pre",
            (
                "changed",
                "hc_fn",
                "hc_scale",
                "hc_base",
                "attn_pre",
                "ffn_pre",
                "ffn_post",
                "ffn_comb",
                "collapsed",
                "physical",
            ),
        ),
        ("norm_cast", ("collapsed", "ffn_norm", "normed", "router_input", "physical")),
        (
            "router",
            (
                "router_input",
                "gate",
                "bias",
                "bias_vl",
                "token_ids",
                "logits",
                "ids",
                "weights",
                "routing_rows",
            ),
        ),
    ),
    "moe_experts_l2": (
        ("dispatch_unpack", ("packet_routes", "packet_hidden", "packet_counts", "recv_routes", "recv_hidden",
                             "expert_x", "expert_counts", "packet_map", "received_total", "owner_rank")),
        ("routed", ("expert_x", "gate_up_weight", "gate_up_scale", "down_weight", "down_scale",
                    "expert_counts", "expert_y")),
        ("return_pack", ("expert_y", "packet_counts", "packet_map", "sent_hidden")),
    ),
    "moe_finish_l2": (
        ("mask_moe", ("combined", "valid", "physical")),
        ("hc_post", ("combined", "changed", "post", "comb", "output", "physical")),
    ),
}

STAGE_REGIONS = {
    "embedding": ("input_l2",),
    "output": ("head_l2",),
    "layer": ("qkv_l2", "attention_pack_l2", "output_projection_l2", "ffn_dispatch_l2",
              "moe_experts_l2", "moe_finish_l2"),
}


def region_bindings(steps):
    """Return the stable external ABI, including intermediate observation buffers."""
    return tuple(dict.fromkeys(binding for _, bindings in steps for binding in bindings))


def _parameter_type(param, capacity):
    """Bind only the existing leading dynamic token dimension to the network capacity."""
    shape = (
        None
        if param.shape is None
        else tuple(capacity if size == -1 and axis == 0 else size for axis, size in enumerate(param.shape))
    )
    return shape, str(param.dtype), param.layout


def _annotation(param, writable, capacity):
    """Serialize a specialized child parameter without changing its storage type."""
    names = {
        "bfloat16": "BF16",
        "fp16": "FP16",
        "fp32": "FP32",
        "int8": "INT8",
        "uint8": "UINT8",
        "int16": "INT16",
        "int32": "INT32",
        "int64": "INT64",
    }
    dtype = names.get(str(param.dtype))
    if dtype is None:
        raise ValueError("Unsupported A3 L2 parameter dtype: " + str(param.dtype))
    if param.shape is None:
        if writable:
            raise ValueError("An L2 scalar cannot be writable")
        return f"pl.Scalar[pl.{dtype}]"
    shape, _, _ = _parameter_type(param, capacity)
    if any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError("L2 regions require the exact positive specialized child shape")
    if param.layout not in ("ND", "NZ"):
        raise ValueError("Unsupported A3 L2 parameter layout: " + str(param.layout))
    layout = ", pl.NZ" if param.layout == "NZ" else ""
    annotation = f"pl.Tensor[{list(shape)}, pl.{dtype}{layout}]"
    return "pl.InOut[" + annotation + "]" if writable else annotation


def compose_l2(name, definitions, steps, capacity):
    """Inline the same child bodies and thread every returned tensor alias in call order.

    The generated body contains only calls and return bindings. It introduces no
    arithmetic, casts, reductions, quantization or cache-layout transformations.
    """
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("L2 regions require an explicit positive token capacity")
    bindings = region_bindings(steps)
    if not steps or any(not value.isidentifier() or keyword.iskeyword(value) for value in (name, *bindings)):
        raise ValueError("L2 regions require safe Python identifiers and at least one computation")
    namespace = {"pl": pl}
    params, writable, calls = {}, set(), []
    for index, (child_name, arguments) in enumerate(steps):
        child = definitions[child_name]
        program = child.specialize()
        infos, _, returns = extract_param_infos(program)
        if len(infos) != len(arguments):
            raise ValueError("L2 child arity differs: " + child_name)
        alias_program = passes.convert_to_ssa()(program)
        functions = [
            item for item in alias_program.functions.values() if item.func_type == ir.FunctionType.Orchestration
        ]
        if len(functions) != 1:
            raise ValueError("L2 children require one unambiguous orchestration entry")
        aliases = ir.returned_param_indices(functions[0], alias_program)
        if len(aliases) != len(returns) or any(
            type(slot) is not int or not 0 <= slot < len(infos) for slot in aliases
        ):
            raise ValueError("L2 child returns must alias actual arguments: " + child_name)
        for slot, (binding, info) in enumerate(zip(arguments, infos)):
            signature = _parameter_type(info, capacity)
            if binding in params:
                prior = params[binding]
                if signature != _parameter_type(prior, capacity):
                    raise ValueError("L2 binding has incompatible child types: " + binding)
            params[binding] = info
            if info.direction.name != "In" or slot in aliases:
                writable.add(binding)
        symbol = f"child_{index}"
        namespace[symbol] = pl.jit.inline(child._func)
        invocation = f"{symbol}({', '.join(arguments)})"
        targets = [arguments[slot] for slot in aliases]
        calls.append("    " + (", ".join(targets) + " = " if targets else "") + invocation)
    signature = [
        f"    {binding}: {_annotation(params[binding], binding in writable, capacity)},"
        for binding in bindings
    ]
    results = [binding for binding in bindings if binding in writable]
    body = ["@pl.jit", f"def {name}(", *signature, "):", *calls, "    return " + ", ".join(results)]
    source = "\n".join(body) + "\n"
    filename = f"<ds41-prefill-l2-{sha256(source.encode()).hexdigest()}>"
    # PyPTO reads the function source through inspect, including generated orchestration.
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    exec(compile(source, filename, "exec"), namespace)
    return namespace[name]


def make_network_l2_entries(definitions, capacity, stage="layer"):
    """Compose each stage only between its existing host communication boundaries."""
    regions = {name: REGIONS[name] for name in STAGE_REGIONS[stage]}
    covered = {name for steps in regions.values() for name, _ in steps}
    entries = {name: function for name, function in definitions.items() if name not in covered}
    entries.update({name: compose_l2(name, definitions, steps, capacity) for name, steps in regions.items()})
    return entries


def execute_region(runtime, name, rank, case, operations):
    """Run a network L2 region or the same child sequence in independent-layer validation."""
    steps = REGIONS[name]
    if tuple(item[0] for item in operations) != tuple(item[0] for item in steps):
        raise ValueError("L2 call order differs from the shared layer: " + name)
    callback = runtime.get("execute_l2")
    if callback is None:
        for operation in operations:
            child, arguments, outputs, role = operation[:4]
            child_case = operation[4] if len(operation) == 5 else case
            runtime["execute"](child, rank, child_case, arguments, outputs, role)
        return
    arguments, outputs, roles = region_arguments(name, operations)
    callback(name, rank, case, arguments, outputs, roles)


def region_arguments(name, operations):
    """Bind the shared child operands and observations without copying or changing tensors."""
    steps = REGIONS[name]
    if tuple(item[0] for item in operations) != tuple(item[0] for item in steps):
        raise ValueError("L2 call order differs from the shared stage: " + name)
    values, outputs, roles = {}, {}, []
    for operation, (_, bindings) in zip(operations, steps):
        child, arguments, observed, role = operation[:4]
        if len(arguments) != len(bindings):
            raise ValueError("L2 argument count differs: " + child)
        for binding, value in zip(bindings, arguments):
            if binding in values:
                prior = values[binding]
                same = (
                    prior is value
                    if hasattr(value, "shape")
                    else type(prior) is type(value) and prior == value
                )
                if not same:
                    raise ValueError("L2 binding refers to different runtime operands: " + binding)
            values[binding] = value
        outputs.update({child + "." + key: value for key, value in observed.items()})
        roles.extend([role] if isinstance(role, str) else role)
    return [values[binding] for binding in region_bindings(steps)], outputs, tuple(roles)

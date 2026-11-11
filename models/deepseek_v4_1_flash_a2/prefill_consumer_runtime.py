# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Schedule consumer SWA publication and attention over its readonly source state."""

import torch

from .prefill_compressed_runtime import physical_cache_views


def run_consumer_attention(state, query, operands, execute, rank, case,
                           capacity, cache_blocks, requests, table_pages, layer_id=3, *,
                           q_input=None, latent=None, cosine=None, sine=None):
    """Publish computed own KV and read the supplied source selection and physical caches."""
    original, total, nodes = state["original"], state["total"], state["nodes"]
    start, end, _per_rank, _padded = original["cp_token_range"]
    query_rows = max(0, min(end, total) - min(start, total))
    if query_rows != original["actual_local_query_rows"]:
        raise ValueError("Consumer CP rows differ from the original source boundary")
    role = state["role"]
    source_id = role["kv_source"]
    if role["physical_layer"] != layer_id or role["mode"] not in ("reuse", "reindex"):
        raise ValueError("Require the original physical consumer role")
    metadata = original["cache_metadata"]
    if metadata["physical_layer_id"] != layer_id or metadata["source_physical_layer_id"] != source_id:
        raise ValueError("Consumer and readonly cache source differ from the requested physical layer")
    source_prefix = f"model.layers.{source_id}.self_attn."
    own_prefix = f"model.layers.{layer_id}.self_attn."
    global_meta = {name: item["global_metadata"] for name, item in metadata["actual_metadata"].items()}
    live = state["live"]
    raw = live["raw"]
    views = physical_cache_views(raw, live["layout"])
    own_name, long_name = own_prefix + "swa_cache", source_prefix + "long_kv_cache"
    own_descriptor = live["layout"]["views"][own_name]
    long_descriptor = live["layout"]["views"][long_name]
    network = live.get("network_geometry")
    if network is not None:
        from .prefill_network_cache import load_network_index, network_cache_geometry, owned_cache_pages
        from .prefill_network_cache import raw_cache_matrix, validate_network_sparse_indices

        if network != network_cache_geometry(live["layout"], role, cache_blocks):
            raise ValueError("Network consumer geometry differs from its actual source and own cache aliases")
        source_pages = owned_cache_pages(original, long_name, network["long"])
        own_pages = owned_cache_pages(original, own_name, network["own"])
        if (network["long"]["storage_key"] == network["own"]["storage_key"]
                and set(source_pages).intersection(own_pages)):
            raise ValueError("Consumer source and own-SWA owner pages overlap in their shared arena")
    elif own_descriptor["storage_key"] != long_descriptor["storage_key"]:
        raise ValueError("Consumer attention requires the original shared physical cache backing")
    if network is not None:
        source_cache = raw_cache_matrix(raw, network["long"])
        cache = raw_cache_matrix(raw, network["own"])
    elif role["ratio"] == 1:
        from .prefill_cache_layout import physical_consumer_geometry

        geometry = physical_consumer_geometry(live["layout"], own_prefix, source_prefix, cache_blocks)
        raw_cache = raw[geometry["storage_key"]].view(torch.bfloat16).reshape(cache_blocks, geometry["stride"])
        dense = torch.empty((cache_blocks, geometry["payload"]), dtype=torch.bfloat16)
        execute("consumer_cache_load", rank, case, [raw_cache, dense], {"payload": dense}, "physical_cache_load")
        cache = dense.reshape(cache_blocks, 128, 1, 512)
    else:
        cache = raw[own_descriptor["storage_key"]].view(torch.bfloat16).reshape(cache_blocks, 128, 1, 512)
    if views[own_name].dtype != torch.bfloat16 or views[long_name].dtype != torch.bfloat16:
        raise ValueError("Consumer caches require the original BF16 physical views")

    def pad(value, fill=0):
        result = torch.full((capacity, *value.shape[1:]), fill, dtype=value.dtype)
        result[:len(value)].copy_(value)
        return result

    def table(name):
        value = global_meta[name]["block_table"]
        target = torch.full((requests, table_pages), -1, dtype=torch.int32)
        target[:value.shape[0], :value.shape[1]].copy_(value)
        return target

    shared = live["shared"]
    slots = global_meta[own_name]["slot_mapping"]
    publish = "network_swa_publish" if network is not None else "cache_publish"
    execute(publish, rank, case, [state["kv_rope"], pad(slots.to(torch.int64), -1), cache, total],
            dict(cache=cache), "cache_publish")
    request_ids = torch.full((capacity,), -1, dtype=torch.int32)
    starts = original["metadata"]["query_start_loc"].tolist()
    for request, (begin, finish) in enumerate(zip(starts, starts[1:])):
        request_ids[begin:finish].fill_(request)
    positions = pad(original["positions"][start:start + query_rows])
    query_requests = pad(request_ids[start:start + query_rows], -1)
    if role["mode"] == "reindex":
        from .prefill_cache_layout import physical_index_geometry
        from .prefill_indexer_runtime import run_index_query_selection

        if network is not None:
            index_pages = owned_cache_pages(original, source_prefix + "indexer.k_cache", network["codes"])
            if index_pages != source_pages:
                raise ValueError("Refresh source index and long cache owner pages differ")
            dense_index = load_network_index(raw, network, index_pages, execute, rank, case)
            codes, scales = dense_index["codes"], dense_index["scales"]
        else:
            geometry = physical_index_geometry(live["layout"], source_prefix, cache_blocks)
            dense_index = {}
            for name, descriptor in geometry.items():
                dtype = torch.int8 if name == "codes" else torch.float16
                source = raw[descriptor["storage_key"]].view(dtype).reshape(cache_blocks, descriptor["stride"])
                payload = torch.empty((cache_blocks, descriptor["payload"]), dtype=dtype)
                execute("refresh_cache_load_" + name, rank, case, [source, payload], {"payload": payload}, "physical_cache_load")
                dense_index[name] = payload
            codes = dense_index["codes"].reshape(cache_blocks, 128, 1, 128)
            scales = dense_index["scales"].reshape(cache_blocks, 128, 1, 1)
        selection = run_index_query_selection(state, q_input, latent, cosine, sine, operands,
            execute, rank, case, capacity, requests, table_pages, source_prefix, own_prefix, codes, scales)
        selected, lengths = selection["selected"], selection["lengths"]
        compressed, window = selection["compressed"], selection["window"]
        execute("shared_index_publish", rank, case,
            [selected, lengths, shared["topk_indices"], shared["topk_lengths"], query_rows],
            dict(indices=shared["topk_indices"], lengths=shared["topk_lengths"]), "index_select")
    else:
        source_selected = pad(shared["topk_indices"][:query_rows], -1)
        selected = torch.full((capacity, 512), -1, dtype=torch.int32)
        lengths = torch.zeros((capacity, 1), dtype=torch.int32)
        compressed = torch.full((capacity, 512), -1, dtype=torch.int32)
        window = torch.full((capacity, 128), -1, dtype=torch.int32)
        control_args = [source_selected, positions, query_requests, table(long_name), table(own_name),
                        selected, lengths, compressed, window, query_rows]
        execute("sparse_controls", rank, case, control_args,
                dict(selected=selected, lengths=lengths, compressed=compressed, window=window), "compressed_attention")
    indices = torch.full((capacity, 640), -1, dtype=torch.int32)
    execute("sparse_index_pack", rank, case, [window, compressed, indices, query_rows],
            dict(indices=indices), "compressed_attention")
    context = torch.zeros((capacity, 32768), dtype=torch.bfloat16)
    if network is not None:
        validate_network_sparse_indices(indices, original, network, query_rows)
        execute("network_compressed_attention", rank, case,
                [query, indices, operands["sink"], source_cache, cache, context, query_rows],
                dict(context=context), ("swa", "compressed_attention"))
    else:
        execute("compressed_attention", rank, case, [query, indices, operands["sink"], cache, context, query_rows],
                dict(context=context), ("swa", "compressed_attention"))
    if role["ratio"] == 1 and network is None:
        execute("consumer_cache_store", rank, case, [dense, raw_cache], {"raw": raw_cache}, "physical_cache_store")
    nodes["consumer.selected_indices"] = selected[:query_rows].clone()
    nodes["consumer.own_swa_publication"] = dict(slots=slots.clone(), values=state["kv_rope"][:total].clone())
    state.update(raw_cache_after=raw, shared_state_after=shared)
    return context, views[own_name] if network is not None else cache

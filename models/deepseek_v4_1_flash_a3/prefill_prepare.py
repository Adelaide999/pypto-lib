# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare single-request chunk inputs and persistent paged-cache coordinates."""

import json
from copy import deepcopy
from pathlib import Path
import torch
from .cache import resolve_layer_role
from .runtime_utils import file_sha, write_json


MAX_CONTEXT = 2048
PAGE_ROWS = 128


def tokenize_prompt(inputs, checkpoint, vocabulary):
    """Encode raw completion text or validate explicit original token IDs."""
    if isinstance(inputs, str):
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(Path(checkpoint) / "tokenizer.json"))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        inputs = tokenizer.encode(inputs).ids
    if (not isinstance(inputs, (list, tuple)) or not 1 <= len(inputs) <= MAX_CONTEXT
            or any(type(token) is not int or not 0 <= token < vocabulary for token in inputs)):
        raise ValueError("Require 1..2048 integer token IDs within the checkpoint vocabulary")
    return list(inputs)


def reserve_cache_layouts(templates, context_length):
    """Resize fresh byte arenas and reserve disjoint group pages for the complete prompt."""
    if len(templates) != 16 or not 1 <= context_length <= MAX_CONTEXT:
        raise ValueError("Require all 16 cache-layout templates and a bounded complete prompt")
    pages = (context_length + PAGE_ROWS - 1) // PAGE_ROWS
    layouts, group_pages, maximum = [], [], 0
    for template in templates:
        layout, occupied, allocations, covered = deepcopy(template["layout"]), {}, [], set()
        for group in template["groups"]:
            names = group["names"]
            if not names or covered.intersection(names):
                raise ValueError("Cache groups must partition the original typed cache names")
            covered.update(names)
            keys = {plane["storage_key"] for name in names for plane in
                    (layout["views"][name] if isinstance(layout["views"][name], list) else [layout["views"][name]])}
            first = max((occupied.get(key, 1) for key in keys), default=1)
            owned = list(range(first, first + pages))
            for key in keys:
                occupied[key] = first + pages
            allocations.append(owned)
            maximum = max(maximum, first + pages)
        if covered != set(layout["views"]):
            raise ValueError("Cache groups do not cover the original typed cache layout")
        layouts.append(layout)
        group_pages.append(allocations)
    blocks = maximum + 1
    for layout in layouts:
        strides = {}
        for descriptor in layout["views"].values():
            for plane in descriptor if isinstance(descriptor, list) else [descriptor]:
                size = torch.empty((), dtype=getattr(torch, plane["dtype"].split(".")[-1])).element_size()
                page_bytes = plane["stride"][0] * size
                key = plane["storage_key"]
                if key in strides and strides[key] != page_bytes:
                    raise ValueError("Aliased typed planes disagree on the original byte-page stride")
                strides[key] = page_bytes
                plane["shape"][0] = blocks
                plane["storage_bytes"] = blocks * page_bytes
        for key, raw in layout["raw"].items():
            raw["shape"] = [blocks * strides[key]]
            raw["storage_bytes"] = raw["shape"][0]
    return layouts, group_pages, blocks - 1


def build_chunk_metadata(template, layout, allocations, *, tokens, context, tp_rank, active, reserved):
    """Build CPU controls for SWA, completed compression pairs and per-request ring state."""
    total = len(tokens) if active else 0
    positions = torch.arange(context, context + total, dtype=torch.int64)
    physical = (total + 3) // 4
    start, end = tp_rank * physical, (tp_rank + 1) * physical
    span = [start, end, physical, physical * 4]
    valid = max(0, min(end, total) - min(start, total))
    lengths = torch.tensor([context + total] if active else [], dtype=torch.int32)
    starts = torch.tensor([0, total] if active else [0], dtype=torch.int32)
    metadata, groups = {}, []
    for source, pages in zip(template["groups"], allocations, strict=True):
        table = torch.tensor([pages], dtype=torch.int32) if active else torch.empty((0, len(pages)), dtype=torch.int32)
        raw_slots = [[pages[int(position) // PAGE_ROWS], int(position) % PAGE_ROWS] for position in positions]
        group = dict(group_id=source["group_id"], names=list(source["names"]), block_table=table,
            slot_mapping=torch.tensor(raw_slots, dtype=torch.int64).reshape(total, 2),
            live_block_ids=list(pages) if active else [], reserved_untouched_block_id=reserved)
        groups.append(group)
        for name in source["names"]:
            layer = int(name.split(".")[2])
            role = resolve_layer_role(layer)
            compressed = name.endswith(("long_kv_cache", "indexer.k_cache"))
            ring = name.endswith("compressor.state_cache")
            ratio = role["ratio"] if compressed else 1
            slots = []
            for position in positions.tolist():
                if ring or (position + 1) % ratio:
                    slots.append([-1, -1])
                else:
                    slots.append([pages[position // PAGE_ROWS], position % PAGE_ROWS // ratio])
            global_metadata = dict(block_table=table[:, :1] if ring else table,
                query_start_loc=starts.clone(), seq_lens=lengths.clone(), positions=positions.clone(),
                slot_mapping=torch.tensor(slots, dtype=torch.int64).reshape(total, 2),
                cache_seq_lens=lengths // ratio, cmp_residual=lengths % ratio if ratio == 2 else None,
                num_actual_tokens=total, num_reqs=len(lengths), max_query_len=total,
                max_cache_seq_len=int((lengths // ratio).max()) if active else 0)
            if ring:
                global_metadata["c2_source_positions"] = torch.where(positions % 2 == 1, positions - 1, 0)
            metadata[name] = dict(cp_token_range=span, num_actual_tokens=valid, global_metadata=global_metadata)
    return dict(layout=layout, metadata=metadata, groups=groups, positions=positions, lengths=lengths,
                query_start_loc=starts, hash_constants=template["hash_constants"])


def prepare_token_chunks(inputs, *, checkpoint, cache, config, output, chunk_size=16, dp_rank=0, prepare_engram=False):
    """Yield prepared chunks lazily while retaining a complete fixed page reservation."""
    topology = dict(dp=4, tp=4, ep=16, pp=1, pcp=1, dcp=1)
    if (config["topology"] != topology or type(chunk_size) is not int or not 1 <= chunk_size <= 16
            or type(dp_rank) is not int or not 0 <= dp_rank < 4):
        raise ValueError("Automatic chunks require DP4/TP4/EP16, chunk_size 1..16 and dp_rank 0..3")
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    model = json.loads((checkpoint / "config.json").read_text())
    if model != config["model_config"]:
        raise ValueError("Prepared model configuration differs from the selected checkpoint")
    tokens = tokenize_prompt(inputs, checkpoint, model["text_config"]["vocab_size"])
    layouts, allocations, reserved = reserve_cache_layouts(cache["cache_ABI"], len(tokens))
    identities = {name: file_sha(checkpoint / name) for name in
                  ("config.json", "quant_model_weights.safetensors.index.json", "tokenizer.json")
                  if (checkpoint / name).is_file()}
    output.mkdir(parents=True, exist_ok=False)
    from contextlib import ExitStack

    from .engram_host import HostEngram

    with ExitStack() as stack:
        host = (stack.enter_context(HostEngram(checkpoint, cache["engram_contracts"], output.name))
                if prepare_engram else None)
        predecessor = None
        for epoch, begin in enumerate(range(0, len(tokens), chunk_size)):
            directory = output / f"chunk-{epoch}"
            directory.mkdir()
            current = tokens[begin:begin + chunk_size]
            request = dict(request_slot=0, token_ids=current, position_ids=list(range(begin, begin + len(current))),
                           live_token_mask=[1] * len(current), initial_context_length=begin, history_tokens=tokens[:begin])
            engines = [dict(dp_rank=rank, active=len(current) if rank == dp_rank else 0,
                            requests=[request] if rank == dp_rank else []) for rank in range(4)]
            actual_input = dict(topology=topology, checkpoint=str(checkpoint), checkpoint_sha256=identities,
                epoch=epoch, predecessor_input_sha256=predecessor, DP_engines=engines)
            input_path = directory / "input-contract.json"
            write_json(input_path, actual_input)
            identity = file_sha(input_path)
            payloads, sources = [], []
            for rank, template in enumerate(cache["cache_ABI"]):
                payload = build_chunk_metadata(template, layouts[rank], allocations[rank], tokens=current, context=begin,
                    tp_rank=rank % 4, active=rank // 4 == dp_rank, reserved=reserved)
                path = directory / f"rank-{rank}-metadata.pt"
                torch.save(payload, path)
                sources.append(dict(rank=rank, path=str(path), sha256=file_sha(path)))
                payloads.append(payload)
            contract = dict(config, checkpoint=str(checkpoint), input_contract_path=str(input_path),
                input_contract_sha256=identity)
            prepared_inputs = dict(input_contract=actual_input, prepare_input_stage=True)
            prepared_cache = dict(cache, cache_ABI=payloads,
                cache_ABI_sources=[dict(rank=row["rank"], payload_sha256=row["sha256"], report_sha256=row["sha256"])
                                   for row in sources])
            if host is not None:
                prepared_inputs["engram_lookup"] = host.lookup(prepared_inputs, cache=prepared_cache, config=contract)
            yield dict(inputs=prepared_inputs, contract=contract, metadata_sources=sources, cache=prepared_cache)
            predecessor = identity

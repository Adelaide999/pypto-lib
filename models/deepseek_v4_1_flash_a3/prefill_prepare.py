# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Lower transactional request batches into persistent paged-cache coordinates."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
import torch
from .cache import resolve_layer_role
from .cache_pool import CachePagePool
from .metadata import RequestLedger, RequestQuery
from .runtime_config import ParallelPlan, RuntimeLimits, prefill_execution_capacity
from .utils import file_sha, write_json
from .runtime_config import MAX_CONTEXT_TOKENS as MAX_CONTEXT


PAGE_ROWS = 128


def checkpoint_cache(checkpoint, topology, *, capacity=16):
    """Build cache and host Engram contracts directly from checkpoint metadata."""
    from .engram_host import build_compressed_token_map, engram_regions, make_layout
    checkpoint = Path(checkpoint).resolve()
    model = json.loads((checkpoint / "config.json").read_text())
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    mapping, compressed = build_compressed_token_map(tokenizer)
    text = model["text_config"]
    if compressed != text["engram_compressed_vocab_size"]:
        raise ValueError("Checkpoint tokenizer compressed vocabulary differs")
    hashed = make_layout()
    constants = dict(token_map=torch.tensor(mapping, dtype=torch.int64), multipliers=hashed.multipliers,
                     primes=hashed.primes, offsets=hashed.offsets)
    contracts = {}
    tp, world = topology["tp"], topology["ep"]
    regions = engram_regions(checkpoint)
    for slot, layer in enumerate((1, 14)):
        rows = regions[layer, "codes"].rows
        primes, offsets = hashed.primes[slot], hashed.offsets[slot]
        boundaries = [int(offsets[lane * 24 // tp]) for lane in range(tp)] + [rows]
        if rows != int(offsets[-1] + primes[-1]):
            raise ValueError("Checkpoint Engram rows differ from the original hash layout")
        shards = []
        for lane in range(tp):
            row_range = boundaries[lane:lane + 2]
            shards.append(dict(
                head_range=[lane * 24 // tp, (lane + 1) * 24 // tp],
                row_range=row_range,
                DP_sharing_ranks=list(range(lane, world, tp)),
            ))
        contracts[layer] = dict(
            checkpoint=str(checkpoint), model_config=model, topology=deepcopy(topology), capacity=capacity,
            physical_layer_id=layer,
            hash_constants=dict(constants, layer_ids=[1, 14], pad_id=text["engram_pad_token_id"]),
            engram_shards=shards, host_storage="checkpoint_file_mapping",
            engram_file_identity={kind: regions[layer, kind].identity for kind in ("codes", "scales")},
            engram_storage_contract=dict(dtype="INT8", scale_dtype="FP32", group_size=32,
                                         cpu_offload=True, dp_shared_memory=True, global_columns=24),
        )
    cache = dict(cache_ABI=cache_templates(constants, world=world), engram_contracts=contracts)
    config = dict(
        topology=deepcopy(topology), checkpoint=str(checkpoint), model_config=model,
        required_modes=dict(enable_expert_parallel=True, enable_fused_mc2=True,
                            enable_dsa_cp=True, enable_flashcomm1=True, enable_shared_expert_dp=True),
        capacity=capacity, scheduler_capacity=8192, fresh_shared_initialization={"topk_indices": -1},
        weight_residency="session",
    )
    return cache, config


def tokenize_prompt(inputs, checkpoint, vocabulary, *, max_tokens=MAX_CONTEXT):
    """Encode raw completion text or validate explicit original token IDs."""
    if isinstance(inputs, str):
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(Path(checkpoint) / "tokenizer.json"))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        inputs = tokenizer.encode(inputs).ids
    if (not isinstance(inputs, (list, tuple)) or not 1 <= len(inputs) <= max_tokens
            or any(type(token) is not int or not 0 <= token < vocabulary for token in inputs)):
        raise ValueError(f"Require 1..{max_tokens} integer token IDs within the checkpoint vocabulary")
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


def cache_templates(hash_constants, *, world=16):
    """Construct the A3 typed cache planes shared by Prefill and Decode."""
    views, groups = {}, []
    for layer in range(40):
        root = f"model.layers.{layer}.self_attn."
        groups.append(dict(group_id=len(groups), names=[root + "swa_cache"]))
        views[root + "swa_cache"] = (128, 512, "torch.bfloat16")
        role = resolve_layer_role(layer)
        if role["mode"] == "full":
            compressed = [root + "long_kv_cache", root + "indexer.k_cache"]
            groups.append(dict(group_id=len(groups), names=compressed))
            views[compressed[0]] = (128 // role["ratio"], 512, "torch.bfloat16")
            views[compressed[1]] = [(128 // role["ratio"], 128, "torch.int8"),
                                    (128 // role["ratio"], 1, "torch.float16")]
            if role["ratio"] == 2:
                groups.append(dict(group_id=len(groups), names=[root + "compressor.state_cache"]))
                views[root + "compressor.state_cache"] = (32, 1024, "torch.float32")
    descriptors, aliases = {}, {"shared": [], "index": []}
    for name, dimensions in views.items():
        planes = dimensions if isinstance(dimensions, list) else [dimensions]
        items = []
        for plane_id, (rows, width, dtype) in enumerate(planes):
            size = torch.empty((), dtype=getattr(torch, dtype.split(".")[-1])).element_size()
            key = "index" if name.endswith("indexer.k_cache") and rows == 128 else "shared"
            offset = (65536 + (8192 if plane_id else 0)) if name.endswith("indexer.k_cache") and rows == 64 else 0
            if key == "index" and plane_id:
                offset = 16384
            items.append(dict(storage_key=key, dtype=dtype, shape=[2, rows, 1, width],
                              stride=[131072 // size, width, width, 1],
                              storage_offset_bytes=offset, storage_bytes=262144))
            if name not in aliases[key]:
                aliases[key].append(name)
        descriptors[name] = items if isinstance(dimensions, list) else items[0]
    raw = {key: dict(storage_key=key, dtype="torch.int8", shape=[262144], stride=[1],
                      storage_offset_bytes=0, storage_bytes=262144) for key in aliases}
    layout = dict(raw=raw, views=descriptors, aliases=aliases)
    template = dict(layout=layout, groups=groups, hash_constants=hash_constants)
    return [deepcopy(template) for _ in range(world)]


def build_chunk_metadata(template, layout, allocations, *, tokens, context, tp_rank, active, reserved, tp=4):
    """Build CPU controls for SWA, completed compression pairs and per-request ring state."""
    total = len(tokens) if active else 0
    positions = torch.arange(context, context + total, dtype=torch.int64)
    if tp not in (1, 2, 4):
        raise ValueError("Prefill TP must be 1, 2 or 4")
    physical = (total + tp - 1) // tp
    start, end = tp_rank * physical, (tp_rank + 1) * physical
    span = [start, end, physical, physical * tp]
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


class TokenChunkInputs:
    """Retain request metadata, one page reservation and host lookup tables for a session."""

    def __init__(self, *, checkpoint, cache, config, output, context_capacity, prepare_engram=False):
        from .engram_host import HostEngram

        self.checkpoint, self.output = Path(checkpoint).resolve(), Path(output).resolve()
        capacity = prefill_execution_capacity(config.get("chunk_size", config.get("capacity", 16)))
        self.cache = dict(cache, engram_contracts={slot: dict(item, capacity=capacity)
                          for slot, item in cache["engram_contracts"].items()})
        self.config = dict(config, capacity=capacity)
        self.topology = config["topology"]
        tp, dp = self.topology["tp"], self.topology["dp"]
        if (self.topology != dict(dp=dp, tp=tp, ep=16, pp=1, pcp=1, dcp=1)
                or tp not in (1, 2, 4) or dp * tp != 16):
            raise ValueError("Token chunks require EP16 with TP1/2/4")
        model = json.loads((self.checkpoint / "config.json").read_text())
        if model != config["model_config"]:
            raise ValueError("Prepared model configuration differs from the selected checkpoint")
        self.capacity = context_capacity
        self.layouts, self.allocations, self.reserved = reserve_cache_layouts(cache["cache_ABI"], context_capacity)
        self.identities = {name: file_sha(self.checkpoint / name) for name in
                           ("config.json", "quant_model_weights.safetensors.index.json", "tokenizer.json")
                           if (self.checkpoint / name).is_file()}
        self.output.mkdir(parents=True, exist_ok=False)
        self.host = (HostEngram(self.checkpoint, self.cache["engram_contracts"], self.output.parent.name)
                     if prepare_engram else None)
        self.history, self.epoch, self.predecessor = [], 0, None
        self.closed = False

    def close(self):
        if not self.closed:
            self.closed = True
            if self.host is not None:
                self.host.close()

    def chunk(self, tokens, *, dp_rank=0, phase="prefill"):
        """Prepare dynamic rows and coordinates without replacing any cache backing allocation."""
        tp, dp = self.topology["tp"], self.topology["dp"]
        capacity = 16 if phase == "decode" else self.config["capacity"]
        if phase not in ("prefill", "decode") or (phase == "decode" and len(tokens) != 1):
            raise ValueError("Decode chunks contain exactly one token")
        if self.closed or not 0 <= dp_rank < dp or not 1 <= len(tokens) <= capacity:
            raise ValueError("Require a live session and tokens within its execution capacity")
        context = len(self.history)
        if context + len(tokens) > self.capacity:
            raise ValueError("Token chunk exceeds the session's fixed cache reservation")
        vocabulary = self.config["model_config"]["text_config"]["vocab_size"]
        tokens = tokenize_prompt(list(tokens), self.checkpoint, vocabulary)
        directory = self.output / f"chunk-{self.epoch}"
        directory.mkdir()
        request = dict(request_slot=0, token_ids=tokens, position_ids=list(range(context, context + len(tokens))),
                       live_token_mask=[1] * len(tokens), initial_context_length=context,
                       history_tokens=list(self.history))
        engines = [dict(dp_rank=rank, active=len(tokens) if rank == dp_rank else 0,
                        requests=[request] if rank == dp_rank else []) for rank in range(dp)]
        actual = dict(topology=self.topology, checkpoint=str(self.checkpoint), checkpoint_sha256=self.identities,
                      epoch=self.epoch, predecessor_input_sha256=self.predecessor, DP_engines=engines)
        input_path = directory / "input-contract.json"
        write_json(input_path, actual)
        identity = file_sha(input_path)
        payloads, sources = [], []
        for rank, template in enumerate(self.cache["cache_ABI"]):
            payload = build_chunk_metadata(template, self.layouts[rank], self.allocations[rank], tokens=tokens,
                context=context, tp_rank=rank % tp, active=rank // tp == dp_rank, reserved=self.reserved, tp=tp)
            path = directory / f"rank-{rank}-metadata.pt"
            torch.save(payload, path)
            sources.append(dict(rank=rank, path=str(path), sha256=file_sha(path)))
            payloads.append(payload)
        contract = dict(self.config, capacity=capacity, checkpoint=str(self.checkpoint), input_contract_path=str(input_path),
                        input_contract_sha256=identity)
        inputs = dict(input_contract=actual, prepare_input_stage=True)
        cache = dict(self.cache, engram_contracts={slot: dict(item, capacity=capacity)
                         for slot, item in self.cache["engram_contracts"].items()}, cache_ABI=payloads,
                     cache_ABI_sources=[dict(rank=row["rank"], payload_sha256=row["sha256"],
                                            report_sha256=row["sha256"]) for row in sources])
        if self.host is not None:
            inputs["engram_lookup"] = self.host.lookup(inputs, cache=cache, config=contract)
        self.history.extend(tokens)
        self.epoch += 1
        self.predecessor = identity
        return dict(inputs=inputs, contract=contract, metadata_sources=sources, cache=cache)


def build_packed_chunk_metadata(template, layout, pool, *, batch, rank):
    """Lower packed requests, original positions, C2 pairs and stable ring ownership."""
    dp_rank = rank // batch.plan.tp
    rows = batch.for_dp(dp_rank)
    coordinates = batch.coordinates(dp_rank)
    total, requests = len(coordinates["token_ids"]), batch.request_capacity
    positions = torch.tensor(coordinates["positions"], dtype=torch.int64)
    span = list(batch.plan.token_range(rank, total))
    valid = max(0, min(span[1], total) - min(span[0], total))
    lengths = torch.zeros(requests, dtype=torch.int32)
    lengths[:len(rows)] = torch.tensor(coordinates["seq_lens"], dtype=torch.int32)
    starts = torch.full((requests + 1,), total, dtype=torch.int32)
    starts[:len(rows) + 1] = torch.tensor(coordinates["query_start_loc"], dtype=torch.int32)
    metadata, groups = {}, []
    for index, source in enumerate(template["groups"]):
        ring_group = all(name.endswith("compressor.state_cache") for name in source["names"])
        swa_group = all(name.endswith("swa_cache") for name in source["names"])
        width = 1 if ring_group else pool.table_pages
        table = torch.full((requests, width), -1, dtype=torch.int32)
        raw_slots, live = [], []
        for request, row in enumerate(rows):
            pages = pool.group_pages(row.handle, index)
            if swa_group:
                first = max(0, row.context_length - 127) // PAGE_ROWS
                last = (row.sequence_length + PAGE_ROWS - 1) // PAGE_ROWS
                for logical in range(first, last):
                    table[request, logical] = pool.physical_page(row.handle, index, logical)
            else:
                table[request, :len(pages)] = torch.tensor(pages, dtype=torch.int32)
            live.extend(pages)
            for position in range(row.context_length, row.sequence_length):
                raw_slots.append([-1, -1] if ring_group else
                                 [pool.physical_page(row.handle, index, position // PAGE_ROWS),
                                  position % PAGE_ROWS])
        group = dict(group_id=source["group_id"], names=list(source["names"]), block_table=table,
            slot_mapping=torch.tensor(raw_slots, dtype=torch.int64).reshape(total, 2),
            live_block_ids=sorted(set(live)), reserved_untouched_block_id=pool.reserved)
        groups.append(group)
        for name in source["names"]:
            role = resolve_layer_role(int(name.split(".")[2]))
            compressed = name.endswith(("long_kv_cache", "indexer.k_cache"))
            ring = name.endswith("compressor.state_cache")
            ratio = role["ratio"] if compressed else 1
            slots = []
            for row in rows:
                for position in range(row.context_length, row.sequence_length):
                    if ring or (position + 1) % ratio:
                        slots.append([-1, -1])
                    else:
                        slots.append([pool.physical_page(row.handle, index, position // PAGE_ROWS),
                                      position % PAGE_ROWS // ratio])
            global_metadata = dict(block_table=table,
                query_start_loc=starts.clone(), seq_lens=lengths.clone(), positions=positions.clone(),
                slot_mapping=torch.tensor(slots, dtype=torch.int64).reshape(total, 2),
                cache_seq_lens=lengths // ratio, cmp_residual=lengths % ratio if ratio == 2 else None,
                num_actual_tokens=total, num_reqs=len(rows), max_query_len=coordinates["max_query_len"],
                max_cache_seq_len=int((lengths // ratio).max()))
            if ring:
                global_metadata["c2_source_positions"] = torch.where(positions % 2 == 1, positions - 1, 0)
            metadata[name] = dict(cp_token_range=span, num_actual_tokens=valid, global_metadata=global_metadata)
    return dict(layout=layout, metadata=metadata, groups=groups, positions=positions, lengths=lengths,
                query_start_loc=starts, hash_constants=template["hash_constants"])


class PackedTokenChunkInputs:
    """Own request transactions, fixed physical page pools and incremental host lookup inputs."""

    def __init__(self, *, checkpoint, cache, config, output, context_capacity, prepare_engram=False):
        from .engram_host import HostEngram

        self.checkpoint, self.output = Path(checkpoint).resolve(), Path(output).resolve()
        self.config = dict(config, packed_requests=True)
        self.topology = config["topology"]
        self.plan = ParallelPlan.from_topology(self.topology)
        model = json.loads((self.checkpoint / "config.json").read_text())
        if model != config["model_config"]:
            raise ValueError("Prepared model configuration differs from the selected checkpoint")
        self.capacity = context_capacity
        self.limits = RuntimeLimits(**config["runtime_limits"])
        if not 1 <= context_capacity <= self.limits.max_model_len:
            raise ValueError("Context reservation exceeds the configured model history limit")
        self.config["capacity"] = self.limits.token_capacity(self.limits.max_num_batched_tokens)
        self.cache = dict(cache, engram_contracts={
            layer: dict(contract, capacity=self.config["capacity"])
            for layer, contract in cache.get("engram_contracts", {}).items()
        })
        if self.config["capacity"] > 512:
            raise ValueError("Network token buckets above 512 require separate operator and MoE validation")
        self.plan.validate_model(model["text_config"])
        self.config["request_capacity"] = self.limits.max_seqs_per_dp
        self.ledger = RequestLedger(self.limits, self.plan)
        pages = (context_capacity + PAGE_ROWS - 1) // PAGE_ROWS
        groups = cache["cache_ABI"][0]["groups"]
        swa_pages = min(pages, (self.limits.max_num_batched_tokens + 126) // PAGE_ROWS + 2)
        default_pages = sum(1 if all(name.endswith("compressor.state_cache") for name in group["names"])
                            else swa_pages if all(name.endswith("swa_cache") for name in group["names"])
                            else pages for group in groups) * self.limits.max_seqs_per_dp
        page_capacity = config.get("cache_page_capacity", default_pages)
        if "hbm_budget" in config:
            from .cache_pool import cache_capacity_from_hbm

            budget = cache_capacity_from_hbm([item["layout"] for item in cache["cache_ABI"]],
                                            **config["hbm_budget"])
            page_capacity = min(page_capacity, budget["page_capacity"])
            self.config["cache_budget"] = budget
        self.pool = CachePagePool(cache["cache_ABI"], self.plan,
            page_capacity=page_capacity, table_pages=pages, swa_pages=swa_pages)
        self.layouts, self.reserved = self.pool.layouts, self.pool.reserved
        self.layout_identity = hashlib.sha256(json.dumps(self.layouts, sort_keys=True).encode()).hexdigest()
        self.metadata_transport = config.get("metadata_transport", "files")
        if self.metadata_transport not in ("files", "memory"):
            raise ValueError("metadata_transport must be files or memory")
        self.identities = {name: file_sha(self.checkpoint / name) for name in
                           ("config.json", "quant_model_weights.safetensors.index.json", "tokenizer.json")
                           if (self.checkpoint / name).is_file()}
        self.output.mkdir(parents=True, exist_ok=False)
        self.host = (HostEngram(self.checkpoint, self.cache["engram_contracts"], self.output.parent.name)
                     if prepare_engram else None)
        self.epoch, self.predecessor = 0, None
        self.retired = []
        self.pending = None
        self.closed = False

    def close(self):
        if not self.closed:
            self.closed = True
            if self.host is not None:
                self.host.close()

    def chunk(self, tokens, *, dp_rank=0, phase=None):
        """Prepare one request through the same transactional batch interface."""
        owner = self.ledger.owners.get("default")
        if phase is None:
            phase = "prefill" if owner is None else owner.status
        prompt = self.config.get("prompt_length", len(tokens)) if owner is None else owner.prompt_length
        query = RequestQuery("default", dp_rank, tuple(tokens), prompt,
                             None if owner is None else owner.handle.generation)
        return self.batch([query], phase=phase)

    def batch(self, queries, *, phase):
        """Validate and lower a batch without committing any request history."""
        if self.closed:
            raise ValueError("Require a live session")
        queries = tuple(queries)
        if any(not isinstance(query, RequestQuery) for query in queries):
            raise ValueError("Require explicit RequestQuery records")
        vocabulary = self.config["model_config"]["text_config"]["vocab_size"]
        for query in queries:
            tokenize_prompt(query.token_ids, self.checkpoint, vocabulary, max_tokens=self.limits.max_model_len)
        transaction = self.ledger.begin(queries, phase=phase)
        previous = None
        try:
            previous = self.pool.reserve(transaction)
            result = self._lower(transaction, previous)
        except BaseException:
            if previous is not None:
                self.pool.rollback(previous)
            self.ledger.rollback(transaction)
            raise
        self.pending = dict(transaction=transaction, previous_pool=previous,
                            identity=result["contract"]["input_contract_sha256"], chunk=result)
        return result

    def _lower(self, transaction, changes):
        directory = self.output / f"chunk-{self.epoch}"
        directory.mkdir(exist_ok=True)
        actual = dict(topology=self.topology, checkpoint=str(self.checkpoint), checkpoint_sha256=self.identities,
                      epoch=self.epoch, phase=transaction.phase, predecessor_input_sha256=self.predecessor,
                      DP_engines=transaction.contract_engines(), packed_requests=True,
                      request_capacity=transaction.request_capacity,
                      retired_requests=list(self.retired), output_map=transaction.output_map(),
                      cache_layout_sha256=self.layout_identity,
                      cache_page_updates=[dict(request_id=handle.request_id, dp_rank=handle.dp_rank,
                          request_slot=handle.state_slot, generation=handle.generation, group=group,
                          logical_start=start, pages=pages) for handle, group, start, pages in changes["updates"]])
        input_path = directory / "input-contract.json"
        write_json(input_path, actual)
        identity = file_sha(input_path)
        payloads, sources = [], []
        for rank, template in enumerate(self.cache["cache_ABI"]):
            payload = build_packed_chunk_metadata(template, self.layouts[rank], self.pool, batch=transaction, rank=rank)
            if self.metadata_transport == "files":
                path = directory / f"rank-{rank}-metadata.pt"
                torch.save(payload, path)
                sources.append(dict(rank=rank, path=str(path), sha256=file_sha(path)))
            else:
                receipt = hashlib.sha256(f"{identity}:{self.layout_identity}:{rank}".encode()).hexdigest()
                sources.append(dict(rank=rank, transport="memory", sha256=receipt))
            payloads.append(payload)
        contract = dict(self.config, checkpoint=str(self.checkpoint), input_contract_path=str(input_path),
                        input_contract_sha256=identity)
        inputs = dict(input_contract=actual, prepare_input_stage=True)
        cache = dict(self.cache, cache_ABI=payloads,
                     cache_ABI_sources=[dict(rank=row["rank"], payload_sha256=row["sha256"],
                                            report_sha256=row["sha256"]) for row in sources])
        if self.host is not None:
            inputs["engram_lookup"] = self.host.lookup(inputs, cache=cache, config=contract)
        return dict(inputs=inputs, contract=contract, metadata_sources=sources, cache=cache,
                    batch_metadata=transaction, metadata_owner=self)

    def commit(self, identity=None):
        if self.pending is None:
            raise RuntimeError("No prepared request batch to commit")
        if identity is not None and identity != self.pending["identity"]:
            raise ValueError("Completed input does not identify the prepared request transaction")
        self.ledger.commit(self.pending["transaction"])
        self.predecessor = self.pending["identity"]
        self.epoch += 1
        self.pending = None
        self.retired.clear()

    def rollback(self):
        if self.pending is not None:
            self.pool.rollback(self.pending["previous_pool"])
            self.ledger.rollback(self.pending["transaction"])
            self.pending = None

    def release(self, handle, reset):
        self.ledger.release(handle, lambda owner: self.pool.release(owner, reset))
        self.retired.append(dict(request_id=handle.request_id, dp_rank=handle.dp_rank,
                                 request_slot=handle.state_slot, generation=handle.generation))

    def abort(self, reset):
        if self.pending is not None:
            transaction = self.pending["transaction"]
            try:
                self.ledger.abort(transaction, lambda owner: self.pool.release(owner, reset))
            finally:
                self.pending = None


def prepare_token_chunks(inputs, *, checkpoint, cache, config, output, chunk_size=16, dp_rank=0, prepare_engram=False):
    """Yield bounded request chunks using one metadata and host-table owner."""
    vocabulary = config["model_config"]["text_config"]["vocab_size"]
    packed = "runtime_limits" in config
    if packed:
        limits = RuntimeLimits(**config["runtime_limits"])
        if type(chunk_size) is not int or not 1 <= chunk_size <= min(limits.max_num_batched_tokens, 512):
            raise ValueError("Chunk size must fit the independent network new-token budget (at most 512)")
        tokens = tokenize_prompt(inputs, checkpoint, vocabulary, max_tokens=limits.max_model_len)
        request_type = PackedTokenChunkInputs
        config = dict(config, prompt_length=len(tokens))
    else:
        prefill_execution_capacity(chunk_size)
        tokens = tokenize_prompt(inputs, checkpoint, vocabulary)
        request_type = TokenChunkInputs
        config = dict(config, chunk_size=chunk_size)
    request = request_type(checkpoint=checkpoint, cache=cache, config=config, output=output,
                           context_capacity=config.get("context_capacity", len(tokens)),
                           prepare_engram=prepare_engram)
    try:
        for begin in range(0, len(tokens), chunk_size):
            yield request.chunk(tokens[begin:begin + chunk_size], dp_rank=dp_rank)
            if packed and request.pending is not None:
                request.commit()
    finally:
        request.close()

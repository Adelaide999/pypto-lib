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
import hashlib
import os
from copy import deepcopy
from pathlib import Path
import torch
from .cache import resolve_layer_role
from .utils import file_sha, write_json


MAX_CONTEXT = 2048
PAGE_ROWS = 128


def checkpoint_cache(checkpoint, topology):
    """Build cache and host Engram contracts directly from checkpoint metadata."""
    from .engram_host import build_compressed_token_map, checkpoint_region, make_layout
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
    cache_root = Path(os.environ.get("PYPTO_A3_STATE_DIR", Path.home() / ".cache")) / "engram-contracts"
    cache_root.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256()
    for metadata_name in ("config.json", "quant_model_weights.safetensors.index.json", "tokenizer.json"):
        metadata_path = checkpoint / metadata_name
        identity.update(metadata_name.encode())
        identity.update(metadata_path.read_bytes())
    digest_cache_path = cache_root / (hashlib.sha256(str(checkpoint).encode()).hexdigest() + ".json")
    try:
        digest_cache = json.loads(digest_cache_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        digest_cache = {}
    if digest_cache.get("identity") != identity.hexdigest():
        digest_cache = {"identity": identity.hexdigest(), "payloads": {}}

    def payload_sha256(name, row_range, *, width, element_bytes):
        """Hash exactly one original Engram shard without materializing the table."""
        path, payload, spec = checkpoint_region(checkpoint, name)
        begin, end = row_range
        if spec["shape"][1] != width:
            raise ValueError(f"Unexpected Engram width for {name}")
        cache_key = f"{name}:{row_range[0]}:{row_range[1]}:{width}:{element_bytes}"
        cached = digest_cache["payloads"].get(cache_key)
        if cached:
            return cached
        digest = hashlib.sha256()
        byte_offset = payload + begin * width * element_bytes
        byte_count = (end - begin) * width * element_bytes
        with path.open("rb", buffering=0) as stream:
            stream.seek(byte_offset)
            remaining = byte_count
            while remaining:
                chunk = stream.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise EOFError(f"Incomplete Engram payload while hashing {name}")
                digest.update(chunk)
                remaining -= len(chunk)
        value = digest.hexdigest()
        digest_cache["payloads"][cache_key] = value
        return value

    for slot, layer in enumerate((1, 14)):
        _, _, spec = checkpoint_region(checkpoint, f"layers.{layer}.engram.embed.weight")
        primes, offsets = hashed.primes[slot], hashed.offsets[slot]
        boundaries = [int(offsets[lane * 24 // tp]) for lane in range(tp)] + [spec["shape"][0]]
        if spec["shape"][0] != int(offsets[-1] + primes[-1]):
            raise ValueError("Checkpoint Engram rows differ from the original hash layout")
        shards = []
        for lane in range(tp):
            row_range = boundaries[lane:lane + 2]
            shards.append(dict(
                head_range=[lane * 24 // tp, (lane + 1) * 24 // tp],
                row_range=row_range,
                DP_sharing_ranks=list(range(lane, world, tp)),
                host_payload_sha256=dict(
                    codes=payload_sha256(
                        f"layers.{layer}.engram.embed.weight", row_range, width=256, element_bytes=1
                    ),
                    scales=payload_sha256(
                        f"layers.{layer}.engram.embed.scale", row_range, width=8, element_bytes=4
                    ),
                ),
            ))
        contracts[layer] = dict(
            checkpoint=str(checkpoint), model_config=model, topology=deepcopy(topology), capacity=16,
            physical_layer_id=layer,
            hash_constants=dict(constants, layer_ids=[1, 14], pad_id=text["engram_pad_token_id"]),
            engram_shards=shards, host_storage="checkpoint_file_mapping",
            engram_storage_contract=dict(dtype="INT8", scale_dtype="FP32", group_size=32,
                                         cpu_offload=True, dp_shared_memory=True, global_columns=24),
        )
    cache = dict(cache_ABI=cache_templates(constants, world=world), engram_contracts=contracts)
    config = dict(
        topology=deepcopy(topology), checkpoint=str(checkpoint), model_config=model,
        required_modes=dict(enable_expert_parallel=True, enable_fused_mc2=True,
                            enable_dsa_cp=True, enable_flashcomm1=True, enable_shared_expert_dp=True),
        capacity=16, scheduler_capacity=8192, fresh_shared_initialization={"topk_indices": -1},
        weight_residency="session",
    )
    digest_cache_path.write_text(json.dumps(digest_cache, indent=2) + "\n")
    return cache, config


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
        self.cache, self.config = cache, dict(config)
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
        self.host = (HostEngram(self.checkpoint, cache["engram_contracts"], self.output.parent.name)
                     if prepare_engram else None)
        self.history, self.epoch, self.predecessor = [], 0, None
        self.closed = False

    def close(self):
        if not self.closed:
            self.closed = True
            if self.host is not None:
                self.host.close()

    def chunk(self, tokens, *, dp_rank=0):
        """Prepare dynamic rows and coordinates without replacing any cache backing allocation."""
        tp, dp = self.topology["tp"], self.topology["dp"]
        if self.closed or not 0 <= dp_rank < dp or not 1 <= len(tokens) <= 16:
            raise ValueError("Require a live session and 1..16 tokens on one DP engine")
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
        contract = dict(self.config, checkpoint=str(self.checkpoint), input_contract_path=str(input_path),
                        input_contract_sha256=identity)
        inputs = dict(input_contract=actual, prepare_input_stage=True)
        cache = dict(self.cache, cache_ABI=payloads,
                     cache_ABI_sources=[dict(rank=row["rank"], payload_sha256=row["sha256"],
                                            report_sha256=row["sha256"]) for row in sources])
        if self.host is not None:
            inputs["engram_lookup"] = self.host.lookup(inputs, cache=cache, config=contract)
        self.history.extend(tokens)
        self.epoch += 1
        self.predecessor = identity
        return dict(inputs=inputs, contract=contract, metadata_sources=sources, cache=cache)


def prepare_token_chunks(inputs, *, checkpoint, cache, config, output, chunk_size=16, dp_rank=0, prepare_engram=False):
    """Yield bounded request chunks using one metadata and host-table owner."""
    if type(chunk_size) is not int or not 1 <= chunk_size <= 16:
        raise ValueError("Chunk size must be in 1..16")
    tokens = tokenize_prompt(inputs, checkpoint, config["model_config"]["text_config"]["vocab_size"])
    request = TokenChunkInputs(checkpoint=checkpoint, cache=cache, config=config, output=output,
                               context_capacity=config.get("context_capacity", len(tokens)),
                               prepare_engram=prepare_engram)
    try:
        for begin in range(0, len(tokens), chunk_size):
            yield request.chunk(tokens[begin:begin + chunk_size], dp_rank=dp_rank)
    finally:
        request.close()

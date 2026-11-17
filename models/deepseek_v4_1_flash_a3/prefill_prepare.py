# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Lower transactional request batches into persistent paged-cache coordinates."""

import json
import hashlib
import os
from copy import deepcopy
from pathlib import Path
import torch
from .cache import resolve_layer_role
from .cache_pool import CachePagePool
from .metadata import RequestLedger, RequestQuery
from .runtime_config import ParallelPlan, RuntimeLimits
from .utils import file_sha, write_json


MAX_CONTEXT = 1048576
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
        raise ValueError("Require 1..1048576 integer token IDs within the checkpoint vocabulary")
    return list(inputs)


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


def build_chunk_metadata(template, layout, pool, *, batch, rank):
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


class TokenChunkInputs:
    """Own request transactions, fixed physical page pools and incremental host lookup inputs."""

    def __init__(self, *, checkpoint, cache, config, output, context_capacity, prepare_engram=False):
        from .engram_host import HostEngram

        self.checkpoint, self.output = Path(checkpoint).resolve(), Path(output).resolve()
        self.cache, self.config = cache, dict(config)
        self.topology = config["topology"]
        self.plan = ParallelPlan.from_topology(self.topology)
        model = json.loads((self.checkpoint / "config.json").read_text())
        if model != config["model_config"]:
            raise ValueError("Prepared model configuration differs from the selected checkpoint")
        self.capacity = context_capacity
        settings = dict(config.get("runtime_limits", {}))
        settings.setdefault("max_model_len", context_capacity)
        settings.setdefault("max_num_seqs", 1)
        settings.setdefault("max_seqs_per_dp", settings["max_num_seqs"])
        settings.setdefault("max_num_batched_tokens", config.get("capacity", 16))
        self.limits = RuntimeLimits(**settings)
        if not 1 <= context_capacity <= self.limits.max_model_len:
            raise ValueError("Context reservation exceeds the configured model history limit")
        self.config["capacity"] = self.limits.token_capacity(self.limits.max_num_batched_tokens)
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
        self.host = (HostEngram(self.checkpoint, cache["engram_contracts"], self.output.parent.name)
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

    def chunk(self, tokens, *, dp_rank=0):
        """Prepare one request through the same transactional batch interface."""
        owner = self.ledger.owners.get("default")
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
            tokenize_prompt(query.token_ids, self.checkpoint, vocabulary)
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
                      DP_engines=transaction.contract_engines(), request_capacity=transaction.request_capacity,
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
            payload = build_chunk_metadata(template, self.layouts[rank], self.pool, batch=transaction, rank=rank)
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
    budget = config.get("runtime_limits", {}).get("max_num_batched_tokens", config.get("capacity", 16))
    if type(chunk_size) is not int or not 1 <= chunk_size <= min(budget, 512):
        raise ValueError("Chunk size must fit the independent network new-token budget (at most 512)")
    tokens = tokenize_prompt(inputs, checkpoint, config["model_config"]["text_config"]["vocab_size"])
    request = TokenChunkInputs(checkpoint=checkpoint, cache=cache, config=dict(config, prompt_length=len(tokens)), output=output,
                               context_capacity=config.get("context_capacity", len(tokens)),
                               prepare_engram=prepare_engram)
    try:
        for begin in range(0, len(tokens), chunk_size):
            yield request.chunk(tokens[begin:begin + chunk_size], dp_rank=dp_rank)
            if request.pending is not None:
                request.commit()
    finally:
        request.close()

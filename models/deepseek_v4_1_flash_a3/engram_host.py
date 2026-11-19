# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""HOST Engram storage, hashing, row lookup and offload lifecycle."""

import hashlib
import json
import mmap
import os
import struct
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence
import numpy as np
import torch
from .utils import file_sha


def lookup_columns(codes, scales, rows):
    """Gather INT8/FP32 group32 rows on CPU and round the decoded entries to BF16."""
    values = codes.gather(rows).float().reshape(*rows.shape, 8, 32)
    factors = scales.gather(rows).unsqueeze(-1)
    return (values * factors).reshape(*rows.shape, 256).to(torch.bfloat16)


class HostEngram:
    """Share read-only checkpoint mappings across DP peers and gather current rows."""

    def __init__(self, checkpoint, contracts, namespace):
        if set(contracts) != {1, 14} or not namespace or "/" in namespace or ".." in namespace:
            raise ValueError("Require both original Engram slots and a private host-storage namespace")
        self.checkpoint = str(Path(checkpoint).resolve())
        self.contracts = deepcopy(contracts)
        self.regions = engram_regions(self.checkpoint)
        self.arrays = {}
        self._stack = ExitStack()
        self._closed = False
        self.constants = self.contracts[1]["hash_constants"]
        for slot, layer in enumerate((1, 14)):
            contract = self.contracts[layer]
            if str(Path(contract["checkpoint"]).resolve()) != self.checkpoint:
                raise ValueError("Host lookup and forward must use the same checkpoint")
            if contract["topology"] != self.contracts[1]["topology"]:
                raise ValueError("Both host tables must use the same rank ownership")
            storage = dict(
                dtype="INT8",
                scale_dtype="FP32",
                group_size=32,
                cpu_offload=True,
                dp_shared_memory=True,
                global_columns=24,
            )
            if contract["engram_storage_contract"] != storage:
                raise ValueError("Host lookup requires the original Eco-Tech INT8/FP32 group32 tables")
            other = contract["hash_constants"]
            if (
                tuple(other["layer_ids"]) != (1, 14)
                or other["pad_id"] != self.constants["pad_id"]
                or any(
                    not torch.equal(other[key], self.constants[key])
                    for key in ("token_map", "multipliers", "primes", "offsets")
                )
            ):
                raise ValueError("Both host tables must use the same original hash constants")
            tp = contract["topology"]["tp"]
            if len(contract["engram_shards"]) != tp or tp not in (1, 2, 4):
                raise ValueError("Require complete original TP1/2/4 table ownership")
            offsets = self.constants["offsets"][slot].reshape(-1)
            table_rows = int(offsets[-1] + self.constants["primes"].reshape(2, 24)[slot, -1])
            if self.regions[layer, "codes"].rows != table_rows:
                raise ValueError("Host table size differs from the Engram hash layout")
            boundaries = [int(offsets[rank * 24 // tp]) for rank in range(tp)] + [table_rows]
            for rank, shard in enumerate(contract["engram_shards"]):
                if list(shard["row_range"]) != boundaries[rank:rank + 2]:
                    raise ValueError("Host table row range differs from its assigned hash columns")
                if shard["head_range"] != [rank * 24 // tp, (rank + 1) * 24 // tp]:
                    raise ValueError("Host table shards must preserve original column order")
                if shard["DP_sharing_ranks"] != list(range(rank, contract["topology"]["ep"], tp)):
                    raise ValueError("Host tables must be shared by matching DP peers")
        bind_engram_files(self.contracts, self.regions)
        self.layout = HashLayout(
            (1, 14),
            self.constants["multipliers"],
            self.constants["primes"].reshape(2, 24),
            self.constants["offsets"],
            4,
            8,
            int(self.constants["token_map"].max()) + 1,
            self.constants["pad_id"],
            len(self.constants["token_map"]),
        )

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    def close(self):
        """Close checkpoint mappings; materialized rows own their storage."""
        if not self._closed:
            self._closed = True
            self._stack.close()
            self.arrays.clear()

    def _table(self, layer, rank, kind):
        if self._closed:
            raise RuntimeError("Host Engram is closed")
        key = layer, rank, kind
        if key not in self.arrays:
            shard = self.contracts[layer]["engram_shards"][rank]
            region = self.regions[layer, kind]
            array = MappedEngramArray(region, shard["row_range"])
            self._stack.callback(array.close)
            self.arrays[key] = array
        return self.arrays[key]

    def lookup(self, inputs, *, cache, config):
        """Return ready rows for an explicit request, including single-token continuations.

        Hashes use the request's explicit last three history tokens. No device
        workers, registrations, descriptors or transfers are used here.
        """
        if self._closed:
            raise RuntimeError("Host Engram is closed")
        topology, capacity = config["topology"], config["capacity"]
        actual = inputs["input_contract"]
        if (
            topology != self.contracts[1]["topology"]
            or actual["topology"] != topology
            or str(Path(config["checkpoint"]).resolve()) != self.checkpoint
        ):
            raise ValueError("Host rows must match the forward checkpoint and topology")
        tp, world = topology["tp"], topology["ep"]
        payloads = cache["cache_ABI"]
        if len(payloads) != world or len(actual["DP_engines"]) != topology["dp"]:
            raise ValueError("Host preparation needs the actual all-rank request metadata")
        records = []
        model = self.contracts[1]["model_config"]
        image_id = model["image_token_id"]
        image_pad_id = model.get("image_pad_token_id", image_id + 1)
        for dp, engine in enumerate(actual["DP_engines"]):
            tokens, positions, starts, lookback = [], [], [0], []
            for request in engine["requests"]:
                prefix = request["history_tokens"]
                current = request["token_ids"]
                if (
                    not current
                    or any(
                        type(token) is not int or not 0 <= token < len(self.constants["token_map"])
                        for token in [*prefix, *current]
                    )
                    or request["initial_context_length"] != len(prefix)
                    or request["live_token_mask"] != [1] * len(current)
                    or request["position_ids"] != list(range(len(prefix), len(prefix) + len(current)))
                ):
                    raise ValueError("Host lookup requires complete contiguous request coordinates")
                tokens.extend(current)
                positions.extend(request["position_ids"])
                starts.append(len(tokens))
                lookback.append((list(reversed(prefix[-3:])) + [-1] * 3)[:3])
            if engine["dp_rank"] != dp or engine["active"] != len(tokens) or len(tokens) > capacity:
                raise ValueError("Host input does not match the bounded DP engine")
            ids = torch.tensor(tokens, dtype=torch.int64)
            pos = torch.tensor(positions, dtype=torch.int64)
            if torch.any(ids < 0) or torch.any(ids >= len(self.constants["token_map"])):
                raise ValueError("Host lookup token IDs exceed the compressed token map")
            dead = (ids == image_id) | (ids == image_pad_id)
            hashes = torch.full((capacity, 2, 24), -1, dtype=torch.int32)
            mask = torch.zeros(capacity, dtype=torch.int32)
            if tokens:
                previous = torch.tensor(lookback, dtype=torch.int64)
                hashes[: len(tokens)] = hash_ids(
                    ids,
                    self.constants["token_map"],
                    pos,
                    torch.tensor(starts, dtype=torch.int32),
                    layout=self.layout,
                    dead_mask=dead,
                    lookback_token_ids=previous,
                    lookback_dead_mask=(previous == image_id) | (previous == image_pad_id),
                )
                mask[: len(tokens)] = (~dead).to(torch.int32)
            full = {layer: torch.zeros((capacity, 6144), dtype=torch.bfloat16) for layer in (1, 14)}
            if tokens:
                for layer, slot in ((1, 0), (14, 1)):
                    for rank, shard in enumerate(self.contracts[layer]["engram_shards"]):
                        begin, end = shard["head_range"]
                        rows = hashes[: len(tokens), slot, begin:end]
                        decoded = lookup_columns(
                            self._table(layer, rank, "codes"), self._table(layer, rank, "scales"), rows
                        )
                        full[layer][: len(tokens), begin * 256 : end * 256] = decoded.flatten(1)
            for lane in range(tp):
                payload = payloads[dp * tp + lane]
                native = payload["metadata"]["model.layers.1.self_attn.swa_cache"]
                metadata = native["global_metadata"]
                if (
                    not torch.equal(payload["positions"], pos)
                    or payload["query_start_loc"].tolist() != starts
                    or metadata["query_start_loc"].tolist() != starts
                ):
                    raise ValueError("Host row coordinates differ from the original cache metadata")
                span = tuple(native["cp_token_range"])
                begin, end, physical, padded = span
                if (
                    begin != lane * physical
                    or end != begin + physical
                    or padded != physical * tp
                    or padded < len(tokens)
                    or physical > capacity
                ):
                    raise ValueError("Host rows must retain the original SP/TP token partition")
                valid = max(0, min(end, len(tokens)) - min(begin, len(tokens)))
                local = {layer: torch.zeros_like(value) for layer, value in full.items()}
                local_mask = torch.zeros_like(mask)
                local_mask[:valid] = mask[begin : begin + valid]
                for layer in full:
                    local[layer][:valid] = full[layer][begin : begin + valid]
                records.append(
                    dict(
                        full_input_ids=ids,
                        positions=pos,
                        cp_token_range=span,
                        hashes=hashes,
                        mask=mask,
                        full=full,
                        local=local,
                        local_mask=local_mask,
                    )
                )
        return dict(
            epoch=actual.get("epoch", 0),
            token_map_id=config["input_contract_sha256"],
            checkpoint=self.checkpoint,
            topology=deepcopy(topology),
            capacity=capacity,
            cases={"fresh": records},
            storage={str(key): dict(array.receipt) for key, array in self.arrays.items()},
            hash_constants=self.constants,
        )


DEAD_ID = -1


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % p == 0:
            return value == p
    d, r = value - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, value)
        if x in (1, value - 1):
            continue
        for _ in range(r - 1):
            x = x * x % value
            if x == value - 1:
                break
        else:
            return False
    return True


def _next_prime(start: int, seen: set[int]) -> int:
    candidate = start + 1
    while candidate in seen or not _is_prime(candidate):
        candidate += 1
    return candidate


@dataclass(frozen=True)
class HashLayout:
    """Frozen V4.1 hash parameters for the configured Engram layers."""

    layer_ids: tuple[int, ...]
    multipliers: torch.Tensor  # [layers, max_ngram]
    primes: torch.Tensor  # [layers, (max_ngram - 1) * heads]
    offsets: torch.Tensor  # [layers, (max_ngram - 1) * heads]
    max_ngram_size: int
    n_heads: int
    compressed_vocab_size: int
    pad_token_id: int
    vocab_size: int

    @property
    def lookback_depth(self) -> int:
        return self.max_ngram_size - 1

    @property
    def n_hash_cols(self) -> int:
        return self.lookback_depth * self.n_heads


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Build the V4.1 normalized token map used by the hash state.

    The tokenizer object is deliberately supplied by the host runner.  This
    keeps tokenizer I/O out of PyPTO graphs while preserving the exact
    normalization contract (including one-space and partial UTF-8 tokens).
    """
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    # ``transformers.PreTrainedTokenizerFast`` exposes the Rust tokenizer as
    # ``backend_tokenizer`` while the checkpoint preparation path uses the
    # native ``tokenizers.Tokenizer`` directly.  Both provide the same decode
    # and id-to-token operations needed here.
    backend = getattr(tokenizer, "backend_tokenizer", tokenizer)
    vocabulary_size = len(tokenizer) if hasattr(tokenizer, "__len__") else tokenizer.get_vocab_size()
    key_to_new: dict[str, int] = {}
    lookup = [0] * vocabulary_size
    for token_id in range(vocabulary_size):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def make_layout(
    layer_ids: Sequence[int] = (1, 14),
    *,
    max_ngram_size: int = 4,
    n_heads: int = 8,
    compressed_vocab_size: int = 99092,
    vocab_size: int = 16_000_000,
    pad_token_id: int = 2,
) -> HashLayout:
    """Construct the same deterministic layout as upstream vLLM."""
    layers = tuple(int(x) for x in layer_ids)
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // compressed_vocab_size) // 2)
    multipliers = []
    primes = []
    offsets = []
    seen: set[int] = set()
    for layer_id in layers:
        rng = np.random.default_rng(10007 * layer_id)
        values = rng.integers(0, bound, size=(max_ngram_size,), dtype=np.int64)
        multipliers.append(torch.from_numpy(values * 2 + 1))
        layer_primes: list[int] = []
        for _ in range(max_ngram_size - 1):
            for _ in range(n_heads):
                prime = _next_prime(vocab_size - 1, seen)
                seen.add(prime)
                layer_primes.append(prime)
        primes.append(torch.tensor(layer_primes, dtype=torch.int64))
        running = 0
        layer_offsets = []
        for prime in layer_primes:
            layer_offsets.append(running)
            running += prime
        offsets.append(torch.tensor(layer_offsets, dtype=torch.int64))
    return HashLayout(
        layers,
        torch.stack(multipliers),
        torch.stack(primes),
        torch.stack(offsets),
        max_ngram_size,
        n_heads,
        compressed_vocab_size,
        pad_token_id,
        vocab_size,
    )


def update_slot_history(
    input_ids: torch.Tensor,
    token_map: torch.Tensor,
    dead_mask: torch.Tensor,
    slot_mapping: torch.Tensor,
    hash_cache: torch.Tensor,
) -> None:
    """Publish compressed ids to physical slots, matching Ascend's writer."""
    if input_ids.ndim != 1 or token_map.ndim != 1 or dead_mask.ndim != 1:
        raise ValueError("input_ids, token_map and dead_mask must be vectors")
    if slot_mapping.shape != input_ids.shape:
        raise ValueError("slot_mapping shape mismatch")
    valid = (slot_mapping >= 0) & (slot_mapping < hash_cache.numel())
    values = token_map[input_ids.clamp(0, token_map.numel() - 1).long()].to(torch.int32)
    values = torch.where(dead_mask.bool(), torch.full_like(values, DEAD_ID), values)
    hash_cache[slot_mapping[valid].long()] = values[valid]


def hash_ids(
    input_ids: torch.Tensor,
    token_map: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    layout: HashLayout,
    dead_mask: torch.Tensor | None = None,
    lookback_token_ids: torch.Tensor | None = None,
    lookback_dead_mask: torch.Tensor | None = None,
    hash_cache: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    cache_block_size: int = 128,
) -> torch.Tensor:
    """Generate ``[tokens, layers, 24]`` hash ids with vLLM precedence.

    Current chunk tokens win over explicit request lookback, which wins over
    the physical slot cache.  A dead token or a missing/negative history
    position blocks the rolling hash and emits ``pad_token_id`` for all later
    n-gram sizes at that position.
    """
    input_ids = input_ids.to(torch.int64).flatten()
    token_map = token_map.to(torch.int64).flatten()
    positions = positions.to(torch.int64).flatten()
    query_start_loc = query_start_loc.to(torch.int64).flatten()
    n_tokens = int(input_ids.numel())
    if positions.numel() != n_tokens or dead_mask is not None and dead_mask.numel() != n_tokens:
        raise ValueError("input/position/dead-mask lengths differ")
    dead = torch.zeros(n_tokens, dtype=torch.bool) if dead_mask is None else dead_mask.bool().flatten()
    if query_start_loc.numel() < 2 or int(query_start_loc[-1]) != n_tokens:
        raise ValueError("query_start_loc must bound every input token")
    lookback_depth = layout.lookback_depth
    if lookback_token_ids is None:
        lookback_token_ids = torch.full((query_start_loc.numel() - 1, lookback_depth), -1, dtype=torch.int64)
    else:
        lookback_token_ids = lookback_token_ids.to(torch.int64)
    if lookback_dead_mask is None:
        lookback_dead_mask = torch.zeros_like(lookback_token_ids, dtype=torch.bool)
    else:
        lookback_dead_mask = lookback_dead_mask.bool()
    if lookback_token_ids.shape != lookback_dead_mask.shape:
        raise ValueError("lookback token and mask shapes differ")
    output = torch.full(
        (n_tokens, len(layout.layer_ids), layout.n_hash_cols), layout.pad_token_id, dtype=torch.int32
    )
    if n_tokens == 0:
        return output
    req_ends = query_start_loc[1:]
    for token in range(n_tokens):
        # The native binary search advances when token >= end, i.e. upper
        # bound.  A token exactly at a request boundary belongs to the next
        # request rather than the preceding chunk.
        req = int(torch.searchsorted(req_ends, torch.tensor(token), right=True))
        req = min(req, query_start_loc.numel() - 2)
        chunk_start = int(positions[int(query_start_loc[req])])
        rolling = [0] * len(layout.layer_ids)
        blocked = False
        for shift in range(layout.max_ngram_size):
            lookback = int(positions[token]) - shift
            in_batch = lookback >= chunk_start
            batch_idx = max(token - shift, 0)
            source: int
            source_dead = False
            if in_batch:
                source = int(token_map[int(input_ids[batch_idx])])
                source_dead = bool(dead[batch_idx])
            else:
                col = chunk_start - 1 - lookback
                in_window = 0 <= col < lookback_depth
                known = in_window and int(lookback_token_ids[req, col]) >= 0
                if known:
                    source_token = int(lookback_token_ids[req, col])
                    source = int(token_map[source_token])
                    source_dead = bool(lookback_dead_mask[req, col])
                elif hash_cache is not None and block_table is not None:
                    clamped = min(max(lookback, 0), int(block_table.shape[1]) * cache_block_size - 1)
                    block = int(block_table[min(req, block_table.shape[0] - 1), clamped // cache_block_size])
                    slot = min(
                        max(block * cache_block_size + clamped % cache_block_size, 0), hash_cache.numel() - 1
                    )
                    source = int(hash_cache[slot])
                else:
                    source = layout.pad_token_id
            blocked = blocked or lookback < 0 or source_dead or source == DEAD_ID
            value = layout.pad_token_id if blocked else source
            for li in range(len(layout.layer_ids)):
                rolling[li] ^= int(value) * int(layout.multipliers[li, shift])
                if shift:
                    base = (shift - 1) * layout.n_heads
                    for head in range(layout.n_heads):
                        col = base + head
                        output[token, li, col] = int(
                            rolling[li] % int(layout.primes[li, col]) + int(layout.offsets[li, col])
                        )
    return output


def bounded_host_gather(
    table: torch.Tensor,
    row_ids: torch.Tensor,
    *,
    full_rows: int,
    chunk_rows: int = 32,
) -> torch.Tensor:
    """Gather a bounded host table window with explicit range checks.

    ``table`` is a resident or mmap-backed host tensor.  The function records
    the full checkpoint row count separately so callers cannot accidentally
    treat a 4,096-row audit window as the 384,006,168-row production table.
    """
    if table.ndim != 2 or row_ids.ndim != 2:
        raise ValueError("table and row_ids must be rank-2")
    if (
        full_rows < table.shape[0]
        or row_ids.numel()
        and (int(row_ids.min()) < 0 or int(row_ids.max()) >= full_rows)
    ):
        raise ValueError("row id outside declared full table range")
    if int(row_ids.max()) >= table.shape[0]:
        raise LookupOutsideResidentWindow(int(row_ids.max()), int(table.shape[0]), full_rows)
    pieces = []
    for start in range(0, row_ids.shape[0], chunk_rows):
        pieces.append(
            table.index_select(0, row_ids[start : start + chunk_rows].reshape(-1).long()).reshape(
                -1, row_ids.shape[1], table.shape[1]
            )
        )
    return torch.cat(pieces, dim=0) if pieces else table.new_empty((0, row_ids.shape[1], table.shape[1]))


class LookupOutsideResidentWindow(IndexError):
    """A valid full-table row was not present in the bounded audit window."""

    def __init__(self, row: int, resident_rows: int, full_rows: int):
        super().__init__(f"row {row} needs host offload (resident={resident_rows}, full={full_rows})")
        self.row, self.resident_rows, self.full_rows = row, resident_rows, full_rows


@dataclass(frozen=True)
class EngramTableGeometry:
    """Physical and declared dimensions of one Engram table."""

    full_rows: int
    resident_rows: int
    row_width: int = 256
    scale_width: int = 8
    scale_block: int = 32

    def __post_init__(self) -> None:
        if self.full_rows <= 0 or self.resident_rows <= 0:
            raise ValueError("Engram table dimensions must be positive")
        if self.resident_rows > self.full_rows:
            raise ValueError("resident table cannot exceed declared full table")
        if self.row_width != self.scale_width * self.scale_block:
            raise ValueError("scale geometry does not cover one row")


class A2EngramHostTable:
    """Decode and stage a bounded E4M3/E8M0 Engram table window.

    ``weight_path`` and ``scale_path`` contain raw bytes, one row after
    another.  The byte tensors are memory mapped by ``torch.from_file`` and
    therefore do not duplicate the resident table in host memory.  A complete
    production table can use the same API if its files are supplied; callers
    still receive only requested rows.
    """

    def __init__(
        self,
        weight_path: str | Path,
        scale_path: str | Path,
        *,
        full_rows: int,
        row_width: int = 256,
        scale_width: int = 8,
        scale_block: int = 32,
    ) -> None:
        self.weight_path = Path(weight_path)
        self.scale_path = Path(scale_path)
        if not self.weight_path.is_file() or not self.scale_path.is_file():
            raise FileNotFoundError("Engram raw weight or scale file is missing")
        if row_width != scale_width * scale_block:
            raise ValueError("scale geometry does not cover one row")
        weight_bytes = self.weight_path.stat().st_size
        scale_bytes = self.scale_path.stat().st_size
        if weight_bytes % row_width:
            raise ValueError("raw E4M3 file is not an integral number of rows")
        if scale_bytes % scale_width:
            raise ValueError("raw E8M0 file is not an integral number of rows")
        resident_rows = weight_bytes // row_width
        scale_rows = scale_bytes // scale_width
        if resident_rows != scale_rows:
            raise ValueError("weight and scale files have different row counts")
        self.geometry = EngramTableGeometry(
            full_rows=int(full_rows),
            resident_rows=int(resident_rows),
            row_width=int(row_width),
            scale_width=int(scale_width),
            scale_block=int(scale_block),
        )
        # ``size`` is required because torch.from_file defaults to an empty
        # mapping when the size is omitted.
        self._weight_bytes = torch.from_file(
            str(self.weight_path), dtype=torch.uint8, size=weight_bytes
        ).reshape(self.geometry.resident_rows, self.geometry.row_width)
        self._scale_bytes = torch.from_file(
            str(self.scale_path), dtype=torch.uint8, size=scale_bytes
        ).reshape(self.geometry.resident_rows, self.geometry.scale_width)

    @property
    def full_rows(self) -> int:
        return self.geometry.full_rows

    @property
    def resident_rows(self) -> int:
        return self.geometry.resident_rows

    def coverage(self) -> dict[str, int | str]:
        """Return an audit-friendly declaration of the available range."""

        return {
            "full_rows": self.geometry.full_rows,
            "resident_rows": self.geometry.resident_rows,
            "resident_start": 0,
            "resident_end_exclusive": self.geometry.resident_rows,
            "row_width": self.geometry.row_width,
            "scale_width": self.geometry.scale_width,
            "scale_block": self.geometry.scale_block,
            "coverage": (
                "full"
                if self.geometry.resident_rows == self.geometry.full_rows
                else "bounded_resident_window"
            ),
        }

    def _validate_ids(self, row_ids: torch.Tensor) -> torch.Tensor:
        ids = torch.as_tensor(row_ids, dtype=torch.int64).reshape(-1)
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= self.full_rows):
            raise ValueError(f"row id outside declared table range [0, {self.full_rows})")
        if ids.numel() and int(ids.max()) >= self.resident_rows:
            raise LookupOutsideResidentWindow(int(ids.max()), self.resident_rows, self.full_rows)
        return ids

    def read_rows(self, row_ids: torch.Tensor) -> torch.Tensor:
        """Decode requested rows to BF16 while preserving row-id order."""

        original = torch.as_tensor(row_ids, dtype=torch.int64)
        ids = self._validate_ids(original)
        if ids.numel() == 0:
            return torch.empty((*original.shape, self.geometry.row_width), dtype=torch.bfloat16)
        raw = self._weight_bytes.index_select(0, ids)
        raw_fp32 = raw.view(torch.float8_e4m3fn).float()
        scale_raw = self._scale_bytes.index_select(0, ids)
        # vLLM's A2 path materializes E8M0 as an FP32 exponent bit pattern.
        # Byte 0 consequently maps to +0.0, matching the native bitcast.
        scale_fp32 = (scale_raw.to(torch.int32) << 23).view(torch.float32)
        decoded = raw_fp32.reshape(-1, self.geometry.scale_width, self.geometry.scale_block)
        decoded = (decoded * scale_fp32[:, :, None]).reshape(-1, self.geometry.row_width)
        return decoded.to(torch.bfloat16).reshape(*original.shape, self.geometry.row_width)

    def iter_rows(
        self, row_ids: Sequence[int] | torch.Tensor, *, chunk_rows: int = 32
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield ``(ids, decoded_rows)`` in bounded chunks."""

        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        ids = torch.as_tensor(row_ids, dtype=torch.int64).reshape(-1)
        for start in range(0, ids.numel(), chunk_rows):
            chunk = ids[start : start + chunk_rows]
            yield chunk, self.read_rows(chunk)

    def stage_rows(self, row_ids: torch.Tensor, device: torch.device | str) -> torch.Tensor:
        """Decode on host and copy rows to an A2 device, when available."""

        rows = self.read_rows(row_ids)
        staged = rows.to(device)
        if getattr(staged.device, "type", None) == "npu":
            torch.npu.synchronize()
        return staged


def token_map_from_tokenizer(tokenizer, *, expected_size: int = 99092) -> torch.Tensor:
    """Build the native normalized token map and enforce its hash contract."""

    values, compressed_size = build_compressed_token_map(tokenizer)
    if compressed_size != expected_size:
        raise ValueError(f"tokenizer compressed vocab mismatch: {compressed_size} != {expected_size}")
    return torch.tensor(values, dtype=torch.int32)


def _file_identity(stat):
    return dict(device=stat.st_dev, inode=stat.st_ino, size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


@dataclass(frozen=True)
class EngramRegion:
    name: str
    path: Path
    offset: int
    rows: int
    width: int
    dtype: np.dtype
    identity: dict


def engram_regions(checkpoint):
    """Read the four Engram tensor descriptors and bind them to their checkpoint files."""
    root = Path(checkpoint)
    index = json.loads((root / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    regions = {}
    for layer in (1, 14):
        for kind, suffix, width, tag, dtype in (("codes", "weight", 256, "I8", np.int8),
                                               ("scales", "scale", 8, "F32", np.float32)):
            name = f"layers.{layer}.engram.embed.{suffix}"
            path = root / index[name]
            with path.open("rb") as stream:
                identity = _file_identity(os.fstat(stream.fileno()))
                header_size, = struct.unpack("<Q", stream.read(8))
                if header_size > min(100_000_000, identity["size"] - 8):
                    raise ValueError(f"Invalid Engram safetensors header size for {name}")
                spec = json.loads(stream.read(header_size))[name]
            if spec["dtype"] != tag or len(spec["shape"]) != 2 or spec["shape"][1] != width:
                raise ValueError(f"Invalid Engram dtype or shape for {name}")
            rows = spec["shape"][0]
            begin, end = spec["data_offsets"]
            dtype = np.dtype(dtype)
            if rows <= 0 or begin < 0 or end - begin != rows * width * dtype.itemsize:
                raise ValueError(f"Invalid Engram payload span for {name}")
            if 8 + header_size + end > identity["size"]:
                raise EOFError(f"Truncated Engram checkpoint payload for {name}")
            regions[layer, kind] = EngramRegion(name, path, 8 + header_size + begin, rows, width, dtype, identity)
        if regions[layer, "codes"].rows != regions[layer, "scales"].rows:
            raise ValueError(f"Engram codes and scales have different row counts at layer {layer}")
    return regions


def bind_engram_files(contracts, regions):
    """Verify legacy shard digests once and bind the contracts in place to these file identities."""
    for layer, contract in contracts.items():
        identities = {kind: regions[layer, kind].identity for kind in ("codes", "scales")}
        if "engram_file_identity" in contract and contract["engram_file_identity"] != identities:
            raise ValueError("Engram checkpoint files differ from the bound contract")
        verified = []
        for shard in contract["engram_shards"]:
            expected = shard.get("host_payload_sha256")
            if expected is None:
                if "engram_file_identity" not in contract:
                    raise ValueError("Engram contract lacks a file identity or a frozen shard digest")
                continue
            for kind in ("codes", "scales"):
                region = regions[layer, kind]
                with region.path.open("rb", buffering=0) as stream:
                    begin, end = shard["row_range"]
                    if not 0 <= begin < end <= region.rows:
                        raise ValueError("Engram digest range exceeds its tensor")
                    row_bytes = region.width * region.dtype.itemsize
                    stream.seek(region.offset + begin * row_bytes)
                    remaining = (end - begin) * row_bytes
                    digest = hashlib.sha256()
                    while remaining:
                        block = stream.read(min(8 * 1024**2, remaining))
                        if not block:
                            raise EOFError(f"Incomplete Engram shard: {region.name}")
                        digest.update(block)
                        remaining -= len(block)
                    if _file_identity(os.fstat(stream.fileno())) != region.identity:
                        raise ValueError(f"Engram checkpoint changed during verification: {region.name}")
                    if digest.hexdigest() != expected[kind]:
                        raise ValueError(f"Engram shard differs from its frozen digest: {region.name}")
            verified.append(shard)
        for shard in verified:
            shard["verified_payload_sha256"] = shard.pop("host_payload_sha256")
        contract["engram_file_identity"] = identities


class MappedEngramArray:
    """Map one TP shard directly from an immutable safetensors file."""

    def __init__(self, region, row_range):
        begin, end = row_range
        if not 0 <= begin < end <= region.rows:
            raise ValueError("Engram shard exceeds its checkpoint tensor")
        row_bytes = region.width * region.dtype.itemsize
        offset = region.offset + begin * row_bytes
        size = (end - begin) * row_bytes
        aligned = offset - offset % mmap.ALLOCATIONGRANULARITY
        delta = offset - aligned
        self.row_start = begin
        self.region = region
        with region.path.open("rb") as stream:
            actual = _file_identity(os.fstat(stream.fileno()))
            if actual != region.identity:
                raise ValueError(f"Engram checkpoint file changed: {region.name}")
            self.mapping = mmap.mmap(stream.fileno(), delta + size, access=mmap.ACCESS_READ, offset=aligned)
        self.array = None
        try:
            self.array = np.ndarray((end - begin, region.width), region.dtype, buffer=self.mapping, offset=delta)
            self.mapping.madvise(mmap.MADV_RANDOM)
            self.receipt = dict(checkpoint_file=str(region.path), tensor=region.name, row_range=[begin, end],
                                shape=list(self.array.shape), dtype="I8" if region.dtype == np.int8 else "F32",
                                byte_offset=offset, bytes=size, file_identity=actual, storage="readonly_file_mapping")
        except BaseException:
            self.close()
            raise

    def gather(self, rows):
        """Gather unique valid global rows and restore their order and multiplicity."""
        if self.array is None or rows.device.type != "cpu" or rows.dtype not in (torch.int32, torch.int64):
            raise ValueError("Lookup requires an open host table and CPU integer row IDs")
        if _file_identity(self.region.path.stat()) != self.region.identity:
            raise ValueError(f"Engram checkpoint file changed: {self.region.name}")
        local = rows.numpy().astype(np.int64, copy=False) - self.row_start
        valid = (local >= 0) & (local < self.array.shape[0])
        unique, inverse = np.unique(local[valid], return_inverse=True)
        result = np.zeros((*local.shape, self.array.shape[1]), dtype=self.array.dtype)
        result[valid] = self.array[unique][inverse]
        return torch.from_numpy(result)

    def close(self):
        self.array = None
        self.mapping.close()


def load_rotation(checkpoint, hidden_size):
    """Load and validate repeated block32 checkpoint rotation by byte identity."""
    from safetensors import safe_open

    path = Path(checkpoint) / "optional/quarot.safetensors"
    if not path.is_file():
        raise FileNotFoundError("The physical A3 Engram requires its checkpoint rotation")
    with safe_open(path, framework="pt", device="cpu") as source:
        rotation = source.get_tensor("global_rotation")
    if tuple(rotation.shape) != (hidden_size, hidden_size):
        raise ValueError("Checkpoint rotation has a different full hidden dimension")
    block = rotation[:32, :32].contiguous()
    zero = torch.zeros_like(block)
    for row in range(0, hidden_size, 32):
        for col in range(0, hidden_size, 32):
            expected = block if row == col else zero
            if not torch.equal(rotation[row : row + 32, col : col + 32], expected):
                raise ValueError("Checkpoint rotation is not repeated block32")
    return block.float(), {"path": str(path), "sha256": file_sha(path)}


def engram_owner(contract, layer_id, role=None):
    """Resolve the physical checkpoint prefix and original hash slot."""
    layer_ids = tuple(contract["hash_constants"]["layer_ids"])
    if role is None:
        if (
            layer_id not in (1, 14)
            or contract.get("physical_layer_id") != layer_id
            or layer_id not in layer_ids
        ):
            raise ValueError("Require the actual physical Engram owner in the original hash layout")
    else:
        from .cache import resolve_layer_role

        role = resolve_layer_role(layer_id, role)
        if role["engram_slot"] is None or layer_id not in layer_ids:
            raise ValueError("Require the original Engram owner role")
    slot = layer_ids.index(layer_id)
    if contract.get("engram_slot", slot) != slot or (role is not None and role["engram_slot"] != slot):
        raise ValueError("Engram slot differs from the original physical-layer hash layout")
    return f"layers.{layer_id}.engram", slot

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed request coordinates and transactional generation ownership."""

from dataclasses import dataclass

from .runtime_config import ParallelPlan, RuntimeLimits


@dataclass(frozen=True)
class RequestHandle:
    request_id: str
    dp_rank: int
    state_slot: int
    generation: int


@dataclass(frozen=True)
class RequestQuery:
    request_id: str
    dp_rank: int
    token_ids: tuple
    prompt_length: int
    generation: int | None = None


@dataclass
class RequestProgress:
    handle: RequestHandle
    prompt_length: int
    committed_length: int = 0
    lookback: tuple = ()
    status: str = "prefill"


@dataclass(frozen=True)
class RequestSlice:
    handle: RequestHandle
    token_ids: tuple
    context_length: int
    prompt_length: int
    lookback: tuple

    @property
    def sequence_length(self):
        return self.context_length + len(self.token_ids)


@dataclass(frozen=True)
class BatchMetadata:
    """Keep caller output order independent of DP packing and persistent slots."""

    epoch: int
    phase: str
    requests: tuple
    plan: ParallelPlan
    token_capacity: int
    request_capacity: int

    @property
    def num_tokens(self):
        return sum(len(row.token_ids) for row in self.requests)

    def for_dp(self, dp_rank):
        if not 0 <= dp_rank < self.plan.dp:
            raise ValueError("DP owner is outside the batch plan")
        return tuple(row for row in self.requests if row.handle.dp_rank == dp_rank)

    def coordinates(self, dp_rank):
        """Return original token coordinates without copying complete history."""
        rows = self.for_dp(dp_rank)
        starts, tokens, positions, token_to_req = [0], [], [], []
        for index, row in enumerate(rows):
            tokens.extend(row.token_ids)
            positions.extend(range(row.context_length, row.sequence_length))
            token_to_req.extend([index] * len(row.token_ids))
            starts.append(len(tokens))
        return dict(
            query_start_loc=starts, context_lens=[row.context_length for row in rows],
            seq_lens=[row.sequence_length for row in rows], positions=positions, token_ids=tokens,
            token_to_req=token_to_req, request_ids=[row.handle.request_id for row in rows],
            state_slots=[row.handle.state_slot for row in rows],
            generations=[row.handle.generation for row in rows],
            max_query_len=max((len(row.token_ids) for row in rows), default=0),
            output_rows=[starts[index + 1] - 1 for index, row in enumerate(rows)
                         if row.sequence_length >= row.prompt_length],
        )

    def contract_engines(self):
        engines = []
        for dp_rank in range(self.plan.dp):
            requests = []
            for row in self.for_dp(dp_rank):
                handle = row.handle
                requests.append(dict(
                    request_id=handle.request_id, request_slot=handle.state_slot, generation=handle.generation,
                    token_ids=list(row.token_ids), initial_context_length=row.context_length,
                    position_ids=list(range(row.context_length, row.sequence_length)),
                    live_token_mask=[1] * len(row.token_ids), prompt_length=row.prompt_length,
                    history_tokens=list(row.lookback), history_start=row.context_length - len(row.lookback),
                ))
            engines.append(dict(dp_rank=dp_rank, active=sum(len(row["token_ids"]) for row in requests),
                                requests=requests))
        return engines

    def output_map(self):
        """Map requested samples to the TP leader and packed row for each request."""
        result = []
        for row in self.requests:
            packed = self.for_dp(row.handle.dp_rank)
            index = packed.index(row)
            output = sum(len(item.token_ids) for item in packed[:index + 1]) - 1
            result.append(dict(request_id=row.handle.request_id, generation=row.handle.generation,
                               state_slot=row.handle.state_slot, dp_rank=row.handle.dp_rank,
                               rank=self.plan.rank(row.handle.dp_rank), row=output,
                               logit_row=sum(item.sequence_length >= item.prompt_length for item in packed[:index]),
                               sample=row.sequence_length >= row.prompt_length))
        return result


class RequestLedger:
    """Commit one homogeneous batch while retaining omitted and paused requests."""

    def __init__(self, limits, plan, *, lookback_tokens=8):
        if not isinstance(limits, RuntimeLimits) or not isinstance(plan, ParallelPlan):
            raise TypeError("RequestLedger requires explicit RuntimeLimits and ParallelPlan")
        if type(lookback_tokens) is not int or lookback_tokens < 1:
            raise ValueError("Engram lookback must be a bounded positive extent")
        self.limits, self.plan, self.lookback_tokens = limits, plan, lookback_tokens
        self.owners, self.pending, self.epoch = {}, None, 0
        self.generations = [[0] * limits.max_seqs_per_dp for _ in range(plan.dp)]
        self.free = [list(range(limits.max_seqs_per_dp - 1, -1, -1)) for _ in range(plan.dp)]
        self.poisoned = False
        self._new_requests = set()

    def _idle(self):
        if self.poisoned:
            raise RuntimeError("Failed device reset requires closing this request owner")
        if self.pending is not None:
            raise RuntimeError("A request transaction is already in flight")

    def get(self, handle):
        if not isinstance(handle, RequestHandle):
            raise ValueError("Require an explicit request-generation handle")
        owner = self.owners.get(handle.request_id)
        if owner is None or owner.handle != handle:
            raise ValueError("Stale or foreign request generation")
        return owner

    def begin(self, queries, *, phase):
        self._idle()
        queries = tuple(queries)
        if phase not in ("prefill", "decode"):
            raise ValueError("Dispatch must have one homogeneous Prefill or Decode phase")
        if not queries or any(not isinstance(row, RequestQuery) for row in queries):
            raise ValueError("Require one or more explicit request queries")
        if len({row.request_id for row in queries}) != len(queries):
            raise ValueError("A request may appear only once in a batch")
        total = sum(len(row.token_ids) for row in queries)
        capacity = self.limits.token_capacity(total)
        available = [list(slots) for slots in self.free]
        additions, slices = {}, []
        for query in queries:
            if not isinstance(query.request_id, str) or not query.request_id:
                raise ValueError("Request ID must be nonempty")
            if type(query.dp_rank) is not int or not 0 <= query.dp_rank < self.plan.dp:
                raise ValueError("Request DP owner is outside the plan")
            if query.generation is not None and (type(query.generation) is not int or query.generation < 0):
                raise ValueError("Request generation must be a nonnegative integer")
            if (not query.token_ids or any(type(token) is not int or token < 0 for token in query.token_ids)
                    or type(query.prompt_length) is not int
                    or not 1 <= query.prompt_length <= self.limits.max_model_len):
                raise ValueError("Invalid request tokens or prompt length")
            owner = self.owners.get(query.request_id)
            if owner is None:
                if phase != "prefill" or query.generation is not None:
                    raise ValueError("New request generations require Prefill from position zero")
                if len(self.owners) + len(additions) >= self.limits.max_num_seqs or not available[query.dp_rank]:
                    raise ValueError("Live request capacity exhausted independently of the token budget")
                slot = available[query.dp_rank].pop()
                generation = self.generations[query.dp_rank][slot]
                handle = RequestHandle(query.request_id, query.dp_rank, slot, generation)
                owner = RequestProgress(handle, query.prompt_length)
                additions[query.request_id] = owner
            elif (query.generation != owner.handle.generation or query.dp_rank != owner.handle.dp_rank
                  or query.prompt_length != owner.prompt_length):
                raise ValueError("Continuation changed request generation, DP ownership or prompt")
            if owner.status in ("paused", "finished", "failed"):
                raise ValueError("Paused, finished or failed requests cannot execute")
            end = owner.committed_length + len(query.token_ids)
            if end > self.limits.max_model_len:
                raise ValueError("Request exceeds the independent history-context limit")
            if phase == "prefill" and (owner.status != "prefill" or end > owner.prompt_length):
                raise ValueError("Prefill must remain within the unfinished original prompt")
            if phase == "decode" and (owner.status != "decode" or len(query.token_ids) != 1):
                raise ValueError("Decode requires completed Prefill and one query token per request")
            slices.append(RequestSlice(owner.handle, tuple(query.token_ids), owner.committed_length,
                                       owner.prompt_length, owner.lookback))
        self.free = available
        self.owners.update(additions)
        self._new_requests = set(additions)
        self.pending = BatchMetadata(self.epoch, phase, tuple(slices), self.plan,
                                     capacity, self.limits.max_seqs_per_dp)
        return self.pending

    def _current(self, batch):
        if batch is not self.pending or batch.epoch != self.epoch:
            raise ValueError("Stale or foreign batch completion")
        for row in batch.requests:
            self.get(row.handle)

    def commit(self, batch):
        self._current(batch)
        for row in batch.requests:
            owner = self.get(row.handle)
            owner.committed_length = row.sequence_length
            owner.lookback = (owner.lookback + row.token_ids)[-self.lookback_tokens:]
            owner.status = "decode" if row.sequence_length >= owner.prompt_length else "prefill"
        self.pending, self._new_requests = None, set()
        self.epoch += 1

    def rollback(self, batch):
        """Cancel a preparation failure before any mutable device state was exposed."""
        self._current(batch)
        for request_id in self._new_requests:
            owner = self.owners.pop(request_id)
            self.free[owner.handle.dp_rank].append(owner.handle.state_slot)
        self.pending, self._new_requests = None, set()

    def pause(self, handle):
        self._idle()
        owner = self.get(handle)
        if owner.status not in ("prefill", "decode"):
            raise ValueError("Only a live request can pause")
        owner.status = "paused"

    def resume(self, handle):
        self._idle()
        owner = self.get(handle)
        if owner.status != "paused":
            raise ValueError("Only a paused request can resume")
        owner.status = "decode" if owner.committed_length >= owner.prompt_length else "prefill"

    def finish(self, handle):
        self._idle()
        owner = self.get(handle)
        if owner.status not in ("decode", "paused") or owner.committed_length < owner.prompt_length:
            raise ValueError("Only a request with completed Prefill can finish")
        owner.status = "finished"

    def release(self, handle, reset):
        """Synchronously reset mutable state before publishing a slot for another generation."""
        self._idle()
        owner = self.get(handle)
        try:
            reset(handle)
        except BaseException:
            owner.status, self.poisoned = "failed", True
            raise
        del self.owners[handle.request_id]
        self.generations[handle.dp_rank][handle.state_slot] += 1
        self.free[handle.dp_rank].append(handle.state_slot)

    def abort(self, batch, reset):
        """Invalidate every potentially partially written request after device execution fails."""
        self._current(batch)
        self.pending, self._new_requests = None, set()
        self.epoch += 1
        errors = []
        for row in batch.requests:
            try:
                owner = self.get(row.handle)
                reset(row.handle)
            except BaseException as error:
                self.owners[row.handle.request_id].status = "failed"
                errors.append(error)
            else:
                handle = owner.handle
                del self.owners[handle.request_id]
                self.generations[handle.dp_rank][handle.state_slot] += 1
                self.free[handle.dp_rank].append(handle.state_slot)
        if errors:
            self.poisoned = True
            raise RuntimeError("Failed to reset all invalidated request generations") from errors[0]

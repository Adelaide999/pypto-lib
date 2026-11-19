# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Retain A3 workers, typed device allocations and HCCL groups across a complete generation."""

import logging
import os
import struct
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from threading import Condition, Lock, RLock, Thread
import torch
from .utils import write_json


_LOGGER = logging.getLogger(__name__)


def _host_bytes(value):
    """View the complete CPU allocation as bytes, including typed aliases."""
    if value.device.type != "cpu" or not value.is_contiguous():
        raise ValueError("Resident staging requires contiguous CPU tensors")
    storage = value.untyped_storage()
    return torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))


def _device_view(allocation, value, base_offset=0):
    """Bind a contiguous typed view to the allocation's owning Buffer."""
    from pypto.runtime import DeviceTensor

    offset = value.storage_offset() * value.element_size() - base_offset
    if offset < 0:
        raise ValueError("Resident tensor view begins before its allocation")
    if offset + value.numel() * value.element_size() > allocation.nbytes:
        raise ValueError("Resident tensor view exceeds its allocation")
    return DeviceTensor(allocation.data_ptr + offset, value.shape, value.dtype,
                        buffer=allocation.buffer, byte_offset=offset)


class _RankBatch:
    def __init__(self, function, ranks, world):
        self.world = world
        self.results, self.errors = [None] * world, [None] * world
        self.reset(function, ranks)

    def reset(self, function, ranks):
        order = tuple(ranks)
        members = frozenset(order)
        if len(members) != len(order) or any(rank < 0 or rank >= self.world for rank in order):
            raise ValueError("Rank batch requires distinct allocated ranks")
        self.function, self.order, self.ranks = function, order, members
        for rank in range(self.world):
            self.results[rank] = self.errors[rank] = None
        self.remaining = len(order)


class _RankExecutor:
    """Publish complete batches atomically to persistent rank-affine threads."""

    def __init__(self, world):
        self._world = world
        lock = Lock()
        self._condition = Condition(lock)
        self._complete = Condition(lock)
        self._threads = [None] * world
        self._admission = Lock()
        self._slot = _RankBatch(None, (), world)
        self._batch, self._generation = None, 0
        self._closing = self._closed = False

    def start(self):
        for rank in range(self._world):
            thread = Thread(target=self._work, args=(rank,),
                            name=f"ThreadPoolExecutor-a3-transport-{rank}")
            self._threads[rank] = thread
            thread.start()

    def _work(self, rank):
        generation = 0
        while True:
            with self._condition:
                while not self._closed and self._generation == generation:
                    self._condition.wait()
                if self._closed:
                    return
                generation, batch = self._generation, self._batch
                if batch is None or rank not in batch.ranks:
                    continue
            result, error = None, None
            try:
                result = batch.function(rank)
            except BaseException as failure:
                error = failure
            with self._condition:
                batch.results[rank], batch.errors[rank] = result, error
                batch.remaining -= 1
                if not batch.remaining:
                    self._batch = None
                    self._complete.notify_all()

    def run(self, function, ranks):
        with self._admission:
            try:
                return self._run(function, ranks)
            finally:
                self._slot.function = None

    def _run(self, function, ranks):
        batch = self._slot
        batch.reset(function, ranks)
        if not batch.order:
            return []
        with self._condition:
            if self._closing:
                raise RuntimeError("Rank executor is closed")
            while self._batch is not None:
                self._complete.wait()
                if self._closing:
                    raise RuntimeError("Rank executor is closed")
            self._batch = batch
            self._generation += 1
            try:
                self._condition.notify_all()
            except BaseException:
                # Notified threads cannot inspect the batch until this lock
                # is released, so failed admission can roll back atomically.
                self._generation -= 1
                self._batch = None
                raise
            wait_error = None
            while batch.remaining:
                try:
                    self._complete.wait()
                except BaseException as error:
                    if wait_error is None:
                        wait_error = error
        if wait_error is not None:
            raise wait_error
        for rank in batch.order:
            if batch.errors[rank] is not None:
                raise batch.errors[rank]
        return [batch.results[rank] for rank in batch.order]

    def shutdown(self):
        with self._condition:
            self._closing = True
            while self._batch is not None:
                self._complete.wait()
            self._closed = True
            self._condition.notify_all()
        error = None
        for thread in self._threads:
            if thread is not None and thread.ident is not None:
                try:
                    thread.join()
                except BaseException as failure:
                    if error is None:
                        error = failure
        if error is not None:
            raise error
        self._threads.clear()
        with self._admission:
            self._slot.reset(None, ())


class NetworkProfiler:
    """Collect one device trace per dispatch, grouped by phase, chunk and logical rank."""

    def __init__(self, devices, *, level=0, phase="all", output_dir=None):
        if type(level) is not int or not 0 <= level <= 4:
            raise ValueError("Chip swimlane level must be an integer in 0..4")
        if phase not in ("all", "prefill", "decode"):
            raise ValueError("Profile phase must be all, prefill or decode")
        if level and output_dir is None:
            raise ValueError("Enabled network profiling requires an output directory")
        self.level, self.phase = level, phase
        self.devices = tuple(devices)
        self.output = Path(output_dir).resolve() if output_dir is not None else None
        self.current, self.chunks, self.name_maps = None, [], {}
        self.started = False

    def begin_chunk(self, phase, epoch):
        """Open a fresh capture scope after model initialization and before dispatch."""
        if self.current is not None:
            raise RuntimeError("The previous profiling chunk has not been finalized")
        if not self.level or self.phase not in ("all", phase):
            return
        if phase not in ("prefill", "decode") or type(epoch) is not int or epoch < 0:
            raise ValueError("Profiling requires a valid phase and nonnegative chunk epoch")
        if not self.started:
            self.output.mkdir(parents=True, exist_ok=False)
            self.started = True
        directory = self.output / phase / f"chunk-{epoch:06d}"
        directory.mkdir(parents=True, exist_ok=False)
        self.current = dict(phase=phase, epoch=epoch, directory=directory,
                            ranks=[[] for _ in self.devices])

    def prepare_call(self, compiled, rank, staged, arguments, parameters, *, empty_capture=False, ring_heap=None):
        """Reserve a unique output prefix without changing cached dispatch arguments."""
        if self.current is None:
            return None, None
        from pypto.runtime import RunConfig

        calls = self.current["ranks"][rank]
        directory = self.current["directory"] / f"rank-{rank:02d}" / f"call-{len(calls):06d}"
        directory.mkdir(parents=True, exist_ok=False)
        record = dict(rank=rank, physical_device=int(self.devices[rank]), call=len(calls),
                      phase=self.current["phase"], epoch=self.current["epoch"], level=self.level,
                      compiled_directory=str(compiled.output_dir), directory=str(directory),
                      staged=staged, empty_capture=empty_capture, ring_heap=ring_heap, status="started")
        record["scalar_arguments"] = {
            parameter.get("name", f"arg{index}"): value
            for index, (value, parameter) in enumerate(zip(arguments, parameters)) if parameter["shape"] is None
        }
        calls.append(record)
        config = RunConfig(enable_chip_swimlane=self.level, ring_heap=ring_heap)
        return record, compiled._build_call_config(config, dfx_dir=directory)

    def finish_chunk(self, *, completed=True):
        """Index raw captures and offline conversion commands without replaying model state."""
        if self.current is None:
            return None
        from pypto.runtime.runner import _write_name_map

        chunk, self.current = self.current, None
        missing, ranks = [], []
        for rank, calls in enumerate(chunk["ranks"]):
            for record in calls:
                directory = Path(record["directory"])
                source = Path(record["compiled_directory"])
                raw = directory / "chip_swimlane_records.json"
                record["raw_trace"] = str(raw) if raw.is_file() else None
                if raw.is_file():
                    if source not in self.name_maps:
                        symbols = self.output / "symbols" / f"program-{len(self.name_maps):04d}"
                        symbols.mkdir(parents=True, exist_ok=False)
                        prebuilt = (source / "binary_manifest.json").is_file()
                        self.name_maps[source] = _write_name_map(source, symbols, prebuilt=prebuilt)
                    names = self.name_maps[source]
                    command = [sys.executable, "-m", "simpler_setup.tools.swimlane_converter", str(raw),
                               "-o", str(directory / "swimlane.json")]
                    if names is not None:
                        command.extend(("--func-names", str(names)))
                    record["convert_command"] = command
                    record["capture_status"] = "captured"
                elif any(record["scalar_arguments"].get(name) == 0 for name in ("active", "num_tokens")):
                    record["capture_status"] = "not_emitted_zero_active"
                elif record["empty_capture"]:
                    record["capture_status"] = "not_emitted_empty_dispatch"
                elif record["status"] == "completed":
                    record["capture_status"] = "missing"
                    missing.append(str(directory))
                else:
                    record["capture_status"] = "dispatch_failed"
                write_json(directory / "dispatch.json", record)
            index = chunk["directory"] / f"rank-{rank:02d}" / "index.json"
            index.parent.mkdir(parents=True, exist_ok=True)
            write_json(index, dict(rank=rank, physical_device=int(self.devices[rank]), calls=calls))
            ranks.append(dict(rank=rank, calls=len(calls), index=str(index)))
        status = "complete" if completed and not missing else "incomplete"
        summary = dict(phase=chunk["phase"], epoch=chunk["epoch"], level=self.level,
                       status=status, ranks=ranks, missing_traces=missing,
                       index=str(chunk["directory"] / "index.json"))
        write_json(Path(summary["index"]), summary)
        self.chunks.append(summary)
        write_json(self.output / "index.json", self.summary())
        if completed and missing:
            raise RuntimeError(f"Profiling did not emit {len(missing)} dispatch traces; see {summary['index']}")
        return summary

    def summary(self):
        """Describe capture coverage and paths without implying cross-rank clock alignment."""
        return dict(enabled=bool(self.level), level=self.level, phase=self.phase,
                    output_dir=str(self.output) if self.output is not None else None,
                    scope="per-dispatch PyPTO device compute; host-trace.json records host dispatch and collectives",
                    cross_rank_clock_alignment=False, chunks=list(self.chunks))


class ResidentNetworkRuntime:
    """Own one worker, program registry and allocation bank per logical rank."""

    def __init__(self, devices, topology, *, worker_factory=None, device_copy=None, device_readback=None,
                 device_fill=None, profiling=None, ring_heap=None):
        begin = perf_counter()
        self.devices, self.topology = tuple(devices), dict(topology)
        os.environ.setdefault("HCCL_EXEC_TIMEOUT", "180")
        os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "180")
        if len(self.devices) != topology["ep"] or len(set(self.devices)) != len(self.devices):
            raise ValueError("Resident runtime requires the exact complete distinct device mapping")
        self.profiling = NetworkProfiler(self.devices, **(profiling or {}))
        from pypto.runtime import RunConfig

        self.ring_heap = ring_heap
        self._run_config = RunConfig(ring_heap=ring_heap) if ring_heap is not None else None
        self.workers, self.pinned, self.channels = [], {}, {}
        self.device_owners = {}
        self.device_readonly = {}
        self.device_active = {}
        self.alias_leases = {}
        self.rotations = {}
        self.programs = {}
        self.static_parameters = {}
        self.parameter_abis = {}
        self.dynamic = {}
        self.values = {}
        self.free_values = {}
        self.free_workspaces = {}
        self._stats_lock = Lock()
        self.transfer_stats = [dict() for _ in devices]
        self._device_copy = device_copy or self._copy_device
        self._device_readback = device_readback or self._copy_from_device
        self._device_fill = device_fill or self._fill_device
        self._memory = None
        self.allocations = [[] for _ in devices]
        self.failed_ranks = set()
        self.rank_locks = [RLock() for _ in devices]
        self.compute_queues = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"a3-rank-{rank}")
                               for rank in range(len(devices))]
        self.collective_queue = ThreadPoolExecutor(max_workers=1, thread_name_prefix="a3-collective")
        self.trace = []
        self._stack = ExitStack()
        self._worker_owners = [None] * len(devices)
        self._staging_owners = [None] * len(devices)
        self._dispatch_locks = [RLock() for _ in devices]
        self._device_views = {}
        self._dispatch_cache = {}
        self.staging_workers = {}
        self._exchange_executor = None
        self._exchange_lock = RLock()
        self._transport_channels = []
        self._closed = False
        self._native_dispatch = worker_factory is None
        self._select = lambda rank: None
        self.stats = dict(worker_initializations=0, dispatches=0, h2d_bytes=0, d2h_bytes=0,
                          weight_bytes=[0] * len(devices), state_bytes=[0] * len(devices),
                          transport_calls=0, transport_seconds=0.0, device_weight_upload_calls=0,
                          device_weight_upload_seconds=0.0,
                          dynamic_allocations=0, cache_h2d_bytes=0, control_d2h_bytes=0, output_d2h_bytes=0,
                          activation_d2h_bytes=0, d2d_bytes=0, program_evictions=0, staging_worker_initializations=0,
                          workspace_fill_bytes=0, workspace_requests=0, workspace_reuses=0)
        try:
            if worker_factory is None:
                from pypto.backend import BackendType
                from pypto.runtime import ChipWorker, ExecutionMode, RunConfig

                def worker_factory(rank):
                    config = RunConfig(arch=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                                       device_id=rank)
                    return ChipWorker(config=config)

                from .moe_transport import CANNMemory

                # CANN's current device belongs to the calling thread. Creating
                # several workers leaves that thread on the last initialized
                # device; every later allocation/copy must select its owner.
                memory = CANNMemory(diagnostics=False)
                self._memory = memory
                self._select = memory.select

            self._worker_factory = worker_factory
            for rank in range(len(devices)):
                self._select(rank)
                worker = worker_factory(rank)
                self._worker_owners[rank] = worker
                self.workers.append(worker)
                self.stats["worker_initializations"] += 1
            self._stack.callback(self._close_allocations)
            self.stats["worker_initialization_seconds"] = perf_counter() - begin
        except BaseException as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        try:
            self.close()
        except BaseException:
            try:
                self.profiling.finish_chunk(completed=False)
            except BaseException:
                _LOGGER.exception("Profiling finalization failed during resident cleanup")
            if error is None:
                raise
            _LOGGER.exception("Resident cleanup failed while unwinding %s", error_type.__name__)

    def _close_worker(self, rank, worker):
        self._select(rank)
        worker.close()

    def _close_allocations(self):
        """Free all completed session allocations before closing their workers."""
        first_error = None
        for rank, allocations in enumerate(self.allocations):
            self._select(rank)
            if rank in self.failed_ranks:
                allocations.clear()
                continue
            for allocation in reversed(allocations):
                try:
                    self.workers[rank].free_tensor(allocation)
                except BaseException as error:
                    first_error = first_error or error
                    _LOGGER.exception("Session allocation cleanup failed on rank %s", rank)
            allocations.clear()
        if first_error is not None:
            raise first_error

    def close(self):
        """Drain every communicator and worker, including partial initialization."""
        self.collective_queue.shutdown(wait=True, cancel_futures=True)
        for queue in self.compute_queues:
            queue.shutdown(wait=True, cancel_futures=True)
        with self._exchange_lock, ExitStack() as dispatches:
            for lock in self._dispatch_locks:
                dispatches.enter_context(lock)
            self._closed = True
            # A failed shutdown retains all device owners for a later retry.
            if self._exchange_executor is not None:
                self._exchange_executor.shutdown()
                self._exchange_executor = None
            remaining, errors = [], []
            for channel in self._transport_channels:
                if channel is None:
                    continue
                try:
                    channel.close()
                except BaseException as error:
                    remaining.append(channel)
                    errors.append(error)
            self._transport_channels = remaining
            if errors:
                raise errors[0]
            self.channels.clear()
            self._stack.close()
            for owners in (self._staging_owners, self._worker_owners):
                for rank in reversed(range(len(owners))):
                    worker = owners[rank]
                    if worker is None:
                        continue
                    try:
                        self._close_worker(rank, worker)
                        owners[rank] = None
                    except BaseException as error:
                        errors.append(error)
            if errors:
                raise errors[0]
            self.profiling.finish_chunk(completed=False)
            self.pinned.clear()
            self.device_owners.clear()
            self.rotations.clear()
            self.device_readonly.clear()
            self.programs.clear()
            self.static_parameters.clear()
            self.dynamic.clear()
            self.staging_workers.clear()
            self.device_active.clear()
            self.alias_leases.clear()
            self.parameter_abis.clear()
            self.values.clear()
            self.free_values.clear()
            self.free_workspaces.clear()
            self._device_views.clear()
            self._dispatch_cache.clear()

    def map_ranks(self, function):
        """Submit independent ranks before joining their ordered per-rank queues."""
        if self._closed or self.failed_ranks:
            raise RuntimeError("A closed or failed runtime cannot submit another stage")
        def invoke(rank):
            self._select(rank)
            started = perf_counter()
            try:
                return function(rank)
            except BaseException:
                self.failed_ranks.add(rank)
                raise
            finally:
                self.trace.append(dict(kind="rank_stage", stage=function.__name__, rank=rank,
                                       start=started, end=perf_counter()))
        pending = [queue.submit(invoke, rank) for rank, queue in enumerate(self.compute_queues)]
        failure, results = None, []
        for future in pending:
            try:
                results.append(future.result())
            except BaseException as error:
                failure = failure or error
        if failure is not None:
            raise failure
        return results

    def pin(self, value, ranks, *, readonly, name):
        """Upload one owning allocation per rank and retain all typed aliases."""
        if self._closed:
            raise RuntimeError("Resident runtime is closed")
        full_host = _host_bytes(value)
        for rank in ranks:
            self._select(rank)
            key = (rank, value.untyped_storage()._cdata)
            existing = self.pinned.get(key)
            if existing is not None:
                if existing["readonly"] != readonly:
                    raise ValueError("An allocation cannot be both a weight and mutable state: " + name)
                existing["versions"][id(value)] = (value, value._version)
                offset = value.storage_offset() * value.element_size() - existing["base_offset"]
                if offset < 0 or offset + value.numel() * value.element_size() > existing["allocation"].nbytes:
                    raise ValueError("Resident weight views require one complete owning span per rank")
                continue
            offset = value.storage_offset() * value.element_size() if readonly else 0
            extent = value.numel() * value.element_size() if readonly else full_host.numel()
            host = full_host.narrow(0, offset, extent)
            upload_begin = perf_counter()
            allocation = self.workers[rank].alloc_tensor(host.shape, host.dtype, init=host)
            self.allocations[rank].append(allocation)
            self.device_owners[id(allocation.buffer)] = rank
            self.device_readonly[id(allocation.buffer)] = readonly
            self.device_active[id(allocation.buffer)] = True
            self.pinned[key] = dict(host=host, nbytes=extent, source=value, allocation=allocation, readonly=readonly,
                                    base_offset=offset,
                                    name=name, versions={id(value): (value, value._version)})
            kind = "weight_bytes" if readonly else "state_bytes"
            self.stats[kind][rank] += allocation.nbytes
            self._record_transfer(rank, "h2d_bytes", allocation.nbytes)
            if readonly:
                self.stats["device_weight_upload_calls"] += 1
                self.stats["device_weight_upload_seconds"] += perf_counter() - upload_begin
            elif name.startswith("raw/"):
                self._record_transfer(rank, "cache_h2d_bytes", allocation.nbytes)

    def handle(self, value, rank):
        """Return a typed alias of an already uploaded host allocation."""
        if self._closed:
            raise RuntimeError("Resident runtime is closed")
        owner = self.pinned[rank, value.untyped_storage()._cdata]
        result = _device_view(owner["allocation"], value, owner["base_offset"])
        self.alias_leases[id(result)] = result
        return result

    def reset_ranges(self, value, rank, ranges):
        """Synchronously zero bounded byte spans without replacing their owning allocation."""
        if self._closed or rank in self.failed_ranks:
            raise RuntimeError("Cannot certify a page reset on a failed resident worker")
        owner = self.pinned[rank, value.untyped_storage()._cdata]
        if owner["readonly"]:
            raise ValueError("Request reset cannot modify resident weights")
        host = _host_bytes(value)
        allocation = owner["allocation"]
        buffer, base = allocation.buffer, allocation.byte_offset
        if (buffer is None or type(base) is not int or base < 0
                or int(buffer.base) + base != allocation.data_ptr
                or base + allocation.nbytes > int(buffer.nbytes)
                or host.numel() != allocation.nbytes):
            raise ValueError("Page reset requires a live, bounded owning byte allocation")
        ranges = tuple(ranges)
        for offset, length in ranges:
            if (type(offset) is not int or type(length) is not int
                    or offset < 0 or length <= 0 or offset + length > host.numel()):
                raise ValueError("Page reset escapes its owning byte allocation")
        self._select(rank)
        worker = self.workers[rank]
        for offset, length in ranges:
            zeros = host.narrow(0, offset, length).zero_()
            if self._native_dispatch:
                worker._impl.copy_to(buffer, zeros.numpy(),
                                     dst_offset=base + offset, src_offset=0, nbytes=length)
            else:
                self._copy(worker, allocation.data_ptr + offset, zeros.data_ptr(), length, 1)
            self.stats["page_reset_bytes"] = self.stats.get("page_reset_bytes", 0) + length
        for tensor, _version in list(owner["versions"].values()):
            owner["versions"][id(tensor)] = (tensor, tensor._version)

    def register(self, program):
        """Admit every phase and boundary entry to the same worker registry."""
        if self._closed:
            raise RuntimeError("Resident runtime is closed")
        static_names = {
            "weight", "hc_function", "hc_fn", "hc_scale", "hc_base", "norm_weight",
            "wq_a", "wq_a_scale", "q_norm", "wq_b", "wq_b_scale", "wkv", "wkv_scale", "kv_norm",
            "sink", "weight_a", "weight_b", "ffn_norm", "bias", "bias_vl",
            "gate_up_weight", "gate_up_scale", "down_weight", "down_scale",
            "wgate", "gamma", "channel_scale", "q_weight", "k_weight", "rotation",
        }
        phase_entries = {}
        for entries in program.stages.values():
            for name, entry in entries.items():
                phase_entries[id(entry.compiled)] = entry.compiled
                if self._native_dispatch and id(entry.compiled) not in self.programs:
                    entry.compiled.chip_callable
                self.programs[id(entry.compiled)] = entry.compiled
                names = static_names | {"gate"} if name == "ffn_dispatch_l2" else static_names
                self.static_parameters[id(entry.compiled)] = names.intersection(entry.compiled.param_names)
        if len(phase_entries) > 64:
            raise ValueError("One execution shape exceeds the worker's 64 program slots")

    def weight(self, value, rank):
        """Upload a complete immutable owned expert operand without retaining host payloads."""
        if self._closed:
            raise RuntimeError("Resident runtime is closed")
        self._select(rank)
        value = value.contiguous()
        upload_begin = perf_counter()
        result = self.workers[rank].alloc_tensor(value.shape, value.dtype, init=value)
        self.allocations[rank].append(result)
        self.device_owners[id(result.buffer)] = rank
        self.device_readonly[id(result.buffer)] = True
        self.device_active[id(result.buffer)] = True
        self.stats["weight_bytes"][rank] += result.nbytes
        self._record_transfer(rank, "h2d_bytes", result.nbytes)
        self.stats["device_weight_upload_calls"] += 1
        self.stats["device_weight_upload_seconds"] += perf_counter() - upload_begin
        self.alias_leases[id(result)] = result
        return result

    def staging_worker(self, rank):
        """Create a separate public worker whose buffers never alias resident allocations."""
        if self._closed:
            raise RuntimeError("Resident runtime is closed")
        self._select(rank)
        if rank not in self.staging_workers:
            worker = self._worker_factory(rank)
            self._staging_owners[rank] = worker
            self.staging_workers[rank] = worker
            self.stats["worker_initializations"] += 1
            self.stats["staging_worker_initializations"] += 1
        return self.staging_workers[rank]

    def execute(self, compiled, rank, arguments, abi, *, staged=False, empty_capture=False):
        """Dispatch shared arithmetic with cached bindings and device-owned operands."""
        with self._dispatch_locks[rank]:
            self._execute(compiled, rank, arguments, abi, staged=staged, empty_capture=empty_capture)

    def _execute(self, compiled, rank, arguments, abi, *, staged, empty_capture=False):
        from pypto.runtime import DeviceTensor

        if self._closed:
            raise RuntimeError("Resident runtime is closed")
        if len(arguments) != len(abi["params"]):
            raise ValueError("Resident dispatch requires the complete compiled parameter ABI")
        worker = self.workers[rank]
        bound = []
        self._select(rank)
        for value, parameter, direction in zip(arguments, abi["params"], abi["effective_directions"]):
            if parameter["shape"] is None:
                if isinstance(value, (torch.Tensor, DeviceTensor)) or direction != "In":
                    raise ValueError("Resident scalar ABI differs")
                bound.append(value)
                continue
            if isinstance(value, DeviceTensor):
                if self.device_owners.get(id(value.buffer)) != rank:
                    raise ValueError("Resident device operand belongs to another rank or runtime")
                if not self.device_active.get(id(value.buffer)) or id(value) not in self.alias_leases:
                    raise ValueError("Resident device operand is retired or released")
                if direction != "In" and self.device_readonly[id(value.buffer)]:
                    raise ValueError("A compiled entry writes an immutable resident weight")
                bound.append(value)
                continue
            if value.device.type != "cpu" or not value.is_contiguous():
                raise ValueError("Resident staging requires contiguous CPU tensors")
            key = rank, value.untyped_storage()._cdata
            if (key not in self.pinned and parameter.get("name")
                    in self.static_parameters.get(id(compiled), ())):
                raise RuntimeError("Static operand is missing its resident handle: " + parameter["name"])
            owner = self._value_owner(value, rank, initialize=direction != "Out")
            if direction != "In":
                if owner["readonly"]:
                    raise ValueError("A compiled entry writes an immutable resident weight")
                owner["device_written"] = True
            bound.append(self._view(owner["allocation"], value, owner.get("base_offset", 0)))
        started = perf_counter()
        capture, call_config = self.profiling.prepare_call(compiled, rank, staged, bound, abi["params"],
                                                         empty_capture=empty_capture, ring_heap=self.ring_heap)
        try:
            # Retire completed callable slots; tensor allocations belong to the worker.
            if self._native_dispatch:
                cache = worker._cid_cache
                key = id(compiled.chip_callable)
                if key not in cache and len(cache) >= 64:
                    oldest = next(iter(cache))
                    worker._impl.unregister(cache[oldest])
                    del cache[oldest]
                    self.stats["program_evictions"] += 1
            self._run_compiled(compiled, rank, staged, worker, bound, call_config=call_config)
            if capture is not None:
                capture["status"] = "completed"
        except BaseException as error:
            if capture is not None:
                capture.update(status="failed", error_type=type(error).__name__, error=str(error))
            self.failed_ranks.add(rank)
            raise
        finally:
            if capture is not None:
                capture.update(host_start=started, host_end=perf_counter())
                write_json(Path(capture["directory"]) / "dispatch.json", capture)
        self.trace.append(dict(kind="worker_run", rank=rank, start=started, end=perf_counter()))
        self.stats["dispatches"] += 1

    def _view(self, allocation, value, base_offset):
        """Retain bounded immutable typed views on each live device allocation."""
        key = value.storage_offset() * value.element_size() - base_offset, tuple(value.shape), value.dtype
        record = self._device_views.get(id(allocation))
        if record is None:
            record = allocation, {}
            self._device_views[id(allocation)] = record
        views = record[1]
        view = views.get(key)
        if view is None:
            view = _device_view(allocation, value, base_offset)
            if len(views) == 8:
                views.pop(next(iter(views)))
            views[key] = view
        return view

    def _copy(self, worker, destination, source, nbytes, direction):
        """Copy complete staging spans while the selected worker owns their allocations."""
        if self._memory is not None:
            self._memory.copy(destination, source, nbytes, direction)
        elif direction == 1:
            worker.copy_to(destination, source, nbytes)
        else:
            worker.copy_from(destination, source, nbytes)

    def _run_compiled(self, compiled, rank, staged, worker, bound, *, call_config=None):
        """Reuse validated wire descriptors for the same live device spans and scalar values."""
        from pypto.runtime import DeviceTensor

        if not self._native_dispatch and call_config is None:
            if self._run_config is None:
                worker.run(compiled, *bound)
            else:
                worker.run(compiled, *bound, config=self._run_config)
            return
        worker._check_binding(compiled)
        runtime_config = compiled.runtime_config
        thread_count = runtime_config.get("aicpu_thread_num")
        host_indices = runtime_config.get("host_tensor_indices", ())
        cacheable = (type(host_indices) in (tuple, list) and not host_indices
                     and type(thread_count) in (int, type(None)))
        signature = []
        for value in bound:
            if isinstance(value, DeviceTensor):
                signature.append((id(value.buffer), value.data_ptr, value.shape, value.dtype, value.byte_offset))
            elif type(value) is float:
                signature.append((float, struct.pack("!d", value)))
            elif type(value) in (bool, int):
                signature.append((type(value), value))
            else:
                cacheable = False
                break
        if not cacheable:
            if call_config is None:
                if self._run_config is None:
                    worker.run(compiled, *bound)
                else:
                    worker.run(compiled, *bound, config=self._run_config)
            else:
                arguments, coerced, _ = compiled._build_orch_args(*bound, worker=worker._impl)
                worker._run_chip(compiled.chip_callable, arguments, call_config)
            return
        key = id(compiled), thread_count, tuple(signature)
        cache = self._dispatch_cache.setdefault((rank, staged), OrderedDict())
        record = cache.get(key)
        if record is None:
            arguments, coerced, _ = compiled._build_orch_args(*bound, worker=worker._impl)
            config = compiled._build_call_config(self._run_config)
            record = compiled, coerced, compiled.chip_callable, arguments, config
            cache[key] = record
            if len(cache) > 64:
                cache.popitem(last=False)
        else:
            for value in bound:
                if isinstance(value, DeviceTensor):
                    worker._require_owned_resident_tensor(value, "Resident dispatch argument")
            cache.move_to_end(key)
        worker._run_chip(record[2], record[3], record[4] if call_config is None else call_config)

    def _value_owner(self, value, rank, *, initialize=True):
        with self.rank_locks[rank]:
            return self._value_owner_locked(value, rank, initialize=initialize)

    def _value_owner_locked(self, value, rank, *, initialize=True):
        """Resolve a host storage descriptor to its authoritative device allocation."""
        if self._closed:
            raise RuntimeError("Resident runtime is closed")
        host = _host_bytes(value)
        key = rank, value.untyped_storage()._cdata
        owner = self.pinned.get(key)
        if owner is not None:
            changed = any(tensor._version != version for tensor, version in owner["versions"].values())
            if changed:
                if owner["readonly"] or owner["name"].startswith("raw/") or owner.get("device_written"):
                    raise ValueError("A device-owned allocation changed on the host: " + owner["name"])
                copy = owner["host"]
                self._copy(self.workers[rank], owner["allocation"].data_ptr, copy.data_ptr(), copy.numel(), 1)
                self._record_transfer(rank, "h2d_bytes", copy.numel())
                owner["versions"] = {index: (tensor, tensor._version)
                                     for index, (tensor, _) in owner["versions"].items()}
            owner["versions"][id(value)] = value, value._version
            return owner
        owner = self.values.get(key)
        if owner is not None:
            if owner["source"]._version != owner["version"]:
                if owner["device_written"]:
                    raise ValueError("A device activation descriptor was modified on the host")
                self._copy(self.workers[rank], owner["allocation"].data_ptr, host.data_ptr(), host.numel(), 1)
                self._record_transfer(rank, "h2d_bytes", host.numel())
                owner["version"] = value._version
            return owner
        pool = self.free_values.setdefault((rank, host.numel()), [])
        if pool:
            allocation = pool.pop()
            if initialize:
                self._copy(self.workers[rank], allocation.data_ptr, host.data_ptr(), host.numel(), 1)
        else:
            allocation = self.workers[rank].alloc_tensor(host.shape, host.dtype, init=host if initialize else None)
            self.allocations[rank].append(allocation)
            self.device_owners[id(allocation.buffer)] = rank
            self.device_readonly[id(allocation.buffer)] = False
            self.stats["dynamic_allocations"] += 1
        self.device_active[id(allocation.buffer)] = True
        if initialize:
            self._record_transfer(rank, "h2d_bytes", host.numel())
        owner = dict(source=value, version=value._version, allocation=allocation, readonly=False,
                     device_written=False)
        self.values[key] = owner
        return owner

    def _record_transfer(self, rank, kind, amount):
        """Accumulate aggregate and per-rank transfer counters under one lock."""
        with self._stats_lock:
            self.stats[kind] += amount
            row = self.transfer_stats[rank]
            row[kind] = row.get(kind, 0) + amount

    def _fill_device(self, destination, nbytes, value, rank):
        self._memory_for(rank).fill(destination, nbytes, value)

    def workspace(self, shape, dtype, rank, *, fill=0):
        """Reuse device scratch, optionally skipping fill for a full overwrite."""
        integral = dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        if fill is not None and fill != 0 and not (fill == -1 and integral):
            raise ValueError("Device workspaces support no fill, zero or integer -1 initialization")
        shape = tuple(shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("Device workspaces require positive dimensions")
        with self.rank_locks[rank]:
            if self._closed:
                raise RuntimeError("Resident runtime is closed")
            self._select(rank)
            pool = self.free_workspaces.setdefault((rank, shape, dtype), [])
            if pool:
                owner = pool.pop()
                value = owner["source"]
                owner["version"] = value._version
                self.values[rank, value.untyped_storage()._cdata] = owner
                self.device_active[id(owner["allocation"].buffer)] = True
                self._record_transfer(rank, "workspace_reuses", 1)
            else:
                value = torch.empty(shape, dtype=dtype)
                owner = self._value_owner_locked(value, rank, initialize=False)
                owner["workspace"] = True
            allocation = owner["allocation"]
            if fill is not None:
                self._device_fill(allocation.data_ptr, allocation.nbytes, 0 if fill == 0 else 255, rank)
                owner["device_written"] = True
                self._record_transfer(rank, "workspace_fill_bytes", allocation.nbytes)
            self._record_transfer(rank, "workspace_requests", 1)
            return value

    def tensor(self, value, rank):
        """Return a typed device alias without copying an authoritative payload."""
        self._select(rank)
        owner = self._value_owner(value, rank)
        result = _device_view(owner["allocation"], value, owner.get("base_offset", 0))
        self.alias_leases[id(result)] = result
        return result

    def _readback(self, value, rank, kind):
        """Copy an explicitly requested host boundary without changing device authority."""
        if self._closed:
            raise RuntimeError("Resident runtime is closed")
        self._select(rank)
        key = rank, value.untyped_storage()._cdata
        if key not in self.values and key not in self.pinned:
            return value.clone()
        owner = self._value_owner(value, rank)
        source = _device_view(owner["allocation"], value, owner.get("base_offset", 0))
        result = torch.empty(value.shape, dtype=value.dtype)
        if source.byte_offset:
            self._device_readback(result.data_ptr(), source.data_ptr, source.nbytes, rank)
        else:
            self._copy(self.workers[rank], result.data_ptr(), source.data_ptr, source.nbytes, 2)
        self._record_transfer(rank, "d2h_bytes", source.nbytes)
        self._record_transfer(rank, kind + "_d2h_bytes", source.nbytes)
        return result

    def read_control(self, value, rank):
        """Read integer routing/index/status controls; reject activation payloads."""
        if value.dtype not in (torch.int32, torch.int64, torch.bool):
            raise ValueError("Only integer control tensors may cross an intermediate host boundary")
        return self._readback(value, rank, "control")

    def read_output(self, value, rank):
        """Materialize the public vocabulary output at the end of a forward."""
        return self._readback(value, rank, "output")

    def _memory_for(self, rank):
        from .moe_transport import CANNMemory

        if self._memory is None:
            self._memory = CANNMemory(diagnostics=False)
        self._memory.select(rank)
        return self._memory

    def _copy_device(self, destination, source, nbytes, rank):
        memory = self._memory_for(rank)
        status = memory.library.aclrtMemcpy(destination, nbytes, source, nbytes, 3)
        if status:
            raise RuntimeError(f"Device activation copy returned CANN status {status}")

    def _copy_from_device(self, destination, source, nbytes, rank):
        """Read a typed interior alias through CANN's byte-addressed interface."""
        memory = self._memory_for(rank)
        status = memory.library.aclrtMemcpy(destination, nbytes, source, nbytes, 2)
        if status:
            raise RuntimeError(f"Device boundary readback returned CANN status {status}")

    def copy_tensor(self, destination, source, rank):
        """Copy contiguous activation aliases directly between device buffers."""
        if self._closed:
            raise RuntimeError("Resident runtime is closed")
        if destination.shape != source.shape or destination.dtype != source.dtype:
            raise ValueError("Device copy requires matching tensor shapes and dtypes")
        if not destination.numel():
            return destination
        key = rank, destination.untyped_storage()._cdata
        if key in self.pinned and self.pinned[key]["readonly"]:
            raise ValueError("Cannot copy into an immutable device weight")
        source_key = rank, source.untyped_storage()._cdata
        source_owner = self.pinned.get(source_key, self.values.get(source_key))
        target_owner = self.pinned.get(key, self.values.get(key))
        if (source_owner is None or not source_owner.get("device_written")) and (
                target_owner is None or not target_owner.get("device_written")):
            destination.copy_(source)
            return destination
        destination_owner = self._value_owner(destination, rank)
        source_owner = self._value_owner(source, rank)
        target = _device_view(destination_owner["allocation"], destination, destination_owner.get("base_offset", 0))
        value = _device_view(source_owner["allocation"], source, source_owner.get("base_offset", 0))
        self._device_copy(target.data_ptr, value.data_ptr, value.nbytes, rank)
        self._record_transfer(rank, "d2d_bytes", value.nbytes)
        key = rank, destination.untyped_storage()._cdata
        owner = self.pinned.get(key, self.values.get(key))
        if owner["readonly"]:
            raise ValueError("Cannot copy into an immutable device weight")
        owner["device_written"] = True
        return destination

    def release_values(self, keep=()):
        """Recycle completed layer scratch while retaining live layer-boundary aliases."""
        if self._closed:
            raise RuntimeError("Resident runtime is closed")
        retained = {(rank, value.untyped_storage()._cdata) for rank, value in keep}
        self._retire_values(set(self.values) - retained)

    def release_tensors(self, values):
        """Recycle completed communication roots after all rank consumers finish."""
        if self._closed:
            raise RuntimeError("Resident runtime is closed")
        keys = {(rank, value.untyped_storage()._cdata) for rank, value in values}
        if keys.intersection(self.pinned):
            raise ValueError("Cannot recycle pinned weights or cache storage")
        self._retire_values(keys.intersection(self.values))

    def _retire_values(self, keys):
        """Return retired allocations to their owning rank's reuse pool."""
        retired = set()
        for key in keys:
            owner = self.values.pop(key)
            allocation = owner["allocation"]
            retired.add(id(allocation.buffer))
            self.device_active[id(allocation.buffer)] = False
            if owner.get("workspace"):
                value = owner["source"]
                shape_key = key[0], tuple(value.shape), value.dtype
                self.free_workspaces.setdefault(shape_key, []).append(owner)
            else:
                self.free_values.setdefault((key[0], allocation.nbytes), []).append(allocation)
        self.alias_leases = {key: value for key, value in self.alias_leases.items() if id(value.buffer) not in retired}

    def _group_ranks(self, group):
        world, tp = self.topology["ep"], self.topology["tp"]
        groups = dict(ep=[list(range(world))],
                      tp=[list(range(start, start + tp)) for start in range(0, world, tp)],
                      dp=[list(range(lane, world, tp)) for lane in range(tp)])
        if group not in groups:
            raise ValueError("Unknown resident HCCL group: " + group)
        return groups[group]

    def load_rotation(self, checkpoint, hidden_size):
        """Retain the validated checkpoint rotation for both slots and subsequent tokens."""
        from .engram_host import load_rotation

        key = str(checkpoint), hidden_size
        if key not in self.rotations:
            self.rotations[key] = load_rotation(checkpoint, hidden_size)
        return self.rotations[key]

    def exchange(self, stage, packets, collective, group, *, host_control=False):
        """Exchange device-owned payloads or explicit startup controls through HCCL."""
        with self._exchange_lock:
            return self._exchange(stage, packets, collective, group, host_control=host_control)

    def _exchange_rank_tasks(self, function, ranks):
        """Quarantine rank resources after a failed batch until close drains them."""
        try:
            return self._exchange_executor.run(function, ranks)
        except BaseException:
            self._closed = True
            raise

    def _exchange(self, stage, packets, collective, group, *, host_control):
        from .moe_transport import CANNMemory, HCCLBytes

        if self._closed:
            raise RuntimeError("Resident runtime is closed")
        world = len(self.workers)
        if len(packets) != world or not packets:
            raise ValueError("Every allocated resident rank must participate in a collective")
        schema = [(case, tuple(packet)) for case, packet in packets[0].items()]
        if any([(case, tuple(packet)) for case, packet in row.items()] != schema for row in packets):
            raise ValueError("Resident collective rank packet orders differ")
        if collective not in ("all_gather", "all_to_all", "reduce_scatter", "all_reduce"):
            raise ValueError("Unsupported resident collective")
        groups = self._group_ranks(group)
        for ranks in groups:
            for case, names in schema:
                for name in names:
                    reference = packets[ranks[0]][case][name]
                    if not isinstance(reference, torch.Tensor):
                        raise ValueError("Resident collective requires actual tensor packets")
                    for rank in ranks:
                        value = packets[rank][case][name]
                        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                                or not value.is_contiguous() or value.ndim < 1 or value.numel() == 0
                                or value.shape != reference.shape or value.dtype != reference.dtype):
                            raise ValueError("Resident collective peer tensor ABIs differ")
                        partitioned = collective in ("all_to_all", "reduce_scatter")
                        if partitioned and value.numel() * value.element_size() % len(ranks):
                            raise ValueError("Resident collective byte count does not divide across peers")
                        if collective == "reduce_scatter" and (
                                value.dtype not in (torch.bfloat16, torch.float32) or value.shape[0] % len(ranks)):
                            raise ValueError("Resident reduce-scatter requires equal BF16/FP32 row buckets")
                        if collective == "all_reduce" and value.dtype not in (torch.bfloat16, torch.float32):
                            raise ValueError("Resident all-reduce requires BF16/FP32 values")
        if self._exchange_executor is None:
            # Keep each communicator on its own host thread. CANN context and
            # HCCL thread-local state must not migrate between logical ranks.
            self._exchange_executor = _RankExecutor(world)
            try:
                self._exchange_executor.start()
            except BaseException:
                self._closed = True
                raise
        if group not in self.channels:
            memory = CANNMemory(diagnostics=False)
            channels = {}
            for ranks in groups:
                first_slot = len(self._transport_channels)
                slots = {rank: first_slot + index for index, rank in enumerate(ranks)}
                # Reserve ownership slots before creating any native channel.
                self._transport_channels.extend([None] * len(ranks))

                def retain(channel):
                    self._transport_channels[slots[channel.rank]] = channel

                channels.update(HCCLBytes.create_group(memory, ranks, _register_owner=retain))
            self.channels[group] = channels
        channels, outputs = self.channels[group], [{} for _ in packets]
        begin = perf_counter()

        prepared = [None] * world

        def prepare(rank):
            channel = channels[rank]
            transfers, destination_owners = [], []
            for case, names in schema:
                outputs[rank][case] = {}
                for name in names:
                    value = packets[rank][case][name]
                    identity = stage + "/" + case + "/" + name
                    shape = list(value.shape)
                    if collective == "all_gather":
                        shape[0] *= len(channel.group_ranks)
                    elif collective == "reduce_scatter":
                        shape[0] //= len(channel.group_ranks)
                    received = torch.empty(shape, dtype=value.dtype)
                    self._select(rank)
                    owner = self._value_owner(received, rank, initialize=False)
                    destination = _device_view(owner["allocation"], received)
                    source_owner = self._value_owner(value, rank)
                    source = _device_view(source_owner["allocation"], value, source_owner.get("base_offset", 0))
                    transfers.append((source, destination, identity))
                    destination_owners.append(owner)
                    outputs[rank][case][name] = received
            prepared[rank] = transfers, destination_owners

        def transfer(rank):
            channel = channels[rank]
            if host_control:
                for case, names in schema:
                    outputs[rank][case] = {}
                    for name in names:
                        identity = stage + "/" + case + "/" + name
                        outputs[rank][case][name] = channel.exchange(packets[rank][case][name], identity, collective)
                return
            transfers, destination_owners = prepared[rank]
            if transfers:
                try:
                    channel.exchange_device_batch(transfers, collective)
                except BaseException:
                    self.failed_ranks.add(rank)
                    raise
                for owner in destination_owners:
                    owner["device_written"] = True

        if not host_control:
            self._exchange_rank_tasks(prepare, range(world))
        self._exchange_rank_tasks(transfer, range(world))
        seconds = perf_counter() - begin
        self.trace.append(dict(kind="collective", stage=stage, start=begin, end=perf_counter()))
        self.stats["transport_calls"] += 1
        self.stats["transport_seconds"] += seconds
        receipts = [dict(rank=rank, backend="device CANN/HCCL", activation_staging=False, group_ranks=channels[rank].group_ranks,
                         collective=collective, stage=stage, seconds=seconds, files_used=False,
                         field_count=sum(len(names) for _, names in schema),
                         stream_waits_per_rank=sum(len(names) for _, names in schema) if host_control else 1,
                         memory=dict(scope="Transport buffers only; not total HBM or a peak",
                                     samples=[], intervals=[], transport_buffer_peak_bytes=channels[rank].buffers_peak))
                    for rank in range(world)]
        return outputs, receipts

    def summary(self):
        """Report actual allocations and dispatches separately from numerical acceptance."""
        return dict(self.stats, transfer_by_rank=[dict(rank=rank, **row) for rank, row in enumerate(self.transfer_stats)],
                    profiling=self.profiling.summary(),
                    ring_heap=self.ring_heap,
                    assigned_devices=list(self.devices),
                    runtime_identity=id(self), worker_identities=[id(worker) for worker in self.workers],
                    registered_programs=len(self.programs), closed=self._closed,
                    compute_queues=len(self.compute_queues), communication_threads=len(self.devices),
                    execution_model="persistent_A3_ChipWorkers_with_host_layer_scheduler",
                    single_device_graph=False, cache_readback="integer controls only",
                    activation_residency="device storage aliases across stages and layers",
                    numerical_acceptance="not_established_by_residency")

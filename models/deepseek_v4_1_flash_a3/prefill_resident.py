# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Retain A3 workers, typed device allocations and HCCL groups across a complete generation."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import os
from time import perf_counter

import torch


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


class ResidentNetworkRuntime:
    """Own resident compute workers and independent CPU-staged IO workers per logical rank."""

    def __init__(self, devices, topology, *, worker_factory=None):
        begin = perf_counter()
        self.devices, self.topology = tuple(devices), dict(topology)
        os.environ.setdefault("HCCL_EXEC_TIMEOUT", "180")
        os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "180")
        if len(self.devices) != topology["ep"] or len(set(self.devices)) != len(self.devices):
            raise ValueError("Resident runtime requires the exact complete distinct device mapping")
        self.workers, self.pinned, self.channels = [], {}, {}
        self.device_owners = {}
        self.staging_workers = {}
        self.rotations = {}
        self._stack = ExitStack()
        self._closed = False
        self._select = lambda rank: None
        self.stats = dict(worker_initializations=0, dispatches=0, h2d_bytes=0, d2h_bytes=0,
                          weight_bytes=[0] * len(devices), state_bytes=[0] * len(devices),
                          transport_calls=0, transport_seconds=0.0,
                          staging_worker_initializations=0, staging_dispatches=0)
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
                self._select = memory.select

            self._worker_factory = worker_factory
            for rank in range(len(devices)):
                self._select(rank)
                worker = worker_factory(rank)
                self._stack.callback(self._close_worker, rank, worker)
                self.workers.append(worker)
                self.stats["worker_initializations"] += 1
            self.stats["worker_initialization_seconds"] = perf_counter() - begin
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    def _close_worker(self, rank, worker):
        self._select(rank)
        worker.close()

    def close(self):
        """Drain every communicator and worker, including partial initialization."""
        if not self._closed:
            self._closed = True
            self._stack.close()
            self.channels.clear()
            self.pinned.clear()
            self.device_owners.clear()
            self.rotations.clear()
            self.staging_workers.clear()

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
            allocation = self.workers[rank].alloc_tensor(host.shape, host.dtype, init=host)
            self.pinned[key] = dict(host=host, source=value, allocation=allocation, readonly=readonly,
                                    base_offset=offset,
                                    name=name, versions={id(value): (value, value._version)})
            kind = "weight_bytes" if readonly else "state_bytes"
            self.stats[kind][rank] += allocation.nbytes
            self.stats["h2d_bytes"] += allocation.nbytes

    def weight(self, value, rank):
        """Upload a complete immutable owned expert operand without retaining host payloads."""
        self._select(rank)
        value = value.contiguous()
        result = self.workers[rank].alloc_tensor(value.shape, value.dtype, init=value)
        self.device_owners[id(result.buffer)] = rank
        self.stats["weight_bytes"][rank] += result.nbytes
        self.stats["h2d_bytes"] += result.nbytes
        return result

    def staging_worker(self, rank):
        """Create a separate public worker whose buffers never alias resident allocations."""
        if self._closed:
            raise RuntimeError("Resident runtime is closed")
        self._select(rank)
        if rank not in self.staging_workers:
            worker = self._worker_factory(rank)
            self._stack.callback(self._close_worker, rank, worker)
            self.staging_workers[rank] = worker
            self.stats["worker_initializations"] += 1
            self.stats["staging_worker_initializations"] += 1
        return self.staging_workers[rank]

    def execute(self, compiled, rank, arguments, abi, *, staged=False):
        """Dispatch existing arithmetic with resident constants and exact aliased staging."""
        from pypto.runtime import DeviceTensor

        if self._closed:
            raise RuntimeError("Resident runtime is closed")
        if len(arguments) != len(abi["params"]):
            raise ValueError("Resident dispatch requires the complete compiled parameter ABI")
        worker = self.staging_worker(rank) if staged else self.workers[rank]
        local, bound = {}, []
        self._select(rank)
        writable, transient = {}, []
        try:
            for value, parameter, direction in zip(arguments, abi["params"], abi["effective_directions"]):
                if parameter["shape"] is None:
                    if isinstance(value, (torch.Tensor, DeviceTensor)) or direction != "In":
                        raise ValueError("Resident scalar ABI differs")
                    bound.append(value)
                    continue
                if isinstance(value, DeviceTensor):
                    if staged:
                        raise ValueError("Staged IO requires CPU tensors, not resident device operands")
                    if direction != "In":
                        raise ValueError("Owned expert operands must remain read-only")
                    if self.device_owners.get(id(value.buffer)) != rank:
                        raise ValueError("Resident device operand belongs to another rank or runtime")
                    bound.append(value)
                    continue
                host = _host_bytes(value)
                key = (rank, value.untyped_storage()._cdata)
                resident_owner = self.pinned.get(key)
                if staged and resident_owner is not None and direction != "In":
                    raise ValueError("Staged IO may not write resident state or immutable weights")
                owner = None if staged else resident_owner
                if owner is not None:
                    if owner["readonly"] and direction != "In":
                        raise ValueError("A compiled entry writes an immutable resident weight")
                    changed = any(tensor._version != version for tensor, version in owner["versions"].values())
                    if changed or (not owner["readonly"] and id(value) not in owner["versions"]):
                        if owner["readonly"]:
                            raise ValueError("Immutable resident weight changed on the host")
                        copy = owner["host"]
                        worker.copy_to(owner["allocation"].data_ptr, copy.data_ptr(), copy.numel())
                        self.stats["h2d_bytes"] += copy.numel()
                        owner["versions"] = {key: (tensor, tensor._version)
                                             for key, (tensor, _) in owner["versions"].items()}
                    owner["versions"][id(value)] = (value, value._version)
                elif key in local:
                    owner = local[key]
                else:
                    allocation = worker.alloc_tensor(host.shape, host.dtype, init=host)
                    transient.append(allocation)
                    owner = local[key] = dict(host=host, allocation=allocation, readonly=False)
                    self.stats["h2d_bytes"] += allocation.nbytes
                bound.append(_device_view(owner["allocation"], value, owner.get("base_offset", 0)))
                if direction != "In":
                    writable[key] = owner
            worker.run(compiled, *bound)
            self.stats["dispatches"] += 1
            self.stats["staging_dispatches"] += int(staged)
            for owner in writable.values():
                host, allocation = owner["host"], owner["allocation"]
                worker.copy_from(host.data_ptr(), allocation.data_ptr, host.numel())
                self.stats["d2h_bytes"] += host.numel()
        finally:
            for allocation in reversed(transient):
                worker.free_tensor(allocation)

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

    def exchange(self, stage, packets, collective, group):
        """Exchange actual CPU boundaries through persistent HCCL without files or subprocesses."""
        from .moe_transport import CANNMemory, HCCLBytes

        if self._closed:
            raise RuntimeError("Resident runtime is closed")
        world = len(self.workers)
        if len(packets) != world or not packets:
            raise ValueError("Every allocated resident rank must participate in a collective")
        schema = [(case, tuple(packet)) for case, packet in packets[0].items()]
        if any([(case, tuple(packet)) for case, packet in row.items()] != schema for row in packets):
            raise ValueError("Resident collective rank packet orders differ")
        if collective not in ("all_gather", "all_to_all", "reduce_scatter"):
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
                        if collective != "all_gather" and value.numel() * value.element_size() % len(ranks):
                            raise ValueError("Resident collective byte count does not divide across peers")
                        if collective == "reduce_scatter" and (
                                value.dtype not in (torch.bfloat16, torch.float32) or value.shape[0] % len(ranks)):
                            raise ValueError("Resident reduce-scatter requires equal BF16/FP32 row buckets")
        if group not in self.channels:
            memory = CANNMemory(diagnostics=False)
            channels = {}
            for ranks in groups:
                root_info = HCCLBytes.root_info(memory)

                def connect(rank):
                    channel = HCCLBytes(memory, rank, len(ranks), None, ranks, root_info=root_info)
                    self._stack.callback(channel.close)
                    return rank, channel

                with ThreadPoolExecutor(max_workers=len(ranks)) as pool:
                    channels.update(pool.map(connect, ranks))
            self.channels[group] = channels
        channels, outputs = self.channels[group], [{} for _ in packets]
        begin = perf_counter()

        def transfer(rank):
            channel = channels[rank]
            for case, names in schema:
                outputs[rank][case] = {}
                for name in names:
                    value = packets[rank][case][name]
                    outputs[rank][case][name] = channel.exchange(value, stage + "/" + case + "/" + name, collective)

        with ThreadPoolExecutor(max_workers=world) as pool:
            list(pool.map(transfer, range(world)))
        seconds = perf_counter() - begin
        self.stats["transport_calls"] += 1
        self.stats["transport_seconds"] += seconds
        receipts = [dict(rank=rank, backend="persistent CANN/HCCL", group_ranks=channels[rank].group_ranks,
                         collective=collective, stage=stage, seconds=seconds, files_used=False,
                         memory=dict(scope="Transport buffers only; not total HBM or a peak",
                                     samples=[], intervals=[], transport_buffer_peak_bytes=channels[rank].buffers_peak))
                    for rank in range(world)]
        return outputs, receipts

    def summary(self):
        """Report actual allocations and dispatches separately from numerical acceptance."""
        return dict(self.stats, assigned_devices=list(self.devices),
                    execution_model="persistent_A3_ChipWorkers_with_host_layer_scheduler",
                    single_device_graph=False, cache_readback="retained for current host consumers",
                    numerical_acceptance="not_established_by_residency")

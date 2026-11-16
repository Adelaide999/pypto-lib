# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exchange device-resident PyPTO route buckets through CANN and HCCL."""

import ctypes
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"
import torch


def _check(status, operation):
    if status:
        raise RuntimeError(f"{operation} returned CANN/HCCL status {status}")


def _bind(library, name, argument_types):
    function = getattr(library, name)
    function.argtypes = argument_types
    function.restype = ctypes.c_int
    return function


class CANNMemory:
    """Allocate/copy CANN memory and retain raw device-memory samples."""

    def __init__(self, *, diagnostics=False):
        self.diagnostics = diagnostics
        self.library = ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
        pointer, size, integer = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        _bind(self.library, "aclInit", [ctypes.c_char_p])
        _bind(self.library, "aclrtSetDevice", [integer])
        _bind(self.library, "aclrtCreateStream", [ctypes.POINTER(pointer)])
        _bind(self.library, "aclrtDestroyStream", [pointer])
        _bind(self.library, "aclrtSynchronizeStream", [pointer])
        _bind(self.library, "aclrtMalloc", [ctypes.POINTER(pointer), size, integer])
        _bind(self.library, "aclrtFree", [pointer])
        _bind(self.library, "aclrtMemcpy", [pointer, size, pointer, size, integer])
        _bind(self.library, "aclrtGetMemInfo", [integer, ctypes.POINTER(size), ctypes.POINTER(size)])
        status = self.library.aclInit(None)
        # A persistent ChipWorker can already own this process's ACL initialization.
        if status != 100002:  # ACL_ERROR_REPEAT_INITIALIZE
            _check(status, "aclInit")
        self.samples = []
        self.intervals = []

    @contextmanager
    def interval(self, rank, phase, stage):
        """Record a stage on the allocation observer's common host clock."""
        if not self.diagnostics:
            yield
            return
        begin = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        try:
            yield
        finally:
            end = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            self.intervals.append({"rank": rank, "phase": phase, "stage": stage,
                                   "begin_ns": begin, "end_ns": end})

    def select(self, rank):
        _check(self.library.aclrtSetDevice(rank), "aclrtSetDevice")

    def sample(self, rank, stage):
        self.select(rank)
        if not self.diagnostics:
            return None
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        _check(self.library.aclrtGetMemInfo(1, ctypes.byref(free), ctypes.byref(total)), "aclrtGetMemInfo")
        sample = {"rank": rank, "stage": stage, "time_ns": time.time_ns(),
                  "free_bytes": free.value, "total_bytes": total.value, "used_bytes": total.value - free.value}
        self.samples.append(sample)
        return sample


class HCCLBytes:
    """Exchange equal contiguous peer blocks without interpreting model tensors."""

    @classmethod
    def create_group(cls, memory, group_ranks):
        """Create one single-process communication domain in logical-device order."""
        ranks = list(group_ranks)
        if not ranks or len(set(ranks)) != len(ranks) or any(type(rank) is not int or rank < 0 for rank in ranks):
            raise ValueError("HCCL groups require distinct nonnegative logical device IDs")
        library = ctypes.CDLL("libhccl.so", mode=ctypes.RTLD_GLOBAL)
        pointer, u64, integer = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int
        init_types = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(pointer)]
        initialize = _bind(library, "HcclCommInitAll", init_types)
        _bind(library, "HcclCommDestroy", [pointer])
        _bind(library, "HcclAlltoAll", [pointer, u64, integer, pointer, u64, integer, pointer, pointer])
        _bind(library, "HcclAllGather", [pointer, pointer, u64, integer, pointer, pointer])
        _bind(library, "HcclAllReduce", [pointer, pointer, u64, integer, integer, pointer, pointer])
        _bind(library, "HcclReduceScatter", [pointer, pointer, u64, integer, integer, pointer, pointer])
        devices, comms = (ctypes.c_int32 * len(ranks))(*ranks), (pointer * len(ranks))()
        _check(initialize(len(ranks), devices, comms), "HcclCommInitAll")
        channels = {}
        try:
            for index, rank in enumerate(ranks):
                channel = cls(memory, rank, ranks, comms[index], library)
                channels[rank] = channel
                memory.select(rank)
                _check(memory.library.aclrtCreateStream(ctypes.byref(channel.stream)), "aclrtCreateStream")
        except BaseException:
            for index, rank in enumerate(ranks):
                try:
                    if rank in channels:
                        channels[rank].close()
                    elif comms[index]:
                        memory.select(rank)
                        _check(library.HcclCommDestroy(comms[index]), "HcclCommDestroy")
                except BaseException:
                    logging.getLogger(__name__).exception("HCCL group initialization cleanup failed")
            raise
        return channels

    def __init__(self, memory, rank, group_ranks, comm, library):
        self.memory, self.rank, self.world = memory, rank, len(group_ranks)
        self.group_ranks, self.library = list(group_ranks), library
        self.stream, self.comm = ctypes.c_void_p(), ctypes.c_void_p(comm)
        self.buffers_peak = 0
        self.transfers = []

    def _collective(self, send_buffer, recv_buffer, byte_count, output_count, dtype, stage, collective):
        """Complete one byte exchange or typed sum on the channel stream."""
        peer_bytes = byte_count // self.world
        with self.memory.interval(self.rank, "communication", stage):
            if collective == "all_to_all":
                arguments = [send_buffer, peer_bytes, 7, recv_buffer, peer_bytes, 7, self.comm, self.stream]
                status = self.library.HcclAlltoAll(*arguments)
            elif collective == "all_gather":
                status = self.library.HcclAllGather(send_buffer, recv_buffer, byte_count, 7, self.comm, self.stream)
            elif collective == "reduce_scatter":
                hccl_dtype = 11 if dtype == torch.bfloat16 else 4
                arguments = [send_buffer, recv_buffer, output_count, hccl_dtype, 0, self.comm, self.stream]
                status = self.library.HcclReduceScatter(*arguments)
            elif collective == "all_reduce":
                hccl_dtype = 11 if dtype == torch.bfloat16 else 4
                arguments = [send_buffer, recv_buffer, output_count, hccl_dtype, 0, self.comm, self.stream]
                status = self.library.HcclAllReduce(*arguments)
            else:
                raise ValueError("Unsupported basic collective: " + collective)
            _check(status, "HCCL/" + collective)
            _check(self.memory.library.aclrtSynchronizeStream(self.stream), "aclrtSynchronizeStream")

    def exchange_device(self, value, received, stage, collective="all_to_all"):
        """Exchange caller-owned device allocations after their producer completes.

        The caller validates rank ownership and retains both allocations through
        this synchronous call. This method does not allocate or copy host data.
        """
        from pypto.runtime import DeviceTensor

        if not isinstance(value, DeviceTensor) or not isinstance(received, DeviceTensor):
            raise ValueError("Device collectives require DeviceTensor input and output handles")
        if value.buffer is None or received.buffer is None:
            raise ValueError("Device collectives require owning allocation buffers")
        if not value.shape or value.nbytes <= 0 or value.data_ptr <= 0 or received.data_ptr <= 0:
            raise ValueError("Device collectives require nonempty tensor allocations")
        if value.dtype != received.dtype:
            raise ValueError("Device collective input and output dtypes differ")
        shape = tuple(value.shape)
        if collective == "all_gather":
            expected_shape = (self.world * shape[0], *shape[1:])
            expected_bytes = value.nbytes * self.world
        elif collective == "reduce_scatter":
            if value.dtype not in (torch.bfloat16, torch.float32) or shape[0] % self.world:
                raise ValueError("Reduce-scatter requires equal BF16/FP32 row buckets")
            expected_shape = (shape[0] // self.world, *shape[1:])
            expected_bytes = value.nbytes // self.world
        elif collective in ("all_to_all", "all_reduce"):
            expected_shape, expected_bytes = shape, value.nbytes
            if collective == "all_to_all" and value.nbytes % self.world:
                raise ValueError("All-to-all bytes must divide into equal peer blocks")
            if collective == "all_reduce" and value.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError("All-reduce requires BF16/FP32 values")
        else:
            raise ValueError("Unsupported basic collective: " + collective)
        if tuple(received.shape) != expected_shape or received.nbytes != expected_bytes:
            raise ValueError("Device collective output allocation has the wrong shape or byte extent")
        if max(value.data_ptr, received.data_ptr) < min(value.data_ptr + value.nbytes, received.data_ptr + received.nbytes):
            raise ValueError("Device collective input and output allocations overlap")
        self.memory.select(self.rank)
        self.buffers_peak = max(self.buffers_peak, value.nbytes + received.nbytes)
        self.memory.sample(self.rank, stage + "/device")
        send_buffer = ctypes.c_void_p(value.data_ptr)
        recv_buffer = ctypes.c_void_p(received.data_ptr)
        output_count = received.nbytes // (2 if value.dtype == torch.bfloat16 else 4)
        self._collective(send_buffer, recv_buffer, value.nbytes, output_count, value.dtype, stage, collective)
        self.transfers.append({"stage": stage, "bytes_per_peer": value.nbytes // self.world, "world": self.world,
                               "shape": list(value.shape), "dtype": str(value.dtype), "collective": collective,
                               "transport_type": "BF16/FP32 SUM" if collective in ("reduce_scatter", "all_reduce") else "UINT8",
                               "host_staging": False})
        return received

    def exchange(self, value, stage, collective="all_to_all"):
        self.memory.select(self.rank)
        if value.device.type != "cpu" or not value.is_contiguous():
            raise ValueError("HCCL byte transport requires a contiguous CPU staging container")
        byte_count = value.numel() * value.element_size()
        if not byte_count or (collective in ("all_to_all", "reduce_scatter") and byte_count % self.world):
            raise ValueError("Every rank must provide an equal nonempty block for every peer")
        if collective == "all_gather":
            received = torch.empty((self.world * value.shape[0], *value.shape[1:]), dtype=value.dtype)
        elif collective == "reduce_scatter":
            if value.dtype not in (torch.bfloat16, torch.float32) or value.shape[0] % self.world:
                raise ValueError("Reduce-scatter requires equal BF16/FP32 row buckets")
            received = torch.empty((value.shape[0] // self.world, *value.shape[1:]), dtype=value.dtype)
        elif collective == "all_reduce":
            if value.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError("All-reduce requires BF16/FP32 values")
            received = torch.empty_like(value)
        elif collective == "all_to_all":
            received = torch.empty_like(value)
        elif collective == "all_reduce":
            if value.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError("All-reduce requires BF16/FP32 values")
            received = torch.empty_like(value)
        else:
            raise ValueError("Unsupported basic collective: " + collective)
        received_bytes = received.numel() * received.element_size()
        send_buffer, recv_buffer = ctypes.c_void_p(), ctypes.c_void_p()
        acl = self.memory.library
        _check(acl.aclrtMalloc(ctypes.byref(send_buffer), byte_count, 0), "aclrtMalloc/send")
        try:
            _check(acl.aclrtMalloc(ctypes.byref(recv_buffer), received_bytes, 0), "aclrtMalloc/receive")
            self.buffers_peak = max(self.buffers_peak, byte_count + received_bytes)
            self.memory.sample(self.rank, stage + "/load")
            with self.memory.interval(self.rank, "load", stage + "/staging"):
                _check(acl.aclrtMemcpy(send_buffer, byte_count, value.data_ptr(), byte_count, 1), "aclrtMemcpy/H2D")
            peer_bytes = byte_count // self.world
            self._collective(send_buffer, recv_buffer, byte_count, received.numel(), value.dtype, stage, collective)
            self.memory.sample(self.rank, stage + "/execute")
            arguments = [received.data_ptr(), received_bytes, recv_buffer, received_bytes, 2]
            _check(acl.aclrtMemcpy(*arguments), "aclrtMemcpy/D2H")
            self.transfers.append({"stage": stage, "bytes_per_peer": peer_bytes, "world": self.world,
                                   "shape": list(value.shape), "dtype": str(value.dtype), "collective": collective,
                                   "transport_type": "BF16/FP32 SUM" if collective in ("reduce_scatter", "all_reduce") else "UINT8",
                                   "host_staging": True})
        finally:
            if recv_buffer.value:
                _check(acl.aclrtFree(recv_buffer), "aclrtFree/receive")
            _check(acl.aclrtFree(send_buffer), "aclrtFree/send")
        return received

    def close(self):
        self.memory.select(self.rank)
        try:
            if self.comm.value:
                _check(self.library.HcclCommDestroy(self.comm), "HcclCommDestroy")
                self.comm = ctypes.c_void_p()
        finally:
            if self.stream.value:
                _check(self.memory.library.aclrtDestroyStream(self.stream), "aclrtDestroyStream")
                self.stream = ctypes.c_void_p()

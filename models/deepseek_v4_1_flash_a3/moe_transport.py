# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Transport contiguous PyPTO route buckets through CANN memory and HCCL bytes."""

import ctypes
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

    @staticmethod
    def root_info(memory):
        """Create communicator bootstrap bytes in the coordinating process."""
        library = ctypes.CDLL("libhccl.so", mode=ctypes.RTLD_GLOBAL)
        function = _bind(library, "HcclGetRootInfo", [ctypes.c_void_p])
        root = ctypes.create_string_buffer(4108)
        _check(function(root), "HcclGetRootInfo")
        return root.raw

    def __init__(self, memory, rank, world, group_ranks=None, *, root_info):
        self.memory, self.rank, self.world = memory, rank, world
        self.group_ranks = list(range(world)) if group_ranks is None else group_ranks
        local_rank = self.group_ranks.index(rank)
        memory.select(rank)
        self.library = ctypes.CDLL("libhccl.so", mode=ctypes.RTLD_GLOBAL)
        pointer, u64, integer = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int
        _bind(self.library, "HcclGetRootInfo", [pointer])
        _bind(self.library, "HcclCommInitRootInfo", [ctypes.c_uint32, pointer, ctypes.c_uint32, ctypes.POINTER(pointer)])
        _bind(self.library, "HcclCommDestroy", [pointer])
        _bind(self.library, "HcclAlltoAll", [pointer, u64, integer, pointer, u64, integer, pointer, pointer])
        _bind(self.library, "HcclAllGather", [pointer, pointer, u64, integer, pointer, pointer])
        _bind(self.library, "HcclReduceScatter", [pointer, pointer, u64, integer, integer, pointer, pointer])
        self.stream, self.comm = pointer(), pointer()
        _check(memory.library.aclrtCreateStream(ctypes.byref(self.stream)), "aclrtCreateStream")
        try:
            root = ctypes.create_string_buffer(4108)
            if len(root_info) != ctypes.sizeof(root):
                raise ValueError("Invalid in-memory HCCL root information size")
            ctypes.memmove(root, root_info, len(root_info))
            arguments = [world, root, local_rank, ctypes.byref(self.comm)]
            _check(self.library.HcclCommInitRootInfo(*arguments), "HcclCommInitRootInfo")
        except BaseException:
            self.close()
            raise
        self.buffers_peak = 0
        self.transfers = []

    def exchange(self, value, stage, collective="all_to_all"):
        self.memory.select(self.rank)
        if value.device.type != "cpu" or not value.is_contiguous():
            raise ValueError("HCCL byte transport requires a contiguous CPU staging container")
        byte_count = value.numel() * value.element_size()
        if not byte_count or (collective != "all_gather" and byte_count % self.world):
            raise ValueError("Every rank must provide an equal nonempty block for every peer")
        if collective == "all_gather":
            received = torch.empty((self.world * value.shape[0], *value.shape[1:]), dtype=value.dtype)
        elif collective == "reduce_scatter":
            if value.dtype not in (torch.bfloat16, torch.float32) or value.shape[0] % self.world:
                raise ValueError("Reduce-scatter requires equal BF16/FP32 row buckets")
            received = torch.empty((value.shape[0] // self.world, *value.shape[1:]), dtype=value.dtype)
        elif collective == "all_to_all":
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
            with self.memory.interval(self.rank, "communication", stage):
                if collective == "all_to_all":
                    arguments = [send_buffer, peer_bytes, 7, recv_buffer, peer_bytes, 7, self.comm, self.stream]
                    status = self.library.HcclAlltoAll(*arguments)
                elif collective == "all_gather":
                    status = self.library.HcclAllGather(send_buffer, recv_buffer, byte_count, 7, self.comm, self.stream)
                else:
                    dtype = 11 if value.dtype == torch.bfloat16 else 4
                    count = received.numel()
                    status = self.library.HcclReduceScatter(send_buffer, recv_buffer, count, dtype, 0, self.comm, self.stream)
                _check(status, "HCCL/" + collective)
                _check(acl.aclrtSynchronizeStream(self.stream), "aclrtSynchronizeStream")
            self.memory.sample(self.rank, stage + "/execute")
            arguments = [received.data_ptr(), received_bytes, recv_buffer, received_bytes, 2]
            _check(acl.aclrtMemcpy(*arguments), "aclrtMemcpy/D2H")
            self.transfers.append({"stage": stage, "bytes_per_peer": peer_bytes, "world": self.world,
                                   "shape": list(value.shape), "dtype": str(value.dtype), "collective": collective,
                                   "transport_type": "BF16/FP32 SUM" if collective == "reduce_scatter" else "UINT8"})
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

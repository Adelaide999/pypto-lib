# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Transport contiguous PyPTO route buckets through CANN memory and HCCL bytes."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

import argparse
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import sys
import time

import torch

from .runtime_config import configure_devices
from .runtime_utils import file_sha


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

    def __init__(self, *, diagnostics=None):
        self.diagnostics = os.environ.get("DS41_DIAGNOSTICS") != "0" if diagnostics is None else diagnostics
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

    def __init__(self, memory, rank, world, directory, group_ranks=None, *, root_info=None):
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
            path = None if directory is None else directory / ("hccl-root-" + "-".join(map(str, self.group_ranks)) + ".bin")
            if root_info is not None:
                if len(root_info) != ctypes.sizeof(root):
                    raise ValueError("Invalid in-memory HCCL root information size")
                ctypes.memmove(root, root_info, len(root_info))
            elif path is None:
                raise ValueError("HCCL bootstrap requires explicit root bytes or a rendezvous directory")
            elif local_rank == 0:
                _check(self.library.HcclGetRootInfo(root), "HcclGetRootInfo")
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(root.raw)
                temporary.replace(path)
            else:
                deadline = time.monotonic() + 180
                while not path.exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Timed out waiting for the HCCL root communicator")
                    time.sleep(0.02)
                data = path.read_bytes()
                if len(data) != ctypes.sizeof(root):
                    raise ValueError("Invalid HCCL root information size")
                ctypes.memmove(root, data, len(data))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()
    runtime = json.loads(args.runtime_config.read_text())
    world, rank = runtime["topology"]["ep"], args.rank
    if not 0 <= rank < world:
        raise ValueError("HCCL rank is outside the explicit runtime device mapping")
    assigned = configure_devices(runtime["assigned_devices"], world=world)
    os.environ["HCCL_EXEC_TIMEOUT"] = "180"
    settings = runtime.get("transport", {"collective": "all_to_all", "group": "ep", "stage": args.directory.name})
    collective = settings["collective"]
    group_ranks = next((group for group in runtime.get("groups", {}).get(settings["group"], [])
                        if rank in group), list(range(world)))
    memory = CANNMemory()
    memory.sample(rank, "transport/load/before")
    with memory.interval(rank, "communication", "communicator_init"):
        transport = HCCLBytes(memory, rank, len(group_ranks), args.directory, group_ranks)
    path = args.directory / f"rank-{rank}-inputs.pt"
    values = torch.load(path, map_location="cpu", weights_only=True)
    outputs = {}
    try:
        for case, packet in values.items():
            outputs[case] = {}
            for name, value in packet.items():
                outputs[case][name] = transport.exchange(value, case + "/" + name, collective)
            print({"rank": rank, "case": case, "transport": "HcclAlltoAll/UINT8"}, flush=True)
    finally:
        transport.close()
    output_path = args.directory / f"rank-{rank}-outputs.pt"
    torch.save(outputs, output_path)
    libraries = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        path_string = line.split()[-1]
        if path_string.startswith("/") and ".so" in path_string and Path(path_string).is_file():
            if any(part in path_string.lower() for part in ("libhccl", "libascendcl", "libruntime")):
                libraries[path_string] = file_sha(path_string)
    report = {"rank": rank, "assigned_device": assigned[rank], "backend": "CANN/HCCL byte transport",
              "collective": collective, "group_ranks": group_ranks, "stage": settings["stage"],
              "artifacts": {str(path): file_sha(path), str(output_path): file_sha(output_path)},
              "command": [sys.executable, *sys.argv], "source_sha256": file_sha(__file__),
              "inputs_sha256": file_sha(path), "outputs_sha256": file_sha(output_path),
              "libraries": libraries, "transfers": transport.transfers,
              "memory": {"scope": "CANN whole-device HBM samples at transport stage boundaries; not a peak",
                         "samples": memory.samples, "intervals": memory.intervals,
                         "transport_buffer_peak_bytes": transport.buffers_peak,
                         "buffer_scope": "Only the two explicit transport buffers; excludes runtime and communicator"}}
    receipt_path = args.directory / f"rank-{rank}-report.json"
    receipt_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

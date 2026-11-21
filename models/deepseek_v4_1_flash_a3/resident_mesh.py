# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bind PyPTO distributed communication windows on the session's original chip workers."""

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from pypto.backend import BackendType
from pypto.ir import DistributedConfig
from pypto.runtime import ExecutionMode, RunConfig
from simpler.task_interface import _CommContextStruct

from .routed_exchange import D, EP, TOPK, one_token_expert_graph


class ResidentExpertMesh:
    """Use one communication domain with one rank on each resident worker."""

    def __init__(self, runtime, directory):
        self.runtime = runtime
        self.sequence = 0
        self.window_bytes = 16 << 20
        self.streams = [ctypes.c_void_p() for _ in range(EP)]
        self.handles = [None] * EP
        self.contexts = [None] * EP
        self.rendezvous = TemporaryDirectory(prefix="ds41-ep-")
        output = Path(directory) / "routed-exchange"
        config = RunConfig(arch=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                           distributed_config=DistributedConfig(device_ids=list(range(EP)), num_sub_workers=0),
                           save_kernels=True, save_kernels_dir=str(output))
        one_token_expert_graph.compile(config=config)
        self.output_dir = output / "next_levels" / "routed_rank"
        libraries = {row.split()[-1] for row in Path("/proc/self/maps").read_text().splitlines()
                     if row.endswith("/a2a3/onboard/tensormap_and_ringbuffer/libhost_runtime.so")}
        if len(libraries) != 1:
            raise RuntimeError("Expected one loaded A3 tensormap_and_ringbuffer runtime")
        self.library = ctypes.CDLL(libraries.pop())
        self.acl = ctypes.CDLL("libascendcl.so")
        pointer, integer, u64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
        for dll, name, result, args in (
                (self.library, "comm_init", pointer, (integer, integer, pointer, ctypes.c_char_p)),
                (self.library, "comm_alloc_windows", integer, (pointer, ctypes.c_size_t, ctypes.POINTER(u64))),
                (self.library, "comm_destroy", integer, (pointer,)),
                (self.acl, "aclrtCreateStream", integer, (ctypes.POINTER(pointer),)),
                (self.acl, "aclrtDestroyStream", integer, (pointer,))):
            fn = getattr(dll, name)
            fn.restype, fn.argtypes = result, args
        try:
            runtime.map_ranks(self._initialize_rank)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _check(status, operation):
        if status:
            raise RuntimeError(f"{operation} returned {status}")

    def _initialize_rank(self, rank):
        stream = self.streams[rank]
        self._check(self.acl.aclrtCreateStream(ctypes.byref(stream)), "aclrtCreateStream")
        path = os.fsencode(Path(self.rendezvous.name) / "root")
        handle = self.library.comm_init(rank, EP, stream, path)
        if not handle:
            raise RuntimeError(f"comm_init failed on rank {rank}")
        self.handles[rank] = handle
        context = ctypes.c_uint64()
        window_bytes = self.window_bytes
        self._check(self.library.comm_alloc_windows(handle, window_bytes, ctypes.byref(context)), "comm_alloc_windows")
        info = _CommContextStruct()
        self.runtime._memory.copy(ctypes.addressof(info), context.value, ctypes.sizeof(info), 2)
        if info.rankId != rank or info.rankNum != EP or not all(info.windowsIn[:EP]):
            raise RuntimeError(f"Incomplete EP16 window mapping on rank {rank}")
        self.runtime._memory.fill(info.windowsIn[rank], window_bytes, 0)
        self.contexts[rank] = context.value, int(info.windowsIn[rank])

    def close(self):
        def close_rank(rank):
            self.runtime._select(rank)
            if self.handles[rank]:
                self._check(self.library.comm_destroy(self.handles[rank]), "comm_destroy")
                self.handles[rank] = None
            if self.streams[rank]:
                self._check(self.acl.aclrtDestroyStream(self.streams[rank]), "aclrtDestroyStream")
                self.streams[rank] = ctypes.c_void_p()
        with ThreadPoolExecutor(max_workers=EP, thread_name_prefix="decode-comm-close") as pool:
            tasks = [pool.submit(close_rank, rank) for rank in range(EP)]
            for task in tasks:
                task.result()
        self.rendezvous.cleanup()

    def run(self, states, owner, case, grouped_weights):
        """Execute one complete routed Top6 and combine on the source SP rank."""
        self.sequence += 1
        if self.sequence >= 2 ** 31:
            raise OverflowError("EP16 signal sequence exhausted")
        runtime = self.runtime

        def bind(rank):
            state = states[rank][case]
            if rank == owner:
                token = state["normed"][:1]
                ids = state["route_ids"][:1]
                weights = state["weights"][:1]
                shared = state["shared"][:1]
                combined = state["combined"][:1]
            else:
                token = runtime.workspace((1, D), torch.bfloat16, rank, fill=None)
                ids = runtime.workspace((1, TOPK), torch.int32, rank, fill=None)
                weights = runtime.workspace((1, TOPK), torch.float32, rank, fill=None)
                shared = runtime.workspace((1, D), torch.bfloat16, rank, fill=None)
                combined = runtime.workspace((1, D), torch.bfloat16, rank, fill=None)
            handles = [runtime.tensor(token, rank), runtime.tensor(ids, rank), *grouped_weights(rank),
                       runtime.tensor(weights, rank), runtime.tensor(shared, rank), runtime.tensor(combined, rank)]
            runtime.values[rank, combined.untyped_storage()._cdata]["device_written"] = True
            runtime.recording.experts(rank, handles, owner, self.sequence)

        runtime.map_ranks(bind)

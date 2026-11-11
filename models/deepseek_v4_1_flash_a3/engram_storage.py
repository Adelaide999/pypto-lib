# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Own complete shared INT8/FP32 checkpoint backings and same-process CANN mappings."""

import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import tempfile

from .moe_transport import _bind, _check


# CANN standard host registration flags.
HOST_REGISTER_FLAGS = 0x10000002

# Host byte-copy chunk.
COPY_TILE = 16 * 1024 * 1024


def checkpoint_region(checkpoint, name):
    """Resolve an original complete safetensors payload without tensor arithmetic."""
    root = Path(checkpoint)
    index = json.loads((root / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    path = root / index[name]
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
    item = header[name]
    return path, 8 + header_size + item["data_offsets"][0], item


class SharedEngramArray:
    """Retain one complete TP array in a task-owned MAP_SHARED backing."""

    def __init__(self, checkpoint, name, row_range, width, dtype, expected_sha, owner):
        path, payload, spec = checkpoint_region(checkpoint, name)
        if spec["dtype"] != dtype or spec["shape"][1:] != [width]:
            raise ValueError("Original Engram checkpoint dtype/shape differs: " + name)
        begin, end = row_range
        if not 0 <= begin < end <= spec["shape"][0]:
            raise ValueError("Engram shard exceeds original complete checkpoint")
        element_bytes = {"I8": 1, "F32": 4}[dtype]
        self.shape = (end - begin, width)
        self.byte_count = (end - begin) * width * element_bytes
        self.path = None
        self.mapping = None
        self.view = None
        self.live = set()
        self.host_pointer = 0
        available = os.statvfs("/dev/shm")
        if self.byte_count > available.f_bavail * available.f_frsize:
            raise RuntimeError("Insufficient shared host capacity for complete Engram array")
        fd, filename = tempfile.mkstemp(prefix=f"ds41-{owner}-", dir="/dev/shm")
        self.path = Path(filename)
        try:
            os.ftruncate(fd, self.byte_count)
            self.mapping = mmap.mmap(fd, self.byte_count, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        except BaseException:
            self.path.unlink()
            raise
        finally:
            os.close(fd)
        try:
            self.view = memoryview(self.mapping)
            digest = hashlib.sha256()
            with path.open("rb", buffering=0) as source:
                source.seek(payload + begin * width * element_bytes)
                for offset in range(0, self.byte_count, COPY_TILE):
                    chunk = self.view[offset:min(offset + COPY_TILE, self.byte_count)]
                    try:
                        read = source.readinto(chunk)
                        if read != len(chunk):
                            raise EOFError("Incomplete original Engram checkpoint payload")
                        digest.update(chunk)
                    finally:
                        chunk.release()
            actual_sha = digest.hexdigest()
            if actual_sha != expected_sha:
                raise ValueError("Complete assigned Engram bytes differ from frozen checkpoint: " + name)
            self.host_pointer = ctypes.addressof(ctypes.c_ubyte.from_buffer(self.mapping))
            stat = self.path.stat()
            self.receipt = {"checkpoint_file": str(path), "tensor": name, "row_range": list(row_range),
                            "shape": list(self.shape), "dtype": dtype, "bytes": self.byte_count,
                            "payload_sha256": actual_sha, "shared_path": str(self.path),
                            "inode": stat.st_ino, "device": stat.st_dev, "owner_pid": os.getpid()}
        except BaseException:
            self.close()
            raise

    def register(self, memory, rank):
        """Register this full host range in the calling PID and current rank context."""
        if rank in self.live or self.mapping is None:
            raise RuntimeError("Engram mapping is already registered or closed")
        memory.select(rank)
        lib = memory.library
        pointer, u64, u32 = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32
        register = _bind(lib, "aclrtHostRegisterV2", [pointer, u64, u32])
        mapped = _bind(lib, "aclrtHostGetDevicePointer", [pointer, ctypes.POINTER(pointer), u32])
        unregister = _bind(lib, "aclrtHostUnregister", [pointer])
        host = pointer(self.host_pointer)
        _check(register(host, self.byte_count, HOST_REGISTER_FLAGS), "aclrtHostRegisterV2/Engram")
        self.live.add(rank)
        device = pointer()
        try:
            _check(mapped(host, ctypes.byref(device), 0), "aclrtHostGetDevicePointer/Engram")
            if not device.value:
                raise RuntimeError("CANN returned an empty Engram mapped address")
        except BaseException:
            _check(unregister(host), "aclrtHostUnregister/partial-Engram")
            self.live.remove(rank)
            raise
        return device.value

    def unregister(self, memory, rank):
        """Unregister only after the owning synchronous ChipWorker has retired."""
        if rank not in self.live:
            return
        memory.select(rank)
        function = _bind(memory.library, "aclrtHostUnregister", [ctypes.c_void_p])
        _check(function(ctypes.c_void_p(self.host_pointer)), "aclrtHostUnregister/Engram")
        self.live.remove(rank)

    def close(self):
        """Release shared storage after every DP peer registration has closed."""
        if self.live:
            raise RuntimeError("Cannot close complete Engram storage with live rank mappings")
        if self.view is not None:
            self.view.release()
            self.view = None
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        if self.path is not None:
            self.path.unlink(missing_ok=True)
            self.path = None
        self.host_pointer = 0

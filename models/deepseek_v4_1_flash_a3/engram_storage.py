# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Own complete shared INT8/FP32 checkpoint backings for CPU row lookup."""

import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import tempfile

import torch

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
        self.tensor = None
        self.row_start = begin
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
            self.tensor = torch.frombuffer(self.mapping, dtype={"I8": torch.int8, "F32": torch.float32}[dtype]).reshape(self.shape)
            stat = self.path.stat()
            self.receipt = {"checkpoint_file": str(path), "tensor": name, "row_range": list(row_range),
                            "shape": list(self.shape), "dtype": dtype, "bytes": self.byte_count,
                            "payload_sha256": actual_sha, "shared_path": str(self.path),
                            "inode": stat.st_ino, "device": stat.st_dev, "owner_pid": os.getpid()}
        except BaseException:
            self.close()
            raise

    def gather(self, rows):
        """Copy selected global rows; returned tensors do not alias the host mapping."""
        if self.tensor is None or rows.device.type != "cpu" or rows.dtype not in (torch.int32, torch.int64):
            raise ValueError("Lookup requires an open host table and CPU integer row IDs")
        local = rows.to(torch.int64) - self.row_start
        valid = (local >= 0) & (local < self.shape[0])
        result = torch.zeros((*rows.shape, self.shape[1]), dtype=self.tensor.dtype)
        result[valid] = self.tensor[local[valid]]
        return result

    def close(self):
        """Release CPU views and shared storage after the host lookup owner closes."""
        self.tensor = None
        if self.view is not None:
            self.view.release()
            self.view = None
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        if self.path is not None:
            self.path.unlink(missing_ok=True)
            self.path = None

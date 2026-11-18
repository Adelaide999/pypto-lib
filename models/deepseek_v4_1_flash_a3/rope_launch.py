# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Launch supported resident Prefill RoPE leaves on a CANN stream."""

import ctypes
import hashlib
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


ROPE_DIRECT_ENTRIES = {
    "compressed_rope": (512, "a2_adjacent_rope", ((2, 16), (1, 32)), False),
    "index_rope": (128, "a2_adjacent_rope", ((1, 64),), False),
    "index_query_rope": (4096, "index_query_rope_heads", ((3, 1),), False),
    "q_rope": (32768, "a2_adjacent_rope", ((3, 1),), True),
    "kv_rope": (512, "a2_adjacent_rope", ((1, 32),), True),
    "inverse_rope": (32768, "attention_rope", ((3, 1),), True),
}
_TENSOR_ORDER = {
    1: ("output(output_heads)", "input(x_heads)", "input(ext_cosine)", "input(ext_sine)"),
    2: ("input(source)", "input(ext_cosine)", "input(ext_sine)", "output(target)"),
    3: ("input(source)", "input(ext_cosine)", "input(ext_sine)", "output(target)"),
}
_SCALAR_ORDER = {
    1: ("blocks_per_worker", "blocks", "active"),
    2: ("blocks", "workers", "active"),
    3: ("blocks", "workers"),
}


def _launch_abi(orchestration, symbol, modes, inplace):
    """Match a single generated vector task and its zero-active branch."""
    code = re.sub(r"\s+|_inline\d+", "", orchestration)
    submits = re.findall(r"\brt_submit_(\w+)_task\(", orchestration)
    if submits not in (["aiv"], ["aiv", "dummy"]):
        return None
    if f"_spmd:{symbol}" not in code or "rt_submit_aiv_task(0,params_t0)" not in code:
        return None
    if "params_t0.launch_spec.set_block_num(workers)" not in code:
        return None
    if "int64_tworkers=std::min<int64_t>(blocks,48);" not in code:
        return None
    tensors = re.findall(r"params_t0\.add_((?:input|output|inout)\(\w+\))", code)
    scalars = re.findall(r"params_t0\.add_scalar\((\w+)\)", code)
    for mode, tile in modes:
        blocks = "static_cast<int64_t>(active)" if tile == 1 else f"((static_cast<int64_t>(active)+{tile-1})/{tile})"
        if f"int64_tblocks={blocks};" not in code:
            continue
        if tuple(tensors) != _TENSOR_ORDER[mode] or tuple(scalars) != _SCALAR_ORDER[mode]:
            continue
        if mode == 1 and "int64_tblocks_per_worker=((blocks+47)/48);" not in code:
            continue
        if inplace:
            aliases = ("x_heads", "output_heads") if mode == 1 else ("source", "target")
            if any(f"Tensor{name}=ext_x.reshape(" not in code for name in aliases):
                continue
        return mode, tile
    return None


def _matches_signature(width, abi, inplace):
    expected = (
        ("x", "In", [8192, width], "bfloat16", "ND"),
        ("cosine", "In", [8192, 32], "fp32", "ND"),
        ("sine", "In", [8192, 32], "fp32", "ND"),
        ("output", "InOut", [8192, width], "bfloat16", "ND"),
        ("active", "In", None, "int32", None),
    )
    fields = ("name", "direction", "shape", "dtype", "layout")
    if inplace:
        expected = (("x", "InOut", [8192, width], "bfloat16", "ND"), *expected[1:3], expected[4])
    if abi["return_aliases"] != ([0] if inplace else [3]):
        return False
    return all(tuple(tuple(p[key] for key in fields) for p in abi[item]) == expected
               for item in ("params", "lowered_params"))


def _build_launch(root, symbol, body, mode, tile):
    """Compile the generated body plus the stream and block adapter."""
    from pypto.runtime import ensure_pto_isa_root

    cann = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
    pto = ensure_pto_isa_root()
    shim = Path(__file__).with_suffix(".cpp")
    digest = hashlib.sha256(shim.read_bytes() + body.read_bytes() + str(cann).encode())
    output = root / f"rope-launch-{mode}-{tile}-{digest.hexdigest()[:20]}.so"
    if not output.is_file():
        with TemporaryDirectory(prefix="rope-launch-", dir=root) as directory:
            temporary = Path(directory) / output.name
            command = [
                str(cann / "bin/bisheng"), "-xcce", "-O2", "-std=c++17", "-shared", "-fPIC", "--cce-fatobj-link",
                "--cce-aicore-arch=dav-c220-vec", "-fenable-matrix", "--cce-auto-infer-kernel-type=true",
                "-DREGISTER_BASE", f'-DROPE_BODY="{body.name}"', f"-DROPE_FUNCTION={symbol}",
                f"-DROPE_TILE={tile}", f"-DROPE_ABI={mode}",
                "-Wno-ignored-attributes", "-Wno-macro-redefined",
                "-mllvm", "-cce-aicore-stack-size=0x8000", "-mllvm", "-cce-aicore-function-stack-size=0x8000",
                "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
                f"-I{body.parent}", f"-I{pto}/include", f"-I{cann}/include", f"-L{cann}/lib64",
                str(shim), "-lascendcl", "-o", str(temporary),
            ]
            subprocess.run(command, check=True)
            os.replace(temporary, output)
    return ctypes.CDLL(str(output))


def _check(status, operation):
    if status != 0:
        raise RuntimeError(f"CANN RoPE {operation} returned {status}")


class DirectRope:
    def __init__(self, library, width, tile, inplace):
        self.library = library
        self.launch = library.launch_rope_on_stream
        pointer, integer = ctypes.c_void_p, ctypes.c_int32
        self.launch.argtypes = (pointer, pointer, pointer, pointer, integer, integer, pointer)
        self.launch.restype = ctypes.c_int
        self.width = width
        self.tile = tile
        self.inplace = inplace

    @classmethod
    def from_compiled(cls, name, compiled, abi):
        width, symbol, modes, inplace = ROPE_DIRECT_ENTRIES[name]
        if compiled.platform != "a2a3" or not _matches_signature(width, abi, inplace):
            return None
        root = Path(compiled.output_dir).resolve()
        orchestrations = list((root / "orchestration").glob("*.cpp"))
        body = root / "ptoas" / (symbol + ".cpp")
        if len(orchestrations) != 1 or not body.is_file():
            return None
        matched = _launch_abi(orchestrations[0].read_text(), symbol, modes, inplace)
        if matched is None:
            return None
        mode, tile = matched
        return cls(_build_launch(root, symbol, body, mode, tile), width, tile, inplace)

    def run(self, arguments, stream, synchronize):
        if self.inplace:
            x, cosine, sine, active = arguments
            output = x
        else:
            x, cosine, sine, output, active = arguments
        matrix, tables = (8192, self.width), (8192, 32)
        for value, shape, dtype in (
            (x, matrix, torch.bfloat16), (output, matrix, torch.bfloat16),
            (cosine, tables, torch.float32), (sine, tables, torch.float32),
        ):
            if value.shape != shape or value.dtype != dtype or value.data_ptr % 32:
                raise ValueError("RoPE direct operands differ from the compiled ND layout")
        start, end = output.data_ptr, output.data_ptr + output.nbytes
        for source in ((cosine, sine) if self.inplace else (x, cosine, sine)):
            if max(start, source.data_ptr) < min(end, source.data_ptr + source.nbytes):
                raise ValueError("RoPE direct output overlaps an input")
        if type(active) is not int or not 0 <= active <= matrix[0]:
            raise ValueError("RoPE direct active rows are outside the compiled capacity")
        if active:
            workers = min(48, (active + self.tile - 1) // self.tile)
            status = self.launch(x.data_ptr, cosine.data_ptr, sine.data_ptr, start, active, workers, stream)
            _check(status, "launch")
            _check(synchronize(stream, 50000), "synchronize")


class RopeStreams:
    """Rank-local stream ownership and synchronization."""

    def __init__(self, library):
        self.library = library
        pointer = ctypes.c_void_p
        self.synchronize = library.aclrtSynchronizeStreamWithTimeout
        self.synchronize.argtypes = (pointer, ctypes.c_int32)
        self.synchronize.restype = ctypes.c_int
        self.streams = {}

    def stream(self, rank):
        if rank not in self.streams:
            stream = ctypes.c_void_p()
            _check(self.library.aclrtCreateStream(ctypes.byref(stream)), "stream creation")
            self.streams[rank] = stream
        return self.streams[rank]

    def close(self, select):
        for rank, stream in tuple(self.streams.items()):
            select(rank)
            _check(self.synchronize(stream, 50000), "stream drain")
            _check(self.library.aclrtDestroyStream(stream), "stream destruction")
            del self.streams[rank]

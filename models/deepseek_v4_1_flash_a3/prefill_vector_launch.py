# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Synchronously launch supported generated Prefill vector leaves on A3."""

import ctypes
import hashlib
import json
import os
import re
import subprocess
from math import prod
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from pypto.runtime import ChipWorker, DeviceTensor, ensure_pto_isa_root


_CAPACITY = 8192
_D = 5120
_I32 = ctypes.c_int32
_PTR = ctypes.c_void_p
_DTYPES = {
    "bfloat16": (torch.bfloat16, 2), "fp32": (torch.float32, 4),
    "int8": (torch.int8, 1), "int64": (torch.int64, 8),
}


def _abi(leaf, width):
    matrix = [_CAPACITY, width]
    active = ("active", "In", None, "int32", None)
    if leaf == "rms_norm_cast" and width == _D:
        return "prefill_norm_dual_output", (
            ("collapsed", "In", matrix, "bfloat16", "ND"),
            ("weight", "In", [_D], "bfloat16", "ND"),
            ("normed", "InOut", matrix, "bfloat16", "ND"),
            ("router_input", "InOut", matrix, "fp32", "ND"),
            active,
        ), (2, 3), 8
    if leaf == "projection_quantize" and width in (1280, _D):
        from .quant import VEC_M_TILE

        return "prefill_token_quantize_whole_row", (
            ("x", "In", matrix, "bfloat16", "ND"),
            ("payload", "InOut", matrix, "int8", "ND"),
            ("scales", "InOut", [_CAPACITY, 8], "fp32", "ND"),
            active,
        ), (1, 2), VEC_M_TILE
    if leaf == "rope_tables" and width == 32:
        from .rope import ROPE_TABLE_ROWS_TILE

        return "prefill_rope_tables", (
            ("positions", "In", [_CAPACITY], "int64", "ND"),
            ("cosine", "InOut", matrix, "fp32", "ND"),
            ("sine", "InOut", matrix, "fp32", "ND"),
            active,
        ), (1, 2), ROPE_TABLE_ROWS_TILE
    raise ValueError(f"Unsupported Prefill vector leaf: {leaf}, width={width}")


def _matches_orchestration(source, symbol, params, aliases, tile):
    hint = f"// Spmd {symbol}_spmd: {symbol}"
    if hint not in source:
        return False
    code = re.sub(r"//[^\n]*|\s+|_inline\d+", "", source)
    decl = "".join(f"constTensor&ext_{param[0]}=orch_args.tensor({i}).ref();" for i, param in enumerate(params[:-1]))
    first = f"ext_{params[0][0]}"
    ret, aux = (f"ext_{params[i][0]}" for i in aliases)
    is_table = symbol == "prefill_rope_tables"
    bound = "blocks" if is_table else "tiles"
    grid = f"int64_t{bound}=((static_cast<int64_t>(active)+{tile - 1})/{tile});" if tile > 1 else ""
    if tile == 1:
        bound = "static_cast<int64_t>(active)"
    grid += f"int64_tworkers=std::min<int64_t>({bound},48);"
    reshape = f"uint32_tposition_rows_shapes[2]={{1,{_CAPACITY}}};"
    reshape += "Tensorposition_rows=ext_positions.reshape(position_rows_shapes,2);"
    inputs = "".join(f"params_t0.add_input(ext_{param[0]});" for param in params[:-1] if param[1] == "In")
    if is_table and reshape in code:
        grid += reshape
        inputs = "params_t0.add_input(position_rows);"
    outputs = "".join(f"params_t0.add_inout(ext_{params[i][0]});" for i in aliases)
    args = inputs + outputs if is_table else outputs + inputs
    scalars = (bound, "workers", "active") if tile > 1 else ("active", "workers")
    args += "".join(f"params_t0.add_scalar({scalar});" for scalar in scalars)
    prefix = f"(constChipTaskArgs&orch_args){{{decl}int32_tactive=orch_args.scalar<int32_t>(0);"
    grid = f"SIMPLER_SCOPE(){{{grid}if((0<workers)){{SIMPLER_SCOPE(){{"
    task = f"CoreTaskArgsparams_t0;{args}params_t0.launch_spec.set_block_num(workers);rt_submit_aiv_task(0,params_t0);"
    suffix = "}}else{SIMPLER_SCOPE(){"
    actual = code.partition("voidaicpu_orchestration_entry")[2]
    close = "}}}}}"
    if actual == prefix + grid + task + suffix + close:
        return True
    initial = f"Tensorret={first};Tensorret__tmp_v0_guarded_1={first};"
    returned = f"ret={ret};ret__tmp_v0_guarded_1={aux};"
    return actual == prefix + initial + grid + task + returned + suffix + returned + close


def _generated_body(root, symbol, params, aliases, tile):
    meta = json.loads((root / "compiled_meta.json").read_text())
    fields = ("name", "direction", "shape", "dtype", "layout")
    if (
        meta["platform"] != "a2a3" or meta["backend_type"] != "Ascend910" or meta["schema"] != 3
        or meta["num_return_types"] != len(aliases) or tuple(meta["return_aliases"]) != aliases
        or any(tuple(tuple(p[key] for key in fields) for p in meta[item]) != params
               for item in ("params", "lowered_params"))
    ):
        raise ValueError("Unsupported Prefill vector signature")
    sources = list((root / "orchestration").glob("*.cpp"))
    if len(sources) != 1 or not _matches_orchestration(sources[0].read_text(), symbol, params, aliases, tile):
        raise ValueError("Expected exactly one supported Prefill vector task")
    body = root / "ptoas" / f"{symbol}.cpp"
    config = (root / "kernel_config.py").read_text()
    kernels = re.findall(r'"func_id"\s*:\s*(\d+),\s*"name"\s*:\s*"(\w+)"[^\n]*"core_type"\s*:\s*"(\w+)"', config)
    if kernels != [("0", symbol, "aiv")] or not body.is_file():
        raise ValueError("Expected the generated A3 vector kernel body")
    return body


def _compile(root, body, leaf, tile, cann):
    pto = Path(ensure_pto_isa_root()).resolve()
    shim = Path(__file__).with_suffix(".cpp")
    compiler = cann / "bin/bisheng"
    digest = hashlib.sha256(shim.read_bytes() + body.read_bytes())
    digest.update(f"{leaf}:{tile}".encode())
    for path in (cann, pto):
        digest.update(str(path.resolve()).encode())
    digest.update(str(compiler.stat().st_mtime_ns).encode())
    output = root / f"prefill-vector-{digest.hexdigest()[:20]}.so"
    if not output.is_file():
        with TemporaryDirectory(prefix="prefill-vector-", dir=root) as directory:
            temporary = Path(directory) / output.name
            command = [
                str(compiler), "-xcce", "-O2", "-std=c++17", "-shared", "-fPIC", "--cce-fatobj-link",
                "--cce-aicore-arch=dav-c220-vec", "-fenable-matrix", "--cce-auto-infer-kernel-type=true",
                "-DREGISTER_BASE", f'-DPYPTO_VECTOR_BODY="{body.name}"',
                f"-DPYPTO_VECTOR_QUANT_ROWS={tile if leaf == 'projection_quantize' else 0}",
                f"-DPYPTO_VECTOR_ROPE_TABLE_ROWS={tile if leaf == 'rope_tables' else 0}",
                "-Wno-ignored-attributes", "-Wno-macro-redefined",
                "-mllvm", "-cce-aicore-stack-size=0x8000", "-mllvm", "-cce-aicore-function-stack-size=0x8000",
                "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
                f"-I{body.parent}", f"-I{pto}/include", f"-I{cann}/include", f"-L{cann}/lib64",
                str(shim), "-lascendcl", "-o", str(temporary),
            ]
            subprocess.run(command, check=True)
            os.replace(temporary, output)
    return output


def _check(status, operation):
    if status:
        raise RuntimeError(f"CANN Prefill vector {operation} returned {status}")


def _bind(library, name, *args):
    call = getattr(library, name)
    call.argtypes = args
    call.restype = ctypes.c_int
    return call


class PrefillVectorLaunch:
    """Single-threaded, synchronous A3 launch borrowing one live ChipWorker."""

    @classmethod
    def build(cls, leaf, width, compiled):
        """Build once from an onboard CompiledProgram with a supported ABI."""
        if compiled.platform != "a2a3":
            raise ValueError("Prefill vector launch requires onboard A3")
        root = Path(compiled.output_dir).resolve()
        symbol, params, aliases, tile = _abi(leaf, width)
        body = _generated_body(root, symbol, params, aliases, tile)
        cann = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
        library = ctypes.CDLL(str(_compile(root, body, leaf, tile, cann)))
        return cls(library, params[:-1], aliases, tile)

    def __init__(self, library, params, aliases, tile):
        self._library = library
        self._launch = _bind(library, "launch_prefill_vector_leaf", _PTR, _PTR, _PTR, _PTR, _I32, _I32, _PTR)
        acl = self._acl = ctypes.CDLL("libascendcl.so")
        self._select = _bind(acl, "aclrtSetDevice", _I32)
        self._create = _bind(acl, "aclrtCreateStream", ctypes.POINTER(_PTR))
        self._sync = _bind(acl, "aclrtSynchronizeStreamWithTimeout", _PTR, _I32)
        self._destroy = _bind(acl, "aclrtDestroyStream", _PTR)
        self._operands = []
        for name, _, shape, dtype, _ in params:
            torch_dtype, itemsize = _DTYPES[dtype]
            self._operands.append((name, tuple(shape), torch_dtype, prod(shape) * itemsize))
        self._aliases = aliases
        self._tile = tile

    def open(self, worker):
        """Create a stream scoped inside the caller's ChipWorker lifetime."""
        if not isinstance(worker, ChipWorker) or worker.platform != "a2a3":
            raise ValueError("Prefill vector launch requires an A3 ChipWorker")
        return _VectorStream(self, worker)


class _VectorStream:
    def __init__(self, entry, worker):
        self._entry = entry
        self._worker = worker
        self._stream = _PTR()
        self._failed = False

    def __enter__(self):
        if self._stream:
            raise RuntimeError("Prefill vector stream is already open")
        self._worker._require_ready("open Prefill vector stream")
        _check(self._entry._select(self._worker.device_id), "select device")
        _check(self._entry._create(ctypes.byref(self._stream)), "create stream")
        self._failed = False
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._stream:
            _check(self._entry._select(self._worker.device_id), "select device")
            _check(self._entry._sync(self._stream, 50000), "drain stream")
            _check(self._entry._destroy(self._stream), "destroy stream")
            self._stream = _PTR()

    def run(self, *args, active):
        """Validate resident operands, run the active prefix and return both output aliases."""
        if not self._stream or self._failed:
            raise RuntimeError("Prefill vector stream is closed or failed")
        entry = self._entry
        if type(active) is not int or not 0 <= active <= _CAPACITY:
            raise ValueError("Prefill vector active rows are outside the compiled capacity")
        if len(args) != len(entry._operands):
            raise ValueError("Incorrect number of Prefill vector operands")
        regions = []
        for value, (name, shape, dtype, size) in zip(args, entry._operands):
            if not isinstance(value, DeviceTensor) or value.shape != shape or value.dtype != dtype:
                raise ValueError(f"Incorrect Prefill vector shape or dtype for {name}")
            self._worker._require_owned_resident_tensor(value, name, worker_id=0)
            start = value.data_ptr
            if start % 32:
                raise ValueError(f"Unaligned Prefill vector operand: {name}")
            regions.append((start, start + size))
        for i in entry._aliases:
            start, end = regions[i]
            if any(i != j and start < right and left < end for j, (left, right) in enumerate(regions)):
                raise ValueError("Prefill vector outputs cannot alias another operand")
        if active:
            pointers = [start for start, _ in regions]
            if len(pointers) == 3:
                pointers.append(None)
            workers = min(48, (active + entry._tile - 1) // entry._tile)
            _check(entry._select(self._worker.device_id), "select device")
            self._failed = True
            _check(entry._launch(*pointers, active, workers, self._stream), "launch")
            _check(entry._sync(self._stream, 50000), "synchronize")
            self._failed = False
        return tuple(args[i] for i in entry._aliases)

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compose the session's compiled Decode stages into one distributed rank program per token."""

import hashlib
import math
from pathlib import Path
from time import perf_counter

import torch
from pypto.backend import BackendType
from pypto.ir import DistributedConfig
from pypto.runtime import DeviceTensor, ExecutionMode, RunConfig
from pypto.runtime.device_runner import _compile_and_assemble
from pypto.runtime.runner import _build_call_config
from pypto.runtime.tensor_arg import make_tensor_arg
from simpler.task_interface import ArgDirection, ChipCallable, DataType, TaskArgs, TensorArgType
from simpler_setup.kernel_compiler import KernelCompiler

from .cache import resolve_layer_role
from .decode_replay_inputs import DecodeReplayInputs
from .indexer import INDEXER_SELECT_KEY_CAPACITY
from .resident_memory import copy_bytes, make_fill_bytes
from .tp_exchange import make_tp_exchange


DTYPE = {torch.bfloat16: DataType.BFLOAT16.value, torch.float32: DataType.FLOAT32.value,
         torch.float16: DataType.FLOAT16.value, torch.int8: DataType.INT8.value,
         torch.uint8: DataType.UINT8.value, torch.int32: DataType.INT32.value,
         torch.int64: DataType.INT64.value}
TP_BASE = 1 << 17
TP_DATA = TP_BASE + 512


class ResidentDecodeGraph:
    """Own the linked binary, the distributed window and per-step operand bindings."""

    def __init__(self, runtime, program):
        self.runtime = runtime
        self.mesh = runtime.expert_mesh(program.directory)
        self.output_dir = program.directory / "full-decode"
        self.output_dir.mkdir()
        self.stages = {}
        self.tp = {}
        self.tp_sequence = 0
        self.chip = None
        self.registrations = []
        self.signature = None
        self.prepared = None
        self.pools = ({}, {}, {})
        self.temporary = [[] for _ in runtime.devices]
        runtime._stack.callback(self.close)
        self.runtime_config = {}
        for entries in program.stages.values():
            for entry in entries.values():
                self._stage(entry.compiled.output_dir)
        self.expert_stage = self._stage(self.mesh.output_dir)
        self.memory_stages = {}
        for key, kernel in (("copy", copy_bytes), (0, make_fill_bytes(0)), (255, make_fill_bytes(255))):
            path = self.output_dir / f"memory-{key}"
            config = RunConfig(arch=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                               save_kernels=True, save_kernels_dir=str(path))
            kernel.compile(config=config)
            self.memory_stages[key] = self._stage(path)

    def _stage(self, path):
        path = Path(path).resolve()
        if path not in self.stages:
            self.stages[path] = len(self.stages)
            self.chip = None
        return self.stages[path]

    def begin(self, provider, settings, lookup, metadata):
        begin = perf_counter()
        # The ordinary request preallocates its page tables. Owner or layout changes
        # invalidate instructions containing physical page lists or rank-specific branches.
        signature = (tuple(e["active"] for e in settings["DP_engines"]),
                     tuple(tuple((tuple(group["names"]), tuple(group["live_block_ids"]), group["reserved_untouched_block_id"],
                                  group["block_table"].numpy().tobytes()) for group in row["groups"])
                           for row in metadata))
        rt = self.runtime
        self.outer_pools = rt.values, rt.free_values, rt.free_workspaces
        rt.values, rt.free_values, rt.free_workspaces = self.pools
        if self.signature != signature:
            rt.release_values()
            self.close()
            self.signature = signature
            self.commands = [[] for _ in rt.devices]
            self.initial = [{} for _ in rt.devices]
            self.inputs = [
                DecodeReplayInputs(rank) for rank in range(len(rt.devices))]
            self.roles = [None for _ in rt.devices]
            self.cache_blocks = provider.runtime_geometry()["cache_blocks"]
            self.routes = {}
            self.pending = None
            self.tp_start, self.ep_start = self.tp_sequence, self.mesh.sequence
        self.provider, self.settings, self.lookup = provider, settings, lookup
        self.replaying = self.prepared is not None
        self.executed = False
        self.admission_seconds = perf_counter() - begin
        self.prepare_seconds = 0.0
        self.graph_seconds = None

    def kept_values(self):
        return ([(rank, value) for rank, row in enumerate(self.pending) for value, _ in row.values()]
                + [(rank, bank) for rank, (bank, _) in self.routes.items()])

    def close(self):
        for rank, tensors in enumerate(self.temporary):
            self.runtime._select(rank)
            if rank in self.runtime.failed_ranks:
                self.runtime.allocations[rank].extend(tensors)
            else:
                for tensor in reversed(tensors):
                    self.runtime.workers[rank].free_tensor(tensor)
            tensors.clear()
        self.prepared = None

    def _build_call_config(self, config, *, dfx_dir=None):
        return _build_call_config(config, runtime_config=self.runtime_config, dfx_dir=dfx_dir)

    def _build(self):
        import runpy

        if self.chip is not None:
            return
        body = []
        children = []
        names = []
        for path, index in self.stages.items():
            child_chip, _, _ = _compile_and_assemble(path, "a2a3")
            config = runpy.run_path(str(path / "kernel_config.py"))
            source = Path(config["ORCHESTRATION"]["source"])
            offset = len(names)
            for i in range(child_chip.child_count):
                children.append((offset + child_chip.child_func_id(i), child_chip.child(i)))
            ids = {entry["func_id"]: entry["name"] for entry in config["KERNELS"]}
            names.extend(ids.get(i, "") for i in range(max(ids) + 1))
            body.extend([f"#define aicpu_orchestration_entry decode_stage_{index}",
                         f"#define aicpu_orchestration_config decode_config_{index}",
                         f"#define rt_submit_aic_task(id, a) finish_task((rt_submit_aic_task)({offset} + id, a))",
                         f"#define rt_submit_aiv_task(id, a) finish_task((rt_submit_aiv_task)({offset} + id, a))",
                         f"#define rt_submit_task(ids, a) submit_mixed({offset}, ids, a)",
                         f'#include "{source}"', "#undef aicpu_orchestration_config", "#undef aicpu_orchestration_entry",
                         "#undef rt_submit_task", "#undef rt_submit_aic_task", "#undef rt_submit_aiv_task"])
        body.append("static void (*const stages[])(const ChipTaskArgs &) = {")
        body.extend(f"    decode_stage_{i}," for i in range(len(self.stages)))
        body.append("};")
        (self.output_dir / "resident_decode_stages.inc").write_text("\n".join(body) + "\n")
        wrapper = self.output_dir / "resident_decode_entry.cpp"
        wrapper.write_text(Path(__file__).with_name(wrapper.name).read_text())
        binary = KernelCompiler(platform="a2a3").compile_orchestration("tensormap_and_ringbuffer", str(wrapper))
        self.chip = ChipCallable.build(signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.IN],
                                       func_name="aicpu_orchestration_entry", binary=binary, children=children)
        # NetworkProfiler uses kernel_config to resolve child IDs in the raw trace.
        config = [{"func_id": i, "name": name} for i, name in enumerate(names)]
        (self.output_dir / "kernel_config.py").write_text("KERNELS = " + repr(config) + "\n")
        self.registrations.append(self.chip)

    def _describe(self, value, rank, *, write=False):
        runtime = self.runtime
        runtime.workers[rank]._require_owned_resident_tensor(value, "Decode graph operand")
        key = id(value.buffer)
        if runtime.device_owners.get(key) != rank or not runtime.device_active.get(key):
            raise ValueError("A Decode operand belongs to a different rank or to a retired scratch lease")
        if write and runtime.device_readonly[key]:
            raise ValueError("A Decode graph operation would modify an immutable weight")
        if value.dtype not in DTYPE or len(value.shape) > 5:
            raise ValueError("Unsupported Decode operand dtype or rank")
        return (value.data_ptr, DTYPE[value.dtype], len(value.shape), *value.shape)

    def emit(self, rank, op, tensors, scalars=()):
        tape = self.commands[rank]
        tape.extend((op, len(tensors), len(scalars)))
        for descriptor in tensors:
            tape.extend(descriptor)
        tape.extend(scalars)

    def stage(self, rank, compiled, bound, directions):
        worker = self.runtime.workers[rank]
        _, coerced, _ = compiled._build_orch_args(*bound, worker=worker._impl)
        tensors, scalars = [], []
        layer, name, arguments = self.roles[rank]
        bindings = self.inputs[rank]
        for value, direction, param, original in zip(coerced, directions, compiled.param_names, arguments, strict=True):
            if isinstance(value, DeviceTensor):
                descriptor = self._describe(value, rank, write=direction != "In")
                address = bindings.tensor(name, param, layer, value, original)
                tensors.append(descriptor if address is None else (address, *descriptor[1:]))
            elif type(value.value) in (int, bool):
                if name == "index_key_gather" and param == "length":
                    scalars.append(bindings.scalar("cache_length", layer))
                else:
                    scalars.append(int(value.value))
            else:
                raise TypeError(f"Unsupported Decode graph scalar {value!r}")
        self.emit(rank, self.stages[Path(compiled.output_dir).resolve()], tensors, scalars)

    def initialize(self, rank, allocation, host):
        data = memoryview(host.numpy())
        key = hashlib.sha256(data).digest()
        initial = self.initial[rank]
        if key not in initial:
            source = self.runtime.workers[rank].alloc_tensor(host.shape, host.dtype, init=host)
            self.temporary[rank].append(source)
            initial[key] = source
        source = initial[key]
        self._describe(allocation, rank, write=True)
        self._copy_addresses(rank, allocation.data_ptr, source.data_ptr, source.nbytes)

    @staticmethod
    def _bytes(address, count):
        return (address, DTYPE[torch.uint8], 2, 1, count)

    def _copy_addresses(self, rank, target, source, count):
        if source == target or count == 0:
            return
        dst, src = self._bytes(target, count), self._bytes(source, count)
        if abs(target - source) < count:
            self.emit(rank, -1, (dst, src))
        else:
            self.emit(rank, self.memory_stages["copy"], (src, dst), (count,))

    def copy(self, rank, target, source):
        self._describe(target, rank, write=True)
        self._describe(source, rank)
        self._copy_addresses(rank, target.data_ptr, source.data_ptr, source.nbytes)

    def fill(self, rank, allocation, byte):
        self._describe(allocation, rank, write=True)
        self.emit(rank, self.memory_stages[byte], (self._bytes(allocation.data_ptr, allocation.nbytes),),
                  (allocation.nbytes,))

    def check(self, op, rank, tensors, scalars=()):
        if not tensors[0].numel():
            return
        descriptors = [self._describe(self.runtime.tensor(value, rank), rank, write=op == -5 and i == 0)
                       for i, value in enumerate(tensors)]
        self.emit(rank, op, descriptors, scalars)

    def sparse(self, controls, original, geometry, active, rank):
        if not active:
            return
        if (active != 1 or original["metadata"]["query_start_loc"].tolist() != [0, 1]
                or original["cp_token_range"][0] != 0):
            raise ValueError("Single-token sparse controls require one request and one query")
        role = resolve_layer_role(original["physical_layer_id"], original.get("role"))
        keys = (f"model.layers.{original['physical_layer_id']}.self_attn.swa_cache",
                f"model.layers.{role['kv_source']}.self_attn.long_kv_cache")
        meta = original["cache_metadata"]["actual_metadata"]
        pages = [meta[key]["global_metadata"]["block_table"][0].to(torch.int32).contiguous() for key in keys]
        self.check(-5, rank, (controls[:1], *pages), (geometry["own"]["rows"], geometry["long"]["rows"]))

    def candidates(self, original, shared, source_id, rank):
        if not original["actual_local_query_rows"]:
            return
        key = f"model.layers.{source_id}.self_attn.indexer.k_cache"
        meta = original["cache_metadata"]["actual_metadata"][key]["global_metadata"]
        length, = meta["cache_seq_lens"].tolist()
        if not 0 <= length <= INDEXER_SELECT_KEY_CAPACITY:
            raise ValueError("Refresh history exceeds supported index-key capacity")
        self.check(-6, rank, (shared["candidates"][:1],),
                   (self.inputs[rank].scalar("candidate_blocks", original["physical_layer_id"]),))

    def window(self, rank, offset, shape, dtype):
        limit = offset + math.prod(shape) * dtype.itemsize
        if offset < 0 or limit > self.mesh.window_bytes:
            raise ValueError("Decode communication view exceeds its rank window")
        return (self.mesh.contexts[rank][1] + offset, DTYPE[dtype], len(shape), *shape)

    def route_ids(self, rank, capacity):
        """Let the router and the expert exchange share one layer slice until graph completion."""
        if rank not in self.routes:
            bank = self.runtime.workspace((40, capacity, 6), torch.int32, rank)
            self.routes[rank] = (bank, 0)
        bank, layer = self.routes[rank]
        if layer >= bank.size(0):
            raise ValueError("Too many expert router invocations in a Decode graph")
        self.routes[rank] = (bank, layer + 1)
        return bank[layer]

    def experts(self, rank, handles, owner, sequence):
        inputs = [self._describe(t, rank, write=i == 8) for i, t in enumerate(handles)]
        cursor = 0
        for shape, dtype in (((1, 5120), torch.bfloat16), ((1, 8), torch.int32), ((6, 5120), torch.bfloat16),
                             ((1, 16), torch.int32), ((16, 16), torch.int32)):
            inputs.append(self.window(rank, cursor, shape, dtype))
            cursor += (math.prod(shape) * dtype.itemsize + 63) // 64 * 64
        ctx = self.mesh.contexts[rank][0]
        self.emit(rank, self.expert_stage, inputs, (owner, rank, self.inputs[rank].scalar("ep_signal", sequence - self.ep_start), *([ctx] * 5)))

    def collective(self, packets, collective, group):
        if group != "tp":
            raise ValueError("The one-token TP transport received a different communication group")
        outputs = [{} for _ in packets]
        for case, fields in packets[0].items():
            for name, sample in fields.items():
                count = sample.numel()
                if sample.dtype != torch.bfloat16:
                    raise ValueError("Decode TP packet must be BF16")
                key = collective, count
                if key not in self.tp:
                    path = self.output_dir / f"tp-{collective}-{count}"
                    config = RunConfig(arch=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                        distributed_config=DistributedConfig(device_ids=list(range(16)), num_sub_workers=0),
                        save_kernels=True, save_kernels_dir=str(path))
                    make_tp_exchange(collective, count).compile(config=config)
                    self.tp[key] = self._stage(path / "next_levels" / "tp_rank")
                self.tp_sequence += 1
                def bind(rank):
                    runtime = self.runtime
                    value = packets[rank][case][name]
                    if value.shape != sample.shape or value.dtype != sample.dtype:
                        raise ValueError("Decode TP peer packet ABIs differ")
                    shape = list(value.shape)
                    if collective == "all_gather":
                        shape[0] *= 4
                    result = runtime.workspace(shape, value.dtype, rank, fill=None)
                    outputs[rank].setdefault(case, {})[name] = result
                    src = runtime.tensor(value.view(1, -1), rank)
                    dst = runtime.tensor(result.view(1, -1), rank)
                    runtime.values[rank, result.untyped_storage()._cdata]["device_written"] = True
                    width = count // 4 if collective == "all_to_all" else count
                    inputs = [self._describe(src, rank), self._describe(dst, rank, write=True),
                              self.window(rank, TP_DATA, (4, width), value.dtype),
                              self.window(rank, TP_BASE, (4, 16), torch.int32),
                              self.window(rank, TP_BASE + 256, (4, 16), torch.int32)]
                    ctx = self.mesh.contexts[rank][0]
                    self.emit(rank, self.tp[key], inputs, (rank, self.inputs[rank].scalar("tp_signal", self.tp_sequence - self.tp_start), ctx, ctx, ctx))
                self.runtime.map_ranks(bind)
        return outputs

    def _validation_error(self, rank, step):
        checks = {-3: "finite activations", -5: "sparse cache indices", -6: "refresh candidate coverage"}
        instructions = self.commands[rank]
        cursor = 0
        stage = None
        for _ in range(step):
            op, tensors, scalars = instructions[cursor:cursor + 3]
            cursor += 3
            if op >= 0:
                stage = op
            for _ in range(tensors):
                cursor += 3 + instructions[cursor + 2]
            cursor += scalars
        program = next((path.name for path, index in self.stages.items() if index == stage), None)
        return dict(rank=rank, instruction=step, check=checks.get(op, op), preceding_program=program)

    def launch(self):
        if self.executed or any(not row for row in self.commands):
            raise RuntimeError("Expected exactly one complete command sequence per rank")
        begin = perf_counter()
        self._build()
        runtime = self.runtime
        if self.replaying:
            self.tp_start, self.ep_start = self.tp_sequence, self.mesh.sequence
            self.tp_sequence += self.tp_stride
            self.mesh.sequence += self.ep_stride
        else:
            self.tp_stride = self.tp_sequence - self.tp_start
            self.ep_stride = self.mesh.sequence - self.ep_start
        if max(self.tp_sequence, self.mesh.sequence) >= 2**31:
            raise OverflowError("Decode communication signal exhausted")
        def prepare(rank):
            worker = runtime.workers[rank]
            payload = self.inputs[rank].update(self.provider, self.settings, self.lookup, self.tp_start, self.ep_start, self.cache_blocks)
            if self.replaying:
                args, status, device_inputs = self.prepared[rank]
                runtime._memory.copy(device_inputs.data_ptr, payload.data_ptr(), payload.nbytes, 1)
            else:
                tape = torch.tensor(self.commands[rank], dtype=torch.int64)
                owned = worker.alloc_tensor(tape.shape, tape.dtype, init=tape)
                self.temporary[rank].append(owned)
                status = worker.alloc_tensor((1,), torch.int32)
                self.temporary[rank].append(status)
                device_inputs = worker.alloc_tensor(payload.shape, payload.dtype, init=payload)
                self.temporary[rank].append(device_inputs)
                args = TaskArgs()
                for tensor, direction in ((owned, TensorArgType.INPUT), (status, TensorArgType.OUTPUT_EXISTING),
                                          (device_inputs, TensorArgType.INPUT)):
                    args.add_tensor(make_tensor_arg(worker._impl, tensor), direction)
            runtime._record_transfer(rank, "h2d_bytes", payload.nbytes)
            capture, config = runtime.profiling.prepare_call(self, rank, False, (), (), ring_heap=runtime.ring_heap)
            if id(self.chip) not in worker._cid_cache and len(worker._cid_cache) >= 64:
                first = next(iter(worker._cid_cache))
                worker._impl.unregister(worker._cid_cache.pop(first))
            if id(self.chip) not in worker._cid_cache:
                worker._cid_cache[id(self.chip)] = worker._impl.register(self.chip)
            return args, status, device_inputs, capture, config or self._build_call_config(runtime._run_config or RunConfig())
        prepared = runtime.map_ranks(prepare)
        if not self.replaying:
            self.prepared = [row[:3] for row in prepared]
        self.prepare_seconds = perf_counter() - begin
        def run(rank):
            args, status, _, capture, config = prepared[rank]
            begin = perf_counter()
            try:
                runtime.workers[rank]._run_chip(self.chip, args, config)
                if capture is not None:
                    capture["status"] = "completed"
            finally:
                end = perf_counter()
                if capture is not None:
                    capture.update(host_start=begin, host_end=end)
                runtime.trace.append(dict(kind="worker_run", rank=rank, start=begin, end=end))
            value = torch.empty(1, dtype=torch.int32)
            runtime._memory.copy(value.data_ptr(), status.data_ptr, value.nbytes, 2)
            failures = [self._validation_error(rank, value.item())] if value.item() else []
            if rank in self.routes:
                bank, layers = self.routes[rank]
                source = runtime.tensor(bank, rank)
                ids = torch.empty_like(bank)
                runtime._memory.copy(ids.data_ptr(), source.data_ptr, ids.nbytes, 2)
                routes = ids[:layers, 0, :]
                sorted_ids = routes.sort(dim=1).values
                invalid = (sorted_ids[:, 0] < 0) | (sorted_ids[:, -1] >= 384)
                invalid |= (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any(dim=1)
                if invalid.any():
                    layer = int(invalid.nonzero()[0].item())
                    failures.append(dict(rank=rank, layer=layer, check="Top6 expert IDs",
                                         expert_ids=routes[layer].tolist()))
            return failures
        begin = perf_counter()
        errors = runtime.map_ranks(run)
        self.graph_seconds = perf_counter() - begin
        runtime.stats["dispatches"] += len(prepared)
        if any(errors):
            failures = [failure for rank_errors in errors for failure in rank_errors]
            raise ValueError(f"Decode graph validation failed: {failures}")
        self.executed = True

    def end(self):
        rt = self.runtime
        self.pools = rt.values, rt.free_values, rt.free_workspaces
        rt.values, rt.free_values, rt.free_workspaces = self.outer_pools
        self.provider = self.settings = self.lookup = None

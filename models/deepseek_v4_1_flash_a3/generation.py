# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run one TP-group request through resident A3 prefill and seven decode calls."""

import dataclasses
import hashlib
import json
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from pathlib import Path

import torch
from tokenizers import Tokenizer
from pypto.backend import BackendType
from pypto.ir import DistributedConfig
from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
from pypto.runtime import DeviceTensor, ExecutionMode, RunConfig, StackedDeviceTensor
from pypto.runtime.distributed_runner import DistributedWorker

import config as C
from config import FLASH as M
from engram import HostEngram
from host import prepare_resident_inputs
from utils import DECODE_CACHE_NAMES, decode_cache_views, prepare_decode_inputs
from weights import Checkpoint, ResidentWeights
from replay import LogitValidationError, reset_caches, run_request
from swimlane import profile_program_dirs, summarize_swimlanes


class SwimlaneWorker(DistributedWorker):
    """Write Level 1 records and task dependencies into the same step directory."""

    _capture_dir = ContextVar("a3_step_capture_dir", default=None)

    def run_profiled_step(self, compiled, *arguments, config, directory):
        directory.mkdir()
        token = self._capture_dir.set(directory.resolve())
        try:
            config = dataclasses.replace(config, enable_dep_gen=True, enable_chip_swimlane=0)
            self.run(compiled, *arguments, config=config)
        finally:
            self._capture_dir.reset(token)

    def _prepare_dispatch_config(self, compiled, config):
        call_config, directory, two_pass = super()._prepare_dispatch_config(compiled, config)
        capture = self._capture_dir.get()
        if capture is not None:
            # Enable timing on the native dispatch. The public swimlane flag requests two passes.
            call_config.enable_chip_swimlane = 1
            call_config.output_prefix = str(capture)
        return call_config, directory, two_pass


class HbmSampler:
    """Sample board HBM occupancy and retain the largest observation per phase."""

    def __init__(self, devices):
        self.devices = devices
        self.phase = "prepare"
        self.samples = []
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self):
        completed = subprocess.run(["npu-smi", "info"], text=True, capture_output=True, timeout=20, check=True)
        values = {}
        for line in completed.stdout.splitlines():
            fields = [part.strip() for part in line.split("|")[1:-1]]
            if len(fields) != 3:
                continue
            identity = fields[0].split()
            usage = re.search(r"(\d+)\s*/\s*(\d+)\s*$", fields[2])
            if len(identity) == 2 and all(part.isdigit() for part in identity) and usage:
                values[int(identity[1])] = int(usage[1]) * 1024**2
        if any(device not in values for device in self.devices):
            raise RuntimeError("npu-smi did not report every allocated physical chip")
        self.samples.append(dict(phase=self.phase, monotonic=time.monotonic(),
                                 bytes_per_rank=[values[device] for device in self.devices]))

    def _run(self):
        while not self.stopped.is_set():
            try:
                self.sample()
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                self.samples.append(dict(phase=self.phase, error=str(error)))
            self.stopped.wait(2.0)

    def report(self):
        valid = [sample for sample in self.samples if "bytes_per_rank" in sample]
        if not valid:
            raise RuntimeError("no HBM occupancy sample was collected")
        phases = {}
        for sample in valid:
            phase = phases.setdefault(sample["phase"], [0] * len(self.devices))
            for rank, value in enumerate(sample["bytes_per_rank"]):
                phase[rank] = max(phase[rank], value)
        peak = max((value, rank, name) for name, values in phases.items() for rank, value in enumerate(values))
        return dict(method="npu-smi physical-chip occupancy sampled every 2 s", includes_board_baseline=True,
                    peak_observed_bytes=peak[0], peak_rank=peak[1], peak_phase=peak[2],
                    phases=phases, samples=self.samples)


def audit_caches(runtime, ranks, host_cache, owner, step, directory=None):
    """Check TP replicas, inactive request preservation and retained cache addresses."""
    checks = {}
    for name, initial in host_cache[0].items():
        actual = []
        for rank in range(16):
            value = torch.empty_like(initial)
            handle = ranks[0][rank][name]
            runtime.copy_from(value.data_ptr(), handle.data_ptr, value.numel() * value.element_size(), worker_id=rank)
            actual.append(value)
        active_equal = all(torch.equal(actual[owner].view(torch.uint8), actual[rank].view(torch.uint8))
                           for rank in range(owner, owner + 4))
        inactive_equal = all(torch.equal(actual[rank].view(torch.uint8), host_cache[rank][name].view(torch.uint8))
                             for rank in range(16) if rank // 4 != owner // 4)
        checks[name] = dict(tp_replicas_equal=active_equal, inactive_preserved=inactive_equal)
        if directory is not None:
            torch.save(actual[owner], directory / f"cache_step{step + 1}_{name}.pt")
        if not active_equal or not inactive_equal:
            raise RuntimeError(f"cache ownership validation failed for {name}: {checks[name]}")
    return checks


def read_request(path, tokenizer, artifact_dir):
    """Read one exact 8192-token request or persist a deterministic development replay."""
    development = path is None
    if development:
        block = tokenizer.encode("Explain the mathematical reasoning step by step. ", add_special_tokens=False).ids
        if not block:
            raise ValueError("development replay text encoded to an empty sequence")
        ids = (block * ((8192 + len(block) - 1) // len(block)))[:8192]
        record = {"request_id": "development-8192", "dp_rank": 0, "input_ids": ids}
        path = Path(artifact_dir) / "development_input_ids.json"
        path.write_text(json.dumps(record) + "\n")
    else:
        record = json.loads(Path(path).read_text())
    if isinstance(record, list):
        if len(record) != 1:
            raise ValueError("the acceptance workload contains exactly one request")
        record = record[0]
    if not isinstance(record, dict) or not all(key in record for key in ("request_id", "dp_rank", "input_ids")):
        raise ValueError("replay requires request_id, dp_rank and input_ids")
    if type(record["dp_rank"]) is not int or record["dp_rank"] not in range(4):
        raise ValueError("dp_rank must be an integer in [0, 4)")
    ids = record["input_ids"]
    if not isinstance(ids, list) or len(ids) != 8192 or any(type(token) is not int or not 0 <= token < M.vocab_size for token in ids):
        raise ValueError("input_ids must contain exactly 8192 valid integer token IDs")
    return record, {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    "development_replay": development}


def generate(args, devices, *, manifests):
    """Reuse the same rank-local weight and cache handles across both compiled graphs."""
    if (args.capacity, args.chunk_size, args.max_new_tokens) != (8192, 8192, 8):
        raise ValueError("acceptance requires capacity=8192, chunk-size=8192, max-new-tokens=8")
    checkpoint = Checkpoint(args.checkpoint)
    tokenizer = Tokenizer.from_file(str(checkpoint.root / "tokenizer.json"))
    request, replay = read_request(args.input_ids_json, tokenizer, args.runtime_dir)
    owner = request["dp_rank"] * 4
    weights = ResidentWeights(checkpoint)
    selected_layers = [int(value) for value in args.validate_layers.split(",")] if args.validation_only else None
    engram = HostEngram(checkpoint, tokenizer)
    reference = None
    if args.reference_dir:
        policy = json.loads((args.reference_dir / "policy.json").read_text())
        if policy["checkpoint_identity"] != checkpoint.identity or policy["input_sha256"] != replay["sha256"]:
            raise ValueError("reference checkpoint or exact input replay differs")
    elif args.teacher_forced:
        raise ValueError("teacher forcing requires a frozen eight-step reference")
    config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
                       ring_heap=(4 * 1024**3, 4 * 1024**3, 2 * 1024**3, 1024**3), ring_dep_pool=131072)
    program_dirs = (args.runtime_dir, args.decode_runtime_dir)
    if args.profile_all_steps is not None:
        program_dirs = profile_program_dirs(args.profile_all_steps, program_dirs)
    programs = [DistributedCompiledProgram.from_dir(path, backend_type=config.backend_type,
                execution_mode=config.execution_mode, distributed_config=config.distributed_config)
                for path in program_dirs]
    names = [[parameter.name.removesuffix("__ssa_v0") for parameter in program._get_metadata()[0]]
             for program in programs]
    cache_names = set(DECODE_CACHE_NAMES.values())
    constants = set(weights.tensor_layouts) | cache_names | {"freqs_cos", "freqs_sin"}
    counts = {name: torch.zeros(16, dtype=torch.int32).share_memory_()
              for name in ("moe_num_tokens", "attention_num_tokens")}
    outputs = {"x_hc_out": ((8192, M.hc_mult, M.hidden_size), torch.float32),
               "attn_hc_out": ((8192, M.hc_mult, M.hidden_size), torch.float32),
               "pre_mix_out": ((8192, M.hc_mult), torch.float32),
               "logits": ((16, M.vocab_size), torch.float32), "sampled_ids": ((16, 8), torch.int32)}
    report = {"status": "running", "platform": "a2a3", "topology": manifests[0]["topology"],
              "checkpoint_identity": checkpoint.identity, "weight_layout": C.WEIGHT_LAYOUT_VERSION,
              "cache_layout": C.CACHE_LAYOUT_VERSION, "replay": replay, "request_id": request["request_id"],
              "dp_rank": request["dp_rank"], "resident_weight_bytes_per_rank": weights.bytes_per_rank,
              "history_capacity": 8200, "prefill_calls": 0, "decode_calls": 0,
              "output_sequences": [], "numerical_acceptance": "pending independent W8A8 logits reference",
              "artifacts": [str(args.runtime_dir), str(args.decode_runtime_dir)],
              "teacher_forced": args.teacher_forced, "logit_checks": [], "cache_checks": [],
              "instrumented_cache_audits": args.audit_caches}
    report["initialized_weight_layers"] = selected_layers if selected_layers is not None else list(range(40))
    memory = HbmSampler(devices)
    memory.sample()
    memory.thread.start()
    load_start = time.monotonic()
    try:
        print("A3_PREPARE_RESIDENT_GRAPHS", flush=True)
        worker = SwimlaneWorker if args.profile_all_steps is not None else DistributedWorker
        with worker(programs, config, persistent=True, reset_persistent_windows=False,
                    inherited_host_tensors=list(counts.values())) as runtime:
            memory.phase = "graphs_prepared"
            memory.sample()
            print("A3_RESIDENT_GRAPHS_READY", flush=True)
            torch.set_num_threads(4)
            ranks = [[{} for _ in range(16)] for _ in programs]
            host_cache = []
            rope_tables = None
            for rank in range(16):
                memory.phase = "weight_load"
                values = prepare_resident_inputs(weights, [], 8192, max_seq_len=8200, rope_tables=rope_tables,
                                                  engram=engram, request_id=rank)
                rope_tables = values["freqs_cos"], values["freqs_sin"]
                host_cache.append({name: values[name] for name in cache_names})
                for name in names[0]:
                    if name in weights.tensor_layouts or name in outputs:
                        shape, dtype = weights.tensor_layouts[name] if name in weights.tensor_layouts else outputs[name]
                        handle = runtime.alloc_tensor(shape, dtype, worker_id=rank)
                    else:
                        value = values[name]
                        handle = runtime.alloc_tensor(value.shape, value.dtype, init=value, worker_id=rank)
                    ranks[0][rank][name] = handle
                values = prepare_resident_inputs(weights, [], C.DECODE_INPUT_MAX_TOKENS, max_seq_len=8200,
                                                  cache_state=host_cache[rank], rope_tables=rope_tables,
                                                  engram=engram, request_id=rank)
                values = prepare_decode_inputs(values)
                views = decode_cache_views(host_cache[rank])
                for name in names[1]:
                    if name in counts:
                        continue
                    if name in DECODE_CACHE_NAMES:
                        source = ranks[0][rank][DECODE_CACHE_NAMES[name]]
                        handle = DeviceTensor(source.data_ptr, views[name].shape, source.dtype, buffer=source.buffer)
                    elif name in constants or name in ("logits", "sampled_ids"):
                        handle = ranks[0][rank][name]
                    else:
                        value = values[name]
                        handle = runtime.alloc_tensor(value.shape, value.dtype, init=value, worker_id=rank)
                    ranks[1][rank][name] = handle
            with ThreadPoolExecutor(max_workers=8) as uploaders:
                uploads = {uploaders.submit(weights.upload, runtime, rank, ranks[0][rank], layers=selected_layers): rank
                           for rank in range(16)}
                for uploaded in as_completed(uploads):
                    uploaded.result()
                    print(f"A3_RANK_WEIGHTS_READY {uploads[uploaded]}", flush=True)
            torch.set_num_threads(16)
            memory.phase = "weights_resident"
            memory.sample()
            report["weight_load_seconds"] = time.monotonic() - load_start
            report["allocator_committed_bytes_per_rank"] = [runtime.committed_device_memory(rank)
                                                            for rank in range(16)]
            report["resident_weights_shared"] = all(ranks[0][rank][name] is ranks[1][rank][name]
                                                     for rank in range(16) for name in weights.tensor_layouts)
            report["resident_caches_shared"] = all(
                ranks[0][rank][source].data_ptr == ranks[1][rank][target].data_ptr
                for rank in range(16) for target, source in DECODE_CACHE_NAMES.items()
            )
            arguments = []
            for index in range(2):
                stacked = {name: StackedDeviceTensor([rank[name] for rank in ranks[index]],
                           (16, *ranks[index][0][name].shape), tuple(range(16)))
                           for name in names[index] if name not in counts}
                stacked.update(counts if index == 1 else {})
                arguments.append(stacked)
            if args.validate_layers:
                from validation import LayerValidationError, validate_layers

                layer_records = report.setdefault("layer_validation", [])
                try:
                    validate_layers(
                        runtime, programs, arguments, names, ranks, weights, engram, tokenizer, request,
                        host_cache, rope_tables, args, config, memory, audit_caches,
                        records=layer_records,
                    )
                except LayerValidationError as error:
                    if args.validation_only:
                        raise
                    report["layer_validation_error"] = str(error)
            if args.validation_only:
                report.update(status="passed", acceptance_scope="representative_layers",
                              numerical_acceptance="full-model logits were not run")
                return report
            if args.reference_dir:
                reference = torch.load(args.reference_dir / "logits.pt", map_location="cpu", weights_only=True)
                if tuple(reference["logits"].shape) != (8, M.vocab_size) or len(reference["token_ids"]) != 8:
                    raise ValueError("reference must contain all eight complete-vocabulary outputs")
            replays = []
            report["request_replays"] = replays
            if args.validate_teacher_forced:
                teacher = {}
                report["teacher_forced_validation"] = teacher
                reset_caches(runtime, ranks, host_cache)
                try:
                    run_request(
                        runtime, programs, arguments, names, ranks,
                        weights, engram, tokenizer, request, host_cache, rope_tables,
                        config, memory, teacher, reference=reference, teacher_forced=True,
                        audit=audit_caches if args.audit_caches else None, label="teacher_forced",
                    )
                except LogitValidationError as error:
                    report["teacher_forced_error"] = str(error)
            total = args.warmup_requests + args.measured_requests
            for iteration in range(total):
                if iteration or args.validate_teacher_forced:
                    reset_caches(runtime, ranks, host_cache)
                label = f"warmup_{iteration}" if iteration < args.warmup_requests else f"measured_{iteration}"
                replay_report = {}
                capture_dir = args.profile_all_steps if iteration == args.warmup_requests else None
                if capture_dir is not None:
                    report["profile_request"] = replay_report
                replays.append(replay_report)
                audit_request = args.audit_caches and (iteration < args.warmup_requests or not args.warmup_requests)
                try:
                    run_request(
                        runtime, programs, arguments, names, ranks,
                        weights, engram, tokenizer, request, host_cache, rope_tables,
                        config, memory, replay_report, reference=reference, teacher_forced=args.teacher_forced,
                        audit=audit_caches if audit_request else None, save_logits=args.save_logits, label=label,
                        profile=capture_dir is not None, profile_dir=capture_dir,
                    )
                finally:
                    report.update(replay_report)
            measured = replays[args.warmup_requests:]
            timed = [replay for replay in measured if not replay["profiled"]] or measured
            report["benchmark"] = dict(warmup_requests=args.warmup_requests,
                                       measured_requests=args.measured_requests,
                                       sample_labels=[replay["label"] for replay in timed],
                                       instrumented=any(r["profiled"] or r["instrumented_cache_audits"] for r in timed),
                                       clock="host monotonic")
            for metric in ("ttft_seconds", "tpot_seconds", "end_to_end_seconds"):
                values = torch.tensor([replay[metric] for replay in timed], dtype=torch.float64)
                report["benchmark"][metric] = dict(median=float(values.quantile(0.5)),
                                                  p95=float(values.quantile(0.95)), samples=values.tolist())
            if args.profile_last_decode:
                reset_caches(runtime, ranks, host_cache)
                profiled = {}
                report["profile_request"] = profiled
                run_request(
                    runtime, programs, arguments, names, ranks,
                    weights, engram, tokenizer, request, host_cache, rope_tables,
                    config, memory, profiled, reference=reference, label="profile", profile=True,
                )
            memory.sample()
            if "layer_validation_error" in report:
                raise ValueError(report["layer_validation_error"])
            if "teacher_forced_error" in report:
                raise ValueError(report["teacher_forced_error"])
        if args.profile_all_steps is not None:
            report["operator_timings"] = summarize_swimlanes(args.profile_all_steps, devices, owner, convert=True)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        memory.stopped.set()
        memory.thread.join(timeout=22)
        report["hbm"] = memory.report()
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report

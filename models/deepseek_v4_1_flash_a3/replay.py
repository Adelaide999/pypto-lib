# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Replay eight-token requests with resident weights and cache allocations."""

import dataclasses
import json
import time

import torch

import config as C
from config import FLASH as M
from host import prepare_resident_inputs
from reference import compare_logits
from swimlane import step_name
from telemetry import route_distribution
from utils import DECODE_CACHE_NAMES, prepare_decode_inputs


class LogitValidationError(ValueError):
    """Report completed requests whose logits did not satisfy the frozen policy."""


def reset_caches(runtime, ranks, host_cache):
    """Reset the same seven resident allocations for the next request-slot reuse."""
    for rank in range(16):
        for name, value in host_cache[rank].items():
            runtime.copy_to(ranks[0][rank][name].data_ptr, value.data_ptr(),
                            value.numel() * value.element_size(), worker_id=rank)


def run_request(
    runtime, programs, arguments, names, ranks,
    weights, engram, tokenizer, request, host_cache, rope_tables,
    config, memory, report, *, reference=None, teacher_forced=False,
    audit=None, save_logits=None, label="greedy", profile=False, profile_dir=None,
):
    """Submit one full prefill and seven decode calls, then check frozen logits."""
    if teacher_forced and reference is None:
        raise ValueError("teacher forcing requires the frozen reference")
    owner = request["dp_rank"] * 4
    constants = set(weights.tensor_layouts) | set(DECODE_CACHE_NAMES.values()) | {"freqs_cos", "freqs_sin"}
    counts = {name: arguments[1][name] for name in ("moe_num_tokens", "attention_num_tokens")}
    logits = torch.empty(M.vocab_size, dtype=torch.float32)
    sampled = torch.empty(8, dtype=torch.int32)
    report.update(prefill_calls=0, decode_calls=0, output_sequences=[], logit_checks=[], cache_checks=[],
                  teacher_forced=teacher_forced, label=label, instrumented_cache_audits=audit is not None,
                  profiled=profile)
    committed = 0
    generated, token_times, engine_times, observed_logits = [], [], [], []
    tokenizer_config = json.loads((weights.checkpoint.root / "tokenizer_config.json").read_text())
    eos = tokenizer_config["eos_token"]
    eos_id = tokenizer.token_to_id(eos["content"] if isinstance(eos, dict) else eos)
    start = time.monotonic()
    for step in range(8):
        index = int(step > 0)
        capacity = C.DECODE_INPUT_MAX_TOKENS if index else 8192
        current_ids = [generated[-1]] if index else request["input_ids"]
        if index and teacher_forced:
            current_ids = reference["token_ids"][step - 1:step]
        for rank in range(16):
            active = rank // 4 == request["dp_rank"]
            ids = current_ids if active else []
            values = prepare_resident_inputs(weights, ids, capacity, start_pos=committed if active else 0,
                                             max_seq_len=8200, cache_state=host_cache[rank],
                                             rope_tables=rope_tables, engram=engram, request_id=rank)
            if rank != owner:
                values["logit_rows"].fill_(-1)
            if index:
                values = prepare_decode_inputs(values)
                for tensor in counts.values():
                    tensor[rank] = len(ids)
            for name, value in values.items():
                if name in constants or name not in ranks[index][rank]:
                    continue
                runtime.copy_to(ranks[index][rank][name].data_ptr, value.data_ptr(),
                                value.numel() * value.element_size(), worker_id=rank)
        engine_start = time.monotonic()
        memory.phase = "decode" if index else "prefill"
        marker = dict(label=label, step=step + 1, mode=memory.phase)
        if profile_dir is not None:
            marker["profiled"] = True
        print("A3_ENGINE_BEGIN", json.dumps(marker), flush=True)
        program_args = [arguments[index][name] for name in names[index]]
        if profile_dir is not None:
            runtime.run_profiled_step(programs[index], *program_args, config=config, directory=profile_dir / step_name(step))
        else:
            dispatch_config = dataclasses.replace(config, enable_chip_swimlane=4) if profile and step == 7 else config
            runtime.run(programs[index], *program_args, config=dispatch_config)
        engine_times.append(time.monotonic() - engine_start)
        print("A3_ENGINE_END", json.dumps(marker), flush=True)
        for name, value in (("logits", logits), ("sampled_ids", sampled)):
            runtime.copy_from(value.data_ptr(), ranks[index][owner][name].data_ptr,
                              value.numel() * value.element_size(), worker_id=owner)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("nonfinite owner logits")
        token = int(logits.argmax())
        if token != int(sampled[0]):
            raise RuntimeError("device greedy token differs from the complete vocabulary argmax")
        generated.append(token)
        observed_logits.append(logits.clone())
        committed += len(current_ids)
        token_times.append(time.monotonic())
        report["decode_calls" if index else "prefill_calls"] += 1
        report["output_sequences"] = [{"request_id": request["request_id"], "token_ids": generated.copy()}]
        print("A3_TOKEN", json.dumps(dict(label=label, step=step + 1, token_id=token, committed=committed)), flush=True)
        if audit is not None:
            report["cache_checks"].append(audit(runtime, ranks, host_cache, owner, step))
        if token == eos_id and step < 7:
            raise RuntimeError("acceptance replay reached EOS before producing eight tokens")
    if save_logits is not None:
        torch.save(dict(logits=torch.stack(observed_logits), token_ids=generated,
                        checkpoint_identity=weights.checkpoint.identity), save_logits)
    report.update(status="passed" if reference is not None else "executed_pending_numerical_acceptance",
                  committed_positions=committed, ttft_seconds=token_times[0] - start,
                  tpot_seconds=(token_times[-1] - token_times[0]) / 7,
                  end_to_end_seconds=token_times[-1] - start,
                  token_intervals_seconds=[b - a for a, b in zip(token_times, token_times[1:])],
                  engine_seconds=engine_times, text=tokenizer.decode(generated, skip_special_tokens=False))
    report["prefill_routing"] = route_distribution(runtime, ranks, owner)
    if reference is not None:
        report["numerical_acceptance"] = reference["policy"]
        for step, value in enumerate(observed_logits):
            checked = compare_logits(value, reference["logits"][step], reference["policy"])
            report["logit_checks"].append(checked)
        failed_steps = [step + 1 for step, checked in enumerate(report["logit_checks"]) if not checked["passed"]]
        if failed_steps:
            report["status"] = "failed"
            raise LogitValidationError(f"frozen W8A8 logit acceptance failed at steps {failed_steps}")
    return report

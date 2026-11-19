# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Greedy DP4/TP4/EP16 generation with one resident Prefill/Decode weight and cache bank."""

import argparse
import json
import sys
import traceback
from pathlib import Path
from time import perf_counter

from tokenizers import Tokenizer

from .decode_network import DecodeNetwork, TOPOLOGY
from .prefill_fwd import prefill_fwd
from .prefill_prepare import checkpoint_cache, tokenize_prompt
from .runtime_config import (
    MAX_CONTEXT_TOKENS,
    MAX_MODEL_CONTEXT_TOKENS,
    MAX_PREFILL_CHUNK_SIZE,
    PREFILL_MOE_SLAB_ROWS,
    ParallelPlan,
    configure_devices,
    prefill_execution_capacity,
)
from .utils import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--assets", type=Path, help="Deprecated; contracts are derived from checkpoint metadata")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompt")
    inputs.add_argument("--tokens")
    inputs.add_argument("--tokens-file", type=Path, help="JSON array of prompt token IDs")
    parser.add_argument("--devices", type=lambda value: [int(item) for item in value.split(",")], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, help="Defaults to 128, bounded by any experimental token budget")
    parser.add_argument("--prefill-moe-slab-rows", choices=PREFILL_MOE_SLAB_ROWS, type=int,
                        help="Maximum source rows per rank in one Prefill expert dispatch/return round")
    parser.add_argument("--max-num-batched-tokens", type=int, choices=(16, 64, 256, 512),
                        help="Opt into experimental packed requests with this new-token budget")
    parser.add_argument("--tp", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--hbm-budget", type=Path, help="JSON per-rank HBM budget for expanded token/history runs")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--logits-mode", choices=("last", "all"), default="last",
                        help="Project the last active position or every position in each chunk")
    parser.add_argument("--dump-logits", action="store_true", help="Save vocabulary tensors for diagnostics")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5),
                        help="Device capture level: 0 disables capture; the bare flag selects 1")
    parser.add_argument("--profile-phase", choices=("all", "prefill", "decode"), default="all",
                        help="Capture all forward phases or only Prefill/Decode")
    parser.add_argument("--profile-output", type=Path, help="Device trace root; defaults to OUTPUT/profiling")
    args = parser.parse_args(argv)
    started = perf_counter()
    experimental = args.max_num_batched_tokens is not None
    if not experimental and args.tp != TOPOLOGY["tp"]:
        parser.error("Alternate TP requires --max-num-batched-tokens to opt into experimental packed requests")
    if experimental and args.logits_mode == "all":
        parser.error("Experimental packed requests project only final request positions")
    chunk_limit = args.max_num_batched_tokens if experimental else MAX_PREFILL_CHUNK_SIZE
    if args.chunk_size is None:
        args.chunk_size = min(128, chunk_limit)
    if not 1 <= args.chunk_size <= chunk_limit or not 1 <= args.max_new_tokens <= 256:
        parser.error(f"Require chunk size 1..{chunk_limit} and a new-token limit in 1..256")
    topology = dict(TOPOLOGY, tp=args.tp, dp=16 // args.tp)
    ParallelPlan.from_topology(topology, args.devices)
    args.checkpoint, args.output = args.checkpoint.resolve(), args.output.resolve()
    result_dir = (args.result or args.output / "generation").resolve()
    model = json.loads((args.checkpoint / "config.json").read_text())
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    if args.tokens_file is not None:
        raw = json.loads(args.tokens_file.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            parser.error("--tokens-file must contain a JSON array of token IDs")
    else:
        raw = args.prompt if args.prompt is not None else [int(item) for item in args.tokens.split(",")]
    context_limit = MAX_MODEL_CONTEXT_TOKENS if experimental else MAX_CONTEXT_TOKENS
    tokens = tokenize_prompt(raw, args.checkpoint, model["text_config"]["vocab_size"], max_tokens=context_limit)
    if len(tokens) + args.max_new_tokens > context_limit:
        parser.error(f"Prompt plus generation exceeds the {context_limit}-position session reservation")
    devices = configure_devices(args.devices, world=16)
    args.output.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(parents=True, exist_ok=True)
    preparation_started = perf_counter()
    capacity = args.max_num_batched_tokens if experimental else prefill_execution_capacity(args.chunk_size)
    decode_capacity = capacity if experimental else 16
    cache, config = checkpoint_cache(args.checkpoint, topology, capacity=capacity)
    checkpoint_contract_seconds = perf_counter() - preparation_started
    config.update(chunk_size=args.chunk_size, context_capacity=len(tokens) + args.max_new_tokens,
                  logits_mode=args.logits_mode, dump_logits=args.dump_logits,
                  prefill_moe_slab_rows=args.prefill_moe_slab_rows)
    profiling = dict(level=args.enable_chip_swimlane, phase=args.profile_phase,
                     output_dir=str((args.profile_output or args.output / "profiling").resolve()))
    config["profiling"] = profiling
    if experimental:
        config["runtime_limits"] = dict(max_num_seqs=1, max_seqs_per_dp=1,
                                        max_num_batched_tokens=args.max_num_batched_tokens,
                                        max_model_len=MAX_MODEL_CONTEXT_TOKENS)
    if args.hbm_budget is not None:
        config["hbm_budget"] = json.loads(args.hbm_budget.read_text())
    eos = model.get("eos_token_id", model["text_config"].get("eos_token_id"))
    report = dict(status="executing", checkpoint=str(args.checkpoint), topology=topology,
                  devices=devices, chunk_size=args.chunk_size, prefill_capacity=config["capacity"],
                  decode_capacity=decode_capacity, requested_prefill_moe_slab_rows=args.prefill_moe_slab_rows,
                  prompt_token_ids=tokens, generated_token_ids=[], phases=[],
                  profiling=profiling, logits_mode="request-final" if experimental else args.logits_mode,
                  dump_logits=args.dump_logits, accuracy="not_run")
    if experimental:
        report["runtime_limits"] = config["runtime_limits"]
    write_json(result_dir / "prompt.json", report)
    session, network = None, None
    generated, decode_timings, token_timings = [], [], []
    report["timings"] = dict(
        scope="host wall time; includes dispatch, communication, validation and report I/O",
        checkpoint_contract_seconds=checkpoint_contract_seconds,
    )
    report["decode_timings"] = decode_timings
    report["token_timings"] = token_timings
    prefill_started = None
    previous_token = started

    def emit(token, phase):
        nonlocal previous_token
        emitted_at = perf_counter()
        token_timings.append(dict(step=len(generated), phase=phase, token_id=token,
                                 since_start_seconds=emitted_at - started,
                                 since_previous_token_seconds=emitted_at - previous_token))
        previous_token = emitted_at
        report["token_timings"] = token_timings
        generated.append(token)
        report["generated_token_ids"] = list(generated)
        report["phases"].append(phase)
        text = tokenizer.decode(generated, skip_special_tokens=False)
        (result_dir / "answer.txt").write_text(text, encoding="utf-8")
        write_json(result_dir / "tokens.json", generated)
        print("[generation-token] " + json.dumps(dict(step=len(generated) - 1, token_id=token, text=text),
                                                ensure_ascii=False), flush=True)

    try:
        prefill_started = perf_counter()
        prefill = prefill_fwd(tokens, weights=args.checkpoint, cache=cache, config=config, devices=devices,
            output=args.output / "prefill", keep_runtime=True,
            progress=lambda event: print(json.dumps(dict(event, phase="prefill")), flush=True))
        prefill_call_seconds = perf_counter() - prefill_started
        session = prefill.pop("session")
        prefill_identity = id(session.runtime)
        before_decode = session.weights.counters()
        chunks = prefill.get("completed_prefill_chunks", [prefill])
        prefill_forward_seconds = sum(chunk["timings"]["chunk_seconds"] for chunk in chunks)
        prefill_moe = session.program.moe_geometry[0]
        decode_moe = session.phase_programs[decode_capacity].moe_geometry[0]
        report.update(moe_slab_rows=prefill_moe["slab_rows"], prefill_expert_capacity=prefill_moe["expert_capacity"],
                      decode_moe_slab_rows=decode_moe["slab_rows"], decode_expert_capacity=decode_moe["expert_capacity"])
        report["timings"].update(
            prefill_call_seconds=prefill_call_seconds,
            compilation=prefill["compilation"],
            weights=dict(session.weights.initialization_timings),
            prefill_forward_seconds=prefill_forward_seconds,
        )
        logits = prefill["request_outputs"][0]["logits"] if experimental else prefill["logits"][0]["fresh"][-1]
        emit(int(logits.argmax()), "prefill")
        network = DecodeNetwork(runtime=session.runtime, resident_weights=session.weights,
            cache_views=session.cache_views, session=session, output=args.output / "decode",
            progress=lambda event: print(json.dumps(dict(event, phase="decode")), flush=True))
        for _ in range(args.max_new_tokens - 1):
            if not args.ignore_eos and generated[-1] == eos:
                break
            logits = network.step(generated[-1])
            decode_timings.append(dict(network.last_timings))
            emit(int(logits.argmax()), "decode")
        session.weights.assert_static_unchanged()
        if prefill_identity != id(network.runtime) or before_decode != session.weights.counters():
            raise RuntimeError("Decode changed the generation runtime or static weight bank")
        report.update(status="executed", generated_text=tokenizer.decode(generated, skip_special_tokens=False),
            prefill_layers=len(prefill["completed_layers"]), decode_layers=40 if decode_timings else 0,
            decode_steps=len(decode_timings), static_checkpoint_reads=session.weights.reader.read_calls,
            static_device_uploads=session.weights.device_upload_calls,
            checkpoint_read_calls=session.weights.reader.read_calls,
            host_weight_preparation_seconds=session.weights.initialization_timings["host_weight_preparation_seconds"],
            device_weight_upload_calls=session.weights.device_upload_calls,
            prefill_runtime_identity=prefill_identity, decode_runtime_identity=id(network.runtime),
            same_runtime=True, static_counters_before_decode=before_decode,
            static_counters_after_decode=session.weights.counters(), decode_timings=decode_timings,
            session=session.summary())
        report["timings"].update(
            time_to_first_token_seconds=token_timings[0]["since_start_seconds"],
            decode_step_seconds=[entry["total_seconds"] for entry in decode_timings],
            decode_total_seconds=sum(entry["total_seconds"] for entry in decode_timings),
            elapsed_before_cleanup_seconds=perf_counter() - started,
        )
    except BaseException as error:
        if prefill_started is not None:
            report["timings"].setdefault("prefill_call_seconds", perf_counter() - prefill_started)
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(result_dir / "failure.json", dict(report, traceback=traceback.format_exc()))
        raise
    finally:
        primary_error = sys.exc_info()[0] is not None
        cleanup_error = None
        cleanup_failures = []
        for owner in (network, session):
            if owner is not None:
                try:
                    owner.close()
                except BaseException as error:
                    cleanup_error = cleanup_error or error
                    cleanup_failures.append(dict(error_type=type(error).__name__, error=str(error)))
        if session is not None:
            report["session_closed"] = session.closed
            report["runtime_closed"] = session.runtime.summary()["closed"]
            report["profiling"] = session.runtime.profiling.summary()
        if cleanup_failures:
            report["cleanup_failures"] = cleanup_failures
            if not primary_error:
                report.update(status="failed", error_type=type(cleanup_error).__name__, error=str(cleanup_error))
        report["timings"].update(
            decode_step_seconds=[entry["total_seconds"] for entry in decode_timings],
            decode_total_seconds=sum(entry["total_seconds"] for entry in decode_timings),
        )
        report["elapsed_including_cleanup_seconds"] = perf_counter() - started
        write_json(result_dir / "result.json", report)
        if cleanup_error is not None and not primary_error:
            write_json(result_dir / "failure.json", report)
            raise cleanup_error
    print("[generation] RUN PASS; " + json.dumps(dict(
        tokens=generated, prefill_layers=report["prefill_layers"], decode_layers=report["decode_layers"],
        decode_steps=len(decode_timings), same_runtime=True, topology=topology)), flush=True)


if __name__ == "__main__":
    main()

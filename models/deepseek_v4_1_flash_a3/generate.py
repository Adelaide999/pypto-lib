# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run checkpoint Prefill40, transfer request state and generate with Decode40."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import multiprocessing
from pathlib import Path
import sys
from time import perf_counter
import traceback

import torch

from .runtime_config import configure_devices
from .runtime_utils import write_json


TOPOLOGY = dict(dp=16, tp=1, ep=16, pp=1, pcp=1, dcp=1)


def run_prompt(checkpoint, tokens, devices, output, chunk_size, compile_only=False):
    """Execute genuine Prefill in a process whose device workers end at the phase boundary."""
    begin = perf_counter()
    from .prefill_fwd import compile_prefill, prefill_fwd
    from .prefill_prepare import checkpoint_cache

    cache, config = checkpoint_cache(checkpoint, TOPOLOGY)
    timings = dict(checkpoint_cache_preparation_seconds=perf_counter() - begin)
    config.update(replicate_prompt=True, chunk_size=chunk_size, export_state=str(output / "handoff.pt"))
    def progress(event):
        print(json.dumps(dict(event, phase="prefill", at=datetime.now(timezone.utc).isoformat())), flush=True)
    if compile_only:
        program = compile_prefill(tokens, weights=checkpoint, cache=cache, config=config,
            output=output / "prefill-compiled")
        write_json(output / "prefill-compilation.json", program.summary())
        return
    stage_begin = perf_counter()
    result = prefill_fwd(tokens, weights=checkpoint, cache=cache, config=config, devices=devices,
        output=output / "prefill", diagnostics=False, progress=progress)
    timings["forward_seconds"] = perf_counter() - stage_begin
    timings["runtime"] = result["runtime_timings"]
    chunks = result.get("completed_prefill_chunks", [result])
    timings["chunks"] = [chunk["stage_timings"] for chunk in chunks]
    timings["automatic_chunk_preparation_and_cleanup_seconds"] = timings["forward_seconds"] - timings["runtime"]["body_seconds"]
    stage_begin = perf_counter()
    if result["completed_layers"] != list(range(40)):
        raise RuntimeError("Prefill did not complete every layer")
    logits = torch.stack([row["fresh"][-1] for row in result["logits"]])
    if list(logits.shape) != [16, 129280] or not torch.isfinite(logits).all():
        raise RuntimeError("Prefill head did not produce a finite complete vocabulary row per rank")
    torch.save(dict(logits=logits, tokens=logits.argmax(-1)), output / "prefill-head.pt")
    timings["head_validation_and_save_seconds"] = perf_counter() - stage_begin
    timings["child_seconds"] = perf_counter() - begin
    write_json(output / "prefill-result.json", dict(status="executed", layers=40,
        topology=TOPOLOGY, context_length=len(tokens), prefill_chunks=(len(tokens) + chunk_size - 1) // chunk_size,
        logits_shape=list(logits.shape), finite=True, accuracy="not_run", timings=timings))
    print("[generation] Prefill40+LMhead RUN PASS; state exported", flush=True)


def _prompt_process(settings):
    """Publish failures before releasing the Prefill runtime process."""
    try:
        run_prompt(**settings)
    except BaseException:
        write_json(settings["output"] / "prefill-failure.json", dict(traceback=traceback.format_exc()))
        raise


def prepare_prompt(args):
    """Load the tokenizer and render an optional single-user checkpoint chat template."""
    from tokenizers import Tokenizer

    from .prefill_prepare import tokenize_prompt

    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    model = json.loads((args.checkpoint / "config.json").read_text())
    rendered = args.prompt
    if args.chat:
        spec = importlib.util.spec_from_file_location("checkpoint_chat_encoding", args.checkpoint / "encoding/encoding.py")
        encoder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(encoder)
        rendered = encoder.encode_messages([dict(role="user", content=args.prompt)], thinking_mode="chat")
        raw = tokenizer.encode(rendered, add_special_tokens=False).ids
    elif args.prompt is not None:
        raw = tokenizer.encode(args.prompt).ids
    else:
        raw = [int(token) for token in args.tokens.split(",")]
    tokens = tokenize_prompt(raw, args.checkpoint, model["text_config"]["vocab_size"])
    return model, tokenizer, tokens, rendered


def publish_token(output, tokenizer, generated, *, source, position, started):
    """Write actual decoded text and return one timestamped token event."""
    ids = [row[0] for row in generated]
    raw_text = tokenizer.decode(ids, skip_special_tokens=False)
    text = tokenizer.decode(ids, skip_special_tokens=True)
    event = dict(step=len(generated), source=source, token_ids=generated[-1], token_id=ids[-1],
        token_piece=tokenizer.id_to_token(ids[-1]), text=text, raw_text=raw_text,
        logits_position=position, available_seconds=perf_counter() - started,
        at=datetime.now(timezone.utc).isoformat())
    (output / "answer.txt").write_text(text, encoding="utf-8")
    (output / "raw-answer.txt").write_text(raw_text, encoding="utf-8")
    write_json(output / "tokens.json", generated)
    with (output / "token-events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    print("[generation-token] " + json.dumps(event, ensure_ascii=False), flush=True)
    return event


def main(argv=None):
    """Generate a bounded greedy continuation on the caller's allocated sixteen devices."""
    begin = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--devices", type=lambda value: [int(item) for item in value.split(",")])
    parser.add_argument("--ep", type=int, choices=(16,), default=16)
    parser.add_argument("--tp", type=int, choices=(1,), default=1)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompt", help="Raw completion text encoded with the checkpoint tokenizer")
    inputs.add_argument("--tokens", help="Comma-separated explicit vocabulary token IDs")
    parser.add_argument("--chat", action="store_true", help="Wrap --prompt with the checkpoint's chat template")
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--ignore-eos", action="store_true", help="Run the fixed token count for execution acceptance")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.compile_only and args.devices is None:
        parser.error("Execution requires explicit allocated --devices")
    if not 1 <= args.chunk_size <= 16 or args.max_new_tokens < 1:
        parser.error("Require chunk size 1..16 and a positive new-token limit")
    if args.chat and args.prompt is None:
        parser.error("--chat requires a text --prompt")
    if args.devices is not None:
        configure_devices(args.devices, world=16)
    model, tokenizer, tokens, rendered = prepare_prompt(args)
    if len(tokens) + args.max_new_tokens - 1 > 256:
        parser.error("Prompt plus consumed Decode tokens must fit the 256-position cache")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "prompt.json", dict(prompt=args.prompt, rendered_prompt=rendered,
        input_ids=tokens, input_tokens=len(tokens), chat_template=args.chat,
        checkpoint=str(args.checkpoint.resolve()), max_new_tokens=args.max_new_tokens,
        topology=TOPOLOGY, assigned_devices=args.devices))
    timings = dict(prompt_preparation_seconds=perf_counter() - begin, decode_steps=[])
    stage_begin = perf_counter()
    settings = dict(checkpoint=args.checkpoint, tokens=tokens, devices=args.devices, output=args.output,
        chunk_size=args.chunk_size, compile_only=args.compile_only)
    process = multiprocessing.get_context("spawn").Process(target=_prompt_process, args=(settings,))
    process.start()
    process.join()
    timings["prefill_process_seconds"] = perf_counter() - stage_begin
    if process.exitcode != 0:
        raise RuntimeError(f"Prefill process failed with exit code {process.exitcode}; see prefill-failure.json")
    # The parent has not initialized CANN before the distributed Decode workers fork.
    # Decode's existing factories read their parallel specialization during import.
    stage_begin = perf_counter()
    if not any(arg == "--ep" or arg.startswith("--ep=") for arg in sys.argv):
        sys.argv.extend(("--ep", "16"))
    if not any(arg == "--tp" or arg.startswith("--tp=") for arg in sys.argv):
        sys.argv.extend(("--tp", "1"))
    from .decode_network import DecodeNetwork, DecodeState
    from .decode_network_weights import DecodeNetworkWeights

    network = DecodeNetwork(devices=args.devices, output=args.output / "decode",
        progress=lambda event: print(json.dumps(dict(event, phase="decode", at=datetime.now(timezone.utc).isoformat())), flush=True))
    timings["decode_network_preparation_seconds"] = perf_counter() - stage_begin
    if args.compile_only:
        write_json(args.output / "decode-compilation.json", network.compile())
        print("[generation] Prefill40 + Decode40 + both heads COMPILE PASS; execution=not_run", flush=True)
        return 0
    stage_begin = perf_counter()
    packet = torch.load(args.output / "handoff.pt", map_location="cpu", weights_only=True)
    if (packet["checkpoint"] != str(args.checkpoint.resolve())
            or packet["histories"] != [tokens] * 16 or packet["topology"] != TOPOLOGY):
        raise RuntimeError("Prefill handoff belongs to another checkpoint, prompt or topology")
    state = DecodeState.from_generation_state(packet)
    write_json(args.output / "handoff.json", state.handoff)
    timings["state_load_and_import_seconds"] = perf_counter() - stage_begin
    stage_begin = perf_counter()
    logits = torch.load(args.output / "prefill-head.pt", map_location="cpu", weights_only=True)["logits"]
    timings["prefill_logits_load_seconds"] = perf_counter() - stage_begin
    generated, phases, positions, events = [], [], [], []
    eos = model.get("eos_token_id", model["text_config"].get("eos_token_id"))
    stage_begin = perf_counter()
    weights = DecodeNetworkWeights(args.checkpoint, 16)
    timings["decode_weight_reader_preparation_seconds"] = perf_counter() - stage_begin
    timings["selection_text_and_save_seconds"] = 0.0
    timings["decode_compilation_seconds"] = 0.0
    try:
        for step in range(args.max_new_tokens):
            stage_begin = perf_counter()
            current = logits.argmax(-1).tolist()
            generated.append(current)
            phases.append("prefill" if step == 0 else "decode")
            positions.append([len(history) - 1 for history in state.histories])
            events.append(publish_token(args.output, tokenizer, generated, source=phases[-1],
                position=positions[-1][0], started=begin))
            timings["selection_text_and_save_seconds"] += perf_counter() - stage_begin
            if (not args.ignore_eos and all(token == eos for token in current)) or step + 1 == args.max_new_tokens:
                break
            if not network.compiled:
                stage_begin = perf_counter()
                write_json(args.output / "decode-compilation.json", network.compile())
                timings["decode_compilation_seconds"] = perf_counter() - stage_begin
            stage_begin = perf_counter()
            logits = network.step(current, weights=weights, state=state)
            timings["decode_steps"].append(dict(epoch=state.epoch, seconds=perf_counter() - stage_begin,
                details=network.last_timings))
            write_json(args.output / "timings-partial.json", timings)
        stage_begin = perf_counter()
        weights.close()
        network.close()
        timings["decode_cleanup_seconds"] = perf_counter() - stage_begin
        timings["prefill"] = json.loads((args.output / "prefill-result.json").read_text())["timings"]
        timings["time_to_first_token_seconds"] = events[0]["available_seconds"]
        timings["inter_token_seconds"] = [b["available_seconds"] - a["available_seconds"] for a, b in zip(events, events[1:])]
        timings["total_seconds"] = perf_counter() - begin
        exclusive = ("prompt_preparation_seconds", "prefill_process_seconds", "decode_network_preparation_seconds",
            "state_load_and_import_seconds", "prefill_logits_load_seconds", "decode_weight_reader_preparation_seconds",
            "selection_text_and_save_seconds", "decode_compilation_seconds", "decode_cleanup_seconds")
        accounted = sum(timings[name] for name in exclusive) + sum(row["seconds"] for row in timings["decode_steps"])
        timings["other_host_seconds"] = timings["total_seconds"] - accounted
        timings["scope"] = "One real generation; host wall seconds from main entry, excluding queue, shell environment activation and Python module imports"
        write_json(args.output / "timings.json", timings)
        report = dict(status="executed", topology=TOPOLOGY, prompt_ids=tokens, generated_ids=generated,
            token_sources=phases, logits_positions=positions, prefill_layers=40, decode_layers=40,
            decode_steps=state.epoch, final_context_lengths=[len(h) for h in state.histories],
            stop_reason="eos" if not args.ignore_eos and all(token == eos for token in generated[-1]) else "token_limit",
            handoff=state.handoff, finite=True, accuracy="not_run", seconds=timings["total_seconds"],
            prompt=args.prompt, chat_template=args.chat, generated_text=events[-1]["text"],
            raw_generated_text=events[-1]["raw_text"], timings_file=str(args.output / "timings.json"))
        write_json(args.output / "result.json", report)
        print("[generated-text] " + events[-1]["text"], flush=True)
        print("[generation] Prefill40 -> state handoff -> Decode40 RUN PASS; accuracy=not_run; "
            + json.dumps(report), flush=True)
    finally:
        weights.close()
        network.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

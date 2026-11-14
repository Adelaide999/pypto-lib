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
import traceback
from pathlib import Path

from tokenizers import Tokenizer

from .decode_network import DecodeNetwork, TOPOLOGY
from .prefill_fwd import prefill_fwd
from .prefill_prepare import checkpoint_cache, tokenize_prompt
from .runtime_config import configure_devices
from .utils import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--assets", type=Path, help="Deprecated; contracts are derived from checkpoint metadata")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompt")
    inputs.add_argument("--tokens")
    parser.add_argument("--devices", type=lambda value: [int(item) for item in value.split(",")], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--ignore-eos", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.chunk_size <= 16 or not 1 <= args.max_new_tokens <= 256:
        parser.error("Require chunk size 1..16 and a new-token limit in 1..256")
    args.checkpoint, args.output = args.checkpoint.resolve(), args.output.resolve()
    result_dir = (args.result or args.output / "generation").resolve()
    model = json.loads((args.checkpoint / "config.json").read_text())
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    raw = args.prompt if args.prompt is not None else [int(item) for item in args.tokens.split(",")]
    tokens = tokenize_prompt(raw, args.checkpoint, model["text_config"]["vocab_size"])
    if len(tokens) + args.max_new_tokens > 2048:
        parser.error("Prompt plus generation exceeds the 2048-position session reservation")
    devices = configure_devices(args.devices, world=16)
    args.output.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(parents=True, exist_ok=True)
    cache, config = checkpoint_cache(args.checkpoint, TOPOLOGY)
    config.update(chunk_size=args.chunk_size, context_capacity=len(tokens) + args.max_new_tokens)
    eos = model.get("eos_token_id", model["text_config"].get("eos_token_id"))
    report = dict(status="executing", checkpoint=str(args.checkpoint), topology=TOPOLOGY,
                  devices=devices, prompt_token_ids=tokens, generated_token_ids=[], phases=[], accuracy="not_run")
    write_json(result_dir / "prompt.json", report)
    session, network = None, None
    generated, decode_timings = [], []

    def emit(token, phase):
        generated.append(token)
        report["generated_token_ids"] = list(generated)
        report["phases"].append(phase)
        text = tokenizer.decode(generated, skip_special_tokens=False)
        (result_dir / "answer.txt").write_text(text, encoding="utf-8")
        write_json(result_dir / "tokens.json", generated)
        print("[generation-token] " + json.dumps(dict(step=len(generated) - 1, token_id=token, text=text),
                                                ensure_ascii=False), flush=True)

    try:
        prefill = prefill_fwd(tokens, weights=args.checkpoint, cache=cache, config=config, devices=devices,
            output=args.output / "prefill", keep_runtime=True,
            progress=lambda event: print(json.dumps(dict(event, phase="prefill")), flush=True))
        session = prefill.pop("session")
        prefill_identity = id(session.runtime)
        before_decode = session.weights.counters()
        emit(int(prefill["logits"][0]["fresh"][-1].argmax()), "prefill")
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
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(result_dir / "failure.json", dict(report, traceback=traceback.format_exc()))
        raise
    finally:
        if network is not None:
            network.close()
        if session is not None:
            session.close()
            report["session_closed"] = session.closed
            report["runtime_closed"] = session.runtime.summary()["closed"]
        write_json(result_dir / "result.json", report)
    print("[generation] RUN PASS; " + json.dumps(dict(
        tokens=generated, prefill_layers=report["prefill_layers"], decode_layers=report["decode_layers"],
        decode_steps=len(decode_timings), same_runtime=True, topology=TOPOLOGY)), flush=True)


if __name__ == "__main__":
    main()

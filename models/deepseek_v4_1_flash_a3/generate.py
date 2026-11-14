# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Feed actual greedy output tokens through persistent complete-network Prefill chunks."""

import argparse
import json
import math
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
import torch
from tokenizers import Tokenizer
from torch.utils._python_dispatch import _disable_current_modes
from .prefill_fwd import prefill_fwd
from .prefill_prepare import build_chunk_metadata, checkpoint_cache, reserve_cache_layouts, tokenize_prompt
from .utils import file_sha, write_json


TOPOLOGY = dict(dp=16, tp=1, ep=16, pp=1, pcp=1, dcp=1)


def atomic_json(path, value):
    """Publish a small receipt with an atomic rename."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


class FeedbackInputs:
    """Own fixed original cache geometry and prepare controls for actual consumed token history."""

    def __init__(self, checkpoint, assets, directory, prompt, maximum):
        self.checkpoint, self.directory, self.prompt = checkpoint, directory, prompt
        self.model = json.loads((checkpoint / "config.json").read_text())
        self.base = json.loads((assets / "input-contract.json").read_text())
        self.templates = torch.load(assets / "cache-ABI.pt", map_location="cpu", weights_only=True)
        self.engram = torch.load(assets / "engram-contracts.pt", map_location="cpu", weights_only=True)["layers"]
        if self.base["topology"] != TOPOLOGY or set(self.engram) != {1, 14}:
            raise ValueError("Require the original complete DP4/TP4/EP16 cache and Engram contracts")
        self.reservation = len(prompt["input_token_ids"]) + maximum
        # Only original host metadata arithmetic is outside the product container-only guard.
        with _disable_current_modes():
            self.layouts, self.allocations, self.reserved = reserve_cache_layouts(self.templates, self.reservation)
        self.identities = {name: record["sha256"] for name, record in prompt["assets"].items()}
        self.directory.mkdir(parents=True, exist_ok=False)

    def chunk(self, epoch, current, history, predecessor):
        """Build positions, valid lengths, original aliases and lineage for one actual chunk."""
        if not 1 <= len(current) <= 16 or len(history) + len(current) > self.reservation:
            raise ValueError("Feedback exceeds the fixed query or persistent context reservation")
        request = dict(request_slot=0, token_ids=list(current), history_tokens=list(history),
            initial_context_length=len(history), live_token_mask=[1] * len(current),
            position_ids=list(range(len(history), len(history) + len(current))))
        actual = dict(topology=TOPOLOGY, checkpoint=str(self.checkpoint), checkpoint_sha256=self.identities,
            epoch=epoch, predecessor_input_sha256=predecessor,
            DP_engines=[dict(dp_rank=dp, active=len(current) if dp == 0 else 0,
                            requests=[request] if dp == 0 else []) for dp in range(4)])
        directory = self.directory / f"chunk-{epoch}"
        directory.mkdir(exist_ok=False)
        input_path = directory / "input-contract.json"
        write_json(input_path, actual)
        identity = file_sha(input_path)
        payloads, sources = [], []
        # These functions operate solely on integer scheduler controls, never model activations or logits.
        with _disable_current_modes():
            for rank, template in enumerate(self.templates):
                payload = build_chunk_metadata(template, self.layouts[rank], self.allocations[rank],
                    tokens=current, context=len(history), tp_rank=rank % 4, active=rank // 4 == 0,
                    reserved=self.reserved)
                payloads.append(payload)
        for rank, payload in enumerate(payloads):
            path = directory / f"rank-{rank}-metadata.pt"
            torch.save(payload, path)
            sources.append(dict(rank=rank, path=str(path), sha256=file_sha(path)))
        contract = dict(topology=TOPOLOGY, checkpoint=str(self.checkpoint), model_config=self.model,
            required_modes=self.base["required_modes"], capacity=16, scheduler_capacity=8192,
            fresh_shared_initialization={"topk_indices": -1}, input_contract_path=str(input_path),
            input_contract_sha256=identity)
        cache = dict(cache_ABI=payloads, engram_contracts=self.engram,
            cache_ABI_sources=[dict(rank=row["rank"], payload_sha256=row["sha256"], report_sha256=row["sha256"])
                               for row in sources])
        return dict(inputs=dict(input_contract=actual, prepare_input_stage=True), cache=cache,
                    contract=contract, metadata_sources=sources)


def load_prompt(path, checkpoint):
    """Verify and re-encode the original chat template rather than a completion prompt."""
    import importlib.util

    prompt = json.loads(path.read_text())
    for name, record in prompt["assets"].items():
        if file_sha(checkpoint / name) != record["sha256"]:
            raise ValueError("Original tokenizer/checkpoint metadata changed: " + name)
    spec = importlib.util.spec_from_file_location("original_chat_encoder", checkpoint / "encoding/encoding.py")
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    rendered = encoder.encode_messages(prompt["messages"], thinking_mode=prompt["thinking_mode"])
    ids = tokenizer.encode(rendered, add_special_tokens=False).ids
    if rendered != prompt["rendered_prompt"] or ids != prompt["input_token_ids"]:
        raise ValueError("Original chat template or exact prompt token IDs differ")
    model = json.loads((checkpoint / "config.json").read_text())
    eos = model.get("eos_token_id", model["text_config"].get("eos_token_id"))
    if eos != prompt["eos_token_id"] or tokenizer.token_to_id("<｜end▁of▁sentence｜>") != eos:
        raise ValueError("Original EOS identity differs")
    return prompt, tokenizer


def main(argv=None):
    """Run Prefill followed by Decode in one generation-scoped session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--assets", type=Path, help="Deprecated; cache contracts are derived from checkpoint")
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
    if (not 1 <= args.chunk_size <= 16 or not 1 <= args.max_new_tokens <= 256):
        parser.error("Require chunk size 1..16 and a new-token limit in 1..256")
    args.output = args.output.resolve()
    result_dir = (args.result or args.output / "generation").resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(parents=True, exist_ok=True)
    model = json.loads((args.checkpoint / "config.json").read_text())
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    raw = args.prompt if args.prompt is not None else [int(item) for item in args.tokens.split(",")]
    tokens = tokenize_prompt(raw, args.checkpoint, model["text_config"]["vocab_size"])
    eos = model.get("eos_token_id", model["text_config"].get("eos_token_id"))
    if len(tokens) + args.max_new_tokens > 2048:
        parser.error("Prompt plus generation exceeds the A3 cache context bound")
    from .runtime_config import configure_devices
    devices = configure_devices(args.devices, world=TOPOLOGY["ep"])
    cache, config = checkpoint_cache(args.checkpoint, TOPOLOGY)
    report = dict(status="executing", checkpoint=str(args.checkpoint), topology=TOPOLOGY,
                  devices=devices, prompt_token_ids=tokens, generated_token_ids=[], phases=[],
                  accuracy="not_run", static_checkpoint_reads=0, static_device_uploads=0)
    (result_dir / "prompt.json").write_text(json.dumps(report, indent=2) + "\n")
    session = None
    network = None
    weights = None
    try:
        prefill = prefill_fwd(tokens, weights=args.checkpoint, cache=cache, config=config, devices=devices,
                              output=args.output / "prefill", keep_runtime=True,
                              progress=lambda event: print(json.dumps(dict(event, phase="prefill")), flush=True))
        session = prefill.pop("session")
        logits = prefill["logits"][0]["fresh"][-1]
        current = int(logits.argmax())
        # Capture the Prefill accounting before closing its workers. A3's
        # standalone Decode DistributedWorker must start after those workers
        # have released the physical devices.
        prefill_runtime = session.summary()
        session.release_runtime()
        generated = [current]
        report["generated_token_ids"] = list(generated)
        report["phases"].append("prefill")
        text = tokenizer.decode(generated, skip_special_tokens=False)
        (result_dir / "answer.txt").write_text(text, encoding="utf-8")
        (result_dir / "tokens.json").write_text(json.dumps(generated) + "\n")
        print("[generation-token] " + json.dumps(
            dict(step=0, token_id=current, text=text), ensure_ascii=False), flush=True)

        decode_result = None
        remaining = args.max_new_tokens - 1
        if remaining and (args.ignore_eos or current != eos):
            # Decode's bounded state can retain only 256 positions. Keep the
            # latest history rows when a longer Prefill prompt is supplied.
            history = tokens[-(256 - remaining):]
            decode_output = args.output / "decode"
            command = [
                sys.executable, "-m", "models.deepseek_v4_1_flash_a3.decode_network",
                "--checkpoint", str(args.checkpoint),
                "--devices", ",".join(str(device) for device in devices),
                "--tokens", str(current),
                "--history-tokens", ",".join(str(token) for token in history),
                "--decode-steps", str(remaining),
                "--output", str(decode_output),
            ]
            print("[generation-decode-process] " + json.dumps(command), flush=True)
            subprocess.run(command, check=True)
            decode_result = json.loads((decode_output / "result.json").read_text())
            for row in decode_result["generated_ids"]:
                next_token = int(row[0])
                if not args.ignore_eos and next_token == eos:
                    break
                generated.append(next_token)
                report["generated_token_ids"] = list(generated)
                report["phases"].append("decode")
                text = tokenizer.decode(generated, skip_special_tokens=False)
                (result_dir / "answer.txt").write_text(text, encoding="utf-8")
                (result_dir / "tokens.json").write_text(json.dumps(generated) + "\n")
                print("[generation-token] " + json.dumps(
                    dict(step=len(generated) - 1, token_id=next_token, text=text),
                    ensure_ascii=False), flush=True)
        resident_weights = prefill_runtime["resident_weights"]
        decode_init = (decode_result or {}).get("resident_weight_initialization", {})
        report.update(status="executed", generated_text=tokenizer.decode(generated, skip_special_tokens=False),
                      prefill_layers=40, decode_layers=40 if decode_result else 0,
                      decode_steps=max(0, len(generated) - 1),
                      static_checkpoint_reads=resident_weights["checkpoint_read_calls"] + decode_init.get("checkpoint_read_calls", 0),
                      static_device_uploads=resident_weights["device_weight_upload_call_count"] + decode_init.get("device_weight_upload_calls", 0),
                      prefill_runtime=prefill_runtime, decode_runtime=decode_result)
        (result_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print("[generation] Prefill40 -> Decode40 RUN PASS; " + json.dumps(report), flush=True)
    except BaseException as error:
        failure = dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        (result_dir / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    finally:
        if weights is not None and (session is None or weights is not session.weights):
            weights.close()
        if network is not None:
            network.close()
        if session is not None:
            session.close()


if __name__ == "__main__":
    main()

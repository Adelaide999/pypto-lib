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
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
import torch
from tokenizers import Tokenizer
from torch.utils._python_dispatch import _disable_current_modes
from .prefill_fwd import prefill_fwd
from .prefill_prepare import build_chunk_metadata, reserve_cache_layouts
from .runtime_utils import file_sha, write_json
from .state import build_state_regions


TOPOLOGY = dict(dp=4, tp=4, ep=16, pp=1, pcp=1, dcp=1)


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
            regions = build_state_regions(payloads, identity)
        for rank, payload in enumerate(payloads):
            path = directory / f"rank-{rank}-metadata.pt"
            torch.save(payload, path)
            sources.append(dict(rank=rank, path=str(path), sha256=file_sha(path)))
        region_path = directory / "state-regions.json"
        write_json(region_path, regions)
        contract = dict(topology=TOPOLOGY, checkpoint=str(self.checkpoint), model_config=self.model,
            required_modes=self.base["required_modes"], capacity=16, scheduler_capacity=8192,
            fresh_shared_initialization={"topk_indices": -1}, input_contract_path=str(input_path),
            input_contract_sha256=identity, state_regions=dict(path=str(region_path), sha256=file_sha(region_path)))
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


def main():
    """Run one actual public-leased greedy generation or a bounded metadata readiness check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--source-identity", type=Path, required=True)
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.result.mkdir(parents=True, exist_ok=True)
    prompt, tokenizer = load_prompt(args.prompt, args.checkpoint)
    identity = json.loads(args.source_identity.read_text())
    report = dict(at=datetime.now(timezone.utc).isoformat(), status="preparing", prompt=prompt["prompt"],
        source=identity, checkpoint=str(args.checkpoint), tokenizer=prompt["assets"],
        rendered_prompt=prompt["rendered_prompt"], input_token_ids=prompt["input_token_ids"],
        eos_token_id=prompt["eos_token_id"], topology=TOPOLOGY, diagnostics=False,
        initial_max_new_tokens=128, maximum_new_tokens=256, active_token_limit=128,
        generated_token_ids=[], raw_decoded_answer="", steps=[], eos=False, truncated=False,
        complete_answer_observed=False, accuracy="not_run", exit_code=None, device_computation="PyPTO only",
        algorithm="Persistent-cache sequential Prefill feedback; one complete 40-layer forward per new token",
        optional_intermediate_precision_captures_hash_gates=False)
    start = monotonic()
    generated, steps = report["generated_token_ids"], report["steps"]

    def publish(event):
        report.update(event)
        report.update(updated_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=monotonic() - start)
        atomic_json(args.result / "generation-result.json", report)
        atomic_json(args.result / "status.json", dict(
            at=report["updated_at"], stage=report.get("stage"), epoch=report.get("epoch"), layer=report.get("layer"),
            task_id=report.get("task_id"), lease=report.get("lease"), actual_devices=report.get("actual_devices"),
            status=report["status"], generated_tokens=len(generated), generated_token_ids=list(generated),
            eos=report["eos"], truncated=report["truncated"], complete_answer_observed=report["complete_answer_observed"],
            elapsed_seconds=report["elapsed_seconds"], result=str(args.result / "generation-result.json")))
        print(json.dumps(event, ensure_ascii=False), flush=True)

    (args.result / "answer.txt").write_text("")
    publish(dict(stage="preparing_metadata"))
    try:
        builder = FeedbackInputs(args.checkpoint, args.assets, args.output / "prepared", prompt, 256)
        history, current = [], list(prompt["input_token_ids"])
        chunk = builder.chunk(0, current, history, None)
        if args.check_only:
            from .prefill_fwd import prepare_network_provider

            provider = prepare_network_provider(dict(chunk["inputs"], **chunk["cache"]), chunk["contract"],
                                                "generation-readiness", geometry_only=True)
            geometry = provider.runtime_geometry()
            if geometry["table_pages"] != 3 or provider.history.cache_rows != geometry["cache_blocks"] * 128:
                raise ValueError("Reserved context/history geometry differs")
            if [row["SP_valid_rows"] for row in provider.batch_for_layer(0)["cases"]["fresh"]] != [2] * 4 + [0] * 12:
                raise ValueError("Original chat input is not mapped to its actual active TP rows")
            publish(dict(status="host_interfaces_ready", stage="readiness_complete", geometry=geometry,
                execution="not_run", compilation="not_run", reserved_context_tokens=builder.reservation, exit_code=0))
            return
        if args.submission is None:
            raise ValueError("Actual generation requires its real public submission receipt")
        submission = json.loads(args.submission.read_text())
        job = Path(os.environ["DS41_JOB_ROOT"])
        lease_path = job / "lease.json"
        lease = json.loads(lease_path.read_text())
        devices = [int(value) for value in os.environ["TASK_DEVICE"].split(",")]
        if lease["assigned_devices"] != devices or len(devices) != 16:
            raise ValueError("Actual device mapping differs from the leased public group")
        report.update(task_id=submission["task_id"], lease=str(lease_path), lease_at_start=lease,
                      actual_devices=devices, logical_rank_mapping=list(range(16)), status="executing")

        from .engram_host import HostEngram

        host = HostEngram(args.checkpoint, chunk["cache"]["engram_contracts"], args.output.name)

        def feedback():
            nonlocal chunk, current, history
            for epoch in range(256):
                network = args.output / "network"
                chunk_output = network if epoch == 0 else network / f"continuation-chunk{epoch}"
                execution = json.loads((chunk_output / "execution.json").read_text())
                runtime = json.loads((chunk_output / "runtime_config.json").read_text())
                if runtime["completed_layers"] != list(range(40)):
                    raise RuntimeError("Greedy feedback requires an actual completed full40 forward")
                logits_path = chunk_output / "actual-layer39-output/rank-0-fresh-network-logits.pt"
                result = torch.load(logits_path, map_location="cpu", weights_only=True)
                positions = list(range(len(history), len(history) + len(current)))
                actual_positions = result["positions"]
                if isinstance(actual_positions, torch.Tensor):
                    actual_positions = actual_positions.tolist()
                if (result["active"] != len(current) or actual_positions != positions
                        or result["DP_rank"] != 0):
                    raise RuntimeError("Final logits do not cover the actual request's last valid row")
                row = result["logits"][len(current) - 1].tolist()
                if len(row) != builder.model["text_config"]["vocab_size"] or not all(map(math.isfinite, row)):
                    raise RuntimeError("Final valid vocabulary logits are incomplete or nonfinite")
                token = max(range(len(row)), key=row.__getitem__)
                generated.append(token)
                steps.append(dict(step=epoch, input_contract=chunk["contract"]["input_contract_path"],
                    input_sha256=chunk["contract"]["input_contract_sha256"], input_token_ids=list(current),
                    positions=positions, context_length=len(history), valid_length=len(current),
                    logits_path=str(logits_path), logits_sha256=file_sha(logits_path),
                    last_valid_row=len(current) - 1, token_id=token, greedy_logit=row[token],
                    timing=execution["timings"]))
                report["raw_decoded_answer"] = tokenizer.decode(generated, skip_special_tokens=False)
                (args.result / "answer.txt").write_text(report["raw_decoded_answer"])
                atomic_json(args.result / "tokens.json", generated)
                publish(dict(stage="generated_real_product_token", epoch=epoch, layer=None, token_id=token,
                             raw_decoded_answer=report["raw_decoded_answer"]))
                if token == prompt["eos_token_id"]:
                    publish(dict(status="completed_eos", stage="generation_complete", eos=True,
                                 stop_reason="eos", complete_answer_observed=True))
                    return
                if len(generated) == 128:
                    publish(dict(stage="extending_token_limit", active_token_limit=256))
                if len(generated) == 256:
                    publish(dict(status="completed_token_limit", stage="generation_complete",
                                 truncated=True, stop_reason="token_limit"))
                    return
                history += current
                current = [token]
                chunk = builder.chunk(epoch + 1, current, history, chunk["contract"]["input_contract_sha256"])
                chunk["inputs"]["engram_lookup"] = host.lookup(chunk["inputs"], cache=chunk["cache"], config=chunk["contract"])
                yield {name: chunk[name] for name in ("inputs", "contract", "metadata_sources")}

        try:
            chunk["inputs"]["engram_lookup"] = host.lookup(chunk["inputs"], cache=chunk["cache"], config=chunk["contract"])
            result = prefill_fwd(chunk["inputs"], weights=args.checkpoint, cache=chunk["cache"],
                config=chunk["contract"], devices=devices, output=args.output / "network",
                continuation=feedback(), diagnostics=False, progress=publish)
        finally:
            host.close()
        report["compilation"] = result["compilation"]
        report["resident_runtime"] = result["resident_runtime"]
        publish(dict(stage="generation_terminal", exit_code=0))
    except BaseException as error:
        failure = dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        atomic_json(args.result / "failure.json", failure)
        publish(dict(status="failed", stage="functional_failure", exit_code=1, **failure))
        raise


if __name__ == "__main__":
    main()


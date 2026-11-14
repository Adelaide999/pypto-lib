# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One-token DP4/TP4/EP16 Decode over the generation session's resident network."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from .config import FLASH
from .cache import resolve_layer_role
from .network import NetworkProgram, _merge_inputs, prepare_network_provider, run_network_runtime
from .prefill_prepare import TokenChunkInputs, checkpoint_cache, tokenize_prompt
from .runtime_config import configure_devices
from .utils import write_json


TOPOLOGY = dict(dp=4, tp=4, ep=16, pp=1, pcp=1, dcp=1)
MAX_CONTEXT = 2048


def network_plan():
    """Report physical cache/source roles without initializing any device."""
    return [dict(layer=layer, **resolve_layer_role(layer)) for layer in range(40)]


class DecodeNetwork:
    """Advance one request token using the same TP4 layer ABI in both execution modes."""

    def __init__(self, *, runtime=None, resident_weights=None, cache_views=None, session=None,
                 devices=None, output=None, progress=None):
        self.session = session
        self.runtime = runtime
        self.weights = resident_weights
        self.cache_views = cache_views
        self.devices = devices
        self.output = Path(output) if output is not None else Path("build_output/ds41-decode-network")
        self.progress = progress
        self._steps = 0
        self._closed = False
        self.last_timings = {}
        self.initialization_timings = {}
        self.dp_rank = 0
        if session is not None:
            self._bind(session)
        elif any(value is not None for value in (runtime, resident_weights, cache_views)):
            raise ValueError("External Decode requires the generation session with its live provider and history")

    def _bind(self, session):
        if session.closed or session.failed or session.contract["topology"] != TOPOLOGY:
            raise ValueError("Decode requires a live DP4/TP4/EP16 generation session")
        if session.request_inputs is None:
            raise ValueError("Decode requires the session's original request metadata owner")
        if self.devices is not None and list(map(int, self.devices)) != list(map(int, session.runtime.devices)):
            raise ValueError("Decode devices differ from the existing session worker mapping")
        for supplied, actual in ((self.runtime, session.runtime), (self.weights, session.weights),
                                 (self.cache_views, session.cache_views)):
            if supplied is not None and supplied is not actual:
                raise ValueError("Decode cannot replace the session runtime, weights or cache allocations")
        self.session, self.runtime, self.weights = session, session.runtime, session.weights
        self.cache_views = session.cache_views
        self.dp_rank = session.request_inputs.config.get("dp_rank", 0)
        self.initialization_timings = dict(session.weights.initialization_timings)
        self.weights.assert_static_unchanged()

    def step(self, token):
        """Execute all forty layers, Engram and the TP-sharded vocabulary head on A3."""
        if self._closed or self.session is None:
            raise RuntimeError("Decode requires an open initialized generation session")
        if type(token) is not int or not 0 <= token < FLASH.vocab_size:
            raise ValueError("Decode requires one vocabulary token ID")
        begin = perf_counter()
        self.weights.assert_static_unchanged()
        before = self.weights.counters()
        cache_h2d = self.runtime.stats["cache_h2d_bytes"]
        chunk = self.session.request_inputs.chunk([token], dp_rank=self.dp_rank)
        result = self.session.advance(chunk, output=self.output / f"step-{self._steps + 1}",
                                      progress=self.progress, phase="decode")
        logits = result["logits"][self.dp_rank * TOPOLOGY["tp"]]["fresh"][-1]
        if not torch.isfinite(logits).all():
            self.session.failed = True
            raise RuntimeError("Non-finite Decode vocabulary logits")
        self.weights.assert_static_unchanged()
        after = self.weights.counters()
        self._steps += 1
        deltas = {name: after[name] - before[name] for name in before}
        self.last_timings = dict(total_seconds=perf_counter() - begin, layers=40,
            weight_load_seconds=result["timings"]["weight_load_seconds"],
            **deltas,
            cache_h2d_bytes=self.runtime.stats["cache_h2d_bytes"] - cache_h2d,
            runtime_identity=id(self.runtime), head_backend="a3", engram_compute_backend="a3",
            engram_lookup="dynamic_host_rows")
        write_json(self.output / f"step-{self._steps}" / "decode.json", self.last_timings)
        return logits

    def close(self):
        """Detach this driver; the generation owner closes its shared resources."""
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("-d", "--devices", type=lambda value: [int(item) for item in value.split(",")])
    parser.add_argument("-p", "--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("--ep", type=int, choices=(16,), default=16)
    parser.add_argument("--tp", type=int, choices=(4,), default=4)
    parser.add_argument("--dp", type=int, choices=(4,), default=4)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--tokens", default="0", help="Prompt IDs consumed one token at a time")
    parser.add_argument("--history-tokens", default="", help="History to execute before the current token")
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("build_output/ds41-decode-network"))
    args = parser.parse_args(argv)
    if args.plan_only:
        print(json.dumps(dict(topology=TOPOLOGY, layers=network_plan()), indent=2))
        return 0
    if args.checkpoint is None or (not args.compile_only and args.devices is None):
        parser.error("Require --checkpoint and, for execution, explicit allocated --devices")
    history = [int(item) for item in args.history_tokens.split(",")] if args.history_tokens else []
    tokens = [int(item) for item in args.tokens.split(",")]
    if args.decode_steps < 1 or not tokens or len(history) + len(tokens) + args.decode_steps - 1 > MAX_CONTEXT:
        parser.error("Require positive steps within the 2048-position session reservation")
    if not args.compile_only:
        configure_devices(args.devices, world=16)
    cache, config = checkpoint_cache(args.checkpoint, TOPOLOGY)
    ids = tokenize_prompt(history + tokens, args.checkpoint, FLASH.vocab_size)
    request = TokenChunkInputs(checkpoint=args.checkpoint, cache=cache, config=config,
        output=args.output / "prepared", context_capacity=len(ids) + args.decode_steps - 1,
        prepare_engram=not args.compile_only)
    session, network = None, None
    try:
        first = request.chunk(ids[:1])
        merged = _merge_inputs(first["inputs"], first["cache"])
        if args.compile_only:
            provider = prepare_network_provider(merged, first["contract"], args.output.name, geometry_only=True)
            program = NetworkProgram(args.output / "compiled").compile(provider, config["model_config"],
                                                                       TOPOLOGY, cache["engram_contracts"])
            write_json(args.output / "compiled.json", program.summary())
            print("[decode-network] shared 40-layer TP4 network + device Engram/head COMPILE PASS; execution=not_run")
            return 0
        progress = lambda record: print(json.dumps(record), flush=True)
        result = run_network_runtime(merged, contract=first["contract"], output=args.output / "network",
            devices=args.devices, keep_runtime=True, request_inputs=request, progress=progress)
        session = result.pop("session")
        network = DecodeNetwork(session=session, output=args.output / "decode", progress=progress)
        logits = result["logits"][0]["fresh"][-1]
        for token in ids[1:]:
            logits = network.step(token)
        generated = [int(logits.argmax())]
        for _ in range(args.decode_steps - 1):
            logits = network.step(generated[-1])
            generated.append(int(logits.argmax()))
        summary = dict(status="executed", layers=40, input_steps=len(ids), decode_steps=args.decode_steps,
            generated_ids=generated, accuracy="not_run", topology=TOPOLOGY,
            resident_weight_initialization=network.initialization_timings, timings=network.last_timings,
            session=session.summary())
        write_json(args.output / "result.json", summary)
        print("[decode-network] 40L+device Engram/head RUN PASS; accuracy=not_run", flush=True)
    finally:
        if network is not None:
            network.close()
        if session is not None:
            session.close()
        request.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

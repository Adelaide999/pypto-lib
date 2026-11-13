# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-scheduled 40-layer Decode with persistent caches and a final LM head."""

from __future__ import annotations

import argparse
import inspect
import json
import traceback
from pathlib import Path
from time import perf_counter

import torch
import torch.multiprocessing as multiprocessing
from golden import normalize_config
from pypto.ir import DistributedConfig

from .config import FLASH
from .decode_engram import make_decode_engram
from .decode_head import make_decode_head
from .decode_layer import EP_SIZE, get_layer_entry
from .decode_network_weights import DecodeNetworkWeights
from .embedding import make_initial_hc, make_vocab_embedding
from .generation_state import validate_generation_state
from .norm_rope import make_rope_tables
from .runtime_config import validate_devices


CAPACITY = 16
MAX_CONTEXT = 256
RING_HEAP = (2 * 1024**3, 0, 0, 0)


def _serve_io(connection):
    """Execute single-device stages outside the parent that forks distributed workers."""
    from pypto.ir import CompiledProgram

    compiled = {}
    try:
        while True:
            request = connection.recv()
            if request is None:
                return
            directory, arguments, settings = request
            try:
                if directory not in compiled:
                    compiled[directory] = CompiledProgram.from_dir(directory)
                compiled[directory](*arguments, config=normalize_config(settings))
                connection.send(None)
            except Exception:
                connection.send(traceback.format_exc())
            del request, arguments
    finally:
        connection.close()


def layer_entry(layer):
    """Select the layer body after the separate checkpoint Engram stage."""
    return get_layer_entry(0 if layer == 1 else layer)


def network_plan():
    """Resolve every original physical layer and its persistent source owners."""
    return [dict(layer=layer, ratio=role.compression_ratio, mode=role.mode.value,
                 kv_source=role.kv_source_layer_id, index_source=role.index_source_layer_id,
                 engram=layer in FLASH.engram_layer_ids, entry=layer_entry(layer)._func.__name__)
            for layer in range(FLASH.num_hidden_layers) for role in [FLASH.layer_config(layer)]]


def entry_layout(entry):
    """Read tensor shapes and scalar names from the product entry's own annotations."""
    function = entry._func
    closure = inspect.getclosurevars(function).nonlocals
    annotations = inspect.get_annotations(function, globals=function.__globals__, locals=closure, eval_str=True)
    result = {}
    for name in inspect.signature(function).parameters:
        value = annotations[name]
        shape = tuple(value.shape) if hasattr(value, "shape") else None
        result[name] = (shape, getattr(torch, {"fp32": "float32", "fp16": "float16"}.get(str(value.dtype), str(value.dtype))))
    return result


class DecodeState:
    """Own per-layer SWA, source KV/index planes, candidates and request history."""

    def __init__(self, world=EP_SIZE):
        if world != EP_SIZE:
            raise ValueError("The state world must match the imported --ep specialization")
        self.world, self.epoch, self.failed = world, 0, False
        self.histories = [[] for _ in range(world)]
        self.layers = []
        self.sources = {}
        for row in network_plan():
            layout = entry_layout(layer_entry(row["layer"]))
            state = {}
            for name in ("cache", "cache_lens", "prepared_indices"):
                if name in layout:
                    shape, dtype = layout[name]
                    state[name] = torch.full(shape, -1 if name == "prepared_indices" else 0, dtype=dtype)
            self.layers.append(state)
            if row["mode"] == "full":
                source = {}
                for name in ("compressed_cache", "index_cache", "index_scale", "compressor_state",
                             "source_positions", "cache_lengths"):
                    if name in layout:
                        shape, dtype = layout[name]
                        source[name] = torch.zeros(shape, dtype=dtype)
                self.sources[row["layer"]] = source
        self.candidates = dict(candidate_block_ids=torch.full((world, 16, 64), -1, dtype=torch.int32),
                               candidate_block_counts=torch.zeros((world, 16), dtype=torch.int32),
                               candidate_topk_lengths=torch.zeros((world, 16), dtype=torch.int32))

    @classmethod
    def from_generation_state(cls, packet):
        """Import logical Prefill rows into Decode's dense cache and source owners."""
        validate_generation_state(packet)
        state = cls(packet["topology"]["ep"])
        state.histories = [list(history) for history in packet["histories"]]
        state.engram_history = [value.clone() for value in packet["engram_history"]]
        state.prefill_chunks = packet["prefill_chunks"]
        copied = 0

        def copy(destination, source):
            nonlocal copied
            destination.copy_(source)
            if not torch.equal(destination.view(torch.uint8), source.view(torch.uint8)):
                raise RuntimeError("State transfer changed cache bytes")
            copied += source.numel() * source.element_size()

        for rank, history in enumerate(state.histories):
            count = len(history)
            for layer, source in enumerate(packet["layers"][rank]):
                target = state.layers[layer]
                copy(target["cache"][rank].reshape(256, 512)[:count], source["swa"])
                for page in range(2):
                    target["cache_lens"][rank, page] = min(128, max(0, count - page * 128))
                if "selection" in source:
                    copy(target["prepared_indices"][rank, 0], source["selection"])
            for owner, rows in packet["sources"].items():
                source, target = rows[rank], state.sources[owner]
                visible = count // source["ratio"]
                offset = 64 if source["ratio"] == 2 else 0
                copy(target["compressed_cache"][rank].reshape(256, 512)[offset:offset + visible], source["kv"])
                copy(target["index_cache"][rank].reshape(256, 128)[:visible], source["codes"])
                copy(target["index_scale"][rank].reshape(256)[:visible], source["scales"])
                if source["ratio"] == 2:
                    copy(target["compressor_state"][rank, 0], source["ring"])
                else:
                    target["source_positions"][rank, 0] = count - 1
                    target["cache_lengths"][rank, 0].fill_(count)
            candidate = packet["candidates"][rank]
            copy(state.candidates["candidate_block_ids"][rank, 0, :candidate["count"]], candidate["ids"])
            state.candidates["candidate_block_counts"][rank, 0] = candidate["count"]
            state.candidates["candidate_topk_lengths"][rank, 0] = candidate["topk_length"]
        state.handoff = dict(status="imported", copied_cache_bytes=copied, copied_tensors_exact=True,
            next_positions=[len(history) for history in state.histories], prefill_chunks=state.prefill_chunks,
            swa_layers=40, source_layers=sorted(state.sources), compressor_residuals=[len(h) % 2 for h in state.histories])
        return state

    def validate_step(self, tokens):
        if self.failed:
            raise RuntimeError("A partially executed Decode state cannot be reused")
        if len(tokens) != self.world or any(type(token) is not int or not 0 <= token < FLASH.vocab_size
                                            for token in tokens):
            raise ValueError("Require one vocabulary token ID per logical rank")
        if any(len(history) >= MAX_CONTEXT for history in self.histories):
            raise ValueError("The bounded Decode cache holds at most 256 token positions per request")

    def layer_arguments(self, row):
        values = dict(self.layers[row["layer"]])
        if row["kv_source"] is not None:
            values.update(self.sources[row["kv_source"]])
        if row["ratio"] == 2 and row["mode"] == "reuse":
            values["prepared_indices_source"] = self.layers[row["index_source"]]["prepared_indices"]
        if row["ratio"] == 1:
            if row["mode"] == "reuse":
                # Reuse entries read the current source's prepared selection directly.
                values["prepared_indices"] = self.layers[row["index_source"]]["prepared_indices"]
            elif row["mode"] == "reindex":
                values.update({name: value.clone() for name, value in self.candidates.items()})
            else:
                values.update(self.candidates)
        return values


class DecodeNetwork:
    """Compile structural layer variants once and stream actual weights across all 40 layers."""

    def __init__(self, *, devices=None, sim=False, output=None, progress=None):
        self.explicit_devices = devices is not None
        self.devices = list(range(EP_SIZE)) if devices is None else list(devices)
        validate_devices(self.devices, world=EP_SIZE)
        self.sim, self.output, self.progress = sim, Path(output) if output else None, progress
        self.config = normalize_config(dict(backend_type="a2a3", execution_mode=sim,
            ring_heap=RING_HEAP,
            distributed_config=DistributedConfig(device_ids=self.devices, num_sub_workers=0)))
        self.functions = {entry._func.__name__: entry for entry in map(layer_entry, range(40))}
        self.io = dict(embedding=make_vocab_embedding(16, 5120, FLASH.vocab_size, 1),
                       initial_hc=make_initial_hc(16, 5120, 4), engram=make_decode_engram(), head=make_decode_head(),
                       rope=make_rope_tables(16, 64, FLASH.rope_theta,
                                            dict(factor=FLASH.rope_factor, beta_fast=FLASH.beta_fast,
                                                 beta_slow=FLASH.beta_slow,
                                                 original_max_position_embeddings=FLASH.original_max_position_embeddings)))
        self.compiled, self.compiled_io, self.scratch = {}, {}, {}
        self._io_connection, self._io_process = None, None

    def close(self):
        """Release the spawned single-device stage process after the Decode session."""
        if self._io_process is not None:
            try:
                if self._io_process.is_alive():
                    self._io_connection.send(None)
            except (BrokenPipeError, EOFError):
                pass
            self._io_process.join(timeout=30)
            if self._io_process.is_alive():
                self._io_process.terminate()
                self._io_process.join(timeout=5)
            self._io_connection.close()
            self._io_connection, self._io_process = None, None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _report(self, **record):
        if self.progress is not None:
            self.progress(record)

    def compile(self):
        """Compile all structural layer and input/output stages without allocating model weights."""
        for name, function in self.functions.items():
            self._report(stage="compile", entry=name)
            self.compiled[name] = function.compile(config=self.config)
        for name, function in self.io.items():
            self._report(stage="compile", entry=name)
            config = self._rank_config(0)
            self.compiled_io[name] = function.compile(config=config)
        return dict(topology=dict(dp=EP_SIZE, tp=1, ep=EP_SIZE), plan=network_plan(),
                    ring_heap=list(RING_HEAP),
                    entries={name: str(entry.output_dir) for name, entry in self.compiled.items()},
                    io={name: str(entry.output_dir) for name, entry in self.compiled_io.items()},
                    execution="not_run", accuracy="not_run")

    def _rank_config(self, rank):
        return normalize_config(dict(backend_type="a2a3", execution_mode=self.sim, device_id=self.devices[rank]))

    def _io(self, name, rank, arguments):
        if self._io_process is None:
            context = multiprocessing.get_context("spawn")
            self._io_connection, child = context.Pipe()
            self._io_process = context.Process(target=_serve_io, args=(child,))
            self._io_process.start()
            child.close()
        settings = dict(backend_type="a2a3", execution_mode=self.sim, device_id=self.devices[rank])
        self._io_connection.send((str(self.compiled_io[name].output_dir), arguments, settings))
        if not self._io_connection.poll(600):
            raise TimeoutError(f"Decode {name} stage did not finish on rank {rank}")
        error = self._io_connection.recv()
        if error is not None:
            raise RuntimeError(f"Decode {name} stage failed on rank {rank}:\n{error}")

    def _scratch(self, name, shape, dtype):
        key = name, shape, dtype
        if key not in self.scratch:
            self.scratch[key] = torch.zeros(shape, dtype=dtype)
        return self.scratch[key]

    def _metadata(self, tokens, state):
        values = dict(input_ids=torch.zeros((EP_SIZE, 16), dtype=torch.int64),
                      positions=torch.zeros((EP_SIZE, 16), dtype=torch.int64),
                      positions32=torch.zeros((EP_SIZE, 16), dtype=torch.int32),
                      slots=torch.full((EP_SIZE, 16, 2), -1, dtype=torch.int64),
                      indices=torch.full((EP_SIZE, 16, 128), -1, dtype=torch.int32),
                      requests=torch.zeros((EP_SIZE, 16), dtype=torch.int32),
                      query_start=torch.tensor([[0, 1]] * EP_SIZE, dtype=torch.int32),
                      state_block_table=torch.zeros((EP_SIZE, 1, 1), dtype=torch.int32),
                      rope_cos=torch.zeros((EP_SIZE, 16, 32), dtype=torch.float32),
                      rope_sin=torch.zeros((EP_SIZE, 16, 32), dtype=torch.float32))
        for rank, token in enumerate(tokens):
            position = len(state.histories[rank])
            values["input_ids"][rank, 0] = token
            values["positions"][rank, 0] = position
            values["positions32"][rank, 0] = position
            values["slots"][rank, 0] = torch.tensor([position // 128, position % 128])
            visible = list(range(max(0, position - 127), position + 1))
            values["indices"][rank, 0, :len(visible)] = torch.tensor(visible, dtype=torch.int32)
            self._io("rope", rank, [values["positions"][rank], values["rope_cos"][rank], values["rope_sin"][rank], 1])
        return values

    def step(self, tokens, *, weights, state):
        """Run one real token per rank through the entire backbone and return vocabulary logits."""
        tokens = list(tokens)
        if not self.explicit_devices and not self.sim:
            raise ValueError("Execution requires explicit allocated physical device IDs")
        if state.world != EP_SIZE or weights.world != EP_SIZE:
            raise ValueError("Weights, state and compiled network must use the same EP world")
        state.validate_step(tokens)
        if not self.compiled or not self.compiled_io:
            self.compile()
        try:
            return self._step(tokens, weights, state)
        except BaseException:
            state.failed = True
            raise

    def _step(self, tokens, weights, state):
        begin = perf_counter()
        metadata = self._metadata(tokens, state)
        timings = dict(metadata_and_rope_seconds=perf_counter() - begin, layers=[])
        stage_begin = perf_counter()
        hidden = torch.zeros((EP_SIZE, 16, 4, 5120), dtype=torch.bfloat16)
        premix = torch.zeros((EP_SIZE, 16, 4), dtype=torch.float32)
        embedding = torch.zeros((16, 5120), dtype=torch.bfloat16)
        live = torch.zeros(16, dtype=torch.int32)
        live[0] = 1
        embedding_weight = weights.embedding()
        timings["embedding_preparation_seconds"] = perf_counter() - stage_begin
        stage_begin = perf_counter()
        for rank in range(EP_SIZE):
            self._io("embedding", rank, [metadata["input_ids"][rank], live, embedding_weight, embedding, 0, 1])
            self._io("initial_hc", rank, [embedding, hidden[rank], premix[rank], 0, 1, 1])
        self.close()
        del embedding_weight
        timings["embedding_execution_seconds"] = perf_counter() - stage_begin
        stage_begin = perf_counter()
        engram_rows, engram_masks = weights.engram_rows(tokens, state.histories)
        timings["engram_host_lookup_seconds"] = perf_counter() - stage_begin
        backbone_begin = perf_counter()
        completed = []
        for row in network_plan():
            layer = row["layer"]
            layer_begin = perf_counter()
            self._report(stage="layer_start", epoch=state.epoch, layer=layer, mode=row["mode"])
            stage_begin = perf_counter()
            operands = weights.layer(layer)
            layer_timing = dict(layer=layer, mode=row["mode"], weight_load_seconds=perf_counter() - stage_begin)
            required = {
                "attn_hc_function", "attn_hc_scale", "attn_hc_base", "attn_norm_weight",
                "wq_a", "wq_a_scale", "q_norm_weight", "wq_b", "wq_b_scale", "wkv", "wkv_scale",
                "kv_norm_weight", "wo_a", "wo_b", "sink", "ffn_hc_function", "ffn_hc_scale",
                "ffn_hc_base", "ffn_norm_weight", "gate_weight", "text_bias", "vision_bias",
                "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale", "routed_w2",
                "routed_w2_scale", "shared_gate_up", "shared_gate_up_scale", "shared_down", "shared_down_scale",
            }
            if row["mode"] in ("full", "reindex"):
                required.update(("index_wq_b", "index_wq_b_scale", "index_weights_proj"))
            if row["mode"] == "full":
                required.update(("compressor_wkv", "compressor_norm", "index_wk", "index_norm"))
                if row["ratio"] == 2:
                    required.add("compressor_wgate")
            if row["engram"]:
                required.update(("engram_wkv", "engram_channel_weight", "engram_rotation"))
            if required - operands.keys():
                raise ValueError(f"L{layer} missing checkpoint operands: {sorted(required - operands.keys())}")
            stage_begin = perf_counter()
            if row["engram"]:
                result = torch.zeros_like(hidden)
                for rank in range(EP_SIZE):
                    self._io("engram", rank, [hidden[rank], engram_rows[layer][rank], operands["engram_wkv"],
                        operands["engram_channel_weight"], operands["engram_rotation"], engram_masks[rank], result[rank], 1])
                self.close()
                hidden = result
            layer_timing["engram_execution_seconds"] = perf_counter() - stage_begin if row["engram"] else 0.0
            for name in ("engram_wkv", "engram_channel_weight", "engram_rotation"):
                operands.pop(name, None)
            values = dict(metadata, **state.layer_arguments(row))
            values.update(operands)
            values.update(x_hc=hidden, attn_incoming=premix, active=1, epoch=state.epoch * 40 + layer + 1,
                          reuse=int(row["mode"] == "reuse"), has_engram=0, c1_mode=int(row["mode"] == "reindex"))
            if row["ratio"]:
                slots = torch.full((EP_SIZE, 16), -1, dtype=torch.int64)
                index_slots = torch.full_like(slots, -1)
                for rank, history in enumerate(state.histories):
                    position = len(history)
                    if row["ratio"] == 1:
                        slots[rank, 0] = position
                        index_slots[rank, 0] = position
                    elif position % 2:
                        slots[rank, 0] = 64 + position // 2
                        index_slots[rank, 0] = position // 2
                values.update(compressed_slots=slots, index_slots=index_slots)
            entry = layer_entry(layer)
            arguments = []
            for name, (shape, dtype) in entry_layout(entry).items():
                if name not in values:
                    values[name] = self._scratch(name, shape, dtype)
                value = values[name]
                if shape is None:
                    value = torch.as_tensor(value, dtype=dtype)
                if shape is not None and (tuple(value.shape) != shape or value.dtype != dtype):
                    raise ValueError(f"L{layer} {name}: expected {shape}/{dtype}, got {value.shape}/{value.dtype}")
                arguments.append(value)
            stage_begin = perf_counter()
            self.compiled[entry._func.__name__](*arguments, config=self.config)
            layer_timing["distributed_call_seconds"] = perf_counter() - stage_begin
            hidden, premix = values["output"].clone(), values["ffn_next_pre"].clone()
            if not torch.isfinite(hidden[:, :1]).all():
                raise RuntimeError(f"Non-finite hidden state after L{layer}")
            for rank, history in enumerate(state.histories):
                for page in range(2):
                    state.layers[layer]["cache_lens"][rank, page] = min(128, max(0, len(history) + 1 - page * 128))
            completed.append(layer)
            del values, arguments, operands
            layer_timing["total_seconds"] = perf_counter() - layer_begin
            measured = sum(layer_timing[name] for name in ("weight_load_seconds", "engram_execution_seconds", "distributed_call_seconds"))
            layer_timing["host_preparation_and_state_seconds"] = layer_timing["total_seconds"] - measured
            timings["layers"].append(layer_timing)
            self._report(stage="layer_done", epoch=state.epoch, **layer_timing)
        timings["backbone_seconds"] = perf_counter() - backbone_begin
        stage_begin = perf_counter()
        norm, head = weights.head()
        logits = torch.zeros((EP_SIZE, 16, FLASH.vocab_size), dtype=torch.bfloat16)
        timings["head_preparation_seconds"] = perf_counter() - stage_begin
        stage_begin = perf_counter()
        for rank in range(EP_SIZE):
            self._io("head", rank, [hidden[rank], premix[rank], norm, head, logits[rank], 1])
        self.close()
        timings["head_execution_seconds"] = perf_counter() - stage_begin
        stage_begin = perf_counter()
        if not torch.isfinite(logits[:, 0]).all():
            raise RuntimeError("Non-finite final logits")
        for rank, (history, token) in enumerate(zip(state.histories, tokens, strict=True)):
            history.append(token)
            if hasattr(state, "engram_history"):
                compressed = -1 if token in weights.image_ids else int(weights.token_map[token])
                state.engram_history[rank] = torch.cat((state.engram_history[rank], torch.tensor([compressed], dtype=torch.int32)))
        state.epoch += 1
        timings["validation_and_history_seconds"] = perf_counter() - stage_begin
        timings["total_seconds"] = perf_counter() - begin
        timings["scope"] = "Host wall time; distributed calls include worker setup, device build, transfers, communication and execution"
        timings["layer_weight_load_seconds"] = sum(row["weight_load_seconds"] for row in timings["layers"])
        timings["layer_distributed_call_seconds"] = sum(row["distributed_call_seconds"] for row in timings["layers"])
        timings["engram_execution_seconds"] = sum(row["engram_execution_seconds"] for row in timings["layers"])
        self.last_timings = timings
        report = dict(status="executed", layers=completed, epoch=state.epoch,
                      logits_shape=[EP_SIZE, FLASH.vocab_size], accuracy="not_run",
                      seconds=timings["total_seconds"], topology=dict(dp=EP_SIZE, tp=1, ep=EP_SIZE), timings=timings)
        self._report(stage="step_done", **report)
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=True)
            torch.save(dict(logits=logits[:, 0].clone(), hidden=hidden[:, :1].clone(), premix=premix[:, :1].clone()),
                       self.output / f"step-{state.epoch}.pt")
            (self.output / f"step-{state.epoch}.json").write_text(json.dumps(report, indent=2) + "\n")
        return logits[:, 0].clone()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("-d", "--devices", type=lambda value: [int(item) for item in value.split(",")])
    parser.add_argument("-p", "--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("--ep", type=int, choices=(2, 16), default=EP_SIZE)
    parser.add_argument("--tp", type=int, choices=(1,), default=1)
    parser.add_argument("--dp", type=int)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--tokens", default="0", help="Comma-separated prompt token IDs, replayed one token at a time")
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("build_output/ds41-decode-network"))
    args = parser.parse_args(argv)
    if args.ep != EP_SIZE or args.dp not in (None, EP_SIZE):
        parser.error("The imported Decode specialization requires DP=EP and TP=1")
    if args.plan_only:
        print(json.dumps(dict(topology=dict(dp=EP_SIZE, tp=1, ep=EP_SIZE), layers=network_plan()), indent=2))
        return 0
    if not args.compile_only and (args.checkpoint is None or args.devices is None):
        parser.error("Execution requires --checkpoint and explicit allocated --devices")
    tokens = [int(item) for item in args.tokens.split(",")]
    if (args.decode_steps < 1 or len(tokens) + args.decode_steps - 1 > MAX_CONTEXT
            or not tokens or any(not 0 <= token < FLASH.vocab_size for token in tokens)):
        parser.error("Require valid prompt IDs and a positive step count within the 256-position cache")
    network = DecodeNetwork(devices=args.devices, sim=args.sim, output=args.output,
                            progress=lambda record: print(json.dumps(record), flush=True))
    manifest = network.compile()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "compiled.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.compile_only:
        print("[decode-network] 40-layer schedule + embedding/Engram/LM head COMPILE PASS; execution=not_run")
        return 0
    weights = DecodeNetworkWeights(args.checkpoint, EP_SIZE)
    state = DecodeState()
    try:
        for token in tokens:
            logits = network.step([token] * EP_SIZE, weights=weights, state=state)
        generated = [logits.argmax(-1).tolist()]
        for _ in range(args.decode_steps - 1):
            logits = network.step(generated[-1], weights=weights, state=state)
            generated.append(logits.argmax(-1).tolist())
        summary = dict(status="executed", layers=40, input_steps=len(tokens), decode_steps=args.decode_steps,
                       generated_ids=generated, accuracy="not_run", topology=dict(dp=EP_SIZE, tp=1, ep=EP_SIZE))
        (args.output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("[decode-network] 40L+LMhead RUN PASS; accuracy=not_run; " + json.dumps(summary), flush=True)
    finally:
        weights.close()
        network.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

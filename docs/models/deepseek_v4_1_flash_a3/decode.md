# DeepSeek V4.1 Flash A3 Decode

Decode and Prefill now reside in `models.deepseek_v4_1_flash_a3`. The former
`models.deepseek_v4_1_flash_a2` package has been removed. Both source histories
remain in Git; this is a source-organization change rather than new device
acceptance or an automatic Prefill-to-Decode state handoff.

## Entries and operator ownership

| Responsibility | Maintained owner |
| --- | --- |
| 40-layer schedule, state lifetime and autoregressive loop | `decode_fwd.py --network`, `decode_network.py` |
| Streaming real checkpoint weights and sparse Engram rows | `decode_network_weights.py`, `decode_weight_abi.py`, `decode_weights.py` |
| Input embedding / HC and final vocabulary head | `embedding.py`, `decode_head.py` |
| Bounded single-step SWA Decode and state replay | `decode_fwd.py`, `decode_replay.py` |
| Physical-layer dispatch | `decode_layer.get_layer_entry(layer_id)` |
| Complete SWA / Engram / C2A / C1A blocks | `decode_swa.py`, `decode_engram.py`, `decode_c2a.py`, `decode_c1a.py` |
| EP2 / EP16 product orchestration with TP1 | `decode_moe.py` |
| Routing, dispatch layout and shared-expert variants | `moe.py` |
| Routed expert kernels | `expert_routed.py`, `decode_expert_routed.py` |
| Shared model, dtype, Engram and compressor contracts | `config.py`, `operator_config.py`, `engram.py`, `compressor_core.py` |
| Decode-specific numerical/cache boundaries | `decode_quant.py`, `decode_gemm.py`, `decode_hc.py`, `decode_swa_cache.py`, `decode_compressor_c1.py`, `decode_compressor_c2.py` |
| HOST contract and no-device dependency scan | `decode_acceptance.py`, `decode_asset_contract.py`, `decode_gate.py` |

The phase-specific implementations retain the exact existing quantization,
rounding, cache and topology contracts. Shared MoE kernels use explicit
`make_decode_*` names where their semantics differ from Prefill. Existing
A2/A3 dtype names, compiler hints and environment option names are ABI and
provenance, not a claim that the hardware is A2.

## 40-layer network

The network entry connects the existing complete Decode operators in physical
layer order, carrying each layer's hidden state and HC premix into its successor.
It runs checkpoint embedding, original-basis Engram at L1/L14, all 40 attention
and MoE blocks, final HC collapse, RMS normalization and the vocabulary projection.
The LM head returns `[EP, 129280]` logits for one active request per rank.

The initial execution topology is **DP16 / TP1 / EP16**, with EP2 also admitted
by the driver. Each rank owns `384 / EP` routed experts; attention and shared
weights are replicated. Pass `--ep` and `--tp` on the command line because the
existing distributed entries specialize their tensor shapes during import.
The DP8/TP2 and DP4/TP4 contracts remain outside this network driver.

From the repository root, after activating the PyPTO environment:

```bash
PYTHONPATH="$PWD:$PYTHONPATH" python -m models.deepseek_v4_1_flash_a3.decode_fwd \
  --network --ep 16 --tp 1 --plan-only
PYTHONPATH="$PWD:$PYTHONPATH" python -m models.deepseek_v4_1_flash_a3.decode_fwd \
  --network --ep 16 --tp 1 --compile-only \
  --output build_output/ds41-decode-network
PYTHONPATH="$PWD:$PYTHONPATH" python -m models.deepseek_v4_1_flash_a3.decode_fwd \
  --network --ep 16 --tp 1 --devices "$DEVICES" --checkpoint "$CHECKPOINT" \
  --tokens 0 --decode-steps 2 --output build_output/ds41-decode-device
```

Set `DEVICES` to the scheduler-allocated comma-separated physical IDs in logical
rank order and `CHECKPOINT` to the real W8A8 checkpoint directory. On systems
using `task-submit`, execute the last command inside the allocated job and use
`TASK_DEVICE` for that mapping. Preserve the activated `PYTHONPATH` so a managed
toolchain's runtime overlay remains visible.

`--tokens` accepts prompt IDs replayed one at a time from an empty Decode state;
the CLI sends the same prompt to each rank. `--decode-steps 2` produces two
greedy output IDs per rank, including a second full forward that consumes the
first generated ID. The Python `DecodeNetwork.step` interface accepts distinct
token IDs per rank and a retained `DecodeState` / `DecodeNetworkWeights` pair.

The state owner retains all layers' SWA buffers, C2 source state at L2/L8/L14,
C1 KV/index state at L20, reindex selections at L24/L28/L32/L36, the L20 candidate
pool and request token histories for Engram. Source/reuse consumers share the
correct physical owner. Cache capacity is **256 token positions per request**,
with a 128-token SWA window, fixed physical row capacity 16 and one active row.
An interrupted step invalidates its partially updated state.

This is a host-scheduled driver. It uploads and reads back state for each layer,
streams one physical layer's real weights and creates communication windows per
invocation. It does not yet provide generation-scoped device residency,
Prefill-state handoff, request slot reuse, or CP/SP and TP2/TP4 execution.
It uses PyPTO compute and its distributed put/notify/wait transport; it does not
add an HCCL runtime driver. Latency and throughput have not been tuned.

The EP16/TP1 structural templates and all input/output stages compile. Host
checkpoint shape/dtype binding checks cover SWA, C2, C1 source and C1 reindex.
A real-weight DP16/TP1/EP16 device run has completed all 40 layers, both Engram
stages and the final head for two consecutive tokens, producing finite
`[16, 129280]` logits at both steps. The second forward consumes the first
generated token with the retained session state. This run printed
`40L+LMhead RUN PASS` and exited successfully.
These execution checks do not establish numerical accuracy.
The run writes `compiled.json`, per-step logits/hidden tensors and progress
records; `result.json` and `40L+LMhead RUN PASS` appear only after every requested
forward reaches the final head with finite active outputs. Accuracy remains
`not_run`: this entry does not invoke vLLM or numerical reference comparisons.

## Acceptance and source provenance

This directory is the independent A3 Decode session. No historical Prefill,
CPU, simulator, or private capture result is carried into its status. The
formal status for layers **0, 1, 2, 3, 14, 15, 19, 20, 21, 24 and 25** starts at
`not_run` and can only change after a same-input NPU PyPTO versus vLLM-Ascend
reference comparison at the requested topology. The floating policy is
`atol=0.05, rtol=0.05`; routes, lengths, masks, source IDs, readonly inputs,
inactive rows and untouched cache regions are exact checks.

`decode_source_manifest.json` maps each ported file to its original source digest.
It also records unchanged definition AST digests, the native-call identities,
and host changes needed for relocation. Its `decode_files` records identify the
new PyPTO Decode entry points and explicitly keep their accuracy evidence
separate from source migration. Dependencies extracted from the original
parent model retain their compute definitions in this flat directory.
The port includes the L1/L14 Engram observation interfaces, C2A source/reuse
cache publication, and the C1A source/reindex/reuse candidate state boundary.

The local tree contains bounded PyPTO L0 entries, complete layer composition
entries for every physical layer, and the independently compiled routed/shared
MoE entry. The retained EP2 acceptance matrix covers
L1/L2/L3/L14/L15/L19/L20/L21/L24/L25. The current paired
representative replays pass for L1, L2, L3, L14, L19, L20, L21, L24 and L25.
L15 reaches the complete replay but remains a precision failure because its
observable `moe_output` exposes a PyPTO routed/shared split while the vLLM
capture exposes a fused boundary. Composition and compile coverage do not
promote a layer without a same-request replay.
The independent vLLM-Ascend capture is described by
[`decode_reference.py`](../../../models/deepseek_v4_1_flash_a3/decode_reference.py)
and is not imported by the product path.

Callers must supply the actual independent request, checkpoint, history cache
and reference capture required by `decode_asset_contract.py`. The package
contains no private capture runner or device queue. Metadata, compile-only
results and the representative historical reports below do not complete the
formal acceptance matrix.

Generic `layer_capture.npz` and `moe_capture.npz` files do not satisfy a
non-SWA layer plan. C2A/C1A captures must also declare complete semantic
`compressed_kv_cache`, `index_cache`, `index_scale` and (for C1A)
`candidate_pool` state in `reference_capture.json`; L1/L14 must mark the
captured Engram rows.

## Independent Decode acceptance

The Decode session starts with every formal gate at `not_run`. Search K=2 is
registered in [`decode_candidates.py`](../../../models/deepseek_v4_1_flash_a3/decode_candidates.py);
both candidates use PyPTO entry points and share the same page size, scale
contract, input boundary, topology and tolerance. The acceptance contract and
product dependency scan are executable without importing a model kernel:
The four-commit ancestry and tree IDs are recorded in
[`decode_cherry_pick_receipt.json`](../../../models/deepseek_v4_1_flash_a3/decode_cherry_pick_receipt.json).
The generated `git cherry-pick -x` proof is kept in a temporary clean clone;
the product checkout still retains the requested source commit IDs unchanged.
[`decode_shape_contract.py`](../../../models/deepseek_v4_1_flash_a3/decode_shape_contract.py)
records the global dimensions, the TP1/TP2/TP4 matrix, EP2 expert ownership,
the replicated single KV head, page layout, and dispatch/combine contract.
`decode_config.py` records the constructed-runtime fields for sequence
parallel MoE, cache block size, request scheduling and communication groups;
unresolved fields stay `not_run`. Capacity and per-rank peak memory are also
left `not_run` until real checkpoint buffers are allocated.

```bash
PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a3.decode_acceptance --json
PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a3.decode_acceptance \
  --record .pypto-a3-state/decode-acceptance.json
PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a3.decode_gate \
  --topology ep2_gate --record .pypto-a3-state/decode-ep2-gate.json
PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a3.decode_gate \
  --topology main --record .pypto-a3-state/decode-main-gate.json
```

The scan rejects vLLM, vLLM-Ascend and Torch-NPU imports in the product import
closure. Files ending in `_runtime.py` are reference-only and cannot be loaded
by a product entry. The scan records Torch imports used by the golden/reference
side separately; they do not establish product accuracy.

`decode_gate.py` is intentionally fail-closed. It emits the required
`task-submit --device auto --device-num 2/16` command and records missing
checkpoint, cache, input, or reference assets; it never launches vLLM or
claims a gate without a same-input capture. `decode_reference.py` records the
separate reference process's Python path, source digest, mapped libraries,
visible devices and runtime library path; it never imports vLLM in the product
process.

The bounded entries accept the full state ABI, while the independent
`decode_moe.py` product entry owns the PyPTO EP2 dispatch/combine,
192-expert-per-rank routed shard, and shared-expert merge. The real two-card
MoE boundary has executed on A3 with the checkpoint and the vLLM-Ascend
capture input; `logits`, `route_ids`, `route_weights`, and the merged MoE
output pass the `rtol=0.05, atol=0.05` gate. The evidence is recorded in
`.pypto-a3-state/product/decode-ep2-l0-v25e.json`.

The same PyPTO MoE boundary replay has also passed for L1, L2, L3, L14, L15,
L19, L20, L21, L24 and L25 using their layer-specific vLLM-Ascend captures.
Those reports compare logits, exact top-six route IDs, route weights and the
merged routed/shared output. They are component evidence only. The complete
physical-layer replays currently pass for L1, L2, L3, L14, L19, L20, L21, L24
and L25. L15 has a complete same-request replay with attention, cache/index,
HC, routing and final output passing; its observable `moe_output` remains
outside the fixed gate (`max_abs=0.132812`, `mean_abs=0.0017201`) because the
reference capture fuses routed and shared expert results before exposing that
boundary.

The complete L0 composition has also executed on the same two-card A3
topology with mHC attention, SWA cache read/write, FFN residual, routed/shared
MoE, and output restoration. The representative result compares the layer
output, router input, logits, route IDs and weights, MoE output, FFN state,
and cache pages against the reference at `rtol=0.05, atol=0.05`; see
`.pypto-a3-state/product/decode-layer-ep2-l0-v29.json`. The matching joint
vLLM-Ascend contract is
`.pypto-a3-state/reference/decode_ep2_l0_20261106_v29/reference_capture.json`.

The formal EP2 status remains `not_run` until the required no-local-input
collective, finished/inactive rows, unequal-length batch, slot reuse, and
additional request-state cases are covered. The new C1A/C2A and Engram
compositions still require same-request paired captures and NPU replay before
they can change the layer precision status; CP/SP and the requested 16-device
gates remain separate blockers.

## Checks without a device

Compile the bounded Decode ABI without allocating a device:

```bash
PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a3.decode_fwd -p a2a3 --sim --compile-only
PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a3.decode_moe_compile
PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a3.decode_layer_l0_compile
```

Compile the deterministic single-context state replay for each supported step
count. These commands exercise PyPTO code generation only; they are not NPU
checkpoint accuracy evidence. The EP2 compiler entry reports physical
FRACTAL_NZ lowering separately from execution.

```bash
PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a3.decode_replay --steps 1 -p a2a3 --sim
PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a3.decode_replay --steps 2 -p a2a3 --sim
PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a3.decode_replay --steps 8 -p a2a3 --sim
```

The external A5-shaped validation tree is not part of this Decode session and
must not be used as an acceptance shortcut. Actual operator validation should
use the existing [Golden Harness](../../run-and-validate/golden-harness.md)
only to compile the PyPTO entry, then compare NPU outputs and state against the
independent reference manifest under the same checkpoint, layer, cache and
topology. Native boundary reports verify identity and interface integrity;
they do not replace numerical comparison against an independent reference.

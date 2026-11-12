# DeepSeek V4.1 Flash A3 Decode

Decode and Prefill now reside in `models.deepseek_v4_1_flash_a3`. The former
`models.deepseek_v4_1_flash_a2` package has been removed. Both source histories
remain in Git; this is a source-organization change rather than new device
acceptance or an automatic Prefill-to-Decode state handoff.

## Entries and operator ownership

| Responsibility | Maintained owner |
| --- | --- |
| Bounded single-step SWA Decode and state replay | `decode_fwd.py`, `decode_replay.py` |
| Physical-layer dispatch | `decode_layer.get_layer_entry(layer_id)` |
| Complete SWA / Engram / C2A / C1A blocks | `decode_swa.py`, `decode_engram.py`, `decode_c2a.py`, `decode_c1a.py` |
| EP2 product orchestration | `decode_moe.py` |
| Routing, dispatch layout and shared-expert variants | `moe.py` |
| Routed expert kernels | `expert_routed.py`, `decode_expert_routed.py` |
| Shared model, dtype, Engram and compressor contracts | `config.py`, `operator_config.py`, `engram.py`, `compressor_core.py` |
| Decode-specific numerical/cache boundaries | `decode_quant.py`, `decode_gemm.py`, `decode_hc.py`, `decode_swa_cache.py`, `decode_compressor_c1.py`, `decode_compressor_c2.py` |
| HOST contract and no-device dependency scan | `decode_acceptance.py`, `decode_asset_contract.py`, `decode_gate.py` |

The phase-specific implementations retain the exact existing quantization,
rounding, cache and topology contracts. Shared MoE kernels use explicit
`make_decode_*` names where their semantics differ from Prefill. Existing
A2/A3 dtype names, compiler hints and environment option names are ABI and
provenance, not a claim that the hardware is A2. The bounded Decode forward
is not a full 40-layer Decode driver.

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
entries for the retained L1/L2/L3/L14/L15/L19/L20/L21/L24/L25 EP2 matrix, and
the independently compiled EP2 routed/shared MoE entry. The current paired
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

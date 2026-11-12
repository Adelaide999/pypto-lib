# DeepSeek V4.1 Flash A3 Decode

Decode and Prefill reside in `models.deepseek_v4_1_flash_a3`. The old fixed
EP2 product path, two-device compile drivers and complete-layer wrappers
have been removed. Their source and historical results remain in Git at
`2704263`; they are not current entry points or EP16 acceptance evidence.

## Entries and operator ownership

| Responsibility | Maintained owner |
| --- | --- |
| Bounded single-step SWA Decode and state replay | `decode_fwd.py`, `decode_replay.py` |
| Local attention and mHC composition | `decode_swa.py` |
| Routing, dispatch primitives and shared-expert variants | `moe.py` |
| Routed expert kernels | `expert_routed.py` |
| Compression, Indexer and Engram computation | `compressor.py`, `indexer.py`, `engram.py` |
| Quantization, linear, normalization, RoPE and cache computation | `quant.py`, `gemm.py`, `rmsnorm.py`, `rope.py`, `qkv_proj_rope.py`, `o_proj.py`, `swa.py` |
| Checkpoint and EP16 MoE weight layouts | `weights.py` |
| HOST contracts and dependency scan | `decode_acceptance.py`, `decode_contract.py`, `decode_gate.py` |

The retained numerical kernels preserve their arithmetic, rounding and cache
semantics. The bounded SWA forward contains attention computation; it is not
an EP16 MoE, a complete layer or a 40-layer Decode network. `decode_swa.py`
composes local attention with mHC and does not import distributed MoE.

## EP16 contract and current limits

The declared topology matrix is DP16/TP1/EP16, DP8/TP2/EP16 and
DP4/TP4/EP16. A Decode configuration with EP2 is rejected. Declaring one of
these EP16 configurations does not shard the bounded SWA kernel or provide
a distributed network driver.

`weights.load_decode_rank_weights(checkpoint, layer, rank)` loads the router,
shared expert and one contiguous 24-expert MoE shard for an EP16 owner.
Ranks 0 through 15 cover all 384 experts exactly once. This preserves the
existing per-expert packing, scale handling and offset checks; it does not
bind TP-sharded attention weights, source/cache lifetimes or a network plan.
`weights.abi_summary()` describes that host layout without loading weights.

The following requirements remain unimplemented or unverified:

- Complete-network EP16 Decode dispatch, communication and output reduction.
- Per-layer TP attention, cache ownership and Prefill-to-Decode handoff.
- DecodeState binding and request/position/finished/epoch transitions.
- Same-input NPU versus independent vLLM-Ascend reference captures.
- Device memory peaks, resident model coverage and performance.

`decode_gate.py` reports `ready_for_submission=false`, no product command and
no formal network entry point while the EP16 driver is unavailable. It does
not substitute a bounded operator run for the requested topology.

## Source and acceptance evidence

`decode_source_manifest.json` and `source_manifest.json` retain original
export identities, including retired files. `package_refactor_map.json`
records retired definitions separately from current owners. The original
four-commit history is recorded in `decode_cherry_pick_receipt.json`.

Historical two-card layer and MoE replays remain evidence for their exact
old source, inputs and topology. They do not establish current EP16
correctness. Every current formal layer and topology gate remains `not_run`.
Floating comparisons use `rtol=0.05, atol=0.05`; discrete control, indices,
readonly state, inactive rows and untouched cache regions remain exact.

`decode_contract.py` still requires identical checkpoint, request, history
cache and independent reference captures. C2A/C1A reference assets must
include compressed KV, index and candidate state, and Engram layers must
identify the captured rows. A contract scan or a CPU golden is not an NPU
acceptance result.

## Checks

The following HOST reports initialize no devices:

```bash
python -m models.deepseek_v4_1_flash_a3.decode_acceptance --json
python -m models.deepseek_v4_1_flash_a3.decode_gate --topology main --json
```

The dependency scan rejects vLLM, vLLM-Ascend and Torch-NPU imports in the
product closure. Independent reference capture remains in
`decode_reference.py`; it is not part of product computation.

The bounded SWA entry can be compiled with the existing
[Golden Harness](../../run-and-validate/golden-harness.md):

```bash
python -m models.deepseek_v4_1_flash_a3.decode_fwd -p a2a3 --sim --compile-only
python -m models.deepseek_v4_1_flash_a3.decode_replay --steps 1 -p a2a3 --sim --compile-only
```

These are bounded operator checks, not EP16 or full-network acceptance.

# DeepSeek V4.1 Flash A3 Decode

Decode and Prefill share the A3 package and small computation primitives.
The bounded SWA Decode entry is implemented; a complete 40-layer EP16 Decode
network is not implemented. Old EP2 drivers, state replay, task acceptance
gates and independent reference-capture scripts are removed.

## Owners

| Responsibility | Module |
| --- | --- |
| Bounded single-step SWA forward and local golden | `decode_fwd.py` |
| Local attention and mHC composition | `decode_swa.py` |
| Routing, dispatch and shared-expert variants | `decode_moe.py` |
| Routed expert computation | `expert_routed.py` |
| Compression, Indexer and Engram | `compressor.py`, `indexer.py`, `engram.py` |
| Linear, quantization, Norm, RoPE and cache operations | Shared child operators |
| Checkpoint loading and EP16 MoE layout | `weights.py` |

The declared topology matrix is DP16/TP1/EP16, DP8/TP2/EP16 and
DP4/TP4/EP16. EP2 is rejected. These declarations do not turn the local
SWA step into a distributed MoE or a complete network.

`weights.load_decode_rank_weights(checkpoint, layer, rank)` loads router,
shared-expert operands and a contiguous 24-expert shard. Ranks 0..15 cover
all 384 experts once. Per-expert packing, scale handling and offset checks
are preserved. This loader does not bind TP attention weights or source/cache
ownership into a network plan.

The bounded entry retains its own Torch golden and standard
[Golden Harness](../../run-and-validate/golden-harness.md) command:

```bash
python -m models.deepseek_v4_1_flash_a3.decode_fwd -p a2a3 --sim --compile-only
```

This compiles an operator case; it is not a full-network accuracy result.
`decode_moe.py` additionally owns bounded router/shared-expert validation:

```bash
python -m models.deepseek_v4_1_flash_a3.decode_moe -p a2a3 --compile-only
```

Its local Torch goldens retain the FP32 router input and both standalone BF16
and fused FP32 activation boundaries, including empty and partial token slabs.
Remaining work includes the EP16 driver and collectives, per-layer weight and
cache binding, request/position/finished/epoch transitions, Prefill state
handoff, and device accuracy/memory/performance verification. Continuous
generation currently uses the complete Prefill feedback path described in
the [model documentation](index.md).

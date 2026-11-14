# DeepSeek V4.1 Flash A3 Decode

Decode and Prefill share the A3 package and small computation primitives.
The bounded SWA entry and a host-scheduled 40-layer EP16 Decode network are
implemented. The network composes the existing physical-layer entries, keeps
Decode weights resident across steps, and exposes embedding, Engram and the
vocabulary head. Device accuracy and automatic Prefill-to-Decode generation
remain separate validation work.

## Owners

| Responsibility | Module |
| --- | --- |
| 40-layer schedule, state lifetime and autoregressive loop | `decode_fwd.py --network`, `decode_network.py` |
| Resident checkpoint weights and sparse Engram rows | `decode_network_weights.py`, `decode_network_decode_weight_abi.py`, `decode_network_decode_weights.py` |
| Bounded single-step SWA forward and local golden | `decode_fwd.py` |
| Bounded attention and mHC composition | `decode_swa.py` |
| Full Decode SWA and Engram composition | `decode_network_swa.py`, `decode_network_engram.py` |
| Routing, dispatch and shared-expert variants | `decode_moe.py` |
| Routed expert computation | `expert_routed.py` |
| Compression, Indexer and Engram | `compressor.py`, `indexer.py`, `engram.py` |
| Linear, quantization, Norm, RoPE and cache operations | Shared child operators |
| Checkpoint loading and EP16 MoE layout | `weights.py` |

The complete network currently specializes to DP16/TP1/EP16. Its Decode MoE
loader assigns 24 routed experts to each rank; the bounded single-step entry
retains its existing topology matrix and validation contract.

`decode_network_weights.DecodeNetworkWeights` loads one physical layer at a
time, adapts the target Prefill-packed QKV/Indexer matrices back to the
canonical Decode layout, and retains the complete bank on the device. Routed
experts are streamed by owner rank, so ranks 0..15 cover all 384 experts once.
Per-expert NZ packing, scale handling and offset checks are preserved.

## 40-layer network

The network runs embedding, the L1/L14 Engram stages, all forty physical
layers, and the final vocabulary head. `DecodeState` owns SWA pages, compressed
source state, Indexer selections, the candidate pool and request histories.
The public `DecodeNetwork` and `DecodeNetworkWeights` objects can be retained
for multiple Decode steps.

After activating the PyPTO environment, the following checks do not allocate
or execute an NPU device:

```bash
python -m models.deepseek_v4_1_flash_a3.decode_fwd --network --ep 16 --dp 16 --tp 1 --plan-only
python -m models.deepseek_v4_1_flash_a3.decode_fwd --network --ep 16 --dp 16 --tp 1 --compile-only
```

The plan reports all 40 layers and the compile-only command compiles the four
layer variants plus embedding, initial HC, Engram, RoPE and the vocabulary
head. It does not establish numerical accuracy or device performance.

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
Remaining work includes Prefill state handoff, request slot reuse and device
accuracy/memory/performance verification. Continuous generation still uses
the existing Prefill feedback path described in the [model documentation](index.md).

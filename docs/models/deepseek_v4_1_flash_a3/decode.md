# DeepSeek V4.1 Flash A3 Decode

The 40-layer network uses DP4/TP4/EP16 and one persistent generation session.
Prefill and one-token Decode use the same TP4 layer ABI, compiled program
registry, static weight handles, HCCL groups and paged cache allocations.
The scheduler remains on the host; model arithmetic runs on A3.

## Owners

| Responsibility | Module |
| --- | --- |
| Shared programs, layer schedule and generation session | `network.py` |
| Prefill request preparation and public forward | `prefill_fwd.py` |
| One-token Decode and standalone Decode command | `decode_network.py` |
| Session static weight ownership and read/upload audit | `resident_weights.py` |
| Decode binding to existing canonical operands | `resident_weights.py` (`DecodeNetworkWeights`) |
| Checkpoint reading, packing and expert ownership | `weights.py` |
| Original paged cache geometry and aliases | `cache.py`, `state.py` |
| Persistent workers, dynamic buffers and HCCL channels | `prefill_resident.py` |
| Request metadata, cache reservation and host lookup lifetime | `prefill_prepare.py` |
| Device Engram projection and gating | `engram.py`, `prefill_network_engram.py` |
| Device final collapse, normalization and vocabulary shards | `lm_head.py` |
| Full-width Decode layer dispatch and SWA/Engram block composition | `decode_layer.py` |
| SWA and compressed-attention compositions | `decode_swa.py`, `decode_c1a.py`, `decode_c2a.py` |
| Full-width EP dispatch, experts and rank composition | `decode_moe_ep.py` |

Decode uses the same operator owners as Prefill: `gemm.py`, `quant.py`,
`rmsnorm.py`, `rope.py`, `qkv_proj_rope.py`, `o_proj.py`, `hc.py`, `indexer.py`
and `compressor.py`. Distinct Decode rounding and weight-layout variants keep
separate function names in those owners. The `_kn` suffix identifies entries
that consume unpacked `[K, N]` INT8 weights; existing entries for blocked
weights retain their interfaces. `decode_compressed_attention.py` retains the
full-width compressed-attention specialization and its original configuration
checks. `decode_config.py` and `decode_layer_config.py` retain their separate
topology defaults and validation rules.

`A3ResidentWeights.preload_once()` prepares the complete static bank before
executing any tokens. Each logical checkpoint tensor is read once, including
shared experts. Each checkpoint shard is opened once during preload and its
initialization handle is closed before token execution. Every EP rank owns 24 contiguous routed experts. Identical
Prefill/Decode representations use the same pointer and owning buffer;
Decode's ABI adapter cannot create a reader or preload another bank.

The network retains the original blocked W8A8 operands for both phases. The
legacy full-width TP1 network and its independent FRACTAL_NZ weight bank are
replaced by this shared ABI. Bounded operator validation entries retain their
separate operator interfaces.

## Generation

```bash
python -m models.deepseek_v4_1_flash_a3.generate \
  --checkpoint weights/deepseek-v4.1-flash-w8a8 \
  --prompt "Hello" --max-new-tokens 3 --ignore-eos \
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --output build_output/ds41-generation
```

Use the site's allocator and its assigned device list for execution. The
first predicted token comes from Prefill; a three-token limit exercises two
Decode steps. Prompt chunks honor `--chunk-size`. Cache pages are reserved
for the prompt and all requested output tokens before initialization.
History is preserved in full up to the 2048-position context bound.

Decode updates token IDs, positions, masks and lookup rows through reusable
dynamic device buffers. It advances the original cache provider without a
state file, cache copy, new worker or second cache pool. Cache aliases keep
the original buffer, pointer and byte offset. A failed step invalidates the
session and cannot be replayed in place.

Engram projection, rotation and gating run on A3 in both phases. As in
Prefill, the complete INT8/FP32 lookup tables remain mapped on the host;
hashing and gathering the small dynamic lookup rows remain CPU input
preparation. This mode does not claim that the complete tables reside in HBM.
The LM head, normalization and vocabulary shard assembly run on A3.

`generation/result.json` records both runtime identities, checkpoint and
static upload counts, per-step zero-load checks, expert ownership, cache
pointer aliases and cleanup status. `prefill/network/weight-audit.json`
records per-logical-weight reads and host preparation groups. Each Decode
step writes `decode/step-N/decode.json` and the full execution receipt.
`generation/answer.txt` and `generation/tokens.json` are updated after each
generated token. Execution success does not establish independent accuracy.

## Standalone Decode

Standalone Decode initializes the same shared ABI once and consumes the
supplied history on the device before autoregressive steps. A history string
alone does not fabricate a populated cache.

```bash
python -m models.deepseek_v4_1_flash_a3.decode_fwd --network \
  --dp 4 --tp 4 --ep 16 --plan-only
python -m models.deepseek_v4_1_flash_a3.decode_network \
  --checkpoint weights/deepseek-v4.1-flash-w8a8 \
  --dp 4 --tp 4 --ep 16 --tokens 1 --decode-steps 3 --compile-only \
  --output build_output/ds41-decode-compiled
```

Plan-only reads no weights and initializes no device. Compile-only requires
checkpoint metadata to derive the shared cache and Engram contracts; it
compiles the complete network and its device boundaries without executing
kernels. Omit `--compile-only` and supply allocated `--devices` to execute.

The bounded SWA and MoE operator commands retain their Torch goldens and the
[Golden Harness](../../run-and-validate/golden-harness.md). Complete-network
accuracy and throughput require separate device comparison and measurement.

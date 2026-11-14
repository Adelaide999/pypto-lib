# A3 module ownership

The organization follows the shared
[DeepSeek V4.1 Flash model](https://github.com/hw-native-sys/pypto-lib/tree/e1c3522fa2670e13efaab80def5c04e7cfab024a/models/deepseek_v4_1_flash):
forward entries compose attention and MoE, while independent child files own
the computation and corresponding Torch goldens. A3 retains Eco-Tech W8A8
arithmetic and layouts; it does not substitute the reference model's
MXFP8/MXFP4 kernels or topology.

## Maintained owners

| Responsibility | Modules |
| --- | --- |
| Shared TP4 programs, network execution and generation session | `network.py` |
| Whole-network Prefill preparation | `prefill_fwd.py` |
| One-token Decode and token feedback | `decode_network.py`, `generate.py` |
| Session weight bank and Decode ABI binding | `resident_weights.py`, `decode_network_weights.py` |
| Generic layer and L2 composition | `prefill_layer.py`, `prefill_layer_plan.json` |
| Bounded Decode forward | `decode_fwd.py` |
| Phase attention compositions | `prefill_swa.py`, `prefill_c2a.py`, `prefill_compressed_reuse.py`, `decode_swa.py`, `decode_network_swa.py` |
| Vocabulary embedding and final head | `input_pack.py`, `lm_head.py` |
| Compression projection, pooling and publication | `compressor.py` |
| Index cache, query projection, selection and candidate pooling | `indexer.py`, `prefill_indexer.py` |
| Engram device computation | `engram.py`, `decode_network_engram.py` |
| Host Engram tables, rows, rotation and slot ownership | `engram_host.py` |
| Prefill routing, dispatch, shared expert and combination | `prefill_moe.py` |
| Bounded Decode routing, dispatch and shared expert | `decode_moe.py` |
| Bounded full-width Decode routing and layer composition | `decode_network_moe.py`, `decode_network_decode_*.py` |
| Routed expert computation | `expert_routed.py` |
| Linear, quantization, Norm, RoPE, QKV and attention leaves | `gemm.py`, `quant.py`, `rmsnorm.py`, `rope.py`, `qkv_proj_rope.py`, `o_proj.py`, `swa.py`, `sparse_attention.py` |
| Checkpoint readers and phase weight layouts | `weights.py` |
| Cache and request lifetime | `cache.py`, `state.py` |
| Persistent device workers and HCCL | `prefill_resident.py`, `moe_transport.py` |
| Weight packing, integer window scheduling and host file utilities | `utils.py` |

The full Decode network follows the same flat package layout as Prefill. Its
network-only leaves and orchestration helpers are top-level `decode_` modules;
the existing bounded Decode and Prefill owners remain available to their
original callers. Standalone audits, independent-layer runners,
native reference-capture workers, state replay and identity manifests are
removed. Historical code and provenance remain in Git history. Production
request/cache checks and operator-local golden entries remain.

Network request preparation lives in `prefill_prepare.py`; shared program
and generation lifetime live in `network.py`.
Indexer dispatch lives in `prefill_indexer.py`; live candidate ownership checks
live in `cache.py`. Decode projection definitions live in their Linear, Norm,
RoPE, activation, Indexer, attention or output-projection owner. There is no
mixed `decode_projection.py` module or forwarding facade. Integer SWA scheduling
and file identities use `utils.py`, which contains no PyPTO computation.

Unused native SWA/QLI/SMLA adapters and their SMLA scheduling code are removed.
The two small FP32/BF16 sparse staging kernels remain in `sparse_attention.py`;
their casts and cache-publication boundaries are unchanged.

## Numerical variants

Identical small definitions are shared. Phase-specific implementations keep
separate names within their computation owner when numerical semantics differ:

- `quant.py` preserves activation quantization, dequantization, scale encoding
  and the Decode CANN FP16 intermediate option.
- `rmsnorm.py` preserves normalization and FP32 router boundaries.
- `compressor.py` and `indexer.py` retain norm, RoPE, candidate bounds and
  cache publication semantics.
- `prefill_moe.py` and `decode_moe.py` own their phase kernels, Torch goldens
  and validation entries. Prefill retains controlled routing and FP16/FP32
  fused-channel options; Decode retains its authoritative FP32 router input
  and its separate activation/quantization boundaries. Neither imports the other.
  Both use independent `gemm.py` and `quant.py` primitives.

`expert_routed.py` owns grouped expert computation, and `moe_transport.py`
owns CANN/HCCL resources. The unused native dispatch adapter is removed;
there is no compatibility `moe.py` facade.

L2 regions inline existing children between host communication boundaries.
They do not change operation order, BF16 buffers or source/cache ownership.
`decode_sparse_attention.py` stays TP-specialized and separate from Prefill
configuration. EP2 product orchestration is removed. The generation weight owner assigns 24 experts per EP rank.
`decode_network.py` advances the shared TP4 schedule by one token over the
original Prefill cache aliases. Both phases execute device embedding, Engram
and the vocabulary head; independent device accuracy remains separate validation work.

Compute weights, workers and communication groups persist across Prefill
chunks and one-token Decode steps. Engram tables stay on the host; device kernels consume prepared rows.
Runtime shape, dtype, alias, version and lifecycle checks are production
interface requirements and remain in place.

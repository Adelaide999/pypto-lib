# A3 module ownership

## Default distributed forward

| Responsibility | Owner |
| --- | --- |
| Prefill rank loop, layer schedule, six attention modes and round-based MoE | `prefill_fwd.py` |
| Decode rank loop, encoder and decoder layer schedules | `decode_fwd.py` |
| Phase-specific attention and indexing | `operators/prefill_*.py`, `operators/decode_*.py` |
| Routing, expert dispatch and combine, point-to-point signaling | `operators/moe.py`, `operators/expert_routed.py`, `operators/expert_shared.py` |
| Topology, phase geometry, checkpoint loading and resident weight layouts | `operators/config.py`, `operators/weights.py` |
| Prepared tensor inputs, cache geometry and Decode aliases | `host.py`, `operators/utils.py` |
| Compile, verify manifests and select execution | `driver.py`, `generate.py` |
| Weight and cache lifetime, request submission and sampling | `generation.py`, `replay.py` |
| Layer and logit comparison, routing reports and profiling | `validation.py`, `reference.py`, `telemetry.py`, `swimlane.py` |

The default forward imports the operators under `operators/`. The following
tables describe the separate host-scheduled implementation selected by
`--staged-network` and its standalone cases.

## Staged session and bounded operators

| Responsibility | Modules |
| --- | --- |
| Staged TP4 programs, layer execution and session | `network.py` |
| Whole-network Prefill preparation | `staged_prefill.py` |
| One-token Decode and token feedback | `decode_network.py`, `generate.py` |
| Session weight bank and Decode ABI binding | `resident_weights.py` |
| Generic layer and L2 composition | `prefill_layer.py`, `prefill_layer_plan.json` |
| Bounded SWA validation | `bounded_decode.py` |
| Phase attention compositions | `prefill_swa.py`, `prefill_c2a.py`, `prefill_compressed_reuse.py`, `decode_swa.py`, `decode_c1a.py`, `decode_c2a.py` |
| Vocabulary embedding and final head | `input_pack.py`, `lm_head.py` |
| Compression projection, pooling and publication | `compressor.py` |
| Index cache, query projection, selection and candidate pooling | `indexer.py`, `prefill_indexer.py` |
| Engram device computation | `engram.py` |
| Host Engram tables, rows, rotation and slot ownership | `engram_host.py` |
| Prefill routing, dispatch, shared expert and combination | `prefill_moe.py` |
| Bounded Decode routing, dispatch and shared expert | `decode_moe.py` |
| Full-width Decode routing and layer composition | `decode_moe_ep.py`, `decode_layer.py` |
| Decode compressed-attention specialization | `decode_compressed_attention.py` |
| Resident blocked-INT8 routed experts | `grouped_moe.py` |
| Bounded Decode NZ routed experts | `expert_routed.py` |
| Linear, quantization, Norm, RoPE, QKV and attention leaves | `gemm.py`, `quant.py`, `rmsnorm.py`, `rope.py`, `qkv_proj_rope.py`, `o_proj.py`, `swa.py`, `sparse_attention.py` |
| Checkpoint readers and phase weight layouts | `weights.py` |
| Cache and request lifetime | `cache.py`, `state.py` |
| Persistent device workers and HCCL | `prefill_resident.py`, `moe_transport.py` |
| Weight packing, integer window scheduling and host file utilities | `utils.py` |

The full Decode network follows the same flat package layout as Prefill.
Attention variants compose shared operator owners; `decode_layer.py` selects
the layer entry and combines attention, MoE and residual streams.
Standalone audits, independent-layer runners,
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

## Layers and relocated modules

| Layer | Owners |
| --- | --- |
| Model configuration | `config.py` |
| Primitive computation | `gemm.py`, `quant.py`, `rmsnorm.py`, `rope.py`, `qkv_proj_rope.py`, `o_proj.py`, `hc.py`, `swa.py`, `sparse_attention.py`, `activation.py`, `input_pack.py`, `engram.py`, `lm_head.py` |
| Functional modules | `indexer.py`, `compressor.py`, `prefill_moe.py`, `decode_moe.py`, `decode_moe_ep.py`, `grouped_moe.py`, `expert_routed.py` |
| Attention variants | `prefill_swa.py`, `prefill_c2a.py`, `prefill_compressed_reuse.py`, `decode_swa.py`, `decode_c1a.py`, `decode_c2a.py` |
| Layer composition | `prefill_layer.py`, `decode_layer.py` |
| Network and generation | `staged_prefill.py`, `decode_network.py`, `network.py`, `staged_generate.py` |

Runtime topology controls stay in `operator_config.py`, `runtime_config.py`,
`decode_config.py` and `decode_layer_config.py`, with their original defaults
and validation. Tiling remains with the operator that owns it.

| Previous internal modules | Current owner |
| --- | --- |
| `decode_network_gemm`, `decode_network_linear_primitives`, `decode_network_ecotech_linear`, `decode_network_decode_primitives` | `gemm`, `quant`, `rmsnorm`, `rope` |
| `decode_network_quant`, `decode_network_norm_rope`, `decode_network_decode_rmsnorm` | `quant`, `rmsnorm`, `rope` |
| `decode_network_hc`, `decode_network_decode_hc` | `hc` |
| `decode_network_qkv_proj_rope`, `decode_network_decode_qkv_proj_rope`, `decode_network_decode_qkv_w8a8` | `qkv_proj_rope` |
| `decode_network_decode_projection`, `decode_network_decode_output_proj` | `activation`, `gemm`, `indexer`, `o_proj`, `rmsnorm`, `rope`, `swa` |
| `decode_network_indexer`, `decode_network_compressor` | `indexer`, `compressor` |
| `decode_network_embedding`, `decode_network_decode_head` | `input_pack`, `lm_head` |
| `decode_network_swa`, `decode_network_engram` | `swa`, `decode_swa`, `engram`, `decode_layer` |
| `decode_network_decode_c1a`, `decode_network_decode_c2a`, `decode_network_decode_layer` | `decode_c1a`, `decode_c2a`, `decode_layer` |
| `decode_network_moe`, `decode_network_decode_config`, `decode_network_decode_sparse_attention` | `decode_moe_ep`, `decode_layer_config`, `decode_compressed_attention` |
| `decode_network_weights`, `decode_network_generation_state` | `resident_weights`, `state` |

Consumers import the current owner directly. Exact duplicate operators share
one definition. Unpacked `[K, N]` projection interfaces use `_kn` names beside
the existing blocked-weight entries. Different quantization, normalization,
attention, Engram and expert boundaries retain their original bodies and
specializations.

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

`grouped_moe.py` owns blocked-INT8 grouped experts in the shared generation
network. `expert_routed.py` retains the separate NZ experts used by bounded
Decode compositions. `moe_transport.py` owns CANN/HCCL resources.
The unused native dispatch adapter is removed;
there is no compatibility `moe.py` facade.

L2 regions inline existing children between host communication boundaries.
They do not change operation order, BF16 buffers or source/cache ownership.
`decode_sparse_attention.py` stays TP-specialized and separate from Prefill
configuration. Full-width rank compositions retain their original interfaces;
the generation weight owner assigns 24 experts per EP rank.
`decode_network.py` advances the shared TP4 schedule by one token over the
original Prefill cache aliases. Both phases execute device embedding, Engram
and the vocabulary head; independent device accuracy remains separate validation work.

Compute weights, workers and communication groups persist across Prefill
chunks and one-token Decode steps. Stage activations retain device allocations,
with one Decode or two alternating Prefill boundary buffers per rank. HCCL
reads and writes these device allocations directly; integer control validation
and final vocabulary output are the explicit device-to-host boundaries.
Engram tables stay on the host; device kernels consume prepared rows.
Runtime shape, dtype, alias, version and lifecycle checks are production
interface requirements and remain in place.

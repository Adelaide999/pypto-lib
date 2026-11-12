# A3 module ownership

The flat A3 package contains Prefill and Decode. It follows the shared model's
operator-to-layer-to-forward dependency direction. Phase-specific operators
retain separate implementations when their arithmetic, state or topology
contracts differ; neither phase imports the other.

| Responsibility | Modules |
| --- | --- |
| Public whole-network Prefill and compilation reuse | `prefill_fwd.py`, `prefill_program.py`, `prefill_l2.py` |
| Generic layer composition and all physical-layer roles | `prefill_layer.py`, `prefill_layer_state.py`, `prefill_layer_plan.json` |
| SWA and compressed attention | `prefill_swa.py`, `prefill_c2a.py`, `prefill_c1a.py`, `prefill_compressed_reuse.py`, `prefill_sparse_attention.py` |
| Compressor projection, pooling and observation | `compressor_projection.py`, `prefill_c2_pool.py`, `prefill_c2a_observed.py`, `prefill_c2a_readback.py` |
| Public bounded Decode and complete layer selection | `decode_fwd.py`, `decode_replay.py`, `decode_layer.py` |
| Actual Decode SWA, Engram and compressed blocks | `decode_swa.py`, `decode_engram.py`, `decode_c2a.py`, `decode_c1a.py` |
| Phase orchestration and shared MoE computation | `prefill_moe.py`, `decode_moe.py`, `moe.py` |
| Routed expert computation | `expert_routed.py`, `decode_expert_routed.py` |
| Explicit resource lifetime and communication | `prefill_resident.py`, `prefill_resident_weights.py`, `prefill_network_state.py`, `moe_transport.py` |
| HOST Engram preparation and device projection/gate | `engram_host.py`, `engram_storage.py`, `prefill_network_engram.py`, `prefill_engram.py` |
| Shared configuration and leaf contracts | `config.py`, `operator_config.py`, `compressor_core.py`, `engram.py`, `indexer_prepare.py`, `utils.py` |
| Independent phase numerical/cache variants | `quant.py` / `decode_quant.py`, `hc.py` / `decode_hc.py`, `swa_cache.py` / `decode_swa_cache.py` |
| Original provenance and maintained identity | `source_manifest.json`, `decode_source_manifest.json`, `source_refactor_map.json`, `package_refactor_map.json`, `maintenance_baseline.json` |

Routing, dispatch, shared experts, combination, their Torch goldens and their
bounded validators live in `moe.py`. Routed expert leaf kernels remain in
dedicated files, as in the shared model. Transport and memory ownership have
independent runtime responsibilities and remain separate from computation.
The two phases keep distinct MoE functions where they have different FP16,
FP32 or BF16 rounding boundaries. Import aliases bind callers to the same
variant they used before this reorganization.

Unused historical fixed-layer modules and CPU-reference Decode facades are
removed. The generic `prefill_layer.py --layer-id` and
`decode_layer.get_layer_entry(layer_id)` select the real computations.
Active layer-2 kernels have moved to `compressor_projection.py`; the actual
TP grouped output kernel belongs to `qkv_proj_rope.py`, alongside the shared
BF16 projection and RoPE kernels. Observation and cache ownership are retained.

The two immutable source manifests retain their historical paths. The
package relocation map records their maintained module/symbol owners and
function identities against the pre-refactor commit. It does not grant
numerical acceptance. Runtime identity and state checks remain independent
of the optional maintenance hash audit.

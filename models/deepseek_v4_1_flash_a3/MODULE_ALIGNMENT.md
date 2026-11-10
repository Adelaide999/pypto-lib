# Module ownership

| Responsibility | Product modules |
| --- | --- |
| Public token-to-logit Prefill, whole-network compilation and IR reuse | `prefill_fwd.py`, `prefill_program.py` |
| L2 compute composition between communication boundaries | `prefill_l2.py` |
| Complete layer and full 40-layer composition | `prefill_layer.py`, `prefill_layer_state.py`, `prefill_layer_plan.json` |
| Actual network runtime, original L39 output | `prefill_network_runtime.py`, `prefill_network_input.py`, `prefill_output.py` |
| Persistent cache allocation, aliases, owner/version handoffs | `prefill_network_state.py`, `prefill_network_cache.py`, `prefill_cache_layout.py`, `prefill_cache_publish.py` |
| Exact state and discrete observations | `prefill_network_guard.py`, `prefill_network_discrete_observer.py`, `prefill_network_shared_observer.py` |
| SWA and compressed attention | `prefill_swa.py`, `prefill_full2*.py`, `prefill_consumer*.py`, `prefill_ratio1.py`, `prefill_indexer*.py` |
| Network Engram lifetime and storage | `prefill_network_engram.py`, `prefill_engram*.py`, `engram_storage.py`, `engram_ecotech.py` |
| EP dispatch, routed/shared experts and reduction | `prefill_moe_ep.py`, `moe_dispatch.py`, `moe_router.py`, `moe_shared.py`, `moe_combine.py` |
| Actual device memory and communication | `moe_transport.py`, `runtime_utils.py` |
| Shared projection, quantization, normalization and mHC | `ecotech_linear*.py`, `quant.py`, `qkv*.py`, `norm_rope.py`, `hc.py`, `gemm.py` |
| Explicit checkpoint operands and expert ownership | `weights.py`, `raw_checkpoint.py`, `runtime_config.py` |

Prefill-only modules carry the `prefill_` prefix. Shared primitives retain
general names. `grouped_output.py` becomes `prefill_grouped_output.py` because
its only exported kernel is `prefill_grouped_output`. Internal `a2` name hints
and shared A2/A3 ABI names retain their source identity.

The maintained public entry is `prefill_fwd.py`. Historical task release
checks live in `audit_task.py`; original and maintained source identities are
checked separately by `audit_source.py`. The source manifest is immutable;
`source_refactor_map.json` tracks consolidation into these owners:

| Former owner | Maintained computation owner |
| --- | --- |
| `ops.py`: INT8 packing, quantization, dot, scaling | `linear_primitives.py` |
| `ops.py`: RMSNorm, adjacent-pair RoPE | `rmsnorm.py`, `norm_rope.py` |
| `staged.py`: BF16/FP32/direct Linear | `gemm.py` |
| `staged.py`: grouped projection, SwiGLU | `output_proj.py`, `activation.py` |
| `staged.py`: norm/RoPE, index quantization, SWA | `norm_rope.py`, `index_cache.py`, `swa_attention.py` |
| Six `decode_*` reference facades | `layer14_15.py`, `layers19_25.py`, `swa_attention.py` directly |
| Fixed L1/L3 command wrappers | `prefill_layer.py --layer-id` |

Cache ownership, Engram lifetime, communication and numerical variants with
distinct rounding contracts remain separate. See the public migration notes
for optional native-helper consumers and validation limits.

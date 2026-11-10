# Module ownership

| Responsibility | Product modules |
| --- | --- |
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

Legacy `decode_*` modules and native-boundary/reference helpers are retained
as part of the tracked source closure. They do not establish new A3 Decode
validation or replace the PyPTO full-network arithmetic with native forward.
No custom product C++ translation unit is required by this source closure;
the actual transport binds the installed CANN/HCCL libraries through ctypes.

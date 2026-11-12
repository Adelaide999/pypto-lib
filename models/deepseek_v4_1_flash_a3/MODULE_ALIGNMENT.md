# A3 module ownership

The reference is the shared model at
[`e1c3522`](https://github.com/hw-native-sys/pypto-lib/tree/e1c3522fa2670e13efaab80def5c04e7cfab024a/models/deepseek_v4_1_flash).
Its forward files select layer roles and slice stacked weights and caches;
attention and MoE compositions call independent child operators. Shared files
such as `compressor.py`, `engram.py`, `gemm.py`, `hc.py`, `quant.py` and
`rmsnorm.py` own their computations and corresponding Torch goldens. Large
Prefill and Decode compositions remain separate.

This package follows that responsibility split. It retains the A3 Eco-Tech
W8A8 implementation: the reference model's MXFP8/MXFP4 kernels, cache layouts
and distributed topology are not substituted for the existing implementation.
The package has 123 Python files, down from 151 at `c8297f4`; 28 fragmented
modules have been removed rather than moved into another directory.

## Maintained owners

| Responsibility | Modules |
| --- | --- |
| Public forward and reusable Prefill compilation | `prefill_fwd.py`, `prefill_program.py`, `prefill_l2.py`, `decode_fwd.py` |
| Generic layer selection and composition | `prefill_layer.py`, `decode_layer.py`, `prefill_layer_plan.json` |
| SWA and compressed attention composition | `prefill_swa.py`, `prefill_c2a.py`, `prefill_compressed_reuse.py`, `decode_swa.py`, `decode_c2a.py`, `decode_c1a.py` |
| Compression projection, ratio-1/2 pooling and cache publication | `compressor.py` |
| Index cache, query projection, candidate pooling and Decode selection | `indexer.py` |
| Extended causal Prefill Indexer selection and composition | `indexer_select.py`, `prefill_indexer.py`, `decode_indexer_projection.py` |
| Engram lookup, projection, gating and bounded validators | `engram.py` |
| HOST Engram storage, hashing and row preparation | `engram_host.py`, `engram_storage.py`, `engram_hash.py`, `engram_offload.py` |
| Shared MoE routing, dispatch, shared expert and combination | `moe.py` |
| Phase MoE orchestration and routed expert computation | `prefill_moe.py`, `decode_moe.py`, `expert_routed.py`, `decode_expert_routed.py` |
| Linear, normalization, QKV and attention leaves | `gemm.py`, `linear_primitives.py`, `rmsnorm.py`, `quant.py`, `qkv_proj_rope.py`, `sparse_attention.py` |
| Checkpoint readers and explicit phase weight layouts | `weights.py`, `prefill_resident_weights.py` |
| Cache ownership, resource lifetime and communication | `prefill_layer_state.py`, `prefill_network_state.py`, `prefill_resident.py`, `moe_transport.py` |
| HOST control checks and provenance audits | `validate.py`, `validate_resident.py`, `validate_engram_host.py`, `audit_source.py`, `audit_task.py` |

`prefill_c2a.py` owns its observed source operations and cache readback.
Compressor operations no longer live in fixed-layer files. Public package
exports keep their names while resolving directly to the maintained owners;
there are no forwarding files for removed modules. Explicit runtime workers
remain separate because they own process, native-library and resource lifetime
boundaries, rather than merely forwarding a kernel call.

## Shared definitions and numerical variants

Seven modules were already shared when the Decode history was integrated.
This pass additionally reuses 16 definitions whose bodies and transitive
bindings match: six W8A8 Torch helpers, INT8 weight packing and dot primitives,
compressed cache publication, compressor projection and Indexer publication,
Index cache dequantization, the RMSNorm golden, and three GEMM factories.

The following implementations remain distinct inside their owning files:

| Boundary | Preserved implementations |
| --- | --- |
| Dynamic activation quantization | `quant.make_dynamic_int8_quantize` and `make_decode_dynamic_int8_quantize`; Decode retains its CANN FP16 intermediate option |
| Linear quantization and dequantization | `linear_primitives.make_quantize` / `make_decode_quantize` and `make_dequantize` / `make_decode_dequantize`; tile widths, scale application and narrowing boundaries remain unchanged |
| Normalization | `rmsnorm.make_prefill_norm`, `make_decode_norm`, tiled normalization and router FP32 streams retain their original rounding and storage |
| Compression and indexing | Decode-prefixed factories retain their original norm, RoPE, candidate bounds and cache publication dependencies |
| MoE | Existing phase rounding, routing, expert sharding and communication contracts remain separate |

Other phase files remain separate when annotation handling, tiling, numerical
boundaries or state contracts differ. Similar names alone do not establish
interchangeability. Shared parents do not import Prefill or Decode composition
files, and the maintained local import graph is acyclic.

## Weight ABI provenance

The former `decode_weight_abi.py` was introduced by Lun7777 in
[`2fd4aa3`](https://github.com/Adelaide999/pypto-lib/commit/2fd4aa37d4e1c16564233f9c06985653bf42533d)
and updated in `0284d5e`. Its current owner is `weights.py`.
`load_ep2_rank_weights` still loads the real router, routed shard and shared
expert operands for one EP2 rank. Moving that reader neither implements EP16
Decode nor supplies a full-network Decode state and cache ownership plan.

## Preservation and validation

`source_manifest.json` and `decode_source_manifest.json` remain immutable.
`package_refactor_map.json` records historical definitions and their current
owners, plus the scope-aware structural relocation from `c8297f4`.
`maintenance_baseline.json` is an optional identity audit, separate from
correctness and device admission. Arithmetic order, shapes, dtypes, rounding,
cache ownership and the existing acceptance criteria are unchanged.

Local checks compare Torch golden outputs and real PyPTO specializations
before and after reorganization. HOST control tests also exercise explicit
device mappings, chunk continuation and resource cleanup. These checks do not
execute an NPU, establish end-to-end accuracy or measure performance.

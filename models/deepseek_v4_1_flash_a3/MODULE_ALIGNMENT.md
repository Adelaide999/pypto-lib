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
The package has 77 Python files, down from 123 at `5e463e4` and 151 at
`c8297f4`. Removed modules are consolidated into their computation owners,
without compatibility forwarding files or a second implementation directory.

## Maintained owners

| Responsibility | Modules |
| --- | --- |
| Public forward and reusable Prefill compilation | `prefill_fwd.py` (network loop, compile reuse, runtime), `decode_fwd.py` (bounded Decode step) |
| Generic layer selection and composition | `prefill_layer.py` (including L2 region bindings), `prefill_layer_plan.json` |
| SWA and compressed attention composition | `prefill_swa.py`, `prefill_c2a.py`, `prefill_compressed_reuse.py`, `decode_swa.py` |
| Input packing and final output head | `input_pack.py`, `lm_head.py` |
| Compression projection, ratio-1/2 pooling and cache publication | `compressor.py` |
| Index cache, query projection, candidate pooling and Decode selection | `indexer.py` |
| Extended causal Prefill Indexer selection and composition | `indexer.py`, `prefill_indexer.py` |
| Engram lookup, projection, gating and bounded validators | `engram.py` |
| HOST Engram storage, hashing and row preparation | `engram_host.py` |
| Shared MoE routing, dispatch, shared expert and combination | `moe.py` |
| Phase MoE orchestration and routed expert computation | `prefill_moe.py`, `expert_routed.py` |
| Linear, normalization, QKV and attention leaves | `gemm.py`, `quant.py`, `rmsnorm.py`, `rope.py`, `qkv_proj_rope.py`, `o_proj.py`, `swa.py`, `sparse_attention.py` |
| Checkpoint readers and explicit phase weight layouts | `weights.py` |
| Cache ownership, resource lifetime and communication | `cache.py`, `state.py`, `prefill_resident.py`, `moe_transport.py` |
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
| Linear quantization and dequantization | `quant.make_quantize` / `make_decode_quantize` and `make_dequantize` / `make_decode_dequantize`; tile widths, scale application and narrowing boundaries remain unchanged |
| Normalization | `rmsnorm.make_prefill_norm`, `make_decode_norm`, tiled normalization and router FP32 streams retain their original rounding and storage |
| Compression and indexing | Decode-prefixed factories retain their original norm, RoPE, candidate bounds and cache publication dependencies |
| MoE | Existing phase rounding, routing, expert sharding and communication contracts remain separate |

The additional consolidation shares six definitions with equivalent bodies
and dependencies: the RoPE block helper, QKV shape class and constructor,
sparse-attention builder and reference, and routed-weight generator.

Prefill and Decode variants remain separately named inside shared files when
tiling, numerical boundaries or state contracts differ. Concrete tensor
annotations are retained in shared DSL modules so shape-only specialization
and L2 composition continue to work without sample payloads. Similar names alone do not establish
interchangeability. Shared parents do not import Prefill or Decode composition
files, and the maintained local import graph is acyclic.

## Weight ABI provenance

The former `decode_weight_abi.py` was introduced by Lun7777 in
[`2fd4aa3`](https://github.com/Adelaide999/pypto-lib/commit/2fd4aa37d4e1c16564233f9c06985653bf42533d)
and updated in `0284d5e`. Its current owner is `weights.py`.
`load_decode_rank_weights` loads router/shared operands and one 24-expert
EP16 shard. Per-expert arithmetic and packing remain unchanged. The old EP2
drivers and fixed two-device compile interfaces are retired; their history
remains at `2704263`. The maintained loader does not provide TP attention
binding, network communication or source/cache ownership transitions.

## Preservation and validation

`source_manifest.json` and `decode_source_manifest.json` remain immutable.
`package_refactor_map.json` records historical definitions and their current
owners, plus the scope-aware structural relocations from `c8297f4` and
`5e463e4`.
`maintenance_baseline.json` is an optional identity audit, separate from
correctness and device admission. Arithmetic order, shapes, dtypes, rounding,
cache ownership and the existing acceptance criteria are unchanged.

Local checks compare 43 Torch golden outputs bit for bit and 116 real PyPTO
IR structural hashes before and after reorganization. The 22 compilation-reuse
and CI-selection tests retain their assertions and use the new module paths. HOST control tests also exercise explicit
device mappings, chunk continuation and resource cleanup. These checks do not
execute an NPU, establish end-to-end accuracy or measure performance.

## Boundaries retained deliberately

- `prefill_fwd.py` owns the 40-layer network loop, compilation cache, input/head
  dispatch and runtime state. `prefill_layer.py` owns role selection and L2
  regions between existing HOST HCCL boundaries. L2 denotes the PyPTO
  orchestration level; there is no file specialized to physical layer 2.
- Resource workers remain separate when they own subprocess, native extension,
  communication or allocation lifetime. Merging their process boundaries would
  change runtime behavior and requires separate device validation.
- `runtime_config.py` remains independent of the legacy Decode CLI configuration
  in `decode_config.py` and `decode_sparse_attention.py`; importing Prefill
  topology does not parse Decode command-line arguments. The TP-bound Decode
  sparse kernels remain separate, while their parameterized builder and Torch
  reference live in `sparse_attention.py`.
- Fixed EP2 Decode orchestration and complete-layer wrappers are removed.
  Local attention, routed/shared expert and other independent computation
  remain available. EP16 full-network execution remains unimplemented.
- This is not a single fused 40-layer L2 device graph. Existing communication,
  cache ownership and execution order are retained; compile and execution
  latency improvements require measurements on allocated hardware.

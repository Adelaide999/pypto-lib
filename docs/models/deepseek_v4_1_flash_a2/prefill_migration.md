# Prefill product migration boundary

The current A3 product and future whole-network product use PyPTO for all model
tensor computation. Historical native boundaries remain reference/diagnostic
code while their callers are migrated; their historical numerical results are
not inherited by the new product. Acceptance is scoped to each actual queued
topology and case set; the EP2 delivery acceptance below does not validate the
other rows as complete-layer computations.

| Model responsibility | Legacy boundary or consumer | PyPTO replacement and required validation |
| --- | --- | --- |
| Delayed mHC collapse and residual expansion | `hc_pre_runtime.py`, `prefill_hc_full2_runtime.py`, staged HC callers; `_C_ascend.npu_hc_pre_v3` and vendor libraries | `hc.py` pre/post definitions; delayed coefficients, Sinkhorn, BF16 boundaries and residual states |
| RMSNorm | `norm_runtime.py`; Torch-NPU RMSNorm | `rmsnorm.py`, `norm_rope.py`, `prefill_norm.py`; full reduction and exact effective rows |
| BF16/FP32 projections | `projection_runtime.py`, `bf16_linear_native.py`; ATen/native linear | `staged.py`, `gemm.py`, shared projection primitives; source dtype, accumulation and narrowing order |
| W8A8 QKV and grouped output | Current PyPTO projection leaves and their composite callers | `qkv_w8a8.py`, `ops.py`, `grouped_output.py`; per-boundary TP1/2/4, CP/SP full-head Q, scale reductions and output collective |
| SWA and sparse MLA | `swa_native.py`, `smla_native.py`, `smla_staging.py`; registered native MLA | `swa_attention.py`, `swa_cache.py`, `sparse_attention.py`; same causal controls, cache ABI, sink and BF16 ordering |
| Compressor and cache publication | Prefill ratio1/ratio2 stages and staged projection boundaries | `compressor_core.py`, `prefill_c2_pool.py`, `prefill_cache_publish.py`; physical source IDs, initial/ring/cache state and untouched regions |
| Quantized indexer and candidate selection | `qli_runtime.py`, `prefill_qli_full2_runtime.py`, `qli_boundary.py`; native QLI/metadata | `indexer_query.py`, `indexer_prepare.py`, `indexer_select.py`, Prefill indexer; exact indices, source mapping, scales, Top512 and refresh branches |
| Engram | Existing row-preparation and projection/gating stages | `engram.py`, `engram_ecotech.py` and common PyPTO leaves; INT8 rows/scales, actual TP columns, host offload and DP shared memory |
| Router/shared/routed experts | Existing PyPTO leaves plus historical native dispatch | `prefill_moe_ep.py` consumes `moe_router.py` and `moe_shared.py`; original384 experts/top6, natural/controlled routing, channel-first W8A8 scales, distinct shared BF16 and routed FP32 activation boundaries |
| EP dispatch and token restoration | `moe_dispatch_runtime.py`, `moe_boundary.py`; Torch-NPU/native routing and host tensor packing | `prefill_moe_ep.py` consumes PyPTO `moe_dispatch.py` pack/unpack/return-pack and `moe_combine.py`; `moe_transport.py` exchanges their contiguous buckets using CANN memory and HCCL bytes; real DP2/TP1/EP2 first, then EP16 |
| Complete physical layer | Stage factories, staged native callers and CPU layer goldens | Compose the migrated common PyPTO operators; no golden injection; compare full output and all effective states at DP4/TP4/EP16 |

Physical3's independent consumer entry composes `prefill_layer.py` with
`prefill_consumer_runtime.py`. It loads physical3 operands, publishes its own
computed BF16 SWA values and reads source2's original raw cache and shared
selection. `prefill_indexer.make_sparse_controls(selection_ready=True)` maps
those logical IDs without repeating producer filtering or sorting. Source2
executes no layer computation, and its bytes and shared buffers remain
readonly. The seven-case physical3 gate and actual affected accepted
regressions establish acceptance only after their frozen process succeeds.

Physical14 uses the shared parameterized complete-layer owner with original
Engram slot1 and its own ratio-two KV/index producer. `engram_owner` resolves
physical checkpoint prefixes and verifies the original hash slot. Compressed
metadata and raw byte backings retain the actual physical owner. The lookup
uses token-column tasks and preserves every group32 scale and dtype boundary.
This route retains all 384 experts and assigns 96 per rank at DP2/TP2/EP4.
Its matching seven-family EP4 gate and affected accepted regressions remain
not_run. The retained EP16 quantized-key/cache and query failures require
current-source owning diagnostics; earlier layout experiments confer no
acceptance on this route.

Product process search paths exclude reference source/package paths. Product
execution records actual imported source, PyPTO artifacts, runtime captures,
descendant process commands and mapped libraries. A static import check alone
does not establish independence. Torch goldens remain colocated according to
repository rules but may not supply product computation.

Engram lookup metadata lives on `engram_ecotech.make_resident_int8_lookup`;
projection metadata lives on `qkv_proj_rope.make_bf16_projection_staged`.
The Prefill registry compiles these owners directly, while resident and
attention/index consumers use inline views of the same bodies. Guarded row
selection, INT8 widening, FP32 group32 multiplication and BF16 RNE share one
PyPTO core region. The staged code/scale outputs and separate dequantization
entry have been removed. Host-table/UVA ownership
remains in `MappedLookup`; computation and reference paths remain separate.
The [owning Golden inventory](index.md) covers row guards/scales, projection,
hash/history, TP/SP packing, gate intermediates and readonly readback. Its
compile-only checks require fresh device validation before acceptance.
The fused lookup definition uses the earlier run0010 A implementation as
research provenance only; its source and scores confer no current acceptance.
The current baseline's hash/history, projection, gate arithmetic, full-table
backings and DP sharing remain the shared owning implementations.

True zero-request controls belong to `swa_scheduler`: zero requests retain
starts `[0]`, empty control lists, zero maxima and complete cache metadata.
`prefill_layer` retains the `[0,2]` empty slot ABI. The current necessary L1
wrapper checks 13 malformed controls and six complete native boundary cases,
then regresses affected old-layer dependencies. No dummy requests are added;
empty ranks keep their independent expert ownership and collective duties.
The gate's probability publication now uses one owner per 16-token block
across all four branches. The public token-major shape and floating formula
are unchanged. QR validation samples use contiguous copies of the same
rotation values; frozen complete-layer fixtures remain unchanged.

The EP2 computation contract (retained validation record `a3_prefill_ep2_pypto`)
and its v3 process lifecycle gate (retained validation record `a3_prefill_ep2_pypto_v3`)
combine static call inventory, dynamic execution constraints and the unchanged
independent numerical reference. Candidate promotion and delivery replay require
all rank/cases and the dependency gate. A passed operator does not validate a
complete layer or future network; missing branches remain not_run.
Official device evaluation uses only
`launch_v4.py` (retained validation record `a3_prefill_ep2_pypto_v3`),
with terminal-state queue ownership (retained validation record `a3_prefill_ep2_pypto_v3`).
The launcher retains the common NPU lock until its single bounded two-card job
finishes or is cleaned up. Earlier launcher entries remain historical evidence.

`prefill_moe_ep.py` accepts only `--inputs` and `--output`. It loads original
checkpoint operands through `CheckpointOperands`, composes the shared PyPTO
definitions, and retains actual rank/case captures. Weight layout preparation
uses host tensor containers; dispatch packing, expert slabs and weighted token
restoration execute in PyPTO. Empty ranks participate in both HCCL exchanges.
The parameterized ownership contract supports EP2/EP16 and TP1/2/4 rank-local
controls; the current gate uses DP2/TP1/EP2 without a TP split.

The current implementation reuses candidate A's retained CPU-complete snapshot
from run_0004, with source identities and earlier failure provenance in
`source_manifest.json`. Dispatch counters remain INDEX until storage, metadata
and payload indices have separate names, typed converters consume slices or
explicitly typed buffers, and quantization pads after conversion.

The corrected snapshot compiled all eight entries: natural and controlled
router, shared and routed experts, dispatch pack/unpack, return pack and combine.
All 96 Python source hashes matched the initial product artifact exactly.
All 109 snapshot hashes, 30 retained PTO artifacts and 32 child audit records were verified before reuse;
the CPU process observation records complete audits and zero static violations.
The failed first CPU attempt remains preserved. Documentation and provenance
updates leave the active compiled definitions and their shapes unchanged.

The first official execution produced all 16 rank/case outputs and passed the
product dependency gate, then failed while inventorying a synthetic PyPTO
module path. Source inventory now hashes imported Python files within the
model directory; a missing file within that directory still raises an error.
The uncalled legacy `moe_combine_test` and its unused token-count environment
setting were removed. The product combine factory, its independent colocated
golden and its Golden Harness validator remain unchanged. These host metadata
and obsolete-definition changes require no new kernel or compile shape.

The corrected product passed all16 DP2/TP1/EP2 rank-cases, static/dynamic
independence, exact routing/state and actual remote dispatch/combine, followed
by independent promotion and delivery replays. Both K=2 candidates retained
their own evidence; no performance claim follows from equal correctness scores.
The accepted delivery is627c23a9. Earlier failures and evidence remain preserved.

The complete L0 short contract (retained validation record `a3_prefill_l0_pypto`)
freezes48 native DP4/TP4/EP16 rank-cases, effective CP/SP boundary shapes,
BF16 output projections, FP32 router input, full cache/HC outputs and unchanged
comparators. Product validity adds actual16-rank groups/collectives and CANN
allocation lifetimes, with reference computation excluded from every product
process. Main's applied-delivery short replay and the independent
page/window extension (retained validation record `a3_prefill_l0_boundary`)
passed all 96 rank-cases. Physical L0 is accepted for these six cases at
DP4/TP4/EP16, PP1, PCP1 and DCP1. Other physical layers, TP1/2 device
compatibility and whole-network execution remain not_run.

The physical L0 entry composes the existing HC, tiled RMSNorm, W8A8 QKV,
RoPE, paged SWA, BF16 grouped output, router and shared/routed expert owners.
`prefill_cp.py` packs complete 64-head local CP results into equal HCCL peer
buckets and compacts the resulting 16-head and SP views. Cache publication
uses three pages. HC state, cache writes and externally supplied output buffers
have explicit read/write directions, including writes through reshaped views.
All tensor calculations and dispatch/combine packing remain PyPTO operations.

Physical SP rows and valid dispatch rows are separate. HC, normalization and
shared experts process physical rows. MC2 routing observations additionally
include zero-logit padding to the maximum physical SP row count across the EP
group. Dispatch transports only valid rows; every rank participates. The common
combine owner treats masked routes as zero and adds shared results over all
physical rows. The physical-layer composition then zeros masked MoE rows
before final HC and masked hidden rows after final HC with PyPTO stores,
using the original SP valid and physical bounds. Shared-expert observations,
delayed HC state, valid rows and inactive capacity remain unchanged.
Empty CP queries retain their zero-row observation tensors.

The physical L0 implementation uses candidate B's staged tile and compact
CP/SP layout. Its retained run-6 ancestor has no acceptance under the
version-2 contract (retained validation record `a3_prefill_l0_pypto_v2`).
Both later K=2 candidates were evaluated under that immutable contract;
the selected B artifact was independently replayed, applied and revalidated
by Main before the separate page/window acceptance.
FFN normalization emits BF16 expert values and widens those rounded values for
the FP32 router. Independent CPU analysis confirms this native boundary across
all 48 frozen captures. SWA narrows softmax probabilities with ties away from
zero, divides the FP32 weighted accumulation by its denominator, and narrows
the final output with ties to even. Exact valid-row Top6 ordering passed
both applied-delivery suites with real cross-rank dispatch and combination.
Received and returned route validators inspect actual product metadata.

The tiled RMSNorm owner stages only valid one-row input slices into an
eight-row tile before each reduction and output pass. Its callers include
`prefill_norm.py`, `prefill_norm_projections.py`, `prefill_attention_block.py`
and `prefill_layer0_local.py`. Effective row counts of 1 and 7 remain distinct
from physical capacity; compact capacities also require bounded input slices.
FP32 routing values are exported before BF16 expert narrowing. Compilation
coverage must include both output dtypes, compact capacities and these affected
callers; device correctness and inactive-state checks remain separate gates.

`runtime_utils.py` owns common compilation, captures and HCCL process launch
helpers used by both Prefill entries. `make_l0_attention`, the colocated CPU
goldens and historical native consumers retain their existing interfaces for
their documented callers. Those compatibility entries can be removed only
after their consumers migrate and their corresponding gates are replaced.

Physical L1 remains a separate migration unit. Its outer-model Engram runs
before that same decoder layer: token hash/history, full INT8 codes with
group32 FP32 scales, TP-owned column lookup, SP token sharding, BF16 projection
and original-basis gating must all execute through PyPTO. A bounded CPU row
gather or an imported native helper cannot establish this product path.
TP4 owns six of the 24 hash columns per shard; matching DP peers share the
complete host shard and retain their own runtime mappings and lifetimes.
Existing ABI inspection and large-shape compilation do not establish actual
registration, numerical/state agreement or complete L1 acceptance.


The physical-layer orchestration now lives in `prefill_layer.py`; both L0 and
L1 select their checkpoint prefix in that owner. The L0 command and its public
factory aliases retain their accepted interfaces. `prefill_layer1.py` composes
outer Engram and this shared computation; it does not execute a prior layer.
All L1 validation is pending, including the independent short gate and future
history/dead-token/page-window extensions. Shared refactoring additionally
requires EP2 and both accepted L0 regression suites before promotion.

`engram_storage.py` owns complete MAP_SHARED INT8/FP32 checkpoint arrays,
streaming their original contiguous bytes into one backing per TP shard and
array kind. Every DP peer registers its own live CANN mapping and executes
public `ChipWorker.run` descriptors for both complete arrays. A lookup stages
INT8 codes and FP32 scales separately. The common Engram operator then applies
group32 scaling and BF16 rounding. Whole-column TP gathering and token-only SP
sharding remain PyPTO operations. The historical raw E4M3 host adapter remains
an independent asset interface for its documented legacy consumers.

`--compile-only` on either physical entry compiles the complete bounded entry
set using Golden's a2a3 configuration. Full host storage and device contexts are
created only by the device path. This source-level separation does not establish
numerical acceptance, actual mapping validity or TP1/2 device compatibility.

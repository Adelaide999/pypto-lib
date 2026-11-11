# DeepSeek V4.1 Flash on A2

Physical L0, L1, L2 and L3 have passed independent complete-layer A3 acceptance at
DP4/TP4/EP16, PP1, PCP1 and DCP1. Main replayed the applied PyPTO product on
six L0 short and page/window cases, covering 96 rank-cases. The accepted L1
run0012 covers its six necessary-boundary cases and 96 rank-cases, including
dead current/history tokens, multiple requests and an actually empty DP.
The accepted L2 product is recorded by the physical3 gate's immutable
dependency-regression baseline. Acceptance belongs to each exact process-verified
source artifact. Physical L14, L15, L19, L20, L21, L24 and L25 have also passed their own
independent complete-layer acceptance and required impact regressions at
DP2/TP2/EP4. These seven deliveries preserve all 384 experts, 96 per rank,
Top6, and the matching CP/SP, quantization, cache and shared-expert modes.
Together with L0鈥揕3 they complete the eleven independent-layer targets.
Each delivery remains bound to its own actual source, inputs, initial state,
independent reference and topology. The retained L14 EP16 failures keep their
original status. EP4 success does not establish EP16 whole-network acceptance.

The source now includes complete forty-layer Prefill composition through
`prefill_network_runtime.run_network_runtime`, persistent cache and source
ownership, shared Engram tables, and the final HC/norm/vocabulary boundary.
Its execution guard requires all eleven formal deliveries, verified actual
capacity and sharding, a matching independent full-network reference and a
complete queue-owned DP4/TP4/EP16 allocation. Source installation, compilation
and output-arithmetic checks do not establish complete-network accuracy.
Complete-network execution, the actual layer39 output boundary and fixed16
TP1/TP2 product compatibility remain unvalidated.

TP1 disables several CP/SP modes in the fixed native implementation; its
distinct execution route still needs validation. Each topology requires its
own inputs, initial state and independent reference.

Physical L0 short acceptance uses the
version-2 L0 gate (retained validation record `a3_prefill_l0_pypto_v2`).
It retains the frozen native valid reference and requires deterministic zero
product state in masked combine rows. The original gate and captures remain
historical evidence; valid routing, cache, readonly and untouched-state checks
keep their original requirements.

`models/deepseek_v4_1_flash_a2/` packages the A2 single-layer Prefill
operators separately from the original TP4/DP2/EP8 model scaffold. It includes
W8A8 and BF16 projections, SWA and compressed-cache stages, mHC, Engram,
384-expert top-six MoE, and the native operator boundaries used during
independent layer alignment.

The file-level ownership and migration map against the A5 implementation is
maintained in [`MODULE_ALIGNMENT.md`](../../../models/deepseek_v4_1_flash_a2/MODULE_ALIGNMENT.md).
Use it when adding a Decode or Prefill stage: it records the corresponding A5
semantic entry, the A2 owner, the dtype translation, and whether the stage is
currently partial or accepted.

## Development priorities

| Field | Current priority |
| --- | --- |
| Performance target | Correct all-PyPTO independent A3 Prefill layers on explicitly bound multi-rank EP topologies, followed by complete-network Prefill on eight physical cards / sixteen logical NPUs, initially DP4/TP4/EP16. Keep original full-width operands and numerical boundaries. |
| Deferred performance work | Performance tuning and serving. Complete-network Prefill follows the eleven independent layer gates and verified asset/capacity readiness. Decode, dataset scoring and HTTP serving remain outside this task. |
| Compatibility requirements | Parameterize TP1/2/4 and EP ownership; preserve the frozen operators' dtypes, quantization boundaries, page layouts, request controls, source ownership, readonly inputs, inactive rows, and untouched cache regions. Fixed 16-rank compatibility targets are DP16/TP1/EP16 and DP8/TP2/EP16; unexecuted paths remain unvalidated. |

The A3 task started from the PR #2 source baseline with every layer
unvalidated. Historical A2 acceptance below keeps its original
meaning. The retained EP16 layer topology uses PP1, PCP1, DCP1, W8A8, INT8 Engram
with CPU offload and DP shared memory, DSA_CP, FlashComm1, fused MC2 and the
shared-expert-DP path. EP spans the same 16 DP-by-TP ranks, not 256 devices.
The EP2 operator reference has a separate configuration and cannot establish
formal single-layer correctness. Its contract is described in
`verification/a3_prefill_ep2_pypto/README.md` (retained validation record `a3_prefill_ep2_pypto`).
The complete PyPTO EP2 delivery gate has passed all16 rank-cases, including
actual remote dispatch/combine, exact routing/state and product independence.
This acceptance applies to DP2/TP1/EP2 only. The new
L0 short gate (retained validation record `a3_prefill_l0_pypto`) freezes
same-topology native inputs and complete observable states at DP4/TP4/EP16;
its applied-delivery replay and the independent
page/window extension (retained validation record `a3_prefill_l0_boundary`)
both passed. The six cases are `active1`, `uneven7`, `uneven16`,
`page127-active1`, `window128-active7` and `page-tail-active16`.
All required outputs and states passed the unchanged floating policies,
with exact routes, request/token mappings and readonly/inactive/untouched
checks. Static and dynamic product independence, actual PyPTO/HCCL artifacts
and closed allocation lifetimes were independently audited. This establishes
physical L0 only. The separate accepted L1 run0012 establishes complete Engram
hash/history, INT8/FP32 host offload and DP sharing, projection/gating and its
own complete layer under the necessary-boundary contract.

Current A3 product computation must be independent of vLLM/vLLM-Ascend,
Torch/Torch-NPU model arithmetic, ATB, ACLNN and vendor model operators.
Native boundaries below describe retained historical/reference paths and
are not product fallbacks. Separate native capture supplies independent
numerical references. Product dispatch packing, attention, mHC, projections,
quantization, Engram, experts and combination use PyPTO; basic memory/stream
and HCCL transport remain permitted. See the
[migration inventory](prefill_migration.md) for remaining consumers and gates.

## Construction and ownership

The directory exposes operators, bounded stage factories and the physical L0
Prefill entry described below. Callers supply actual checkpoint operands,
scheduler controls, caches and output buffers. Historical acceptance belongs
to the retained independent snapshots.

`python -m models.deepseek_v4_1_flash_a2.prefill_layer0 --tp 4 --inputs INPUTS
--output OUTPUT` composes CP/SP attention, valid-row EP16 MoE and delayed HC.
TP is read once at interpreter startup. This entry consumes the frozen input
bundle, reads original checkpoint weights and writes all rank/case state and
artifact receipts. It requires the allocated sixteen-device mapping from
`TASK_DEVICE`. Use the immutable L0 launcher for device evaluation; CPU
compilation and source checks do not establish complete-layer correctness.
The [migration boundary](prefill_migration.md) describes padding and remaining
legacy consumers.

`prefill_layer.py` owns the complete reusable physical-layer orchestration.
Its dispatch factories receive EP explicitly. Rank groups, expert ownership,
head exchange and grouped output dimensions follow the frozen input topology;
all callers must use the matching startup TP setting. Public queue leases
must cover the complete configured device set. Up to two independent small
jobs may run on disjoint leases with separate build, cache and communication
directories. The final sixteen-rank job requires the entire device set.
The physical-layer commands select their own checkpoint prefix; L1 additionally
executes outer Engram before HC-pre. The L1 implementation uses complete
DP-shared INT8/FP32 host backings and same-process public ChipWorker lookup.
The owning PyPTO lookup fuses guarded row selection, signed INT8 widening,
FP32 group32 scaling and BF16 round-to-nearest in one core region. Its BF16
output feeds TP column gathering, SP token sharding, the shared BF16
projection and original-basis gating. TP1/2 complete-layer compatibility
remains not_run; kernel construction does not establish that compatibility.

The same parameterized entry composes independent physical14 with
`--layer-id 14 --tp 2`. It consumes the supplied real physical1 hidden and
delayed-HC boundary, selects original Engram hash slot1, and loads physical14
checkpoint operands. Its own ratio-two KV and index producer follows Engram
and HC. Actual physical14 metadata and raw cache descriptors retain their
original names and ownership. Attention, shared and routed MoE, delayed HC,
and request/token restoration use the shared complete-layer implementation.
The EP4 successor gate (retained validation record `a3_prefill_l14_ep4_v2`) requires
seven matching families, all 28 rank-cases and affected accepted regressions.
This route has not passed that gate. The earlier EP16 numerical failure
remains a counterexample; TP1 and EP16 compatibility remain unvalidated.
Startup validates every expert/local-slot pair before device initialization.
Equivalent list and tuple pairs share the existing JSON list representation.

The shared per-token quantizer publishes the true row maximum divided by
127, including an exact zero scale for all-zero rows. Its inverse uses a
separate positive division guard. The retained native-boundary zero-scale
counterexample motivates this repair; nonzero code/query and route failures
remain unresolved until current-source owning-prefix and complete-layer
checks pass. The predecessor complete-layer diagnostic observed 22 of 28
rank-cases passing and grants no acceptance.

The physical-layer commands accept `--compile-only`. Run this bounded mode
through the public `task-submit --no-device` queue with the same `--tp 4`,
`--inputs` and a fresh `--output` directory. It constructs every layer entry
and retains Golden-configured PTO artifacts and source digests, without
initializing a device or materializing the full host tables. Device evaluation
uses the immutable complete-layer launchers and common task mutex. The
L1 short gate (retained validation record `a3_prefill_l1_pypto`) remains
historical evidence for its exact source. The current
necessary-boundary contract (retained validation record `a3_prefill_l1_required`)
covers six native cases and 96 L1 rank-cases: page127, window128 with explicit
lookback3, page crossing/tail, dead current tokens, dead history, and multiple
requests with a true empty DP. Its wrapper selects affected EP2/L0 regressions
from actual dependency changes and checks exact identities before reuse.
Current acceptance belongs to the exact source and terminal process report.

The Engram entries compile `engram_ecotech.make_resident_int8_lookup` and the
shared staged BF16 projection directly. These definitions own output `InOut`
directions; the projection accepts `capacity=None` for its existing dynamic
row extent or an explicit static capacity. Nested consumers use inline views
of the same bodies. Complete-table descriptor order, mapped addresses and
synchronous lifetimes remain owned by `prefill_engram_runtime.MappedLookup`.
The shared projection uses N16 tiles and K512 slabs, reduced to divisors for
smaller widths. Adjacent N16 tiles share one N32 output owner. Eligible owner
groups traverse K slabs in paired parity and phase order, with one continuous
FP32 Cube accumulator initialized at logical step zero. Vector stages the
final FP32 result and narrows once to BF16 RNE. The original 12-by-512 BF16
operand leaf matched native exactly; complete-layer and changed-dependency
acceptance require the frozen original minimum, full and affected-layer gates.
The lookup partitions work over active tokens and assigned columns. Each
task writes one complete 256-value BF16 column with eight group32 FP32
scales, signed INT8-to-FP16-to-FP32 widening, BF16 RNE and original row guards.
Both complete host tables are mapped on every rank, including active-zero
ranks. The lookup descriptor contains global INT32 IDs, full assigned INT8
codes, full assigned FP32 scales, a BF16 output and the active scalar. No
intermediate code/scale buffers or separate dequantization dispatch remain.

`swa_scheduler.from_controls` accepts true zero-request controls `[0]`, `[]`,
`[]`, retaining the complete metadata/cache ABI and zero maxima. Nonempty
requests still require positive lengths and valid distinct live pages. The
layer materializes empty slot rows with shape `[0, 2]`. Empty DP/SP ranks
participate in every collective and retain their 24 owned experts; incoming
remote tokens use the same routed-expert computation.

The following operator modules provide bounded Golden Harness validators.
Invoke them from the repository root with `python -m
models.deepseek_v4_1_flash_a2.MODULE --compile-only` for CPU code generation.
Compilation checks ordered shapes, dtypes and directions; it skips numerical
execution. Device validation requires the public queue and common task mutex.

| Module | Default correctness cases |
| --- | --- |
| `engram_ecotech` | Five fused lookup cases: signed INT8, FP32 group32 scales, BF16 midpoint rounding, row guards, shard offsets, TP1/2/4 and inactive output sentinels. |
| `qkv_proj_rope` | Four staged projection cases: dynamic/static rows, zero/one/partial/full active extents and BF16 narrowing. |
| `prefill_engram` | Four hash/history cases, twelve TP1/2/4 layout/mask cases and four original-basis/probability/output gate cases. |
| `engram_ecotech_observed` | One resident full-width three-token composition with all readonly input readbacks and inactive output sentinels. Its BF16 projection weight is initialized lazily. |

Torch references live after the kernels in their owning files. Hash/history
validation reuses the independent `engram_hash` reference; gate composites
reuse the owning original-basis and gate goldens. Existing Golden comparators
check exact integer/layout/readonly state and inactive sentinels separately
from floating prefixes. These validators do not supply product intermediates.
Changes to entry organization require fresh complete-layer and operator
device evidence; historical acceptance does not transfer to a new source tree.

Gate and observed-composition samples materialize QR rotation values in
contiguous storage for the runtime argument ABI. The shared gate gives each
16-token task all four branches and publishes probabilities with scalar
stores into the public token-major buffer. Each task owns complete 64-byte
probability cache lines; the gate arithmetic and rotated residual stay in
their existing vector computation. Fresh CPU and device validation of these
changes remains pending.

The masked-gate comparator reads a pure input mask from the harness input
context, or its Golden output when an observed entry returns the readonly
mask as `InOut`. Both paths retain the same exact masked-state checks.
CPU validation exercises tensor initialization, Golden construction and
comparison through the harness with compiled parameter directions.


| Files | Responsibility |
| --- | --- |
| `ops.py`, `qkv_w8a8.py`, `ecotech_linear*.py` | Per-token INT8 quantization, KN weight packing, full INT32 accumulation, explicit FP32 scaling and BF16 narrowing. |
| `prefill_swa.py`, `swa.py`, `grouped_output.py` | Full 64-head SWA, inverse partial RoPE, eight grouped Wo-A products and BF16 Wo-B. The raw operand adapter in `prefill_swa.py` uses layer-zero checkpoint key names. |
| `prefill_layer2_c1c2.py`, `prefill_c2_pool.py`, `ecotech_full2_*.py` | Ratio-two projection, ring state, pooling, cache publication and index observation. The width-128 index publisher includes the accepted 16-row quantization and reciprocal correction. |
| `prefill_ratio1.py` | Ratio-one publication starting at an actual BF16 compressor projection. This is a distinct input boundary from the FP32 ratio-two compressor. |
| `prefill_consumer.py`, `prefill_indexer.py`, `prefill_sparse_consumer.py` | Consumer projections, causal controls and sparse attention stages. |
| `decode_c2a_full.py`, `decode_c2a_reuse.py`, `decode_c1a_full.py`, `decode_c1a_reuse.py`, `decode_c1a_reindex.py` | A5-shaped A2/A3 decode entry points for L14/L15/L19/L20/L21/L24/L25. `layer14_15.py` and `layers19_25.py` remain aggregate compatibility providers. |
| `candidate_pool.py`, `raw_checkpoint.py` | A2/A3 candidate metadata and raw checkpoint loading. |
| `swa_scheduler.py`, `smla_scheduler.py` | Integer-only request, page and causal-selection scheduling. Original and compressed lanes retain separate page tables. |
| `swa_native.py`, `smla_native.py`, `smla_staging.py` | Registered BF16 SWA/ratio-one SMLA leaves, fresh metadata, and PyPTO dtype boundaries. |
| `engram_ecotech*.py`, `engram_hash.py`, `engram_offload.py` | INT8 row lookup, BF16 projection, original-basis gating with probability observation, and host row preparation. |
| `hc.py`, `ecotech_mhc_*.py`, `moe_*.py` | Four-stream mHC, complete shared expert, routing, native dispatch, routed expert and route combination. |
| `prefill_moe_ep.py`, `moe_dispatch.py`, `moe_combine.py`, `moe_transport.py` | Checkpoint-backed PyPTO Prefill EP consumer, dispatch pack/unpack, expert return packing, ordered weighted combination and CANN/HCCL byte transport. DP2/TP1/EP2 passed16/16 rank-cases; complete physical L0 and L1 also passed with real EP16 under their separate exact-source contracts. Later layers remain unvalidated. |
| `prefill_layer0.py`, `prefill_layer.py`, `prefill_cp.py`, `runtime_utils.py` | Physical L0 composition, PyPTO head/SP packing, actual operator captures, source identities and collective scheduling. Independent A3 short and page/window delivery acceptance passed96/96 rank-cases at DP4/TP4/EP16. |
| `prefill_layer1.py`, `prefill_engram.py`, `prefill_engram_runtime.py`, `engram_storage.py` | Complete physical L1 entry, fused mapped lookup and complete shared-host Engram ownership. Acceptance is scoped to exact process evidence. |
| `*_runtime.py`, `bf16_linear_native.py` | Actual-operand native boundaries and input/output/library identity checks. |

Construct the principal stages from the repository root in a Python 3.10+
environment containing Torch and the selected PyPTO checkout:

```python
from models.deepseek_v4_1_flash_a2.prefill_swa import GEOMETRY, make_builders
from models.deepseek_v4_1_flash_a2.prefill_full2 import make_entries
from models.deepseek_v4_1_flash_a2.prefill_ratio1 import make_ratio1_compress_publish

swa = make_builders(capacity=16)
full2 = make_entries(active=7)
full2_projected = make_entries(active=7, compressor_projected=True)
ratio1 = make_ratio1_compress_publish(16, dict(GEOMETRY, index_head_dim=128))
```

These calls construct DSL entries without executing a device workload. The
Full2 observation entries retain the bounded active counts 1, 7 and 16.
`compressor_projected=True` consumes supplied FP32 projection buffers rather
than computing them in DSL. Preserve the selected projection boundary when
comparing with an external runtime.

Quantized checkpoint preprocessing remains on the host; current A3 product
forward numerical operations execute in PyPTO. INT8
weights retain their channel scales, full reduction widths and INT32
accumulation. BF16 rounding and the order of scale multiplication are part
of the operator contract.

SWA uses BF16 `PA_BBND` pages with 128 rows. Ratio-two compressed/index
storage also uses lossless physical views of native 64-row pages; retain
the original slot mapping when using these views. Index payloads are INT8
with FP16 row scales; compressor state and projections are FP32. Consumers
must receive their declared source layer's cache and index state. Independent
layer acceptance does not validate chaining these buffers across layers.

## Historical and reference native runtime dependencies

The retained historical aligned path includes native NPU dispatch. An independent
reference environment needs a compatible CANN, Torch-NPU and vLLM-Ascend runtime exposing:

- `_C_ascend.npu_hc_pre_v3` for the HC-pre boundary;
- `_C_ascend.npu_quant_lightning_indexer_v2_metadata` and
  `_C_ascend.npu_quant_lightning_indexer_v3` for QLI;
- `_C_ascend.npu_sparse_flash_mla_metadata` and
  `_C_ascend.npu_sparse_flash_mla` for BF16 SWA and ratio-one SMLA;
- `torch_npu.npu_moe_init_routing_v2` for dispatch;
- `torch_npu.npu_rms_norm` and Ascend `aten.linear` dispatch for the explicit
  normalization and BF16/FP32 projection boundaries.

The BF16 linear registration preserves BF16 operands and dispatch; it does
not widen the projection to FP32. Keep its returned library handles alive.
The SWA and SMLA registrations retain their library handles internally and
must each be initialized once per interpreter. Their public namespaces are
`pypto_dsv41_swa_bf16_v1`, `pypto_dsv41_ratio1_smla_bf16_v1`, and
`pypto_dsv41_bf16_linear_v1`.

Fill in `runtime.example.json` with your installed runtime paths. Before
calling a subprocess host boundary, set `ASCEND_RT_VISIBLE_DEVICES` to the
single physical device specified in that configuration. The subprocess uses
logical device zero. Device scheduling belongs to the caller; these modules
do not reserve cards or start persistent holders.

HC additionally requires an installed OPP vendor and a manifest whose
`files` mapping pins these relative library names to SHA256 digests:

```text
op_api/lib/libcust_opapi.so
op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so
```

The HC boundary checks both the digests and the libraries actually mapped in
the process. QLI callers similarly supply the expected native library
identities. Runtime manifests and compiled libraries are external inputs;
they are not bundled in this model directory. A different native build
requires its own precision validation.

## Acceptance and source provenance

The historical frozen snapshots passed independent same-input complete-block
checks for layers **0, 1, 2, 3, 14, 15, 19, 20, 21, 24 and 25**. Layer zero
has six complete calls; each of the other selected layers has twelve.
`/data/disk/sunhaolun/captrue_and_vllm/models/deepseek_v4_1_flash_a2/acceptance_manifest.json`
records the original evidence digests and limits.
The most recent L20 gate uses six cases and twelve calls with all 384 experts,
top-six routing and 64 attention heads. Its floating policy is
`atol=0.05, rtol=0.05`, with zero violations and nonfinite values; discrete
controls, routes, readonly inputs, inactive rows and untouched regions are
checked exactly. Historical acceptance belongs to those frozen snapshots.

`source_manifest.json` maps each ported file to its original source digest.
It also records unchanged definition AST digests, the native-call identities,
and host changes needed for relocation. Dependencies extracted from the
original parent model retain their compute definitions in this flat directory.
The port includes the L14 Engram observation interface, L1 native boundaries,
and the final L20 index-publication candidate.

The external capture and validation tree has host validation and
factory-construction coverage; it has not been replayed on an NPU. It makes no claim of a newly validated
full-block runner, stacked layers, whole-network dataset score, Decode,
full-table Engram residency, multi-card execution or serving support. The
original L25 evidence's missing large-readonly and dispatch-input artifacts
remain a limitation of that historical record.

## Checks without a device

Run the source receipt, dependency-closure and prefill-control checks from the
external A5-shaped validation tree:

```bash
PYTHONPATH=/data/disk/sunhaolun/captrue_and_vllm:/data/disk/sunhaolun/pypto-lib \
python -m models.deepseek_v4_1_flash_a2.validate
```

With Torch and PyPTO installed, also import every module and construct the
bounded SWA, Full2, Ratio1, Engram, and L19/L20/L21/L24/L25 decoder entries:

```bash
PYTHONPATH=/data/disk/sunhaolun/captrue_and_vllm:/data/disk/sunhaolun/pypto-lib \
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m models.deepseek_v4_1_flash_a2.validate --factories
```

These checks do not allocate NPU buffers. Actual operator validation should
use the existing [Golden Harness](../../run-and-validate/golden-harness.md)
with the same compute implementation, checkpoint operands and comparison
policy. Native boundary reports verify identity and interface integrity;
they do not replace numerical comparison against an independent reference.

## Decoder operators

The A2/A3 decoder adaptation is integrated into this same flat package. The
decoder contract is shared through `config.py`, `operator_config.py`,
`engram.py`, and `sparse_attention.py`; W8A8 quantization and GEMM live in
`quant.py` and `gemm.py`. The decoder entry points use the A5 file names:
`decode_c2a_full.py`, `decode_c2a_reuse.py`, `decode_c1a_full.py`,
`decode_c1a_reuse.py`, and `decode_c1a_reindex.py`. The CPU references and
bounded PyPTO validators are direct modules in this directory, with no nested
`verifier` package. The original `layer14_15.py` and `layers19_25.py` imports
remain available during migration.

The decoder validator suite can be run without a device:

```bash
PYTHONPATH=/data/disk/sunhaolun/captrue_and_vllm:/data/disk/sunhaolun/pypto-lib \
python -m pytest /data/disk/sunhaolun/captrue_and_vllm/models/deepseek_v4_1_flash_a2/test_operators.py -q
PYTHONPATH=/data/disk/sunhaolun/captrue_and_vllm:/data/disk/sunhaolun/pypto-lib \
python -m models.deepseek_v4_1_flash_a2.validate_layer14 --sim
PYTHONPATH=/data/disk/sunhaolun/captrue_and_vllm:/data/disk/sunhaolun/pypto-lib \
python -m models.deepseek_v4_1_flash_a2.validate_layer15 --sim
# Supply the checkpoint through --checkpoint or DSV41_A2_CHECKPOINT.
PYTHONPATH=/data/disk/sunhaolun/captrue_and_vllm:/data/disk/sunhaolun/pypto-lib \
python -m models.deepseek_v4_1_flash_a2.validate_layers19_25 --checkpoint /path/to/checkpoint
```

The validators retain the A2/A3 ABI: BF16 hidden and cache rows, INT8 weights
and index payloads, FP32 W8A8 scales, FP16 index-cache scales, and INT32 GEMM
accumulation. Prefill callers continue to use the existing `prefill_qkv_swa.py`,
`prefill_attention.py`, and cache modules unchanged.

## Physical L2 compressed Prefill

`python -m models.deepseek_v4_1_flash_a2.prefill_layer2 --tp 4 --inputs INPUTS
--output OUTPUT` composes the shared physical-layer runner with own C2 and
index computation. `--compile-only` constructs the same entries without
initializing a device. TP1/2/4 construction is parameterized; device acceptance
requires the independent DP4/TP4/EP16 seven-case, 112-rank-case gate.

The compressed path uses the actual byte backing and reconstructs typed
aliases from the supplied offsets and strides. Long-cache 64-row logical pages
occupy padded 128-row physical BF16 pages. Ring updates retain FP32 values;
pooled rows narrow to BF16 before normalization. Index keys use 16-row padded
quantization and a single publication writer for INT8 codes and adjacent FP16
scales. Query keys are packed per request for QLI, and all 64 query heads read
the packed SWA and compressed physical indices. Shared selection publication
updates only current CP rows and retains the supplied history and tails.

The serialized BF16 cache observation uses independent snapshot storage.
The raw byte backing and its typed-view descriptors retain the physical cache
ABI. Index publication reuses the shared per-token quantizer with 16-row,
128-column tiles and direct FP32 division of 127 by the row maximum.

RoPE tables specialize plain or YaRN scaling from host scalar constants.
Positive-power evaluation and its reciprocal remain separate FP32 PyPTO operations.
QLI reuses its owning score and sort definition inline; padded key tiles are
initialized in FP16 and cast exactly to INT8. The current selection arena
supports at most 512 compressed rows per request. Longer Top512 pruning and
actual TP1/2 device compatibility remain separate acceptance items.

The L2 CLI reuses HC, CP/SP exchange, output projections, EP dispatch, experts
and combination from `prefill_layer.py`. The new cache/control adapter owns
memory views and scheduling in `prefill_compressed_runtime.py`; it does not
compute model arithmetic. Accepted L2 execution is recorded in the immutable
`a3_prefill_l3_required` regression baseline retained with the validation artifacts;
changed dependencies require the corresponding frozen process regression.

## Physical L3 consumer Prefill

`python -m models.deepseek_v4_1_flash_a2.prefill_layer3 --tp 4 --inputs INPUTS
--output OUTPUT` selects physical3 operands in the same complete-layer
orchestrator. The explicit `physical_layer_id` selects the input boundary;
the frozen bundle retains its inherited source2 `layer` header unchanged.
`--compile-only` constructs those entries through the public
queue without a device. TP1/2/4 shapes are parameterized at interpreter startup;
actual TP1/2 device compatibility remains unvalidated.

`prefill_consumer_runtime.py` schedules the explicit readonly source2 boundary
and physical3's own cache. It reconstructs the supplied physical byte backing,
expands the original shared state, and passes source-selected logical indices
to the owning sparse controls without filtering or sorting them again.
Compressed addressing retains 64-row logical pages in padded 128-row physical
pages. Own SWA addressing uses physical3's request tables and current slots.
The source2 byte span, compressed/index/ring/SWA views and all shared buffers
remain readonly. Current physical3 Q/KV, RoPE, own SWA publication, full-head
compressed attention, output projections, HC and shared/routed EP16 MoE execute
in the existing PyPTO owners; physical3 executes no compressor or indexer.

The `a3_prefill_l3_required` independent consumer gate retained with the validation artifacts
covers seven cases and 112 rank-cases, including true empty DP and audited own
SWA history at page127, window128 and tail boundaries. Publication observations
retain the product's computed BF16 rows and original slots. No device or
numerical acceptance is claimed before the exact frozen process succeeds.


## Complete network Prefill implementation

`prefill_network_runtime.run_network_runtime` composes all 40 physical
layers, final normalization, the vocabulary projection and TP vocabulary
gather. `prefill_network_state.py` owns the retained cache views;
`prefill_network_engram.py` owns Engram history and table preparation.
Inputs, checkpoint manifests, initial state and distributed transport are
provided by the caller. This is an implementation interface, not a bundled
checkpoint or a standalone text-generation service.

The A3 execution configuration used DP4/TP4/EP16 across eight physical
cards and sixteen logical devices. One diagnostics-disabled full40 run
completed with exit code zero, including actual physical layer 39 and the
final logits path. Its subsequent greedy first-token comparison matched
two of four requests. Whole-network numerical acceptance therefore remains
incomplete. A complete autoregressive answer and continuous Decode have
not been validated by this result.

The eleven independently delivered layers are 0, 1, 2, 3, 14, 15, 19, 20,
21, 24 and 25. Layers 0 through 3 used DP4/TP4/EP16; the seven later layers
used their frozen DP2/TP2/EP4 intermediate contracts. These per-layer
results do not establish whole-network accuracy, history, TP compatibility
or resource acceptance.

This publication includes the diagnostics-off host paths used by the
completed full40 execution; its PyPTO program and JIT definitions match the
registered numerical source. Experimental
shared-expert candidates with local exact results are not installed here.
Frozen tensor fixtures, private execution reports and machine-local
provenance pointers remain with their original validation records.

## Public complete-network forward and compilation reuse

`prefill_fwd.prefill_fwd` (also exported as `run_prefill`) accepts explicit
weights, request inputs, cache state, device mapping and output configuration.
It executes token embedding, the original forty layers and the vocabulary
head. `prefill_fwd.compile_prefill` compiles from storage-free geometry before
checkpoint reads or device allocation; its returned `PrefillProgram` can be
reused for compatible subsequent forwards. See
[`prefill.example.json`](../../../models/deepseek_v4_1_flash_a2/prefill.example.json)
for the prepared-input configuration.

The compiler pool deduplicates structurally equal IR only when parameter
names and ABI also match. Each layer composes five L2 compute regions between
existing host HCCL and expert-dispatch boundaries. This is not one device L2
for the entire forty-layer network. The established compute functions retain
their arithmetic. Legacy Decode helpers remain for existing callers and do
not establish complete-network Decode support.

`compiled/compilation.json` records unique compilations, stage bindings,
compiler seconds and preparation wall time. Per-chunk execution records keep
checkpoint read and operand preparation time separate. A ten-minute total
load-and-compile bound has not been established. The public forward defaults
to diagnostics disabled; explicit diagnostics retain state and readonly
checks. Disabling diagnostics never establishes numerical acceptance.

### Continuous Prefill chunks

For a single request containing 1..2048 tokens, pass a token-ID list or raw
completion text to the same public entry:

```python
result = run_prefill(
    token_ids, weights=checkpoint, cache=cache_state, config=network_contract,
    devices=devices, output="build_output/ds41-prefill-2048",
)
```

The default is `chunk_size=16` and `dp_rank=0`. Set these fields in the Python
`config` to select chunks of 1..16 tokens or DP engine 0..3. All other DP
engines are empty, and all 16 ranks remain represented. A 2048-token request
produces 128 chunks at the default size, retaining earlier cache/history state.
This is sequential chunked Prefill; it does not make the executable query
capacity 2048 or implement Decode, sampling or an autoregressive text loop.
The final result points to the last chunk's output-head artifacts.

For the CLI, replace `inputs` with either `prompt` or `token_ids`, omit
`continuations`, and optionally set `chunk_size` and `dp_rank`:

```json
{
  "checkpoint": "weights/deepseek-v4.1-flash-w8a8",
  "prompt": "Please continue this passage.",
  "chunk_size": 16,
  "dp_rank": 0,
  "cache_state": "inputs/cache-state.pt",
  "contract": "inputs/network-contract.json",
  "devices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  "output": "build_output/ds41-prefill-2048"
}
```

Text uses the checkpoint's `tokenizer.json` through `tokenizers`; it does not
apply a chat template. Supply token IDs for an already formatted conversation.
The limit applies after tokenization, and oversized prompts fail rather than
being truncated. The model configuration must match the selected checkpoint.

Automatic preparation still needs the 16 original `cache_ABI` templates
(`layout`, `groups`, `hash_constants`) and both TP4 Engram contracts in
`cache_state`. The templates describe physical storage and model constants;
their previous token coordinates and cache contents are not imported. Preparation
reserves fresh pages for the entire prompt once, preserving storage keys,
typed-plane offsets, strides and aliases. It creates each chunk's positions,
query starts, sequence/cache lengths, SWA/C1/C2 slots, C2 RoPE source positions,
input lineage and exact state-write complements. The numerical configuration,
required modes and deterministic initial TopK state remain explicit in
`network_contract`; per-input hashes and state regions are regenerated.

Prepared files are written under `output/prepared/chunk-N`; execution files
are under `output/network`. Later chunks are prepared only after the preceding
chunk completes all 40 layers. Embedding uses the existing PyPTO input stage.
This does not change the network weight-loading or resident-weight policy.

The Indexer searches up to 2048 cached keys. Ratio2 layers therefore see up to
1024 completed keys for a 2048-token prompt, while ratio1 layers see 2048.
TopK remains 512, and attention retains its 128-window plus 512-compressed
slots. Causal masking happens before TopK selection, followed by chronological
index preparation. Layer20 publishes all visible eight-key candidate blocks
(at most 256 here), and layer24 retains its full-coverage guard. Longer histories
requiring candidate pruning, request eviction and multi-request automatic
scheduling are outside this bounded path.


### Shared-expert precision and compiler reuse

The fused shared expert retains FP32 channel scaling through gate/up and down
rather than inserting the routed expert's FP16 channel boundary. The routed
expert keeps its previous default, and standalone BF16 activation is unchanged.
The selected repair comes from candidate
`34dd9543a59808bb07e55e3efa11c29dfa036361`; its original localized A3 report
matched 10240/10240 elements bit for bit. The alternative candidate is not
stacked with it. This localized result does not establish repaired full-layer
or complete-network accuracy on this publication branch.

The network compiler validates model constants, topology, cache layout and
Engram geometry before reusing a program across chunks. Unchanged chunks
reuse stage bindings directly without reconstructing and specializing all
forty layers. Changed compile-time geometry is rejected before execution.
`compile_seconds` measures PyPTO IR/PTO/C++ generation; device binary building
and first-call initialization belong to the runtime and are not included in
that number. Checkpoint loading and complete first-token latency must be
measured separately on the allocated device group. Neither CPU code generation
nor the IR deduplication count establishes a ten-minute end-to-end bound.

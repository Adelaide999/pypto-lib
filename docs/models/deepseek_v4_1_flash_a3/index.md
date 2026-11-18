# DeepSeek V4.1 Flash A3 Prefill and Decode

A3 Eco-Tech W8A8 operators, the original 40-layer Prefill composition and
persistent generation session, bounded Decode operators, and a shared TP4
40-layer network for Prefill and one-token Decode. See [Decode](decode.md) for its implementation scope and
remaining work.

See [Native vLLM-Ascend accuracy comparison](accuracy.md) for fixed
TP4/DP4/EP16 prompt/Decode traces, full-vocabulary captures and CPU metrics.
See [Experimental packed requests and paged history](dynamic-runtime.md) for
request admission, independent limits, TP plans, HBM estimates and device validation status.

## Development priorities

- Performance target: complete-network correctness and continuous generation
  with generation-scoped workers, compute weights, caches and communication,
  retaining Eco-Tech real W8A8, 384 routed experts, Top6 and original Engram state.
- Deferred performance work: fused device layer scheduling and final throughput
  tuning. Residency is an active target; elapsed-time improvements require
  measurements on an allocated device group.
- Compatibility requirements: preserve original layer roles, byte aliases,
  cache/source lifetime, quantization, dtypes, and explicit DP/TP/EP mappings.
  TP1/TP2 full-network device compatibility remains unvalidated.

## Structure

| Responsibility | Owner |
| --- | --- |
| Shared network compilation, execution and session | `network.py` |
| Whole-network Prefill and request preparation | `prefill_fwd.py` |
| 40-layer Decode schedule, resident weights and state | `decode_fwd.py --network`, `decode_network.py` |
| Generic layer composition and L2 bindings | `prefill_layer.py`, `prefill_layer_plan.json` |
| Bounded SWA Decode | `decode_fwd.py` |
| Vocabulary embedding, initial HC and output head | `input_pack.py`, `prefill_fwd.py`, `lm_head.py` |
| Attention, compression and Indexer | `swa.py`, `sparse_attention.py`, `compressor.py`, `indexer.py`, phase compositions |
| Phase MoE routing, dispatch and shared experts | `prefill_moe.py`, `decode_moe.py` |
| Full-width Decode layer dispatch and attention | `decode_layer.py`, `decode_swa.py`, `decode_c1a.py`, `decode_c2a.py` |
| Full-width Decode expert-parallel composition | `decode_moe_ep.py` |
| Resident blocked-INT8 grouped experts | `grouped_moe.py` |
| Bounded Decode NZ routed experts | `expert_routed.py` |
| Linear, quantization, Norm, RoPE and HC | `gemm.py`, `quant.py`, `rmsnorm.py`, `rope.py`, `hc.py` |
| Checkpoint operands and phase layouts | `weights.py` |
| Cache ownership and device lifetime | `cache.py`, `state.py`, `prefill_resident.py` |
| CANN/HCCL communication | `moe_transport.py` |
| Host Engram tables and row preparation | `engram_host.py` |
| Device Engram projection and gating | `engram.py`, `prefill_network_engram.py` |

The package keeps Prefill and bounded Decode owners flat. Full Decode network
leaves and orchestration helpers use the `decode_` prefix alongside those
owners; there is no nested Decode implementation package. The
[ownership map](../../../models/deepseek_v4_1_flash_a3/MODULE_ALIGNMENT.md)
describes shared primitives and phase-specific numerical variants.
Historical audits, independent-layer runners, reference-capture workers,
state replay, tensor collectors and identity-lock manifests are removed.
Their source remains in Git history. Operators retain their local Torch
goldens and standard Golden Harness entries.

MoE phase files own their kernels, Torch goldens and validation entries.
Shared GEMM and quantization primitives remain in independent child modules;
the phase modules do not import each other. Prefill controlled routing and
FP16/FP32 fused-channel variants remain distinct from Decode FP32 router
inputs and activation boundaries. Standalone validation uses the same kernels
as composition:

```bash
python -m models.deepseek_v4_1_flash_a3.prefill_moe -p a2a3 --compile-only
python -m models.deepseek_v4_1_flash_a3.decode_moe -p a2a3 --compile-only
```

These commands compile bounded operator cases; they do not execute a network.
Omit `--compile-only` on an allocated A3 device to run operator comparisons.

## Operator CPU goldens

Operator files own their Torch `golden_fn`, `TensorSpec` fixtures and
`golden.run(..., compare_fn=...)` entries. CPU references reuse child
operator references; they do not call the NPU runtime or CANN. PyPTO and
CPU-capable PyTorch are still required to import the operator definitions.

| Owner | Reference and bounded coverage |
| --- | --- |
| `input_pack.py`, `qkv_proj_rope.py` | Embedding, initial HC and BF16 projections |
| `hc.py`, `swa.py`, `indexer.py` | HC mixing, paged attention and discrete selection |
| `prefill_moe.py`, `decode_moe.py` | Phase-specific 384-expert/Top6 routing and W8A8 shared experts |
| `engram.py` | Group32 INT8 lookup, Prefill hash/layout/gate and Host-prepared Decode rows |
| `decode_swa.py` | Full-width HC, RMSNorm, W8A8 QKV/RoPE, paged SWA, inverse RoPE, BF16 output projections and residual mixing |

The full-width SWA entry uses the existing `make_attention_kn_block` kernel
and `golden_swa`; `build_swa_specs` constructs its KN weights, RoPE values,
page-edge history, skipped publication slots and all-masked attention row.
The default cases have 0, 1, 7 and 16 active rows at capacity 16. Numerical
outputs retain `rtol=0.05, atol=0.05` with no failing-element allowance.
Inactive InOut rows, positions, cache lengths and unmapped cache rows remain
exact; the output projection explicitly clears its inactive output rows to zero.
Only published cache rows use numerical comparison, through the harness's
mapped-pool comparator.

The Engram `decode-rows` entry validates the existing `make_decode_engram`
kernel with `golden_decode_engram_rows` and `build_decode_engram_specs`.
The Host has already looked up and dequantized the 24 table entries into
6144 BF16 values per token. Those values enter the kernel as `lookup`;
the kernel performs projection and original-basis gating. Its golden never
moves a whole table into the kernel or performs a device lookup. Masked
residuals and inactive rows are exact. The separate `decode-lookup` entry
retains bounded legacy lookup coverage with the actual 256-value table row
width and eight group32 scales.

Compile or simulate the same operator entry without allocating an NPU:

```bash
python -m models.deepseek_v4_1_flash_a3.decode_swa -p a2a3 --active 7 --compile-only
python -m models.deepseek_v4_1_flash_a3.decode_swa -p a2a3 --active 7 --sim
python -m models.deepseek_v4_1_flash_a3.engram -p a2a3 --component decode-rows --active 7 --sim
```

To evaluate only the Torch reference, construct values from the owning specs
and call its golden directly. This does not compile or execute a kernel:

```python
from golden import TensorSpec
from models.deepseek_v4_1_flash_a3.decode_swa import build_swa_specs, golden_swa

values = {
    spec.name: spec.create_tensor().clone()
    if isinstance(spec, TensorSpec) else spec.to_python()
    for spec in build_swa_specs(active=7)
}
golden_swa(values)
```

The native SWA and Engram Decode-row entries accept `--save-data`,
`--golden-data` and `--runtime-dir` for the standard harness workflow:

```bash
python -m models.deepseek_v4_1_flash_a3.decode_swa --active 7 --sim --save-data
python -m models.deepseek_v4_1_flash_a3.decode_swa --active 7 --sim \
  --golden-data build_output/<compiled-case>/data
```

Use the data directory printed by the first run. Replay requires an explicit
active-row case and matching shapes, dtypes, scalar controls, weight layout
and source contract. A standard harness `data/out` directory contains Torch
expected outputs; it is not an actual NPU readback. Historical actual outputs
are reusable only with every corresponding initial input buffer and the same
operator ABI. Logs, token IDs or logits alone do not reconstruct these inputs.
CPU reference evaluation and simulator comparisons do not establish physical
NPU accuracy, distributed communication, whole-network accuracy or performance.

## Whole-network Prefill

`prefill_fwd.prefill_fwd` executes embedding, initial HC, layers 0 through 39,
final HC collapse, RMSNorm and the vocabulary head. Each layer consumes the
previous layer's actual activations. The default DP4/TP4/EP16, PP1/PCP1/DCP1
path retains Prefill capacities up to 8192 and Decode capacity 16. The separate
experimental packed path adds DP8/TP2/EP16 and DP16/TP1/EP16 plans and supports
new-token budgets up to 512 in its entry points. See the packed runtime guide
for the limits of device coverage. Physical devices are explicit and ordered
by logical rank. Duplicate IDs, rank-count mismatches and a
conflicting `ASCEND_RT_VISIBLE_DEVICES` mapping are rejected before initialization.

```python
from models.deepseek_v4_1_flash_a3.prefill_fwd import compile_prefill, prefill_fwd

program = compile_prefill(
    inputs, cache=cache_state, config=network_contract,
    output="build_output/ds41-compiled", weights=checkpoint,
)
result = prefill_fwd(
    inputs, weights=checkpoint, cache=cache_state, config=network_contract,
    devices=list(range(16)), compiled=program, output="build_output/ds41-run",
)
logits = result["logits"][0]["fresh"]
```

For automatic inputs, `inputs` is raw completion text or 1..16384 token IDs.
The entry tokenizes with the checkpoint tokenizer, reserves pages for the
complete prompt and processes chunks of up to 8192 tokens. `chunk_size` can
select 1..8192 and `dp_rank` can select 0..3. Other DP engines remain empty.
Positions, sequence lengths, page tables, slot mappings, C2 cross-chunk state
and Engram history persist across chunks. Indexer searches up to 16384
compressed keys, keeps TopK 512 and excludes future positions. This automatic path
handles one request; it is not a multi-request serving scheduler.

`cache_state` supplies canonical cache-layout and both Engram templates.
A checkpoint and prompt alone do not define cache aliases or source ownership.
The cache dictionary contains `cache_ABI`, `cache_ABI_sources` and
`engram_contracts` with integer slots 1 and 14. The network contract contains
`topology`, `model_config`, `required_modes`, `capacity`, `scheduler_capacity`,
`fresh_shared_initialization`, `input_contract_path` and
`input_contract_sha256`. Prepared dictionary inputs contain `input_contract`.
These are request/storage interfaces, without private task admission receipts.
Embedding always runs in the forward; captured backbone activation files
are no longer accepted as an alternative input path.

### Compilation and execution

`compile_prefill` specializes canonical metadata with storage-free meta tensors.
It compiles all stages without allocating full caches, initializing CANN or
executing kernels. Automatic inputs require `weights=checkpoint` for checkpoint
metadata. A `PrefillProgram` is reusable within the same process; structural
equality, argument names, shapes, constants and cache geometry must match.
`compilation.json` records unique entries, stage bindings and compiler time.
Omitting `compiled` compiles inside the forward.

| L2 region | Existing computation |
| --- | --- |
| Input | Embedding summation and initial SP/HC streams |
| QKV | Q/KV projection, normalization and RoPE |
| Attention packing | Inverse RoPE and head packet packing |
| Output projection | Head unpack, grouped projection and reduction packing |
| FFN dispatch | HC, normalization and router |
| MoE slabs | Bounded source slabs, EP dispatch, fused unpack/experts/return packing, and combine |
| MoE finish | Valid-row masking and FFN HC post |
| Head | SP unpack, RMSNorm and vocabulary projection |

L2 denotes PyPTO orchestration, not physical layer 2. The network retains host
scheduling and HCCL boundaries; it is not a single fused 40-layer device graph.
Intermediate activations remain in device allocations across these boundaries.
CPU tensors in the scheduler describe shapes, storage aliases and initial values;
a written activation is authoritative only on the device. Host arithmetic or
cloning must not consume its stale descriptor. Contiguous activation padding and
request stitching use device-to-device copies. Decode reuses one hidden/pre-mix
boundary per rank, while Prefill alternates two boundaries. Completed layer
scratch is recycled without releasing live cache or boundary aliases.

Activation HCCL collectives consume the producer device pointer and write
directly to the consumer device allocation without CPU payload staging buffers.
Only startup checks, integer routing/index/status controls and final vocabulary
outputs are read back. Finite checks run on A3 and return integer row flags;
route-capacity and sparse-page validation still run on the explicit controls.
Runtime accounting separates `control_d2h_bytes`, `output_d2h_bytes`,
`activation_d2h_bytes` and `d2d_bytes`. Complete forwards keep
`activation_d2h_bytes` at zero; this does not imply zero control traffic or
establish a latency improvement.
No loading, compilation or token-latency guarantee follows from this organization.

Execution uses persistent ChipWorkers, preloaded compute weights, cache
allocations and HCCL groups for the complete stream. Input/output entries and both generation phases share the same worker
registry and allocation owners.
The single-process runtime creates EP, TP and DP communication domains with
`HcclCommInitAll`, preserving logical-device order within each group.
The streaming-weight diagnostic alternative and task-directory compiler reuse
manifests are removed. Runtime dtype, shape, rank, alias, write-direction,
source-version and lifecycle checks remain in force.

If a synchronous device run fails, its transient allocations remain owned by
the public worker until worker teardown drains or reclaims the failed run.
Cleanup failures are logged without replacing the original dispatch error.
After a completed layer, scratch allocations return to the session pool.
Session teardown frees each owning allocation once; device aliases do not own
additional buffers.

### Concurrent execution and bounded MoE storage

The generation command defaults to `--chunk-size 128`. Each request uses the
smallest fitting capacity from 16, 128, 256, 512, 1024, 2048, 4096 and 8192,
including its partial final chunk. Decode always uses capacity 16. Both shapes are compiled
before execution and share static weights, cache backing storage, workers,
and Engram history. The scheduler capacity is 8192; the context bound is 16384,
including reserved generation positions. An 8192-token prompt can therefore
continue through all seven Decode steps needed for eight output tokens.
Shared index and candidate publication buffers retain their 8192-row ABI
when execution capacity switches from Prefill to 16-row Decode.
Long-history Indexer selection sorts at most 2048 keys per tile and merges
the tile results into the global Top-512. Ratio-1 layers retain one key per
context token; ratio-2 layers retain one key per completed token pair.
Candidate pooling covers all 2048 eight-key blocks while retaining the
original 2048-entry publication ABI.

Each rank has one persistent compute queue. Independent ranks are submitted
before the stage joins; owned experts execute in order within their rank.
Communication uses separate persistent threads. The first EP slab dispatch
runs alongside the shared expert, and its completion future is joined before
unpack. All ranks execute the same number of slab collectives, including
empty ranks. Slab token IDs are local to the current slice; combine writes
into the original SP slice at the corresponding slab offset.

Set `--prefill-moe-slab-rows` on the generation command, or
`prefill_moe_slab_rows` in the Python forward configuration, to 16, 64 or
128 to cap the number of source rows per rank in each round. The default cap is 64 at
Prefill capacity 8192 and 16 for the smaller capacities. The compiler rounds
the initial valid SP row count up to a power of two and caps it at that limit.
The per-peer bucket holds `slab * TopK` rows.
Per-expert storage rounds `min(DP * capacity, EP * slab)` up to a multiple of
16. Later admissions reuse these bounds and execute as many rounds as their
physical SP rows require; all ranks retain the same collective count.
The separately compiled one-token Decode phase uses a one-row slab, a 960 KiB
hidden packet per rank and direction, and 16 rows per expert. A full 128-token Prefill chunk retains
16-row slabs by default, 15 MiB packets and 256 rows per expert. For a single
8192-token request on TP4/DP4/EP16, the busiest source rank has 2048 rows:

| Slab cap | Rounds per layer | Dispatch and return stages, 40 layers | Hidden packet per rank | Reserved rows per expert |
|---:|---:|---:|---:|---:|
| 16 | 128 | 10240 | 15 MiB | 256 |
| 64 | 32 | 2560 | 60 MiB | 1024 |
| 128 | 16 | 1280 | 120 MiB | 2048 |

Wide packet copies use complete hidden rows to fit the A3 SPMD block count.
The runtime sizes the grouped-expert scratch heap from the compiled number
of expert rows and checks received route counts before unpack. Shared computation
retains the phase capacity. These are communication workspace bounds, not
total HBM measurements.

DP groups with no tokens skip local attention normalization, QKV, indexing,
cache arithmetic and output projection. They supply initialized zero packets
to the unchanged TP collectives and retain cache-version publication and EP
expert participation. A TP lane without a local query still executes its KV
work when its DP group has tokens.

SP shards with no valid source tokens also skip FFN routing, shared experts,
source packing, combine and FFN HC output computation. Their physical output
rows are zeroed and their packet counts remain zero. Every EP rank still
receives routes; a nonempty received bucket runs unpack, grouped experts and
return packing in one L2 region, even when that rank's DP group is empty.
Received route metadata is checked against the per-expert capacity before
the fused region executes. Ranks with no received routes skip that region.

Device collectives enqueue every packet field in the same order on each
rank's communication stream and synchronize once after the batch. Source and
destination allocations remain owned until the batch completes or failure
cleanup drains the stream. MoE routes, hidden payloads and counts retain their
separate tensor layouts and collective calls.

Intermediate CPU tensors describe device-owned storage. Activations and HCCL
payloads stay on device; integer controls and final logits are explicitly read
back. Attention and MoE scratch descriptors and device allocations return to a
rank/shape/dtype workspace pool after each layer. Reuse resets zero and integer
`-1` padding directly on device through synchronous CANN byte fills, without
creating initialized CPU payloads or uploading them. Host-prepared metadata,
static weights, persistent caches and layer-boundary buffers retain their
existing ownership. Other completed allocations return to the size-based pool.
Host descriptors must not be mutated after the device becomes authoritative.

Indexer codes and scales transfer all selected owner pages in one invocation
per plane and direction. The device copy consumes a bounded page-ID list;
unselected pages and bytes outside the typed payload remain unchanged. Prefill
and Decode use the same copy implementation and retain the original page,
reserved-page and cache-alias checks.

The generation command defaults to `--logits-mode last`: final normalization
and vocabulary projection consume only the last active position of each DP
engine. A 16-row device buffer retains the GEMM alignment; only its valid row
is returned to the host. One TP rank unpacks and reads the full vocabulary,
and the host result is replicated across the TP result slots. Empty DP groups
return empty tensors. Each Prefill chunk still computes its final-position
distribution, and Decode uses the same output policy.

Logit tensor files are opt-in for the generation command with `--dump-logits`.
Use `--logits-mode all --dump-logits` to retain every position and the full
diagnostic files. The Python forward API retains `logits_mode="all"` and
`dump_logits=True` when these keys are absent from `config`; set them explicitly
to select compact output. The compiled program freezes the output mode, which
cannot change during a generation session.

Bounded byte-copy and output-selection checks use the shared Golden Harness
and compare both written values and untouched regions exactly:

```bash
python -m models.deepseek_v4_1_flash_a3.cache -p a2a3 -d 0
python -m models.deepseek_v4_1_flash_a3.input_pack -p a2a3 -d 0 --output-unpack
```

Run these device commands through the site's device allocator.

Runtime receipts include per-rank transfer totals, workspace requests, reuse
counts and device fill bytes. Device fill bytes are separate from H2D bytes;
the aggregate transfer counters and their per-rank sums share one lock.

Per-chunk `execution.json` schema 2 stores dynamic timings and communication
receipts. Its `static_audit` references include paths and SHA256 digests for
the generation's `weight-audit.json` and the phase program's
`execution-audit.json`. Checkpoint reads and host weight accounting are
written once per generation; full compiled layer bindings and compiler
receipts are written once per phase. The referenced files retain the complete
audit data without repeating it in each token report.

The pinned PyPTO worker retains at most 64 callable registrations. Completed
registrations are retired through the underlying simpler worker before a slot
is reused; compiled artifacts, weights and cache allocations remain live.
This integration uses the pinned wrapper's `_cid_cache` and `_impl` fields;
recheck it when updating PyPTO. A device smoke verifies retirement followed by
another dispatch on the same workers and allocations.

`generation/result.json` records initialization, complete Prefill forward,
each Decode step, and time to first token. These are host wall times, including
dispatch, communication and execution receipt I/O. The first-token interval
starts after argument parsing and includes contracts, host table preparation,
compilation and weight initialization. Per-chunk `host-trace.json` records
rank stages, synchronous worker calls and collective completion intervals.
Host intervals are not a substitute for aligned device swimlanes.

This change is a performance experiment. Numerical comparison, boundary-matrix
validation and full-network vLLM acceptance are not established by execution.

### Fixed INT8 weight layout

Initialization packs QKV, Indexer query and Prefill shared/routed expert
weights on the CPU before uploading their final compute operands. The shared Prefill/Decode network accepts the same blocked operands.
For logical `[K, N]` INT8 weights, the layout is `[N/64, K, 64]`, flattened
to `[K*(N/64), 64]`. `utils.pack_blocked_weight` performs only a byte reorder.
`prepare_ecotech_dynamic_weight` and the expert loaders return this packed
shape; callers must not pass the original `[K, N]` matrix to these builders.
QKV channel scales retain the existing BF16-to-FP32 conversion, while expert
loaders retain their original FP32 scales. At TP4, the shared expert prepares
four independently padded 576-channel groups. Routed experts apply token-first
FP32 Gate-Up scaling and FP16 activation requantization. See
[numerical boundaries](accuracy.md#numerical-boundaries) for the generation
network's shared and routed quantization rules.

The QKV and shared-expert Torch reference interfaces accept these loader
outputs directly and unpack only for reference computation. The QKV references
also accept explicit `weight_layout="kn"` for the bounded Decode direct-tile
entry. Shared-expert references reconstruct each intermediate group from
the packed down operand and apply its configured activation rounding. Routed-expert
references continue to take the documented output-major checkpoint tensors;
their separate NZ or blocked device operands are not interchangeable.

`ResidentNetworkWeights.preload` prepares these operands once per generation.
Layer and chunk calls reuse the pinned projection tensors and each EP owner's
resident expert handles, without repacking or uploading the fixed weights.
Each rank retains four expert-major weight owners and per-expert borrowed
views of those same bytes. After each dispatch slab is unpacked,
`grouped_moe.py` computes all nonempty local experts in one invocation.
Empty ranks skip the routed invocation and its weight lookup.
Activation quantization and routed-token dispatch packing still run per call.
The generation network reuses the blocked routed-expert representation in
both phases. Bounded legacy Decode operators retain their FRACTAL_NZ contract,
including down projection padding. The bounded `decode_fwd` direct-tile GEMMs retain logical
`[K, N]` weights and do not perform a whole-weight pack. A5 FP8/FP4 layouts are
not used by these A3 W8A8 kernels.

Changing this operand ABI requires fresh compilation and reloading weights;
compiled programs created for the former canonical-weight ABI are incompatible.
Device accuracy and latency require separate A3 validation; CPU byte checks
and compilation alone do not establish a speedup.

### Engram and continuation

Engram lookup runs on the host before the prepared forward call. `HostEngram`
owns both complete CPU INT8/FP32 tables, hashes request history, gathers selected
entries and applies group32 dequantization with BF16 rounding. Device kernels
consume only rank-local `[capacity, 6144]` BF16 rows for projection and gating.
The NPU does not dereference the host table. Tables are not registered with
CANN or counted as full HBM model residency.

Automatic inputs and the CLI prepare these rows. For prepared Python inputs:

```python
from models.deepseek_v4_1_flash_a3.engram_host import HostEngram

with HostEngram(checkpoint, cache_state["engram_contracts"], "generation") as host:
    rows = host.lookup(inputs, cache=cache_state, config=network_contract)
    result = prefill_fwd(
        inputs, weights=checkpoint, cache=cache_state, config=network_contract,
        devices=list(range(16)), output="build_output/ds41-host-rows",
        engram_lookup=rows,
    )
```

Each prepared continuation supplies fresh `inputs["engram_lookup"]`, its next
epoch and predecessor input identity, and ordered metadata sources for all
ranks. The cache provider retains allocations and publishes layer/source
versions. Failed or completed chunks cannot be replayed in place.

## Commands

Copy [the configuration example](../../../models/deepseek_v4_1_flash_a3/prefill.example.json)
and supply checkpoint, request/cache templates, contract, devices and a new
output directory. JSON paths resolve relative to the configuration file.
For automatic inputs, replace `inputs` with `prompt` or `token_ids` and
optionally add `chunk_size` and `dp_rank`.

```bash
python -m models.deepseek_v4_1_flash_a3.prefill_fwd --config prefill.json --check-config
python -m models.deepseek_v4_1_flash_a3.prefill_fwd --config prefill.json --compile-only
python -m models.deepseek_v4_1_flash_a3.prefill_fwd --config prefill.json -p a2a3
```

`--checkpoint`, `--devices` and `--output` override the JSON settings.
`--check-config` checks only configuration and device mapping. Compile-only
does not execute NPU work. Output includes final logits, compilation accounting,
execution progress and failures, without per-layer reference-comparison captures.

`generate.py` runs Prefill and one-token Decode in one DP4/TP4/EP16
session with one static weight bank and the original cache allocations:

```bash
python -m models.deepseek_v4_1_flash_a3.generate \
  --checkpoint weights/deepseek-v4.1-flash-w8a8 --prompt "Hello" \
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --max-new-tokens 3 --ignore-eos --output build_output/ds41-generation
```

For batch size 1, 8192 input tokens, one 8192-token Prefill chunk and eight
output tokens, supply a JSON array containing exactly 8192 token IDs:

```bash
python -m models.deepseek_v4_1_flash_a3.generate \
  --checkpoint weights/deepseek-v4.1-flash-w8a8 \
  --tokens-file prompt-8192.json --devices "$TASK_DEVICE" \
  --chunk-size 8192 --max-new-tokens 8 --ignore-eos \
  --output build_output/ds41-generation-8k
```

Run this command in a job allocated 16 devices. Only DP rank 0 receives the
request; every EP rank participates in expert execution. The first output
token comes from Prefill and the remaining seven from Decode. The 8K chunk
is a single network admission, not a sequence of 128-token forwards. MoE
continues to use bounded internal communication slabs. This configuration
increases activation and full-vocabulary output storage; it does not change
Engram's CPU lookup placement or reduce the model's expert count.

The resident runtime sizes ring 1 for active-query BF16 sparse gathers and
full-capacity HC/shared-expert intermediates, and ring 2 for mixed-kernel
pipeline buffers including index selection. Required sizes include 1 MiB
of additional space and are rounded to runtime-supported powers of two.
Capacity 8192 with TP4 uses 2 GiB on ring 1 and 1 GiB on ring 2; other rings
retain their defaults. These settings apply with profiling enabled or disabled
and remain in effect through Decode. Each TP rank gathers at most 2048 query
rows, requiring 1280 MiB; shared-expert intermediates require about 1337 MiB.
MoE communication receive buffers are recycled after each completed round.
All ranks allocate collective buffers before any rank submits the collective.
Ring heaps are additional to weights, KV caches and host tables.
Wide embedding, packing, routing, attention, projection and output launches
use at most 256 SPMD workers, with each worker iterating over its assigned
blocks. This bounds the runtime's 16-bit subtask counters and pipeline memory
while retaining one complete network admission for the 8K chunk.

The command derives its cache and Engram contracts from checkpoint metadata,
reserves prompt and output pages once, and feeds each predicted token into a
single-token Decode step. It retains the same workers, static handles, cache
pools and request history until the session closes. Engram projection/gating
and the vocabulary head run on A3; dynamic Engram lookup rows are prepared
from the persistent host tables. See [Decode](decode.md) for standalone
commands and the read/upload/cache-alias audit fields.

## Network profiling

The persistent generation runtime supports optional chip swimlane capture
across embedding, all 40 layers and the vocabulary head. The same controls
apply to Prefill and Decode, including cached resident dispatches:

```bash
python -m models.deepseek_v4_1_flash_a3.generate \
  --checkpoint weights/deepseek-v4.1-flash-w8a8 \
  --tokens-file prompt-8192.json --devices "$TASK_DEVICE" \
  --chunk-size 8192 --max-new-tokens 8 --ignore-eos \
  --enable-chip-swimlane 1 --profile-phase all \
  --output build_output/ds41-generation-8k-profile
```

Run on 16 allocated devices. The capture starts after compilation and static
weight initialization. It observes each actual dispatch once and retains the
same weight, KV cache and Engram state throughout generation.

| Option | Behavior |
| --- | --- |
| `--enable-chip-swimlane 0` | Default; no device trace files are created. |
| `--enable-chip-swimlane` or `--enable-chip-swimlane 1` | Capture AICore task timing. |
| `--enable-chip-swimlane 4` | Include detailed runtime scheduler records. Levels 2 and 3 are also passed through to PyPTO. |
| `--profile-phase all` | Default; capture both Prefill and Decode. |
| `--profile-phase prefill` / `decode` | Capture only the selected phase on the persistent runtime. |
| `--profile-output PATH` | Fresh capture root; defaults to `OUTPUT/profiling`. Existing roots are rejected. |

Python callers of `prefill_fwd` can set the same controls in the network
configuration before creating the resident session:

```python
config["profiling"] = {
    "level": 1,
    "phase": "all",
    "output_dir": "build_output/ds41-device-profile",
}
```

For the `prefill_fwd --config` entry, place this object in the referenced
network contract JSON. The retained Decode session inherits these settings.
If `output_dir` is omitted, the Python entry uses its output directory's
`profiling` subdirectory. Settings are fixed for that session.

The output structure separates phases, chunk epochs, logical ranks and
invocations. Each rank index also records its physical device ID:

```text
profiling/
  index.json
  symbols/program-*/name_map_*.json
  prefill/chunk-000000/
    index.json
    rank-00/
      index.json
      call-000000/
        chip_swimlane_records.json
        dispatch.json
      call-000001/...
    rank-01/...
  decode/chunk-000001/rank-00/call-000000/...
```

Every chunk has an index for every allocated rank, including empty ranks.
A zero-active dispatch may emit no device records; its `dispatch.json` then
has `raw_trace: null` and `capture_status: "not_emitted_zero_active"`.
The output mask also has no device tasks when every physical row is valid;
an absent trace is then marked `not_emitted_empty_dispatch`. These calls still
execute once. A missing trace from other completed dispatches fails capture and
lists the affected calls in the chunk index. Failure cleanup preserves partial
captures as `incomplete`. Reuse a new output directory for the next run.

Raw traces are collected during execution. Convert selected calls offline in
the same activated PyPTO environment using the command stored in each
`dispatch.json`. For example, convert every captured call for Prefill rank 0:

```python
import json
import subprocess
from pathlib import Path

root = Path("build_output/ds41-generation-8k-profile/profiling")
for path in sorted(root.glob("prefill/chunk-*/rank-00/call-*/dispatch.json")):
    record = json.loads(path.read_text())
    if "convert_command" in record:
        subprocess.run(record["convert_command"], check=True)
```

Open a resulting `call-*/swimlane.json` in Perfetto. Each file represents one
invocation on one card. This first version does not stitch calls or align
clocks across cards; `cross_rank_clock_alignment` is explicitly false in the
capture summary. Dependency-arrow capture is not enabled by these controls.

`generation/result.json`, per-chunk `execution.json` and `runtime_config.json`
reference capture coverage and indexes. Device traces cover PyPTO compute;
CPU Engram lookup and HCCL device kernels are outside their scope. The existing
`host-trace.json` retains host dispatch and collective intervals. Profiling
adds instrumentation and file I/O to forward execution, so disable it when
comparing the existing host-wall-time reports with vLLM-Ascend.

## Correctness scope

See [Indexer and MoE operator timings](kernel-timings.md) for isolated A3
measurements and the corresponding baseline and shapes.

The generation network shares its Prefill and Decode numerical boundaries,
including native TP shared-expert partitions and routed-expert rounding;
see [accuracy comparison](accuracy.md#numerical-boundaries). Cache layouts and
ownership remain unchanged.
Operator-local checks use the [Golden Harness](../../run-and-validate/golden-harness.md).
CPU goldens, imports and IR specialization do not establish NPU accuracy.
Complete-network accuracy, dataset scores, full-model HBM residency and
performance still require device evidence. Serving scheduler integration and independent full-network accuracy
acceptance remain outside this generation entry.

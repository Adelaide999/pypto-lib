# DeepSeek V4.1 Flash A3 Prefill

This A3 implementation contains real PyPTO operators and the runtime wiring
for the original 40-layer text backbone. The exported product snapshot is
`ebf19ad0ae7d668b4f7f2b04b526c9dcff2785a9`. Its historical source directory
was `models/deepseek_v4_1_flash_a2`; this delivery uses the separate
`models/deepseek_v4_1_flash_a3` package and actual A3 task sources.

## Development priorities

- Performance target: correctness of the A3 complete Prefill path, with
  Eco-Tech real W8A8, 384 routed experts, Top6, and original Engram state.
- Deferred performance work: final throughput and latency tuning until
  complete-network accuracy is independently established.
- Compatibility requirements: preserve original layer roles, byte aliases,
  cache/source lifetime, quantization, dtypes, and explicit DP/TP/EP mappings.
  TP1/TP2 fixed-16 full-network compatibility remains unvalidated.

## Evidence and scope

| Item | Evidence scope |
| --- | --- |
| Original A3 independent complete layers | 11/11: physical layers 0, 1, 2, 3, 14, 15, 19, 20, 21, 24, 25 |
| Independent-layer admitted source | `afef9572cfe9c518f4e60d5ea7b00368b7297fbb` |
| Early four layers | DP4 / TP4 / EP16, PP1 / PCP1 / DCP1 |
| Late seven layers | DP2 / TP2 / EP4, PP1 / PCP1 / DCP1 |
| Exported latest network implementation | Full original 40-layer composition, DP4 / TP4 / EP16, capacity 16 |
| Full-network accuracy and actual L39 output | Not established at export; original task validation was in progress |
| Fixed-16 TP1/TP2 full-network compatibility | Not established |
| Full real-weight and both Engram tables in HBM | Not established by code or host weight-byte counts |
| Dataset score / Decode / serving integration | Not established by this delivery |
| Relocated package device execution | Not run; HOST checks do not replace NPU validation |

Independent alignment uses the same complete-layer inputs and legal source/cache
state for product and reference. Floating tensors retain the original
`rtol=0.05, atol=0.05` criterion; discrete indices, controls, and untouched state
remain exact. An independent-layer pass does not imply the original 40-layer
chain passes. In particular, late-layer EP4 results do not establish EP16
full-network correctness.

`source_manifest.json` records the exported Git identity, every original file's
Git blob and SHA256, its identity at independent-layer admission, and normalized
definition hashes. It distinguishes later network changes from admitted files.
The original source names in this manifest are intentional provenance.

## Code structure and entry points

| Responsibility | Implementation |
| --- | --- |
| Public whole-network forward and reusable compilation | `prefill_fwd.prefill_fwd`, `prefill_fwd.compile_prefill`, `prefill_program.py` |
| Shared-computation L2 regions and exact argument bindings | `prefill_l2.py` |
| Complete layer | `prefill_layer.make_prefill_layer_entries`, `run_prefill_layer_inmemory` |
| Original 40-layer composition | `prefill_layer.run_prefill_network_inmemory` |
| Actual checkpoint, persistent state, HCCL and L39 head | `prefill_network_runtime.run_network_runtime` |
| Input embedding / HC and final output kernels | `prefill_embedding.py`, `prefill_network_input.py`, `prefill_output.py` |
| SWA and C2/C1 attention | `prefill_swa.py`, `prefill_full2*.py`, `prefill_consumer*.py`, `prefill_ratio1.py` |
| Persistent source/cache ownership | `prefill_network_state.py`, `prefill_network_cache.py`, `prefill_layer_state.py` |
| Two Engram slots and singleton history | `prefill_network_engram.py`, `prefill_engram_runtime.py`, `engram_storage.py` |
| EP MoE and actual communication | `prefill_moe_ep.py`, `moe_dispatch.py`, `moe_transport.py` |
| Shared primitives | `quant.py`, `ecotech_linear*.py`, `qkv*.py`, `hc.py`, `gemm.py` |

The public entry is `prefill_fwd.prefill_fwd`; `run_prefill` remains an alias
for existing callers. It explicitly accepts inputs,
checkpoint weights, canonical cache state, numerical configuration, physical
device IDs and a new output directory. It has no dependency on historical
`formal_layer_receipts`, task releases, `TASK_DEVICE` or `DS41_JOB_ROOT`.
The runtime forwards those IDs in `assigned_devices` to both HCCL workers and
Engram preparation. Each uses the same explicit mapping in logical rank order;
duplicate IDs, rank-count mismatches and conflicting existing
`ASCEND_RT_VISIBLE_DEVICES` mappings are rejected before device initialization.
Historical release and continuation admission can be checked separately with
`audit_task.audit_task_release` and `audit_task.audit_task_continuation`.
Shape, dtype, topology, legal cache aliases and exact state preservation remain
runtime checks. Input/cache identity hashes describe the supplied operands;
they are not credentials from a private task.

```python
from models.deepseek_v4_1_flash_a3.prefill_fwd import compile_prefill, prefill_fwd

program = compile_prefill(
    inputs, cache=cache_state, config=network_contract,
    output="build_output/ds41-prefill-compiled",
)
result = prefill_fwd(
    inputs, weights=checkpoint, cache=cache_state, config=network_contract,
    devices=list(range(16)), compiled=program, output="build_output/ds41-prefill-new-run",
)
logits = result["logits"][0]["fresh"]  # Actual valid rows for logical rank 0.
```

The forward executes vocabulary embedding and initial HC, all original layers
0 through 39 with their original roles, final HC collapse, RMSNorm and the
vocabulary head. Each layer consumes the previous layer's actual activations.
Omitting `compiled` performs compilation inside the forward. A supplied program
is reusable in the same process; changed entry sets, IR, shapes or argument
names are rejected before execution. The pool uses structural equality as well
as the IR hash, so a hash collision cannot reuse a different computation.
During cold preparation, equivalent layer definitions also share their builder
result and specialized IR. The key includes the model, topology, routed capacity,
attention role and all cache plane offsets, strides, dtypes and shapes. Physical
cache ownership is still validated for every layer; sharing generated code does
not share live cache objects or bypass layer/source admission.

`compile_prefill` derives shapes from canonical metadata using storage-free
meta tensors. It compiles every stage without reading checkpoint payloads,
allocating full cache backing buffers, initializing CANN or running NPU work.
`compilation.json` records unique entries, all layer/stage bindings and actual
compiler time. Only structurally identical entries with the same argument names
share a compiled artifact across layers; different attention roles remain
distinct. The full-network runtime now composes five continuous compute regions
per layer into L2 entries, directly inlining the existing child functions:

| L2 region | Existing child calls |
| --- | --- |
| Input | TP embedding summation and initial SP/HC streams, after vocabulary gather |
| QKV | Q projection, normalization and RoPE; KV projection, normalization and RoPE |
| Attention packing | Inverse RoPE and head packet packing |
| Output projection | Head unpack, grouped output projection, output projection and reduce packing |
| FFN dispatch | Attention HC post, FFN HC pre, normalization, router, shared expert and EP packing |
| MoE finish | Combine, valid-row masking and FFN HC post |
| Head | SP unpack, RMSNorm and vocabulary projection, before vocabulary gather |

These regions replace 22 separate compute invocations with five invocations
within each layer and rank. This counts Python/runtime invocations, not
measured latency or the number of generated device tasks. Child arithmetic,
quantization, rounding, BF16 communication buffers, cache writes and comparison
tolerances are unchanged. Intermediate observation buffers remain explicit;
returned aliases are threaded between calls. The existing leading dynamic
token dimension is bound to the forward's explicit capacity. Inconsistent
types, child order or operand identity are rejected. Independent-layer
validation still executes the same child sequence with its original captures.
The input boundary additionally replaces two compute invocations with one;
the output head replaces three with one, per rank and chunk. Lookup, HC collapse
and vocabulary unpack stay on their respective sides of host communication.
`compilation.json` reports layer definition specializations and specialized
functions separately from unique compiled entries and total stage bindings.

This is **not a single L2 graph for all forty layers**; `compilation.json`
continues to report `single_l2: false`. Python owns the HCCL boundaries and
active routed-expert checkpoint reads. The existing HCCL BF16 reduction is
retained, rather than replacing its reduction order during a composition
change. The standalone child path remains a consumer for independent-layer
validation; removing the host network scheduler requires a separately
validated device communication and resident-expert interface.

The structure was compared with `models/deepseek_v4_1_flash/prefill_fwd.py`
and `models/qwen3_14b/prefill_fwd.py`: host preparation and resident operands
surround device L2 composition. The former uses A5 quantization and topology,
which are not substitutes for A3 Eco-Tech W8A8. Its host-prepared Engram lookup
also shows that table lookup does not itself prevent a single L2 backbone;
Engram projection and gating must consume the actual hidden at layers 1 and 14.
No Decode graph is exposed by this Prefill entry.

For the CLI, copy `models/deepseek_v4_1_flash_a3/prefill.example.json`, supply
actual files, and run from the repository root:

```bash
python -m models.deepseek_v4_1_flash_a3.prefill_fwd --config prefill.json --check-config
python -m models.deepseek_v4_1_flash_a3.prefill_fwd --config prefill.json --compile-only --output build_output/ds41-compiled
python -m models.deepseek_v4_1_flash_a3.prefill_fwd --config prefill.json -p a2a3
```

Paths in the JSON are relative to that JSON file. `--checkpoint`, `--devices`
and `--output` override the corresponding fields. `--check-config` checks the
configuration and mapping only; it does not load weights, prove input/cache
compatibility, compile or execute. Output directories must be new.
Compile-only also accepts the automatic `prompt` or `token_ids` configuration.
For the Python API, pass `weights=checkpoint` when compiling automatic inputs;
only checkpoint metadata is read. The complete prompt reserves the same cache
geometry as execution, while temporary first-chunk descriptors are removed after
compilation. Prepared dictionary inputs retain the existing API.

The standalone `prefill_moe_ep` driver also requires `--devices`, for example
`--devices 3,1` for a matching EP2 contract.
It writes that ordered mapping to `runtime_config.json` for its HCCL workers.

`inputs` and `cache_state` are Torch `weights_only=True` dictionaries. The first
contains `input_contract`, including actual per-request token IDs and ownership
metadata. To use an explicitly prepared backbone boundary instead of running
embedding, additionally supply `product_input_paths` and `product_input_files`.
The second contains `cache_ABI`, `cache_ABI_sources` and `engram_contracts`
with integer Engram slots 1 and 14. Their canonical rank/page metadata and
physical aliases are interpreted by `NetworkCacheProvider`; metadata must
match the actual allocated tensors. The JSON `contract` supplies `topology`,
`model_config`, `required_modes`, `capacity`, `scheduler_capacity`,
`fresh_shared_initialization`, `input_contract_sha256` and `state_regions`
(path and SHA256). These remain explicit state/operand contracts. This entry
requires canonical cache-layout and Engram templates. The automatic entry below
prepares request metadata and tokenizes raw text from these templates; a checkpoint
and prompt alone do not define cache ownership or a generation scheduler.

Weights retain the existing streaming policy: one host layer and the real
owned active experts are loaded as needed. Full weight residency is not added.
`execution.json` reports per-chunk elapsed time, layer/expert weight preparation
and checkpoint reads. The weight and read counters overlap; do not add them as
independent phases. Compiler timing is recorded separately. A ten-minute total
for loading and compilation requires measurement on the intended hardware and
checkpoint storage; HOST checks cannot establish that bound.

The current runtime supports the original 40-layer plan with DP4/TP4/EP16,
PP1/PCP1/DCP1 and capacity 16. Device IDs are runtime devices, not a claim of
16 separate physical cards. Multi-chunk continuation additionally retains
exact own-state identities, metadata lineage and unchanged numerical settings.
Dataset accuracy remains a separate check after execution.

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



## Dependencies and execution

Use the repository's PyPTO installation and `golden` harness, with compatible
Torch, NumPy, safetensors, tokenizers, and the PyPTO runtime. Actual A3 execution
also requires the installed CANN and HCCL shared libraries and an explicit
healthy device allocation. The Python transport calls CANN/HCCL through ctypes;
there is no omitted custom product `.cpp` or `.h` implementation. Compiler
outputs such as generated C++, PTO, and shared objects are build artifacts.

Native/reference helper modules additionally use Torch-NPU, vLLM-Ascend, and
custom native operators when explicitly invoked. They are not required by the
device-free checks and do not substitute for the A3 PyPTO network arithmetic.
`runtime.example.json` describes only this retained native-helper configuration.

Run bounded host controls and operator checks from the repository root:

```bash
python -B -m models.deepseek_v4_1_flash_a3.validate
python -m models.deepseek_v4_1_flash_a3.hc -p a2a3 --cpu-only
python -m models.deepseek_v4_1_flash_a3.swa -p a2a3 --cpu-only
python -m models.deepseek_v4_1_flash_a3.indexer_select -p a2a3 --cpu-only
python -m models.deepseek_v4_1_flash_a3.prefill_embedding -p a2a3 -d 0
```

`validate` uses only the standard library. It checks the maintained import DAG,
public exports, all 40 roles, scheduler/EP ownership, TP import independence and
device mappings. It deliberately does not require historical AST equality.
`audit_source` performs the separate opt-in source identity audit.

Executable operator entries accept `-p/--platform a2a3`, `-d/--device`,
`--sim` and `--compile-only`. HC, SWA and QLI additionally support `--cpu-only`;
that mode checks Torch reference fixtures without running kernels and cannot
be combined with `--compile-only`. Default device runs use `golden.run` with
the same compute factories as model composition. HC/SWA compare floating
outputs at `rtol=0.05, atol=0.05`, and untouched state exactly. SWA also checks
the entire updated cache exactly. QLI checks active discrete indices exactly;
its inactive pure-`Out` rows are undefined and are not compared. The QLI golden
models both FP16 rounding boundaries and the fixed 512-candidate score order;
`indexer_select_reference` remains the separate ideal logical visibility oracle.

CI owns this directory on `a2a3` and resolves relative imports for downstream
selection. Drivers needing external weights, cache state or whole-group device
allocation use `# ci: manual`, keeping their dependencies visible while excluding
them from automatic bounded execution.

HOST validation passed 19 CI-selection tests, all 114 module imports and 84
public exports, 20 CPU golden cases, and TP4/TP1/TP2/TP4 factory construction.
All 20 HC/SWA/QLI cases and eight grouped-output/sparse TP specializations
compiled with PTOAS 0.70. Simulator dispatch was checked
to resolve to `a2a3sim`, but simulation could not execute because the validation
host lacks `g++-15`. Device accuracy, full simulator CI and full-network execution
of the refactored package remain unverified; earlier A3 acceptance does not
certify this refactor. An AST comparison preserved all 200 original JIT definitions;
this structural check does not replace numerical regression.
No original task source, Goal or device job was changed for these checks.
See [migration notes](prefill_migration.md) for provenance and helper ownership.

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

Local CPU checks on 2026-11-11 used PyPTO `f32ecf3f`, Torch 2.10.0 CPU and
PTOAS 0.70, without connecting to an A3 server. With capacity 16, DP4/TP4/EP16
and two cache pages, all 926 bindings across forty layer roles produced IR
structurally equal to the previous builder path, retaining 55 unique entries.
Only seven layer definitions were constructed. One local comparison reduced
IR preparation from 90.38 seconds to 24.59 seconds; compilation, checkpoint
reads and device execution were excluded from this measurement.

All seven L2 region types generated PTO and C++ successfully, including the
new input and head regions. CPU golden scheduling checks covered TP2/TP4 and
active row counts 0, 1, 7 and 16, including empty DP engines and live masks.
The BF16 linear's eager shape annotations also produced IR equal to the old
sample-tensor specialization in three configurations; its function body was
unchanged. These are local composition and code-generation checks, not NPU
precision, HCCL, full-network logits or elapsed-time acceptance.

## Recorded complete-network execution

Before the shared-expert repair, one diagnostics-disabled A3 full40 run
completed with exit code zero, including physical layer39 and the final
logits path. It used DP4/TP4/EP16 across eight physical cards and sixteen
logical devices. Its greedy first-token comparison matched two of four
requests. Whole-network numerical acceptance therefore remains incomplete;
this historical run does not validate the subsequently repaired publication.
Continuous Decode and complete autoregressive answers remain unvalidated.

## Branch continuity

This package continues the maintained A3 branch at `d84e5d8`, including its
explicit public entry, operator ownership and five L2 compute regions. The
2048-token preparation and causal Indexer update come from cherry-picked
commit `79a584e64cb8cee5e0f3b816f77c014cba9911cc`. Shared-expert precision and
admitted-program reuse retain their documented validation limits above.
The original A3 source manifest remains unchanged.

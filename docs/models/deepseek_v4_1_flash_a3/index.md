# DeepSeek V4.1 Flash A3 Prefill and Decode

This A3 implementation contains real PyPTO operators and the runtime wiring
for the original 40-layer text backbone. The exported product snapshot is
`ebf19ad0ae7d668b4f7f2b04b526c9dcff2785a9`. Its historical source directory
was `models/deepseek_v4_1_flash_a2`; the maintained Prefill and Decode
implementations now share `models/deepseek_v4_1_flash_a3`.
See [Decode](decode.md) for its independent entry points and evidence scope.

## Development priorities

- Performance target: execute genuine Prefill40, transfer its state, and run
  continuous Decode40 on a common DP16/TP1/EP16 topology, retaining Eco-Tech real
  W8A8, 384 routed experts, Top6 and original Engram state.
- Deferred performance work: full weight residency, fused scheduling, throughput
  optimization, CP/SP expansion and slot reuse. Numerical and vLLM comparison
  are deferred for this execution milestone.
- Compatibility requirements: preserve original layer roles, byte aliases,
  cache/source lifetime, quantization, dtypes, and explicit DP/TP/EP mappings.
  TP2 fixed-16 full-network compatibility remains unvalidated; TP1 execution
  evidence is scoped to the bounded generation case below.

## Evidence and scope

| Item | Evidence scope |
| --- | --- |
| Original A3 independent complete layers | 11/11: physical layers 0, 1, 2, 3, 14, 15, 19, 20, 21, 24, 25 |
| Independent-layer admitted source | `afef9572cfe9c518f4e60d5ea7b00368b7297fbb` |
| Early four layers | DP4 / TP4 / EP16, PP1 / PCP1 / DCP1 |
| Late seven layers | DP2 / TP2 / EP4, PP1 / PCP1 / DCP1 |
| Exported latest network implementation | Full original 40-layer composition, DP4 / TP4 / EP16, capacity 16 |
| Current 40-layer Decode implementation | Host schedule plus embedding / Engram / LM head, DP16 / TP1 / EP16; two consecutive real-weight tokens completed with finite logits |
| Full-network accuracy and actual L39 output | Not established at export; original task validation was in progress |
| Fixed-16 TP1 Prefill and Decode execution | Prefill40 → state handoff → two Decode40 steps and all heads completed; three generated token IDs |
| Fixed-16 TP2 full-network compatibility | Not established |
| Full real-weight and both Engram tables in HBM | Not established by code or host weight-byte counts |
| Dataset score / serving integration | Not established by this delivery |
| Relocated package device execution | Complete Prefill-to-Decode generation executed on DP16/TP1/EP16; numerical accuracy remains unvalidated |

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

## Prefill to Decode generation

`generate.py` builds metadata directly from the selected checkpoint and its
original tokenizer. It runs genuine Prefill operators for the prompt, exports
completed request state, then feeds greedy tokens through the Decode network.
Both phases use DP16/TP1/EP16 with PP1/PCP1/DCP1. The initial execution entry
replicates one prompt across the sixteen DP engines, with one fixed request
slot per rank. Each phase executes all forty original layers and its LM head.

```bash
# Submit this command through the site allocator for sixteen devices.
python -m models.deepseek_v4_1_flash_a3.generate \
  --checkpoint "$CHECKPOINT" --ep 16 --tp 1 --devices "$TASK_DEVICE" \
  --tokens 0,1,2 --max-new-tokens 3 --ignore-eos \
  --output build_output/ds41-generation
```

Use `--prompt "text"` for raw completion text and add `--chat` to render the
checkpoint's original single-user chat template in chat mode. Use
`--chunk-size 1..16` for prompt chunks, or `--compile-only` for device-free compilation. The prompt plus the
consumed Decode tokens must fit 256 positions. By default generation stops
when every rank selects EOS; `--ignore-eos` requests a fixed execution count.
The first generated token comes from the final prompt row; three new tokens
therefore execute two Decode steps.

Weights stream by layer. Engram rows come from checkpoint file mappings,
without copying complete tables into shared memory. Prefill owns persistent
workers and HCCL groups within its process. That process exits before Decode
workers start. This phase boundary is deliberate because the existing Decode
runtime forks workers and requires a parent without initialized CANN state.

`handoff.pt` stores forty SWA owners in logical token order, the C2 sources at
layers 2/8/14, the C1 source at layer 20, index codes/scales, compressor rings,
selection owners, the candidate pool, and token/Engram history. Import retains
source aliases, remaps physical pages into dense Decode storage, and checks
copied tensor bytes. Odd prompt lengths retain the incomplete C2 pair in the
32-row compressor ring. Route/dispatch/combine scratch is rebuilt after the
preceding token has fully combined.

`prefill-result.json`, `handoff.json`, `decode/step-*.json` and `result.json`
record actual phase completion, transferred bytes, next positions, finite
vocabulary output, and generated token IDs. Execution acceptance does not
establish numerical accuracy. Host transfer checks cover contexts 1, 3, 127,
128, 129 and 255, including odd compression boundaries and page crossings.

Each selected token updates `answer.txt`, `raw-answer.txt`, `tokens.json` and
`token-events.jsonl` using the original tokenizer. Text output represents
logical rank 0; token events retain all sixteen rank IDs. `timings.json`
records one generation's host wall time from entry into `main`, including
compilation, weight streaming, transfers and runtime overhead. It excludes
the allocator queue, shell activation and Python module imports. Top-level
phases and Decode step durations partition the total with an explicit host
remainder. Prefill runtime/chunk/layer/call timings and Decode layer timings
are nested breakdowns, not additional time to add to their enclosing phases.
First-token and inter-token times come from the actual selected-token events.
`timings-partial.json` is saved after each completed Decode step.

### Execution acceptance, 2026-11-12/13

The real `DeepSeek-V4.1-Flash-w8a8` checkpoint completed the complete pipeline
on allocated physical devices 0 through 15. The run exited with code 0 and
printed:

```text
[generation] Prefill40 -> state handoff -> Decode40 RUN PASS; accuracy=not_run
```

| Check | Observed result |
| --- | --- |
| Topology | DP16 / TP1 / EP16, PP1 / PCP1 / DCP1 in both phases |
| Prompt | `[0, 1, 2]`, replicated across sixteen fixed request slots; one Prefill chunk |
| Prefill | Layers 0..39, both Engram slots and final LM head completed |
| Handoff | 9,613,504 cache bytes copied and checked; forty SWA owners and sources 2/8/14/20 |
| Odd C2 boundary | Residual 1 retained; first Decode consumed position 3 |
| Decode | Two consecutive complete 40-layer forwards, both Engram slots and both LM heads completed |
| Generated IDs | `[50388, 777, 777]` on every rank; token sources Prefill / Decode / Decode |
| Position continuity | Logit positions 2 → 3 → 4; final consumed context length 5 |
| Vocabulary output | Each phase returned finite logits with shape `[16, 129280]` |
| Observed elapsed time | Prefill 602.56 s; Decode steps 1220.88 s and 1157.13 s; total 3121.68 s |
| Numerical/reference acceptance | Not run |

These elapsed times include host scheduling and layer weight streaming.
The run artifacts are under `build_output/ds41-generation-tp1-r1/`, including
`verification.json`, `result.json`, `handoff.pt`, `handoff-byte-verification.json`,
the actual Prefill head and both Decode output tensors. Verification checks
that each generated ID is the argmax of its saved vocabulary output and that
the next full forward consumes that generated token at the next position.
This execution case covers a single prompt chunk and two Decode steps;
multichunk Prefill device acceptance and contexts beyond this short prompt
remain outside this run's evidence. Host metadata planning additionally passed
2048 tokens in 128 chunks, and host transfer fixtures cover the six boundary
contexts listed above.

## Code structure and entry points

| Responsibility | Implementation |
| --- | --- |
| Public whole-network forward and reusable compilation | `prefill_fwd.prefill_fwd`, `prefill_fwd.compile_prefill`, `prefill_program.py` |
| Shared-computation L2 regions and exact argument bindings | `prefill_l2.py` |
| Complete layer | `prefill_layer.make_prefill_layer_entries`, `run_prefill_layer_inmemory` |
| Original 40-layer composition | `prefill_layer.run_prefill_network_inmemory` |
| 40-layer Decode schedule and persistent host state | `decode_fwd.py --network`, `decode_network.py` |
| Actual checkpoint, persistent state, HCCL and L39 head | `prefill_network_runtime.run_network_runtime` |
| Input embedding / HC and final output kernels | `embedding.py`, `prefill_network_input.py`, `prefill_output.py`, `decode_head.py` |
| SWA and C2/C1 attention | `prefill_swa.py`, `prefill_c2a*.py`, `prefill_compressed_reuse.py`, `prefill_sparse_attention.py`, `prefill_c1a.py` |
| Persistent source/cache ownership | `prefill_network_state.py`, `prefill_network_cache.py`, `prefill_layer_state.py` |
| Two Engram slots and singleton history | `engram_host.py`, `prefill_network_engram.py`, `prefill_engram_runtime.py`, `engram_storage.py` |
| EP MoE and actual communication | `prefill_moe.py`, `moe.py`, `moe_transport.py` |
| Shared primitives | `quant.py`, `ecotech_linear*.py`, `qkv*.py`, `hc.py`, `gemm.py` |

The public entry is `prefill_fwd.prefill_fwd`; `run_prefill` remains an alias
for existing callers. It explicitly accepts inputs,
checkpoint weights, canonical cache state, numerical configuration, physical
device IDs and a new output directory. It has no dependency on historical
`formal_layer_receipts`, task releases, `TASK_DEVICE` or `DS41_JOB_ROOT`.
The runtime forwards those IDs in `assigned_devices` to HCCL and compute workers.
They use the same explicit mapping in logical rank order;
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
continues to report `single_l2: false`. Python owns the HCCL boundaries. Ordinary generation now preloads routed
experts once; diagnostic execution retains on-demand expert reads. The existing HCCL BF16 reduction is
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

Engram table lookup runs on the host **before** the prepared forward call.
`engram_host.py::HostEngram` owns the complete shared CPU INT8/FP32 tables,
hashes the explicit request history, gathers selected entries and applies group32
dequantization with BF16 rounding. Pass the resulting rows to `prefill_fwd`:

```python
from models.deepseek_v4_1_flash_a3.engram_host import HostEngram
from models.deepseek_v4_1_flash_a3.prefill_fwd import prefill_fwd

with HostEngram(checkpoint, cache_state["engram_contracts"], "generation") as host:
    rows = host.lookup(inputs, cache=cache_state, config=network_contract)
    result = prefill_fwd(
        inputs, weights=checkpoint, cache=cache_state, config=network_contract,
        devices=devices, output="build_output/ds41-host-rows", engram_lookup=rows,
    )
```

Each continuation prepares fresh rows for its actual tokens and history, then
supplies them as `chunk["inputs"]["engram_lookup"]`. The host owner can retain
both tables across the entire stream; returned BF16 rows have independent storage.
The CLI, automatic input preparation and `generate` caller perform this step.
No Engram table is registered with CANN or passed as a device address. Forward
kernels consume rank-local `[capacity, 6144]` BF16 rows for projection and gating;
the forward also publishes the existing compressed slot history once per chunk.
This Prefill interface can still process single-token feedback. The separate
[40-layer Decode entry](decode.md#40-layer-network) reads sparse rows from the
same checkpoint tables and retains its own per-request histories; it does not
yet consume a Prefill cache state.

`python -m models.deepseek_v4_1_flash_a3.validate_engram_host` checks small real
safetensors backings, exact group32 row values, chunk history, image masks,
TP placement, empty DP replicas, teardown and forward admission on CPU.
These checks do not establish complete-network NPU accuracy or token latency.
Add `--compile-only` to compile the full-size projection and gate without a
device, or pass `--device` with an allocated device ID for the bounded NPU check.

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

The separate `generate` caller supplies a lazy greedy feedback stream to this
same forward. It uses the original encoded chat prompt, selects each token
from the last valid product vocabulary row, advances positions and consumed
history, and stops at the original EOS or the recorded token limit. It reserves
the bounded context up front and retains one cache/history provider, compiled
program and finite Engram table owner across all chunks. Every new token costs
one complete 40-layer Prefill forward; this is not a Decode implementation.
With diagnostics disabled, continuation checks allocation identities and
functional metadata lineage without optional persistent-tensor byte hashes.
The optional `progress` callback reports execution stages without tensor captures.

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


## A5-style generation-scoped construction

The existing `prefill_fwd(..., diagnostics=False)` residency policy constructs
one runtime around the complete lazy Prefill chunk stream. The generation entry
selects `weight_residency="layer_streaming"`; callers that omit this setting
retain the complete weight preload described below. This follows
the resource lifetime used by A5 `prefill_fwd.py::generate`, while retaining
A3 Eco-Tech W8A8, DP4/TP4/EP16 and the existing arithmetic. A3 does not import
A5 MXFP8/MXFP4 operands or substitute its communication order.

- `prefill_resident_weights.py` preloads all forty compute layer banks, all 384
  routed experts per layer on their actual EP owner, one shared expert replica
  per rank, and the vocabulary shards. Expert source tensors are released
  after upload; device operands persist until generation ends. Nonexpert host
  views remain the binding identities. No weight payload hashes are taken by
  the ordinary loader. Diagnostics preserve the previous hashes and captures.
- `prefill_resident.py` owns one public PyPTO `ChipWorker` per logical rank,
  persistent typed weight/cache allocations and retained HCCL communicators.
  Boundary transfers use concurrent in-process CANN/HCCL calls, preserving the
  existing gather, all-to-all and BF16/FP32 sum semantics. They do not write
  transport packets to files or launch new transport processes.
- Both full Engram tables retain their existing INT8/FP32 shared-host contract.
  The input caller retains their CPU mappings across tokens and passes only
  gathered BF16 rows into the forward. No Engram lookup worker or device table
  registration is created. Compute-weight residency is not an all-model-HBM
  acceptance claim.
- Existing cache typed views bind to their complete owning byte allocation.
  Current host consumers still receive writable state readback. CPU metadata,
  active-expert selection and the layer scheduler remain on the host. Each
  generated token still executes all forty layers, reusing the same program,
  workers, weights and cache owners.

This is a resident host-scheduled A3 runtime, **not** an A5-style single device
backbone graph. Reports explicitly set `single_device_graph: false` and retain
`numerical_acceptance: not_established_by_residency`. The independent
`diagnostics=True` path still needs streaming weights and exact captures; it is
an actual diagnostic consumer rather than a generation performance fallback.
An allocation, worker or collective failure aborts generation and attempts to
release the owned resources. A poisoned device can also fail during cleanup;
the observed second-token failure below includes this case. The requested weight residency policy is explicit; allocation failures do not
change it.

`resident_runtime` records initializations, allocated compute-weight/state
bytes per rank, actual dispatch/transfer counts and H2D/D2H traffic. Startup
preload time is separate from chunk time. These counters are not observed
whole-device HBM peaks or speedup measurements.

Bounded validation commands, run from the repository root:

```bash
python -B -m models.deepseek_v4_1_flash_a3.validate
python -B -m models.deepseek_v4_1_flash_a3.validate_resident
# Device IDs must come from one two-rank site allocation.
python -B -m models.deepseek_v4_1_flash_a3.validate_resident_device \
  --devices "$TASK_DEVICE" --output build_output/resident-validation
```

The resident HOST validation uses real Torch byte aliases and the public
`DeviceTensor` representation with bounded CPU worker substitutes. It checks
multi-rank weight slices, repeated calls, independent typed cache aliases,
overlapping argument views, host mutation detection and failure cleanup.
A synthetic forty-layer bank checks all expert owners and two traversals
without checkpoint reloads. It initializes no NPU and establishes no model
accuracy or latency result. The separate device entry checks actual repeated
RMSNorm dispatch and three two-rank HCCL collectives; even a pass there is not
a full40 or two-token performance acceptance.

### Resident execution snapshot (2026-11-12)

This snapshot extends `1a91a1964d13880c1306623cdd123656cf52c79a` with the
resident implementation used by a cold full40 generation attempt. It includes
the operator sources, the flattened INT64 Engram slot binding and explicit
current-device selection for allocation, copies, dispatch and retirement.
Embedding and output-head entries use separate CPU-staged workers per rank
to stay within the primary worker's 64-callable limit. Those workers still
share the primary worker's process; process isolation is not included here.

The attempt allocated sixteen device IDs with DP4/TP4/EP16, capacity 16 and
four fixed-baseline requests. It used PyPTO `72a62d13efab`, simpler
`6b6b88f2353c`, PTOAS 0.70, CANN 9.1.0 and Torch 2.10.0 CPU. The private
fixed-baseline driver invoked the public forward path; it did not validate
the separate chat-oriented `generate.py` entry end to end.

| Observed phase | Elapsed time | Scope |
| --- | ---: | --- |
| Queue start to first complete logits | 1937.819 s | About 32 min 18 s; file-write completion, not client delivery |
| Total failed attempt | 2178.038 s | About 36 min 18 s, including error handling and cleanup |
| Compilation preparation | 357.143 s | Includes 314.673 s of IR/PTO/C++ generation; excludes lazy device-binary builds |
| Compute-weight preload | 746.580 s | Includes checkpoint reads, CPU preparation, allocation and upload |
| First network chunk | 814.991 s | Includes input/Engram preparation, all forty layers and output head |
| Layer0 through layer39 | About 212.924 s | Estimated from timestamped progress boundaries inside the chunk |

Compilation retained 76 unique entries for 947 stage bindings. Compute-weight
allocations totaled 653,354,818,560 bytes across the sixteen ranks. Both Engram
tables remained shared host mappings, totaling 221,190,580,800 bytes of unique
codes and scales. Their cold preparation still copies and SHA-256 checks whole
table shards in `engram_storage.py`, even with ordinary tensor diagnostics
disabled. None of these counters is a measured whole-device HBM peak.

All forty layers and the output head executed for the first token of each
request. The floating comparison against the captured reference failed at
`rtol=0.05, atol=0.05`, and greedy token IDs matched only two of four requests.
The second token failed in Engram hash dispatch with `valid_reshape` and host
error 507018; a subsequent resource-retirement error also occurred. This is an
execution snapshot with known failures, not numerical acceptance, continuous
generation acceptance or a steady-state latency benchmark. No second-token
latency or complete answer was measured.

The bounded `validate` and `validate_resident` commands passed on this Python
snapshot. They cover host controls, aliases, expert ownership, retained worker
lifetimes and partial HCCL cleanup with substitutes; they establish no further
device precision or performance result.

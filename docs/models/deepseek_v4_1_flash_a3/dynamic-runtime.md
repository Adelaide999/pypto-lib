# Packed requests and paged history

The shared network supports explicit request admission and homogeneous Prefill
or Decode dispatches. A session retains its workers, compiled program, weights,
cache allocations and source/reuse publications across dispatches. Device
correctness and performance acceptance must be recorded separately from HOST
contract and compiler checks.

## Independent limits

`runtime_config.RuntimeLimits` separates three quantities:

| Setting | Meaning |
| --- | --- |
| `max_num_seqs` | Maximum live requests in the complete session, including paused requests |
| `max_num_batched_tokens` | Total new query tokens across all DP owners in one dispatch |
| `max_model_len` | Maximum original positions in each request's history |
| `max_seqs_per_dp` | Persistent request-slot capacity on each DP owner |

The network uses a fixed physical token bucket selected from the configured
new-token budget. Supported network buckets are 16, 64, 256 and 512; 1024 and
2048 are planning values and are rejected by the executable network path.
Actual token counts, request counts, query boundaries and sample rows vary
inside the reserved capacities. Changing physical token capacity, slot capacity,
page-table width or parallel topology creates a new compiled geometry.

The history reservation can cover 1,048,576 original positions. This is a
supported addressing/planning extent, not a claim of completed 1M network
acceptance. Expanded token buckets and reservations above 2048 positions require
an explicit per-rank HBM budget before device execution.

## Request lifecycle

`metadata.RequestQuery(request_id, dp_rank, token_ids, prompt_length, generation)`
describes the new tokens only. New requests enter Prefill with `generation=None`.
Every continuation supplies the handle's generation and retains its DP owner
and original prompt length. A Decode query contains exactly one token.

Requests omitted from a dispatch keep their state. `pause_request` and
`resume_request` control eligibility; `finish_request` requires completed
Prefill, and releases the request. `cancel_request` releases an unfinished or
paused request. Release resets every leased physical page on all TP owners
before publishing a free slot and incrementing its generation. A stale handle
cannot execute or retire a reused slot.

Admission and metadata preparation are transactional. A preparation failure
returns newly reserved pages and slots without advancing history. History is
committed after all forty layers and the required output head complete. A
device execution or reset failure invalidates the session; close it rather
than replaying a possibly partially written batch.

`prefill_fwd.prefill_batch` starts the session, and
`GenerationSession.advance_batch` continues it. For example, with a checkpoint
and a measured `budget` dictionary:

```python
from models.deepseek_v4_1_flash_a3 import ParallelPlan, RequestQuery, prefill_batch
from models.deepseek_v4_1_flash_a3.prefill_prepare import checkpoint_cache

plan = ParallelPlan(tuple(range(16)), dp=4, tp=4)
cache, config = checkpoint_cache(checkpoint, plan.topology)
config.update(
    context_capacity=8192,
    runtime_limits=dict(max_num_seqs=8, max_seqs_per_dp=4,
                        max_num_batched_tokens=64, max_model_len=8192),
    hbm_budget=budget,
)
result = prefill_batch(
    [RequestQuery("a", 0, (11, 12, 13), 3), RequestQuery("b", 1, (21, 22), 4)],
    weights=checkpoint, cache=cache, config=config, devices=plan.devices,
    output="build_output/packed-first",
)
session = result["session"]
try:
    a, b = session.request_handle("a"), session.request_handle("b")
    session.advance_batch([RequestQuery("b", 1, (23, 24), 4, b.generation)],
                          phase="prefill", output="build_output/packed-b-tail")
    decoded = session.advance_batch(
        [RequestQuery("a", 0, (14,), 3, a.generation),
         RequestQuery("b", 1, (25,), 4, b.generation)],
        phase="decode", output="build_output/packed-decode",
    )
    # decoded["request_outputs"] follows caller order and includes request identity.
    session.finish_request(a)
    session.cancel_request(b)
finally:
    session.close()
```

Each required HBM budget field is a list of sixteen nonnegative byte counts:
`usable_bytes`, `weight_bytes`, `scratch_bytes`, `communication_bytes` and
`runtime_reserve_bytes`. Planning subtracts the latter four reservations from
`usable_bytes`, counts aliased storage once, and takes the smallest rank's page
capacity. After weights preload, execution samples actual free device memory
and rejects a cache pool that cannot fit with scratch/communication/reserves.
The per-rank decision is saved as `hbm-admission.json`.

## Metadata and cache ownership

`metadata.py` owns packed query coordinates, stable slots, generations and
request output mapping. `cache_pool.py` owns physical page leases. Original A3
raw/typed aliases, C2 odd/even residuals, compressor rings, TP token ranges and
source/reuse publication checks remain in the native provider.

Each dispatch carries bounded Engram lookback, `history_start`, an input epoch
and predecessor identity, retired generations, and incremental
`cache_page_updates`. The automatic path uses in-memory rank metadata with an
owner-bound receipt and writes a small input contract. Explicit `files`
transport remains available for prepared inputs. Native fixed-width page tables
are still lowered and validated on the HOST each dispatch; fully incremental
device page-table updates are a further optimization.

SWA leases retain enough pages for the entire query chunk plus every 128-token
causal window. Compressed source history grows by pages. Release clears original
byte arenas in place, preserving resident aliases. Raw cache writes stay on the
device; shared selector publications and output buffers read back active rows.

`paged_indexer.py` reads native index-cache pages into a bounded 2048-row scratch
tile, retains the existing QLI arithmetic, and merges global TopK512 positions.
Source candidate selection merges maxima of eight-position blocks globally,
retains 2048 blocks and pins the newest visible block. Candidate consumers read
their retained blocks directly. No dense full-history Indexer arena is built.
The current HOST scheduler still launches and transfers bounded tile scratch;
its overhead at 1M requires device measurements and further scheduling work.

The output head selects final prompt/decode rows before normalization and
projection. Intermediate Prefill chunks with no samples skip the vocabulary
head. Returned `request_outputs` contain request ID, generation, DP owner,
packed row and compact logit row, independent of dispatch order.

## Validation order

1. Reproduce and close the continuous 33-token 16/16/1 `507018` device gate.
2. Run multi-request omission, reordering, pause/resume, finish/cancel, stale
   generation and page-reset checks with the original resident cache aliases.
3. Compare Prefill-to-Decode outputs against the reference; verify unchanged
   workers, weight read/upload counts and compilation across changing B/S.
4. Compare 16/64/256/512 token buckets, dynamic tails and MoE/collective boundaries;
   record L2 coverage and peak HBM rather than inferring a single device graph.
5. Compare paged Indexer leaves and global candidates at page/tile boundaries,
   then 4K/32K/128K/1M histories under the measured memory budget.
6. Repeat complete-network acceptance on DP4/TP4/EP16, DP8/TP2/EP16 and
   DP16/TP1/EP16. These are the implemented plans; other EP sizes are rejected.

HOST contracts and IR checks:

```bash
python -m pytest -q tests/contract/test_ds41_dynamic_metadata.py \
  tests/contract/test_ds41_batch_lowering.py \
  tests/contract/test_ds41_prefill_program.py \
  tests/contract/test_ds41_resident_session.py
python -m models.deepseek_v4_1_flash_a3.paged_indexer --cpu-oracle
```

Leaf compile/device checks use the same functions as the network:

```bash
python -m models.deepseek_v4_1_flash_a3.paged_indexer -p a2a3 --compile-only
python -m models.deepseek_v4_1_flash_a3.lm_head -p a2a3 --compile-only
```

Omit `--compile-only` only on an allocated device. Compilation alone does not
establish numerical accuracy, reset correctness, long-context capacity or speed.

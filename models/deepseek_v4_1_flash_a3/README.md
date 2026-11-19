# DeepSeek V4.1 Flash on A3

This directory implements the text forward for the supplied
`DeepSeek-V4.1-Flash-w8a8` checkpoint on 16 A3 chips. Its model source is
`../deepseek_v4_1_flash`; `source_manifest.json` records the files used.

The supported topology is DP4 x TP4 with EP16 and 24 complete routed experts
per rank. One request occupies one TP group. Prefill receives all 8,192 rows
in one host submission. Its last row produces the first token; seven decode
calls at positions 8192 through 8198 produce the remaining seven tokens.
The two graphs share rank-local weights and all seven cache allocations.

## W8A8 interpretation of the migration standard

The supplied checkpoint replaces the standard's FP8/FP4 weight assumptions:

- Quantized projections and expert matrices retain their original INT8
  payloads and FP32 per-output-channel scales. Zero offsets are checked.
  Inputs use symmetric per-token dynamic INT8 with nearest-even rounding.
  A3 Cube accumulates the integer products in INT32; the epilogue applies
  both scales in FP32 and rounds at the operator's BF16 boundary.
- O-A and O-B retain checkpoint BF16 weights. Each TP member publishes a
  BF16 output partial; the group sums those partials in FP32 and returns BF16.
- Gate weights widen exactly from checkpoint BF16 to FP32. Engram gathers
  the real INT8 table rows, dequantizes them on the host, and uses the
  checkpoint BF16 projection and global rotation metadata.
- Window KV remains E4M3 with BF16 scales per 32 elements. Compressed KV
  remains packed E2M1 with BF16 scales per 16 elements. Index keys remain
  packed E2M1 with E8M0 scales per 32 elements. These are cache codecs,
  independent of the checkpoint's W8A8 weight format. E2M1 payloads use
  nearest-even rounding. C1A and C2A share the prefill index-key publisher;
  its upward E8M0 scales have a minimum value of 2**-126.

The revised named weight budget is **39,242,607,552 bytes = 36.5475 GiB per
rank**. It supersedes the original 20.7779 GiB estimate for FP4/FP8 weights.
The `v41-a3-w8a8-dynamic-nd-v3` weight ABI stores the remaining BF16
Indexer matrices in logical ND order, alongside the ND O and Engram matrices.

| Weight category | GiB/rank |
| --- | ---: |
| Routed experts | 31.6754 |
| Shared expert | 1.3198 |
| TP attention weights and sink | 1.7981 |
| BF16 vocabulary shard | 0.3082 |
| Other replicated weights | 1.4460 |

`weight_audit.json` contains byte counts, checkpoint identities, and shard
checks. Runtime heaps, communication windows, inputs, outputs, and caches
are additional allocations. The prefill root workspace includes a 2.5 GiB
four-slot residual buffer. MoE temporary allocations have a scope per round.
The runtime heap budgets are 4/4/2/1 GiB for rings 0/1/2/3. Board HBM sampling
reports observed occupancy, including the board baseline; it is not an exact
allocator high-water mark.

## Build and replay

Activate the supplied shared A3 environment first. Run from the repository
root and retain the activation script's Python overlay:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
model=models/deepseek_v4_1_flash_a3
devices=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

python "$model/prefill_fwd.py" -p a2a3 --devices "$devices" --compile-only
python "$model/decode_fwd.py" -p a2a3 --devices "$devices" --compile-only
```

Each build prints `A3_ARTIFACT <directory>`. Use those directories as
`PREFILL_ARTIFACT` and `DECODE_ARTIFACT`. An artifact manifest rejects changes
to the kernel source snapshot, compile driver, topology, or weight/cache ABI.

Generate the independent CPU reference before comparing device outputs:

```bash
python "$model/reference.py" --checkpoint "$CHECKPOINT" \
  --input-ids-json "$model/validation_data/input_ids.json" \
  --output "$REFERENCE_DIR" --save-layers 0,1,2,3,14,20,21,24
```

The reference reads the original checkpoint matrices. Its INT8 products are
exact integer sums, evaluated using bounded FP32 partial products and INT32
accumulation. It does not call device kernels. The frozen eight-step policy
requires all 129,280 logits to be finite, relative L2 at most 0.03, cosine
similarity at least 0.999, and identical greedy IDs for all eight steps.
The limits are written before device comparison and are not adjusted to fit
device results.

Strict gate operator checks accumulate the original BF16 values in FP64 and
round logits to FP32 before sqrt-softplus and biased selection. This avoids
CPU GEMM accumulation noise at a Top-6 boundary. The frozen v1 full-model
reference retains its original FP32 gate GEMM; `reference.py` explicitly
selects that arithmetic. The stored reference and its numerical limits stay
unchanged. `validation_reports/gate_reference_precision.json` records the
captured boundary cases and both CPU calculations.

```bash
python "$model/prefill_fwd.py" -p a2a3 --devices "$devices" \
  --checkpoint "$CHECKPOINT" --runtime-dir "$PREFILL_ARTIFACT" \
  --decode-runtime-dir "$DECODE_ARTIFACT" \
  --input-ids-json "$model/validation_data/input_ids.json" \
  --reference-dir "$REFERENCE_DIR" \
  --validate-layers 0,1,2,3,14,20,21,24 --audit-caches \
  --validate-teacher-forced --warmup-requests 1 --measured-requests 3 \
  --profile-last-decode --save-logits "$ACTUAL_LOGITS" \
  --execution-log "$EXECUTION_LOG" --report "$REPORT_JSON" > "$EXECUTION_LOG" 2>&1
```

`--teacher-forced` consumes the frozen reference token at each decode step.
Omit it for free greedy generation. Single-layer replay uses the production
prefill graph, checks DP0 and DP3 placement, and requires 49,152 global
send/receive/combine branches. Layer 14 uses a separately frozen residual
fixture to include its Engram update. Cache auditing requires byte-identical
active TP replicas and untouched inactive request caches.

TTFT starts after resident loading, before input preparation. Token delivery
times follow completion and copy-back. TPOT is the interval between the first
and eighth tokens divided by seven. Cache audits and numerical comparisons
are instrumented validation work and must be excluded from a performance
baseline. With warmup requests enabled, cache audits run on teacher forcing
and warmup requests; measured requests omit the audits. The profile request
runs separately after measurement. Compilation and component PASS results do
not establish full-model acceptance; the resident report records the actual
completion status.

## Performance measurement without numerical acceptance

Reuse the same checkpoint, input IDs, and compiled graphs. Omit numerical
references, layer checks, and cache audits from this run:

```bash
PYPTO_RUNTIME_LOG=timing python "$model/prefill_fwd.py" -p a2a3 --devices "$devices" \
  --checkpoint "$CHECKPOINT" --runtime-dir "$PREFILL_ARTIFACT" \
  --decode-runtime-dir "$DECODE_ARTIFACT" \
  --input-ids-json "$model/validation_data/input_ids.json" \
  --warmup-requests 1 --measured-requests 3 --profile-last-decode \
  --execution-log "$EXECUTION_LOG" --report "$REPORT_JSON" > "$EXECUTION_LOG" 2>&1
```

Weights remain resident throughout all requests. Each request reuses the
same seven cache allocations, reset before the next request. The separate
profile request follows all measured requests; its timings are excluded from
the baseline. This command leaves numerical acceptance pending.

The report contains host TTFT, all seven token intervals, TPOT, end-to-end
latency, engine-call durations, and sampled board HBM occupancy. Complete
STRACE captures additionally provide per-rank device execution windows.
Fast-rank comparisons use the active TP group; all 16 ranks remain visible
for EP imbalance. Missing rank records are explicitly reported.

`validation_reports/performance_previous.json` retains the earlier run's
instrumented warmup timings. That run completed eight tokens but stopped
before measured requests. Its cache audits are included in host token
intervals, and its artifacts precede the current source snapshot. Treat it
as historical diagnostic data rather than a performance baseline.

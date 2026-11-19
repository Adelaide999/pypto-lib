# A3 resident generation: Indexer and MoE operator timings

Speedup is baseline median device time divided by candidate median device
time. Both columns are in microseconds. Measurements use A3 NPUs and the
resident generation operator shapes; they do not measure full-network TTFT
or time per output token.

| Operator and shape | Baseline (us) | Candidate (us) | Speedup |
| --- | ---: | ---: | ---: |
| Indexer Decode, 1 row, ratio 2, 4096 visible keys | 121.0 | 44.3 | 2.73x |
| Indexer Decode, 1 row, ratio 1, 8193 visible keys | 121.2 | 75.6 | 1.60x |
| Indexer Decode, 1 row, ratio 2, full 16384 key bucket | 120.5 | 120.5 | 1.00x |
| Indexer Prefill, 512 rows, ratio 2, ending at 4096 visible keys | 2419.8 | 733.2 | 3.30x |
| Indexer Prefill, 512 rows, ratio 1, ending at 8193 visible keys | 2416.7 | 1297.1 | 1.86x |
| MoE return pack, EP16 geometry, slab 1 | 180.4 | 28.2 | 6.40x |
| MoE return pack, EP16 geometry, slab 2 | 365.7 | 46.1 | 7.93x |
| MoE return pack, EP16 geometry, slab 4 | 711.4 | 84.2 | 8.45x |
| MoE return pack, EP16 geometry, slab 16 | 2835.5 | 2824.9 | 1.00x |
| MoE return pack, EP2 geometry, slab 1 | 33.9 | 14.1 | 2.40x |
| MoE return pack, EP2 geometry, slab 2 | 52.9 | 16.1 | 3.29x |
| MoE return pack, EP2 geometry, slab 4 | 100.8 | 21.3 | 4.73x |
| MoE return pack, EP2 geometry, slab 16 | 386.2 | 380.7 | 1.01x |

The measurements compare two isolated changes against `68d10df9`:

- **Indexer:** 32 heads, head dimension 128, Top512, 16384 key slots.
  The compression ratio is tokens per index entry. The packed-cache path
  scores tiles up to the causal limit and zeros the remaining full tiles.
  TopK matched exactly and scores met rtol/atol 0.05 for the expected
  tiles. Each column uses five warmups and 30 timed rounds.
- **MoE return pack:** hidden width 5120, 384 experts, Top6, BF16. The
  experimental variant copied a full row per block for small slabs. Packet
  contents and untouched rows matched exactly. Each column uses eight
  warmups and 50 timed rounds, with the same frozen input replayed for both.
  EP2 and EP16 denote single-rank packet shapes, not collective timings.
  The 16-row control used identical tiling in both columns.

The resident branch already includes whole-row MoE return packing with
per-peer workers (`6afe0025`). The MoE speedups above refer to the earlier
tiled baseline; they must not be attributed to a change on top of that
newer implementation. A separate two-rank EP2 AllToAll control checked
both ranks' outputs; its timing is not included in the table.

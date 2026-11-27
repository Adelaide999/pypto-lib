# DeepSeek V4.1 Flash A3 Prefill and Decode

Default single-request generation uses separate 40-layer Prefill and Decode graphs
on DP4/TP4/EP16. Both bind the resident W8A8 weights and paged history.
Each L3 program calls a handwritten rank-local L2 schedule. Prefill selects six attention modes and executes EP16 communication in the graph. Decode runs separate encoder and decoder schedules with seven shared cache views. See the [default forward and commands](../../../models/deepseek_v4_1_flash_a3/README.md).

The [module map](../../../models/deepseek_v4_1_flash_a3/MODULE_ALIGNMENT.md) locates
the graph operators, weight loading, replay and validation. The
[accuracy collector](accuracy.md) captures raw logits from the same two graphs
and aligns them with a fixed vLLM-Ascend token history.

## Packed Prefill Indexer

Packed queries use each request's maximum visible compressed-key count. Requests
with up to 16K keys, up to 32K keys, and longer histories are scheduled in
separate groups. Consecutive rows use the original tensors. Interleaved rows
are gathered once per request and their results are scattered back to the input
row order. Up to four queries share a decoded key panel.

Histories over 32K keys stream 16K-key bands. The first decode can overlap
query preparation and subsequent decodes can overlap scoring. Each band merges
both the Top-512 keys and C1A's best 2048 eight-key blocks. The last band maps
keys to physical rows and publishes the sorted logical block IDs. C2A can map
up to 512 visible keys directly. C1A reindex scores
only the candidate blocks for sparse long histories and can reuse request-local
scores for denser short histories. Invalid pages and noncausal keys are excluded
from selection.

Run the single-device regression from the repository root. It checks the
physical Top-512 indices, candidate IDs and counts against independent Torch
references for each request. The attention option also consumes the selected
indices in the production sparse-attention operator.

```bash
python models/deepseek_v4_1_flash_a3/validate_prefill_indexer.py --case pool-holes --attention --device 0
python models/deepseek_v4_1_flash_a3/validate_prefill_indexer.py --case packed-mixed --device 0
python models/deepseek_v4_1_flash_a3/validate_prefill_indexer.py --case published-interleaved --attention --device 0
```

## Development priorities

- Performance target: run one distributed dispatch per prompt chunk and one
  per subsequent token, reusing rank-local weights, communication windows
  and cache allocations.
- Compatibility requirements: preserve the W8A8 checkpoint layout, six
  Prefill attention modes, Decode layer roles, DP4/TP4/EP16 ownership and
  the seven shared cache buffers.
- Deferred performance work: measure end-to-end timing and tune kernels on
  an allocated 16-device group.

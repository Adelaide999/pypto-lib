# DeepSeek V4.1 Flash A3 Prefill and Decode

This directory contains the actual A3 product implementation exported from
commit `ebf19ad0ae7d668b4f7f2b04b526c9dcff2785a9`, including PyPTO kernels,
complete-layer composition, full 40-layer Prefill wiring, cache ownership,
Engram, and CANN/HCCL transport. The independently developed Decode kernels
are consolidated into this same A3 package; the duplicate A2 directory is removed.

See the [model documentation](../../docs/models/deepseek_v4_1_flash_a3/index.md)
for entry points, dependencies, topology, and the limits of the evidence.
See [Decode](../../docs/models/deepseek_v4_1_flash_a3/decode.md) for its separate acceptance contract.
The original source directory was named `deepseek_v4_1_flash_a2`; that historic
path in `source_manifest.json` records provenance rather than the target hardware.

`prefill_fwd.prefill_fwd` executes embedding, the complete 40-layer backbone
and the final head from explicit request/cache metadata and checkpoint weights.
`prefill_fwd.compile_prefill` prepares all stage artifacts for reuse in the
same process without allocating the full cache buffers or running NPU work.
The existing host communication and weight streaming boundaries remain;
the Prefill interface does not claim a single device L2 graph.
`decode_fwd.py --network` executes embedding, all 40 Decode layers and the final
LM head with real checkpoint weights. The initial driver uses DP=EP with TP1
on an explicitly allocated EP2 or EP16 device group. It retains cache and
request state across tokens in the host session and streams one layer's
weights at a time. See [Decode](../../docs/models/deepseek_v4_1_flash_a3/decode.md#40-layer-network)
for commands, the 256-position bound and the current execution evidence.
The bounded SWA step and independent complete-layer entries remain available.

Run the device-free package checks from the repository root:

```bash
python -B -m models.deepseek_v4_1_flash_a3.validate
```

The original A3 task admitted 11 independent complete layers at an earlier
source identity. The current Decode network has completed two consecutive
real-weight 40-layer forwards plus LM head on DP16/TP1/EP16 with persistent
session state. The relocated Prefill network has also executed all forty
layers and its vocabulary head on DP16/TP1/EP16. Full-network accuracy remains
unvalidated. Weights, private fixtures, generated compiler artifacts,
and device allocations are not included.

`generate.py` connects genuine Prefill40 to Decode40 on DP16/TP1/EP16, with
layer weight streaming and an explicit CPU cache handoff. A real-weight run
completed Prefill40 and two Decode40 steps, all with LM heads, generating three
token IDs with finite logits and continuous positions. It builds checkpoint
metadata without captured fixtures and accepts `--prompt` or `--tokens`. See
[Prefill to Decode generation](../../docs/models/deepseek_v4_1_flash_a3/index.md#prefill-to-decode-generation)
for the allocated sixteen-device command, state coverage and context limits.
Add `--chat` for the checkpoint's original chat template. Generation writes
the decoded answer and per-token events as tokens arrive, then exports
host wall timings for Prefill, state transfer, compilation and every Decode step.

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
`decode_fwd.py` provides the bounded SWA step and `decode_layer.py` selects the
complete Decode blocks; a full 40-layer Decode driver remains outside this scope.

Run the device-free package checks from the repository root:

```bash
python -B -m models.deepseek_v4_1_flash_a3.validate
```

The original A3 task admitted 11 independent complete layers at an earlier
source identity. Full-network accuracy and a fresh relocated-package NPU run
are not claimed. Weights, private fixtures, generated compiler artifacts,
and device allocations are not included.

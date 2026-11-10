# DeepSeek V4.1 Flash A3 Prefill

This directory contains the actual A3 product implementation exported from
commit `ebf19ad0ae7d668b4f7f2b04b526c9dcff2785a9`, including PyPTO kernels,
complete-layer composition, full 40-layer Prefill wiring, cache ownership,
Engram, and CANN/HCCL transport. It is independent of the A2 delivery.

See the [model documentation](../../docs/models/deepseek_v4_1_flash_a3/index.md)
for entry points, dependencies, topology, and the limits of the evidence.
The original source directory was named `deepseek_v4_1_flash_a2`; that historic
path in `source_manifest.json` records provenance rather than the target hardware.

Run the device-free package checks from the repository root:

```bash
python -B -m models.deepseek_v4_1_flash_a3.validate
```

The original A3 task admitted 11 independent complete layers at an earlier
source identity. Full-network accuracy and a fresh relocated-package NPU run
are not claimed. Weights, private fixtures, generated compiler artifacts,
and device allocations are not included.

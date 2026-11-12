# A3 Prefill source migration

## Source identity

The product source is the tracked model subtree of commit
`ebf19ad0ae7d668b4f7f2b04b526c9dcff2785a9`, exported read-only from the A3
product Git repository. The initial delivery included all 118 Python modules. The maintenance mapping
records subsequent consolidation and removal of redundant facades. The original
subtree name `models/deepseek_v4_1_flash_a2` is historical; it must not be
confused with a separately published A2 adaptation.

The formal 11-layer A3 admission belongs to source
`afef9572cfe9c518f4e60d5ea7b00368b7297fbb`, applied tree
`f8d43ebff6b90deb35a17e74e45c175fa22d443e`. The manifest stores both blob
identities for each file, so later network-runtime fixes remain distinguishable.
Full-network acceptance is not inferred from those independent-layer results.

## Packaging transformations

1. Relocate the model package to `models.deepseek_v4_1_flash_a3`, including the
   explicit subprocess module and loaded-source discovery paths.
2. Initially rename `grouped_output.py` to `prefill_grouped_output.py`; its
   active computation now belongs to `qkv_proj_rope.py` with its caller updated.
3. Retain numerical definitions, internal native symbols, shared A2/A3 ABI
   names, layouts, parameter directions, and compiler name hints.
4. Include the original 40-layer plan with identical layer data. Remove only
   the private relative source-report path; retain its source SHA256.
5. Replace task-specific/historical delivery documentation with this explicit
   A3 scope and a sanitized source manifest. Keep weights, fixtures, reports,
   secrets, generated compiler files, and machine paths outside the package.

## Maintained file ownership

`ops.py` and `staged.py` were split by computation responsibility. Function
bodies, arithmetic ordering, dtypes, quantization and rounding semantics remain
unchanged. The distinct implementations are intentional: `linear_primitives`
owns blocked INT8 quantization/packing/dequantization, while `gemm` owns direct
and BF16/FP32 linear entries. `rmsnorm.make_blocked_rmsnorm` and
`norm_rope.make_adjacent_rope` keep the former blocked semantics separately
from the existing norm/RoPE contracts. `output_proj`, `activation`, `index_cache`
and `swa_attention` own the remaining staged computations. Pure direct-linear
forwarding now resolves directly to `gemm.make_int8_linear_direct`.

The active layer-2 kernels now belong to `compressor_projection.py`; grouped
TP output belongs to `qkv_proj_rope.py`. Generic layer selection remains in
`prefill_layer.py`. Unused historical fixed-layer implementations and pure
wrappers have been removed after checking retained callers.

Decode is integrated into the same A3 package; its independent contract and
historical evidence are documented in [Decode](decode.md). MoE shared kernels,
goldens and validation live in `moe.py`; actual Prefill/Decode orchestration,
communication, cache ownership and routed-expert kernels have separate roles.
Different phase rounding and state boundaries remain explicit implementations.

## Provenance and correctness checks

- `source_manifest.json` is the immutable original export identity. Its file set
  and hashes describe that export, including modules subsequently consolidated.
- `source_refactor_map.json` maps original definitions/facades to maintained
  owners and documents API/configuration changes.
- `package_refactor_map.json` records the A3/Decode package consolidation,
  scope-aware binding renames and preserved source/destination definitions.
- `decode_source_manifest.json` preserves the incoming Decode provenance.
- `maintenance_baseline.json` records the explicitly reviewed post-refactor
  definitions. `python -m models.deepseek_v4_1_flash_a3.audit_source` checks this
  optional identity baseline. Use `--original-root /path/to/exported-checkout`
  to verify the original snapshot against the original source manifest.
- `python -m models.deepseek_v4_1_flash_a3.validate` checks ongoing model/import,
  scheduler, ownership and runtime-configuration contracts without historical
  AST locks. Operator `golden.run` entries check numerical behavior.
- `audit_task.audit_task_release` and `audit_task.audit_task_continuation` retain
  original release and continuation receipt checks as opt-in historical audits;
  the public network entry does not require them.

The maintenance checks do not establish device numerics, communication,
full-network accuracy, HBM residency or performance. Original A3 independent-layer
acceptance remains evidence for its exact source, inputs and topology. New device
precision regressions are required before making that claim for maintained code.

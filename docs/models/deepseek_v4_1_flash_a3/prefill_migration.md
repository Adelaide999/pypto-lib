# A3 Prefill source migration

## Source identity

The product source is the tracked model subtree of commit
`ebf19ad0ae7d668b4f7f2b04b526c9dcff2785a9`, exported read-only from the A3
product Git repository. All 118 Python modules are included. The original
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
2. Rename only `grouped_output.py` to `prefill_grouped_output.py` and update
   its caller. Other dedicated Prefill files already carry the prefix.
3. Retain numerical definitions, internal native symbols, shared A2/A3 ABI
   names, layouts, parameter directions, and compiler name hints.
4. Include the original 40-layer plan with identical layer data. Remove only
   the private relative source-report path; retain its source SHA256.
5. Replace task-specific/historical delivery documentation with this explicit
   A3 scope and a sanitized source manifest. Keep weights, fixtures, reports,
   secrets, generated compiler files, and machine paths outside the package.

The retained `decode_*` and native helper modules preserve source provenance;
they are not newly validated Decode paths or a full-network native fallback.
Actual transport is implemented in `moe_transport.py` using ctypes bindings to
installed CANN/HCCL libraries. No product-specific C++ source is missing.

## Verification

`models.deepseek_v4_1_flash_a3.validate` checks all relocated source definitions
against their normalized source AST identities, resolves all local imports,
rejects dependency cycles, verifies the complete physical-layer plan against
the model configuration, and exercises bounded scheduler and expert controls.
It never compiles kernels, loads checkpoint weights, or initializes an NPU.
These HOST checks validate the relocation and cannot establish device numerics,
communication, full-network accuracy, HBM residency, or performance.

Original A3 independent-layer acceptance remains evidence for its exact source,
inputs, and topology. A fresh device run of this relocated package remains
necessary before making an equivalent acceptance claim for this branch.

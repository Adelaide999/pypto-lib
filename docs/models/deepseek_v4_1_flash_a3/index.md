# DeepSeek V4.1 Flash A3 Prefill

This A3 implementation contains real PyPTO operators and the runtime wiring
for the original 40-layer text backbone. The exported product snapshot is
`ebf19ad0ae7d668b4f7f2b04b526c9dcff2785a9`. Its historical source directory
was `models/deepseek_v4_1_flash_a2`; this delivery uses the separate
`models/deepseek_v4_1_flash_a3` package and actual A3 task sources.

## Development priorities

- Performance target: correctness of the A3 complete Prefill path, with
  Eco-Tech real W8A8, 384 routed experts, Top6, and original Engram state.
- Deferred performance work: final throughput and latency tuning until
  complete-network accuracy is independently established.
- Compatibility requirements: preserve original layer roles, byte aliases,
  cache/source lifetime, quantization, dtypes, and explicit DP/TP/EP mappings.
  TP1/TP2 fixed-16 full-network compatibility remains unvalidated.

## Evidence and scope

| Item | Evidence scope |
| --- | --- |
| Original A3 independent complete layers | 11/11: physical layers 0, 1, 2, 3, 14, 15, 19, 20, 21, 24, 25 |
| Independent-layer admitted source | `afef9572cfe9c518f4e60d5ea7b00368b7297fbb` |
| Early four layers | DP4 / TP4 / EP16, PP1 / PCP1 / DCP1 |
| Late seven layers | DP2 / TP2 / EP4, PP1 / PCP1 / DCP1 |
| Exported latest network implementation | Full original 40-layer composition, DP4 / TP4 / EP16, capacity 16 |
| Full-network accuracy and actual L39 output | Not established at export; original task validation was in progress |
| Fixed-16 TP1/TP2 full-network compatibility | Not established |
| Full real-weight and both Engram tables in HBM | Not established by code or host weight-byte counts |
| Dataset score / Decode / serving integration | Not established by this delivery |
| Relocated package device execution | Not run; HOST checks do not replace NPU validation |

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

## Code structure and entry points

| Responsibility | Implementation |
| --- | --- |
| Complete layer | `prefill_layer.make_prefill_layer_entries`, `run_prefill_layer_inmemory` |
| Original 40-layer composition | `prefill_layer.run_prefill_network_inmemory` |
| Actual checkpoint, persistent state, HCCL and L39 head | `prefill_network_runtime.run_network_runtime` |
| Input embedding / HC and final output kernels | `prefill_embedding.py`, `prefill_network_input.py`, `prefill_output.py` |
| SWA and C2/C1 attention | `prefill_swa.py`, `prefill_full2*.py`, `prefill_consumer*.py`, `prefill_ratio1.py` |
| Persistent source/cache ownership | `prefill_network_state.py`, `prefill_network_cache.py`, `prefill_layer_state.py` |
| Two Engram slots and singleton history | `prefill_network_engram.py`, `prefill_engram_runtime.py`, `engram_storage.py` |
| EP MoE and actual communication | `prefill_moe_ep.py`, `moe_dispatch.py`, `moe_transport.py` |
| Shared primitives | `quant.py`, `ecotech_linear*.py`, `qkv*.py`, `hc.py`, `gemm.py` |

The network entry consumes explicitly supplied contracts, source receipts,
captured physical cache ABI, checkpoint operands, and an actual whole-group
device allocation. It executes the product layers in memory and observes real
output/state boundaries; it does not invoke a native model forward as a
fallback. Contract/weight paths must be supplied by the caller. Private task
fixtures, leases, reports, and checkpoints are deliberately not packaged.

The layer plan retains all original physical layers 0 through 39. Capacity
16 is the frozen local executable row extent of the network entry; it is not
a claim that arbitrary batch=16 or every TP/DP arrangement has passed.

Prefill-specific files have a `prefill_` prefix. Shared primitives retain
general names. Legacy Decode exports and native boundary/reference utilities
remain traceable to the source snapshot and have no new validation claim.

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

Run the bounded checks from the repository root:

```bash
python -B -m models.deepseek_v4_1_flash_a3.validate
python -B -m models.deepseek_v4_1_flash_a3.prefill_layer --help
```

The first command uses only the Python standard library. It checks import
closure, preserved compute definitions, all 40 layer roles, EP ownership,
request/page scheduling, and negative controls without compilation or device
initialization. The second requires the runtime Python dependencies but does
not dispatch work. Operator files retain their own Torch goldens and validation
entries; invoke them only in a separately allocated execution environment.

No original task files, source branches, validation jobs, or device reservations
were changed to create this package. See the [migration notes](prefill_migration.md)
for the exact packaging transformations and verification limits.

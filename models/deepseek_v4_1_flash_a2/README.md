# DeepSeek V4.1 Flash on A2

Independent single-layer Prefill operators and stage factories, with W8A8
projection, BF16 attention caches, mHC, Engram, and MoE boundaries.

The actual PyPTO implementations are in this directory:

- `prefill_ecotech_swa.py`, `prefill_ecotech_full2.py`, and
  `prefill_ecotech_consumer.py` compose the Prefill attention stages.
- `ops.py`, `ecotech_linear.py`, `hc.py`, `engram_ecotech.py`, and `moe_*.py`
  contain shared numerical implementations.
- `prefill_input.py`, `prefill_io.py`, `output_head.py`, `hc_collapse.py`, and
  `sampling.py` implement embedding and the terminal output stages.

Historical independent-layer alignment covers the eleven Mini-B layers.
The latest staged first-token engineering run still differs from its matching
native reference. This package does not claim whole-network acceptance.

See the [model documentation](../../docs/models/deepseek_v4_1_flash_a2/index.md)
for construction examples, native runtime requirements, and the precise scope
of the historical single-layer acceptance evidence.

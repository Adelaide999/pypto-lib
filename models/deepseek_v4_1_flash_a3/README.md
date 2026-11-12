# DeepSeek V4.1 Flash A3

A3 Eco-Tech W8A8 operators, the complete 40-layer Prefill composition and
persistent runtime, plus bounded Decode operators. The package has 56 Python
modules organized by computation responsibility.

- `prefill_fwd.prefill_fwd`: embedding, all 40 layers and vocabulary head.
- `prefill_fwd.compile_prefill`: reusable compilation from request/cache geometry.
- `prefill_layer.py`: generic layer composition and L2 compute regions.
- `decode_fwd.py`: bounded single-step SWA; the full EP16 driver remains unimplemented.

Historical regression runners, replay/capture paths and task admission tooling
are removed. Operator files retain Torch goldens and standard validation
entries. Arithmetic, rounding and cache semantics are preserved.

See the [model documentation](../../docs/models/deepseek_v4_1_flash_a3/index.md),
[Decode scope](../../docs/models/deepseek_v4_1_flash_a3/decode.md) and
[module owners](MODULE_ALIGNMENT.md) for configuration, usage and current limits.

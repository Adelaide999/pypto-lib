# DeepSeek V4.1 Flash A3

A3 Eco-Tech W8A8 operators, the complete 40-layer Prefill composition and
persistent generation session, plus bounded and full 40-layer Decode paths.
Decode compositions use the `decode_` prefix and share the existing operator
owners for linear algebra, quantization, normalization, RoPE and cache operations.

- `prefill_fwd.prefill_fwd`: embedding, all 40 layers and vocabulary head.
- `prefill_fwd.compile_prefill`: reusable compilation from request/cache geometry.
- `network.py`: shared TP4 program registry, layer execution and generation session.
- `resident_weights.py`: one static weight owner and per-logical-weight audit.
- `generate.py`: DP4/TP4/EP16 Prefill followed by one-token Decode on the same workers.
- `prefill_layer.py`: concurrent rank stages and bounded MoE slabs of up to 16 rows.
- `prefill_resident.py`: persistent compute queues, device activations and HCCL transfers.
- `decode_layer.py`: full-width Decode layer dispatch and SWA/Engram layer bodies.
- `decode_swa.py`, `decode_c1a.py`, `decode_c2a.py`: Decode attention compositions.
- `decode_moe_ep.py`: full-width expert-parallel dispatch and rank composition.
- `decode_fwd.py --network` / `decode_network.py`: host-scheduled 40-layer
  DP4/TP4/EP16 Decode with resident weights, persistent state, Engram stages and the
  vocabulary head.
- `prefill_moe.py` / `decode_moe.py`: phase routing, dispatch, shared experts,
  Torch goldens and standalone validation.
- `grouped_moe.py`: blocked-INT8 grouped experts for the resident generation network.
- `expert_routed.py`: legacy NZ experts for bounded Decode compositions.

Historical regression runners, replay/capture paths and task admission tooling
are removed. Operator files retain Torch goldens and standard validation
entries. Arithmetic, rounding and cache semantics are preserved.

See the [model documentation](../../docs/models/deepseek_v4_1_flash_a3/index.md),
[Decode scope](../../docs/models/deepseek_v4_1_flash_a3/decode.md) and
[module owners](MODULE_ALIGNMENT.md) for configuration, usage and current limits.

The [native vLLM-Ascend accuracy workflow](../../docs/models/deepseek_v4_1_flash_a3/accuracy.md)
uses fixed TP4/DP4/EP16 prompt/Decode traces and compares full-vocabulary logits,
Top1 and TopK through the repository's `tools/` entries.

Generation defaults to 128-token Prefill chunks; Decode retains capacity 16.
Timing reports separate weight initialization, Prefill, Decode and first-token
wall time. Execution-only performance runs do not establish numerical accuracy.

# Native vLLM-Ascend accuracy comparison

The comparison tools use TP4/DP4/EP16, PP1/PCP1/DCP1 and the original 40-layer
Ascend W8A8 checkpoint. vLLM enables Engram CPU offload and DP shared memory.
The reference uses native model, router, sampler and trace-replay code, without
monkeypatches. TP1/DP16 temporary comparison scripts have been removed.

The tooling provides input preparation, independent captures and CPU comparison.
Configuration checks and CPU tests do not establish complete-network accuracy.
HCCL, weight loading and real Prefill/Decode accuracy require a separate
16-device model run.

## Numerical boundaries

The generation network uses the same corrected operators for Prefill and Decode.
Attention applies BF16 TP AllReduce to the local output-projection partials,
then selects the existing SP token buckets.

The shared expert retains four independent 576-channel quantization groups at
TP4. Resident weight preparation pads each group to the GEMM alignment without
changing its checkpoint bytes. Each group computes its own activation scale
and BF16 Down output. The outputs use paired BF16 sums in TP rank order.

Routed experts apply the token scale before the channel scale in FP32 for
Gate-Up. Activation requantization computes `scale = max(abs(x)) / 127`, then
rounds `x * reciprocal(scale)` through FP16 before INT8 rounding. Down projection
uses FP32 channel scaling. Input quantization keeps its existing dynamic INT8
boundary. These choices match the corresponding native shared and routed paths;
their quantization boundaries differ from one another.

MoE combine narrows selected routing probabilities to BF16 before multiplying
the BF16 expert outputs, accumulates those products in FP32, and rounds the
routed sum to BF16 before adding the reduced shared output.

Operator replay and complete-network comparison are separate checks. Exact
agreement for captured local inputs does not establish full-model agreement or
dataset accuracy. Full-model reports must include the frozen inputs, native
reference identity and measured logit errors.

## Prepare one immutable case file

In the PyPTO environment, run from the repository root:

```bash
export CHECKPOINT=/path/to/DeepSeek-V4.1-Flash-w8a8
export RUN_DIR="$PWD/build_output/deepseek-v41-accuracy"
python -m tools.deepseek_v41_accuracy prepare \
    --checkpoint "$CHECKPOINT" \
    --cases tools/deepseek_v41_accuracy.example.json \
    --output "$RUN_DIR/cases.json"
```

Both collectors must use this same prepared file. Preparation freezes the
explicit token IDs, vocabulary, topology and checkpoint location. It hashes
the configuration, quantization description, safetensors index and tokenizer,
and records each indexed weight shard's size and modification timestamp.
This detects metadata or shard-stat changes; it does not hash every weight byte.
Prepared cases and capture/report outputs refuse overwriting existing results.

The source file contains a `cases` list. Each case needs a unique `id`,
`prompt_token_ids` and `trace_token_ids`. Alternatively, replace either token
list with `prompt` or `trace` text. Preparation tokenizes text once using the
checkpoint tokenizer with `add_special_tokens=False`; neither collector
retokenizes the frozen IDs or applies a chat template. Supply fully formatted
chat token IDs when testing chat inputs.

The example's trace is a fixed history, not an expected answer. Each prompt
and trace must be nonempty, the trace may contain at most 256 IDs, and their
combined length must not exceed 2048 positions.

## Align Prefill and Decode

For prompt `P` and trace `[t0, t1, t2]`, both captures contain:

| Step | Phase | Model input history | Token whose score is reported |
| --- | --- | --- | --- |
| 0 | Prefill | `P` | `t0` |
| 1 | Decode | `P + [t0]` | `t1` |
| 2 | Decode | `P + [t0, t1]` | `t2` |

A trace of length N collects N full-vocabulary distributions: one final
Prefill distribution and N-1 actual one-token Decode distributions. The final
trace token is scored but never consumed as input. Top1 and TopK come from
the raw distribution, independently of the forced trace token.

The PyPTO collector calls the existing `prefill_fwd(..., keep_runtime=True)`
and `DecodeNetwork.step` on the same generation session. It verifies that
Decode retains the runtime and does not reload static weights.

The vLLM collector sets `enable_trace_replay=True`,
`SamplingParams(trace_decode_token_ids=..., temperature=0.0, logprobs=-1)` and
`VLLM_USE_V2_MODEL_RUNNER=1`. The engine also sets `max_logprobs=-1` and
`logprobs_mode="raw_logits"`. The default limit of 20 is insufficient for a
full-vocabulary comparison. vLLM keeps its `Logprob.logprob` field name in
raw-logits mode; this collector stores that field as logits, then computes
normalized log probabilities during CPU comparison. `prompt_logprobs` is
incompatible with trace replay and is not requested.

Only DP rank 0 receives a request. Other DP engines participate through native
dummy batches for expert-parallel collectives. All TP ranks in DP rank 0
receive the same IDs. Prefix caching, asynchronous scheduling and speculation
are disabled; the default Prefill chunk size is 16 in both collectors.

## Inspect configuration without devices

In the PyPTO environment:

```bash
python -m tools.deepseek_v41_accuracy pypto \
    --cases "$RUN_DIR/cases.json" --output "$RUN_DIR/pypto" --check-config
```

In the vLLM-Ascend environment, with this repository on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m tools.deepseek_v41_vllm_reference \
    --cases "$RUN_DIR/cases.json" --output "$RUN_DIR/reference" --check-config
```

The vLLM check constructs its native `EngineArgs` configuration and validates
trace replay and full-vocabulary sampling requests. It creates no workers,
loads no model weights and verifies that the NPU remains uninitialized.

## Capture on allocated devices

Run each collector separately through the site's task scheduler with 16
allocated devices exported as comma-separated physical IDs in `TASK_DEVICE`.
An existing `ASCEND_RT_VISIBLE_DEVICES` must match that mapping. Use the same
`--chunk-size` for both captures. The following commands belong inside those
allocated jobs, after activating each corresponding environment.

Native vLLM-Ascend:

```bash
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
export VLLM_USE_V2_MODEL_RUNNER=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=16 \
    --module tools.deepseek_v41_vllm_reference \
    --cases "$RUN_DIR/cases.json" --output "$RUN_DIR/reference" --chunk-size 16
```

The external launcher is required: a single-process offline `LLM` does not
support this four-way data-parallel reference configuration.

PyPTO:

```bash
python -m tools.deepseek_v41_accuracy pypto \
    --cases "$RUN_DIR/cases.json" --output "$RUN_DIR/pypto" --chunk-size 16
```

Each capture writes a manifest and one FP32 `[trace_length, vocabulary_size]`
tensor per case. The manifest records checkpoint identity, case-file hash,
exact histories, phase labels and tensor hashes. Interrupted or failed
captures cannot be compared as completed results.

## Compare on CPU

In the PyPTO environment, first measure without imposing an acceptance policy:

```bash
python -m tools.deepseek_v41_accuracy compare \
    --actual "$RUN_DIR/pypto" --reference "$RUN_DIR/reference" \
    --output "$RUN_DIR/metrics.json" --topk 10
```

The report contains per-step maximum/mean absolute error, RMSE, relative L2,
cosine similarity, log-probability error, KL divergence, Top1 agreement, TopK
overlap and trace-token log probabilities. Aggregates distinguish Prefill
from Decode. The tool rejects differing inputs, histories, checkpoints,
chunk sizes, incomplete vocabularies, modified artifacts and nonfinite data.

Without an explicit acceptance policy the status is `measured`, even when
all values match. To obtain a `pass` or `fail` verdict, supply the agreed
`--rtol` and `--atol` together, optionally `--max-error-ratio` and
`--require-top1`. The logits gate reuses `golden.ratio_allclose`, defaults to
zero allowed outliers, and applies to every step. A failed gate exits with
code 1; invalid input exits with code 2. `--require-top1` can also be used
alone when Top1 agreement is the complete acceptance criterion.

CPU regressions for the tools run without NPU allocation:

```bash
python -m pytest tools/test_deepseek_v41_accuracy.py -q
```

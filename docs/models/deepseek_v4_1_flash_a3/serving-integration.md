# Serving integration and validation scope

The prepared request-pool interface extends the shared generation session;
the standalone scalar Decode entry remains available. See
[Decode](decode.md#serving-request-pool) for the interface and
[model priorities](index.md#development-priorities) for compatibility targets.
The serving adapter belongs in the separate
[pypto-serving repository](https://github.com/hw-native-sys/pypto-serving),
under `models/deepseek_v41_a3/`.

## Integrated interfaces

- `max_requests_per_partition` reserves 1..32 request controls per DP
  partition without increasing the 16-token dispatch capacity.
- Prepared Prefill and `DecodeNetwork.step(chunk=...)` share the complete
  network, resident weights, workers and cache allocations.
- DP, slot and generation identify a request. Inactive requests keep their
  token histories across bounded dispatches. Live page mappings cannot
  overlap another request or shrink while owned.
- `GenerationSession.release_request` clears the retired request's physical
  pages and history, resets completed-dispatch selection workspace and
  requires a newer generation before reusing the slot. Cleanup failure
  invalidates the session; cleanup cannot mutate static weights.

## Version provenance

The device-accepted source pair is native commit
`c6dcabd3cf09e2c1e93758c9bb7b57e03feff824` and serving commit
`13aca1f8596e2d93cbf48536533953b35b0039d4`.
The native interface commit was rebased onto
`2685c70292b17401b6e89fb62ff545915f60cd33`, retaining that revision's Decode
module consolidation and logical worker-device mapping. The rebased
interface commit is `043c3856342940e45633466cfb076731857a1aee`.
Git range comparison reports an unchanged interface patch.

The former TP1 layer-streaming resource hooks are not carried forward:
the shared resident generation session supplies the service lifetime in the
current implementation. Historical branch code is not a second supported
serving path.

## Validation evidence

The accepted source pair completed the following checks on DP4/TP4/EP16:

- Full Prefill and multi-step Decode, with native/offline token equivalence.
- HTTP completions and chat, streaming and non-streaming, including a
  complete question and answer and normal stop handling.
- Multi-chunk 129/130-token prompts across the 128-token page boundary,
  nonzero history and inactive partitions.
- Per-DP capacities 2 and 32, with observed global active peaks 8 and 128.
  The highest tier submitted 256 requests: 254 completed, one was cancelled
  and one timed out as intended. Completed outputs and interrupted prefixes
  matched the serial baselines. Queue refill and slot-generation reuse were
  exercised.
- Clean installation/replay, 149 selected serving HOST tests and 12 native
  resident-session HOST contract tests, plus resource teardown.

On the rebased source, the resident-session and Prefill-program HOST tests
pass together: 15 tests, with three existing PyPTO inline-direction
deprecation warnings. All repository pre-commit hooks pass. These checks do
not execute NPU inference or validate the consolidated Decode kernels.

## Remaining validation and deployment requirements

1. **Rebased device execution and source pairing.** No fresh NPU run is
   claimed for the rebased source. The upstream consolidation also changes
   model files and worker-device mapping. Repeat native/serving equivalence
   and the affected lifecycle checks before assigning device acceptance to
   this version. Serving commit `13aca1f` deliberately accepts only the
   recorded native commit/tree; pointing `PYPTO_LIB_ROOT` at the rebased
   branch is rejected. Update the serving source pair and its reproducible
   delivery artifacts as a separate validated change. Do not bypass that
   identity check.
2. **Independent model accuracy.** Full-network independent numerical
   accuracy and dataset evaluation remain `not_run`. Integration token
   equivalence and a complete answer do not establish model accuracy.
3. **Capacity outside the measured cases.** The 128-active result uses short
   requests and configured context 256. It does not establish 32 requests
   per DP at arbitrary context lengths, larger capacities, or all supported
   model contexts. Ten-second device HBM sampling includes the device
   baseline and can miss short peaks; it is not a peak-memory or long-run
   leak bound.
4. **Other topology and runtime work.** TP1/TP2 full-network compatibility,
   alternative sharding, and uncommitted dynamic metadata/cache-pool work
   are not covered by this serving acceptance. They require their own
   reviewed implementation and validation.
5. **Optional serving modes.** Prefix caching, mixed Prefill/Decode, async
   scheduling, speculative/MTP/DSpark and disaggregated serving are outside
   the accepted greedy path. Keep unsupported modes explicitly disabled or
   rejected until their state contracts are implemented and validated.

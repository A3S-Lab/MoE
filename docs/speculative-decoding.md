# Model-neutral Speculative Decoding

The DSpark-inspired path is a Power runtime capability, not a Qwen3.8 feature.
Qwen3.6, Qwen3.8, Qwen3-MoE, OLMoE, picolm, and later architectures may use the
same orchestration contract when they supply a compatible drafter and target
verification adapter.

The design follows the separation demonstrated by DeepSeek's
[DeepSpec](https://github.com/deepseek-ai/DeepSpec) project and
[DSpark paper](https://arxiv.org/abs/2607.05147): a parallel draft backbone, a
lightweight prefix-dependent head, confidence-guided verification length, and
lossless target verification. Power generalizes the execution protocol; it
does not embed DeepSeek- or Qwen-specific tensors.

## Ownership

Power owns:

- bounded draft and verification batches;
- exact greedy acceptance and distribution-correct sampling rejection;
- begin, commit-prefix, and rollback orchestration for speculative state;
- confidence-, queue-, memory-, and device-aware verification scheduling;
- cancellation, admission, continuous-batch fairness, and telemetry;
- correctness receipts that contain digests and counters, never token content.

Each model crate owns:

- draft checkpoint discovery, tensor names, layouts, and kernels;
- the parallel backbone and sequential or Markov draft head;
- target block verification and logits production;
- snapshots for every architecture-specific mutable state;
- state replay or prefix retention after partial acceptance;
- tokenizer compatibility between drafter and target.

Qwen3.6 therefore supplies transactions for full-attention KV state plus
Gated DeltaNet recurrent and convolution state. A conventional transformer
usually supplies only KV transactions. Power never assumes either layout.

## Runtime Flow

```text
model drafter adapter
  -> DraftProposal(tokens, confidence, adapter revision)
  -> Power scheduler selects a bounded prefix
  -> model target adapter verifies the prefix in one block
  -> Power applies exact acceptance
       -> full accept: commit prefix and optional bonus token
       -> partial accept: retain accepted prefix, discard suffix, apply correction
       -> cancellation/error: restore the transaction checkpoint
  -> telemetry updates the next hardware-aware draft budget
```

A proposal is advisory. Target verification remains the source of truth. The
runtime must be able to disable speculation for an individual request without
changing its decoding policy or output distribution.

## Exactness Contract

Greedy decoding must emit exactly the same token IDs as non-speculative target
decoding. Sampling must use the target and draft distributions with a proven
rejection rule; matching only top-1 IDs is not sufficient for stochastic
decoding. RNG advancement is part of the transaction and must be reproducible
for a fixed seed.

A verification transaction covers all mutable generation state:

- attention KV entries;
- recurrent and convolution state;
- token history used by penalties;
- grammar or structured-output state;
- RNG state;
- streaming decoder buffers and stop-sequence state.

On partial acceptance, the state digest after retaining the accepted prefix
and applying the correction must match ordinary target decoding of the same
tokens. On cancellation or failure, the pre-draft digest must be restored.

## Scheduling Contract

The scheduler selects a verification prefix from declared hard bounds and
observed signals. Hard bounds win over confidence:

1. remaining output and context capacity;
2. model adapter maximum draft length;
3. target verification batch capacity;
4. device and host memory headroom;
5. continuous-batch latency and fairness budget;
6. drafter confidence and recent accepted-token ratio.

The initial implementation should be deterministic for a fixed evidence input.
A later calibrated policy may use per-device cost models, but it must emit the
inputs, selected prefix length, accepted length, and timing counters required
to reproduce its decision.

## Adapter Families

The runtime contract deliberately supports several proposal sources:

- prompt lookup and n-gram proposals as zero-weight correctness baselines;
- a model-provided MTP head as an integration and block-verification baseline;
- a trained DSpark-style parallel backbone plus prefix-dependent head;
- an independent small draft model with tokenizer-compatible output.

The existing picolm prompt-lookup and n-gram implementation is a baseline, not
a trained DSpark model. It should migrate to the shared Power primitives so
that exact acceptance and scheduling have one source of truth.

Qwen3.6-35B-A3B is the first target adapter because its text path and mixed
state transactions already pass public-checkpoint parity on CPU and CUDA.
Qwen3.8-27B is a second dense/hybrid target adapter, not the owner or limiting
scope of the feature. It additionally needs its own target streaming and
quantized execution path before a DSpark speed claim is meaningful on a 24 GiB
GPU.

## Delivery Plan

1. Complete: Power exposes the existing prompt lookup, n-gram, adaptive-length,
   and exact-acceptance primitives through `a3s_power::speculative`; picolm
   consumes compatibility exports and preserves same-seed token parity.
2. Add typed proposal, verification, scheduler-input, decision, and telemetry
   structures with validation and deterministic unit tests.
3. Add an architecture-owned transaction adapter to Qwen3.6 and verify greedy
   token parity plus state-digest parity under full, partial, zero acceptance,
   cancellation, and injected failure.
4. Implement block target verification on CUDA. Use prompt lookup or the
   checkpoint's MTP component only as a baseline; report its acceptance and
   overhead separately.
5. Train or import a provenance-pinned DSpark-compatible Qwen3.6 drafter, then
   measure end-to-end latency and shared-server throughput against the current
   `0.263611 tokens/s` F32 CUDA baseline.
6. Add the Qwen3.8 target and drafter adapters through the same Power contract.
7. Tune continuous batching, quantization, fused kernels, and cache residency
   using checked evidence rather than model-name branches.

## Acceptance Gates

- Greedy token IDs match ordinary target decoding for every fixture and public
  checkpoint prompt.
- Seeded sampling passes distribution and replay tests with speculation both
  enabled and disabled.
- State digests match after full, partial, and zero-token acceptance.
- Cancellation and injected failures restore the pre-draft state and release
  all admission and residency resources.
- Evidence reports proposal length, verified length, accepted length, bonus or
  correction count, acceptance ratio, drafter time, verifier time, rollback
  time, TTFT, end-to-end token rate, concurrent throughput, and peak memory.
- A speedup is accepted only when the same model digest, quantization, prompt
  set, sampling policy, cache bounds, concurrency, and hardware are compared.
- M9 completes only after Qwen3.6 and at least one additional architecture pass
  exactness gates and show a positive measured end-to-end speedup.

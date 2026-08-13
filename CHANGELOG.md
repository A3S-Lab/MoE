# Changelog

All notable changes to this project are documented in this file.

## Unreleased

### Added

- Strict OLMoE configuration and sparse-layer geometry validation.
- Exact full-softmax top-k routing backed by Power's routed expert contract.
- CPU reference implementation of fused gate/up expert execution and weighted
  reduction.
- Pinned tiny numerical oracle and reproducible fixture generator.
- Fully resident F32 OLMoE CPU decoder with causal GQA, split-half RoPE, Q/K
  normalization, transactional KV cache, and greedy generation.
- Strict Hugging Face checkpoint index and shard-name validation for the exact
  published tensor contract.
- Vocabulary-bounded tokenizer loading, encoding, and decoding.
- Complete-decoder oracle, prefill/decode parity coverage, and a range-request
  verifier for the pinned public SafeTensor headers.
- Versioned lossless F32/BF16 expert records and deterministic bounded-buffer
  checkpoint conversion with atomic publication and digest manifests.
- Power-backed asynchronous expert staging, cache reuse and eviction,
  cancellation-safe full-decoder execution, and one-per-request admission.
- `a3s-moe-pack` CLI with machine-readable conversion evidence.
- Ragged expert-unioned forward batches with independent session KV caches and
  isolated-execution numerical parity.
- Continuous greedy scheduling on Power's fair execution lifecycle with
  cancellation reaping, slot compaction, atomic state publication, and
  digest-only evidence.
- Deterministic request-local sampling with temperature, top-p, top-k, min-p,
  repeat, frequency, and presence controls.
- Stable incremental tokenizer decoding and stop-sequence buffering for
  streamed output.
- Architecture-aware `OlmoeBackend`, bounded continuous-batch worker, and
  `a3s-moe-server` composition through Power's OpenAI-compatible transport.
- `a3s-moe-bench` JSON evidence for first/warm TTFT and throughput, Power expert
  I/O/cache telemetry, peak RSS, and an isolated resident CPU baseline.
- Service, benchmark, and real HTTP subprocess regression coverage.
- Pinned-Transformers public-checkpoint oracle generation plus an
  integrity-bound `a3s-moe-validate` CLI that reports full logits, router, and
  route parity as versioned JSON evidence.
- Typed CPU, CUDA, Metal, and automatic device selection for the Power service
  and benchmark, including device-tier expert staging and explicit automatic
  CPU fallback evidence.

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

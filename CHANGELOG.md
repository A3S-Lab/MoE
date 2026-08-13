# Changelog

All notable changes to this project are documented in this file.

## Unreleased

### Added

- Strict OLMoE configuration and sparse-layer geometry validation.
- Exact full-softmax top-k routing backed by Power's routed expert contract.
- CPU reference implementation of fused gate/up expert execution and weighted
  reduction.
- Pinned tiny numerical oracle and reproducible fixture generator.

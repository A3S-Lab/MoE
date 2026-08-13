# Roadmap

| Milestone | Status | Exit criterion |
| --- | --- | --- |
| M0 Contracts and oracle | Complete | Exact OLMoE routes and sparse-layer output match pinned fixtures; packed `U8` records survive Power staging. |
| M1 CPU correctness | In progress | Resident decoder, loader, tokenizer, and KV cache pass tiny oracles; full public-checkpoint parity still required. |
| M2 Streaming residency | Complete | Exact routed experts execute through the sole Power hierarchy under a measured memory bound. |
| M3 Continuous batching | Complete | Route-unioned batches retain single-request parity and load each active expert once per step. |
| M4 Service and performance | Planned | Power backend, OpenAI streaming, chat template, sampling, and reproducible benchmark evidence ship. |
| M5 GPU execution | Planned | Accelerator path passes CPU parity with explicit fallback evidence. |
| M6 TEE | Planned | Seekable encrypted records and confidential execution pass memory, integrity, and cancellation tests. |
| M7 Second architecture | Planned | A second MoE family lands without model-specific changes to Power's runtime contracts. |

Detailed ownership and acceptance rules live in
[docs/architecture.md](docs/architecture.md).

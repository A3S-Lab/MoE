# Roadmap

| Milestone | Status | Exit criterion |
| --- | --- | --- |
| M0 Contracts and oracle | Complete | Exact OLMoE routes and sparse-layer output match pinned fixtures; packed `U8` records survive Power staging. |
| M1 CPU correctness | Complete | Resident decoder, loader, tokenizer, and KV cache pass tiny and pinned public-checkpoint numerical oracles. |
| M2 Streaming residency | Complete | Exact routed experts execute through the sole Power hierarchy under a measured memory bound. |
| M3 Continuous batching | Complete | Route-unioned batches retain single-request parity and load each active expert once per step. |
| M4 Service and performance | Complete | Service, sampling, streaming, HTTP/SSE smoke tests, and a representative public-checkpoint CPU artifact pass. |
| M5 GPU execution | In progress | Device-native graph and explicit fallback evidence are implemented; CUDA/Metal public-checkpoint parity remains. |
| M6 TEE | In progress | Seekable encrypted records pass bounded-memory, integrity, cancellation, and service tests; attested key release on a real confidential host remains. |
| M7 Second architecture | Complete | Qwen3-MoE resident decoding, bounded conversion, Power expert streaming, service composition, and pinned public-checkpoint acceptance pass. |
| M8 Qwen3.6 text architecture | Complete | The exact pinned Qwen3.6-35B-A3B checkpoint passes hashing, bounded conversion, independent-oracle parity, concurrent HTTP inference, and target-host cold/warm measurement. |

Detailed ownership and acceptance rules live in
[docs/architecture.md](docs/architecture.md).

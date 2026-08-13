# A3S MoE

`a3s-moe` provides model-owned mixture-of-experts inference for
[A3S Power](https://github.com/A3S-Lab/Power). Power remains responsible for
model-neutral scheduling, weight residency, integrity, devices, and service
composition. This crate owns architecture-specific tensor names, layouts,
routing equations, kernels, KV-cache semantics, tokenization, and generation.

The supported architectures are
[OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924) and
[Qwen3-30B-A3B-Base](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base).
OLMoE has 64 experts per layer with 8 selected per token, 7B total parameters,
and approximately 1B active parameters. Qwen3-MoE supplies a second
architecture boundary without changing Power's model-neutral residency core.

## Current Status

The M0 numerical contract, resident M1 CPU engine, bounded M2 expert streaming,
M3 continuous batching path, M4 service path, M6 encrypted-weight foundation,
and M7 Qwen3-MoE inference/service path are implemented and tested:

- Hugging Face compatible OLMoE configuration parsing and strict geometry
  validation.
- Full-softmax top-k routing with OLMoE's default unnormalized selected
  probabilities.
- Exact fused gate/up ordering, SiLU activation, down projection, route
  weighting, and expert reduction.
- A pinned tiny oracle covering router logits, selected experts, route weights,
  and final hidden states.
- Conversion of exact model routes into Power's `RoutedExpertBatch` without
  substitution, reordering, or renormalization.
- Complete embeddings, RMSNorm, Q/K normalization, split-half RoPE, grouped
  query causal attention, transactional KV cache, residual decoder layers,
  final norm, and LM head.
- Safe validation of the published Hugging Face index before any tensor bytes
  are loaded, plus a deliberately fully resident F32 correctness loader.
- GPT-NeoX tokenizer loading, vocabulary-bound encode/decode, and greedy
  prefill/decode generation.
- Full-prefill versus incremental-decode parity and a pinned complete-decoder
  oracle covering logits and every layer's routes.
- A versioned lossless F32/BF16 packed expert format with strict header,
  dimension, length, finite-value, and dtype validation.
- A deterministic bounded-buffer converter that publishes a packed checkpoint
  atomically and records source, dense, and expert collection digests.
- A complete asynchronous decoder that keeps dense weights resident, obtains
  the exact routed expert union from Power, computes ready experts while other
  records load, and restores canonical reduction order.
- One Power admission permit per generation request, transactional cancellation,
  measured cache bounds, and resident-versus-streaming parity for prefill and
  incremental greedy decode.
- Ragged continuous batches with independent per-session attention/KV state,
  one route-unioned expert staging operation per layer, and canonical per-row
  outputs matching isolated inference.
- A fair greedy scheduler built on Power's execution lifecycle, including
  bounded admission, direct cancellation-token reaping, slot compaction,
  exact KV state-byte accounting, and digest-only step/lifecycle evidence.
- Deterministic per-request sampling with temperature, top-p, top-k, min-p,
  repeat, frequency, and presence penalties without weakening batched route
  union semantics.
- An architecture-aware Power backend, process-local model manifest, bounded
  service queue, stable incremental UTF-8 decoding, stop-sequence buffering,
  and cancellation when an HTTP stream is abandoned.
- A standalone Power-composed server for OpenAI chat and completion endpoints,
  plus a JSON benchmark that records application-cold and warm generation,
  TTFT, expert bytes read, cache state, peak RSS, and an isolated resident CPU
  baseline.
- Independently authenticated, seekable AES-256-GCM dense and expert
  collections, bound together with plaintext metadata by a pinned checkpoint
  manifest and consumed through the same Power residency hierarchy.
- An `a3s-moe-encrypt` CLI and typed encrypted service source that keep keys out
  of arguments, logs, model manifests, and decrypted intermediate files.
- A fully resident Qwen3-MoE F32 CPU correctness backend with its distinct
  attention head dimension, per-head Q/K normalization, GQA, RoPE,
  sparse/dense layer schedule, normalized top-k policy, official per-expert
  tensors or fused exporter tensors, transactional KV cache, and greedy
  decoding.
- A dependency-free full Qwen3-MoE decoder oracle covering logits, router
  logits, exact routes, dense-to-sparse layer transitions, prefill/decode
  parity, and sliding attention over the shared Power routing boundary.
- Strict Qwen3-MoE Hugging Face shard-index validation for either the official
  18,867-tensor split-expert layout or the 531-tensor fused exporter layout,
  resident checkpoint loading, and the shared vocabulary-bounded
  tokenizer/stream decoder.
- Bounded Qwen3-MoE conversion that reads official gate/up/down matrices one
  expert at a time, reads fused 3-D exporters through verified Power tensor
  subranges, streams oversized dense tensors in chunks, and publishes one
  atomic packed record for each sparse-layer expert.
- A Qwen3-MoE streaming decoder that shares the resident attention, dense MLP,
  normalization, and transactional KV-cache implementation while delegating
  all expert residency to one Power hierarchy.
- Shared continuous scheduling adapters that retain architecture-specific
  route-union output types while reusing Power admission, lifecycle,
  cancellation, sampling, and KV accounting for OLMoE and Qwen3-MoE.
- A Qwen3-MoE Power backend and server composition path with automatic model
  family detection, concurrent request batching, OpenAI completion/chat
  streaming, and the same fail-closed request policy as OLMoE.
- A pinned public Qwen3-MoE acceptance contract and independent Transformers
  BF16 oracle generator, plus an architecture-aware Rust validator and
  performance-evidence harness. The public 30B-A3B run remains an explicit
  acceptance gate until its checked reports are committed.

The HTTP transport, OpenAI response framing, authentication, rate limiting,
metrics, and shutdown lifecycle remain owned by Power. Dense weights remain
resident in the current CPU path. The complete pinned 13.8 GB public
checkpoint has passed Transformers-to-Rust numerical validation, bounded
conversion, Power-streamed generation, resident token-parity comparison, and
real HTTP completion/SSE smoke tests. The published OLMoE checkpoint is a base
model and does not declare a chat template, so the server uses an explicit
generic transcript unless `--chat-template` is supplied for a compatible
fine-tune. Checked raw evidence lives under [`evidence/`](evidence/).

## Architecture Boundary

```text
a3s-moe (model owner)
  config / tensor names / OLMoE + Qwen3-MoE math / tokenizer / generation
                         |
                         | exact routes + atomic weight requests
                         v
a3s-power (runtime owner)
  admission / batching / storage-RAM-device residency / integrity / TEE / API
```

There is one residency hierarchy. Model code consumes weights returned by
Power and does not introduce a second expert cache. Each packed expert remains
an opaque `U8` SafeTensor to Power; this crate validates and interprets its
versioned header and exact scalar payload.

See [Architecture](docs/architecture.md) for invariants and the delivery plan.

## Development

```shell
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
python tools/test_generate_public_oracle.py
python tools/test_generate_qwen3_moe_oracle.py
python tools/test_generate_qwen3_moe_full_oracle.py
python tools/test_generate_qwen3_moe_public_oracle.py
```

Verify the pinned public checkpoint's 3,219 tensor headers without downloading
the 13.8 GB payload:

```shell
python tools/verify_hf_contract.py
```

Generate and verify the full public-checkpoint numerical oracle from the exact
pinned Transformers checkout:

```shell
PYTHONPATH=/src/transformers/src python tools/generate_public_oracle.py \
  /models/OLMoE-1B-7B-0924 tests/fixtures/olmoe_public_oracle.json
cargo run --release --features validation --bin a3s-moe-validate -- \
  /models/OLMoE-1B-7B-0924 tests/fixtures/olmoe_public_oracle.json \
  > olmoe-validation.json
```

The generator refuses a Transformers Git checkout other than
`918dbf131d0df5b46e3f6e1d96174d62aa4d16d6`, or an OLMoE source file whose
SHA-256 differs from the pinned digest. It also verifies the exact byte length
and SHA-256 of every checkpoint file from revision
`6d84c48581ece794365f2b8e9cfb043c68ade9c5` before loading the model. The oracle
binds that inventory and captures every prompt logit, router logit, selected
expert, and route weight. The validator exits non-zero on provenance, tokenizer,
argmax, route, or tolerance failure and always emits a versioned JSON report
for a structurally valid numerical comparison.

Generate the independent public Qwen3-MoE oracle after downloading the exact
`Qwen/Qwen3-30B-A3B-Base` revision pinned in
`tools/qwen3_moe_public_contract.py`, then compare it with the packed
Power-streaming decoder:

```shell
PYTHONPATH=/src/transformers/src python \
  tools/generate_qwen3_moe_public_oracle.py \
  /models/Qwen3-30B-A3B-Base qwen3-moe-oracle.json
cargo run --release --features validation --bin a3s-moe-validate -- \
  /models/Qwen3-30B-A3B-Base qwen3-moe-oracle.json \
  --packed-checkpoint /models/Qwen3-30B-A3B-Base-a3s \
  --host-cache-mib 4096 > qwen3-moe-validation.json
```

The generator pins the model revision, all 16 shard byte lengths and SHA-256
digests, the Transformers Git revision, its Qwen3-MoE source digest, CPU BF16
execution, eager attention, and the input token IDs. The Rust validator
re-hashes those files and the packed source binding before comparing every
captured logit, router logit, route weight, selected expert, and token argmax.

Convert a downloaded Hugging Face checkpoint without buffering a complete
layer or model:

```shell
cargo run --release --bin a3s-moe-pack -- \
  /models/OLMoE-1B-7B-0924 /models/OLMoE-1B-7B-0924-a3s \
  --experts-per-file 8 --max-buffer-mib 512
```

The destination is created only after conversion and digest validation
complete. The command refuses to overwrite an existing destination and emits a
JSON conversion report containing the observed peak buffered bytes. It reads
`model_type` from the validated source configuration and accepts both `olmoe`
and `qwen3_moe`. Each family supplies explicit Power resource limits: Qwen3-MoE
permits at most 64 GiB of source or packed weights and 8,192 SafeTensor files,
which covers the pinned 61 GB checkpoint and the worst supported
one-expert-per-file packing without weakening Power's global defaults. Its
entrypoint residency profile also permits one bounded 4 GiB current-layer
expert union while keeping concurrent reads within 512 MiB; this covers all
128 public-model experts in lossless F32 form. Explicit library policies are
never widened during loading. For example:

```shell
cargo run --release --bin a3s-moe-pack -- \
  /models/Qwen3-30B-A3B-Base /models/Qwen3-30B-A3B-Base-a3s \
  --experts-per-file 8 --max-buffer-mib 512
```

Encrypt a packed checkpoint with a key supplied by the process environment:

```shell
export A3S_MOE_WEIGHT_KEY="$(openssl rand -hex 32)"
cargo run --release --bin a3s-moe-encrypt -- \
  /models/OLMoE-1B-7B-0924-a3s /models/OLMoE-1B-7B-0924-a3s-encrypted \
  --key-env A3S_MOE_WEIGHT_KEY --chunk-mib 1 \
  > olmoe-encryption.json
```

Store the emitted `manifestSha256` as the checkpoint's out-of-band trust
anchor. Dense and expert weights are encrypted; `config.json`, `manifest.json`,
and the optional tokenizer remain plaintext but are SHA-256-bound by that
trusted manifest. Encryption and opening use bounded chunks and never publish
decrypted intermediate files.

Serve the packed checkpoint through Power's OpenAI-compatible API:

```shell
cargo run --release --features server --bin a3s-moe-server -- \
  /models/OLMoE-1B-7B-0924-a3s \
  --model olmoe-1b-7b --device cpu --host-cache-mib 512 \
  --max-concurrent-requests 4
```

The server detects `olmoe` or `qwen3_moe` from the checkpoint's bounded
`config.json`, injects the corresponding typed backend, and registers a
process-local manifest. Supported sampling controls are
`temperature`, `top_p`, `top_k`, `min_p`, `seed`, `repeat_penalty`,
`repeat_last_n`, `frequency_penalty`, and `presence_penalty`. Unsupported
modalities, tools, structured output, cross-request KV sessions, and backend
knobs fail before inference.

Serve a packed Qwen3-MoE checkpoint through the same binary; omitting
`--model` selects the architecture-specific default `qwen3-moe` identifier:

```shell
cargo run --release --features server --bin a3s-moe-server -- \
  /models/Qwen3-30B-A3B-Base-a3s --device cpu \
  --host-cache-mib 512 --max-concurrent-requests 4
```

Encrypted service loading currently applies only to the OLMoE confidential
checkpoint envelope and fails closed for Qwen3-MoE.

Serve the encrypted form by supplying both its pinned trust anchor and the
environment variable that owns the key:

```shell
cargo run --release --features server --bin a3s-moe-server -- \
  /models/OLMoE-1B-7B-0924-a3s-encrypted --model olmoe-1b-7b \
  --encrypted-manifest-sha256 <manifestSha256> \
  --encrypted-key-env A3S_MOE_WEIGHT_KEY
```

Production confidential hosts should construct the typed encrypted source
from their attested key-release mechanism. The environment-based CLI is an
operator boundary that avoids command-line key exposure; it is not itself a
remote-attestation protocol.

The same graph can run on a Power-resolved accelerator. Build exactly one
platform feature and select a typed device explicitly:

```shell
cargo run --release --features server,cuda --bin a3s-moe-server -- \
  /models/OLMoE-1B-7B-0924-a3s --device cuda:0 \
  --host-cache-mib 512 --device-cache-mib 4096
```

Metal uses `--features server,metal --device metal:0` on macOS. An explicit
CUDA or Metal request fails if that backend or ordinal is unavailable. `auto`
is the only mode allowed to fall back to CPU, and the resolved device plus the
fallback decision are exposed as content-free service/benchmark evidence.

Produce raw, reproducible performance evidence and optionally compare with the
fully resident CPU path in an isolated child process:

```shell
cargo run --release --features benchmark --bin a3s-moe-bench -- \
  /models/OLMoE-1B-7B-0924-a3s \
  --prompt "Bitcoin is" --max-tokens 32 --warm-samples 5 \
  --checkpoint-label olmoe-1b-7b-bf16 \
  --resident-checkpoint /models/OLMoE-1B-7B-0924 \
  > olmoe-performance.json
```

Qwen3-MoE uses the same harness and a family-specific evidence schema. Its
public parity gate is the independent BF16 oracle, so it deliberately omits
the memory-intensive resident F32 child:

```shell
cargo run --release --features benchmark --bin a3s-moe-bench -- \
  /models/Qwen3-30B-A3B-Base-a3s \
  --prompt "Bitcoin is" --max-tokens 8 --warm-samples 3 \
  --host-cache-mib 4096 --checkpoint-label qwen3-30b-a3b-bf16 \
  > qwen3-moe-performance.json
```

The first sample starts with an empty Power expert cache. Warm samples retain
only the configured bounded cache. The report explicitly labels the operating
system page cache as uncontrolled; it does not call that condition physical
cold I/O. See [Performance Evidence](docs/performance.md) for the measurement
boundary and comparison rules.

CUDA and Metal are mutually platform-specific Cargo features, so portable CI
uses `--features server,benchmark,validation` rather than `--all-features`.
CUDA and macOS runners must compile and test `cuda` and `metal` separately.

Regenerate the tiny oracle only when intentionally changing the pinned model
contract:

```shell
python tools/generate_tiny_oracle.py
```

The generator prints JSON to stdout and never overwrites the checked fixture.
Review changes before replacing `tests/fixtures/olmoe_tiny_oracle.json`.
The same rule applies to `generate_full_model_oracle.py` and its complete
decoder fixture.

The Qwen3-MoE sparse-layer and full-decoder fixtures follow the same
review-only workflow:

```shell
python tools/generate_qwen3_moe_oracle.py
python tools/generate_qwen3_moe_full_oracle.py
```

M7 now provides resident CPU reference inference, strict official split and
fused-exporter checkpoint loading, tokenizer integration, bounded conversion,
Power-backed expert streaming, route-unioned continuous batching, service
composition, and pinned public acceptance tooling for the second family. It
reuses Power's model-neutral `RoutedExpertBatch`, verified tensor-range I/O,
lifecycle, and sole residency hierarchy. Pinned public-model numerical and
performance reports remain pending, so the implementation is not yet
presented as a production-accepted Qwen3-MoE deployment.

## License

MIT

# A3S MoE

`a3s-moe` provides model-owned mixture-of-experts inference for
[A3S Power](https://github.com/A3S-Lab/Power). Power remains responsible for
model-neutral scheduling, weight residency, integrity, devices, and service
composition. This crate owns architecture-specific tensor names, layouts,
routing equations, kernels, KV-cache semantics, tokenization, and generation.

The first supported architecture is
[OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924): 64 experts per
layer with 8 selected per token, 7B total parameters, and approximately 1B
active parameters.

## Current Status

The M0 numerical contract, resident M1 CPU engine, bounded M2 expert streaming,
M3 continuous batching path, M4 service path, and M6 encrypted-weight
foundation are implemented and tested:

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

The HTTP transport, OpenAI response framing, authentication, rate limiting,
metrics, and shutdown lifecycle remain owned by Power. Dense weights remain
resident in the current CPU path. The 13.8 GB public checkpoint's metadata
contract is verified without downloading weight payloads; a full
real-checkpoint conversion, numerical run, and representative performance
artifact remain acceptance items. The published OLMoE checkpoint is a base
model and does not declare a chat template, so the server uses an explicit
generic transcript unless `--chat-template` is supplied for a compatible
fine-tune.

## Architecture Boundary

```text
a3s-moe (model owner)
  config / tensor names / OLMoE math / tokenizer / generation
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

Convert a downloaded Hugging Face checkpoint without buffering a complete
layer or model:

```shell
cargo run --release --bin a3s-moe-pack -- \
  /models/OLMoE-1B-7B-0924 /models/OLMoE-1B-7B-0924-a3s \
  --experts-per-file 8 --max-buffer-mib 512
```

The destination is created only after conversion and digest validation
complete. The command refuses to overwrite an existing destination and emits a
JSON conversion report containing the observed peak buffered bytes.

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

The server keeps HTTP concerns in Power and injects both the typed OLMoE
backend and a process-local manifest. Supported sampling controls are
`temperature`, `top_p`, `top_k`, `min_p`, `seed`, `repeat_penalty`,
`repeat_last_n`, `frequency_penalty`, and `presence_penalty`. Unsupported
modalities, tools, structured output, cross-request KV sessions, and backend
knobs fail before inference.

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

## License

MIT

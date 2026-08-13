# MoE Inference Architecture

## Goals

The engine targets correct, bounded-memory inference for sparse MoE models on
CPU, accelerators, and confidential-computing hosts. It adopts the useful
Colibri pattern—storage, RAM, and device tiers with just-in-time expert
movement—without coupling generic Power runtime code to one model family.

OLMoE established the first model contract. Qwen3-MoE is the second admitted
architecture and changes head geometry, routing normalization, expert source
layout, and the dense/sparse layer schedule without modifying Power's
residency core.

## Ownership

| Concern | Owner |
| --- | --- |
| Model configuration and tensor aliases | `a3s-moe` |
| Router math and exact top-k selection | `a3s-moe` |
| Attention, expert kernels, KV cache, tokenizer | `a3s-moe` |
| Packed expert header and quantization interpretation | `a3s-moe` |
| Admission, cancellation, and continuous batching | `a3s-power` |
| Storage/RAM/device residency and eviction | `a3s-power` |
| Weight source integrity and TEE evidence | `a3s-power` |
| OpenAI-compatible transport | `a3s-power`, composed downstream |

Dependency direction is one way: `a3s-moe -> a3s-power`. Power never imports a
model implementation. The `a3s-moe` service binary injects its typed backend
through `PowerServerBuilder`.

## Numerical Invariants

OLMoE inference follows the reference order:

1. Compute router logits with a bias-free linear projection.
2. Apply softmax across all experts in F32.
3. Select exactly `num_experts_per_tok` experts.
4. Keep selected full-softmax probabilities when `norm_topk_prob` is false.
5. Split fused `gate_up_proj` into gate rows followed by up rows.
6. Compute `down(silu(gate(x)) * up(x))` for each selected expert.
7. Multiply by that route's weight and add into the original token position.

The Power batch union is solely an I/O schedule. It cannot alter steps 2–7.
Duplicate, out-of-range, empty, or non-finite routes fail closed.

## Weight Representations

The correctness path first reads the published Hugging Face SafeTensor layout.
Dense weights may remain ordinary tensors. Expert weights are addressed by
layer and expert, even when the source checkpoint stores all experts in a
single three-dimensional tensor.

The streaming path includes a deterministic conversion tool. One packed expert
record contains a fixed header followed by exact gate, up, and down scalar
bytes. The header declares its version, F32 or BF16 encoding, dimensions, and
section lengths. The complete record is stored as one `U8` SafeTensor so Power
can verify, stage, and cache it atomically. The expert collection digest in the
completion manifest binds every record name and byte. Unsupported versions,
trailing or truncated data, non-finite values, and dimension mismatches fail
closed.

Packed checkpoints separate `dense/` and `experts/`. Dense tensors retain their
source precision and are split one tensor per file, while expert files contain
a bounded number of atomic records from one layer. Power opens only
`experts/` for residency, preventing resident dense weights from being counted
or duplicated in the expert cache. The converter publishes the destination
only after both collections have been reopened and their digests recorded.
Qwen3-MoE source checkpoints fuse every layer's experts into two large 3-D
tensors. Conversion uses Power's integrity-verified tensor-subrange API to read
one expert slice at a time, and uses the same bounded range API to stream dense
tensors larger than the conversion budget into valid one-tensor SafeTensor
files. The declared buffer limit therefore applies without relying on the
source tensor being smaller than that limit.

Only `WeightHierarchy` owns resident bytes. Model code may hold short-lived
views for the current operation but cannot retain a parallel byte cache.

### Confidential checkpoint representation

The confidential form encrypts both packed collections with Power's seekable
AES-256-GCM container. Every chunk has an independent random nonce and binds
the file header plus chunk index as authenticated data. Power first performs a
complete bounded-chunk authentication pass that reconstructs each logical
collection digest; later tensor ranges decrypt only their covering chunks into
zeroizing buffers. No decrypted SafeTensor collection is written to disk.

An OLMoE-owned `confidential.json` binds the logical packed-weight digest,
plaintext collection digests, Power child-manifest digests, `config.json`, the
packed manifest, and the optional tokenizer. Callers must provide the top
manifest SHA-256 out of band through `OlmoeEncryptedCheckpointSource`; replacing
the checkpoint directory alone cannot replace that trust anchor. The same
Power `WeightHierarchy` then stages encrypted expert records, so confidential
loading does not introduce a model-owned cache.

Remote attestation and policy-controlled key release belong to the deployment
environment. The library accepts a zeroizing typed key owner; the CLI can load
one from an environment variable without placing key bytes in arguments or
logs.

## Execution Flow

```text
tokens -> embedding -> decoder layer
                         |
                         +-> attention + KV cache
                         |
                         +-> router -> exact RoutedExpertBatch
                                         |
                                         v
                         route-coupled microbatch union
                                         |
                                         v
                         Power stages each expert once
                                         |
                                         v
                         model kernel dispatch + weighted reduction
```

The CPU correctness backend executes one request without approximation. Later
milestones reuse the same oracle for batched, streaming, and accelerator paths.

## Delivery Milestones

### M0: Contract and Oracle

- Validate official OLMoE geometry.
- Pin a tiny independent numerical oracle.
- Prove one packed `U8` expert survives Power staging byte-for-byte.
- Establish typed downstream backend composition.

### M1: CPU Correctness

- Implemented: published SafeTensor index validation and resident sharded
  tensor loading.
- Implemented: embeddings, RMSNorm, Q/K normalization, RoPE, causal attention,
  sparse MLP layers, final norm, and LM head.
- Implemented: transactional prefill/decode KV cache and deterministic greedy
  generation.
- Implemented: pinned equation-level logits/routes oracle and
  prefill-versus-incremental parity.
- Implemented: integrity-bound public-oracle generation and a fail-closed
  resident Rust validator for full logits, router logits, routes, and tokenizer
  IDs.
- Accepted on the checked CPU host: the complete 13.8 GB pinned public
  checkpoint matches Transformers token IDs, all logits/router logits within
  declared tolerances, every selected expert, all route weights, and every
  argmax. The raw report is checked under `evidence/`.

### M2: Streaming Residency

- Implemented: deterministic, lossless F32/BF16 conversion into versioned
  atomic records under an explicit buffer bound.
- Implemented: exact route unions mapped one-to-one to Power staged weight
  groups with no model-specific cache.
- Implemented: ready-group compute overlaps remaining current-layer reads, with
  cancellation checks and ascending-expert reduction restored afterward.
- Implemented: complete decoder and greedy generation parity, transactional KV
  state, cache eviction evidence, and one admission permit per request.
- Deferred to performance tuning: speculative next-layer prefetch based on
  Power's route-coupling hints; hints never alter exact current-layer routes.

### M3: Continuous Batching

- Implemented: ragged rows may have different token widths and absolute KV
  positions; their attention and cache updates remain session-local.
- Implemented: normalized MLP rows flatten into one layer input, so the exact
  route union stages every active expert once before results are split back to
  canonical rows.
- Implemented: Power's fair execution lifecycle owns per-member permits,
  admission ordering, cancellation boundaries, aggregate state limits, and
  digest-only step transcripts.
- Implemented: the model scheduler commits lifecycle metadata before
  publishing continuing KV state, compacts completed/cancelled slots, and
  accepts new members for the next immutable roster.
- Implemented: an architecture adapter shares the scheduler state machine,
  sampling, lifecycle, and cancellation logic while OLMoE and Qwen3-MoE retain
  their own row, route-union, and staging report types. Mixed Qwen layers run
  dense MLPs per row and union routes only for sparse layers.

### M4: Service and Performance

- Implemented: an architecture-aware Power `Backend` accepts only
  `SafeTensors` manifests declaring its exact `olmoe` or `qwen3_moe` family and
  verifies the packed logical-weight digest before serving.
- Implemented: one model worker turns a bounded request channel into M3
  continuous batches. Each request retains independent sampling state,
  incremental decoder state, stop policy, cancellation, and response channel.
- Implemented: Power owns HTTP/OpenAI framing, authentication, outer
  concurrency limiting, metrics, and process lifecycle. The downstream binary
  detects the bounded checkpoint configuration, injects the corresponding
  typed backend and its process-local manifest through `PowerServerBuilder`,
  and rejects encrypted Qwen3-MoE input before loading.
- Implemented: deterministic temperature, top-p, top-k, min-p, repetition,
  frequency, and presence sampling; stable UTF-8 token streaming; and
  fail-closed unsupported request controls.
- Implemented: a JSON evidence harness measures application-cold and warm
  throughput, TTFT, expert storage bytes, cache telemetry, and process peak RSS.
  The resident CPU baseline runs in a separate child process.
- Accepted on the checked CPU host: complete-checkpoint conversion, Power
  streaming, real HTTP completion/SSE, eight-token resident parity, and raw
  first/warm performance evidence are checked under `evidence/`. The artifact
  explicitly does not claim physical cold I/O because the operating-system
  page cache was uncontrolled.

The official base OLMoE tokenizer configuration declares no chat template.
The service therefore uses a deterministic generic role transcript by default
and accepts an explicit Hugging Face compatible Jinja template for a matching
fine-tune. It does not present the base model as instruction-tuned.

### M5–M7: Accelerator, TEE, and Second Architecture

- M5 implemented foundation: the same dense and streamed-expert graph executes
  on Power's resolved CPU, CUDA, or Metal tensor device; expert staging uses
  the fastest configured tier; explicit accelerators fail closed; and `auto`
  exposes content-free CPU fallback evidence.
- M5 pending acceptance: run CUDA and Metal parity against the public oracle
  on matching hardware and check in dtype-specific evidence.
- M6 implemented foundation: bounded-chunk encryption and authenticated random
  access cover dense and expert collections; a pinned top manifest binds all
  plaintext metadata; the typed service source reuses Power residency without
  decrypted intermediates; wrong keys, replacement, tampering, cancellation,
  memory bounds, numerical parity, and service generation are tested.
- M6 pending acceptance: integrate an attested key-release provider and record
  peak-memory and inference evidence on a real confidential-computing host.
- M7 implemented inference path: Qwen3-MoE validates its independent head
  dimension, sparse/dense layer schedule, MoE-specific intermediate width, and
  normalized top-k policy. Its resident F32 CPU decoder implements per-head
  Q/K normalization, GQA, RoPE, optional sliding attention, fused 3-D expert
  tensors, canonical reduction, transactional shared KV cache, and greedy
  generation. A dependency-free full-decoder fixture validates logits, router
  logits, and unchanged Power `RoutedExpertBatch` routes across dense and
  sparse layers. The second family also reuses a single hardened SafeTensor
  shard-index loader and vocabulary-bounded tokenizer while supplying its own
  exact mixed dense/sparse tensor inventory.
- M7 implemented streaming path: fused 3-D expert arrays are converted through
  verified bounded subranges into the shared lossless expert record; oversized
  dense tensors copy in bounded chunks; exact packed inventories and digests
  fail closed; and mixed dense/sparse execution shares the resident decoder
  state while every sparse layer uses one Power hierarchy. Tiny F32 forward,
  BF16 generation, exact-route, cache-bound, and CLI auto-detection tests pass.
- M7 implemented service path: the shared continuous scheduler preserves
  architecture-specific route unions, and a typed Qwen3-MoE backend passes
  concurrent completion/chat generation plus real Power HTTP completion and
  SSE tests. The server automatically selects the model family from
  `config.json`.
- M7 pending acceptance: pin and run the public Qwen3-MoE checkpoint numerical
  and performance gates.

## Acceptance Gates

Every optimized path must retain exact expert IDs and match reference route
weights and logits within a declared dtype-specific tolerance. Tests must cover
malformed configs, truncated/corrupt weights, cancellation, cache pressure,
mixed routes, and deterministic fallback. Performance claims require a checked
benchmark artifact containing hardware, storage cache state, model digest,
configuration, and raw samples.

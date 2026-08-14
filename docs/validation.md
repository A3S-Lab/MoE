# Validation Evidence

## Pinned Sources

- Transformers OLMoE equations:
  `918dbf131d0df5b46e3f6e1d96174d62aa4d16d6`
  (`modeling_olmoe.py` SHA-256
  `53a94a479f9904674a5f45aba0387c13466a1f2a2d3cdb9226f9cf58946ebbf8`)
- OLMoE-1B-7B-0924 checkpoint:
  `6d84c48581ece794365f2b8e9cfb043c68ade9c5`
- Transformers Qwen3-MoE equations:
  `918dbf131d0df5b46e3f6e1d96174d62aa4d16d6`
  (`modeling_qwen3_moe.py` SHA-256
  `56d820671d810b68f31056605cec0c674994c8f962370194225911ac6a71a365`)
- Qwen3-30B-A3B-Base checkpoint:
  `1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`
- Transformers Qwen3.6-MoE equations:
  `918dbf131d0df5b46e3f6e1d96174d62aa4d16d6`
  (`modeling_qwen3_5_moe.py` SHA-256
  `16ef7b0dc6e26eae26a6ffd0ad11d93f85424f90ffb85fbcc9eecc0759cc930d`)
- Qwen3.6-35B-A3B checkpoint:
  `995ad96eacd98c81ed38be0c5b274b04031597b0`
- A3S Power composition, process-local manifest, and packed-record contract:
  `42c66460533112cc5cb82931a6beda09fc2361ed`

## Numerical Gates

- Tiny sparse layer: router logits, expert IDs, full-softmax route weights, and
  reduced hidden states match the independent fixture.
- Complete tiny decoder: all token logits and each layer's exact routes match a
  dependency-free transcription of the pinned Transformers equations.
- KV cache: a four-token prefill matches four incremental decode steps within
  `2e-5` F32 tolerance.
- Cache failures are transactional: exceeding the context limit does not
  advance session state.
- Complete streaming decoder logits and routes match the fully resident model
  within `2e-5`, including BF16 source conversion and greedy decode.

### OLMoE public-checkpoint oracle

`tools/generate_public_oracle.py` imports Transformers only from a Git checkout
whose `HEAD` and OLMoE source SHA-256 match the pinned values. It loads the
pinned checkpoint as F32 on CPU with eager attention and captures all prompt
logits, all per-layer router logits, and the exact top-k expert IDs and
full-softmax weights. Before model construction it requires the exact config,
index, tokenizer, tokenizer configuration, and three SafeTensor shard lengths
and SHA-256 values from the pinned checkpoint revision. The emitted inventory
repeats those trust anchors for the Rust validator.

`a3s-moe-validate` re-hashes that inventory before loading the resident Rust
model. It then checks tokenizer IDs, every numerical value, every token argmax,
and every selected expert. Structurally valid comparisons produce
`a3s.moe.olmoe-validation.v1` JSON even when a numerical gate fails; a failed
report also causes a non-zero process exit. Default absolute tolerances are
`2e-2` for output and router logits and `2e-4` for route weights. The committed
public artifact records the observed maxima so these bounds can be tightened
from evidence rather than guessed.

## Streaming Residency Gates

- Each selected expert maps to exactly one atomic Power staged group.
- Ready experts execute while other current-layer records remain in flight;
  contributions reduce only in canonical ascending expert order.
- Repeated routes hit Power's cache, and a cache sized for one tiny record stays
  within its declared byte bound while recording evictions.
- Pre-cancelled work performs zero storage reads, and cancellation cannot
  partially advance the caller's KV cache.
- Conversion rejects insufficient buffer budgets and never publishes a partial
  destination. The tiny conversion test observes at most its explicit 1 MiB
  bound.
- Reopening a packed checkpoint verifies exact dense and expert file
  inventories plus collection digests before model construction.

## Continuous Batching Gates

- Two sessions at different KV positions, with different token widths, produce
  the same logits and per-token routes as isolated streaming forwards.
- Each layer's staging report requests exactly the cardinality of the unioned
  expert set, with one weight request per expert.
- Fair multi-round greedy generation matches isolated generation while a
  cancelled member releases its permit and a new member joins the next roster.
- Cancellation through either the scheduler API or the original member token
  is reaped before the next arithmetic step; remaining rows commit normally.
- Lifecycle evidence accounts for three admissions, two completions, one
  cancellation, three committed steps, five processed rows, and zero leaked
  permits in the regression fixture.

## Service Gates

- Greedy remains the default model-level policy, while fixed-seed stochastic
  requests reproduce identical token IDs across independent runs.
- Temperature, top-p, top-k, min-p, repetition, frequency, and presence
  controls are validated before admission; non-finite or invalid values fail
  closed.
- Two concurrent backend streams reach a Power admission peak of two and use
  the M3 continuous scheduler rather than independent generation loops.
- Dropping a response stream propagates cancellation and releases its model
  permit before a replacement request completes.
- Incremental decoding uses the tokenizer's stable streaming decoder and holds
  possible stop-sequence prefixes until they can be emitted or suppressed.
- Exact token-ID prompt digests are exposed for rendered chat prompts.
- A real `a3s-moe-server` subprocess passes model listing, non-streaming OpenAI
  completion, and SSE completion tests through Power's HTTP router.
- The same real-server fixture auto-detects a Qwen3.6 text pack and verifies two
  simultaneous completion requests plus incremental SSE framing.
- The benchmark regressions validate each architecture's versioned JSON schema,
  TTFT, generated-token count, expert bytes read, cache-state labels, and
  process peak RSS. OLMoE additionally validates token parity with an isolated
  resident CPU child; Qwen3-MoE deliberately uses the independent public
  oracle as its parity gate.

## Encrypted Checkpoint Gates

- Dense and expert collections use independently authenticated seekable Power
  containers; encrypted output contains no plaintext `.safetensors` files.
- The pinned top manifest binds both Power child manifests, all logical weight
  digests, configuration, packed metadata, and the optional tokenizer.
- Wrong trust anchors, wrong keys, added artifacts, modified ciphertext, and
  pre-cancelled opens fail before model construction.
- Conversion never publishes a partial destination and reports a peak
  plaintext chunk no larger than the configured chunk size.
- Encrypted and plaintext checkpoints produce the same logits, exact routes,
  execution binding, and greedy token IDs through the same Power hierarchy.
- The service accepts an explicit typed encrypted source, reuses it for the
  process-local Power manifest callback, and rejects an untrusted plain reload
  of the same model name.

These gates validate the software boundary and bounded decrypted buffers. They
do not claim remote attestation or hardware-isolated execution; that evidence
remains an M6 acceptance item on a confidential-computing host.

## Qwen3-MoE Resident and Streaming Gates

- The published 30B-A3B geometry validates an explicit 128-wide attention head
  even though `hidden_size / num_attention_heads` is 64, preventing OLMoE's
  attention assumption from leaking into the second family.
- Sparse schedules reject zero cadence, duplicate/out-of-range dense-only
  layers, invalid head geometry, and invalid token IDs.
- A dependency-free fixture covers router logits, normalized selected
  full-softmax weights, exact expert IDs, fused gate/up ordering, and final
  weighted hidden states for two positions.
- A second dependency-free fixture covers the complete decoder with an
  explicit head dimension that differs from `hidden_size / heads`, per-head
  Q/K normalization, GQA, RoPE, one dense MLP layer, one fused sparse layer,
  final logits, router logits, and exact routes.
- Full prefill matches token-at-a-time KV-cache decode for both unrestricted
  causal attention and a two-token sliding window. Cache-limit failures leave
  the caller's transactional state unchanged.
- The Hugging Face loader requires either the exact 18,867-tensor official
  split-expert inventory or the exact 531-tensor fused-exporter inventory,
  validates each shard as a regular path-safe file, checks index-to-shard
  ownership while loading, and handles mixed dense/sparse MLP names without
  duplicating OLMoE's security boundary.
- The Qwen3-MoE sparse result is expressed directly as Power's unchanged
  `RoutedExpertBatch`; no model-specific Power type or cache was added.
- Official split gate/up/down matrices are read one expert at a time. Fused
  gate/up and down tensors larger than the conversion budget are sliced through
  Power's verified subrange API. Dense tensors larger than the same budget are
  copied in bounded chunks into valid SafeTensor files.
- Packed manifests bind exact dense and sparse-only expert inventories. Dense
  schedule layers have no expert records, and each sparse layer has exactly
  `num_experts` atomic records.
- The Qwen3-MoE family profile covers all 61,066,575,648 bytes in the pinned
  16 SafeTensor files (61,064,245,248 tensor-payload bytes) and all 6,144
  possible one-expert files under explicit 64 GiB / 8,192-file hard limits. A
  geometry-only unit test proves the generic Power defaults are too small and
  the selected profile is sufficient without allocating weights.
- The family profile's 512M element bound covers both the public-model
  embedding/head and the largest accepted fused expert export. Power enforces
  this bound while indexing every ordinary, encrypted, or lossless tensor
  source before mmap or materialization.
- The same geometry test proves that one complete public-model F32 KV cache is
  6 GiB and that the 24 GiB family state limit covers four concurrent 32K
  sessions. Power's generic 4 GiB state limit remains unchanged.
- Its entrypoint residency profile covers a complete 128-expert layer union in
  lossless F32 form under a 4 GiB batch limit while retaining a separate 512
  MiB concurrent-read window. A geometry-only test proves the generic 1 GiB
  batch default is insufficient for that valid public-model case.
- Packed OLMoE and Qwen3-MoE loaders account for the validated dense
  inventory's target F32 bytes plus both expert-cache tiers before
  materializing any dense tensor. Tests prove the shared Power resident-weight
  limit rejects a one-byte cache or fixed-weight overrun.
- Mixed-layer streaming F32 logits, router logits, and exact routes match the
  resident backend within `2e-5`; a BF16 packed checkpoint produces identical
  greedy token IDs while using one bounded Power host cache.
- The pack CLI detects `qwen3_moe` from the validated source configuration and
  emits the same path-explicit conversion evidence schema as OLMoE.
- Two Qwen3-MoE sessions with different generation state use the shared fair
  scheduler, reach a Power admission peak of two, and preserve per-layer
  sparse route unions while dense layers remain session-local.
- The typed Qwen3-MoE backend verifies a `qwen3_moe` process-local manifest,
  streams completion and rendered chat output, exposes the architecture's
  effective prompt digest, and terminates each stream with a final event.
- A real server subprocess automatically detects a packed Qwen3-MoE
  checkpoint and passes Power model listing, non-streaming OpenAI completion,
  and SSE completion tests using the default `qwen3-moe` model identifier.
- The public oracle generator requires the exact pinned 16-shard checkpoint,
  Transformers revision and source digest, retains BF16 parameter storage while
  promoting each weight-bearing operation to eager CPU F32 with the attention
  and expert implementations pinned to eager, and captures every prompt logit
  plus each sparse layer's router logits and
  normalized top-k routes. The Rust validator re-hashes the source inventory,
  verifies the packed checkpoint's source digest/config/tokenizer binding, and
  compares all values, expert IDs, and token argmaxes. Its report also records
  the exact MoE source revision and pinned Power revision used by the validator;
  a `-dirty` MoE suffix makes a non-release build explicit.
- Provenance rejection, numerical failure reporting, official split conversion,
  and Qwen3-MoE performance-schema output are covered by deterministic tiny
  regression checkpoints.

These gates accept the resident CPU reference backend, source checkpoint
loader, tokenizer boundary, packed conversion, Power-backed expert streaming,
continuous batching, service composition, and public acceptance tooling. The
checked Qwen3-30B-A3B-Base report compares 303,872 logits, 12,288 router
logits, and 768 selected routes. It records zero numerical, expert-ID, and
argmax mismatches; the maximum absolute differences are `4.9591064e-5` for
logits, `3.6239624e-5` for router logits, and `1.1920929e-6` for normalized
route weights. A separate checked HTTP report records two simultaneous
requests completing with identical one-token output.

## Qwen3.6-35B-A3B Text Gates

- Configuration validation requires the exact outer `qwen3_5_moe` and inner
  `qwen3_5_moe_text` families, 40 declared layer types, a three-linear/one-full
  schedule, explicit 256-wide attention heads, partial RoPE, F32 recurrent
  state, 256 experts, normalized Top-8 routing, and the supported token IDs.
- The source checkpoint contract pins 34 files. Its 26 BF16 shards total
  71,903,776,776 bytes and index exactly 1,045 tensors: 693 text tensors, 333
  authenticated-but-unsupported vision tensors, and 19 authenticated MTP
  tensors. Missing, added, remapped, resized, or rehashed files fail closed.
- Dependency-free tiny fixtures cover offset RMSNorm, depthwise causal
  convolution, Gated DeltaNet recurrence, full attention with per-head Q/K
  normalization and output gates, mixed-cache prefill/decode parity, shared
  expert gating, normalized routes, resident-versus-streamed logits, BF16
  packing, cache bounds, cancellation, and greedy generation.
- Every one of the 40 layers has exactly 256 atomic expert records after
  conversion. The converter reads official fused arrays through Power's
  verified subrange API, never buffers more than its declared budget, and
  rejects any dense, expert, manifest, or source-binding discrepancy.
- The Qwen3.6 family profile covers the exact checkpoint under 80 GiB and
  16,384-file bounds, four maximum-context mixed states under 48 GiB, a 640M
  element per-tensor limit that covers the 536,870,912-element fused gate/up
  array, and one complete 256-expert F32 union under the 4 GiB residency
  staging limit.
- Ragged batches retain independent mixed cache positions while unioning exact
  routes per layer. Continuous generation uses Power admission, fairness,
  cancellation, aggregate-state checks, and one immutable roster per step.
- The public oracle imports the pinned Transformers checkout, maps only the
  complete official `model.language_model` namespace into
  `Qwen3_5MoeForCausalLM`, rejects missing or mismatched text keys, retains
  BF16 storage, and promotes embedding, linear, and depthwise-convolution
  operations to CPU F32. It captures every vocabulary logit and normalized
  Top-8 route from all layers. Its default five-token prompt covers a complete
  four-token convolution window, and shorter public-oracle prompts fail closed.
- The Rust validator re-hashes all 34 source files, including the vision and
  MTP shards excluded from the text pack, verifies the packed source/config/
  tokenizer binding, and compares logits, router logits, route weights, expert
  IDs, and argmax tokens. Provenance rejection and numerical-failure reports
  are deterministic regression tests.
- Full public parity and target-host performance are acceptance outputs, not
  inferred claims. Their checked artifacts are added only after exact weight
  download, conversion, oracle execution, real concurrent HTTP execution, and
  release benchmarking complete.

The benchmark's first generation is application-cold with respect to Power's
expert cache. Integrity verification may already populate the operating-system
page cache, so the artifact records that state as uncontrolled rather than
claiming physical cold storage.

## Public Checkpoint Metadata

`tools/verify_hf_contract.py` uses bounded HTTP range requests to parse only the
three SafeTensor headers. On 2026-08-13 it reported:

```json
{
  "tensorCount": 3219,
  "parameterCount": 6919161856,
  "dtype": "BF16",
  "status": "verified"
}
```

Every tensor name, index-to-shard mapping, dtype, and shape matched the
architecture contract. The complete pinned payload was subsequently
downloaded and SHA-256 verified. The exact Transformers oracle and independent
Rust resident run produced [`olmoe-public-validation.json`](../evidence/olmoe-public-validation.json):

- 384 selected routes checked with zero expert-ID mismatches;
- zero argmax mismatches;
- maximum absolute differences of `9.915829e-4` for logits,
  `1.1062622e-4` for router logits, and `7.480383e-6` for route weights;
- zero values outside the declared acceptance tolerances.

## Tokenizer

The pinned public `tokenizer.json` loaded with 50,280 tokens under the model's
50,304-entry padded vocabulary. `Bitcoin is` encoded as `[12871, 9669, 310]`
and decoded back to the original text.

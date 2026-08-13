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
- A3S Power composition, process-local manifest, and packed-record contract:
  `be82555`

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
- The Qwen3-MoE family profile covers the pinned 61,064,245,248-byte source and
  all 6,144 possible one-expert files under explicit 64 GiB / 8,192-file hard
  limits. A geometry-only unit test proves the generic Power defaults are too
  small and the selected profile is sufficient without allocating weights.
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
  Transformers revision and source digest, runs eager CPU BF16 equations, and
  captures every prompt logit plus each sparse layer's router logits and
  normalized top-k routes. The Rust validator re-hashes the source inventory,
  verifies the packed checkpoint's source digest/config/tokenizer binding, and
  compares all values, expert IDs, and token argmaxes.
- Provenance rejection, numerical failure reporting, official split conversion,
  and Qwen3-MoE performance-schema output are covered by deterministic tiny
  regression checkpoints.

These gates accept the resident CPU reference backend, source checkpoint
loader, tokenizer boundary, packed conversion, Power-backed expert streaming,
continuous batching, service composition, and public acceptance tooling. They
do not yet accept a public 30B-A3B numerical or performance run; that status
requires checked reports from the pinned complete checkpoint.

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

# Validation Evidence

## Pinned Sources

- Transformers OLMoE equations:
  `918dbf131d0df5b46e3f6e1d96174d62aa4d16d6`
- OLMoE-1B-7B-0924 checkpoint:
  `6d84c48581ece794365f2b8e9cfb043c68ade9c5`
- A3S Power composition, process-local manifest, and packed-record contract:
  `f1ec432`

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

### Public-checkpoint oracle

`tools/generate_public_oracle.py` imports Transformers only from a Git checkout
whose `HEAD` is the pinned revision. It loads the pinned checkpoint as F32 on
CPU with eager attention and captures all prompt logits, all per-layer router
logits, and the exact top-k expert IDs and full-softmax weights. Its file
inventory binds the config, index, tokenizer, and every SafeTensor shard by
byte length and SHA-256.

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
- The benchmark regression validates the versioned JSON schema, TTFT,
  generated-token count, expert bytes read, cache-state labels, process peak
  RSS, and token parity with an isolated resident CPU child process.

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
architecture contract. This metadata check does not claim real-weight
numerical parity; that requires downloading and executing the full checkpoint.

## Tokenizer

The pinned public `tokenizer.json` loaded with 50,280 tokens under the model's
50,304-entry padded vocabulary. `Bitcoin is` encoded as `[12871, 9669, 310]`
and decoded back to the original text.

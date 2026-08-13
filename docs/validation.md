# Validation Evidence

## Pinned Sources

- Transformers OLMoE equations:
  `918dbf131d0df5b46e3f6e1d96174d62aa4d16d6`
- OLMoE-1B-7B-0924 checkpoint:
  `6d84c48581ece794365f2b8e9cfb043c68ade9c5`
- A3S Power composition and packed-record contract:
  `7ce6fa5`

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

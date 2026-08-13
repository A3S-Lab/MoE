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

The M0 numerical contract and the resident M1 CPU engine are implemented and
tested:

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

This is a complete correctness baseline, not yet a bounded-memory serving
engine. Streaming expert residency, continuous batching, sampling, chat
templates, and the Power HTTP backend are the next milestones. The 13.8 GB
public checkpoint's metadata contract is verified without downloading weight
payloads; a full real-checkpoint numerical run remains an M1 acceptance item.

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

There is one residency hierarchy. Model code must consume weights returned by
Power and must not introduce a second expert cache. A future packed expert
record remains an opaque `U8` SafeTensor to Power; this crate validates and
interprets the record header and quantization data.

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

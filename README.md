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

The M0 numerical contract is implemented and tested:

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

This is a correctness baseline, not yet a complete text-generation engine.
Model loading, transformer attention, KV cache, streaming expert residency,
tokenization, and the Power HTTP backend are the next milestones.

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

Regenerate the tiny oracle only when intentionally changing the pinned model
contract:

```shell
python tools/generate_tiny_oracle.py
```

The generator prints JSON to stdout and never overwrites the checked fixture.
Review changes before replacing `tests/fixtures/olmoe_tiny_oracle.json`.

## License

MIT

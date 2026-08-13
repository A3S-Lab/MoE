# MoE Inference Architecture

## Goals

The engine targets correct, bounded-memory inference for sparse MoE models on
CPU, accelerators, and confidential-computing hosts. It adopts the useful
Colibri pattern—storage, RAM, and device tiers with just-in-time expert
movement—without coupling generic Power runtime code to one model family.

The initial model is OLMoE. A second architecture is admitted only after the
boundary has proven that model-owned semantics can change without modifying
Power's residency core.

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

Only `WeightHierarchy` owns resident bytes. Model code may hold short-lived
views for the current operation but cannot retain a parallel byte cache.

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
- Pending acceptance: run the complete 13.8 GB pinned public checkpoint against
  Transformers token IDs and logits on a sufficiently provisioned host.

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

### M4: Service and Performance

- Implement the Power `Backend` adapter and an `a3s-moe` server binary.
- Add tokenizer/chat-template and OpenAI chat/completions streaming.
- Publish cold/warm throughput, time-to-first-token, bytes read, and peak RAM
  against the CPU baseline.

### M5–M7: Accelerator, TEE, and Second Architecture

- Add device execution with CPU parity and declared fallback evidence.
- Add seekable encrypted expert reads and TEE memory acceptance.
- Add a second MoE family without changing Power's model-neutral contracts.

## Acceptance Gates

Every optimized path must retain exact expert IDs and match reference route
weights and logits within a declared dtype-specific tolerance. Tests must cover
malformed configs, truncated/corrupt weights, cancellation, cache pressure,
mixed routes, and deterministic fallback. Performance claims require a checked
benchmark artifact containing hardware, storage cache state, model digest,
configuration, and raw samples.

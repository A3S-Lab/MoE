# Performance Evidence

`a3s-moe-bench` emits raw JSON rather than embedding headline numbers in the
codebase. Run release builds on the target host and retain the complete output
with the model revision and operational configuration under review.

## Measurement Boundary

The report separates these phases:

1. Packed checkpoint validation, dense-weight materialization, and construction
   of Power's expert hierarchy.
2. First generation with an empty Power expert residency cache.
3. Repeated warm generations with the bounded Power cache retained.
4. Optional OLMoE fully resident CPU loading and generation in a separate
   process.

Time to first token is measured at the caller-facing backend stream. The
per-request model prompt duration is also retained when supplied by the worker.
Throughput divides emitted model tokens by complete request duration. Expert
storage reads and bytes come from Power placement telemetry, not filesystem
size estimates.

Peak RSS is a process-lifetime operating-system counter. The streaming process
is sampled before the resident baseline starts. When requested, the baseline
runs as a child process so its peak does not include streaming allocations and
the streaming peak does not include resident weights.

## Cache Terminology

"First" means application-cold: no expert has been retained in Power's host or
device cache. It does not mean physical-cold storage. Opening and integrity
checking the checkpoint can populate the operating-system page cache, and the
benchmark deliberately records that cache as uncontrolled.

Use Power's storage benchmark when a verified storage-cache preparation policy
is required. Do not relabel an uncontrolled page cache as cold I/O.

## Example

```shell
cargo run --release --features benchmark --bin a3s-moe-bench -- \
  /models/OLMoE-1B-7B-0924-a3s \
  --prompt "Bitcoin is" \
  --max-tokens 32 \
  --warm-samples 5 \
  --host-cache-mib 512 \
  --checkpoint-label olmoe-1b-7b-bf16 \
  --resident-checkpoint /models/OLMoE-1B-7B-0924 \
  > olmoe-performance.json
```

Accelerator runs add `cuda` or `metal`, `--device <kind>:<ordinal>`, and an
explicit `--device-cache-mib` bound. Evidence records both the typed request
and Power's resolved device. Only `--device auto` can report
`automaticCpuFallback: true`; an explicit unavailable accelerator is an error.

The Qwen3-MoE public run uses the same measurement boundary without a resident
F32 child. Its parity gate is the separately generated, pinned Transformers
F32-operation oracle over the BF16 checkpoint values:

```shell
cargo run --release --features benchmark --bin a3s-moe-bench -- \
  /models/Qwen3-30B-A3B-Base-a3s \
  --prompt "Bitcoin is" \
  --max-tokens 8 \
  --warm-samples 3 \
  --host-cache-mib 4096 \
  --checkpoint-label qwen3-30b-a3b-bf16 \
  > qwen3-moe-performance.json
```

Artifacts use `a3s.moe.olmoe-performance.v1` or
`a3s.moe.qwen3-moe-performance.v1` and include:

- the exact a3s-moe Git revision (with an explicit `-dirty` suffix when code
  inputs differ from `HEAD`) and pinned Power revision;
- logical weight digest and packed byte count;
- OS, architecture, logical parallelism, and available processor identity;
- prompt and generation settings;
- raw first and warm timing samples;
- Power expert I/O, hit, eviction, and residency counters;
- streaming process peak RSS;
- exact generated token IDs;
- for OLMoE when requested, isolated resident peak RSS and
  resident-versus-streaming token parity.

Compare artifacts only when model digest, prompt token IDs, generation policy,
hardware, build profile, cache policy, and page-cache preparation are
equivalent. `packedCheckpoint` is a caller-provided path-free label (or the
checkpoint directory name by default); the logical weights digest is the
machine-independent identity.

## Checked Public CPU Run

[`olmoe-public-cpu-windows.json`](../evidence/olmoe-public-cpu-windows.json)
records a complete OLMoE-1B-7B run on a 20-logical-core Windows x86-64 host.
The prompt `Bitcoin is` generated eight identical token IDs on the
Power-streamed and isolated resident paths. With a 4 GiB expert cache, the
three warm streaming samples averaged `0.3524 tokens/s` and `5.91 s` TTFT.
Peak RSS was 13,612,077,056 bytes for streaming versus 41,563,848,704 bytes for
the resident baseline (about 3.05 times lower), while the resident path was
about 6.94 times faster for this CPU-only sample. The OS page cache was
uncontrolled, so the artifact makes no physical-cold-storage claim.

The pinned Qwen3-30B-A3B-Base performance report remains pending. No public
throughput or memory claim is made until the complete-checkpoint numerical gate
passes and the raw family-specific artifact is checked under `evidence/`.

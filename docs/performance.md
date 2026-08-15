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

When a post-first-token decode rate is quoted, it is derived for samples with
at least two generated tokens as `(generatedTokens - 1) / (total - TTFT)`.
This secondary value excludes prompt evaluation and the first emitted token;
the JSON artifact's `tokensPerSecond` remains the primary end-to-end metric.

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

Qwen3.6 uses the same measurement boundary and independent-oracle parity gate:

```shell
cargo run --release --features benchmark --bin a3s-moe-bench -- \
  /models/Qwen3.6-35B-A3B-a3s \
  --prompt "Bitcoin is" \
  --max-tokens 8 \
  --warm-samples 3 \
  --host-cache-mib 4096 \
  --checkpoint-label qwen3.6-35b-a3b-bf16 \
  > qwen3.6-35b-a3b-performance.json
```

Accelerator evidence uses the same model digest and oracle gate while selecting
the device and its independent cache bound explicitly. The checked Qwen3.6
CUDA command uses `--features benchmark,cuda`, `--device cuda:0`,
`--host-cache-mib 0`, and `--device-cache-mib 8192`. An explicit accelerator
request fails rather than silently falling back to CPU.

Artifacts use `a3s.moe.olmoe-performance.v1` or
`a3s.moe.qwen3-moe-performance.v1`; Qwen3.6 reports use
`a3s.moe.qwen3.6-moe-performance.v1`. All include:

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

## Checked Public Runs

[`olmoe-public-cpu-windows.json`](../evidence/olmoe-public-cpu-windows.json)
records a complete OLMoE-1B-7B run on a 20-logical-core Windows x86-64 host.
The prompt `Bitcoin is` generated eight identical token IDs on the
Power-streamed and isolated resident paths. With a 4 GiB expert cache, the
three warm streaming samples averaged `0.3524 tokens/s` and `5.91 s` TTFT.
Peak RSS was 13,612,077,056 bytes for streaming versus 41,563,848,704 bytes for
the resident baseline (about 3.05 times lower), while the resident path was
about 6.94 times faster for this CPU-only sample. The OS page cache was
uncontrolled, so the artifact makes no physical-cold-storage claim.

[`qwen3-moe-public-cpu-windows.json`](../evidence/qwen3-moe-public-cpu-windows.json)
records the accepted Qwen3-30B-A3B-Base CPU run on the same 20-logical-core
Windows x86-64 host. The complete-checkpoint numerical gate passed before the
sample was admitted. With a 4 GiB expert cache, three warm eight-token samples
averaged `0.1541 tokens/s` and `10.46 s` TTFT. Peak RSS was 26,278,592,512
bytes, host residency remained below the configured bound at 4,293,947,840
bytes, and staged in-flight expert bytes peaked at 37,748,992. The report has
no resident F32 baseline because that would require a second full 30B model;
the independent pinned oracle is the parity gate. Its OS page cache is
explicitly uncontrolled.

[`qwen3-moe-public-http-windows.json`](../evidence/qwen3-moe-public-http-windows.json)
records two simultaneous one-token OpenAI completion requests through the real
Power HTTP server with `maxConcurrentRequests = 2`; both completed with the
same output and no server errors.

[`qwen3.6-35b-a3b-public-cpu-windows.json`](../evidence/qwen3.6-35b-a3b-public-cpu-windows.json)
records the accepted Qwen3.6-35B-A3B text run on the same 20-logical-core
Windows x86-64 host. The model revision is
`995ad96eacd98c81ed38be0c5b274b04031597b0`, the packed text checkpoint is
69,335,962,985 bytes, and execution is the CPU-only F32 dense plus
Power-streamed-expert path. With a 4 GiB host cache, the first eight-token
generation measured `0.161215 tokens/s`; the three warm generations measured
`0.149582`, `0.197711`, and `0.236555 tokens/s`, for an end-to-end arithmetic
mean of `0.194616 tokens/s` and mean TTFT of `11.664 s`. Their derived
post-first-token decode rates were `0.203932`, `0.221544`, and
`0.260715 tokens/s`, for an arithmetic mean of `0.228730 tokens/s`. Peak RSS was
23,509,303,296 bytes, host residency was 4,290,816,640 bytes, and staged
in-flight expert bytes peaked at 25,166,080. The operating-system page cache
was uncontrolled.

[`qwen3.6-35b-a3b-public-cuda-windows.json`](../evidence/qwen3.6-35b-a3b-public-cuda-windows.json)
records the same packed Qwen3.6 text checkpoint on the host's RTX 4090 through
Power's explicit `cuda:0` path, with no automatic CPU fallback, no host expert
cache, and an 8 GiB device expert cache. For 16 generated tokens, the three
warm samples measured `0.268320`, `0.264965`, and `0.257548 tokens/s`, for an
end-to-end arithmetic mean of `0.263611 tokens/s` and mean TTFT of `3.943 s`.
Their derived post-first-token decode rates average `0.264298 tokens/s`. The
first generation measured `0.162680 tokens/s`; the retained cache still read
25,807,815,040 bytes across the warm samples because the 16-token expert
working set exceeds the 8 GiB bound. Peak process RSS was 14,239,883,264 bytes.
This artifact is the unquantized F32 execution baseline and contains no DSpark
drafter or speculative acceptance.

[`qwen3.6-35b-a3b-public-http-windows.json`](../evidence/qwen3.6-35b-a3b-public-http-windows.json)
records two simultaneous one-token requests through the real Power server.
Both returned HTTP 200 with identical output in `35.2554197 s`, an aggregate
completion-token rate of `0.056729 tokens/s`; this is a concurrency smoke
measurement, not the single-request generation headline.

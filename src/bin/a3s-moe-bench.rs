use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use a3s_moe::olmoe::{OlmoeCheckpoint, OlmoeConfig, OlmoeCpuModel, OlmoeTokenizer};
use a3s_moe::service::{OlmoeBackend, OlmoeBackendConfig, OlmoeDeviceSpec};
use a3s_power::backend::types::CompletionRequest;
use a3s_power::backend::Backend;
use a3s_power::inference::{
    DevicePreference, InferenceLimits, PlacementTelemetry, ResidencyPolicy, RuntimeDeviceIdentity,
    TelemetryMode,
};
use anyhow::{Context, Result};
use candle_core::{Device, IndexOp, Tensor};
use clap::Parser;
use futures::StreamExt;
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Emit reproducible JSON performance evidence for packed OLMoE")]
struct Args {
    /// Self-contained checkpoint created by a3s-moe-pack.
    checkpoint: PathBuf,

    /// Prompt to tokenize and run.
    #[arg(long)]
    prompt: String,

    /// Optional original Hugging Face checkpoint for the resident CPU baseline.
    #[arg(long)]
    resident_checkpoint: Option<PathBuf>,

    #[arg(long, default_value = "olmoe-benchmark")]
    model: String,

    #[arg(long, default_value_t = 16)]
    max_tokens: u32,

    /// Number of warm samples after the first application-cold sample.
    #[arg(long, default_value_t = 5)]
    warm_samples: usize,

    #[arg(long, default_value_t = 512)]
    host_cache_mib: u64,

    /// Typed execution device: auto, cpu, cuda:<ordinal>, or metal:<ordinal>.
    #[arg(long, default_value = "cpu")]
    device: OlmoeDeviceSpec,

    #[arg(long, default_value_t = 0)]
    device_cache_mib: u64,

    /// Internal isolated resident-baseline child mode.
    #[arg(long, hide = true)]
    resident_baseline_only: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BenchmarkEvidence {
    schema: &'static str,
    generated_at: String,
    implementation: ImplementationEvidence,
    model: ModelEvidence,
    system: SystemEvidence,
    configuration: ConfigurationEvidence,
    cache_state: CacheStateEvidence,
    load: LoadEvidence,
    first_generation: InferenceSample,
    warm_generations: Vec<InferenceSample>,
    warm_summary: SampleSummary,
    placement: PlacementDelta,
    process_peak_rss_bytes: Option<u64>,
    resident_baseline: Option<ResidentBaseline>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ImplementationEvidence {
    crate_name: &'static str,
    crate_version: &'static str,
    power_revision: &'static str,
    execution: &'static str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelEvidence {
    name: String,
    weights_sha256: String,
    packed_checkpoint: String,
    packed_bytes: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SystemEvidence {
    os: &'static str,
    architecture: &'static str,
    logical_parallelism: usize,
    processor: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ConfigurationEvidence {
    prompt_bytes: usize,
    prompt_tokens: usize,
    max_new_tokens: u32,
    warm_samples: usize,
    temperature: f32,
    host_cache_bytes: u64,
    device_cache_bytes: u64,
    effective_device_cache_bytes: u64,
    requested_device: DevicePreference,
    resolved_device: RuntimeDeviceIdentity,
    automatic_cpu_fallback: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CacheStateEvidence {
    power_first_generation: &'static str,
    power_warm_generations: &'static str,
    operating_system_page_cache: &'static str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct LoadEvidence {
    packed_streaming_ns: u64,
    resident_baseline_ns: Option<u64>,
}

#[derive(Debug, serde::Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct InferenceSample {
    total_ns: u64,
    time_to_first_token_ns: u64,
    model_prompt_eval_ns: Option<u64>,
    generated_tokens: usize,
    tokens_per_second: f64,
    token_ids: Vec<u32>,
    done_reason: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SampleSummary {
    samples: usize,
    mean_total_ns: f64,
    mean_time_to_first_token_ns: f64,
    mean_tokens_per_second: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PlacementDelta {
    first_storage_reads: u64,
    first_storage_bytes_read: u64,
    warm_storage_reads: u64,
    warm_storage_bytes_read: u64,
    host_cache_hits: u64,
    host_resident_bytes: u64,
    host_evictions: u64,
    staged_peak_inflight_bytes: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ResidentBaseline {
    load_ns: u64,
    sample: InferenceSample,
    token_parity_with_streaming: bool,
    process_peak_rss_bytes: Option<u64>,
}

#[derive(serde::Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct ResidentChildEvidence {
    load_ns: u64,
    sample: InferenceSample,
    process_peak_rss_bytes: Option<u64>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if args.prompt.is_empty() {
        anyhow::bail!("--prompt must not be empty");
    }
    if args.max_tokens == 0 {
        anyhow::bail!("--max-tokens must be greater than zero");
    }
    if args.warm_samples == 0 {
        anyhow::bail!("--warm-samples must be greater than zero");
    }
    if args.resident_baseline_only {
        return run_resident_child(&args);
    }

    let checkpoint = args.checkpoint.canonicalize().with_context(|| {
        format!(
            "failed to resolve packed checkpoint '{}'",
            args.checkpoint.display()
        )
    })?;
    let config = OlmoeConfig::from_json_path(checkpoint.join("config.json"))?;
    let tokenizer =
        OlmoeTokenizer::from_file(checkpoint.join("tokenizer.json"), config.vocab_size)?;
    let prompt_tokens = tokenizer.encode(&args.prompt, true)?;
    let host_cache_bytes = mib(args.host_cache_mib)?;
    let device_cache_bytes = mib(args.device_cache_mib)?;
    let limits = InferenceLimits {
        max_context_tokens: config.max_position_embeddings,
        max_generated_tokens: args.max_tokens as usize,
        max_concurrent_requests: 1,
        max_queued_requests: 1,
        ..InferenceLimits::default()
    };
    let backend = Arc::new(OlmoeBackend::new(OlmoeBackendConfig {
        device: args.device.preference(),
        inference_limits: limits,
        residency_policy: ResidencyPolicy {
            host_cache_bytes,
            device_cache_bytes,
            max_background_inflight_bytes: 512 * 1024 * 1024,
            telemetry: TelemetryMode::Aggregate,
            ..ResidencyPolicy::default()
        },
        stream_capacity: 8,
        batch_window: Duration::ZERO,
        default_max_tokens: args.max_tokens as usize,
    })?);

    let load_started = Instant::now();
    let manifest = backend
        .preload(&args.model, &checkpoint, None)
        .await
        .context("failed to preload the packed checkpoint")?;
    let packed_load_ns = duration_ns(load_started.elapsed());
    let device = backend.device_selection(&args.model)?;

    let before_first = backend.telemetry(&args.model)?;
    let first_generation =
        run_streaming(&backend, &args.model, &args.prompt, args.max_tokens).await?;
    let after_first = backend.telemetry(&args.model)?;
    let mut warm_generations = Vec::with_capacity(args.warm_samples);
    for _ in 0..args.warm_samples {
        warm_generations
            .push(run_streaming(&backend, &args.model, &args.prompt, args.max_tokens).await?);
    }
    let after_warm = backend.telemetry(&args.model)?;

    let streaming_process_peak_rss_bytes = process_peak_rss_bytes();
    let resident_baseline = match args.resident_checkpoint.as_ref() {
        Some(path) => {
            let child = run_resident_subprocess(&args, path).await?;
            Some(ResidentBaseline {
                load_ns: child.load_ns,
                token_parity_with_streaming: child.sample.token_ids == first_generation.token_ids,
                sample: child.sample,
                process_peak_rss_bytes: child.process_peak_rss_bytes,
            })
        }
        None => None,
    };
    let resident_load_ns = resident_baseline.as_ref().map(|baseline| baseline.load_ns);

    let evidence = BenchmarkEvidence {
        schema: "a3s.moe.olmoe-performance.v1",
        generated_at: chrono::Utc::now().to_rfc3339(),
        implementation: ImplementationEvidence {
            crate_name: env!("CARGO_PKG_NAME"),
            crate_version: env!("CARGO_PKG_VERSION"),
            power_revision: "f1ec432",
            execution: "f32-dense-power-streamed-experts",
        },
        model: ModelEvidence {
            name: manifest.name,
            weights_sha256: manifest.sha256,
            packed_checkpoint: checkpoint.display().to_string(),
            packed_bytes: manifest.size,
        },
        system: SystemEvidence {
            os: std::env::consts::OS,
            architecture: std::env::consts::ARCH,
            logical_parallelism: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
            processor: std::env::var("PROCESSOR_IDENTIFIER").ok(),
        },
        configuration: ConfigurationEvidence {
            prompt_bytes: args.prompt.len(),
            prompt_tokens: prompt_tokens.len(),
            max_new_tokens: args.max_tokens,
            warm_samples: args.warm_samples,
            temperature: 0.0,
            host_cache_bytes,
            device_cache_bytes,
            effective_device_cache_bytes: device.effective_device_cache_bytes,
            requested_device: device.requested,
            resolved_device: device.resolved,
            automatic_cpu_fallback: device.automatic_cpu_fallback,
        },
        cache_state: CacheStateEvidence {
            power_first_generation: "empty expert residency cache",
            power_warm_generations: "retained bounded expert residency cache",
            operating_system_page_cache:
                "uncontrolled; checkpoint integrity verification may warm file pages",
        },
        load: LoadEvidence {
            packed_streaming_ns: packed_load_ns,
            resident_baseline_ns: resident_load_ns,
        },
        warm_summary: summarize(&warm_generations),
        placement: placement_delta(&before_first, &after_first, &after_warm),
        first_generation,
        warm_generations,
        process_peak_rss_bytes: streaming_process_peak_rss_bytes,
        resident_baseline,
    };
    println!("{}", serde_json::to_string_pretty(&evidence)?);
    Ok(())
}

fn run_resident_child(args: &Args) -> Result<()> {
    let path = args
        .resident_checkpoint
        .as_ref()
        .context("resident child mode requires --resident-checkpoint")?;
    let load_started = Instant::now();
    let checkpoint = OlmoeCheckpoint::open(path)?;
    let tokenizer = checkpoint.load_tokenizer()?;
    let prompt = tokenizer.encode(&args.prompt, true)?;
    let config = checkpoint.config().clone();
    let resident = checkpoint.load_cpu_resident()?;
    let load_ns = duration_ns(load_started.elapsed());
    let sample = run_resident(
        &resident,
        &prompt,
        args.max_tokens as usize,
        config.eos_token_id,
    )?;
    let evidence = ResidentChildEvidence {
        load_ns,
        sample,
        process_peak_rss_bytes: process_peak_rss_bytes(),
    };
    println!("{}", serde_json::to_string(&evidence)?);
    Ok(())
}

async fn run_resident_subprocess(
    args: &Args,
    resident_checkpoint: &std::path::Path,
) -> Result<ResidentChildEvidence> {
    let executable = std::env::current_exe().context("failed to resolve benchmark executable")?;
    let checkpoint = args.checkpoint.clone();
    let prompt = args.prompt.clone();
    let max_tokens = args.max_tokens.to_string();
    let resident_checkpoint = resident_checkpoint.to_path_buf();
    let output = tokio::task::spawn_blocking(move || {
        std::process::Command::new(executable)
            .arg(checkpoint)
            .args(["--prompt", &prompt])
            .args(["--max-tokens", &max_tokens])
            .arg("--resident-checkpoint")
            .arg(resident_checkpoint)
            .arg("--resident-baseline-only")
            .output()
    })
    .await
    .context("resident baseline subprocess task failed")??;
    if !output.status.success() {
        anyhow::bail!(
            "resident baseline subprocess failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    serde_json::from_slice(&output.stdout).context("resident baseline emitted invalid JSON")
}

async fn run_streaming(
    backend: &OlmoeBackend,
    model: &str,
    prompt: &str,
    max_tokens: u32,
) -> Result<InferenceSample> {
    let request: CompletionRequest = serde_json::from_value(serde_json::json!({
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": true
    }))?;
    let started = Instant::now();
    let mut stream = backend.complete(model, request).await?;
    let mut first_token = None;
    let mut token_ids = Vec::new();
    let mut prompt_eval = None;
    let mut done_reason = None;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if chunk.token_id.is_some() && first_token.is_none() {
            first_token = Some(started.elapsed());
        }
        if let Some(token) = chunk.token_id {
            token_ids.push(token);
        }
        prompt_eval = prompt_eval.or(chunk.prompt_eval_duration_ns);
        done_reason = done_reason.or(chunk.done_reason);
    }
    let total = started.elapsed();
    let first_token = first_token.context("generation ended without producing a token")?;
    let generated_tokens = token_ids.len();
    Ok(InferenceSample {
        total_ns: duration_ns(total),
        time_to_first_token_ns: duration_ns(first_token),
        model_prompt_eval_ns: prompt_eval,
        generated_tokens,
        tokens_per_second: throughput(generated_tokens, total),
        token_ids,
        done_reason: done_reason.unwrap_or_else(|| "missing".to_string()),
    })
}

fn run_resident(
    model: &OlmoeCpuModel,
    prompt: &[u32],
    max_tokens: usize,
    eos_token_id: Option<u32>,
) -> Result<InferenceSample> {
    let started = Instant::now();
    let mut cache = model.new_cache();
    let mut input = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), &Device::Cpu)?;
    let mut token_ids = Vec::with_capacity(max_tokens);
    let mut first_token = None;
    let mut done_reason = "length".to_string();
    for _ in 0..max_tokens {
        let output = model.forward(&input, &mut cache)?;
        let sequence_length = output.logits.dim(1)?;
        let token = output
            .logits
            .i((0, sequence_length - 1, ..))?
            .argmax(0)?
            .to_scalar::<u32>()?;
        token_ids.push(token);
        first_token.get_or_insert_with(|| started.elapsed());
        if eos_token_id == Some(token) {
            done_reason = "stop".to_string();
            break;
        }
        input = Tensor::from_vec(vec![token], (1, 1), &Device::Cpu)?;
    }
    let total = started.elapsed();
    let first_token = first_token.context("resident generation produced no token")?;
    Ok(InferenceSample {
        total_ns: duration_ns(total),
        time_to_first_token_ns: duration_ns(first_token),
        model_prompt_eval_ns: Some(duration_ns(first_token)),
        generated_tokens: token_ids.len(),
        tokens_per_second: throughput(token_ids.len(), total),
        token_ids,
        done_reason,
    })
}

fn summarize(samples: &[InferenceSample]) -> SampleSummary {
    let count = samples.len() as f64;
    SampleSummary {
        samples: samples.len(),
        mean_total_ns: samples
            .iter()
            .map(|sample| sample.total_ns as f64)
            .sum::<f64>()
            / count,
        mean_time_to_first_token_ns: samples
            .iter()
            .map(|sample| sample.time_to_first_token_ns as f64)
            .sum::<f64>()
            / count,
        mean_tokens_per_second: samples
            .iter()
            .map(|sample| sample.tokens_per_second)
            .sum::<f64>()
            / count,
    }
}

fn placement_delta(
    before: &PlacementTelemetry,
    first: &PlacementTelemetry,
    warm: &PlacementTelemetry,
) -> PlacementDelta {
    PlacementDelta {
        first_storage_reads: first.storage_reads.saturating_sub(before.storage_reads),
        first_storage_bytes_read: first
            .storage_bytes_read
            .saturating_sub(before.storage_bytes_read),
        warm_storage_reads: warm.storage_reads.saturating_sub(first.storage_reads),
        warm_storage_bytes_read: warm
            .storage_bytes_read
            .saturating_sub(first.storage_bytes_read),
        host_cache_hits: warm.host_cache_hits.saturating_sub(before.host_cache_hits),
        host_resident_bytes: warm.host_resident_bytes,
        host_evictions: warm.host_evictions.saturating_sub(before.host_evictions),
        staged_peak_inflight_bytes: warm.staged_peak_inflight_bytes,
    }
}

fn throughput(tokens: usize, duration: Duration) -> f64 {
    if duration.is_zero() {
        0.0
    } else {
        tokens as f64 / duration.as_secs_f64()
    }
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn mib(value: u64) -> Result<u64> {
    value
        .checked_mul(1024 * 1024)
        .context("MiB value overflowed u64")
}

#[cfg(windows)]
fn process_peak_rss_bytes() -> Option<u64> {
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    // SAFETY: the pseudo-handle is valid for the current process, the output
    // points to a correctly sized initialized structure, and the call does not
    // retain either value.
    unsafe {
        let mut counters = std::mem::zeroed::<PROCESS_MEMORY_COUNTERS>();
        counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        let status = GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb);
        (status != 0).then_some(counters.PeakWorkingSetSize as u64)
    }
}

#[cfg(unix)]
fn process_peak_rss_bytes() -> Option<u64> {
    // SAFETY: `usage` points to writable storage of the exact `rusage` type and
    // `RUSAGE_SELF` requests only counters for the current process.
    unsafe {
        let mut usage = std::mem::zeroed::<libc::rusage>();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) != 0 || usage.ru_maxrss < 0 {
            return None;
        }
        #[cfg(target_os = "macos")]
        let bytes = usage.ru_maxrss as u64;
        #[cfg(not(target_os = "macos"))]
        let bytes = (usage.ru_maxrss as u64).saturating_mul(1024);
        Some(bytes)
    }
}

#[cfg(not(any(windows, unix)))]
fn process_peak_rss_bytes() -> Option<u64> {
    None
}

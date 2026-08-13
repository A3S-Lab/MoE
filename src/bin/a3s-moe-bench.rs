use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use a3s_moe::olmoe::{OlmoeCheckpoint, OlmoeConfig, OlmoeCpuModel};
use a3s_moe::qwen3_moe::Qwen3MoeConfig;
use a3s_moe::service::{
    MoeBackendConfig, MoeDeviceSelection, MoeDeviceSpec, OlmoeBackend, Qwen3MoeBackend,
};
use a3s_moe::{MoeArchitecture, MoeTokenizer};
use a3s_power::backend::types::CompletionRequest;
use a3s_power::backend::Backend;
use a3s_power::error::Result as PowerResult;
use a3s_power::inference::{PlacementTelemetry, ResidencyPolicy, TelemetryMode};
use a3s_power::model::manifest::ModelManifest;
use anyhow::{Context, Result};
use candle_core::{Device, IndexOp, Tensor};
use clap::Parser;
use futures::StreamExt;

#[path = "a3s-moe-bench/evidence.rs"]
mod evidence;
use evidence::*;

#[derive(Debug, Parser)]
#[command(about = "Emit reproducible JSON performance evidence for packed MoE inference")]
struct Args {
    /// Self-contained checkpoint created by a3s-moe-pack.
    checkpoint: PathBuf,

    /// Prompt to tokenize and run.
    #[arg(long)]
    prompt: String,

    /// Optional original OLMoE checkpoint for an isolated resident baseline.
    #[arg(long)]
    resident_checkpoint: Option<PathBuf>,

    /// Path-free checkpoint label recorded in evidence (defaults to the directory name).
    #[arg(long)]
    checkpoint_label: Option<String>,

    #[arg(long)]
    model: Option<String>,

    #[arg(long, default_value_t = 16)]
    max_tokens: u32,

    /// Number of warm samples after the first application-cold sample.
    #[arg(long, default_value_t = 5)]
    warm_samples: usize,

    #[arg(long, default_value_t = 512)]
    host_cache_mib: u64,

    /// Typed execution device: auto, cpu, cuda:<ordinal>, or metal:<ordinal>.
    #[arg(long, default_value = "cpu")]
    device: MoeDeviceSpec,

    #[arg(long, default_value_t = 0)]
    device_cache_mib: u64,

    /// Internal isolated resident-baseline child mode.
    #[arg(long, hide = true)]
    resident_baseline_only: bool,
}

enum BenchmarkBackend {
    Olmoe(Arc<OlmoeBackend>),
    Qwen3Moe(Arc<Qwen3MoeBackend>),
}

impl BenchmarkBackend {
    fn new(architecture: MoeArchitecture, config: MoeBackendConfig) -> Result<Self> {
        Ok(match architecture {
            MoeArchitecture::Olmoe => Self::Olmoe(Arc::new(OlmoeBackend::new(config)?)),
            MoeArchitecture::Qwen3Moe => Self::Qwen3Moe(Arc::new(Qwen3MoeBackend::new(config)?)),
        })
    }

    async fn preload(&self, model: &str, checkpoint: &Path) -> PowerResult<ModelManifest> {
        match self {
            Self::Olmoe(backend) => backend.preload(model, checkpoint, None).await,
            Self::Qwen3Moe(backend) => backend.preload(model, checkpoint, None).await,
        }
    }

    fn telemetry(&self, model: &str) -> PowerResult<PlacementTelemetry> {
        match self {
            Self::Olmoe(backend) => backend.telemetry(model),
            Self::Qwen3Moe(backend) => backend.telemetry(model),
        }
    }

    fn device_selection(&self, model: &str) -> PowerResult<MoeDeviceSelection> {
        match self {
            Self::Olmoe(backend) => backend.device_selection(model),
            Self::Qwen3Moe(backend) => backend.device_selection(model),
        }
    }

    async fn sample(&self, model: &str, prompt: &str, max_tokens: u32) -> Result<InferenceSample> {
        match self {
            Self::Olmoe(backend) => {
                run_streaming(backend.as_ref(), model, prompt, max_tokens).await
            }
            Self::Qwen3Moe(backend) => {
                run_streaming(backend.as_ref(), model, prompt, max_tokens).await
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    validate_arguments(&args)?;
    if args.resident_baseline_only {
        return run_resident_child(&args);
    }

    let checkpoint = args.checkpoint.canonicalize().with_context(|| {
        format!(
            "failed to resolve packed checkpoint '{}'",
            args.checkpoint.display()
        )
    })?;
    let architecture = MoeArchitecture::detect(&checkpoint)?;
    if architecture == MoeArchitecture::Qwen3Moe && args.resident_checkpoint.is_some() {
        anyhow::bail!(
            "the public Qwen3-MoE gate uses the independent oracle for parity; \
             --resident-checkpoint is available only for OLMoE"
        );
    }
    let model_name = args
        .model
        .clone()
        .unwrap_or_else(|| format!("{}-benchmark", architecture.default_model_name()));
    let checkpoint_label = checkpoint_label(&args, &checkpoint)?;
    let (vocabulary_size, context_length) = model_geometry(architecture, &checkpoint)?;
    let tokenizer = MoeTokenizer::from_file(checkpoint.join("tokenizer.json"), vocabulary_size)?;
    let prompt_tokens = tokenizer.encode(&args.prompt, true)?;
    let host_cache_bytes = mib(args.host_cache_mib)?;
    let device_cache_bytes = mib(args.device_cache_mib)?;
    let mut inference_limits = architecture.inference_limits();
    inference_limits.max_context_tokens = context_length;
    inference_limits.max_generated_tokens = args.max_tokens as usize;
    inference_limits.max_concurrent_requests = 1;
    inference_limits.max_queued_requests = 1;
    let backend = BenchmarkBackend::new(
        architecture,
        MoeBackendConfig {
            device: args.device.preference(),
            inference_limits,
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
        },
    )?;

    let load_started = Instant::now();
    let manifest = backend
        .preload(&model_name, &checkpoint)
        .await
        .context("failed to preload the packed checkpoint")?;
    let packed_load_ns = duration_ns(load_started.elapsed());
    let device = backend.device_selection(&model_name)?;

    let before_first = backend.telemetry(&model_name)?;
    let first_generation = backend
        .sample(&model_name, &args.prompt, args.max_tokens)
        .await?;
    let after_first = backend.telemetry(&model_name)?;
    let mut warm_generations = Vec::with_capacity(args.warm_samples);
    for _ in 0..args.warm_samples {
        warm_generations.push(
            backend
                .sample(&model_name, &args.prompt, args.max_tokens)
                .await?,
        );
    }
    let after_warm = backend.telemetry(&model_name)?;

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
        schema: match architecture {
            MoeArchitecture::Olmoe => OLMOE_SCHEMA,
            MoeArchitecture::Qwen3Moe => QWEN3_MOE_SCHEMA,
        },
        generated_at: chrono::Utc::now().to_rfc3339(),
        implementation: ImplementationEvidence {
            crate_name: env!("CARGO_PKG_NAME"),
            crate_version: env!("CARGO_PKG_VERSION"),
            power_revision: env!("A3S_POWER_REVISION"),
            execution: "f32-dense-power-streamed-experts",
        },
        model: ModelEvidence {
            name: manifest.name,
            family: architecture.model_type(),
            weights_sha256: manifest.sha256,
            packed_checkpoint: checkpoint_label,
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

fn validate_arguments(args: &Args) -> Result<()> {
    if args.prompt.is_empty() {
        anyhow::bail!("--prompt must not be empty");
    }
    if args.max_tokens == 0 {
        anyhow::bail!("--max-tokens must be greater than zero");
    }
    if args.warm_samples == 0 {
        anyhow::bail!("--warm-samples must be greater than zero");
    }
    Ok(())
}

fn checkpoint_label(args: &Args, checkpoint: &Path) -> Result<String> {
    match args.checkpoint_label.as_deref() {
        Some(label) if label.trim().is_empty() => {
            anyhow::bail!("--checkpoint-label must not be empty")
        }
        Some(label) => Ok(label.to_string()),
        None => Ok(checkpoint
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("packed-checkpoint")
            .to_string()),
    }
}

fn model_geometry(architecture: MoeArchitecture, checkpoint: &Path) -> Result<(usize, usize)> {
    Ok(match architecture {
        MoeArchitecture::Olmoe => {
            let config = OlmoeConfig::from_json_path(checkpoint.join("config.json"))?;
            (config.vocab_size, config.max_position_embeddings)
        }
        MoeArchitecture::Qwen3Moe => {
            let config = Qwen3MoeConfig::from_json_path(checkpoint.join("config.json"))?;
            (config.vocab_size, config.max_position_embeddings)
        }
    })
}

fn run_resident_child(args: &Args) -> Result<()> {
    let path = args
        .resident_checkpoint
        .as_ref()
        .context("resident child mode requires --resident-checkpoint")?;
    if MoeArchitecture::detect(path)? != MoeArchitecture::Olmoe {
        anyhow::bail!("resident baseline child supports only OLMoE");
    }
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
    resident_checkpoint: &Path,
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

async fn run_streaming<B: Backend>(
    backend: &B,
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

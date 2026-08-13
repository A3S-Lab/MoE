use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use a3s_moe::olmoe::OlmoeEncryptedCheckpointSource;
use a3s_moe::service::{OlmoeBackend, OlmoeBackendConfig, OlmoeDeviceSpec};
use a3s_power::config::PowerConfig;
use a3s_power::inference::{InferenceLimits, ResidencyPolicy, SeekableWeightKey, TelemetryMode};
use a3s_power::server::PowerServerBuilder;
use anyhow::{bail, Context, Result};
use clap::Parser;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(about = "Serve a packed OLMoE checkpoint through A3S Power")]
struct Args {
    /// Self-contained checkpoint created by a3s-moe-pack.
    checkpoint: PathBuf,

    /// API model identifier.
    #[arg(long, default_value = "olmoe")]
    model: String,

    /// Optional Power ACL configuration.
    #[arg(long)]
    power_config: Option<PathBuf>,

    /// Override the Power bind host.
    #[arg(long)]
    host: Option<String>,

    /// Override the Power HTTP port.
    #[arg(long)]
    port: Option<u16>,

    /// Optional Hugging Face compatible Jinja chat template.
    #[arg(long)]
    chat_template: Option<PathBuf>,

    /// Pinned SHA-256 of confidential.json for an encrypted checkpoint.
    #[arg(long, requires = "encrypted_key_env")]
    encrypted_manifest_sha256: Option<String>,

    /// Environment variable containing the 64-hex-character encryption key.
    #[arg(long, requires = "encrypted_manifest_sha256")]
    encrypted_key_env: Option<String>,

    /// Power host-tier expert cache bound in MiB; zero streams every record.
    #[arg(long, default_value_t = 512)]
    host_cache_mib: u64,

    /// Typed execution device: auto, cpu, cuda:<ordinal>, or metal:<ordinal>.
    #[arg(long, default_value = "cpu")]
    device: OlmoeDeviceSpec,

    /// Power device-tier expert cache bound in MiB.
    #[arg(long, default_value_t = 0)]
    device_cache_mib: u64,

    /// Maximum simultaneously active continuous-batch members.
    #[arg(long, default_value_t = 4)]
    max_concurrent_requests: usize,

    /// Maximum requests waiting for model admission.
    #[arg(long, default_value_t = 64)]
    max_queued_requests: usize,

    /// Runtime context bound, including prompt and generated tokens.
    #[arg(long, default_value_t = 4096)]
    max_context_tokens: usize,

    /// Per-request generated-token hard bound.
    #[arg(long, default_value_t = 2048)]
    max_generated_tokens: usize,

    /// Default generated-token count when the request omits max_tokens.
    #[arg(long, default_value_t = 512)]
    default_max_tokens: usize,

    /// Admission window used to form a route-unioned batch.
    #[arg(long, default_value_t = 2)]
    batch_window_ms: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();
    let mut power = match &args.power_config {
        Some(path) => {
            let path_text = path
                .to_str()
                .with_context(|| format!("Power config path '{}' is not UTF-8", path.display()))?;
            PowerConfig::load_from(path_text)
                .with_context(|| format!("failed to load Power config '{}'", path.display()))?
        }
        None => PowerConfig::default(),
    };
    if let Some(host) = args.host {
        power.host = host;
    }
    if let Some(port) = args.port {
        power.port = port;
    }
    power.num_parallel = args.max_concurrent_requests;
    power.max_concurrent_requests = u64::try_from(args.max_concurrent_requests)
        .context("max concurrent request count exceeds u64")?;

    let limits = InferenceLimits {
        max_concurrent_requests: args.max_concurrent_requests,
        max_queued_requests: args.max_queued_requests,
        max_context_tokens: args.max_context_tokens,
        max_generated_tokens: args.max_generated_tokens,
        ..InferenceLimits::default()
    };
    let host_cache_bytes = mib(args.host_cache_mib)?;
    let device_cache_bytes = mib(args.device_cache_mib)?;
    let residency = ResidencyPolicy {
        host_cache_bytes,
        device_cache_bytes,
        max_background_inflight_bytes: 512 * 1024 * 1024,
        telemetry: TelemetryMode::Aggregate,
        ..ResidencyPolicy::default()
    };
    let backend = Arc::new(OlmoeBackend::new(OlmoeBackendConfig {
        device: args.device.preference(),
        inference_limits: limits,
        residency_policy: residency,
        batch_window: Duration::from_millis(args.batch_window_ms),
        default_max_tokens: args.default_max_tokens,
        ..OlmoeBackendConfig::default()
    })?);
    let template_override = match args.chat_template {
        Some(path) => Some(
            std::fs::read_to_string(&path)
                .with_context(|| format!("failed to read chat template '{}'", path.display()))?,
        ),
        None => None,
    };
    let manifest = match (args.encrypted_manifest_sha256, args.encrypted_key_env) {
        (Some(manifest_sha256), Some(key_environment)) => {
            let key = SeekableWeightKey::from_env(&key_environment)
                .context("failed to load encrypted checkpoint key")?;
            let source = OlmoeEncryptedCheckpointSource::new(args.checkpoint, manifest_sha256, key)
                .context("invalid encrypted checkpoint source")?;
            backend
                .preload_encrypted(args.model, source, template_override)
                .await?
        }
        (None, None) => {
            backend
                .preload(args.model, args.checkpoint, template_override)
                .await?
        }
        _ => bail!("--encrypted-manifest-sha256 and --encrypted-key-env must be provided together"),
    };
    let device = backend.device_selection(&manifest.name)?;
    tracing::info!(
        model = %manifest.name,
        weights_sha256 = %manifest.sha256,
        size = manifest.size,
        requested_device = %args.device,
        resolved_device = %device.resolved.name(),
        automatic_cpu_fallback = device.automatic_cpu_fallback,
        effective_device_cache_bytes = device.effective_device_cache_bytes,
        "verified and loaded packed OLMoE model"
    );

    PowerServerBuilder::new(power)
        .with_backend(backend)
        .with_model_manifest(manifest)
        .without_default_backends()
        .start()
        .await?;
    Ok(())
}

fn mib(value: u64) -> Result<u64> {
    value
        .checked_mul(1024 * 1024)
        .context("MiB value overflowed u64")
}

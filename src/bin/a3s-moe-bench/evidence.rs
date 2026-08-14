use std::time::Duration;

use a3s_power::inference::{DevicePreference, PlacementTelemetry, RuntimeDeviceIdentity};
use serde::{Deserialize, Serialize};

pub(super) const OLMOE_SCHEMA: &str = "a3s.moe.olmoe-performance.v1";
pub(super) const QWEN3_MOE_SCHEMA: &str = "a3s.moe.qwen3-moe-performance.v1";

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct BenchmarkEvidence {
    pub schema: &'static str,
    pub generated_at: String,
    pub implementation: ImplementationEvidence,
    pub model: ModelEvidence,
    pub system: SystemEvidence,
    pub configuration: ConfigurationEvidence,
    pub cache_state: CacheStateEvidence,
    pub load: LoadEvidence,
    pub first_generation: InferenceSample,
    pub warm_generations: Vec<InferenceSample>,
    pub warm_summary: SampleSummary,
    pub placement: PlacementDelta,
    pub process_peak_rss_bytes: Option<u64>,
    pub resident_baseline: Option<ResidentBaseline>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ImplementationEvidence {
    pub crate_name: &'static str,
    pub crate_version: &'static str,
    pub moe_revision: &'static str,
    pub power_revision: &'static str,
    pub execution: &'static str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ModelEvidence {
    pub name: String,
    pub family: &'static str,
    pub weights_sha256: String,
    pub packed_checkpoint: String,
    pub packed_bytes: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct SystemEvidence {
    pub os: &'static str,
    pub architecture: &'static str,
    pub logical_parallelism: usize,
    pub processor: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ConfigurationEvidence {
    pub prompt_bytes: usize,
    pub prompt_tokens: usize,
    pub max_new_tokens: u32,
    pub warm_samples: usize,
    pub temperature: f32,
    pub host_cache_bytes: u64,
    pub device_cache_bytes: u64,
    pub effective_device_cache_bytes: u64,
    pub requested_device: DevicePreference,
    pub resolved_device: RuntimeDeviceIdentity,
    pub automatic_cpu_fallback: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct CacheStateEvidence {
    pub power_first_generation: &'static str,
    pub power_warm_generations: &'static str,
    pub operating_system_page_cache: &'static str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct LoadEvidence {
    pub packed_streaming_ns: u64,
    pub resident_baseline_ns: Option<u64>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct InferenceSample {
    pub total_ns: u64,
    pub time_to_first_token_ns: u64,
    pub model_prompt_eval_ns: Option<u64>,
    pub generated_tokens: usize,
    pub tokens_per_second: f64,
    pub token_ids: Vec<u32>,
    pub done_reason: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct SampleSummary {
    pub samples: usize,
    pub mean_total_ns: f64,
    pub mean_time_to_first_token_ns: f64,
    pub mean_tokens_per_second: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct PlacementDelta {
    pub first_storage_reads: u64,
    pub first_storage_bytes_read: u64,
    pub warm_storage_reads: u64,
    pub warm_storage_bytes_read: u64,
    pub host_cache_hits: u64,
    pub host_resident_bytes: u64,
    pub host_evictions: u64,
    pub staged_peak_inflight_bytes: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ResidentBaseline {
    pub load_ns: u64,
    pub sample: InferenceSample,
    pub token_parity_with_streaming: bool,
    pub process_peak_rss_bytes: Option<u64>,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ResidentChildEvidence {
    pub load_ns: u64,
    pub sample: InferenceSample,
    pub process_peak_rss_bytes: Option<u64>,
}

pub(super) fn summarize(samples: &[InferenceSample]) -> SampleSummary {
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

pub(super) fn placement_delta(
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

pub(super) fn throughput(tokens: usize, duration: Duration) -> f64 {
    if duration.is_zero() {
        0.0
    } else {
        tokens as f64 / duration.as_secs_f64()
    }
}

pub(super) fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

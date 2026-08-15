use std::path::Path;

use a3s_power::inference::{
    DevicePreference, EmbeddedRuntime, InferenceLimits, ResidencyPolicy, RuntimeDeviceKind,
    TelemetryMode,
};
use candle_core::Tensor;
use tokio_util::sync::CancellationToken;

use crate::{MoeArchitecture, MoeError, Result};

use super::{Qwen36MoeCheckpoint, Qwen36MoeLayerType, Qwen36MoePackedCheckpoint};

mod input;
mod schema;

use input::{
    read_oracle, validate_files, validate_oracle_shape, validate_provenance, validate_token_ids,
};
pub use schema::{
    Qwen36MoeNumericComparison, Qwen36MoeOracleFile, Qwen36MoeOracleInput, Qwen36MoeOracleModel,
    Qwen36MoeOracleOutput, Qwen36MoeOracleRoute, Qwen36MoePublicOracle, Qwen36MoeValidationReport,
    Qwen36MoeValidationStatus, Qwen36MoeValidationTolerances, QWEN36_MOE_PUBLIC_MODEL_ID,
    QWEN36_MOE_PUBLIC_MODEL_REVISION, QWEN36_MOE_PUBLIC_ORACLE_SCHEMA,
    QWEN36_MOE_TRANSFORMERS_DTYPE, QWEN36_MOE_TRANSFORMERS_REVISION,
    QWEN36_MOE_TRANSFORMERS_SOURCE_SHA256, QWEN36_MOE_VALIDATION_SCHEMA,
};

const DEFAULT_HOST_CACHE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Resource and numerical policy for public Qwen3.6 validation.
#[derive(Debug, Clone)]
pub struct Qwen36MoeValidationOptions {
    pub device: DevicePreference,
    pub tolerances: Qwen36MoeValidationTolerances,
    pub inference_limits: InferenceLimits,
    pub residency_policy: ResidencyPolicy,
}

impl Default for Qwen36MoeValidationOptions {
    fn default() -> Self {
        Self {
            device: DevicePreference::Cpu,
            tolerances: Qwen36MoeValidationTolerances::default(),
            inference_limits: MoeArchitecture::Qwen35Moe.inference_limits(),
            residency_policy: ResidencyPolicy {
                host_cache_bytes: DEFAULT_HOST_CACHE_BYTES,
                telemetry: TelemetryMode::Aggregate,
                ..MoeArchitecture::Qwen35Moe.residency_policy()
            },
        }
    }
}

/// Compare the pinned source checkpoint and its packed Power-streaming form
/// with the independent F32 Transformers text oracle.
pub async fn validate_public_checkpoint(
    source_root: impl AsRef<Path>,
    packed_root: impl AsRef<Path>,
    oracle_path: impl AsRef<Path>,
    options: Qwen36MoeValidationOptions,
) -> Result<Qwen36MoeValidationReport> {
    validate_options(&options)?;
    let Qwen36MoeValidationOptions {
        device,
        tolerances,
        inference_limits,
        mut residency_policy,
    } = options;
    let oracle = read_oracle(oracle_path.as_ref())?;
    validate_provenance(&oracle)?;

    let source = Qwen36MoeCheckpoint::open(source_root)?;
    let source_weights_sha256 = validate_files(&source, &oracle.model.files)?;
    let source_tokenizer = source.load_tokenizer()?;
    let encoded = source_tokenizer.encode(&oracle.input.text, oracle.input.add_special_tokens)?;
    validate_token_ids(&encoded, &oracle)?;
    validate_oracle_shape(&oracle, source.config())?;

    let runtime = EmbeddedRuntime::new(device, inference_limits)?;
    let resolved_device = runtime.device().identity();
    let automatic_cpu_fallback =
        device == DevicePreference::Auto && resolved_device.kind == RuntimeDeviceKind::Cpu;
    if resolved_device.kind == RuntimeDeviceKind::Cpu {
        residency_policy.device_cache_bytes = 0;
    }
    let host_cache_bytes = residency_policy.host_cache_bytes;
    let device_cache_bytes = residency_policy.device_cache_bytes;
    let packed = Qwen36MoePackedCheckpoint::open(packed_root, runtime.clone())?;
    if packed.manifest().source_weights_sha256 != source_weights_sha256 {
        return Err(MoeError::InvalidConfig(format!(
            "packed checkpoint source digest '{}' does not match the pinned source digest '{source_weights_sha256}'",
            packed.manifest().source_weights_sha256
        )));
    }
    if packed.config() != source.config() {
        return Err(MoeError::InvalidConfig(
            "packed Qwen3.6 configuration does not match the pinned source".to_string(),
        ));
    }
    let packed_tokens = packed
        .load_tokenizer()?
        .encode(&oracle.input.text, oracle.input.add_special_tokens)?;
    if packed_tokens != encoded {
        return Err(MoeError::InvalidConfig(
            "packed tokenizer does not match the pinned source tokenizer".to_string(),
        ));
    }

    let packed_weights_sha256 = packed.manifest().weights_sha256();
    let model = packed.load_streaming(residency_policy)?;
    let input = Tensor::from_vec(
        encoded.clone(),
        (1, encoded.len()),
        runtime.device().tensor_device(),
    )?;
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation)?;
    let output = model
        .forward(&input, &mut model.new_cache(), &permit, &cancellation)
        .await?;
    let actual_logits = output.logits.to_vec3::<f32>()?;
    let actual_logits = actual_logits.first().ok_or_else(|| {
        MoeError::InvalidTensor("streaming decoder returned no batch rows".to_string())
    })?;

    let mut logits = Qwen36MoeNumericComparison::new(tolerances.logits_abs);
    let mut argmax_token_mismatches = 0_u64;
    for (actual, expected) in actual_logits.iter().zip(&oracle.output.logits) {
        for (&actual, &expected) in actual.iter().zip(expected) {
            logits.observe(actual, expected)?;
        }
        if argmax(actual)? != argmax(expected)? {
            argmax_token_mismatches = argmax_token_mismatches.saturating_add(1);
        }
    }

    let mut router_logits = Qwen36MoeNumericComparison::new(tolerances.router_logits_abs);
    let mut route_weights = Qwen36MoeNumericComparison::new(tolerances.route_weights_abs);
    let mut route_expert_mismatches = 0_u64;
    let mut route_order_mismatches = 0_u64;
    let mut routes_checked = 0_u64;
    for layer in 0..source.config().num_hidden_layers {
        let actual_router = output.layer_router_logits[layer].to_vec3::<f32>()?;
        let actual_router = actual_router.first().ok_or_else(|| {
            MoeError::InvalidTensor("router logits returned no batch rows".to_string())
        })?;
        for (actual_row, expected_row) in actual_router
            .iter()
            .zip(&oracle.output.layer_router_logits[layer])
        {
            for (&actual, &expected) in actual_row.iter().zip(expected_row) {
                router_logits.observe(actual, expected)?;
            }
        }
        for (actual_row, expected_row) in output.layer_routes[layer]
            .selections()
            .iter()
            .zip(&oracle.output.layer_routes[layer])
        {
            for (actual, expected_at_rank) in actual_row.iter().zip(expected_row) {
                routes_checked = routes_checked.saturating_add(1);
                if actual.expert != expected_at_rank.expert {
                    route_order_mismatches = route_order_mismatches.saturating_add(1);
                }
                if let Some(expected) = expected_row
                    .iter()
                    .find(|expected| expected.expert == actual.expert)
                {
                    route_weights.observe(actual.weight, expected.weight)?;
                } else {
                    route_expert_mismatches = route_expert_mismatches.saturating_add(1);
                }
            }
        }
    }

    let failed = logits.mismatch_count > 0
        || router_logits.mismatch_count > 0
        || route_weights.mismatch_count > 0
        || route_expert_mismatches > 0
        || argmax_token_mismatches > 0;
    let linear_attention_layers = source
        .config()
        .layer_types
        .iter()
        .filter(|kind| **kind == Qwen36MoeLayerType::LinearAttention)
        .count();
    Ok(Qwen36MoeValidationReport {
        schema: QWEN36_MOE_VALIDATION_SCHEMA,
        moe_revision: env!("A3S_MOE_REVISION"),
        power_revision: env!("A3S_POWER_REVISION"),
        requested_device: device,
        resolved_device,
        automatic_cpu_fallback,
        host_cache_bytes,
        device_cache_bytes,
        model_id: oracle.model.id,
        model_revision: oracle.model.revision,
        source_weights_sha256,
        packed_weights_sha256,
        prompt_tokens: encoded.len(),
        vocabulary_size: source.config().vocab_size,
        layers: source.config().num_hidden_layers,
        linear_attention_layers,
        full_attention_layers: source.config().num_hidden_layers - linear_attention_layers,
        experts: source.config().num_experts,
        routes_checked,
        logits,
        router_logits,
        route_weights,
        route_expert_mismatches,
        route_order_mismatches,
        argmax_token_mismatches,
        status: if failed {
            Qwen36MoeValidationStatus::Failed
        } else {
            Qwen36MoeValidationStatus::Passed
        },
    })
}

fn validate_options(options: &Qwen36MoeValidationOptions) -> Result<()> {
    options.inference_limits.validate()?;
    options.residency_policy.validate()?;
    if options.device == DevicePreference::Cpu && options.residency_policy.device_cache_bytes != 0 {
        return Err(MoeError::InvalidConfig(
            "an explicit CPU validation device cannot reserve an accelerator cache".to_string(),
        ));
    }
    for (name, value) in [
        ("logits_abs", options.tolerances.logits_abs),
        ("router_logits_abs", options.tolerances.router_logits_abs),
        ("route_weights_abs", options.tolerances.route_weights_abs),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(MoeError::InvalidConfig(format!(
                "validation tolerance {name} must be finite and non-negative"
            )));
        }
    }
    Ok(())
}

fn argmax(values: &[f32]) -> Result<usize> {
    let mut values = values.iter().copied().enumerate();
    let (mut best_index, mut best_value) = values
        .next()
        .ok_or_else(|| MoeError::InvalidTensor("cannot take argmax of an empty row".to_string()))?;
    if !best_value.is_finite() {
        return Err(MoeError::InvalidTensor(
            "argmax row contains a non-finite value".to_string(),
        ));
    }
    for (index, value) in values {
        if !value.is_finite() {
            return Err(MoeError::InvalidTensor(
                "argmax row contains a non-finite value".to_string(),
            ));
        }
        if value > best_value {
            best_index = index;
            best_value = value;
        }
    }
    Ok(best_index)
}

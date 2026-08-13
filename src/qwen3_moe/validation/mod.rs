use std::path::Path;

use a3s_power::inference::{
    DevicePreference, EmbeddedRuntime, InferenceLimits, ResidencyPolicy, TelemetryMode,
};
use candle_core::{Device, Tensor};
use tokio_util::sync::CancellationToken;

use crate::{MoeArchitecture, MoeError, Result};

use super::{Qwen3MoeCheckpoint, Qwen3MoePackedCheckpoint};

mod input;
mod schema;

use input::{
    read_oracle, validate_files, validate_oracle_shape, validate_provenance, validate_token_ids,
};
pub use schema::{
    Qwen3MoeNumericComparison, Qwen3MoeOracleFile, Qwen3MoeOracleInput, Qwen3MoeOracleModel,
    Qwen3MoeOracleOutput, Qwen3MoeOracleRoute, Qwen3MoePublicOracle, Qwen3MoeValidationReport,
    Qwen3MoeValidationStatus, Qwen3MoeValidationTolerances, QWEN3_MOE_PUBLIC_MODEL_ID,
    QWEN3_MOE_PUBLIC_MODEL_REVISION, QWEN3_MOE_PUBLIC_ORACLE_SCHEMA, QWEN3_MOE_TRANSFORMERS_DTYPE,
    QWEN3_MOE_TRANSFORMERS_REVISION, QWEN3_MOE_TRANSFORMERS_SOURCE_SHA256,
    QWEN3_MOE_VALIDATION_SCHEMA,
};

const DEFAULT_HOST_CACHE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Resource and numerical policy for public Qwen3-MoE validation.
#[derive(Debug, Clone)]
pub struct Qwen3MoeValidationOptions {
    pub tolerances: Qwen3MoeValidationTolerances,
    pub inference_limits: InferenceLimits,
    pub residency_policy: ResidencyPolicy,
}

impl Default for Qwen3MoeValidationOptions {
    fn default() -> Self {
        Self {
            tolerances: Qwen3MoeValidationTolerances::default(),
            inference_limits: MoeArchitecture::Qwen3Moe.inference_limits(),
            residency_policy: ResidencyPolicy {
                host_cache_bytes: DEFAULT_HOST_CACHE_BYTES,
                telemetry: TelemetryMode::Aggregate,
                ..MoeArchitecture::Qwen3Moe.residency_policy()
            },
        }
    }
}

/// Compare a pinned source checkpoint and its packed Power-streaming form with
/// an independently generated BF16 Transformers oracle.
pub async fn validate_public_checkpoint(
    source_root: impl AsRef<Path>,
    packed_root: impl AsRef<Path>,
    oracle_path: impl AsRef<Path>,
    options: Qwen3MoeValidationOptions,
) -> Result<Qwen3MoeValidationReport> {
    validate_options(&options)?;
    let oracle = read_oracle(oracle_path.as_ref())?;
    validate_provenance(&oracle)?;

    let source = Qwen3MoeCheckpoint::open(source_root)?;
    let source_weights_sha256 = validate_files(&source, &oracle.model.files)?;
    let source_tokenizer = source.load_tokenizer()?;
    let encoded = source_tokenizer.encode(&oracle.input.text, oracle.input.add_special_tokens)?;
    validate_token_ids(&encoded, &oracle)?;
    validate_oracle_shape(&oracle, source.config())?;

    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, options.inference_limits)?;
    let packed = Qwen3MoePackedCheckpoint::open(packed_root, runtime.clone())?;
    if packed.manifest().source_weights_sha256 != source_weights_sha256 {
        return Err(MoeError::InvalidConfig(format!(
            "packed checkpoint source digest '{}' does not match the pinned source digest '{source_weights_sha256}'",
            packed.manifest().source_weights_sha256
        )));
    }
    if packed.config() != source.config() {
        return Err(MoeError::InvalidConfig(
            "packed Qwen3-MoE configuration does not match the pinned source".to_string(),
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
    let model = packed.load_cpu_streaming(options.residency_policy)?;
    let input = Tensor::from_vec(encoded.clone(), (1, encoded.len()), &Device::Cpu)?;
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation)?;
    let output = model
        .forward(&input, &mut model.new_cache(), &permit, &cancellation)
        .await?;
    let actual_logits = output.logits.to_vec3::<f32>()?;
    let actual_logits = actual_logits.first().ok_or_else(|| {
        MoeError::InvalidTensor("streaming decoder returned no batch rows".to_string())
    })?;

    let mut logits = Qwen3MoeNumericComparison::new(options.tolerances.logits_abs);
    let mut argmax_token_mismatches = 0_u64;
    for (actual, expected) in actual_logits.iter().zip(&oracle.output.logits) {
        for (&actual, &expected) in actual.iter().zip(expected) {
            logits.observe(actual, expected)?;
        }
        if argmax(actual)? != argmax(expected)? {
            argmax_token_mismatches = argmax_token_mismatches.saturating_add(1);
        }
    }

    let mut router_logits = Qwen3MoeNumericComparison::new(options.tolerances.router_logits_abs);
    let mut route_weights = Qwen3MoeNumericComparison::new(options.tolerances.route_weights_abs);
    let mut route_expert_mismatches = 0_u64;
    let mut routes_checked = 0_u64;
    for layer in 0..source.config().num_hidden_layers {
        match (
            &output.layer_router_logits[layer],
            &oracle.output.layer_router_logits[layer],
        ) {
            (Some(actual), Some(expected)) => {
                let actual = actual.to_vec3::<f32>()?;
                let actual = actual.first().ok_or_else(|| {
                    MoeError::InvalidTensor("router logits returned no batch rows".to_string())
                })?;
                for (actual_row, expected_row) in actual.iter().zip(expected) {
                    for (&actual, &expected) in actual_row.iter().zip(expected_row) {
                        router_logits.observe(actual, expected)?;
                    }
                }
            }
            (None, None) => {}
            _ => {
                return Err(MoeError::InvalidTensor(format!(
                    "router presence differs from the oracle at layer {layer}"
                )))
            }
        }
        match (
            &output.layer_routes[layer],
            &oracle.output.layer_routes[layer],
        ) {
            (Some(actual), Some(expected)) => {
                for (actual_row, expected_row) in actual.selections().iter().zip(expected) {
                    for (actual, expected) in actual_row.iter().zip(expected_row) {
                        routes_checked = routes_checked.saturating_add(1);
                        if actual.expert != expected.expert {
                            route_expert_mismatches = route_expert_mismatches.saturating_add(1);
                        }
                        route_weights.observe(actual.weight, expected.weight)?;
                    }
                }
            }
            (None, None) => {}
            _ => {
                return Err(MoeError::InvalidTensor(format!(
                    "route presence differs from the oracle at layer {layer}"
                )))
            }
        }
    }

    let failed = logits.mismatch_count > 0
        || router_logits.mismatch_count > 0
        || route_weights.mismatch_count > 0
        || route_expert_mismatches > 0
        || argmax_token_mismatches > 0;
    Ok(Qwen3MoeValidationReport {
        schema: QWEN3_MOE_VALIDATION_SCHEMA,
        model_id: oracle.model.id,
        model_revision: oracle.model.revision,
        source_weights_sha256,
        packed_weights_sha256,
        prompt_tokens: encoded.len(),
        vocabulary_size: source.config().vocab_size,
        layers: source.config().num_hidden_layers,
        sparse_layers: (0..source.config().num_hidden_layers)
            .filter(|layer| source.config().is_sparse_layer(*layer))
            .count(),
        experts: source.config().num_experts,
        routes_checked,
        logits,
        router_logits,
        route_weights,
        route_expert_mismatches,
        argmax_token_mismatches,
        status: if failed {
            Qwen3MoeValidationStatus::Failed
        } else {
            Qwen3MoeValidationStatus::Passed
        },
    })
}

fn validate_options(options: &Qwen3MoeValidationOptions) -> Result<()> {
    options.inference_limits.validate()?;
    options.residency_policy.validate()?;
    if options.residency_policy.device_cache_bytes != 0 {
        return Err(MoeError::InvalidConfig(
            "CPU public validation cannot reserve a device cache".to_string(),
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

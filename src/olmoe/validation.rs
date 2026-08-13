use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Component, Path};

use candle_core::{Device, Tensor};
use sha2::{Digest, Sha256};

use crate::{MoeError, Result};

use super::OlmoeCheckpoint;

mod schema;

pub use schema::{
    OlmoeNumericComparison, OlmoeOracleFile, OlmoeOracleInput, OlmoeOracleModel, OlmoeOracleOutput,
    OlmoeOracleRoute, OlmoePublicOracle, OlmoeValidationReport, OlmoeValidationStatus,
    OlmoeValidationTolerances, OLMOE_PUBLIC_MODEL_ID, OLMOE_PUBLIC_MODEL_REVISION,
    OLMOE_PUBLIC_ORACLE_SCHEMA, OLMOE_TRANSFORMERS_REVISION,
};

const OLMOE_VALIDATION_SCHEMA: &str = "a3s.moe.olmoe-validation.v1";
const MAX_ORACLE_BYTES: u64 = 32 * 1024 * 1024;

/// Load the exact resident checkpoint and compare it with the pinned,
/// independently generated Transformers oracle.
pub fn validate_public_checkpoint(
    checkpoint_root: impl AsRef<Path>,
    oracle_path: impl AsRef<Path>,
    tolerances: OlmoeValidationTolerances,
) -> Result<OlmoeValidationReport> {
    validate_tolerances(tolerances)?;
    let oracle = read_oracle(oracle_path.as_ref())?;
    validate_provenance(&oracle)?;

    let checkpoint = OlmoeCheckpoint::open(checkpoint_root)?;
    validate_files(&checkpoint, &oracle.model.files)?;
    let tokenizer = checkpoint.load_tokenizer()?;
    let encoded = tokenizer.encode(&oracle.input.text, oracle.input.add_special_tokens)?;
    if encoded != oracle.input.token_ids {
        return Err(MoeError::InvalidConfig(format!(
            "tokenizer emitted {encoded:?}, but the oracle declares {:?}",
            oracle.input.token_ids
        )));
    }
    if encoded.is_empty() {
        return Err(MoeError::InvalidConfig(
            "public oracle prompt must produce at least one token".to_string(),
        ));
    }

    validate_oracle_shape(&oracle, checkpoint.config())?;
    let model = checkpoint.load_cpu_resident()?;
    let input = Tensor::from_vec(encoded.clone(), (1, encoded.len()), &Device::Cpu)?;
    let output = model.forward(&input, &mut model.new_cache())?;
    let actual_logits = output.logits.to_vec3::<f32>()?;
    let actual_logits = actual_logits.first().ok_or_else(|| {
        MoeError::InvalidTensor("resident decoder returned no batch rows".to_string())
    })?;

    let mut logits = OlmoeNumericComparison::new(tolerances.logits_abs);
    let mut argmax_token_mismatches = 0_u64;
    for (actual, expected) in actual_logits.iter().zip(&oracle.output.logits) {
        for (&actual, &expected) in actual.iter().zip(expected) {
            logits.observe(actual, expected)?;
        }
        if argmax(actual)? != argmax(expected)? {
            argmax_token_mismatches = argmax_token_mismatches.saturating_add(1);
        }
    }

    let mut router_logits = OlmoeNumericComparison::new(tolerances.router_logits_abs);
    for (actual_layer, expected_layer) in output
        .layer_router_logits
        .iter()
        .zip(&oracle.output.layer_router_logits)
    {
        let actual_layer = actual_layer.to_vec3::<f32>()?;
        let actual_layer = actual_layer.first().ok_or_else(|| {
            MoeError::InvalidTensor("router logits returned no batch rows".to_string())
        })?;
        for (actual_row, expected_row) in actual_layer.iter().zip(expected_layer) {
            for (&actual, &expected) in actual_row.iter().zip(expected_row) {
                router_logits.observe(actual, expected)?;
            }
        }
    }

    let mut route_weights = OlmoeNumericComparison::new(tolerances.route_weights_abs);
    let mut route_expert_mismatches = 0_u64;
    let mut routes_checked = 0_u64;
    for (actual_layer, expected_layer) in
        output.layer_routes.iter().zip(&oracle.output.layer_routes)
    {
        for (actual_row, expected_row) in actual_layer.selections().iter().zip(expected_layer) {
            for (actual, expected) in actual_row.iter().zip(expected_row) {
                routes_checked = routes_checked.saturating_add(1);
                if actual.expert != expected.expert {
                    route_expert_mismatches = route_expert_mismatches.saturating_add(1);
                }
                route_weights.observe(actual.weight, expected.weight)?;
            }
        }
    }

    let failed = logits.mismatch_count > 0
        || router_logits.mismatch_count > 0
        || route_weights.mismatch_count > 0
        || route_expert_mismatches > 0
        || argmax_token_mismatches > 0;
    Ok(OlmoeValidationReport {
        schema: OLMOE_VALIDATION_SCHEMA,
        model_id: oracle.model.id,
        model_revision: oracle.model.revision,
        prompt_tokens: encoded.len(),
        vocabulary_size: checkpoint.config().vocab_size,
        layers: checkpoint.config().num_hidden_layers,
        experts: checkpoint.config().num_experts,
        routes_checked,
        logits,
        router_logits,
        route_weights,
        route_expert_mismatches,
        argmax_token_mismatches,
        status: if failed {
            OlmoeValidationStatus::Failed
        } else {
            OlmoeValidationStatus::Passed
        },
    })
}

fn validate_tolerances(tolerances: OlmoeValidationTolerances) -> Result<()> {
    for (name, value) in [
        ("logits_abs", tolerances.logits_abs),
        ("router_logits_abs", tolerances.router_logits_abs),
        ("route_weights_abs", tolerances.route_weights_abs),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(MoeError::InvalidConfig(format!(
                "validation tolerance {name} must be finite and non-negative"
            )));
        }
    }
    Ok(())
}

fn read_oracle(path: &Path) -> Result<OlmoePublicOracle> {
    let metadata = path.symlink_metadata()?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > MAX_ORACLE_BYTES
    {
        return Err(MoeError::InvalidConfig(format!(
            "oracle '{}' must be a regular non-symlink file no larger than {MAX_ORACLE_BYTES} bytes",
            path.display()
        )));
    }
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

fn validate_provenance(oracle: &OlmoePublicOracle) -> Result<()> {
    for (found, expected, label) in [
        (
            oracle.schema.as_str(),
            OLMOE_PUBLIC_ORACLE_SCHEMA,
            "oracle schema",
        ),
        (oracle.model.id.as_str(), OLMOE_PUBLIC_MODEL_ID, "model ID"),
        (
            oracle.model.revision.as_str(),
            OLMOE_PUBLIC_MODEL_REVISION,
            "model revision",
        ),
        (
            oracle.model.transformers_revision.as_str(),
            OLMOE_TRANSFORMERS_REVISION,
            "Transformers revision",
        ),
    ] {
        if found != expected {
            return Err(MoeError::InvalidConfig(format!(
                "{label} must be '{expected}', found '{found}'"
            )));
        }
    }
    validate_sha256(
        &oracle.model.transformers_source_sha256,
        "Transformers source",
    )
}

fn validate_files(checkpoint: &OlmoeCheckpoint, files: &[OlmoeOracleFile]) -> Result<()> {
    if files.is_empty() {
        return Err(MoeError::InvalidConfig(
            "oracle checkpoint file inventory must not be empty".to_string(),
        ));
    }
    let mut declared = BTreeSet::new();
    for file in files {
        validate_file_name(&file.name)?;
        validate_sha256(&file.sha256, &file.name)?;
        if !declared.insert(file.name.clone()) {
            return Err(MoeError::InvalidConfig(format!(
                "oracle declares checkpoint file '{}' more than once",
                file.name
            )));
        }
        let path = checkpoint.root().join(&file.name);
        let metadata = path.symlink_metadata().map_err(|error| {
            MoeError::InvalidConfig(format!(
                "failed to inspect oracle checkpoint file '{}': {error}",
                path.display()
            ))
        })?;
        if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() != file.bytes
        {
            return Err(MoeError::InvalidConfig(format!(
                "checkpoint file '{}' does not match the oracle byte length {}",
                file.name, file.bytes
            )));
        }
        let actual = sha256_file(&path)?;
        if actual != file.sha256 {
            return Err(MoeError::InvalidConfig(format!(
                "checkpoint file '{}' has SHA-256 {actual}, expected {}",
                file.name, file.sha256
            )));
        }
    }

    let mut required = BTreeSet::from([
        "config.json".to_string(),
        "model.safetensors.index.json".to_string(),
        "tokenizer.json".to_string(),
    ]);
    for shard in checkpoint.shards() {
        let name = shard
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                MoeError::InvalidConfig(format!(
                    "checkpoint shard '{}' has a non-UTF-8 filename",
                    shard.display()
                ))
            })?;
        required.insert(name.to_string());
    }
    let missing = required.difference(&declared).cloned().collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(MoeError::InvalidConfig(format!(
            "oracle is missing required checkpoint files {missing:?}"
        )));
    }
    Ok(())
}

fn validate_oracle_shape(oracle: &OlmoePublicOracle, config: &super::OlmoeConfig) -> Result<()> {
    let positions = oracle.input.token_ids.len();
    require_length("oracle logits", oracle.output.logits.len(), positions)?;
    for row in &oracle.output.logits {
        require_length("oracle logit row", row.len(), config.vocab_size)?;
        require_finite("oracle logit row", row)?;
    }
    require_length(
        "oracle router layers",
        oracle.output.layer_router_logits.len(),
        config.num_hidden_layers,
    )?;
    require_length(
        "oracle route layers",
        oracle.output.layer_routes.len(),
        config.num_hidden_layers,
    )?;
    for layer in &oracle.output.layer_router_logits {
        require_length("oracle router positions", layer.len(), positions)?;
        for row in layer {
            require_length("oracle router row", row.len(), config.num_experts)?;
            require_finite("oracle router row", row)?;
        }
    }
    for layer in &oracle.output.layer_routes {
        require_length("oracle route positions", layer.len(), positions)?;
        for row in layer {
            require_length(
                "oracle selected routes",
                row.len(),
                config.num_experts_per_tok,
            )?;
            let mut experts = BTreeSet::new();
            for route in row {
                if route.expert as usize >= config.num_experts || !experts.insert(route.expert) {
                    return Err(MoeError::InvalidTensor(
                        "oracle route contains an invalid or duplicate expert".to_string(),
                    ));
                }
                if !route.weight.is_finite() || route.weight < 0.0 {
                    return Err(MoeError::InvalidTensor(
                        "oracle route contains an invalid weight".to_string(),
                    ));
                }
            }
        }
    }
    Ok(())
}

fn require_length(label: &str, found: usize, expected: usize) -> Result<()> {
    if found != expected {
        return Err(MoeError::InvalidTensor(format!(
            "{label} has length {found}, expected {expected}"
        )));
    }
    Ok(())
}

fn require_finite(label: &str, values: &[f32]) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(MoeError::InvalidTensor(format!(
            "{label} contains a non-finite value"
        )));
    }
    Ok(())
}

fn validate_file_name(name: &str) -> Result<()> {
    let path = Path::new(name);
    let mut components = path.components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(MoeError::InvalidConfig(format!(
            "oracle file name '{name}' must be one relative filename"
        )));
    }
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(MoeError::InvalidConfig(format!(
            "{label} SHA-256 must contain 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
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

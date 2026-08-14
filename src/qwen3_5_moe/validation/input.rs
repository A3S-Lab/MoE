use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Component, Path};

use sha2::{Digest, Sha256};

use crate::qwen3_5_moe::Qwen36MoeConfig;
use crate::{MoeError, Result};

use super::{
    Qwen36MoeCheckpoint, Qwen36MoeOracleFile, Qwen36MoePublicOracle, QWEN36_MOE_PUBLIC_MODEL_ID,
    QWEN36_MOE_PUBLIC_MODEL_REVISION, QWEN36_MOE_PUBLIC_ORACLE_SCHEMA,
    QWEN36_MOE_TRANSFORMERS_DTYPE, QWEN36_MOE_TRANSFORMERS_REVISION,
    QWEN36_MOE_TRANSFORMERS_SOURCE_SHA256,
};

const MAX_ORACLE_BYTES: u64 = 128 * 1024 * 1024;

pub(super) fn read_oracle(path: &Path) -> Result<Qwen36MoePublicOracle> {
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

pub(super) fn validate_provenance(oracle: &Qwen36MoePublicOracle) -> Result<()> {
    for (found, expected, label) in [
        (
            oracle.schema.as_str(),
            QWEN36_MOE_PUBLIC_ORACLE_SCHEMA,
            "oracle schema",
        ),
        (
            oracle.model.id.as_str(),
            QWEN36_MOE_PUBLIC_MODEL_ID,
            "model ID",
        ),
        (
            oracle.model.revision.as_str(),
            QWEN36_MOE_PUBLIC_MODEL_REVISION,
            "model revision",
        ),
        (
            oracle.model.transformers_revision.as_str(),
            QWEN36_MOE_TRANSFORMERS_REVISION,
            "Transformers revision",
        ),
        (
            oracle.model.transformers_dtype.as_str(),
            QWEN36_MOE_TRANSFORMERS_DTYPE,
            "Transformers dtype",
        ),
    ] {
        if found != expected {
            return Err(MoeError::InvalidConfig(format!(
                "{label} must be '{expected}', found '{found}'"
            )));
        }
    }
    if oracle.model.transformers_source_sha256 != QWEN36_MOE_TRANSFORMERS_SOURCE_SHA256 {
        return Err(MoeError::InvalidConfig(format!(
            "Transformers source SHA-256 must be '{QWEN36_MOE_TRANSFORMERS_SOURCE_SHA256}', found '{}'",
            oracle.model.transformers_source_sha256
        )));
    }
    Ok(())
}

pub(super) fn validate_files(
    checkpoint: &Qwen36MoeCheckpoint,
    files: &[Qwen36MoeOracleFile],
) -> Result<String> {
    if files.is_empty() {
        return Err(MoeError::InvalidConfig(
            "oracle checkpoint file inventory must not be empty".to_string(),
        ));
    }
    let mut files = files.iter().collect::<Vec<_>>();
    files.sort_by(|left, right| left.name.cmp(&right.name));
    let mut declared = BTreeSet::new();
    let mut collection = Sha256::new();
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
        let is_weight = file.name.ends_with(".safetensors");
        if is_weight {
            collection.update((file.name.len() as u64).to_le_bytes());
            collection.update(file.name.as_bytes());
            collection.update(file.bytes.to_le_bytes());
        }
        let actual = sha256_file(&path, is_weight.then_some(&mut collection))?;
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
        "tokenizer_config.json".to_string(),
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
    Ok(format!("{:x}", collection.finalize()))
}

pub(super) fn validate_token_ids(encoded: &[u32], oracle: &Qwen36MoePublicOracle) -> Result<()> {
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
    Ok(())
}

pub(super) fn validate_oracle_shape(
    oracle: &Qwen36MoePublicOracle,
    config: &Qwen36MoeConfig,
) -> Result<()> {
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
    for layer in 0..config.num_hidden_layers {
        let router = &oracle.output.layer_router_logits[layer];
        let routes = &oracle.output.layer_routes[layer];
        require_length("oracle router positions", router.len(), positions)?;
        for row in router {
            require_length("oracle router row", row.len(), config.num_experts)?;
            require_finite("oracle router row", row)?;
        }
        require_length("oracle route positions", routes.len(), positions)?;
        for row in routes {
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

fn sha256_file(path: &Path, mut collection: Option<&mut Sha256>) -> Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
        if let Some(collection) = collection.as_deref_mut() {
            collection.update(&buffer[..read]);
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

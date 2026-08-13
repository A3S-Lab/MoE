use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use a3s_power::error::{PowerError, Result as PowerResult};
use a3s_power::inference::ExecutionBatchBinding;
use a3s_power::model::manifest::{ManifestMessage, ModelFormat, ModelManifest, ModelParameters};

use crate::{MoeError, MoeTokenizer, PackedScalarType, Result};

use super::config::OlmoeDeviceSelection;
use super::source::CheckpointSource;

#[doc(hidden)]
pub struct LoadedArtifacts<M> {
    pub canonical_path: PathBuf,
    pub tokenizer: Arc<MoeTokenizer>,
    pub model: Arc<M>,
    pub binding: ExecutionBatchBinding,
    pub size: u64,
    pub device: OlmoeDeviceSelection,
    pub weights_sha256: String,
    pub scalar_type: PackedScalarType,
    pub context_length: usize,
    pub hidden_size: usize,
    pub eos_token_ids: Vec<u32>,
}

pub(super) struct LoadSpec {
    pub name: String,
    pub source: CheckpointSource,
    pub expected_sha256: Option<String>,
    pub system_prompt: Option<String>,
    pub template_override: Option<String>,
    pub default_parameters: Option<HashMap<String, serde_json::Value>>,
    pub messages: Vec<ManifestMessage>,
}

pub(super) fn power_manifest<M>(
    spec: &LoadSpec,
    artifacts: &LoadedArtifacts<M>,
    family: &str,
) -> ModelManifest {
    ModelManifest {
        name: spec.name.clone(),
        format: ModelFormat::SafeTensors,
        size: artifacts.size,
        sha256: artifacts.weights_sha256.clone(),
        parameters: Some(ModelParameters {
            context_length: u32::try_from(artifacts.context_length).ok(),
            embedding_length: u32::try_from(artifacts.hidden_size).ok(),
            parameter_count: None,
            quantization: Some(match artifacts.scalar_type {
                PackedScalarType::F32 => "F32".to_string(),
                PackedScalarType::Bf16 => "BF16".to_string(),
            }),
        }),
        created_at: chrono::Utc::now(),
        path: artifacts.canonical_path.clone(),
        system_prompt: spec.system_prompt.clone(),
        template_override: spec.template_override.clone(),
        default_parameters: spec.default_parameters.clone(),
        modelfile_content: None,
        license: None,
        adapter_path: None,
        projector_path: None,
        messages: spec.messages.clone(),
        family: Some(family.to_string()),
        families: None,
    }
}

pub(super) fn validate_model_name(name: &str) -> PowerResult<()> {
    if name.trim().is_empty() {
        Err(PowerError::InvalidRequest(
            "MoE model name must not be empty".to_string(),
        ))
    } else {
        Ok(())
    }
}

pub(super) fn directory_size(root: &Path) -> Result<u64> {
    let mut total = 0_u64;
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            let metadata = entry.path().symlink_metadata()?;
            if metadata.file_type().is_symlink() {
                return Err(MoeError::InvalidConfig(
                    "packed checkpoints must not contain symbolic links".to_string(),
                ));
            }
            if metadata.is_dir() {
                pending.push(entry.path());
            } else if metadata.is_file() {
                total = total.checked_add(metadata.len()).ok_or_else(|| {
                    MoeError::InvalidConfig("packed checkpoint size overflowed u64".to_string())
                })?;
            }
        }
    }
    Ok(total)
}

pub(super) fn model_load_error(error: MoeError) -> PowerError {
    match error {
        MoeError::Power(error) => error,
        other => PowerError::InferenceFailed(other.to_string()),
    }
}

use std::path::PathBuf;
use std::sync::Arc;

use a3s_power::inference::{EmbeddedRuntime, ResidencyPolicy};

use crate::continuous::ContinuousStreamingModel;
use crate::olmoe::OlmoeStreamingModel;
use crate::qwen3_5_moe::{Qwen36MoePackedCheckpoint, Qwen36MoeStreamingModel};
use crate::qwen3_moe::{Qwen3MoePackedCheckpoint, Qwen3MoeStreamingModel};
use crate::Result;

use super::config::OlmoeDeviceSelection;
use super::load::{directory_size, LoadedArtifacts};
use super::source::CheckpointSource;

/// Model-family hooks used by the architecture-neutral Power backend.
#[doc(hidden)]
pub trait ServiceArchitecture: Send + Sync + 'static {
    type Model: ContinuousStreamingModel;

    const BACKEND_NAME: &'static str;
    const FAMILY: &'static str;
    const DISPLAY_NAME: &'static str;

    fn load_plain(
        path: PathBuf,
        runtime: EmbeddedRuntime,
        residency: ResidencyPolicy,
        device: OlmoeDeviceSelection,
    ) -> Result<LoadedArtifacts<Self::Model>>;
}

#[doc(hidden)]
pub struct OlmoeServiceArchitecture;

impl ServiceArchitecture for OlmoeServiceArchitecture {
    type Model = OlmoeStreamingModel;

    const BACKEND_NAME: &'static str = "a3s-moe-olmoe";
    const FAMILY: &'static str = "olmoe";
    const DISPLAY_NAME: &'static str = "OLMoE";

    fn load_plain(
        path: PathBuf,
        runtime: EmbeddedRuntime,
        residency: ResidencyPolicy,
        device: OlmoeDeviceSelection,
    ) -> Result<LoadedArtifacts<Self::Model>> {
        load_olmoe_source(CheckpointSource::Plain(path), runtime, residency, device)
    }
}

pub(super) fn load_olmoe_source(
    source: CheckpointSource,
    runtime: EmbeddedRuntime,
    residency: ResidencyPolicy,
    device: OlmoeDeviceSelection,
) -> Result<LoadedArtifacts<OlmoeStreamingModel>> {
    let checkpoint = source.open_olmoe(runtime)?;
    let tokenizer = Arc::new(checkpoint.load_tokenizer()?);
    let binding = checkpoint.execution_batch_binding()?;
    let size = directory_size(checkpoint.root())?;
    let canonical_path = checkpoint.root().to_path_buf();
    let weights_sha256 = checkpoint.manifest().weights_sha256();
    let scalar_type = checkpoint.manifest().scalar_type;
    let context_length = checkpoint.config().max_position_embeddings;
    let hidden_size = checkpoint.config().hidden_size;
    let eos_token_ids = checkpoint.config().eos_token_id.into_iter().collect();
    let model = Arc::new(checkpoint.load_streaming(residency)?);
    Ok(LoadedArtifacts {
        canonical_path,
        tokenizer,
        model,
        binding,
        size,
        device,
        weights_sha256,
        scalar_type,
        context_length,
        hidden_size,
        eos_token_ids,
    })
}

#[doc(hidden)]
pub struct Qwen3MoeServiceArchitecture;

impl ServiceArchitecture for Qwen3MoeServiceArchitecture {
    type Model = Qwen3MoeStreamingModel;

    const BACKEND_NAME: &'static str = "a3s-moe-qwen3-moe";
    const FAMILY: &'static str = "qwen3_moe";
    const DISPLAY_NAME: &'static str = "Qwen3-MoE";

    fn load_plain(
        path: PathBuf,
        runtime: EmbeddedRuntime,
        residency: ResidencyPolicy,
        device: OlmoeDeviceSelection,
    ) -> Result<LoadedArtifacts<Self::Model>> {
        let checkpoint = Qwen3MoePackedCheckpoint::open(path, runtime)?;
        let tokenizer = Arc::new(checkpoint.load_tokenizer()?);
        let binding = checkpoint.execution_batch_binding()?;
        let size = directory_size(checkpoint.root())?;
        let canonical_path = checkpoint.root().to_path_buf();
        let weights_sha256 = checkpoint.manifest().weights_sha256();
        let scalar_type = checkpoint.manifest().scalar_type;
        let context_length = checkpoint.config().max_position_embeddings;
        let hidden_size = checkpoint.config().hidden_size;
        let eos_token_ids = checkpoint
            .config()
            .eos_token_id
            .as_ref()
            .map(|tokens| tokens.values().to_vec())
            .unwrap_or_default();
        let model = Arc::new(checkpoint.load_streaming(residency)?);
        Ok(LoadedArtifacts {
            canonical_path,
            tokenizer,
            model,
            binding,
            size,
            device,
            weights_sha256,
            scalar_type,
            context_length,
            hidden_size,
            eos_token_ids,
        })
    }
}

#[doc(hidden)]
pub struct Qwen36MoeServiceArchitecture;

impl ServiceArchitecture for Qwen36MoeServiceArchitecture {
    type Model = Qwen36MoeStreamingModel;

    const BACKEND_NAME: &'static str = "a3s-moe-qwen3.6-moe";
    const FAMILY: &'static str = "qwen3_5_moe";
    const DISPLAY_NAME: &'static str = "Qwen3.6-35B-A3B";

    fn load_plain(
        path: PathBuf,
        runtime: EmbeddedRuntime,
        residency: ResidencyPolicy,
        device: OlmoeDeviceSelection,
    ) -> Result<LoadedArtifacts<Self::Model>> {
        let checkpoint = Qwen36MoePackedCheckpoint::open(path, runtime)?;
        let tokenizer = Arc::new(checkpoint.load_tokenizer()?);
        let binding = checkpoint.execution_batch_binding()?;
        let size = directory_size(checkpoint.root())?;
        let canonical_path = checkpoint.root().to_path_buf();
        let weights_sha256 = checkpoint.manifest().weights_sha256();
        let scalar_type = checkpoint.manifest().scalar_type;
        let context_length = checkpoint.config().max_position_embeddings;
        let hidden_size = checkpoint.config().hidden_size;
        let eos_token_ids = checkpoint
            .config()
            .eos_token_id
            .as_ref()
            .map(|tokens| tokens.values().to_vec())
            .unwrap_or_default();
        let model = Arc::new(checkpoint.load_streaming(residency)?);
        Ok(LoadedArtifacts {
            canonical_path,
            tokenizer,
            model,
            binding,
            size,
            device,
            weights_sha256,
            scalar_type,
            context_length,
            hidden_size,
            eos_token_ids,
        })
    }
}

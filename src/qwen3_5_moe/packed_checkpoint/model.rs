use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use a3s_power::inference::{
    EmbeddedRuntime, ExecutionBatchBinding, ExecutionDigest, ResidencyPolicy, WeightHierarchy,
    WeightStore,
};
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;

use crate::checkpoint::{enforce_metadata_limit, validate_shard_name};
use crate::packing::f32_materialized_bytes;
use crate::{packed_expert_tensor_name, MoeError, MoeTokenizer, PackedExpertRecord, Result};

use super::{
    Qwen36MoePackedManifest, CONFIG_FILE, DENSE_DIRECTORY, EXPERT_DIRECTORY, MANIFEST_FILE,
    TOKENIZER_FILE,
};
use crate::qwen3_5_moe::checkpoint::dense_tensor_names;
use crate::qwen3_5_moe::{Qwen36MoeConfig, Qwen36MoeStreamingModel};

const MAX_CONFIG_BYTES: u64 = 1_048_576;
const MAX_MANIFEST_BYTES: u64 = 4 * 1_048_576;
const MAX_TOKENIZER_BYTES: u64 = 64 * 1_048_576;

/// Validated text-only Qwen3.6 checkpoint bound to one Power runtime.
#[derive(Clone)]
pub struct Qwen36MoePackedCheckpoint {
    root: PathBuf,
    config: Qwen36MoeConfig,
    manifest: Qwen36MoePackedManifest,
    dense_store: Arc<WeightStore>,
    expert_store: Arc<WeightStore>,
    runtime: EmbeddedRuntime,
}

impl Qwen36MoePackedCheckpoint {
    pub fn open(root: impl AsRef<Path>, runtime: EmbeddedRuntime) -> Result<Self> {
        let (root, config, manifest) = read_metadata(root.as_ref())?;
        let dense_store = Arc::new(WeightStore::open(
            root.join(DENSE_DIRECTORY),
            runtime.limits(),
        )?);
        let expert_store = Arc::new(WeightStore::open(
            root.join(EXPERT_DIRECTORY),
            runtime.limits(),
        )?);
        dense_store.verify_integrity(
            "packed Qwen3.6 dense weights",
            &manifest.dense_weights_sha256,
        )?;
        expert_store.verify_integrity(
            "packed Qwen3.6 expert weights",
            &manifest.expert_weights_sha256,
        )?;
        validate_file_inventory(&dense_store, &manifest.dense_files, "dense")?;
        validate_file_inventory(&expert_store, &manifest.expert_files, "expert")?;
        validate_dense_inventory(&dense_store, &config)?;
        validate_expert_inventory(&expert_store, &config, &manifest)?;
        Ok(Self {
            root,
            config,
            manifest,
            dense_store,
            expert_store,
            runtime,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn config(&self) -> &Qwen36MoeConfig {
        &self.config
    }

    pub fn manifest(&self) -> &Qwen36MoePackedManifest {
        &self.manifest
    }

    pub fn runtime(&self) -> &EmbeddedRuntime {
        &self.runtime
    }

    pub fn execution_batch_binding(&self) -> Result<ExecutionBatchBinding> {
        let state_layout = ExecutionDigest::utf8_text(&format!(
            "a3s-moe-qwen3.6-mixed-cache-f32-v1\0{}\0{}\0{}\0{}\0{}\0{}\0{}\0{}\0{}",
            self.config.layer_types.len(),
            self.config.num_key_value_heads,
            self.config.head_dim,
            self.config.linear_conv_kernel_dim,
            self.config.linear_num_value_heads,
            self.config.linear_key_head_dim,
            self.config.linear_value_head_dim,
            self.config.max_position_embeddings,
            self.config.full_attention_interval,
        ));
        let scheduler =
            ExecutionDigest::utf8_text("a3s-moe-qwen3.6-ragged-expert-union-greedy-scheduler-v1");
        Ok(ExecutionBatchBinding::new(
            self.manifest.weights_sha256(),
            state_layout.sha256,
            scheduler.sha256,
        )?)
    }

    pub fn load_tokenizer(&self) -> Result<MoeTokenizer> {
        if !self.manifest.tokenizer_included {
            return Err(MoeError::InvalidConfig(
                "packed checkpoint does not include tokenizer.json".to_string(),
            ));
        }
        MoeTokenizer::from_file(self.root.join(TOKENIZER_FILE), self.config.vocab_size)
    }

    pub fn load_streaming(&self, policy: ResidencyPolicy) -> Result<Qwen36MoeStreamingModel> {
        if !self.runtime.device().tensor_device().is_cpu() {
            return Err(MoeError::InvalidConfig(
                "Qwen3.6 streaming currently requires a CPU Power runtime".to_string(),
            ));
        }
        let fixed_weight_bytes = f32_materialized_bytes(&self.dense_store)?;
        let hierarchy = WeightHierarchy::new_with_fixed_weight_bytes(
            Arc::clone(&self.expert_store),
            self.runtime.clone(),
            policy,
            fixed_weight_bytes,
        )?;
        let mut tensors =
            HashMap::<String, Tensor>::with_capacity(self.dense_store.inventory().len());
        for descriptor in self.dense_store.inventory() {
            let tensor = self
                .dense_store
                .load_tensor(&descriptor.name, self.runtime.device())?
                .to_dtype(DType::F32)?;
            tensors.insert(descriptor.name.clone(), tensor);
        }
        let builder =
            VarBuilder::from_tensors(tensors, DType::F32, self.runtime.device().tensor_device());
        Qwen36MoeStreamingModel::load(self.config.clone(), builder, hierarchy)
    }
}

impl std::fmt::Debug for Qwen36MoePackedCheckpoint {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Qwen36MoePackedCheckpoint")
            .field("root", &self.root)
            .field("config", &self.config)
            .field("manifest", &self.manifest)
            .field("runtime", &self.runtime)
            .finish_non_exhaustive()
    }
}

fn read_metadata(root: &Path) -> Result<(PathBuf, Qwen36MoeConfig, Qwen36MoePackedManifest)> {
    let root = root.canonicalize()?;
    if !root.is_dir() {
        return Err(MoeError::InvalidConfig(format!(
            "packed checkpoint root '{}' is not a directory",
            root.display()
        )));
    }
    let config_path = root.join(CONFIG_FILE);
    enforce_metadata_limit(&config_path, MAX_CONFIG_BYTES, "packed Qwen3.6 config")?;
    let config = Qwen36MoeConfig::from_json_path(&config_path)?;
    let manifest_path = root.join(MANIFEST_FILE);
    enforce_metadata_limit(
        &manifest_path,
        MAX_MANIFEST_BYTES,
        "packed Qwen3.6 manifest",
    )?;
    let manifest: Qwen36MoePackedManifest =
        serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
    validate_manifest(&manifest, &config, &root)?;
    Ok((root, config, manifest))
}

fn validate_manifest(
    manifest: &Qwen36MoePackedManifest,
    config: &Qwen36MoeConfig,
    root: &Path,
) -> Result<()> {
    if manifest.schema != Qwen36MoePackedManifest::SCHEMA
        || manifest.capabilities != [Qwen36MoePackedManifest::TEXT_GENERATION_CAPABILITY]
    {
        return Err(MoeError::InvalidConfig(
            "packed Qwen3.6 manifest must declare exactly the text-generation capability"
                .to_string(),
        ));
    }
    for (label, digest) in [
        ("source", &manifest.source_weights_sha256),
        ("dense", &manifest.dense_weights_sha256),
        ("expert", &manifest.expert_weights_sha256),
    ] {
        if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(MoeError::InvalidConfig(format!(
                "packed manifest {label} digest must be 64 hexadecimal characters"
            )));
        }
    }
    if manifest.hidden_size != config.hidden_size
        || manifest.moe_intermediate_size != config.moe_intermediate_size
        || manifest.num_hidden_layers != config.num_hidden_layers
        || manifest.num_experts != config.num_experts
    {
        return Err(MoeError::InvalidConfig(
            "packed manifest geometry does not match config.json".to_string(),
        ));
    }
    if manifest.experts_per_file == 0 || manifest.experts_per_file > config.num_experts {
        return Err(MoeError::InvalidConfig(format!(
            "packed manifest experts_per_file must be within 1..={}",
            config.num_experts
        )));
    }
    validate_manifest_files(&manifest.dense_files, "dense")?;
    validate_manifest_files(&manifest.expert_files, "expert")?;
    let expected_dense_files = dense_tensor_names(config).len();
    if manifest.dense_files.len() != expected_dense_files {
        return Err(MoeError::InvalidConfig(format!(
            "packed manifest declares {} dense files, expected {expected_dense_files}",
            manifest.dense_files.len()
        )));
    }
    let expected_expert_files = config
        .num_hidden_layers
        .checked_mul(config.num_experts.div_ceil(manifest.experts_per_file))
        .ok_or_else(|| MoeError::InvalidConfig("expert file count overflowed".to_string()))?;
    if manifest.expert_files.len() != expected_expert_files {
        return Err(MoeError::InvalidConfig(format!(
            "packed manifest declares {} expert files, expected {expected_expert_files}",
            manifest.expert_files.len()
        )));
    }
    let tokenizer_path = root.join(TOKENIZER_FILE);
    if manifest.tokenizer_included {
        enforce_metadata_limit(&tokenizer_path, MAX_TOKENIZER_BYTES, "packed tokenizer")?;
    } else if tokenizer_path.exists() {
        return Err(MoeError::InvalidConfig(
            "packed tokenizer exists but the manifest does not declare it".to_string(),
        ));
    }
    Ok(())
}

fn validate_manifest_files(files: &[String], label: &str) -> Result<()> {
    if files.is_empty() {
        return Err(MoeError::InvalidConfig(format!(
            "packed manifest {label} file list must not be empty"
        )));
    }
    let mut unique = BTreeSet::new();
    for file in files {
        validate_shard_name(file)?;
        if !unique.insert(file) {
            return Err(MoeError::InvalidConfig(format!(
                "packed manifest repeats {label} file '{file}'"
            )));
        }
    }
    Ok(())
}

fn validate_file_inventory(store: &WeightStore, expected: &[String], label: &str) -> Result<()> {
    let actual = store
        .files()
        .iter()
        .map(|file| file.relative_path.as_str())
        .collect::<Vec<_>>();
    let expected = expected.iter().map(String::as_str).collect::<Vec<_>>();
    if actual != expected {
        return Err(MoeError::InvalidConfig(format!(
            "packed {label} file inventory does not match the manifest"
        )));
    }
    Ok(())
}

fn validate_dense_inventory(store: &WeightStore, config: &Qwen36MoeConfig) -> Result<()> {
    let expected = dense_tensor_names(config)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let actual = store
        .inventory()
        .map(|descriptor| descriptor.name.clone())
        .collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(MoeError::InvalidConfig(format!(
            "packed dense inventory contains {} tensors, expected {}",
            actual.len(),
            expected.len()
        )));
    }
    Ok(())
}

fn validate_expert_inventory(
    store: &WeightStore,
    config: &Qwen36MoeConfig,
    manifest: &Qwen36MoePackedManifest,
) -> Result<()> {
    let section_bytes = config
        .hidden_size
        .checked_mul(config.moe_intermediate_size)
        .and_then(|elements| elements.checked_mul(manifest.scalar_type.byte_width()))
        .ok_or_else(|| MoeError::InvalidConfig("packed expert size overflowed".to_string()))?;
    let record_bytes = section_bytes
        .checked_mul(3)
        .and_then(|bytes| bytes.checked_add(PackedExpertRecord::HEADER_BYTES))
        .ok_or_else(|| MoeError::InvalidConfig("packed expert record overflowed".to_string()))?;
    let expected_count = config
        .num_hidden_layers
        .checked_mul(config.num_experts)
        .ok_or_else(|| MoeError::InvalidConfig("packed expert count overflowed".to_string()))?;
    if store.inventory().len() != expected_count {
        return Err(MoeError::InvalidConfig(format!(
            "packed expert inventory contains {} tensors, expected {expected_count}",
            store.inventory().len()
        )));
    }
    for layer in 0..config.num_hidden_layers {
        for expert in 0..config.num_experts {
            let layer = u32::try_from(layer).map_err(|_| {
                MoeError::InvalidConfig("layer index exceeds routing contract".to_string())
            })?;
            let expert = u32::try_from(expert).map_err(|_| {
                MoeError::InvalidConfig("expert index exceeds routing contract".to_string())
            })?;
            let name = packed_expert_tensor_name(layer, expert);
            let descriptor = store.descriptor(&name).ok_or_else(|| {
                MoeError::InvalidConfig(format!("packed checkpoint is missing expert '{name}'"))
            })?;
            if descriptor.dtype != "u8" || descriptor.shape != [record_bytes] {
                return Err(MoeError::InvalidTensor(format!(
                    "packed expert '{name}' must be U8[{record_bytes}], found {}{:?}",
                    descriptor.dtype, descriptor.shape
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_checkpoint_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Qwen36MoePackedCheckpoint>();
    }
}

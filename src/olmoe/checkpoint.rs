use std::collections::{BTreeSet, HashMap};
use std::path::{Component, Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

use crate::{MoeError, Result};

use super::{OlmoeConfig, OlmoeCpuModel, OlmoeTokenizer};

const CONFIG_FILE: &str = "config.json";
const INDEX_FILE: &str = "model.safetensors.index.json";
const TOKENIZER_FILE: &str = "tokenizer.json";
const MAX_CONFIG_BYTES: u64 = 1_048_576;
const MAX_INDEX_BYTES: u64 = 16 * 1_048_576;
const MAX_TOKENIZER_BYTES: u64 = 64 * 1_048_576;

#[derive(Debug, Deserialize)]
struct SafeTensorIndex {
    weight_map: HashMap<String, String>,
}

/// Validated Hugging Face OLMoE checkpoint directory.
#[derive(Debug, Clone)]
pub struct OlmoeCheckpoint {
    root: PathBuf,
    config: OlmoeConfig,
    shards: Vec<PathBuf>,
    weight_map: HashMap<String, String>,
}

impl OlmoeCheckpoint {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().canonicalize()?;
        if !root.is_dir() {
            return Err(MoeError::InvalidConfig(format!(
                "checkpoint root '{}' is not a directory",
                root.display()
            )));
        }
        let config_path = root.join(CONFIG_FILE);
        enforce_metadata_limit(&config_path, MAX_CONFIG_BYTES, "OLMoE config")?;
        let config = OlmoeConfig::from_json_path(&config_path)?;

        let index_path = root.join(INDEX_FILE);
        enforce_metadata_limit(&index_path, MAX_INDEX_BYTES, "SafeTensor index")?;
        let index: SafeTensorIndex = serde_json::from_slice(&std::fs::read(&index_path)?)?;
        if index.weight_map.is_empty() {
            return Err(MoeError::InvalidConfig(
                "SafeTensor index weight_map must not be empty".to_string(),
            ));
        }
        let required_names = required_tensor_names(&config);
        for required in &required_names {
            if !index.weight_map.contains_key(required) {
                return Err(MoeError::InvalidConfig(format!(
                    "SafeTensor index is missing required tensor '{required}'"
                )));
            }
        }
        if index.weight_map.len() != required_names.len() {
            let required = required_names.into_iter().collect::<BTreeSet<_>>();
            let unexpected = index
                .weight_map
                .keys()
                .filter(|name| !required.contains(*name))
                .take(5)
                .cloned()
                .collect::<Vec<_>>();
            return Err(MoeError::InvalidConfig(format!(
                "SafeTensor index contains {} tensors, expected {}; unexpected examples: {unexpected:?}",
                index.weight_map.len(),
                required.len()
            )));
        }

        let mut shard_names = BTreeSet::new();
        for shard in index.weight_map.values() {
            validate_shard_name(shard)?;
            shard_names.insert(shard.clone());
        }
        let mut shards = Vec::with_capacity(shard_names.len());
        for shard in shard_names {
            let path = root.join(&shard);
            let metadata = path.symlink_metadata().map_err(|error| {
                MoeError::InvalidConfig(format!(
                    "failed to inspect SafeTensor shard '{}': {error}",
                    path.display()
                ))
            })?;
            if !metadata.is_file() || metadata.file_type().is_symlink() {
                return Err(MoeError::InvalidConfig(format!(
                    "SafeTensor shard '{}' must be a regular non-symlink file",
                    path.display()
                )));
            }
            shards.push(path);
        }
        Ok(Self {
            root,
            config,
            shards,
            weight_map: index.weight_map,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn config(&self) -> &OlmoeConfig {
        &self.config
    }

    pub fn shards(&self) -> &[PathBuf] {
        &self.shards
    }

    pub fn load_tokenizer(&self) -> Result<OlmoeTokenizer> {
        let path = self.root.join(TOKENIZER_FILE);
        enforce_metadata_limit(&path, MAX_TOKENIZER_BYTES, "tokenizer")?;
        OlmoeTokenizer::from_file(path, self.config.vocab_size)
    }

    /// Load every published tensor into CPU memory and construct the F32
    /// correctness backend.
    ///
    /// This deliberately resident baseline is used for numerical validation.
    /// Bounded-memory serving uses the Power residency path instead.
    pub fn load_cpu_resident(&self) -> Result<OlmoeCpuModel> {
        let mut tensors = HashMap::<String, Tensor>::new();
        for shard in &self.shards {
            let shard_name = shard
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| {
                    MoeError::InvalidConfig(format!(
                        "SafeTensor shard '{}' has a non-UTF-8 filename",
                        shard.display()
                    ))
                })?;
            for (name, tensor) in candle_core::safetensors::load(shard, &Device::Cpu)? {
                if self.weight_map.get(&name).map(String::as_str) != Some(shard_name) {
                    return Err(MoeError::InvalidConfig(format!(
                        "tensor '{name}' is not stored in its index-declared shard"
                    )));
                }
                if tensors.insert(name.clone(), tensor).is_some() {
                    return Err(MoeError::InvalidConfig(format!(
                        "tensor '{name}' occurs in more than one SafeTensor shard"
                    )));
                }
            }
        }
        if tensors.len() != self.weight_map.len() {
            return Err(MoeError::InvalidConfig(format!(
                "loaded {} tensors but the index declares {}",
                tensors.len(),
                self.weight_map.len()
            )));
        }
        let builder = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        OlmoeCpuModel::load(self.config.clone(), builder)
    }
}

pub(super) fn enforce_metadata_limit(path: &Path, limit: u64, label: &str) -> Result<()> {
    let metadata = path.symlink_metadata()?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > limit {
        return Err(MoeError::InvalidConfig(format!(
            "{label} '{}' must be a regular non-symlink file no larger than {limit} bytes",
            path.display()
        )));
    }
    Ok(())
}

pub(super) fn validate_shard_name(name: &str) -> Result<()> {
    let path = Path::new(name);
    let mut components = path.components();
    let single_file = matches!(components.next(), Some(Component::Normal(_)))
        && components.next().is_none()
        && path.extension().and_then(|extension| extension.to_str()) == Some("safetensors");
    if !single_file {
        return Err(MoeError::InvalidConfig(format!(
            "SafeTensor shard name '{name}' must be one relative .safetensors filename"
        )));
    }
    Ok(())
}

pub(super) fn required_tensor_names(config: &OlmoeConfig) -> Vec<String> {
    let global_count = if config.tie_word_embeddings { 2 } else { 3 };
    let attention_biases = if config.attention_bias { 4 } else { 0 };
    let layer_tensor_count = 9 + attention_biases + config.num_experts * 3;
    let mut names =
        Vec::with_capacity(global_count + config.num_hidden_layers * layer_tensor_count);
    names.push("model.embed_tokens.weight".to_string());
    names.push("model.norm.weight".to_string());
    if !config.tie_word_embeddings {
        names.push("lm_head.weight".to_string());
    }
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        names.push(format!("{prefix}.input_layernorm.weight"));
        names.push(format!("{prefix}.post_attention_layernorm.weight"));
        names.push(format!("{prefix}.self_attn.q_proj.weight"));
        names.push(format!("{prefix}.self_attn.k_proj.weight"));
        names.push(format!("{prefix}.self_attn.v_proj.weight"));
        names.push(format!("{prefix}.self_attn.o_proj.weight"));
        if config.attention_bias {
            names.push(format!("{prefix}.self_attn.q_proj.bias"));
            names.push(format!("{prefix}.self_attn.k_proj.bias"));
            names.push(format!("{prefix}.self_attn.v_proj.bias"));
            names.push(format!("{prefix}.self_attn.o_proj.bias"));
        }
        names.push(format!("{prefix}.self_attn.q_norm.weight"));
        names.push(format!("{prefix}.self_attn.k_norm.weight"));
        names.push(format!("{prefix}.mlp.gate.weight"));
        for expert in 0..config.num_experts {
            let expert_prefix = format!("{prefix}.mlp.experts.{expert}");
            names.push(format!("{expert_prefix}.gate_proj.weight"));
            names.push(format!("{expert_prefix}.up_proj.weight"));
            names.push(format!("{expert_prefix}.down_proj.weight"));
        }
    }
    names
}

pub(super) fn dense_tensor_names(config: &OlmoeConfig) -> Vec<String> {
    required_tensor_names(config)
        .into_iter()
        .filter(|name| !name.contains(".mlp.experts."))
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn tiny_config() -> OlmoeConfig {
        OlmoeConfig {
            model_type: "olmoe".to_string(),
            vocab_size: 8,
            hidden_size: 4,
            intermediate_size: 3,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            num_experts: 2,
            num_experts_per_tok: 1,
            max_position_embeddings: 8,
            norm_topk_prob: false,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            attention_bias: false,
            clip_qkv: None,
            tie_word_embeddings: false,
            eos_token_id: Some(7),
            pad_token_id: Some(0),
        }
    }

    fn write_checkpoint(
        directory: &Path,
        config: &OlmoeConfig,
        mutate: impl FnOnce(&mut HashMap<String, String>),
    ) {
        std::fs::write(
            directory.join(CONFIG_FILE),
            serde_json::to_vec(config).unwrap(),
        )
        .unwrap();
        let shard = "model-00001-of-00001.safetensors";
        std::fs::write(directory.join(shard), []).unwrap();
        let mut weight_map = required_tensor_names(config)
            .into_iter()
            .map(|name| (name, shard.to_string()))
            .collect::<HashMap<_, _>>();
        mutate(&mut weight_map);
        std::fs::write(
            directory.join(INDEX_FILE),
            serde_json::to_vec(&json!({ "weight_map": weight_map })).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn validates_the_complete_tensor_contract_before_loading_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let config = tiny_config();
        write_checkpoint(directory.path(), &config, |_| {});

        let checkpoint = OlmoeCheckpoint::open(directory.path()).unwrap();
        assert_eq!(checkpoint.config(), &config);
        assert_eq!(checkpoint.shards().len(), 1);
        assert_eq!(required_tensor_names(&config).len(), 18);
    }

    #[test]
    fn rejects_missing_tensors_and_shard_path_traversal() {
        let directory = tempfile::tempdir().unwrap();
        let config = tiny_config();
        write_checkpoint(directory.path(), &config, |weights| {
            weights.remove("model.layers.0.mlp.gate.weight");
        });
        assert!(OlmoeCheckpoint::open(directory.path()).is_err());

        write_checkpoint(directory.path(), &config, |weights| {
            *weights.get_mut("model.layers.0.mlp.gate.weight").unwrap() =
                "../outside.safetensors".to_string();
        });
        assert!(OlmoeCheckpoint::open(directory.path()).is_err());

        write_checkpoint(directory.path(), &config, |weights| {
            weights.insert(
                "unexpected.weight".to_string(),
                "model-00001-of-00001.safetensors".to_string(),
            );
        });
        assert!(OlmoeCheckpoint::open(directory.path()).is_err());
    }

    #[test]
    fn published_shape_requires_exactly_3219_tensor_names() {
        let mut config = tiny_config();
        config.vocab_size = 50_304;
        config.hidden_size = 2_048;
        config.intermediate_size = 1_024;
        config.num_hidden_layers = 16;
        config.num_attention_heads = 16;
        config.num_key_value_heads = 16;
        config.num_experts = 64;
        config.num_experts_per_tok = 8;
        config.max_position_embeddings = 4_096;
        assert_eq!(required_tensor_names(&config).len(), 3_219);
    }
}

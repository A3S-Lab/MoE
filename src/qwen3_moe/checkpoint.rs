use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::checkpoint::{
    enforce_metadata_limit, ShardedSafeTensors, CONFIG_FILE, MAX_CONFIG_BYTES, MAX_TOKENIZER_BYTES,
    TOKENIZER_FILE,
};
use crate::{MoeError, Result};

use super::{Qwen3MoeConfig, Qwen3MoeCpuModel, Qwen3MoeTokenizer};

/// Validated Hugging Face Qwen3-MoE checkpoint directory.
#[derive(Debug, Clone)]
pub struct Qwen3MoeCheckpoint {
    config: Qwen3MoeConfig,
    weights: ShardedSafeTensors,
}

impl Qwen3MoeCheckpoint {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().canonicalize()?;
        if !root.is_dir() {
            return Err(MoeError::InvalidConfig(format!(
                "checkpoint root '{}' is not a directory",
                root.display()
            )));
        }
        let config_path = root.join(CONFIG_FILE);
        enforce_metadata_limit(&config_path, MAX_CONFIG_BYTES, "Qwen3-MoE config")?;
        let config = Qwen3MoeConfig::from_json_path(config_path)?;
        let weights = ShardedSafeTensors::open(&root, required_tensor_names(&config))?;
        Ok(Self { config, weights })
    }

    pub fn root(&self) -> &Path {
        self.weights.root()
    }

    pub fn config(&self) -> &Qwen3MoeConfig {
        &self.config
    }

    pub fn shards(&self) -> &[PathBuf] {
        self.weights.shards()
    }

    pub fn load_tokenizer(&self) -> Result<Qwen3MoeTokenizer> {
        let path = self.root().join(TOKENIZER_FILE);
        enforce_metadata_limit(&path, MAX_TOKENIZER_BYTES, "Qwen3-MoE tokenizer")?;
        Qwen3MoeTokenizer::from_file(path, self.config.vocab_size)
    }

    /// Load the complete checkpoint into the F32 CPU correctness backend.
    pub fn load_cpu_resident(&self) -> Result<Qwen3MoeCpuModel> {
        let tensors = self.weights.load_cpu()?;
        let builder = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        Qwen3MoeCpuModel::load(self.config.clone(), builder)
    }
}

pub(super) fn required_tensor_names(config: &Qwen3MoeConfig) -> Vec<String> {
    let global_count = if config.tie_word_embeddings { 2 } else { 3 };
    let attention_biases = if config.attention_bias { 4 } else { 0 };
    let mut names = Vec::with_capacity(
        global_count
            + config
                .num_hidden_layers
                .saturating_mul(11 + attention_biases),
    );
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
        if config.is_sparse_layer(layer) {
            names.push(format!("{prefix}.mlp.gate.weight"));
            names.push(format!("{prefix}.mlp.experts.gate_up_proj"));
            names.push(format!("{prefix}.mlp.experts.down_proj"));
        } else {
            names.push(format!("{prefix}.mlp.gate_proj.weight"));
            names.push(format!("{prefix}.mlp.up_proj.weight"));
            names.push(format!("{prefix}.mlp.down_proj.weight"));
        }
    }
    names
}

pub(super) fn dense_tensor_names(config: &Qwen3MoeConfig) -> Vec<String> {
    required_tensor_names(config)
        .into_iter()
        .filter(|name| !name.contains(".mlp.experts."))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;

    use crate::checkpoint::INDEX_FILE;
    use crate::qwen3_moe::Qwen3MoeTokenIds;

    use super::*;

    fn tiny_config() -> Qwen3MoeConfig {
        Qwen3MoeConfig {
            model_type: "qwen3_moe".to_string(),
            vocab_size: 16,
            hidden_size: 4,
            intermediate_size: 5,
            moe_intermediate_size: 3,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: Some(4),
            num_experts: 3,
            num_experts_per_tok: 2,
            max_position_embeddings: 8,
            norm_topk_prob: true,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            attention_bias: false,
            attention_dropout: 0.0,
            decoder_sparse_step: 2,
            mlp_only_layers: Vec::new(),
            use_sliding_window: false,
            sliding_window: None,
            tie_word_embeddings: false,
            bos_token_id: Some(1),
            eos_token_id: Some(Qwen3MoeTokenIds::One(15)),
            pad_token_id: Some(0),
        }
    }

    fn write_checkpoint(
        directory: &Path,
        config: &Qwen3MoeConfig,
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
    fn validates_mixed_dense_and_sparse_tensor_inventory() {
        let directory = tempfile::tempdir().unwrap();
        let config = tiny_config();
        write_checkpoint(directory.path(), &config, |_| {});

        let checkpoint = Qwen3MoeCheckpoint::open(directory.path()).unwrap();
        assert_eq!(checkpoint.config(), &config);
        assert_eq!(checkpoint.shards().len(), 1);
        let names = required_tensor_names(&config);
        assert_eq!(names.len(), 25);
        assert!(names.contains(&"model.layers.0.mlp.gate_proj.weight".to_string()));
        assert!(names.contains(&"model.layers.1.mlp.experts.gate_up_proj".to_string()));
    }

    #[test]
    fn rejects_missing_tensors_and_shard_path_traversal() {
        let directory = tempfile::tempdir().unwrap();
        let config = tiny_config();
        write_checkpoint(directory.path(), &config, |weights| {
            weights.remove("model.layers.1.mlp.gate.weight");
        });
        assert!(Qwen3MoeCheckpoint::open(directory.path()).is_err());

        write_checkpoint(directory.path(), &config, |weights| {
            *weights.get_mut("model.layers.1.mlp.gate.weight").unwrap() =
                "../outside.safetensors".to_string();
        });
        assert!(Qwen3MoeCheckpoint::open(directory.path()).is_err());
    }

    #[test]
    fn published_qwen3_30b_a3b_inventory_has_531_tensors() {
        let mut config = tiny_config();
        config.vocab_size = 151_936;
        config.hidden_size = 2_048;
        config.intermediate_size = 6_144;
        config.moe_intermediate_size = 768;
        config.num_hidden_layers = 48;
        config.num_attention_heads = 32;
        config.num_key_value_heads = 4;
        config.head_dim = Some(128);
        config.num_experts = 128;
        config.num_experts_per_tok = 8;
        config.max_position_embeddings = 32_768;
        config.decoder_sparse_step = 1;
        assert_eq!(required_tensor_names(&config).len(), 531);
    }
}

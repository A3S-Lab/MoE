use std::path::{Path, PathBuf};

use crate::checkpoint::{
    enforce_metadata_limit, ShardedSafeTensors, CONFIG_FILE, MAX_CONFIG_BYTES, MAX_TOKENIZER_BYTES,
    TOKENIZER_FILE,
};
use crate::{MoeError, MoeTokenizer, Result};

use super::{Qwen36MoeConfig, Qwen36MoeLayerType};

const IGNORED_SOURCE_PREFIXES: &[&str] = &["model.visual.", "mtp."];

/// Validated text view of a Hugging Face Qwen3.6-35B-A3B checkpoint.
///
/// The official checkpoint also contains vision and MTP tensors. Their names
/// are accepted only under the two pinned namespaces; this text implementation
/// never advertises or packs those capabilities.
#[derive(Debug, Clone)]
pub struct Qwen36MoeCheckpoint {
    config: Qwen36MoeConfig,
    weights: ShardedSafeTensors,
}

impl Qwen36MoeCheckpoint {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().canonicalize()?;
        if !root.is_dir() {
            return Err(MoeError::InvalidConfig(format!(
                "checkpoint root '{}' is not a directory",
                root.display()
            )));
        }
        let config_path = root.join(CONFIG_FILE);
        enforce_metadata_limit(&config_path, MAX_CONFIG_BYTES, "Qwen3.6-MoE config")?;
        let config = Qwen36MoeConfig::from_json_path(config_path)?;
        let weights = ShardedSafeTensors::open_with_ignored_prefixes(
            &root,
            required_text_tensor_names(&config),
            IGNORED_SOURCE_PREFIXES,
        )?;
        Ok(Self { config, weights })
    }

    pub fn root(&self) -> &Path {
        self.weights.root()
    }

    pub fn config(&self) -> &Qwen36MoeConfig {
        &self.config
    }

    pub fn shards(&self) -> &[PathBuf] {
        self.weights.shards()
    }

    pub fn load_tokenizer(&self) -> Result<MoeTokenizer> {
        let path = self.root().join(TOKENIZER_FILE);
        enforce_metadata_limit(&path, MAX_TOKENIZER_BYTES, "Qwen3.6-MoE tokenizer")?;
        MoeTokenizer::from_file(path, self.config.vocab_size)
    }
}

pub(super) fn required_text_tensor_names(config: &Qwen36MoeConfig) -> Vec<String> {
    let mut names = dense_tensor_names(config);
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("model.language_model.layers.{layer}.mlp.experts");
        names.push(format!("{prefix}.gate_up_proj"));
        names.push(format!("{prefix}.down_proj"));
    }
    names
}

pub(super) fn dense_tensor_names(config: &Qwen36MoeConfig) -> Vec<String> {
    let mut names = Vec::with_capacity(3 + config.num_hidden_layers.saturating_mul(16));
    names.push("model.language_model.embed_tokens.weight".to_string());
    names.push("model.language_model.norm.weight".to_string());
    names.push("lm_head.weight".to_string());
    for (layer, layer_type) in config.layer_types.iter().enumerate() {
        let prefix = format!("model.language_model.layers.{layer}");
        names.push(format!("{prefix}.input_layernorm.weight"));
        names.push(format!("{prefix}.post_attention_layernorm.weight"));
        match layer_type {
            Qwen36MoeLayerType::LinearAttention => {
                let prefix = format!("{prefix}.linear_attn");
                names.push(format!("{prefix}.A_log"));
                names.push(format!("{prefix}.conv1d.weight"));
                names.push(format!("{prefix}.dt_bias"));
                names.push(format!("{prefix}.in_proj_a.weight"));
                names.push(format!("{prefix}.in_proj_b.weight"));
                names.push(format!("{prefix}.in_proj_qkv.weight"));
                names.push(format!("{prefix}.in_proj_z.weight"));
                names.push(format!("{prefix}.norm.weight"));
                names.push(format!("{prefix}.out_proj.weight"));
            }
            Qwen36MoeLayerType::FullAttention => {
                let prefix = format!("{prefix}.self_attn");
                names.push(format!("{prefix}.q_proj.weight"));
                names.push(format!("{prefix}.k_proj.weight"));
                names.push(format!("{prefix}.v_proj.weight"));
                names.push(format!("{prefix}.o_proj.weight"));
                names.push(format!("{prefix}.q_norm.weight"));
                names.push(format!("{prefix}.k_norm.weight"));
            }
        }
        let prefix = format!("model.language_model.layers.{layer}.mlp");
        names.push(format!("{prefix}.gate.weight"));
        names.push(format!("{prefix}.shared_expert.gate_proj.weight"));
        names.push(format!("{prefix}.shared_expert.up_proj.weight"));
        names.push(format!("{prefix}.shared_expert.down_proj.weight"));
        names.push(format!("{prefix}.shared_expert_gate.weight"));
    }
    names
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;

    use crate::checkpoint::INDEX_FILE;

    use super::*;
    use crate::qwen3_5_moe::{Qwen36MoeRopeParameters, Qwen36MoeTextConfig, Qwen36MoeTokenIds};

    fn tiny_config() -> Qwen36MoeConfig {
        Qwen36MoeConfig {
            model_type: "qwen3_5_moe".to_string(),
            text_config: Qwen36MoeTextConfig {
                model_type: "qwen3_5_moe_text".to_string(),
                vocab_size: 16,
                hidden_size: 4,
                num_hidden_layers: 4,
                num_attention_heads: 2,
                num_key_value_heads: 1,
                head_dim: 4,
                max_position_embeddings: 8,
                full_attention_interval: 4,
                layer_types: vec![
                    Qwen36MoeLayerType::LinearAttention,
                    Qwen36MoeLayerType::LinearAttention,
                    Qwen36MoeLayerType::LinearAttention,
                    Qwen36MoeLayerType::FullAttention,
                ],
                linear_conv_kernel_dim: 2,
                linear_key_head_dim: 2,
                linear_num_key_heads: 1,
                linear_num_value_heads: 2,
                linear_value_head_dim: 2,
                mamba_ssm_dtype: "float32".to_string(),
                moe_intermediate_size: 3,
                shared_expert_intermediate_size: 3,
                num_experts: 3,
                num_experts_per_tok: 2,
                hidden_act: "silu".to_string(),
                rms_norm_eps: 1e-6,
                rope_parameters: Qwen36MoeRopeParameters {
                    rope_type: "default".to_string(),
                    rope_theta: 10_000.0,
                    partial_rotary_factor: 1.0,
                    mrope_interleaved: true,
                    mrope_section: vec![1, 1, 0],
                },
                attention_bias: false,
                attention_dropout: 0.0,
                attn_output_gate: true,
                tie_word_embeddings: false,
                bos_token_id: Some(1),
                eos_token_id: Some(Qwen36MoeTokenIds::One(15)),
                pad_token_id: Some(0),
            },
            tie_word_embeddings: false,
            image_token_id: None,
            video_token_id: None,
            vision_start_token_id: None,
            vision_end_token_id: None,
        }
    }

    fn write_checkpoint(directory: &Path, mutate: impl FnOnce(&mut HashMap<String, String>)) {
        let config = tiny_config();
        std::fs::write(
            directory.join(CONFIG_FILE),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let shard = "model-00001-of-00001.safetensors";
        std::fs::write(directory.join(shard), []).unwrap();
        let mut weight_map = required_text_tensor_names(&config)
            .into_iter()
            .map(|name| (name, shard.to_string()))
            .collect::<HashMap<_, _>>();
        weight_map.insert(
            "model.visual.patch_embed.proj.weight".to_string(),
            shard.to_string(),
        );
        weight_map.insert(
            "mtp.layers.0.input_layernorm.weight".to_string(),
            shard.to_string(),
        );
        mutate(&mut weight_map);
        std::fs::write(
            directory.join(INDEX_FILE),
            serde_json::to_vec(&json!({ "weight_map": weight_map })).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn validates_text_tensors_and_only_named_ignored_families() {
        let directory = tempfile::tempdir().unwrap();
        write_checkpoint(directory.path(), |_| {});
        let checkpoint = Qwen36MoeCheckpoint::open(directory.path()).unwrap();
        assert_eq!(checkpoint.config(), &tiny_config());
        assert_eq!(checkpoint.shards().len(), 1);
        assert_eq!(required_text_tensor_names(checkpoint.config()).len(), 72);
        assert_eq!(dense_tensor_names(checkpoint.config()).len(), 64);

        write_checkpoint(directory.path(), |weights| {
            weights.insert(
                "untrusted.weight".to_string(),
                "model-00001-of-00001.safetensors".to_string(),
            );
        });
        assert!(Qwen36MoeCheckpoint::open(directory.path()).is_err());
    }

    #[test]
    fn public_text_inventory_contains_exactly_693_tensors() {
        let mut config = tiny_config();
        config.text_config.vocab_size = 248_320;
        config.text_config.hidden_size = 2_048;
        config.text_config.num_hidden_layers = 40;
        config.text_config.num_attention_heads = 16;
        config.text_config.num_key_value_heads = 2;
        config.text_config.head_dim = 256;
        config.text_config.max_position_embeddings = 262_144;
        config.text_config.linear_conv_kernel_dim = 4;
        config.text_config.linear_key_head_dim = 128;
        config.text_config.linear_num_key_heads = 16;
        config.text_config.linear_num_value_heads = 32;
        config.text_config.linear_value_head_dim = 128;
        config.text_config.moe_intermediate_size = 512;
        config.text_config.shared_expert_intermediate_size = 512;
        config.text_config.num_experts = 256;
        config.text_config.num_experts_per_tok = 8;
        config.text_config.rope_parameters.rope_theta = 10_000_000.0;
        config.text_config.rope_parameters.partial_rotary_factor = 0.25;
        config.text_config.layer_types = (0..40)
            .map(|layer| {
                if (layer + 1) % 4 == 0 {
                    Qwen36MoeLayerType::FullAttention
                } else {
                    Qwen36MoeLayerType::LinearAttention
                }
            })
            .collect();
        config.validate().unwrap();
        assert_eq!(required_text_tensor_names(&config).len(), 693);
        assert_eq!(dense_tensor_names(&config).len(), 613);
    }
}

use std::collections::BTreeSet;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{MoeError, MoeLayerConfig, Result};

fn default_model_type() -> String {
    "qwen3_moe".to_string()
}

fn default_hidden_act() -> String {
    "silu".to_string()
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_rope_theta() -> f64 {
    1_000_000.0
}

fn default_decoder_sparse_step() -> usize {
    1
}

/// One or more token IDs as represented by Hugging Face configurations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Qwen3MoeTokenIds {
    One(u32),
    Many(Vec<u32>),
}

impl Qwen3MoeTokenIds {
    pub fn values(&self) -> &[u32] {
        match self {
            Self::One(value) => std::slice::from_ref(value),
            Self::Many(values) => values,
        }
    }
}

/// Hugging Face compatible Qwen3-MoE architecture configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen3MoeConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f64,
    #[serde(default = "default_decoder_sparse_step")]
    pub decoder_sparse_step: usize,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<Qwen3MoeTokenIds>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
}

impl Qwen3MoeConfig {
    pub fn from_json_path(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let config: Self = serde_json::from_slice(&bytes)?;
        config.validate()?;
        Ok(config)
    }

    pub fn attention_head_dim(&self) -> Result<usize> {
        if let Some(head_dim) = self.head_dim {
            return Ok(head_dim);
        }
        if self.num_attention_heads == 0 {
            return Err(MoeError::InvalidConfig(
                "num_attention_heads must be non-zero".to_string(),
            ));
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(MoeError::InvalidConfig(
                "hidden_size must be divisible by num_attention_heads when head_dim is omitted"
                    .to_string(),
            ));
        }
        Ok(self.hidden_size / self.num_attention_heads)
    }

    pub fn is_sparse_layer(&self, layer: usize) -> bool {
        self.decoder_sparse_step != 0
            && layer < self.num_hidden_layers
            && !self.mlp_only_layers.contains(&layer)
            && (layer + 1).is_multiple_of(self.decoder_sparse_step)
    }

    pub fn moe_config(&self) -> Result<MoeLayerConfig> {
        self.validate()?;
        Ok(MoeLayerConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.moe_intermediate_size,
            num_experts: self.num_experts,
            top_k: self.num_experts_per_tok,
            normalize_top_k: self.norm_topk_prob,
        })
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_type != "qwen3_moe" {
            return Err(MoeError::InvalidConfig(format!(
                "model_type must be 'qwen3_moe', found '{}'",
                self.model_type
            )));
        }
        for (name, value) in [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("moe_intermediate_size", self.moe_intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("num_experts", self.num_experts),
            ("num_experts_per_tok", self.num_experts_per_tok),
            ("max_position_embeddings", self.max_position_embeddings),
            ("decoder_sparse_step", self.decoder_sparse_step),
        ] {
            if value == 0 {
                return Err(MoeError::InvalidConfig(format!("{name} must be non-zero")));
            }
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(MoeError::InvalidConfig(
                "num_attention_heads must be divisible by num_key_value_heads".to_string(),
            ));
        }
        let head_dim = self.attention_head_dim()?;
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(MoeError::InvalidConfig(
                "attention head dimension must be even for rotary embeddings".to_string(),
            ));
        }
        self.moe_config_unchecked().validate()?;
        if self.num_hidden_layers > u32::MAX as usize {
            return Err(MoeError::InvalidConfig(
                "layer count exceeds the Power routing contract".to_string(),
            ));
        }
        if self.hidden_act != "silu" {
            return Err(MoeError::InvalidConfig(format!(
                "only Qwen3-MoE's silu activation is supported, found '{}'",
                self.hidden_act
            )));
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(MoeError::InvalidConfig(
                "rms_norm_eps must be finite and positive".to_string(),
            ));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(MoeError::InvalidConfig(
                "rope_theta must be finite and positive".to_string(),
            ));
        }
        if !self.attention_dropout.is_finite() || !(0.0..=1.0).contains(&self.attention_dropout) {
            return Err(MoeError::InvalidConfig(
                "attention_dropout must be finite and within 0..=1".to_string(),
            ));
        }
        if self.use_sliding_window && self.sliding_window.is_none_or(|window| window == 0) {
            return Err(MoeError::InvalidConfig(
                "sliding_window must be non-zero when sliding attention is enabled".to_string(),
            ));
        }

        let mut unique_dense_layers = BTreeSet::new();
        for &layer in &self.mlp_only_layers {
            if layer >= self.num_hidden_layers || !unique_dense_layers.insert(layer) {
                return Err(MoeError::InvalidConfig(
                    "mlp_only_layers must contain unique in-range layer indices".to_string(),
                ));
            }
        }
        if !(0..self.num_hidden_layers).any(|layer| self.is_sparse_layer(layer)) {
            return Err(MoeError::InvalidConfig(
                "Qwen3-MoE configuration must contain at least one sparse layer".to_string(),
            ));
        }
        for (name, token) in [
            ("bos_token_id", self.bos_token_id),
            ("pad_token_id", self.pad_token_id),
        ] {
            if token.is_some_and(|token| token as usize >= self.vocab_size) {
                return Err(MoeError::InvalidConfig(format!(
                    "{name} must be smaller than vocab_size"
                )));
            }
        }
        if let Some(eos) = &self.eos_token_id {
            let mut unique = BTreeSet::new();
            if eos.values().is_empty()
                || eos
                    .values()
                    .iter()
                    .any(|token| *token as usize >= self.vocab_size || !unique.insert(*token))
            {
                return Err(MoeError::InvalidConfig(
                    "eos_token_id values must be unique and smaller than vocab_size".to_string(),
                ));
            }
        }

        let query_elements = self
            .num_attention_heads
            .checked_mul(head_dim)
            .and_then(|rows| rows.checked_mul(self.hidden_size));
        let key_value_elements = self
            .num_key_value_heads
            .checked_mul(head_dim)
            .and_then(|rows| rows.checked_mul(self.hidden_size))
            .and_then(|one| one.checked_mul(2));
        let dense_mlp_elements = self
            .intermediate_size
            .checked_mul(self.hidden_size)
            .and_then(|one| one.checked_mul(3));
        let routed_expert_elements = self
            .moe_intermediate_size
            .checked_mul(self.hidden_size)
            .and_then(|one| one.checked_mul(3))
            .and_then(|per_expert| per_expert.checked_mul(self.num_experts));
        if query_elements
            .and_then(|query| query.checked_mul(2))
            .and_then(|query_and_output| query_and_output.checked_add(key_value_elements?))
            .and_then(|attention| attention.checked_add(dense_mlp_elements?))
            .and_then(|dense| dense.checked_add(routed_expert_elements?))
            .is_none()
        {
            return Err(MoeError::InvalidConfig(
                "declared Qwen3-MoE weight dimensions overflow addressable memory".to_string(),
            ));
        }
        Ok(())
    }

    fn moe_config_unchecked(&self) -> MoeLayerConfig {
        MoeLayerConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.moe_intermediate_size,
            num_experts: self.num_experts,
            top_k: self.num_experts_per_tok,
            normalize_top_k: self.norm_topk_prob,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn published_shape() -> Qwen3MoeConfig {
        Qwen3MoeConfig {
            model_type: "qwen3_moe".to_string(),
            vocab_size: 151_936,
            hidden_size: 2_048,
            intermediate_size: 6_144,
            moe_intermediate_size: 768,
            num_hidden_layers: 48,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            head_dim: Some(128),
            num_experts: 128,
            num_experts_per_tok: 8,
            max_position_embeddings: 32_768,
            norm_topk_prob: true,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            attention_bias: false,
            attention_dropout: 0.0,
            decoder_sparse_step: 1,
            mlp_only_layers: Vec::new(),
            use_sliding_window: false,
            sliding_window: None,
            tie_word_embeddings: false,
            bos_token_id: Some(151_643),
            eos_token_id: Some(Qwen3MoeTokenIds::One(151_643)),
            pad_token_id: None,
        }
    }

    #[test]
    fn accepts_the_published_qwen3_30b_a3b_shape() {
        let config = published_shape();
        config.validate().unwrap();
        assert_eq!(config.attention_head_dim().unwrap(), 128);
        assert_eq!(config.moe_config().unwrap().intermediate_size, 768);
        assert!((0..48).all(|layer| config.is_sparse_layer(layer)));
    }

    #[test]
    fn validates_sparse_schedule_head_geometry_and_token_ids() {
        let mut config = published_shape();
        config.decoder_sparse_step = 2;
        config.mlp_only_layers = vec![3];
        config.validate().unwrap();
        assert!(!config.is_sparse_layer(0));
        assert!(config.is_sparse_layer(1));
        assert!(!config.is_sparse_layer(3));

        config.head_dim = Some(127);
        assert!(config.validate().is_err());
        config.head_dim = Some(128);
        config.eos_token_id = Some(Qwen3MoeTokenIds::Many(vec![1, 1]));
        assert!(config.validate().is_err());
    }
}

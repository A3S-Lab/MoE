use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{MoeError, Result};

fn default_model_type() -> String {
    "olmoe".to_string()
}

fn default_hidden_act() -> String {
    "silu".to_string()
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}

fn default_rope_theta() -> f64 {
    10_000.0
}

/// Hugging Face compatible OLMoE architecture configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OlmoeConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
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
    pub clip_qkv: Option<f64>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
}

/// Validated dimensions required by one OLMoE sparse feed-forward layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OlmoeMoeConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub normalize_top_k: bool,
}

impl OlmoeConfig {
    pub fn from_json_path(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let config: Self = serde_json::from_slice(&bytes)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_type != "olmoe" {
            return Err(MoeError::InvalidConfig(format!(
                "model_type must be 'olmoe', found '{}'",
                self.model_type
            )));
        }
        for (name, value) in [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("num_experts", self.num_experts),
            ("num_experts_per_tok", self.num_experts_per_tok),
            ("max_position_embeddings", self.max_position_embeddings),
        ] {
            if value == 0 {
                return Err(MoeError::InvalidConfig(format!("{name} must be non-zero")));
            }
        }
        if self.num_experts_per_tok > self.num_experts {
            return Err(MoeError::InvalidConfig(format!(
                "num_experts_per_tok {} exceeds num_experts {}",
                self.num_experts_per_tok, self.num_experts
            )));
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(MoeError::InvalidConfig(
                "hidden_size must be divisible by num_attention_heads".to_string(),
            ));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(MoeError::InvalidConfig(
                "num_attention_heads must be divisible by num_key_value_heads".to_string(),
            ));
        }
        if self.hidden_act != "silu" {
            return Err(MoeError::InvalidConfig(format!(
                "only OLMoE's silu activation is supported, found '{}'",
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
        if self
            .clip_qkv
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(MoeError::InvalidConfig(
                "clip_qkv must be finite and positive when present".to_string(),
            ));
        }

        let expert_gate_up = self
            .intermediate_size
            .checked_mul(2)
            .and_then(|rows| rows.checked_mul(self.hidden_size));
        let expert_down = self.intermediate_size.checked_mul(self.hidden_size);
        let all_experts = expert_gate_up
            .and_then(|gate_up| gate_up.checked_add(expert_down?))
            .and_then(|per_expert| per_expert.checked_mul(self.num_experts));
        let router = self.num_experts.checked_mul(self.hidden_size);
        if all_experts
            .and_then(|value| value.checked_add(router?))
            .is_none()
        {
            return Err(MoeError::InvalidConfig(
                "declared OLMoE weight dimensions overflow addressable memory".to_string(),
            ));
        }
        Ok(())
    }

    pub fn moe_config(&self) -> Result<OlmoeMoeConfig> {
        self.validate()?;
        Ok(OlmoeMoeConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_experts: self.num_experts,
            top_k: self.num_experts_per_tok,
            normalize_top_k: self.norm_topk_prob,
        })
    }
}

impl OlmoeMoeConfig {
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_experts == 0
            || self.top_k == 0
        {
            return Err(MoeError::InvalidConfig(
                "MoE dimensions and top_k must be non-zero".to_string(),
            ));
        }
        if self.top_k > self.num_experts {
            return Err(MoeError::InvalidConfig(format!(
                "top_k {} exceeds num_experts {}",
                self.top_k, self.num_experts
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn official_shape() -> OlmoeConfig {
        OlmoeConfig {
            model_type: "olmoe".to_string(),
            vocab_size: 50_304,
            hidden_size: 2_048,
            intermediate_size: 1_024,
            num_hidden_layers: 16,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            num_experts: 64,
            num_experts_per_tok: 8,
            max_position_embeddings: 4_096,
            norm_topk_prob: false,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            attention_bias: false,
            clip_qkv: None,
            tie_word_embeddings: false,
            eos_token_id: Some(50_279),
            pad_token_id: Some(1),
        }
    }

    #[test]
    fn accepts_the_published_olmoe_shape() {
        let config = official_shape();
        config.validate().unwrap();
        assert_eq!(config.moe_config().unwrap().top_k, 8);
        assert!(!config.moe_config().unwrap().normalize_top_k);
    }

    #[test]
    fn rejects_invalid_routing_and_attention_geometry() {
        let mut config = official_shape();
        config.num_experts_per_tok = 65;
        assert!(config.validate().is_err());

        let mut config = official_shape();
        config.num_key_value_heads = 3;
        assert!(config.validate().is_err());
    }
}

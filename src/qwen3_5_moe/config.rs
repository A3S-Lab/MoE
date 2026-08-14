use std::collections::BTreeSet;
use std::ops::Deref;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{MoeError, MoeLayerConfig, Result};

fn default_outer_model_type() -> String {
    "qwen3_5_moe".to_string()
}

fn default_text_model_type() -> String {
    "qwen3_5_moe_text".to_string()
}

fn default_hidden_act() -> String {
    "silu".to_string()
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_mamba_dtype() -> String {
    "float32".to_string()
}

/// One or more token IDs as represented by Hugging Face configurations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Qwen36MoeTokenIds {
    One(u32),
    Many(Vec<u32>),
}

impl Qwen36MoeTokenIds {
    pub fn values(&self) -> &[u32] {
        match self {
            Self::One(value) => std::slice::from_ref(value),
            Self::Many(values) => values,
        }
    }
}

/// Token mixer selected for one Qwen3.6 decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen36MoeLayerType {
    LinearAttention,
    FullAttention,
}

/// Rotary embedding fields used by the text-only Qwen3.6 path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen36MoeRopeParameters {
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    #[serde(default)]
    pub mrope_interleaved: bool,
    #[serde(default)]
    pub mrope_section: Vec<usize>,
}

fn default_rope_type() -> String {
    "default".to_string()
}

/// Hugging Face `qwen3_5_moe_text` configuration used by Qwen3.6-35B-A3B.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen36MoeTextConfig {
    #[serde(default = "default_text_model_type")]
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub full_attention_interval: usize,
    pub layer_types: Vec<Qwen36MoeLayerType>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    #[serde(default = "default_mamba_dtype")]
    pub mamba_ssm_dtype: String,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    pub rope_parameters: Qwen36MoeRopeParameters,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f64,
    #[serde(default)]
    pub attn_output_gate: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<Qwen36MoeTokenIds>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
}

/// Outer multimodal checkpoint configuration.
///
/// This implementation deliberately exposes only the text model. Vision and
/// MTP tensors are accepted as authenticated, ignored source families and are
/// not copied into a packed text checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen36MoeConfig {
    #[serde(default = "default_outer_model_type")]
    pub model_type: String,
    pub text_config: Qwen36MoeTextConfig,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub image_token_id: Option<u32>,
    #[serde(default)]
    pub video_token_id: Option<u32>,
    #[serde(default)]
    pub vision_start_token_id: Option<u32>,
    #[serde(default)]
    pub vision_end_token_id: Option<u32>,
}

impl Deref for Qwen36MoeConfig {
    type Target = Qwen36MoeTextConfig;

    fn deref(&self) -> &Self::Target {
        &self.text_config
    }
}

impl Qwen36MoeConfig {
    pub fn from_json_path(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let config: Self = serde_json::from_slice(&bytes)?;
        config.validate()?;
        Ok(config)
    }

    pub fn is_linear_attention_layer(&self, layer: usize) -> bool {
        self.layer_types.get(layer) == Some(&Qwen36MoeLayerType::LinearAttention)
    }

    pub fn is_full_attention_layer(&self, layer: usize) -> bool {
        self.layer_types.get(layer) == Some(&Qwen36MoeLayerType::FullAttention)
    }

    pub fn rotary_dim(&self) -> Result<usize> {
        let value = self.head_dim as f64 * self.rope_parameters.partial_rotary_factor;
        if !value.is_finite() || value.fract() != 0.0 || value < 0.0 {
            return Err(MoeError::InvalidConfig(
                "partial rotary dimension must be a finite integer".to_string(),
            ));
        }
        Ok(value as usize)
    }

    pub fn moe_config(&self) -> Result<MoeLayerConfig> {
        self.validate()?;
        Ok(MoeLayerConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.moe_intermediate_size,
            num_experts: self.num_experts,
            top_k: self.num_experts_per_tok,
            normalize_top_k: true,
        })
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_type != "qwen3_5_moe" {
            return Err(MoeError::InvalidConfig(format!(
                "model_type must be 'qwen3_5_moe', found '{}'",
                self.model_type
            )));
        }
        if self.text_config.model_type != "qwen3_5_moe_text" {
            return Err(MoeError::InvalidConfig(format!(
                "text_config.model_type must be 'qwen3_5_moe_text', found '{}'",
                self.text_config.model_type
            )));
        }
        for (name, value) in [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("max_position_embeddings", self.max_position_embeddings),
            ("full_attention_interval", self.full_attention_interval),
            ("linear_conv_kernel_dim", self.linear_conv_kernel_dim),
            ("linear_key_head_dim", self.linear_key_head_dim),
            ("linear_num_key_heads", self.linear_num_key_heads),
            ("linear_num_value_heads", self.linear_num_value_heads),
            ("linear_value_head_dim", self.linear_value_head_dim),
            ("moe_intermediate_size", self.moe_intermediate_size),
            (
                "shared_expert_intermediate_size",
                self.shared_expert_intermediate_size,
            ),
            ("num_experts", self.num_experts),
            ("num_experts_per_tok", self.num_experts_per_tok),
        ] {
            if value == 0 {
                return Err(MoeError::InvalidConfig(format!("{name} must be non-zero")));
            }
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(MoeError::InvalidConfig(format!(
                "layer_types contains {} entries, expected {}",
                self.layer_types.len(),
                self.num_hidden_layers
            )));
        }
        for (layer, layer_type) in self.layer_types.iter().enumerate() {
            let expected = if (layer + 1).is_multiple_of(self.full_attention_interval) {
                Qwen36MoeLayerType::FullAttention
            } else {
                Qwen36MoeLayerType::LinearAttention
            };
            if *layer_type != expected {
                return Err(MoeError::InvalidConfig(format!(
                    "layer {layer} has {layer_type:?}, expected {expected:?} from full_attention_interval"
                )));
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
        if !self
            .linear_num_value_heads
            .is_multiple_of(self.linear_num_key_heads)
        {
            return Err(MoeError::InvalidConfig(
                "linear_num_value_heads must be divisible by linear_num_key_heads".to_string(),
            ));
        }
        let rotary_dim = self.rotary_dim()?;
        if rotary_dim == 0 || rotary_dim > self.head_dim || !rotary_dim.is_multiple_of(2) {
            return Err(MoeError::InvalidConfig(
                "partial rotary dimension must be non-zero, even, and no larger than head_dim"
                    .to_string(),
            ));
        }
        if self.rope_parameters.rope_type != "default" {
            return Err(MoeError::InvalidConfig(format!(
                "only default rotary scaling is supported, found '{}'",
                self.rope_parameters.rope_type
            )));
        }
        if !self.rope_parameters.rope_theta.is_finite() || self.rope_parameters.rope_theta <= 0.0 {
            return Err(MoeError::InvalidConfig(
                "rope_theta must be finite and positive".to_string(),
            ));
        }
        if self.hidden_act != "silu" {
            return Err(MoeError::InvalidConfig(format!(
                "only Qwen3.6's silu activation is supported, found '{}'",
                self.hidden_act
            )));
        }
        if self.mamba_ssm_dtype != "float32" {
            return Err(MoeError::InvalidConfig(format!(
                "mamba_ssm_dtype must be 'float32', found '{}'",
                self.mamba_ssm_dtype
            )));
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(MoeError::InvalidConfig(
                "rms_norm_eps must be finite and positive".to_string(),
            ));
        }
        if !self.attention_dropout.is_finite() || self.attention_dropout != 0.0 {
            return Err(MoeError::InvalidConfig(
                "inference requires zero attention_dropout".to_string(),
            ));
        }
        if self.attention_bias || !self.attn_output_gate {
            return Err(MoeError::InvalidConfig(
                "Qwen3.6 text inference requires bias-free attention with its output gate"
                    .to_string(),
            ));
        }
        if self.tie_word_embeddings || self.text_config.tie_word_embeddings {
            return Err(MoeError::InvalidConfig(
                "tied Qwen3.6 text embeddings are not supported".to_string(),
            ));
        }
        self.moe_config_unchecked().validate()?;
        if self.num_hidden_layers > u32::MAX as usize {
            return Err(MoeError::InvalidConfig(
                "layer count exceeds the Power routing contract".to_string(),
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

        let checked_weight_geometry = self
            .num_experts
            .checked_mul(2)
            .and_then(|value| value.checked_mul(self.moe_intermediate_size))
            .and_then(|value| value.checked_mul(self.hidden_size))
            .and_then(|routed| {
                self.vocab_size
                    .checked_mul(self.hidden_size)
                    .and_then(|embedding| embedding.checked_add(routed))
            });
        if checked_weight_geometry.is_none() {
            return Err(MoeError::InvalidConfig(
                "declared Qwen3.6 weight dimensions overflow addressable memory".to_string(),
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
            normalize_top_k: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn published_config() -> Qwen36MoeConfig {
        let layer_types = (0..40)
            .map(|layer| {
                if (layer + 1) % 4 == 0 {
                    Qwen36MoeLayerType::FullAttention
                } else {
                    Qwen36MoeLayerType::LinearAttention
                }
            })
            .collect();
        Qwen36MoeConfig {
            model_type: "qwen3_5_moe".to_string(),
            text_config: Qwen36MoeTextConfig {
                model_type: "qwen3_5_moe_text".to_string(),
                vocab_size: 248_320,
                hidden_size: 2_048,
                num_hidden_layers: 40,
                num_attention_heads: 16,
                num_key_value_heads: 2,
                head_dim: 256,
                max_position_embeddings: 262_144,
                full_attention_interval: 4,
                layer_types,
                linear_conv_kernel_dim: 4,
                linear_key_head_dim: 128,
                linear_num_key_heads: 16,
                linear_num_value_heads: 32,
                linear_value_head_dim: 128,
                mamba_ssm_dtype: "float32".to_string(),
                moe_intermediate_size: 512,
                shared_expert_intermediate_size: 512,
                num_experts: 256,
                num_experts_per_tok: 8,
                hidden_act: "silu".to_string(),
                rms_norm_eps: 1e-6,
                rope_parameters: Qwen36MoeRopeParameters {
                    rope_type: "default".to_string(),
                    rope_theta: 10_000_000.0,
                    partial_rotary_factor: 0.25,
                    mrope_interleaved: true,
                    mrope_section: vec![11, 11, 10],
                },
                attention_bias: false,
                attention_dropout: 0.0,
                attn_output_gate: true,
                tie_word_embeddings: false,
                bos_token_id: Some(248_044),
                eos_token_id: Some(Qwen36MoeTokenIds::One(248_044)),
                pad_token_id: None,
            },
            tie_word_embeddings: false,
            image_token_id: Some(248_056),
            video_token_id: Some(248_057),
            vision_start_token_id: Some(248_053),
            vision_end_token_id: Some(248_054),
        }
    }

    #[test]
    fn accepts_the_pinned_public_configuration() {
        let config = published_config();
        config.validate().unwrap();
        assert_eq!(config.hidden_size, 2_048);
        assert_eq!(config.num_hidden_layers, 40);
        assert_eq!(config.num_experts, 256);
        assert_eq!(config.num_experts_per_tok, 8);
        assert_eq!(config.rotary_dim().unwrap(), 64);
        assert_eq!(
            config
                .layer_types
                .iter()
                .filter(|kind| **kind == Qwen36MoeLayerType::LinearAttention)
                .count(),
            30
        );
        assert!(config.moe_config().unwrap().normalize_top_k);
    }
}

use candle_core::Tensor;
use candle_nn::{Linear, Module, RmsNorm, VarBuilder};

use crate::decoder::{append_cache, causal_mask, repeat_key_value, LayerKvCache, RotaryEmbedding};
use crate::{MoeError, Result};

use super::Qwen3MoeConfig;

#[derive(Debug, Clone)]
pub(super) struct Qwen3MoeAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    query_norm: RmsNorm,
    key_norm: RmsNorm,
    rotary: RotaryEmbedding,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    sliding_window: Option<usize>,
}

impl Qwen3MoeAttention {
    pub(super) fn load(
        config: &Qwen3MoeConfig,
        rotary: RotaryEmbedding,
        builder: VarBuilder<'_>,
    ) -> Result<Self> {
        let head_dim = config.attention_head_dim()?;
        let query_size = config
            .num_attention_heads
            .checked_mul(head_dim)
            .ok_or_else(|| MoeError::InvalidConfig("query width overflowed".to_string()))?;
        let key_value_size = config
            .num_key_value_heads
            .checked_mul(head_dim)
            .ok_or_else(|| MoeError::InvalidConfig("key/value width overflowed".to_string()))?;
        Ok(Self {
            query: candle_nn::linear_b(
                config.hidden_size,
                query_size,
                config.attention_bias,
                builder.pp("q_proj"),
            )?,
            key: candle_nn::linear_b(
                config.hidden_size,
                key_value_size,
                config.attention_bias,
                builder.pp("k_proj"),
            )?,
            value: candle_nn::linear_b(
                config.hidden_size,
                key_value_size,
                config.attention_bias,
                builder.pp("v_proj"),
            )?,
            output: candle_nn::linear_b(
                query_size,
                config.hidden_size,
                config.attention_bias,
                builder.pp("o_proj"),
            )?,
            // Qwen3 normalizes each attention head independently, unlike
            // OLMoE's normalization across the complete projected width.
            query_norm: candle_nn::rms_norm(head_dim, config.rms_norm_eps, builder.pp("q_norm"))?,
            key_norm: candle_nn::rms_norm(head_dim, config.rms_norm_eps, builder.pp("k_norm"))?,
            rotary,
            num_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim,
            hidden_size: config.hidden_size,
            sliding_window: if config.use_sliding_window {
                config.sliding_window
            } else {
                None
            },
        })
    }

    pub(super) fn forward(
        &self,
        hidden_states: &Tensor,
        cache: &mut LayerKvCache,
        position: usize,
    ) -> Result<Tensor> {
        let (batch_size, sequence_length, hidden_size) = hidden_states.dims3()?;
        if hidden_size != self.hidden_size {
            return Err(MoeError::InvalidTensor(format!(
                "attention input width must be {}, found {hidden_size}",
                self.hidden_size
            )));
        }

        let mut query = self.query.forward(hidden_states)?.reshape((
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ))?;
        query = self.query_norm.forward(&query)?.transpose(1, 2)?;
        let mut key = self.key.forward(hidden_states)?.reshape((
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ))?;
        key = self.key_norm.forward(&key)?.transpose(1, 2)?;
        let value = self
            .value
            .forward(hidden_states)?
            .reshape((
                batch_size,
                sequence_length,
                self.num_key_value_heads,
                self.head_dim,
            ))?
            .transpose(1, 2)?;
        query = self.rotary.apply(&query, position, sequence_length)?;
        key = self.rotary.apply(&key, position, sequence_length)?;

        let (key, value) = append_cache(cache, key, value, position, batch_size)?;
        let total_length = key.dim(2)?;
        let key = repeat_key_value(&key, self.num_heads / self.num_key_value_heads)?;
        let value = repeat_key_value(&value, self.num_heads / self.num_key_value_heads)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (query.matmul(&key.transpose(2, 3)?)? * scale)?;
        if sequence_length > 1 || self.sliding_window.is_some() {
            scores = scores.broadcast_add(&causal_mask(
                sequence_length,
                total_length,
                position,
                self.sliding_window,
                scores.dtype(),
                scores.device(),
            )?)?;
        }
        let probabilities = candle_nn::ops::softmax_last_dim(&scores)?;
        let query_size = self
            .num_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| MoeError::InvalidTensor("query width overflowed".to_string()))?;
        let attended = probabilities.matmul(&value)?.transpose(1, 2)?.reshape((
            batch_size,
            sequence_length,
            query_size,
        ))?;
        Ok(self.output.forward(&attended)?)
    }
}

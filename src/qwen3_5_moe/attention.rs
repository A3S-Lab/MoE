use candle_core::{Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};

use crate::decoder::{append_cache, causal_mask, repeat_key_value, LayerKvCache, RotaryEmbedding};
use crate::{MoeError, Result};

use super::norm::OffsetRmsNorm;
use super::Qwen36MoeConfig;

#[derive(Debug, Clone)]
pub(super) struct Qwen36MoeAttention {
    query_and_gate: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    query_norm: OffsetRmsNorm,
    key_norm: OffsetRmsNorm,
    rotary: RotaryEmbedding,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    hidden_size: usize,
}

impl Qwen36MoeAttention {
    pub(super) fn load(
        config: &Qwen36MoeConfig,
        rotary: RotaryEmbedding,
        builder: VarBuilder<'_>,
    ) -> Result<Self> {
        let query_size = config
            .num_attention_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| MoeError::InvalidConfig("query width overflowed".to_string()))?;
        let key_value_size = config
            .num_key_value_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| MoeError::InvalidConfig("key/value width overflowed".to_string()))?;
        Ok(Self {
            query_and_gate: candle_nn::linear_no_bias(
                config.hidden_size,
                query_size.checked_mul(2).ok_or_else(|| {
                    MoeError::InvalidConfig("gated query width overflowed".to_string())
                })?,
                builder.pp("q_proj"),
            )?,
            key: candle_nn::linear_no_bias(
                config.hidden_size,
                key_value_size,
                builder.pp("k_proj"),
            )?,
            value: candle_nn::linear_no_bias(
                config.hidden_size,
                key_value_size,
                builder.pp("v_proj"),
            )?,
            output: candle_nn::linear_no_bias(
                query_size,
                config.hidden_size,
                builder.pp("o_proj"),
            )?,
            query_norm: OffsetRmsNorm::load(
                config.head_dim,
                config.rms_norm_eps,
                builder.pp("q_norm"),
            )?,
            key_norm: OffsetRmsNorm::load(
                config.head_dim,
                config.rms_norm_eps,
                builder.pp("k_norm"),
            )?,
            rotary,
            num_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rotary_dim: config.rotary_dim()?,
            hidden_size: config.hidden_size,
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

        // The checkpoint interleaves one query and one output-gate vector per
        // head: [batch, sequence, heads, query || gate].
        let query_and_gate = self.query_and_gate.forward(hidden_states)?.reshape((
            batch_size,
            sequence_length,
            self.num_heads,
            2 * self.head_dim,
        ))?;
        let mut query = self
            .query_norm
            .forward(&query_and_gate.narrow(D::Minus1, 0, self.head_dim)?)?
            .transpose(1, 2)?;
        let gate = query_and_gate
            .narrow(D::Minus1, self.head_dim, self.head_dim)?
            .reshape((batch_size, sequence_length, self.num_heads * self.head_dim))?;
        let mut key = self
            .key_norm
            .forward(&self.key.forward(hidden_states)?.reshape((
                batch_size,
                sequence_length,
                self.num_key_value_heads,
                self.head_dim,
            ))?)?
            .transpose(1, 2)?;
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
        query = self.apply_partial_rotary(&query, position, sequence_length)?;
        key = self.apply_partial_rotary(&key, position, sequence_length)?;

        let (key, value) = append_cache(cache, key, value, position, batch_size)?;
        let total_length = key.dim(2)?;
        let repetitions = self.num_heads / self.num_key_value_heads;
        let key = repeat_key_value(&key, repetitions)?;
        let value = repeat_key_value(&value, repetitions)?;
        let mut scores =
            (query.matmul(&key.transpose(2, 3)?)? * (1.0 / (self.head_dim as f64).sqrt()))?;
        if sequence_length > 1 {
            scores = scores.broadcast_add(&causal_mask(
                sequence_length,
                total_length,
                position,
                None,
                scores.dtype(),
                scores.device(),
            )?)?;
        }
        let probabilities = candle_nn::ops::softmax_last_dim(&scores)?;
        let attended = probabilities.matmul(&value)?.transpose(1, 2)?.reshape((
            batch_size,
            sequence_length,
            self.num_heads * self.head_dim,
        ))?;
        let gated = attended.mul(&candle_nn::ops::sigmoid(&gate)?)?;
        Ok(self.output.forward(&gated)?)
    }

    fn apply_partial_rotary(
        &self,
        tensor: &Tensor,
        position: usize,
        sequence_length: usize,
    ) -> Result<Tensor> {
        let rotated = self.rotary.apply(
            &tensor.narrow(D::Minus1, 0, self.rotary_dim)?,
            position,
            sequence_length,
        )?;
        if self.rotary_dim == self.head_dim {
            return Ok(rotated);
        }
        Ok(Tensor::cat(
            &[
                &rotated,
                &tensor.narrow(D::Minus1, self.rotary_dim, self.head_dim - self.rotary_dim)?,
            ],
            D::Minus1,
        )?)
    }
}

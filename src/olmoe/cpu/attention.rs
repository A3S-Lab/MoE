use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, RmsNorm, VarBuilder};

use crate::{MoeError, Result};

use super::cache::LayerKvCache;
use crate::olmoe::OlmoeConfig;

#[derive(Debug, Clone)]
pub(super) struct OlmoeRotaryEmbedding {
    cosine: Tensor,
    sine: Tensor,
}

impl OlmoeRotaryEmbedding {
    pub(super) fn new(config: &OlmoeConfig, dtype: DType, device: &Device) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        if !head_dim.is_multiple_of(2) {
            return Err(MoeError::InvalidConfig(format!(
                "attention head dimension {head_dim} must be even for rotary embeddings"
            )));
        }
        let half_dim = head_dim / 2;
        let mut cosine = Vec::with_capacity(config.max_position_embeddings * half_dim);
        let mut sine = Vec::with_capacity(config.max_position_embeddings * half_dim);
        for position in 0..config.max_position_embeddings {
            for dimension in 0..half_dim {
                let exponent = (2 * dimension) as f64 / head_dim as f64;
                let frequency = 1.0 / config.rope_theta.powf(exponent);
                let angle = position as f64 * frequency;
                cosine.push(angle.cos() as f32);
                sine.push(angle.sin() as f32);
            }
        }
        Ok(Self {
            cosine: Tensor::from_vec(cosine, (config.max_position_embeddings, half_dim), device)?
                .to_dtype(dtype)?,
            sine: Tensor::from_vec(sine, (config.max_position_embeddings, half_dim), device)?
                .to_dtype(dtype)?,
        })
    }

    fn apply(&self, tensor: &Tensor, position: usize, sequence_length: usize) -> Result<Tensor> {
        let cosine = self
            .cosine
            .narrow(0, position, sequence_length)?
            .contiguous()?;
        let sine = self
            .sine
            .narrow(0, position, sequence_length)?
            .contiguous()?;
        Ok(candle_nn::rotary_emb::rope(
            &tensor.contiguous()?,
            &cosine,
            &sine,
        )?)
    }
}

#[derive(Debug, Clone)]
pub(super) struct OlmoeAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    query_norm: RmsNorm,
    key_norm: RmsNorm,
    rotary: OlmoeRotaryEmbedding,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    clip_qkv: Option<f64>,
}

impl OlmoeAttention {
    pub(super) fn load(
        config: &OlmoeConfig,
        rotary: OlmoeRotaryEmbedding,
        builder: VarBuilder<'_>,
    ) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let query_size = config.num_attention_heads * head_dim;
        let key_value_size = config.num_key_value_heads * head_dim;
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
            query_norm: candle_nn::rms_norm(query_size, config.rms_norm_eps, builder.pp("q_norm"))?,
            key_norm: candle_nn::rms_norm(
                key_value_size,
                config.rms_norm_eps,
                builder.pp("k_norm"),
            )?,
            rotary,
            num_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim,
            clip_qkv: config.clip_qkv,
        })
    }

    pub(super) fn forward(
        &self,
        hidden_states: &Tensor,
        cache: &mut LayerKvCache,
        position: usize,
    ) -> Result<Tensor> {
        let (batch_size, sequence_length, hidden_size) = hidden_states.dims3()?;
        let query_size = self.num_heads * self.head_dim;
        if hidden_size != query_size {
            return Err(MoeError::InvalidTensor(format!(
                "attention input width must be {query_size}, found {hidden_size}"
            )));
        }

        let mut query = self
            .query_norm
            .forward(&self.query.forward(hidden_states)?)?;
        let mut key = self.key_norm.forward(&self.key.forward(hidden_states)?)?;
        let mut value = self.value.forward(hidden_states)?;
        if let Some(limit) = self.clip_qkv {
            query = query.clamp(-limit, limit)?;
            key = key.clamp(-limit, limit)?;
            value = value.clamp(-limit, limit)?;
        }

        query = query
            .reshape((batch_size, sequence_length, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        key = key
            .reshape((
                batch_size,
                sequence_length,
                self.num_key_value_heads,
                self.head_dim,
            ))?
            .transpose(1, 2)?;
        value = value
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
        if sequence_length > 1 {
            let mask = causal_mask(
                sequence_length,
                total_length,
                position,
                scores.dtype(),
                scores.device(),
            )?;
            scores = scores.broadcast_add(&mask)?;
        }
        let probabilities = candle_nn::ops::softmax_last_dim(&scores)?;
        let attended = probabilities.matmul(&value)?;
        let attended =
            attended
                .transpose(1, 2)?
                .reshape((batch_size, sequence_length, query_size))?;
        Ok(self.output.forward(&attended)?)
    }
}

fn append_cache(
    cache: &mut LayerKvCache,
    key: Tensor,
    value: Tensor,
    position: usize,
    batch_size: usize,
) -> Result<(Tensor, Tensor)> {
    let (key, value) = match (&cache.key, &cache.value) {
        (None, None) if position == 0 => (key, value),
        (Some(cached_key), Some(cached_value)) => {
            if cached_key.dim(0)? != batch_size
                || cached_value.dim(0)? != batch_size
                || cached_key.dim(2)? != position
                || cached_value.dim(2)? != position
            {
                return Err(MoeError::InvalidTensor(
                    "KV cache tensor shape does not match the declared position".to_string(),
                ));
            }
            (
                Tensor::cat(&[cached_key, &key], 2)?,
                Tensor::cat(&[cached_value, &value], 2)?,
            )
        }
        (None, None) => {
            return Err(MoeError::InvalidTensor(
                "KV cache is empty at a non-zero position".to_string(),
            ));
        }
        _ => {
            return Err(MoeError::InvalidTensor(
                "KV cache contains only one of key or value".to_string(),
            ));
        }
    };
    cache.key = Some(key.clone());
    cache.value = Some(value.clone());
    Ok((key, value))
}

fn repeat_key_value(tensor: &Tensor, repetitions: usize) -> Result<Tensor> {
    if repetitions == 1 {
        return Ok(tensor.clone());
    }
    let (batch_size, key_value_heads, sequence_length, head_dim) = tensor.dims4()?;
    Ok(tensor
        .unsqueeze(2)?
        .expand((
            batch_size,
            key_value_heads,
            repetitions,
            sequence_length,
            head_dim,
        ))?
        .reshape((
            batch_size,
            key_value_heads * repetitions,
            sequence_length,
            head_dim,
        ))?)
}

fn causal_mask(
    sequence_length: usize,
    total_length: usize,
    position: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mut values = Vec::with_capacity(sequence_length * total_length);
    for query in 0..sequence_length {
        let absolute_query = position + query;
        for key in 0..total_length {
            values.push(if key <= absolute_query {
                0.0_f32
            } else {
                f32::NEG_INFINITY
            });
        }
    }
    Ok(
        Tensor::from_vec(values, (sequence_length, total_length), device)?
            .to_dtype(dtype)?
            .unsqueeze(0)?
            .unsqueeze(0)?,
    )
}

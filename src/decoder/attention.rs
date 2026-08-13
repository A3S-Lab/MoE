use candle_core::{DType, Device, Tensor};

use crate::decoder::LayerKvCache;
use crate::{MoeError, Result};

#[derive(Debug, Clone)]
pub(crate) struct RotaryEmbedding {
    cosine: Tensor,
    sine: Tensor,
}

impl RotaryEmbedding {
    pub(crate) fn new(
        head_dim: usize,
        max_positions: usize,
        theta: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(MoeError::InvalidConfig(format!(
                "attention head dimension {head_dim} must be non-zero and even"
            )));
        }
        if max_positions == 0 || !theta.is_finite() || theta <= 0.0 {
            return Err(MoeError::InvalidConfig(
                "rotary positions and theta must be positive".to_string(),
            ));
        }
        let half_dim = head_dim / 2;
        let elements = max_positions.checked_mul(half_dim).ok_or_else(|| {
            MoeError::InvalidConfig("rotary embedding size overflowed".to_string())
        })?;
        let mut cosine = Vec::with_capacity(elements);
        let mut sine = Vec::with_capacity(elements);
        for position in 0..max_positions {
            for dimension in 0..half_dim {
                let exponent = (2 * dimension) as f64 / head_dim as f64;
                let frequency = 1.0 / theta.powf(exponent);
                let angle = position as f64 * frequency;
                cosine.push(angle.cos() as f32);
                sine.push(angle.sin() as f32);
            }
        }
        Ok(Self {
            cosine: Tensor::from_vec(cosine, (max_positions, half_dim), device)?.to_dtype(dtype)?,
            sine: Tensor::from_vec(sine, (max_positions, half_dim), device)?.to_dtype(dtype)?,
        })
    }

    pub(crate) fn apply(
        &self,
        tensor: &Tensor,
        position: usize,
        sequence_length: usize,
    ) -> Result<Tensor> {
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

pub(crate) fn repeat_key_value(tensor: &Tensor, repetitions: usize) -> Result<Tensor> {
    if repetitions == 0 {
        return Err(MoeError::InvalidConfig(
            "key/value head repetitions must be non-zero".to_string(),
        ));
    }
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

pub(crate) fn append_cache(
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

pub(crate) fn causal_mask(
    sequence_length: usize,
    total_length: usize,
    position: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    if sequence_length == 0 || total_length == 0 {
        return Err(MoeError::InvalidTensor(
            "causal mask dimensions must be non-zero".to_string(),
        ));
    }
    let mut values = Vec::with_capacity(sequence_length * total_length);
    for query in 0..sequence_length {
        let absolute_query = position
            .checked_add(query)
            .ok_or_else(|| MoeError::InvalidTensor("causal position overflowed".to_string()))?;
        let first_key = sliding_window
            .map(|window| absolute_query.saturating_add(1).saturating_sub(window))
            .unwrap_or(0);
        for key in 0..total_length {
            values.push(if (first_key..=absolute_query).contains(&key) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_causal_mask_keeps_only_the_declared_window() {
        let mask = causal_mask(2, 4, 2, Some(2), DType::F32, &Device::Cpu)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(mask[0], [f32::NEG_INFINITY, 0.0, 0.0, f32::NEG_INFINITY]);
        assert_eq!(mask[1], [f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0, 0.0]);
    }
}

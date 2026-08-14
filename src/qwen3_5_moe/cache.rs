use candle_core::Tensor;

use crate::decoder::LayerKvCache;
use crate::{MoeError, Result};

use super::{Qwen36MoeConfig, Qwen36MoeLayerType};

#[derive(Debug, Clone)]
pub(super) enum Qwen36MoeLayerCache {
    Linear {
        conv_state: Option<Vec<f32>>,
        recurrent_state: Option<Vec<f32>>,
    },
    Full {
        kv: LayerKvCache,
    },
}

/// Per-session heterogeneous state for Qwen3.6 linear and full attention.
#[derive(Debug, Clone)]
pub struct Qwen36MoeCache {
    layers: Vec<Qwen36MoeLayerCache>,
    position: usize,
    max_position_embeddings: usize,
    batch_size: Option<usize>,
}

impl Qwen36MoeCache {
    pub(super) fn new(config: &Qwen36MoeConfig) -> Self {
        let layers = config
            .layer_types
            .iter()
            .map(|layer_type| match layer_type {
                Qwen36MoeLayerType::LinearAttention => Qwen36MoeLayerCache::Linear {
                    conv_state: None,
                    recurrent_state: None,
                },
                Qwen36MoeLayerType::FullAttention => Qwen36MoeLayerCache::Full {
                    kv: LayerKvCache::default(),
                },
            })
            .collect();
        Self {
            layers,
            position: 0,
            max_position_embeddings: config.max_position_embeddings,
            batch_size: None,
        }
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    pub fn resident_bytes(&self) -> Result<u64> {
        self.layers.iter().try_fold(0_u64, |total, layer| {
            let bytes = match layer {
                Qwen36MoeLayerCache::Linear {
                    conv_state,
                    recurrent_state,
                } => conv_state
                    .iter()
                    .chain(recurrent_state)
                    .try_fold(0_u64, |bytes, state| {
                        let state_bytes = state
                            .len()
                            .checked_mul(size_of::<f32>())
                            .and_then(|value| u64::try_from(value).ok())
                            .ok_or_else(|| {
                                MoeError::InvalidTensor(
                                    "linear-attention state byte count overflowed".to_string(),
                                )
                            })?;
                        bytes.checked_add(state_bytes).ok_or_else(|| {
                            MoeError::InvalidTensor(
                                "linear-attention state byte count overflowed".to_string(),
                            )
                        })
                    })?,
                Qwen36MoeLayerCache::Full { kv } => [&kv.key, &kv.value]
                    .into_iter()
                    .flatten()
                    .try_fold(0_u64, |bytes, tensor: &Tensor| {
                        let tensor_bytes = tensor
                            .elem_count()
                            .checked_mul(tensor.dtype().size_in_bytes())
                            .and_then(|value| u64::try_from(value).ok())
                            .ok_or_else(|| {
                                MoeError::InvalidTensor(
                                    "attention cache byte count overflowed".to_string(),
                                )
                            })?;
                        bytes.checked_add(tensor_bytes).ok_or_else(|| {
                            MoeError::InvalidTensor(
                                "attention cache byte count overflowed".to_string(),
                            )
                        })
                    })?,
            };
            total.checked_add(bytes).ok_or_else(|| {
                MoeError::InvalidTensor("cache aggregate byte count overflowed".to_string())
            })
        })
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            match layer {
                Qwen36MoeLayerCache::Linear {
                    conv_state,
                    recurrent_state,
                } => {
                    *conv_state = None;
                    *recurrent_state = None;
                }
                Qwen36MoeLayerCache::Full { kv } => *kv = LayerKvCache::default(),
            }
        }
        self.position = 0;
        self.batch_size = None;
    }

    pub(super) fn validate_step(
        &self,
        layer_count: usize,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<()> {
        if self.layers.len() != layer_count {
            return Err(MoeError::InvalidTensor(format!(
                "cache has {} layers, model requires {layer_count}",
                self.layers.len()
            )));
        }
        if batch_size == 0 || sequence_length == 0 {
            return Err(MoeError::InvalidTensor(
                "cache step requires a non-empty token batch".to_string(),
            ));
        }
        if self
            .batch_size
            .is_some_and(|cached_batch| cached_batch != batch_size)
        {
            return Err(MoeError::InvalidTensor(format!(
                "cache batch size {:?} cannot change to {batch_size}",
                self.batch_size
            )));
        }
        let end = self
            .position
            .checked_add(sequence_length)
            .ok_or_else(|| MoeError::InvalidTensor("cache position overflowed".to_string()))?;
        if end > self.max_position_embeddings {
            return Err(MoeError::InvalidTensor(format!(
                "cache step ends at position {end}, beyond maximum {}",
                self.max_position_embeddings
            )));
        }
        Ok(())
    }

    pub(super) fn linear_mut(
        &mut self,
        layer: usize,
    ) -> Result<(&mut Option<Vec<f32>>, &mut Option<Vec<f32>>)> {
        match self.layers.get_mut(layer) {
            Some(Qwen36MoeLayerCache::Linear {
                conv_state,
                recurrent_state,
            }) => Ok((conv_state, recurrent_state)),
            Some(Qwen36MoeLayerCache::Full { .. }) => Err(MoeError::InvalidTensor(format!(
                "cache layer {layer} is full attention, not linear attention"
            ))),
            None => Err(MoeError::InvalidTensor(format!(
                "cache layer {layer} is unavailable"
            ))),
        }
    }

    pub(super) fn full_mut(&mut self, layer: usize) -> Result<&mut LayerKvCache> {
        match self.layers.get_mut(layer) {
            Some(Qwen36MoeLayerCache::Full { kv }) => Ok(kv),
            Some(Qwen36MoeLayerCache::Linear { .. }) => Err(MoeError::InvalidTensor(format!(
                "cache layer {layer} is linear attention, not full attention"
            ))),
            None => Err(MoeError::InvalidTensor(format!(
                "cache layer {layer} is unavailable"
            ))),
        }
    }

    pub(super) fn commit_step(&mut self, batch_size: usize, sequence_length: usize) {
        self.batch_size = Some(batch_size);
        self.position += sequence_length;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen3_5_moe::{Qwen36MoeRopeParameters, Qwen36MoeTextConfig, Qwen36MoeTokenIds};

    fn config() -> Qwen36MoeConfig {
        Qwen36MoeConfig {
            model_type: "qwen3_5_moe".to_string(),
            text_config: Qwen36MoeTextConfig {
                model_type: "qwen3_5_moe_text".to_string(),
                vocab_size: 8,
                hidden_size: 4,
                num_hidden_layers: 4,
                num_attention_heads: 2,
                num_key_value_heads: 1,
                head_dim: 4,
                max_position_embeddings: 16,
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
                moe_intermediate_size: 2,
                shared_expert_intermediate_size: 2,
                num_experts: 2,
                num_experts_per_tok: 1,
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
                eos_token_id: Some(Qwen36MoeTokenIds::One(7)),
                pad_token_id: None,
            },
            tie_word_embeddings: false,
            image_token_id: None,
            video_token_id: None,
            vision_start_token_id: None,
            vision_end_token_id: None,
        }
    }

    #[test]
    fn heterogeneous_cache_is_transactional_and_resettable() {
        let mut cache = Qwen36MoeCache::new(&config());
        cache.validate_step(4, 1, 3).unwrap();
        *cache.linear_mut(0).unwrap().0 = Some(vec![1.0; 4]);
        assert_eq!(cache.resident_bytes().unwrap(), 16);
        cache.commit_step(1, 3);
        assert!(cache.validate_step(4, 2, 1).is_err());
        assert!(cache.full_mut(0).is_err());
        assert!(cache.linear_mut(3).is_err());
        cache.reset();
        assert_eq!(cache.position(), 0);
        assert_eq!(cache.resident_bytes().unwrap(), 0);
    }
}

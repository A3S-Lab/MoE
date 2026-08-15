use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};

use crate::decoder::RotaryEmbedding;
use crate::{MoeError, Result};

use super::attention::Qwen36MoeAttention;
use super::gated_delta_net::Qwen36GatedDeltaNet;
use super::mlp::Qwen36SharedExpert;
use super::norm::OffsetRmsNorm;
use super::{Qwen36MoeCache, Qwen36MoeConfig, Qwen36MoeLayerType};

#[derive(Debug, Clone)]
enum Qwen36TokenMixer {
    Linear(Qwen36GatedDeltaNet),
    Full(Qwen36MoeAttention),
}

#[derive(Debug, Clone)]
struct Qwen36DenseDecoderLayer {
    input_norm: OffsetRmsNorm,
    mixer: Qwen36TokenMixer,
    post_attention_norm: OffsetRmsNorm,
    shared_expert: Qwen36SharedExpert,
}

/// Resident non-routed Qwen3.6 text state shared by correctness and streaming.
#[derive(Debug, Clone)]
pub(super) struct Qwen36DenseModel {
    config: Qwen36MoeConfig,
    embeddings: Embedding,
    layers: Vec<Qwen36DenseDecoderLayer>,
    final_norm: OffsetRmsNorm,
    lm_head: Linear,
}

impl Qwen36DenseModel {
    pub(super) fn load(config: Qwen36MoeConfig, builder: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        if builder.dtype() != DType::F32 {
            return Err(MoeError::InvalidConfig(format!(
                "Qwen3.6 text execution requires F32, found {:?}",
                builder.dtype()
            )));
        }
        let rotary = RotaryEmbedding::new(
            config.rotary_dim()?,
            config.max_position_embeddings,
            config.rope_parameters.rope_theta,
            builder.dtype(),
            builder.device(),
        )?;
        let embeddings = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            builder.pp("model.language_model.embed_tokens"),
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for (layer, layer_type) in config.layer_types.iter().enumerate() {
            let layer_builder = builder.pp(format!("model.language_model.layers.{layer}"));
            let mixer = match layer_type {
                Qwen36MoeLayerType::LinearAttention => Qwen36TokenMixer::Linear(
                    Qwen36GatedDeltaNet::load(&config, layer_builder.pp("linear_attn"))?,
                ),
                Qwen36MoeLayerType::FullAttention => {
                    Qwen36TokenMixer::Full(Qwen36MoeAttention::load(
                        &config,
                        rotary.clone(),
                        layer_builder.pp("self_attn"),
                    )?)
                }
            };
            layers.push(Qwen36DenseDecoderLayer {
                input_norm: OffsetRmsNorm::load(
                    config.hidden_size,
                    config.rms_norm_eps,
                    layer_builder.pp("input_layernorm"),
                )?,
                mixer,
                post_attention_norm: OffsetRmsNorm::load(
                    config.hidden_size,
                    config.rms_norm_eps,
                    layer_builder.pp("post_attention_layernorm"),
                )?,
                shared_expert: Qwen36SharedExpert::load(&config, layer_builder.pp("mlp"))?,
            });
        }
        let final_norm = OffsetRmsNorm::load(
            config.hidden_size,
            config.rms_norm_eps,
            builder.pp("model.language_model.norm"),
        )?;
        let lm_head = candle_nn::linear_no_bias(
            config.hidden_size,
            config.vocab_size,
            builder.pp("lm_head"),
        )?;
        Ok(Self {
            config,
            embeddings,
            layers,
            final_norm,
            lm_head,
        })
    }

    pub(super) fn config(&self) -> &Qwen36MoeConfig {
        &self.config
    }

    pub(super) fn device(&self) -> &Device {
        self.embeddings.embeddings().device()
    }

    pub(super) fn new_cache(&self) -> Qwen36MoeCache {
        Qwen36MoeCache::new(&self.config)
    }

    pub(super) fn validate_step(
        &self,
        token_ids: &Tensor,
        cache: &Qwen36MoeCache,
    ) -> Result<(usize, usize, usize)> {
        if token_ids.dtype() != DType::U32 {
            return Err(MoeError::InvalidTensor(format!(
                "token IDs must use U32, found {:?}",
                token_ids.dtype()
            )));
        }
        if !token_ids.device().same_device(self.device()) {
            return Err(MoeError::InvalidTensor(
                "token IDs and Qwen3.6 dense weights must use the same device".to_string(),
            ));
        }
        let (batch_size, sequence_length) = token_ids.dims2()?;
        if token_ids
            .flatten_all()?
            .to_vec1::<u32>()?
            .into_iter()
            .any(|token| token as usize >= self.config.vocab_size)
        {
            return Err(MoeError::InvalidTensor(
                "token IDs contain a value outside the model vocabulary".to_string(),
            ));
        }
        cache.validate_step(self.layers.len(), batch_size, sequence_length)?;
        Ok((batch_size, sequence_length, cache.position()))
    }

    pub(super) fn embed(&self, token_ids: &Tensor) -> Result<Tensor> {
        Ok(self.embeddings.forward(token_ids)?)
    }

    pub(super) fn forward_mixer(
        &self,
        layer: usize,
        hidden_states: &Tensor,
        cache: &mut Qwen36MoeCache,
        position: usize,
    ) -> Result<Tensor> {
        let weights = self.layer(layer)?;
        let normalized = weights.input_norm.forward(hidden_states)?;
        let mixed = match &weights.mixer {
            Qwen36TokenMixer::Linear(mixer) => {
                let (conv_state, recurrent_state) = cache.linear_mut(layer)?;
                mixer.forward(&normalized, conv_state, recurrent_state)?
            }
            Qwen36TokenMixer::Full(mixer) => {
                mixer.forward(&normalized, cache.full_mut(layer)?, position)?
            }
        };
        Ok(hidden_states.add(&mixed)?)
    }

    pub(super) fn normalize_mlp_input(
        &self,
        layer: usize,
        hidden_states: &Tensor,
    ) -> Result<Tensor> {
        self.layer(layer)?
            .post_attention_norm
            .forward(hidden_states)
    }

    pub(super) fn forward_shared_expert(
        &self,
        layer: usize,
        hidden_states: &Tensor,
    ) -> Result<Tensor> {
        self.layer(layer)?.shared_expert.forward(hidden_states)
    }

    pub(super) fn finish(&self, hidden_states: &Tensor) -> Result<Tensor> {
        Ok(self
            .lm_head
            .forward(&self.final_norm.forward(hidden_states)?)?
            .to_dtype(DType::F32)?)
    }

    pub(super) fn commit_cache(
        &self,
        cache: &mut Qwen36MoeCache,
        batch_size: usize,
        sequence_length: usize,
    ) {
        cache.commit_step(batch_size, sequence_length);
    }

    fn layer(&self, layer: usize) -> Result<&Qwen36DenseDecoderLayer> {
        self.layers.get(layer).ok_or_else(|| {
            MoeError::Inference(format!("dense decoder layer {layer} is unavailable"))
        })
    }
}

pub(crate) fn validate_generation_request(
    config: &Qwen36MoeConfig,
    prompt: &[u32],
    max_new_tokens: usize,
) -> Result<()> {
    if prompt.is_empty() {
        return Err(MoeError::InvalidTensor(
            "greedy generation requires at least one prompt token".to_string(),
        ));
    }
    if prompt
        .iter()
        .any(|token| *token as usize >= config.vocab_size)
    {
        return Err(MoeError::InvalidTensor(
            "prompt contains a token outside the model vocabulary".to_string(),
        ));
    }
    let requested = prompt
        .len()
        .checked_add(max_new_tokens)
        .ok_or_else(|| MoeError::InvalidTensor("generation token count overflowed".to_string()))?;
    if requested > config.max_position_embeddings {
        return Err(MoeError::InvalidTensor(format!(
            "generation requests {requested} positions, beyond maximum {}",
            config.max_position_embeddings
        )));
    }
    Ok(())
}

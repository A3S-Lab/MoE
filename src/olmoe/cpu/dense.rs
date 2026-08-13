use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::decoder::RotaryEmbedding;
use crate::olmoe::OlmoeConfig;
use crate::{MoeError, Result};

use super::attention::OlmoeAttention;
use super::OlmoeKvCache;

#[derive(Debug, Clone)]
struct OlmoeDenseDecoderLayer {
    input_norm: RmsNorm,
    attention: OlmoeAttention,
    post_attention_norm: RmsNorm,
}

impl OlmoeDenseDecoderLayer {
    fn load(
        config: &OlmoeConfig,
        rotary: RotaryEmbedding,
        builder: VarBuilder<'_>,
    ) -> Result<Self> {
        Ok(Self {
            input_norm: candle_nn::rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                builder.pp("input_layernorm"),
            )?,
            attention: OlmoeAttention::load(config, rotary, builder.pp("self_attn"))?,
            post_attention_norm: candle_nn::rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                builder.pp("post_attention_layernorm"),
            )?,
        })
    }
}

/// Dense OLMoE components shared by resident and Power-streaming execution.
#[derive(Debug, Clone)]
pub(in crate::olmoe) struct OlmoeDenseModel {
    config: OlmoeConfig,
    embeddings: Embedding,
    layers: Vec<OlmoeDenseDecoderLayer>,
    final_norm: RmsNorm,
    lm_head: Linear,
}

impl OlmoeDenseModel {
    pub(in crate::olmoe) fn load(config: OlmoeConfig, builder: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        if !builder.device().is_cpu() {
            return Err(MoeError::InvalidConfig(
                "OLMoE CPU dense execution requires a CPU VarBuilder".to_string(),
            ));
        }
        if builder.dtype() != DType::F32 {
            return Err(MoeError::InvalidConfig(format!(
                "OLMoE CPU dense execution requires F32, found {:?}",
                builder.dtype()
            )));
        }
        let rotary = RotaryEmbedding::new(
            config.hidden_size / config.num_attention_heads,
            config.max_position_embeddings,
            config.rope_theta,
            builder.dtype(),
            builder.device(),
        )?;
        let embeddings = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            builder.pp("model.embed_tokens"),
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            layers.push(OlmoeDenseDecoderLayer::load(
                &config,
                rotary.clone(),
                builder.pp(format!("model.layers.{layer}")),
            )?);
        }
        let final_norm = candle_nn::rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            builder.pp("model.norm"),
        )?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(embeddings.embeddings().clone(), None)
        } else {
            candle_nn::linear_no_bias(config.hidden_size, config.vocab_size, builder.pp("lm_head"))?
        };
        Ok(Self {
            config,
            embeddings,
            layers,
            final_norm,
            lm_head,
        })
    }

    pub(in crate::olmoe) fn config(&self) -> &OlmoeConfig {
        &self.config
    }

    pub(in crate::olmoe) fn device(&self) -> &Device {
        self.embeddings.embeddings().device()
    }

    pub(in crate::olmoe) fn new_cache(&self) -> OlmoeKvCache {
        OlmoeKvCache::new(
            self.config.num_hidden_layers,
            self.config.max_position_embeddings,
        )
    }

    pub(in crate::olmoe) fn validate_step(
        &self,
        token_ids: &Tensor,
        cache: &OlmoeKvCache,
    ) -> Result<(usize, usize, usize)> {
        if token_ids.dtype() != DType::U32 {
            return Err(MoeError::InvalidTensor(format!(
                "token IDs must use U32, found {:?}",
                token_ids.dtype()
            )));
        }
        if !token_ids.device().same_device(self.device()) {
            return Err(MoeError::InvalidTensor(
                "token IDs and OLMoE dense weights must use the same device".to_string(),
            ));
        }
        let (batch_size, sequence_length) = token_ids.dims2()?;
        cache.validate_step(self.layers.len(), batch_size, sequence_length)?;
        Ok((batch_size, sequence_length, cache.position()))
    }

    pub(in crate::olmoe) fn embed(&self, token_ids: &Tensor) -> Result<Tensor> {
        Ok(self.embeddings.forward(token_ids)?)
    }

    pub(in crate::olmoe) fn forward_attention(
        &self,
        layer: usize,
        hidden_states: &Tensor,
        cache: &mut OlmoeKvCache,
        position: usize,
    ) -> Result<Tensor> {
        let weights = self.layers.get(layer).ok_or_else(|| {
            MoeError::Inference(format!("dense decoder layer {layer} is unavailable"))
        })?;
        let layer_cache = cache.layer_mut(layer)?;
        let normalized = weights.input_norm.forward(hidden_states)?;
        let attended = weights
            .attention
            .forward(&normalized, layer_cache, position)?;
        Ok(hidden_states.add(&attended)?)
    }

    pub(in crate::olmoe) fn normalize_mlp_input(
        &self,
        layer: usize,
        hidden_states: &Tensor,
    ) -> Result<Tensor> {
        let weights = self.layers.get(layer).ok_or_else(|| {
            MoeError::Inference(format!("dense decoder layer {layer} is unavailable"))
        })?;
        Ok(weights.post_attention_norm.forward(hidden_states)?)
    }

    pub(in crate::olmoe) fn finish(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.final_norm.forward(hidden_states)?;
        Ok(self.lm_head.forward(&hidden_states)?.to_dtype(DType::F32)?)
    }

    pub(in crate::olmoe) fn commit_cache(
        &self,
        cache: &mut OlmoeKvCache,
        batch_size: usize,
        sequence_length: usize,
    ) {
        cache.commit_step(batch_size, sequence_length);
    }
}

pub(in crate::olmoe) fn validate_generation_request(
    config: &OlmoeConfig,
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

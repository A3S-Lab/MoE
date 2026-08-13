use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::decoder::RotaryEmbedding;
use crate::{MoeError, Result};

use super::attention::Qwen3MoeAttention;
use super::mlp::DenseMlp;
use super::{Qwen3MoeConfig, Qwen3MoeKvCache};

#[derive(Debug, Clone)]
struct Qwen3MoeDenseDecoderLayer {
    input_norm: RmsNorm,
    attention: Qwen3MoeAttention,
    post_attention_norm: RmsNorm,
    dense_mlp: Option<DenseMlp>,
}

impl Qwen3MoeDenseDecoderLayer {
    fn load(
        config: &Qwen3MoeConfig,
        layer: usize,
        rotary: RotaryEmbedding,
        builder: VarBuilder<'_>,
    ) -> Result<Self> {
        Ok(Self {
            input_norm: candle_nn::rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                builder.pp("input_layernorm"),
            )?,
            attention: Qwen3MoeAttention::load(config, rotary, builder.pp("self_attn"))?,
            post_attention_norm: candle_nn::rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                builder.pp("post_attention_layernorm"),
            )?,
            dense_mlp: if config.is_sparse_layer(layer) {
                None
            } else {
                Some(DenseMlp::load(config, builder.pp("mlp"))?)
            },
        })
    }
}

/// Resident non-expert Qwen3-MoE decoder state shared by correctness and
/// Power-streaming execution.
#[derive(Debug, Clone)]
pub(super) struct Qwen3MoeDenseModel {
    config: Qwen3MoeConfig,
    embeddings: Embedding,
    layers: Vec<Qwen3MoeDenseDecoderLayer>,
    final_norm: RmsNorm,
    lm_head: Linear,
}

impl Qwen3MoeDenseModel {
    pub(super) fn load(config: Qwen3MoeConfig, builder: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        if !builder.device().is_cpu() {
            return Err(MoeError::InvalidConfig(
                "Qwen3-MoE CPU dense execution requires a CPU VarBuilder".to_string(),
            ));
        }
        if builder.dtype() != DType::F32 {
            return Err(MoeError::InvalidConfig(format!(
                "Qwen3-MoE CPU dense execution requires F32, found {:?}",
                builder.dtype()
            )));
        }

        let rotary = RotaryEmbedding::new(
            config.attention_head_dim()?,
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
            layers.push(Qwen3MoeDenseDecoderLayer::load(
                &config,
                layer,
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

    pub(super) fn config(&self) -> &Qwen3MoeConfig {
        &self.config
    }

    pub(super) fn device(&self) -> &Device {
        self.embeddings.embeddings().device()
    }

    pub(super) fn new_cache(&self) -> Qwen3MoeKvCache {
        Qwen3MoeKvCache::new(
            self.config.num_hidden_layers,
            self.config.max_position_embeddings,
        )
    }

    pub(super) fn validate_step(
        &self,
        token_ids: &Tensor,
        cache: &Qwen3MoeKvCache,
    ) -> Result<(usize, usize, usize)> {
        if token_ids.dtype() != DType::U32 {
            return Err(MoeError::InvalidTensor(format!(
                "token IDs must use U32, found {:?}",
                token_ids.dtype()
            )));
        }
        if !token_ids.device().same_device(self.device()) {
            return Err(MoeError::InvalidTensor(
                "token IDs and Qwen3-MoE dense weights must use the same device".to_string(),
            ));
        }
        let (batch_size, sequence_length) = token_ids.dims2()?;
        cache.validate_step(self.layers.len(), batch_size, sequence_length)?;
        Ok((batch_size, sequence_length, cache.position()))
    }

    pub(super) fn embed(&self, token_ids: &Tensor) -> Result<Tensor> {
        Ok(self.embeddings.forward(token_ids)?)
    }

    pub(super) fn forward_attention(
        &self,
        layer: usize,
        hidden_states: &Tensor,
        cache: &mut Qwen3MoeKvCache,
        position: usize,
    ) -> Result<Tensor> {
        let weights = self.layer(layer)?;
        let normalized = weights.input_norm.forward(hidden_states)?;
        let attention =
            weights
                .attention
                .forward(&normalized, cache.layer_mut(layer)?, position)?;
        Ok(hidden_states.add(&attention)?)
    }

    pub(super) fn normalize_mlp_input(
        &self,
        layer: usize,
        hidden_states: &Tensor,
    ) -> Result<Tensor> {
        Ok(self
            .layer(layer)?
            .post_attention_norm
            .forward(hidden_states)?)
    }

    pub(super) fn forward_dense_mlp(&self, layer: usize, hidden_states: &Tensor) -> Result<Tensor> {
        self.layer(layer)?
            .dense_mlp
            .as_ref()
            .ok_or_else(|| {
                MoeError::Inference(format!(
                    "decoder layer {layer} is sparse and has no resident dense MLP"
                ))
            })?
            .forward(hidden_states)
    }

    pub(super) fn finish(&self, hidden_states: &Tensor) -> Result<Tensor> {
        Ok(self
            .lm_head
            .forward(&self.final_norm.forward(hidden_states)?)?
            .to_dtype(DType::F32)?)
    }

    pub(super) fn commit_cache(
        &self,
        cache: &mut Qwen3MoeKvCache,
        batch_size: usize,
        sequence_length: usize,
    ) {
        cache.commit_step(batch_size, sequence_length);
    }

    fn layer(&self, layer: usize) -> Result<&Qwen3MoeDenseDecoderLayer> {
        self.layers.get(layer).ok_or_else(|| {
            MoeError::Inference(format!("dense decoder layer {layer} is unavailable"))
        })
    }
}

pub(super) fn validate_generation_request(
    config: &Qwen3MoeConfig,
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

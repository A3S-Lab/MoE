use a3s_power::inference::RoutedExpertBatch;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::decoder::RotaryEmbedding;
use crate::{MoeError, Result};

use super::attention::Qwen3MoeAttention;
use super::mlp::Qwen3MoeMlp;
use super::{Qwen3MoeConfig, Qwen3MoeKvCache};

#[derive(Debug, Clone)]
struct Qwen3MoeDecoderLayer {
    input_norm: RmsNorm,
    attention: Qwen3MoeAttention,
    post_attention_norm: RmsNorm,
    mlp: Qwen3MoeMlp,
}

impl Qwen3MoeDecoderLayer {
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
            mlp: Qwen3MoeMlp::load(config, layer, builder.pp("mlp"))?,
        })
    }
}

/// Complete output from one Qwen3-MoE CPU prefill or decode step.
///
/// Dense MLP layers contain `None` at their corresponding router output and
/// route positions; sparse layers contain the exact Power routing contract.
#[derive(Debug)]
pub struct Qwen3MoeForwardOutput {
    pub logits: Tensor,
    pub layer_router_logits: Vec<Option<Tensor>>,
    pub layer_routes: Vec<Option<RoutedExpertBatch>>,
}

/// Fully resident F32 CPU correctness backend for Qwen3-MoE.
///
/// This backend implements the Hugging Face checkpoint layout directly,
/// including explicit attention head dimensions, per-head Q/K normalization,
/// fused expert projections, and mixed dense/sparse decoder schedules.
#[derive(Debug, Clone)]
pub struct Qwen3MoeCpuModel {
    config: Qwen3MoeConfig,
    embeddings: Embedding,
    layers: Vec<Qwen3MoeDecoderLayer>,
    final_norm: RmsNorm,
    lm_head: Linear,
}

impl Qwen3MoeCpuModel {
    pub fn load(config: Qwen3MoeConfig, builder: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        if !builder.device().is_cpu() {
            return Err(MoeError::InvalidConfig(
                "Qwen3-MoE CPU execution requires a CPU VarBuilder".to_string(),
            ));
        }
        if builder.dtype() != DType::F32 {
            return Err(MoeError::InvalidConfig(format!(
                "Qwen3-MoE CPU execution requires F32, found {:?}",
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
            layers.push(Qwen3MoeDecoderLayer::load(
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

    pub fn config(&self) -> &Qwen3MoeConfig {
        &self.config
    }

    pub fn new_cache(&self) -> Qwen3MoeKvCache {
        Qwen3MoeKvCache::new(
            self.config.num_hidden_layers,
            self.config.max_position_embeddings,
        )
    }

    pub fn forward(
        &self,
        token_ids: &Tensor,
        cache: &mut Qwen3MoeKvCache,
    ) -> Result<Qwen3MoeForwardOutput> {
        let (batch_size, sequence_length, position) = self.validate_step(token_ids, cache)?;

        // Cache updates are transactional: a failed layer cannot partially
        // advance a caller's session.
        let mut next_cache = cache.clone();
        let mut hidden_states = self.embeddings.forward(token_ids)?;
        let mut layer_router_logits = Vec::with_capacity(self.layers.len());
        let mut layer_routes = Vec::with_capacity(self.layers.len());
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let normalized = layer.input_norm.forward(&hidden_states)?;
            let attention = layer.attention.forward(
                &normalized,
                next_cache.layer_mut(layer_index)?,
                position,
            )?;
            hidden_states = hidden_states.add(&attention)?;
            let normalized = layer.post_attention_norm.forward(&hidden_states)?;
            let layer_number = u32::try_from(layer_index).map_err(|_| {
                MoeError::InvalidConfig("layer index exceeds the routing contract".to_string())
            })?;
            let mlp = layer.mlp.forward(layer_number, &normalized)?;
            hidden_states = hidden_states.add(&mlp.hidden_states)?;
            layer_router_logits.push(mlp.router_logits);
            layer_routes.push(mlp.routes);
        }
        let logits = self
            .lm_head
            .forward(&self.final_norm.forward(&hidden_states)?)?
            .to_dtype(DType::F32)?;
        next_cache.commit_step(batch_size, sequence_length);
        *cache = next_cache;
        Ok(Qwen3MoeForwardOutput {
            logits,
            layer_router_logits,
            layer_routes,
        })
    }

    pub fn generate_greedy(
        &self,
        prompt: &[u32],
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
    ) -> Result<Vec<u32>> {
        self.validate_generation_request(prompt, max_new_tokens)?;
        if max_new_tokens == 0 {
            return Ok(prompt.to_vec());
        }

        let mut generated = prompt.to_vec();
        let mut cache = self.new_cache();
        let mut input = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), self.device())?;
        for _ in 0..max_new_tokens {
            let output = self.forward(&input, &mut cache)?;
            let sequence_length = output.logits.dim(1)?;
            let next_token = output
                .logits
                .i((0, sequence_length - 1, ..))?
                .argmax(0)?
                .to_scalar::<u32>()?;
            generated.push(next_token);
            if eos_token_id == Some(next_token) {
                break;
            }
            input = Tensor::from_vec(vec![next_token], (1, 1), self.device())?;
        }
        Ok(generated)
    }

    fn device(&self) -> &Device {
        self.embeddings.embeddings().device()
    }

    fn validate_step(
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
                "token IDs and Qwen3-MoE weights must use the same device".to_string(),
            ));
        }
        let (batch_size, sequence_length) = token_ids.dims2()?;
        cache.validate_step(self.layers.len(), batch_size, sequence_length)?;
        Ok((batch_size, sequence_length, cache.position()))
    }

    fn validate_generation_request(&self, prompt: &[u32], max_new_tokens: usize) -> Result<()> {
        if prompt.is_empty() {
            return Err(MoeError::InvalidTensor(
                "greedy generation requires at least one prompt token".to_string(),
            ));
        }
        if prompt
            .iter()
            .any(|token| *token as usize >= self.config.vocab_size)
        {
            return Err(MoeError::InvalidTensor(
                "prompt contains a token outside the model vocabulary".to_string(),
            ));
        }
        let requested = prompt.len().checked_add(max_new_tokens).ok_or_else(|| {
            MoeError::InvalidTensor("generation token count overflowed".to_string())
        })?;
        if requested > self.config.max_position_embeddings {
            return Err(MoeError::InvalidTensor(format!(
                "generation requests {requested} positions, beyond maximum {}",
                self.config.max_position_embeddings
            )));
        }
        Ok(())
    }
}

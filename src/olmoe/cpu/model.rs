use a3s_power::inference::RoutedExpertBatch;
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::olmoe::OlmoeConfig;
use crate::{MoeError, Result};

use super::attention::{OlmoeAttention, OlmoeRotaryEmbedding};
use super::cache::OlmoeKvCache;
use super::experts::ResidentOlmoeMlp;

/// Complete output from one CPU prefill or decode step.
#[derive(Debug)]
pub struct OlmoeForwardOutput {
    pub logits: Tensor,
    pub layer_router_logits: Vec<Tensor>,
    pub layer_routes: Vec<RoutedExpertBatch>,
}

#[derive(Debug, Clone)]
struct OlmoeDecoderLayer {
    input_norm: RmsNorm,
    attention: OlmoeAttention,
    post_attention_norm: RmsNorm,
    mlp: ResidentOlmoeMlp,
}

impl OlmoeDecoderLayer {
    fn load(
        config: &OlmoeConfig,
        rotary: OlmoeRotaryEmbedding,
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
            mlp: ResidentOlmoeMlp::load(config, builder.pp("mlp"))?,
        })
    }

    fn forward(
        &self,
        layer: u32,
        hidden_states: &Tensor,
        cache: &mut super::cache::LayerKvCache,
        position: usize,
    ) -> Result<(Tensor, Tensor, RoutedExpertBatch)> {
        let residual = hidden_states;
        let normalized = self.input_norm.forward(hidden_states)?;
        let attended = self.attention.forward(&normalized, cache, position)?;
        let hidden_states = residual.add(&attended)?;

        let residual = &hidden_states;
        let normalized = self.post_attention_norm.forward(&hidden_states)?;
        let moe = self.mlp.forward(layer, &normalized)?;
        let hidden_states = residual.add(&moe.hidden_states)?;
        Ok((hidden_states, moe.router_logits, moe.routes))
    }
}

/// Fully resident F32 CPU correctness backend for OLMoE.
///
/// This implementation intentionally prioritizes a transparent reference path.
/// The streaming backend reuses its numerical tests but obtains expert weights
/// from Power's residency hierarchy instead of retaining all experts.
#[derive(Debug, Clone)]
pub struct OlmoeCpuModel {
    config: OlmoeConfig,
    embeddings: Embedding,
    layers: Vec<OlmoeDecoderLayer>,
    final_norm: RmsNorm,
    lm_head: Linear,
}

impl OlmoeCpuModel {
    pub fn load(config: OlmoeConfig, builder: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        if !builder.device().is_cpu() {
            return Err(MoeError::InvalidConfig(
                "OlmoeCpuModel requires a CPU VarBuilder".to_string(),
            ));
        }
        if builder.dtype() != DType::F32 {
            return Err(MoeError::InvalidConfig(format!(
                "OlmoeCpuModel requires F32 execution, found {:?}",
                builder.dtype()
            )));
        }
        let rotary = OlmoeRotaryEmbedding::new(&config, builder.dtype(), builder.device())?;
        let embeddings = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            builder.pp("model.embed_tokens"),
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            layers.push(OlmoeDecoderLayer::load(
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

    pub fn config(&self) -> &OlmoeConfig {
        &self.config
    }

    pub fn new_cache(&self) -> OlmoeKvCache {
        OlmoeKvCache::new(
            self.config.num_hidden_layers,
            self.config.max_position_embeddings,
        )
    }

    pub fn forward(
        &self,
        token_ids: &Tensor,
        cache: &mut OlmoeKvCache,
    ) -> Result<OlmoeForwardOutput> {
        if token_ids.dtype() != DType::U32 {
            return Err(MoeError::InvalidTensor(format!(
                "token IDs must use U32, found {:?}",
                token_ids.dtype()
            )));
        }
        if !token_ids.device().is_cpu() {
            return Err(MoeError::InvalidTensor(
                "OlmoeCpuModel token IDs must reside on CPU".to_string(),
            ));
        }
        let (batch_size, sequence_length) = token_ids.dims2()?;
        cache.validate_step(self.layers.len(), batch_size, sequence_length)?;
        let position = cache.position();

        // Work on a shallow tensor clone and commit only after all layers
        // succeed, so errors cannot leave a partially advanced session cache.
        let mut next_cache = cache.clone();
        let mut hidden_states = self.embeddings.forward(token_ids)?;
        let mut layer_router_logits = Vec::with_capacity(self.layers.len());
        let mut layer_routes = Vec::with_capacity(self.layers.len());
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let layer_index_u32 = u32::try_from(layer_index).map_err(|_| {
                MoeError::InvalidConfig("layer index exceeds the routing contract".to_string())
            })?;
            let (next_hidden_states, router_logits, routes) = layer.forward(
                layer_index_u32,
                &hidden_states,
                &mut next_cache.layers[layer_index],
                position,
            )?;
            hidden_states = next_hidden_states;
            layer_router_logits.push(router_logits);
            layer_routes.push(routes);
        }
        let hidden_states = self.final_norm.forward(&hidden_states)?;
        let logits = self.lm_head.forward(&hidden_states)?.to_dtype(DType::F32)?;
        next_cache.commit_step(batch_size, sequence_length);
        *cache = next_cache;
        Ok(OlmoeForwardOutput {
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
        if max_new_tokens == 0 {
            return Ok(prompt.to_vec());
        }

        let mut generated = prompt.to_vec();
        let mut cache = self.new_cache();
        let mut input = Tensor::from_vec(
            prompt.to_vec(),
            (1, prompt.len()),
            self.embeddings.embeddings().device(),
        )?;
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
            input = Tensor::from_vec(
                vec![next_token],
                (1, 1),
                self.embeddings.embeddings().device(),
            )?;
        }
        Ok(generated)
    }
}

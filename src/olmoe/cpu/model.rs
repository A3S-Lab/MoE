use a3s_power::inference::RoutedExpertBatch;
use candle_core::{IndexOp, Tensor};
use candle_nn::VarBuilder;

use crate::olmoe::OlmoeConfig;
use crate::{MoeError, Result};

use super::cache::OlmoeKvCache;
use super::dense::{validate_generation_request, OlmoeDenseModel};
use super::experts::ResidentOlmoeMlp;

/// Complete output from one CPU prefill or decode step.
#[derive(Debug)]
pub struct OlmoeForwardOutput {
    pub logits: Tensor,
    pub layer_router_logits: Vec<Tensor>,
    pub layer_routes: Vec<RoutedExpertBatch>,
}

/// Fully resident F32 CPU correctness backend for OLMoE.
///
/// This implementation intentionally prioritizes a transparent reference path.
/// The streaming backend shares its dense decoder and obtains only expert
/// weights from Power's residency hierarchy.
#[derive(Debug, Clone)]
pub struct OlmoeCpuModel {
    dense: OlmoeDenseModel,
    mlps: Vec<ResidentOlmoeMlp>,
}

impl OlmoeCpuModel {
    pub fn load(config: OlmoeConfig, builder: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let dense = OlmoeDenseModel::load(config.clone(), builder.clone())?;
        let mut mlps = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            mlps.push(ResidentOlmoeMlp::load(
                &config,
                builder.pp(format!("model.layers.{layer}.mlp")),
            )?);
        }
        Ok(Self { dense, mlps })
    }

    pub fn config(&self) -> &OlmoeConfig {
        self.dense.config()
    }

    pub fn new_cache(&self) -> OlmoeKvCache {
        self.dense.new_cache()
    }

    pub fn forward(
        &self,
        token_ids: &Tensor,
        cache: &mut OlmoeKvCache,
    ) -> Result<OlmoeForwardOutput> {
        let (batch_size, sequence_length, position) = self.dense.validate_step(token_ids, cache)?;

        // Work on a shallow tensor clone and commit only after all layers
        // succeed, so errors cannot leave a partially advanced session cache.
        let mut next_cache = cache.clone();
        let mut hidden_states = self.dense.embed(token_ids)?;
        let mut layer_router_logits = Vec::with_capacity(self.mlps.len());
        let mut layer_routes = Vec::with_capacity(self.mlps.len());
        for (layer_index, mlp) in self.mlps.iter().enumerate() {
            let layer = u32::try_from(layer_index).map_err(|_| {
                MoeError::InvalidConfig("layer index exceeds the routing contract".to_string())
            })?;
            let attended = self.dense.forward_attention(
                layer_index,
                &hidden_states,
                &mut next_cache,
                position,
            )?;
            let normalized = self.dense.normalize_mlp_input(layer_index, &attended)?;
            let moe = mlp.forward(layer, &normalized)?;
            hidden_states = attended.add(&moe.hidden_states)?;
            layer_router_logits.push(moe.router_logits);
            layer_routes.push(moe.routes);
        }
        let logits = self.dense.finish(&hidden_states)?;
        self.dense
            .commit_cache(&mut next_cache, batch_size, sequence_length);
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
        validate_generation_request(self.config(), prompt, max_new_tokens)?;
        if max_new_tokens == 0 {
            return Ok(prompt.to_vec());
        }

        let mut generated = prompt.to_vec();
        let mut cache = self.new_cache();
        let mut input = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), self.dense.device())?;
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
            input = Tensor::from_vec(vec![next_token], (1, 1), self.dense.device())?;
        }
        Ok(generated)
    }
}

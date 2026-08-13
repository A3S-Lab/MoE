use a3s_power::inference::RoutedExpertBatch;
use candle_core::{IndexOp, Tensor};
use candle_nn::VarBuilder;

use crate::{MoeError, Result};

use super::dense::{validate_generation_request, Qwen3MoeDenseModel};
use super::mlp::SparseMlp;
use super::{Qwen3MoeConfig, Qwen3MoeKvCache};

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
    dense: Qwen3MoeDenseModel,
    sparse_mlps: Vec<Option<SparseMlp>>,
}

impl Qwen3MoeCpuModel {
    pub fn load(config: Qwen3MoeConfig, builder: VarBuilder<'_>) -> Result<Self> {
        let dense = Qwen3MoeDenseModel::load(config.clone(), builder.clone())?;
        let mut sparse_mlps = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            sparse_mlps.push(if config.is_sparse_layer(layer) {
                Some(SparseMlp::load(
                    &config,
                    builder.pp(format!("model.layers.{layer}.mlp")),
                )?)
            } else {
                None
            });
        }
        Ok(Self { dense, sparse_mlps })
    }

    pub fn config(&self) -> &Qwen3MoeConfig {
        self.dense.config()
    }

    pub fn new_cache(&self) -> Qwen3MoeKvCache {
        self.dense.new_cache()
    }

    pub fn forward(
        &self,
        token_ids: &Tensor,
        cache: &mut Qwen3MoeKvCache,
    ) -> Result<Qwen3MoeForwardOutput> {
        let (batch_size, sequence_length, position) = self.dense.validate_step(token_ids, cache)?;

        // Cache updates are transactional: a failed layer cannot partially
        // advance a caller's session.
        let mut next_cache = cache.clone();
        let mut hidden_states = self.dense.embed(token_ids)?;
        let mut layer_router_logits = Vec::with_capacity(self.sparse_mlps.len());
        let mut layer_routes = Vec::with_capacity(self.sparse_mlps.len());
        for (layer, sparse_mlp) in self.sparse_mlps.iter().enumerate() {
            let attended =
                self.dense
                    .forward_attention(layer, &hidden_states, &mut next_cache, position)?;
            let normalized = self.dense.normalize_mlp_input(layer, &attended)?;
            if let Some(sparse_mlp) = sparse_mlp {
                let layer_number = u32::try_from(layer).map_err(|_| {
                    MoeError::InvalidConfig("layer index exceeds the routing contract".to_string())
                })?;
                let mlp = sparse_mlp.forward(layer_number, &normalized)?;
                hidden_states = attended.add(&mlp.hidden_states)?;
                layer_router_logits.push(Some(mlp.router_logits));
                layer_routes.push(Some(mlp.routes));
            } else {
                hidden_states = attended.add(&self.dense.forward_dense_mlp(layer, &normalized)?)?;
                layer_router_logits.push(None);
                layer_routes.push(None);
            }
        }
        let logits = self.dense.finish(&hidden_states)?;
        self.dense
            .commit_cache(&mut next_cache, batch_size, sequence_length);
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

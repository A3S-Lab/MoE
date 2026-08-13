use a3s_power::inference::{
    EmbeddedRuntime, ExecutionPermit, PlacementTelemetry, RoutedExpertBatch,
    StagedWeightBatchReport, WeightHierarchy,
};
use candle_core::{IndexOp, Tensor};
use candle_nn::VarBuilder;
use tokio_util::sync::CancellationToken;

use crate::{MoeError, PowerStreamingMlp, Result};

use super::dense::{validate_generation_request, Qwen3MoeDenseModel};
use super::{Qwen3MoeConfig, Qwen3MoeKvCache};

/// Complete output from one Power-backed Qwen3-MoE prefill or decode step.
#[derive(Debug)]
pub struct Qwen3MoeStreamingForwardOutput {
    pub logits: Tensor,
    pub layer_router_logits: Vec<Option<Tensor>>,
    pub layer_routes: Vec<Option<RoutedExpertBatch>>,
    pub layer_staging: Vec<Option<StagedWeightBatchReport>>,
}

/// Qwen3-MoE decoder with resident dense weights and one Power-owned expert
/// residency hierarchy shared by every sparse layer.
#[derive(Debug, Clone)]
pub struct Qwen3MoeStreamingModel {
    pub(super) dense: Qwen3MoeDenseModel,
    pub(super) sparse_mlps: Vec<Option<PowerStreamingMlp>>,
    hierarchy: WeightHierarchy,
}

impl Qwen3MoeStreamingModel {
    pub fn load(
        config: Qwen3MoeConfig,
        builder: VarBuilder<'_>,
        hierarchy: WeightHierarchy,
    ) -> Result<Self> {
        config.validate()?;
        if !builder
            .device()
            .same_device(hierarchy.runtime().device().tensor_device())
        {
            return Err(MoeError::InvalidConfig(
                "streaming dense weights and Power runtime must use the same device".to_string(),
            ));
        }
        let dense = Qwen3MoeDenseModel::load(config.clone(), builder.clone())?;
        let moe = config.moe_config()?;
        let mut sparse_mlps = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            sparse_mlps.push(if config.is_sparse_layer(layer) {
                let layer_number = u32::try_from(layer).map_err(|_| {
                    MoeError::InvalidConfig("layer index exceeds the routing contract".to_string())
                })?;
                let router_weight = builder.get(
                    (config.num_experts, config.hidden_size),
                    &format!("model.layers.{layer}.mlp.gate.weight"),
                )?;
                Some(PowerStreamingMlp::new(
                    layer_number,
                    moe,
                    router_weight,
                    hierarchy.clone(),
                )?)
            } else {
                None
            });
        }
        Ok(Self {
            dense,
            sparse_mlps,
            hierarchy,
        })
    }

    pub fn config(&self) -> &Qwen3MoeConfig {
        self.dense.config()
    }

    pub fn runtime(&self) -> &EmbeddedRuntime {
        self.hierarchy.runtime()
    }

    pub fn hierarchy(&self) -> &WeightHierarchy {
        &self.hierarchy
    }

    pub fn telemetry(&self) -> PlacementTelemetry {
        self.hierarchy.telemetry()
    }

    pub fn new_cache(&self) -> Qwen3MoeKvCache {
        self.dense.new_cache()
    }

    /// Executes one transactional prefill or decode step under a caller-owned
    /// permit, streaming experts only for layers declared sparse by config.
    pub async fn forward(
        &self,
        token_ids: &Tensor,
        cache: &mut Qwen3MoeKvCache,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> Result<Qwen3MoeStreamingForwardOutput> {
        let (batch_size, sequence_length, position) = self.dense.validate_step(token_ids, cache)?;
        if cancellation.is_cancelled() {
            return Err(MoeError::Power(
                a3s_power::error::PowerError::InferenceCancelled,
            ));
        }

        let mut next_cache = cache.clone();
        let mut hidden_states = self.dense.embed(token_ids)?;
        let mut layer_router_logits = Vec::with_capacity(self.sparse_mlps.len());
        let mut layer_routes = Vec::with_capacity(self.sparse_mlps.len());
        let mut layer_staging = Vec::with_capacity(self.sparse_mlps.len());
        for (layer, sparse_mlp) in self.sparse_mlps.iter().enumerate() {
            let attended =
                self.dense
                    .forward_attention(layer, &hidden_states, &mut next_cache, position)?;
            let normalized = self.dense.normalize_mlp_input(layer, &attended)?;
            if let Some(sparse_mlp) = sparse_mlp {
                let moe = sparse_mlp
                    .forward(&normalized, permit, cancellation)
                    .await?;
                hidden_states = attended.add(&moe.hidden_states)?;
                layer_router_logits.push(Some(moe.router_logits));
                layer_routes.push(Some(moe.routes));
                layer_staging.push(Some(moe.staging));
            } else {
                hidden_states = attended.add(&self.dense.forward_dense_mlp(layer, &normalized)?)?;
                layer_router_logits.push(None);
                layer_routes.push(None);
                layer_staging.push(None);
            }
        }
        let logits = self.dense.finish(&hidden_states)?;
        self.dense
            .commit_cache(&mut next_cache, batch_size, sequence_length);
        *cache = next_cache;
        Ok(Qwen3MoeStreamingForwardOutput {
            logits,
            layer_router_logits,
            layer_routes,
            layer_staging,
        })
    }

    /// Greedy generation holds one Power admission permit for the complete
    /// request and reuses the transactional KV cache across decode steps.
    pub async fn generate_greedy(
        &self,
        prompt: &[u32],
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        validate_generation_request(self.config(), prompt, max_new_tokens)?;
        if max_new_tokens == 0 {
            return Ok(prompt.to_vec());
        }
        let permit = self.runtime().begin_wait(cancellation).await?;
        let mut generated = prompt.to_vec();
        let mut cache = self.new_cache();
        let mut input = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), self.dense.device())?;
        for _ in 0..max_new_tokens {
            let output = self
                .forward(&input, &mut cache, &permit, cancellation)
                .await?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_streaming_model_types_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Qwen3MoeStreamingModel>();
        assert_send_sync::<Qwen3MoeStreamingForwardOutput>();
    }
}

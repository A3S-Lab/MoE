use a3s_power::inference::{
    EmbeddedRuntime, ExecutionPermit, PlacementTelemetry, RoutedExpertBatch,
    StagedWeightBatchReport, WeightHierarchy,
};
use candle_core::{IndexOp, Tensor};
use candle_nn::VarBuilder;
use tokio_util::sync::CancellationToken;

use crate::{MoeError, PowerStreamingMlp, Result};

use super::dense::{validate_generation_request, Qwen36DenseModel};
use super::{Qwen36MoeCache, Qwen36MoeConfig};

/// Complete output from one Power-backed Qwen3.6 prefill or decode step.
#[derive(Debug)]
pub struct Qwen36MoeStreamingForwardOutput {
    pub logits: Tensor,
    pub layer_router_logits: Vec<Tensor>,
    pub layer_routes: Vec<RoutedExpertBatch>,
    pub layer_staging: Vec<StagedWeightBatchReport>,
}

/// Qwen3.6 text decoder with resident non-routed weights and one Power-owned
/// expert hierarchy shared by all 40 routed layers.
#[derive(Debug, Clone)]
pub struct Qwen36MoeStreamingModel {
    pub(super) dense: Qwen36DenseModel,
    pub(super) routed_mlps: Vec<PowerStreamingMlp>,
    hierarchy: WeightHierarchy,
}

impl Qwen36MoeStreamingModel {
    pub fn load(
        config: Qwen36MoeConfig,
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
        let dense = Qwen36DenseModel::load(config.clone(), builder.clone())?;
        let moe = config.moe_config()?;
        let mut routed_mlps = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            let layer_number = u32::try_from(layer).map_err(|_| {
                MoeError::InvalidConfig("layer index exceeds the routing contract".to_string())
            })?;
            let router_weight = builder.get(
                (config.num_experts, config.hidden_size),
                &format!("model.language_model.layers.{layer}.mlp.gate.weight"),
            )?;
            routed_mlps.push(PowerStreamingMlp::new(
                layer_number,
                moe,
                router_weight,
                hierarchy.clone(),
            )?);
        }
        Ok(Self {
            dense,
            routed_mlps,
            hierarchy,
        })
    }

    pub fn config(&self) -> &Qwen36MoeConfig {
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

    pub fn new_cache(&self) -> Qwen36MoeCache {
        self.dense.new_cache()
    }

    pub async fn forward(
        &self,
        token_ids: &Tensor,
        cache: &mut Qwen36MoeCache,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> Result<Qwen36MoeStreamingForwardOutput> {
        let (batch_size, sequence_length, position) = self.dense.validate_step(token_ids, cache)?;
        check_cancellation(cancellation)?;
        let mut next_cache = cache.clone();
        let mut hidden_states = self.dense.embed(token_ids)?;
        let mut layer_router_logits = Vec::with_capacity(self.routed_mlps.len());
        let mut layer_routes = Vec::with_capacity(self.routed_mlps.len());
        let mut layer_staging = Vec::with_capacity(self.routed_mlps.len());
        for (layer, routed_mlp) in self.routed_mlps.iter().enumerate() {
            check_cancellation(cancellation)?;
            let attended =
                self.dense
                    .forward_mixer(layer, &hidden_states, &mut next_cache, position)?;
            let normalized = self.dense.normalize_mlp_input(layer, &attended)?;
            let routed = routed_mlp
                .forward(&normalized, permit, cancellation)
                .await?;
            let shared = self.dense.forward_shared_expert(layer, &normalized)?;
            hidden_states = attended.add(&routed.hidden_states.add(&shared)?)?;
            layer_router_logits.push(routed.router_logits);
            layer_routes.push(routed.routes);
            layer_staging.push(routed.staging);
        }
        let logits = self.dense.finish(&hidden_states)?;
        self.dense
            .commit_cache(&mut next_cache, batch_size, sequence_length);
        *cache = next_cache;
        Ok(Qwen36MoeStreamingForwardOutput {
            logits,
            layer_router_logits,
            layer_routes,
            layer_staging,
        })
    }

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

fn check_cancellation(cancellation: &CancellationToken) -> Result<()> {
    if cancellation.is_cancelled() {
        Err(MoeError::Power(
            a3s_power::error::PowerError::InferenceCancelled,
        ))
    } else {
        Ok(())
    }
}

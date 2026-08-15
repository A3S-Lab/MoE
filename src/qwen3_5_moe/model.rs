use a3s_power::inference::RoutedExpertBatch;
use candle_core::{IndexOp, Tensor};
use candle_nn::VarBuilder;

use crate::{MoeError, Result};

use super::dense::{validate_generation_request, Qwen36DenseModel};
use super::mlp::Qwen36ResidentSparseMlp;
use super::{Qwen36MoeCache, Qwen36MoeConfig};

/// Complete output from one resident Qwen3.6 prefill or decode step.
#[derive(Debug)]
pub struct Qwen36MoeForwardOutput {
    pub logits: Tensor,
    pub layer_router_logits: Vec<Tensor>,
    pub layer_routes: Vec<RoutedExpertBatch>,
}

/// Fully resident F32 CPU correctness backend for Qwen3.6-35B-A3B text.
#[derive(Debug, Clone)]
pub struct Qwen36MoeCpuModel {
    dense: Qwen36DenseModel,
    routed_mlps: Vec<Qwen36ResidentSparseMlp>,
}

impl Qwen36MoeCpuModel {
    pub fn load(config: Qwen36MoeConfig, builder: VarBuilder<'_>) -> Result<Self> {
        if !builder.device().is_cpu() {
            return Err(MoeError::InvalidConfig(
                "the resident Qwen3.6 correctness backend requires a CPU VarBuilder".to_string(),
            ));
        }
        let dense = Qwen36DenseModel::load(config.clone(), builder.clone())?;
        let mut routed_mlps = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            routed_mlps.push(Qwen36ResidentSparseMlp::load(
                &config,
                builder.pp(format!("model.language_model.layers.{layer}.mlp")),
            )?);
        }
        Ok(Self { dense, routed_mlps })
    }

    pub fn config(&self) -> &Qwen36MoeConfig {
        self.dense.config()
    }

    pub fn new_cache(&self) -> Qwen36MoeCache {
        self.dense.new_cache()
    }

    pub fn forward(
        &self,
        token_ids: &Tensor,
        cache: &mut Qwen36MoeCache,
    ) -> Result<Qwen36MoeForwardOutput> {
        let (batch_size, sequence_length, position) = self.dense.validate_step(token_ids, cache)?;
        let mut next_cache = cache.clone();
        let mut hidden_states = self.dense.embed(token_ids)?;
        let mut layer_router_logits = Vec::with_capacity(self.routed_mlps.len());
        let mut layer_routes = Vec::with_capacity(self.routed_mlps.len());
        for (layer, routed_mlp) in self.routed_mlps.iter().enumerate() {
            let attended =
                self.dense
                    .forward_mixer(layer, &hidden_states, &mut next_cache, position)?;
            let normalized = self.dense.normalize_mlp_input(layer, &attended)?;
            let layer_number = u32::try_from(layer).map_err(|_| {
                MoeError::InvalidConfig("layer index exceeds the routing contract".to_string())
            })?;
            let routed = routed_mlp.forward(layer_number, &normalized)?;
            let shared = self.dense.forward_shared_expert(layer, &normalized)?;
            let combined_mlp = routed.hidden_states.add(&shared)?;
            hidden_states = attended.add(&combined_mlp)?;
            layer_router_logits.push(routed.router_logits);
            layer_routes.push(routed.routes);
        }
        let logits = self.dense.finish(&hidden_states)?;
        self.dense
            .commit_cache(&mut next_cache, batch_size, sequence_length);
        *cache = next_cache;
        Ok(Qwen36MoeForwardOutput {
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

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use candle_core::Device;

    use super::*;
    use crate::qwen3_5_moe::test_support::{tiny_builder, tiny_config};

    #[test]
    fn full_prefill_matches_incremental_linear_and_attention_state() {
        let config = tiny_config();
        let model = Qwen36MoeCpuModel::load(config, tiny_builder(&tiny_config())).unwrap();
        let tokens = [1_u32, 4, 2];
        let mut prefill_cache = model.new_cache();
        let prefill = model
            .forward(
                &Tensor::from_slice(&tokens, (1, tokens.len()), &Device::Cpu).unwrap(),
                &mut prefill_cache,
            )
            .unwrap();

        let mut decode_cache = model.new_cache();
        let mut incremental_logits = Vec::new();
        let mut incremental_routes = Vec::new();
        for token in tokens {
            let step = model
                .forward(
                    &Tensor::from_slice(&[token], (1, 1), &Device::Cpu).unwrap(),
                    &mut decode_cache,
                )
                .unwrap();
            incremental_logits.push(step.logits.flatten_all().unwrap().to_vec1::<f32>().unwrap());
            incremental_routes.push(step.layer_routes);
        }
        let prefill_logits = prefill.logits.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
        for (actual, expected) in incremental_logits.iter().zip(&prefill_logits) {
            for (actual, expected) in actual.iter().zip(expected) {
                assert_abs_diff_eq!(actual, expected, epsilon = 2e-5);
            }
        }
        for layer in 0..model.config().num_hidden_layers {
            for (position, step) in incremental_routes.iter().enumerate() {
                let expected = &prefill.layer_routes[layer].selections()[position];
                let actual = &step[layer].selections()[0];
                assert_eq!(
                    actual.iter().map(|route| route.expert).collect::<Vec<_>>(),
                    expected
                        .iter()
                        .map(|route| route.expert)
                        .collect::<Vec<_>>()
                );
                for (actual, expected) in actual.iter().zip(expected) {
                    assert_abs_diff_eq!(actual.weight, expected.weight, epsilon = 2e-5);
                }
            }
        }
        assert_eq!(prefill_cache.position(), tokens.len());
        assert_eq!(decode_cache.position(), tokens.len());
        assert_eq!(
            prefill_cache.resident_bytes().unwrap(),
            decode_cache.resident_bytes().unwrap()
        );
    }

    #[test]
    fn failed_step_does_not_advance_the_callers_cache() {
        let config = tiny_config();
        let model = Qwen36MoeCpuModel::load(config, tiny_builder(&tiny_config())).unwrap();
        let mut cache = model.new_cache();
        let invalid = Tensor::from_slice(&[u32::MAX], (1, 1), &Device::Cpu).unwrap();
        assert!(model.forward(&invalid, &mut cache).is_err());
        assert_eq!(cache.position(), 0);
        assert_eq!(cache.resident_bytes().unwrap(), 0);
    }
}

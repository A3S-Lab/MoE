use crate::{Matrix, MoeError, Result};

use super::{GatedExpertWeights, MoeLayerConfig, TopKRouter, TopKRouterOutput};

/// Output of one exact sparse feed-forward layer.
#[derive(Debug, Clone)]
pub struct SparseMoeOutput {
    pub hidden_states: Matrix,
    pub routing: TopKRouterOutput,
}

/// CPU correctness implementation of a full-softmax routed sparse layer.
#[derive(Debug, Clone)]
pub struct SparseMoeLayer {
    config: MoeLayerConfig,
    router: TopKRouter,
    experts: Vec<GatedExpertWeights>,
}

impl SparseMoeLayer {
    pub fn new(
        config: MoeLayerConfig,
        router_weight: Matrix,
        experts: Vec<GatedExpertWeights>,
    ) -> Result<Self> {
        config.validate()?;
        if experts.len() != config.num_experts {
            return Err(MoeError::InvalidTensor(format!(
                "sparse layer requires {} experts, found {}",
                config.num_experts,
                experts.len()
            )));
        }
        let router = TopKRouter::new(config, router_weight)?;
        Ok(Self {
            config,
            router,
            experts,
        })
    }

    pub fn forward(&self, layer: u32, hidden_states: &Matrix) -> Result<SparseMoeOutput> {
        let routing = self.router.forward(layer, hidden_states)?;
        let mut output = Matrix::zeros(hidden_states.rows(), self.config.hidden_size)?;

        // Power returns the canonical expert union in ascending order. The
        // reference families execute active experts in that same order.
        for expert in routing.routes.experts() {
            let expert_index = *expert as usize;
            let weights = self.experts.get(expert_index).ok_or_else(|| {
                MoeError::Inference(format!("routed expert {expert_index} is unavailable"))
            })?;
            for assignment in routing.routes.assignments(*expert) {
                let expert_output = weights.forward_row(hidden_states.row(assignment.position)?)?;
                let destination = output.row_mut(assignment.position)?;
                for (destination, value) in destination.iter_mut().zip(expert_output) {
                    *destination += value * assignment.weight;
                    if !destination.is_finite() {
                        return Err(MoeError::Inference(format!(
                            "expert reduction became non-finite at position {}",
                            assignment.position
                        )));
                    }
                }
            }
        }

        Ok(SparseMoeOutput {
            hidden_states: output,
            routing,
        })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn applies_router_weight_before_expert_reduction() {
        let config = MoeLayerConfig {
            hidden_size: 1,
            intermediate_size: 1,
            num_experts: 2,
            top_k: 1,
            normalize_top_k: false,
        };
        let expert = |down| {
            GatedExpertWeights::new(
                config,
                Matrix::new(2, 1, vec![1.0, 1.0]).unwrap(),
                Matrix::new(1, 1, vec![down]).unwrap(),
            )
            .unwrap()
        };
        let layer = SparseMoeLayer::new(
            config,
            Matrix::new(2, 1, vec![1.0, -1.0]).unwrap(),
            vec![expert(2.0), expert(100.0)],
        )
        .unwrap();
        let output = layer
            .forward(0, &Matrix::new(1, 1, vec![1.0]).unwrap())
            .unwrap();
        let route_weight = output.routing.routes.selections()[0][0].weight;
        let expected = 2.0 * (1.0_f32 / (1.0 + (-1.0_f32).exp())) * route_weight;
        assert_abs_diff_eq!(output.hidden_states.values()[0], expected, epsilon = 1e-6);
    }
}

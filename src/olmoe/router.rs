use a3s_power::inference::{RoutedExpert, RoutedExpertBatch};

use crate::{Matrix, MoeError, Result};

use super::OlmoeMoeConfig;

/// Exact OLMoE full-softmax top-k router.
#[derive(Debug, Clone)]
pub struct OlmoeRouter {
    config: OlmoeMoeConfig,
    weight: Matrix,
}

/// Router logits and the validated, unmodified Power routing batch.
#[derive(Debug, Clone)]
pub struct OlmoeRouterOutput {
    pub logits: Matrix,
    pub routes: RoutedExpertBatch,
}

impl OlmoeRouter {
    pub fn new(config: OlmoeMoeConfig, weight: Matrix) -> Result<Self> {
        config.validate()?;
        if weight.rows() != config.num_experts || weight.columns() != config.hidden_size {
            return Err(MoeError::InvalidTensor(format!(
                "router weight must have shape [{}, {}], found [{}, {}]",
                config.num_experts,
                config.hidden_size,
                weight.rows(),
                weight.columns()
            )));
        }
        Ok(Self { config, weight })
    }

    pub fn forward(&self, layer: u32, hidden_states: &Matrix) -> Result<OlmoeRouterOutput> {
        if hidden_states.columns() != self.config.hidden_size {
            return Err(MoeError::InvalidTensor(format!(
                "router input width must be {}, found {}",
                self.config.hidden_size,
                hidden_states.columns()
            )));
        }

        let logit_count = hidden_states
            .rows()
            .checked_mul(self.config.num_experts)
            .ok_or_else(|| MoeError::Inference("router logit count overflowed".to_string()))?;
        let mut logits = Vec::with_capacity(logit_count);
        let mut selections = Vec::with_capacity(hidden_states.rows());
        for position in 0..hidden_states.rows() {
            let input = hidden_states.row(position)?;
            let mut row_logits = Vec::with_capacity(self.config.num_experts);
            for expert in 0..self.config.num_experts {
                let value = dot(self.weight.row(expert)?, input);
                if !value.is_finite() {
                    return Err(MoeError::Inference(format!(
                        "router produced a non-finite logit at position {position}, expert {expert}"
                    )));
                }
                row_logits.push(value);
            }
            let probabilities = stable_softmax(&row_logits)?;
            let mut indices: Vec<usize> = (0..self.config.num_experts).collect();
            indices.sort_unstable_by(|left, right| {
                probabilities[*right]
                    .total_cmp(&probabilities[*left])
                    .then_with(|| left.cmp(right))
            });
            indices.truncate(self.config.top_k);

            let normalization = if self.config.normalize_top_k {
                let sum = indices
                    .iter()
                    .map(|index| probabilities[*index])
                    .sum::<f32>();
                if !sum.is_finite() || sum <= 0.0 {
                    return Err(MoeError::Inference(format!(
                        "top-k probability normalization failed at position {position}"
                    )));
                }
                sum
            } else {
                1.0
            };
            let routed = indices
                .into_iter()
                .map(|expert| RoutedExpert {
                    expert: expert as u32,
                    weight: probabilities[expert] / normalization,
                })
                .collect();
            logits.extend_from_slice(&row_logits);
            selections.push(routed);
        }

        let logits = Matrix::new(hidden_states.rows(), self.config.num_experts, logits)?;
        let routes = RoutedExpertBatch::new(
            layer,
            selections,
            self.config.num_experts as u32,
            self.config.top_k,
        )?;
        Ok(OlmoeRouterOutput { logits, routes })
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .fold(0.0, |sum, (left, right)| sum + left * right)
}

fn stable_softmax(logits: &[f32]) -> Result<Vec<f32>> {
    let max = logits
        .iter()
        .copied()
        .reduce(f32::max)
        .ok_or_else(|| MoeError::Inference("router received an empty logit row".to_string()))?;
    let mut probabilities: Vec<f32> = logits.iter().map(|value| (*value - max).exp()).collect();
    let sum = probabilities.iter().sum::<f32>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(MoeError::Inference(
            "router softmax normalization was non-finite".to_string(),
        ));
    }
    for probability in &mut probabilities {
        *probability /= sum;
    }
    Ok(probabilities)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    fn config(normalize_top_k: bool) -> OlmoeMoeConfig {
        OlmoeMoeConfig {
            hidden_size: 2,
            intermediate_size: 2,
            num_experts: 3,
            top_k: 2,
            normalize_top_k,
        }
    }

    #[test]
    fn olmoe_default_keeps_full_softmax_probabilities() {
        let router = OlmoeRouter::new(
            config(false),
            Matrix::new(3, 2, vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0]).unwrap(),
        )
        .unwrap();
        let output = router
            .forward(4, &Matrix::new(1, 2, vec![1.0, 0.0]).unwrap())
            .unwrap();
        let selections = &output.routes.selections()[0];

        assert_eq!(selections[0].expert, 0);
        assert_eq!(selections[1].expert, 1);
        assert!(selections.iter().map(|route| route.weight).sum::<f32>() < 1.0);
        assert_abs_diff_eq!(selections[0].weight, 0.665_240_94, epsilon = 1e-6);
        assert_abs_diff_eq!(selections[1].weight, 0.244_728_48, epsilon = 1e-6);
    }

    #[test]
    fn optional_top_k_normalization_changes_only_selected_weights() {
        let weight = Matrix::new(3, 2, vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0]).unwrap();
        let input = Matrix::new(1, 2, vec![1.0, 0.0]).unwrap();
        let unnormalized = OlmoeRouter::new(config(false), weight.clone())
            .unwrap()
            .forward(0, &input)
            .unwrap();
        let normalized = OlmoeRouter::new(config(true), weight)
            .unwrap()
            .forward(0, &input)
            .unwrap();

        assert_eq!(unnormalized.logits, normalized.logits);
        assert_abs_diff_eq!(
            normalized.routes.selections()[0]
                .iter()
                .map(|route| route.weight)
                .sum::<f32>(),
            1.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn equal_probabilities_use_expert_index_as_deterministic_tie_breaker() {
        let router =
            OlmoeRouter::new(config(false), Matrix::new(3, 2, vec![0.0; 6]).unwrap()).unwrap();
        let output = router
            .forward(0, &Matrix::new(1, 2, vec![1.0, 2.0]).unwrap())
            .unwrap();
        let experts: Vec<u32> = output.routes.selections()[0]
            .iter()
            .map(|route| route.expert)
            .collect();
        assert_eq!(experts, [0, 1]);
    }
}

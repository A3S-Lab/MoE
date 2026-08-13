use crate::{Matrix, MoeError, Result};

use super::MoeLayerConfig;

/// One SiLU-gated expert with fused gate rows followed by up rows.
#[derive(Debug, Clone)]
pub struct GatedExpertWeights {
    gate_up: Matrix,
    down: Matrix,
    hidden_size: usize,
    intermediate_size: usize,
}

impl GatedExpertWeights {
    pub fn new(config: MoeLayerConfig, gate_up: Matrix, down: Matrix) -> Result<Self> {
        config.validate()?;
        let expected_gate_up_rows = config
            .intermediate_size
            .checked_mul(2)
            .ok_or_else(|| MoeError::InvalidConfig("gate/up row count overflowed".to_string()))?;
        if gate_up.rows() != expected_gate_up_rows || gate_up.columns() != config.hidden_size {
            return Err(MoeError::InvalidTensor(format!(
                "expert gate_up weight must have shape [{expected_gate_up_rows}, {}], found [{}, {}]",
                config.hidden_size,
                gate_up.rows(),
                gate_up.columns()
            )));
        }
        if down.rows() != config.hidden_size || down.columns() != config.intermediate_size {
            return Err(MoeError::InvalidTensor(format!(
                "expert down weight must have shape [{}, {}], found [{}, {}]",
                config.hidden_size,
                config.intermediate_size,
                down.rows(),
                down.columns()
            )));
        }
        Ok(Self {
            gate_up,
            down,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
        })
    }

    pub(crate) fn forward_row(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.hidden_size {
            return Err(MoeError::InvalidTensor(format!(
                "expert input width must be {}, found {}",
                self.hidden_size,
                input.len()
            )));
        }

        let mut intermediate = Vec::with_capacity(self.intermediate_size);
        for row in 0..self.intermediate_size {
            let gate = dot(self.gate_up.row(row)?, input);
            let up = dot(self.gate_up.row(row + self.intermediate_size)?, input);
            let value = silu(gate) * up;
            if !value.is_finite() {
                return Err(MoeError::Inference(format!(
                    "expert produced a non-finite intermediate value at row {row}"
                )));
            }
            intermediate.push(value);
        }

        let mut output = Vec::with_capacity(self.hidden_size);
        for row in 0..self.hidden_size {
            let value = dot(self.down.row(row)?, &intermediate);
            if !value.is_finite() {
                return Err(MoeError::Inference(format!(
                    "expert produced a non-finite output value at row {row}"
                )));
            }
            output.push(value);
        }
        Ok(output)
    }
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .fold(0.0, |sum, (left, right)| sum + left * right)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn fused_gate_up_order_matches_the_reference_equation() {
        let config = MoeLayerConfig {
            hidden_size: 2,
            intermediate_size: 1,
            num_experts: 1,
            top_k: 1,
            normalize_top_k: false,
        };
        let expert = GatedExpertWeights::new(
            config,
            Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 2.0]).unwrap(),
            Matrix::new(2, 1, vec![3.0, -1.0]).unwrap(),
        )
        .unwrap();

        let output = expert.forward_row(&[1.0, 2.0]).unwrap();
        let hidden = (1.0_f32 / (1.0 + (-1.0_f32).exp())) * 4.0;
        assert_abs_diff_eq!(output[0], hidden * 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(output[1], -hidden, epsilon = 1e-6);
    }
}

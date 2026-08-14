use candle_core::{DType, Tensor};
use candle_nn::{Linear, Module, RmsNorm, VarBuilder};

use crate::{MoeError, Result};

use super::Qwen36MoeConfig;

const L2_NORM_EPSILON: f32 = 1e-6;

#[derive(Debug, Clone)]
pub(super) struct Qwen36GatedDeltaNet {
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    out_proj: Linear,
    norm: RmsNorm,
    conv_weight: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    hidden_size: usize,
    num_key_heads: usize,
    num_value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    conv_kernel_size: usize,
}

impl Qwen36GatedDeltaNet {
    pub(super) fn load(config: &Qwen36MoeConfig, builder: VarBuilder<'_>) -> Result<Self> {
        let key_dim = config
            .linear_num_key_heads
            .checked_mul(config.linear_key_head_dim)
            .ok_or_else(|| MoeError::InvalidConfig("linear key width overflowed".to_string()))?;
        let value_dim = config
            .linear_num_value_heads
            .checked_mul(config.linear_value_head_dim)
            .ok_or_else(|| MoeError::InvalidConfig("linear value width overflowed".to_string()))?;
        let conv_dim = key_dim
            .checked_mul(2)
            .and_then(|value| value.checked_add(value_dim))
            .ok_or_else(|| {
                MoeError::InvalidConfig("linear convolution width overflowed".to_string())
            })?;
        let conv_weight = builder
            .get(
                (conv_dim, 1, config.linear_conv_kernel_dim),
                "conv1d.weight",
            )?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let a_log = builder
            .get(config.linear_num_value_heads, "A_log")?
            .to_vec1::<f32>()?;
        let dt_bias = builder
            .get(config.linear_num_value_heads, "dt_bias")?
            .to_vec1::<f32>()?;
        Ok(Self {
            in_proj_qkv: candle_nn::linear_no_bias(
                config.hidden_size,
                conv_dim,
                builder.pp("in_proj_qkv"),
            )?,
            in_proj_z: candle_nn::linear_no_bias(
                config.hidden_size,
                value_dim,
                builder.pp("in_proj_z"),
            )?,
            in_proj_b: candle_nn::linear_no_bias(
                config.hidden_size,
                config.linear_num_value_heads,
                builder.pp("in_proj_b"),
            )?,
            in_proj_a: candle_nn::linear_no_bias(
                config.hidden_size,
                config.linear_num_value_heads,
                builder.pp("in_proj_a"),
            )?,
            out_proj: candle_nn::linear_no_bias(
                value_dim,
                config.hidden_size,
                builder.pp("out_proj"),
            )?,
            norm: RmsNorm::new(
                builder.get(config.linear_value_head_dim, "norm.weight")?,
                config.rms_norm_eps,
            ),
            conv_weight,
            a_log,
            dt_bias,
            hidden_size: config.hidden_size,
            num_key_heads: config.linear_num_key_heads,
            num_value_heads: config.linear_num_value_heads,
            key_head_dim: config.linear_key_head_dim,
            value_head_dim: config.linear_value_head_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel_size: config.linear_conv_kernel_dim,
        })
    }

    pub(super) fn forward(
        &self,
        hidden_states: &Tensor,
        conv_state: &mut Option<Vec<f32>>,
        recurrent_state: &mut Option<Vec<f32>>,
    ) -> Result<Tensor> {
        if hidden_states.dtype() != DType::F32 || !hidden_states.device().is_cpu() {
            return Err(MoeError::InvalidTensor(
                "Qwen3.6 Gated DeltaNet requires F32 CPU hidden states".to_string(),
            ));
        }
        let (batch_size, sequence_length, hidden_size) = hidden_states.dims3()?;
        if hidden_size != self.hidden_size {
            return Err(MoeError::InvalidTensor(format!(
                "linear-attention input width must be {}, found {hidden_size}",
                self.hidden_size
            )));
        }

        let mixed = self
            .in_proj_qkv
            .forward(hidden_states)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let z = self.in_proj_z.forward(hidden_states)?;
        let beta = self
            .in_proj_b
            .forward(hidden_states)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let decay_input = self
            .in_proj_a
            .forward(hidden_states)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let convolved = self.causal_convolution(&mixed, batch_size, sequence_length, conv_state)?;
        let mut state = self.take_recurrent_state(batch_size, recurrent_state)?;
        let mut output = vec![0.0_f32; batch_size * sequence_length * self.value_dim];
        let repetitions = self.num_value_heads / self.num_key_heads;
        let state_per_head = self.key_head_dim * self.value_head_dim;
        let mut normalized_query = vec![0.0_f32; self.key_head_dim];
        let mut normalized_key = vec![0.0_f32; self.key_head_dim];
        let mut delta = vec![0.0_f32; self.value_head_dim];

        for batch in 0..batch_size {
            for token in 0..sequence_length {
                let mixed_base = (batch * sequence_length + token) * self.conv_dim;
                let control_base = (batch * sequence_length + token) * self.num_value_heads;
                let output_base = (batch * sequence_length + token) * self.value_dim;
                for value_head in 0..self.num_value_heads {
                    let key_head = value_head / repetitions;
                    let query_base = mixed_base + key_head * self.key_head_dim;
                    let key_base = mixed_base + self.key_dim + key_head * self.key_head_dim;
                    l2_normalize(
                        &convolved[query_base..query_base + self.key_head_dim],
                        &mut normalized_query,
                    );
                    l2_normalize(
                        &convolved[key_base..key_base + self.key_head_dim],
                        &mut normalized_key,
                    );
                    let value_base =
                        mixed_base + 2 * self.key_dim + value_head * self.value_head_dim;
                    let state_base = (batch * self.num_value_heads + value_head) * state_per_head;
                    let beta = sigmoid(beta[control_base + value_head]);
                    let g = -self.a_log[value_head].exp()
                        * softplus(
                            decay_input[control_base + value_head] + self.dt_bias[value_head],
                        );
                    let decay = g.exp();
                    for state_value in &mut state[state_base..state_base + state_per_head] {
                        *state_value *= decay;
                    }
                    for value_dimension in 0..self.value_head_dim {
                        let mut memory = 0.0_f32;
                        for key_dimension in 0..self.key_head_dim {
                            memory += state[state_base
                                + key_dimension * self.value_head_dim
                                + value_dimension]
                                * normalized_key[key_dimension];
                        }
                        delta[value_dimension] =
                            (convolved[value_base + value_dimension] - memory) * beta;
                    }
                    for (key_dimension, key) in normalized_key.iter().copied().enumerate() {
                        let row = state_base + key_dimension * self.value_head_dim;
                        for value_dimension in 0..self.value_head_dim {
                            state[row + value_dimension] += key * delta[value_dimension];
                        }
                    }
                    let scale = 1.0_f32 / (self.key_head_dim as f32).sqrt();
                    for value_dimension in 0..self.value_head_dim {
                        let mut attended = 0.0_f32;
                        for key_dimension in 0..self.key_head_dim {
                            attended += state[state_base
                                + key_dimension * self.value_head_dim
                                + value_dimension]
                                * normalized_query[key_dimension];
                        }
                        output[output_base + value_head * self.value_head_dim + value_dimension] =
                            attended * scale;
                    }
                }
            }
        }
        *recurrent_state = Some(state);

        let output = Tensor::from_vec(
            output,
            (
                batch_size * sequence_length * self.num_value_heads,
                self.value_head_dim,
            ),
            hidden_states.device(),
        )?;
        let gated = self
            .norm
            .forward(&output)?
            .mul(&candle_nn::ops::silu(&z.reshape((
                batch_size * sequence_length * self.num_value_heads,
                self.value_head_dim,
            ))?)?)?
            .reshape((batch_size, sequence_length, self.value_dim))?;
        Ok(self.out_proj.forward(&gated)?)
    }

    fn causal_convolution(
        &self,
        mixed: &[f32],
        batch_size: usize,
        sequence_length: usize,
        conv_state: &mut Option<Vec<f32>>,
    ) -> Result<Vec<f32>> {
        let history_length = self.conv_kernel_size - 1;
        let expected_history = batch_size
            .checked_mul(self.conv_dim)
            .and_then(|value| value.checked_mul(history_length))
            .ok_or_else(|| {
                MoeError::InvalidTensor("convolution state size overflowed".to_string())
            })?;
        let history = match conv_state.take() {
            Some(history) if history.len() == expected_history => history,
            Some(history) => {
                return Err(MoeError::InvalidTensor(format!(
                    "convolution state contains {} values, expected {expected_history}",
                    history.len()
                )))
            }
            None => vec![0.0; expected_history],
        };
        let expected_mixed = batch_size
            .checked_mul(sequence_length)
            .and_then(|value| value.checked_mul(self.conv_dim))
            .ok_or_else(|| {
                MoeError::InvalidTensor("convolution input size overflowed".to_string())
            })?;
        if mixed.len() != expected_mixed {
            return Err(MoeError::InvalidTensor(format!(
                "convolution input contains {} values, expected {expected_mixed}",
                mixed.len()
            )));
        }
        let mut output = vec![0.0_f32; mixed.len()];
        for batch in 0..batch_size {
            for token in 0..sequence_length {
                for channel in 0..self.conv_dim {
                    let mut value = 0.0_f32;
                    for tap in 0..self.conv_kernel_size {
                        let source_position =
                            token as isize + tap as isize - history_length as isize;
                        let source = if source_position < 0 {
                            let history_position =
                                (history_length as isize + source_position) as usize;
                            history[(batch * self.conv_dim + channel) * history_length
                                + history_position]
                        } else {
                            mixed[(batch * sequence_length + source_position as usize)
                                * self.conv_dim
                                + channel]
                        };
                        value += source * self.conv_weight[channel * self.conv_kernel_size + tap];
                    }
                    output[(batch * sequence_length + token) * self.conv_dim + channel] =
                        silu(value);
                }
            }
        }
        let mut next_history = vec![0.0_f32; expected_history];
        for batch in 0..batch_size {
            for channel in 0..self.conv_dim {
                for history_position in 0..history_length {
                    let source_position = sequence_length as isize - history_length as isize
                        + history_position as isize;
                    next_history
                        [(batch * self.conv_dim + channel) * history_length + history_position] =
                        if source_position < 0 {
                            history[(batch * self.conv_dim + channel) * history_length
                                + (history_length as isize + source_position) as usize]
                        } else {
                            mixed[(batch * sequence_length + source_position as usize)
                                * self.conv_dim
                                + channel]
                        };
                }
            }
        }
        *conv_state = Some(next_history);
        Ok(output)
    }

    fn take_recurrent_state(
        &self,
        batch_size: usize,
        recurrent_state: &mut Option<Vec<f32>>,
    ) -> Result<Vec<f32>> {
        let expected = batch_size
            .checked_mul(self.num_value_heads)
            .and_then(|value| value.checked_mul(self.key_head_dim))
            .and_then(|value| value.checked_mul(self.value_head_dim))
            .ok_or_else(|| {
                MoeError::InvalidTensor("recurrent state size overflowed".to_string())
            })?;
        match recurrent_state.take() {
            Some(state) if state.len() == expected => Ok(state),
            Some(state) => Err(MoeError::InvalidTensor(format!(
                "recurrent state contains {} values, expected {expected}",
                state.len()
            ))),
            None => Ok(vec![0.0; expected]),
        }
    }
}

fn l2_normalize(input: &[f32], output: &mut [f32]) {
    let inverse_norm = (input.iter().map(|value| value * value).sum::<f32>() + L2_NORM_EPSILON)
        .sqrt()
        .recip();
    for (output, input) in output.iter_mut().zip(input) {
        *output = *input * inverse_norm;
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        value.exp().ln_1p()
    }
}

fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_scalar_functions_cover_extreme_inputs() {
        assert_eq!(sigmoid(100.0), 1.0);
        assert!(sigmoid(-100.0).is_finite());
        assert_eq!(softplus(100.0), 100.0);
        assert!(softplus(-100.0).is_finite());
        let mut output = [0.0; 2];
        l2_normalize(&[3.0, 4.0], &mut output);
        assert!((output[0] - 0.6).abs() < 1e-6);
        assert!((output[1] - 0.8).abs() < 1e-6);
    }
}

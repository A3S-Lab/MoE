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
    conv_weight: Tensor,
    decay_scale: Tensor,
    dt_bias: Tensor,
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
        let repetitions = config.linear_num_value_heads / config.linear_num_key_heads;
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
            .squeeze(1)?;
        let decay_scale = builder
            .get(config.linear_num_value_heads, "A_log")?
            .exp()?
            .neg()?
            .reshape((1, 1, config.linear_num_key_heads, repetitions))?;
        let dt_bias = builder
            .get(config.linear_num_value_heads, "dt_bias")?
            .reshape((1, 1, config.linear_num_key_heads, repetitions))?;
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
            decay_scale,
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
        conv_state: &mut Option<Tensor>,
        recurrent_state: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        if hidden_states.dtype() != DType::F32 {
            return Err(MoeError::InvalidTensor(format!(
                "Qwen3.6 Gated DeltaNet requires F32 hidden states, found {:?}",
                hidden_states.dtype()
            )));
        }
        if !hidden_states
            .device()
            .same_device(self.conv_weight.device())
        {
            return Err(MoeError::InvalidTensor(
                "Qwen3.6 Gated DeltaNet input and weights must use the same device".to_string(),
            ));
        }
        let (batch_size, sequence_length, hidden_size) = hidden_states.dims3()?;
        if hidden_size != self.hidden_size {
            return Err(MoeError::InvalidTensor(format!(
                "linear-attention input width must be {}, found {hidden_size}",
                self.hidden_size
            )));
        }

        let mixed = self.in_proj_qkv.forward(hidden_states)?;
        let z = self.in_proj_z.forward(hidden_states)?;
        let repetitions = self.num_value_heads / self.num_key_heads;
        let beta = candle_nn::ops::sigmoid(&self.in_proj_b.forward(hidden_states)?.reshape((
            batch_size,
            sequence_length,
            self.num_key_heads,
            repetitions,
        ))?)?;
        let decay_input = self.in_proj_a.forward(hidden_states)?.reshape((
            batch_size,
            sequence_length,
            self.num_key_heads,
            repetitions,
        ))?;
        let decay = tensor_softplus(&decay_input.broadcast_add(&self.dt_bias)?)?
            .broadcast_mul(&self.decay_scale)?
            .exp()?;
        let convolved = self.causal_convolution(&mixed, conv_state)?;
        let queries = l2_normalize_tensor(&convolved.narrow(2, 0, self.key_dim)?.reshape((
            batch_size,
            sequence_length,
            self.num_key_heads,
            self.key_head_dim,
        ))?)?;
        let keys =
            l2_normalize_tensor(&convolved.narrow(2, self.key_dim, self.key_dim)?.reshape((
                batch_size,
                sequence_length,
                self.num_key_heads,
                self.key_head_dim,
            ))?)?;
        let values = convolved
            .narrow(2, 2 * self.key_dim, self.value_dim)?
            .reshape((
                batch_size,
                sequence_length,
                self.num_key_heads,
                repetitions,
                self.value_head_dim,
            ))?;
        let mut state =
            self.take_recurrent_state(batch_size, repetitions, hidden_states, recurrent_state)?;
        let mut outputs = Vec::with_capacity(sequence_length);
        let scale = 1.0_f64 / (self.key_head_dim as f64).sqrt();
        for token in 0..sequence_length {
            let query = queries
                .narrow(1, token, 1)?
                .squeeze(1)?
                .unsqueeze(2)?
                .unsqueeze(4)?;
            let key = keys
                .narrow(1, token, 1)?
                .squeeze(1)?
                .unsqueeze(2)?
                .unsqueeze(4)?;
            let value = values.narrow(1, token, 1)?.squeeze(1)?;
            let token_decay = decay
                .narrow(1, token, 1)?
                .squeeze(1)?
                .unsqueeze(3)?
                .unsqueeze(4)?;
            state = state.broadcast_mul(&token_decay)?;
            let memory = state.broadcast_mul(&key)?.sum(3)?;
            let token_beta = beta.narrow(1, token, 1)?.squeeze(1)?.unsqueeze(3)?;
            let delta = value.sub(&memory)?.broadcast_mul(&token_beta)?;
            state = state.add(&key.broadcast_mul(&delta.unsqueeze(3)?)?)?;
            outputs.push(
                (state.broadcast_mul(&query)?.sum(3)? * scale)?
                    .reshape((batch_size, self.value_dim))?,
            );
        }
        *recurrent_state = Some(state);

        let output = Tensor::stack(&outputs, 1)?.reshape((
            batch_size * sequence_length * self.num_value_heads,
            self.value_head_dim,
        ))?;
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
        mixed: &Tensor,
        conv_state: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        let (batch_size, sequence_length, conv_dim) = mixed.dims3()?;
        if conv_dim != self.conv_dim {
            return Err(MoeError::InvalidTensor(format!(
                "linear convolution input width must be {}, found {conv_dim}",
                self.conv_dim
            )));
        }
        let history_length = self.conv_kernel_size - 1;
        let history = match conv_state.take() {
            Some(history)
                if history.dtype() == DType::F32
                    && history.device().same_device(mixed.device())
                    && history.dims() == [batch_size, self.conv_dim, history_length] =>
            {
                history
            }
            Some(history) => {
                return Err(MoeError::InvalidTensor(format!(
                    "convolution state must be F32 [{batch_size}, {}, {history_length}] on the execution device, found {:?} {:?}",
                    self.conv_dim,
                    history.dtype(),
                    history.dims()
                )))
            }
            None => Tensor::zeros(
                (batch_size, self.conv_dim, history_length),
                DType::F32,
                mixed.device(),
            )?,
        };
        let channels = mixed.transpose(1, 2)?.contiguous()?;
        let window = if history_length == 0 {
            channels
        } else {
            Tensor::cat(&[&history, &channels], 2)?
        };
        let mut output = Tensor::zeros(
            (batch_size, self.conv_dim, sequence_length),
            DType::F32,
            mixed.device(),
        )?;
        for tap in 0..self.conv_kernel_size {
            let source = window.narrow(2, tap, sequence_length)?;
            let weight = self
                .conv_weight
                .narrow(1, tap, 1)?
                .reshape((1, self.conv_dim, 1))?;
            output = output.add(&source.broadcast_mul(&weight)?)?;
        }
        *conv_state = if history_length == 0 {
            None
        } else {
            Some(
                window
                    .narrow(2, sequence_length, history_length)?
                    .contiguous()?,
            )
        };
        Ok(candle_nn::ops::silu(&output)?
            .transpose(1, 2)?
            .contiguous()?)
    }

    fn take_recurrent_state(
        &self,
        batch_size: usize,
        repetitions: usize,
        hidden_states: &Tensor,
        recurrent_state: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        let expected = [
            batch_size,
            self.num_key_heads,
            repetitions,
            self.key_head_dim,
            self.value_head_dim,
        ];
        match recurrent_state.take() {
            Some(state)
                if state.dtype() == DType::F32
                    && state.device().same_device(hidden_states.device())
                    && state.dims() == expected =>
            {
                Ok(state)
            }
            Some(state) => Err(MoeError::InvalidTensor(format!(
                "recurrent state must be F32 {expected:?} on the execution device, found {:?} {:?}",
                state.dtype(),
                state.dims()
            ))),
            None => Ok(Tensor::zeros(
                &expected,
                DType::F32,
                hidden_states.device(),
            )?),
        }
    }
}

fn l2_normalize_tensor(input: &Tensor) -> Result<Tensor> {
    let norm = (input.sqr()?.sum_keepdim(3)? + L2_NORM_EPSILON as f64)?.sqrt()?;
    Ok(input.broadcast_div(&norm)?)
}

fn tensor_softplus(input: &Tensor) -> Result<Tensor> {
    let positive = input.maximum(&input.zeros_like()?)?;
    let tail = (input.abs()?.neg()?.exp()? + 1.0)?.log()?;
    Ok(positive.add(&tail)?)
}

#[cfg(test)]
fn l2_normalize(input: &[f32], output: &mut [f32]) {
    let inverse_norm = (input.iter().map(|value| value * value).sum::<f32>() + L2_NORM_EPSILON)
        .sqrt()
        .recip();
    for (output, input) in output.iter_mut().zip(input) {
        *output = *input * inverse_norm;
    }
}

#[cfg(test)]
fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

#[cfg(test)]
fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        value.exp().ln_1p()
    }
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

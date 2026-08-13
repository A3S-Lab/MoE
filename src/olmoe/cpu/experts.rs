use a3s_power::inference::RoutedExpertBatch;
use candle_core::{DType, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::olmoe::{OlmoeConfig, OlmoeMoeConfig};
use crate::sparse::select_routes;
use crate::{Matrix, MoeError, Result};

#[derive(Debug, Clone)]
struct ResidentExpert {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl ResidentExpert {
    fn load(config: &OlmoeConfig, builder: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            gate: candle_nn::linear_no_bias(
                config.hidden_size,
                config.intermediate_size,
                builder.pp("gate_proj"),
            )?,
            up: candle_nn::linear_no_bias(
                config.hidden_size,
                config.intermediate_size,
                builder.pp("up_proj"),
            )?,
            down: candle_nn::linear_no_bias(
                config.intermediate_size,
                config.hidden_size,
                builder.pp("down_proj"),
            )?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate.forward(hidden_states)?)?;
        let up = self.up.forward(hidden_states)?;
        Ok(self.down.forward(&gate.mul(&up)?)?)
    }
}

#[derive(Debug, Clone)]
pub(super) struct ResidentOlmoeMlp {
    router: Linear,
    experts: Vec<ResidentExpert>,
    config: OlmoeMoeConfig,
}

pub(super) struct ResidentOlmoeMlpOutput {
    pub(super) hidden_states: Tensor,
    pub(super) router_logits: Tensor,
    pub(super) routes: RoutedExpertBatch,
}

impl ResidentOlmoeMlp {
    pub(super) fn load(config: &OlmoeConfig, builder: VarBuilder<'_>) -> Result<Self> {
        let moe_config = config.moe_config()?;
        let router =
            candle_nn::linear_no_bias(config.hidden_size, config.num_experts, builder.pp("gate"))?;
        let mut experts = Vec::with_capacity(config.num_experts);
        for expert in 0..config.num_experts {
            experts.push(ResidentExpert::load(
                config,
                builder.pp(format!("experts.{expert}")),
            )?);
        }
        Ok(Self {
            router,
            experts,
            config: moe_config,
        })
    }

    pub(super) fn forward(
        &self,
        layer: u32,
        hidden_states: &Tensor,
    ) -> Result<ResidentOlmoeMlpOutput> {
        let (batch_size, sequence_length, hidden_size) = hidden_states.dims3()?;
        if hidden_size != self.config.hidden_size {
            return Err(MoeError::InvalidTensor(format!(
                "MoE input width must be {}, found {hidden_size}",
                self.config.hidden_size
            )));
        }
        let positions = batch_size
            .checked_mul(sequence_length)
            .ok_or_else(|| MoeError::InvalidTensor("MoE position count overflowed".to_string()))?;
        let flattened = hidden_states.reshape((positions, hidden_size))?;
        let router_logits = self.router.forward(&flattened)?.to_dtype(DType::F32)?;
        let route_matrix = Matrix::new(
            positions,
            self.config.num_experts,
            router_logits.flatten_all()?.to_vec1::<f32>()?,
        )?;
        let routes = select_routes(layer, &route_matrix, self.config)?;
        let mut output = Tensor::zeros(
            (positions, hidden_size),
            hidden_states.dtype(),
            hidden_states.device(),
        )?;

        for expert in routes.experts() {
            let expert_index = *expert as usize;
            let expert_weights = self.experts.get(expert_index).ok_or_else(|| {
                MoeError::Inference(format!("routed expert {expert_index} is unavailable"))
            })?;
            let assignments = routes.assignments(*expert);
            let position_indices = assignments
                .iter()
                .map(|assignment| {
                    u32::try_from(assignment.position).map_err(|_| {
                        MoeError::InvalidTensor(
                            "MoE position exceeds the tensor index representation".to_string(),
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let indices =
                Tensor::from_vec(position_indices, assignments.len(), hidden_states.device())?;
            let route_weights = Tensor::from_vec(
                assignments
                    .iter()
                    .map(|assignment| assignment.weight)
                    .collect::<Vec<_>>(),
                (assignments.len(), 1),
                hidden_states.device(),
            )?
            .to_dtype(hidden_states.dtype())?;
            let selected = flattened.index_select(&indices, 0)?;
            let expert_output = expert_weights
                .forward(&selected)?
                .broadcast_mul(&route_weights)?;
            output = output.index_add(&indices, &expert_output, 0)?;
        }

        Ok(ResidentOlmoeMlpOutput {
            hidden_states: output.reshape((batch_size, sequence_length, hidden_size))?,
            router_logits: router_logits.reshape((
                batch_size,
                sequence_length,
                self.config.num_experts,
            ))?,
            routes,
        })
    }
}

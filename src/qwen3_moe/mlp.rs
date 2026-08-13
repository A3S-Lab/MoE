use a3s_power::inference::RoutedExpertBatch;
use candle_core::{DType, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::sparse::select_routes;
use crate::{Matrix, MoeError, MoeLayerConfig, Result};

use super::Qwen3MoeConfig;

#[derive(Debug, Clone)]
pub(super) struct DenseMlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl DenseMlp {
    fn load(config: &Qwen3MoeConfig, builder: VarBuilder<'_>) -> Result<Self> {
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
        Ok(self
            .down
            .forward(&gate.mul(&self.up.forward(hidden_states)?)?)?)
    }
}

#[derive(Debug, Clone)]
struct SparseExpert {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl SparseExpert {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate.forward(hidden_states)?)?;
        Ok(self
            .down
            .forward(&gate.mul(&self.up.forward(hidden_states)?)?)?)
    }
}

#[derive(Debug, Clone)]
pub(super) struct SparseMlp {
    router: Linear,
    experts: Vec<SparseExpert>,
    config: MoeLayerConfig,
}

impl SparseMlp {
    fn load(config: &Qwen3MoeConfig, builder: VarBuilder<'_>) -> Result<Self> {
        let moe = config.moe_config()?;
        let router =
            candle_nn::linear_no_bias(config.hidden_size, config.num_experts, builder.pp("gate"))?;
        let gate_up_width = config.moe_intermediate_size.checked_mul(2).ok_or_else(|| {
            MoeError::InvalidConfig("expert gate/up width overflowed".to_string())
        })?;
        let gate_up = builder.pp("experts").get(
            (config.num_experts, gate_up_width, config.hidden_size),
            "gate_up_proj",
        )?;
        let down = builder.pp("experts").get(
            (
                config.num_experts,
                config.hidden_size,
                config.moe_intermediate_size,
            ),
            "down_proj",
        )?;
        let mut experts = Vec::with_capacity(config.num_experts);
        for expert in 0..config.num_experts {
            let gate_up = gate_up.narrow(0, expert, 1)?.squeeze(0)?;
            experts.push(SparseExpert {
                gate: Linear::new(
                    gate_up
                        .narrow(0, 0, config.moe_intermediate_size)?
                        .contiguous()?,
                    None,
                ),
                up: Linear::new(
                    gate_up
                        .narrow(
                            0,
                            config.moe_intermediate_size,
                            config.moe_intermediate_size,
                        )?
                        .contiguous()?,
                    None,
                ),
                down: Linear::new(down.narrow(0, expert, 1)?.squeeze(0)?.contiguous()?, None),
            });
        }
        Ok(Self {
            router,
            experts,
            config: moe,
        })
    }

    fn forward(&self, layer: u32, hidden_states: &Tensor) -> Result<Qwen3MoeMlpOutput> {
        let (batch_size, sequence_length, hidden_size) = hidden_states.dims3()?;
        let positions = batch_size
            .checked_mul(sequence_length)
            .ok_or_else(|| MoeError::InvalidTensor("MoE position count overflowed".to_string()))?;
        let flattened = hidden_states.reshape((positions, hidden_size))?;
        let router_logits = self.router.forward(&flattened)?.to_dtype(DType::F32)?;
        let routes = select_routes(
            layer,
            &Matrix::new(
                positions,
                self.config.num_experts,
                router_logits.flatten_all()?.to_vec1::<f32>()?,
            )?,
            self.config,
        )?;
        let mut output = Tensor::zeros(
            (positions, hidden_size),
            hidden_states.dtype(),
            hidden_states.device(),
        )?;
        for expert in routes.experts() {
            let assignments = routes.assignments(*expert);
            let indices = Tensor::from_vec(
                assignments
                    .iter()
                    .map(|assignment| {
                        u32::try_from(assignment.position).map_err(|_| {
                            MoeError::InvalidTensor(
                                "MoE position exceeds tensor indexing".to_string(),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                assignments.len(),
                hidden_states.device(),
            )?;
            let route_weights = Tensor::from_vec(
                assignments
                    .iter()
                    .map(|assignment| assignment.weight)
                    .collect::<Vec<_>>(),
                (assignments.len(), 1),
                hidden_states.device(),
            )?
            .to_dtype(hidden_states.dtype())?;
            let expert_weights = self.experts.get(*expert as usize).ok_or_else(|| {
                MoeError::Inference(format!("routed expert {expert} is unavailable"))
            })?;
            let selected = flattened.index_select(&indices, 0)?;
            let contribution = expert_weights
                .forward(&selected)?
                .broadcast_mul(&route_weights)?;
            output = output.index_add(&indices, &contribution, 0)?;
        }
        Ok(Qwen3MoeMlpOutput {
            hidden_states: output.reshape((batch_size, sequence_length, hidden_size))?,
            router_logits: Some(router_logits.reshape((
                batch_size,
                sequence_length,
                self.config.num_experts,
            ))?),
            routes: Some(routes),
        })
    }
}

#[derive(Debug, Clone)]
pub(super) enum Qwen3MoeMlp {
    Dense(DenseMlp),
    Sparse(SparseMlp),
}

pub(super) struct Qwen3MoeMlpOutput {
    pub hidden_states: Tensor,
    pub router_logits: Option<Tensor>,
    pub routes: Option<RoutedExpertBatch>,
}

impl Qwen3MoeMlp {
    pub(super) fn load(
        config: &Qwen3MoeConfig,
        layer: usize,
        builder: VarBuilder<'_>,
    ) -> Result<Self> {
        if config.is_sparse_layer(layer) {
            Ok(Self::Sparse(SparseMlp::load(config, builder)?))
        } else {
            Ok(Self::Dense(DenseMlp::load(config, builder)?))
        }
    }

    pub(super) fn forward(&self, layer: u32, hidden_states: &Tensor) -> Result<Qwen3MoeMlpOutput> {
        match self {
            Self::Dense(mlp) => Ok(Qwen3MoeMlpOutput {
                hidden_states: mlp.forward(hidden_states)?,
                router_logits: None,
                routes: None,
            }),
            Self::Sparse(mlp) => mlp.forward(layer, hidden_states),
        }
    }
}

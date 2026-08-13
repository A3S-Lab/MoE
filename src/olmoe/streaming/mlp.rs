use a3s_power::error::PowerError;
use a3s_power::inference::{
    ExecutionPermit, PlacementPreference, PlacementTelemetry, RoutedExpertBatch,
    StagedWeightBatchReport, StagedWeightGroup, StagedWeightGroupRequest, WeightHierarchy,
    WeightKey, WeightRequest,
};
use candle_core::{DType, Tensor};
use candle_nn::{Linear, Module};
use tokio_util::sync::CancellationToken;

use crate::olmoe::router::select_routes;
use crate::olmoe::OlmoeMoeConfig;
use crate::{Matrix, MoeError, Result};

use super::packed::{packed_expert_tensor_name, PackedExpertRecord};

/// Output of one Power-backed sparse feed-forward layer.
#[derive(Debug)]
pub struct OlmoeStreamingMlpOutput {
    pub hidden_states: Tensor,
    pub router_logits: Tensor,
    pub routes: RoutedExpertBatch,
    pub staging: StagedWeightBatchReport,
}

/// OLMoE sparse feed-forward execution backed by Power's weight hierarchy.
///
/// The router remains resident. Only the exact union of selected experts is
/// staged, groups may execute as they become ready, and reduction is restored
/// to ascending expert order for deterministic numerical behavior.
#[derive(Clone)]
pub struct OlmoeStreamingMlp {
    layer: u32,
    config: OlmoeMoeConfig,
    router: Linear,
    hierarchy: WeightHierarchy,
}

impl OlmoeStreamingMlp {
    pub fn new(
        layer: u32,
        config: OlmoeMoeConfig,
        router_weight: Tensor,
        hierarchy: WeightHierarchy,
    ) -> Result<Self> {
        config.validate()?;
        let (experts, hidden_size) = router_weight.dims2()?;
        if experts != config.num_experts || hidden_size != config.hidden_size {
            return Err(MoeError::InvalidTensor(format!(
                "streaming router weight must have shape [{}, {}], found [{experts}, {hidden_size}]",
                config.num_experts, config.hidden_size
            )));
        }
        if router_weight.dtype() != DType::F32 {
            return Err(MoeError::InvalidTensor(format!(
                "streaming router requires F32 execution, found {:?}",
                router_weight.dtype()
            )));
        }
        if !router_weight
            .device()
            .same_device(hierarchy.runtime().device().tensor_device())
        {
            return Err(MoeError::InvalidTensor(
                "streaming router and Power runtime must use the same device".to_string(),
            ));
        }
        for expert in 0..config.num_experts {
            let expert = u32::try_from(expert).map_err(|_| {
                MoeError::InvalidConfig("expert index exceeds the routing contract".to_string())
            })?;
            let name = packed_expert_tensor_name(layer, expert);
            let descriptor = hierarchy.store().descriptor(&name).ok_or_else(|| {
                MoeError::InvalidTensor(format!(
                    "Power weight store is missing packed expert '{name}'"
                ))
            })?;
            if descriptor.dtype != "u8"
                || descriptor.shape.len() != 1
                || descriptor.shape[0] < PackedExpertRecord::HEADER_BYTES
            {
                return Err(MoeError::InvalidTensor(format!(
                    "packed expert '{name}' must be a rank-one U8 tensor with a complete header"
                )));
            }
        }
        Ok(Self {
            layer,
            config,
            router: Linear::new(router_weight, None),
            hierarchy,
        })
    }

    pub fn telemetry(&self) -> PlacementTelemetry {
        self.hierarchy.telemetry()
    }

    pub fn hierarchy(&self) -> &WeightHierarchy {
        &self.hierarchy
    }

    pub async fn forward(
        &self,
        hidden_states: &Tensor,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> Result<OlmoeStreamingMlpOutput> {
        if cancellation.is_cancelled() {
            return Err(MoeError::Power(PowerError::InferenceCancelled));
        }
        if hidden_states.dtype() != DType::F32 {
            return Err(MoeError::InvalidTensor(format!(
                "streaming MoE input must use F32, found {:?}",
                hidden_states.dtype()
            )));
        }
        if !hidden_states
            .device()
            .same_device(self.hierarchy.runtime().device().tensor_device())
        {
            return Err(MoeError::InvalidTensor(
                "streaming MoE input and Power runtime must use the same device".to_string(),
            ));
        }
        let (batch_size, sequence_length, hidden_size) = hidden_states.dims3()?;
        if hidden_size != self.config.hidden_size {
            return Err(MoeError::InvalidTensor(format!(
                "streaming MoE input width must be {}, found {hidden_size}",
                self.config.hidden_size
            )));
        }
        let positions = batch_size.checked_mul(sequence_length).ok_or_else(|| {
            MoeError::InvalidTensor("streaming MoE position count overflowed".to_string())
        })?;
        let flattened = hidden_states.reshape((positions, hidden_size))?;
        let router_logits = self.router.forward(&flattened)?.to_dtype(DType::F32)?;
        let route_matrix = Matrix::new(
            positions,
            self.config.num_experts,
            router_logits.flatten_all()?.to_vec1::<f32>()?,
        )?;
        let routes = select_routes(self.layer, &route_matrix, self.config)?;
        let groups = routes
            .experts()
            .iter()
            .map(|expert| {
                StagedWeightGroupRequest::new(vec![WeightRequest::new(
                    WeightKey::new(self.layer, packed_expert_tensor_name(self.layer, *expert)),
                    PlacementPreference::Fastest,
                )])
            })
            .collect();
        let mut batch = self
            .hierarchy
            .start_staged_batch(groups, permit, cancellation.clone())?;
        let mut contributions = (0..routes.experts().len())
            .map(|_| None)
            .collect::<Vec<Option<ExpertContribution>>>();

        for group in batch.ready_groups() {
            self.compute_group(&routes, &flattened, group, &mut contributions, cancellation)?;
        }
        while let Some(group) = batch.next_ready_group().await? {
            self.compute_group(&routes, &flattened, group, &mut contributions, cancellation)?;
        }
        let completion = batch.wait().await?;
        let mut output = Tensor::zeros(
            (positions, hidden_size),
            hidden_states.dtype(),
            hidden_states.device(),
        )?;
        for (canonical_index, contribution) in contributions.into_iter().enumerate() {
            let contribution = contribution.ok_or_else(|| {
                MoeError::Inference(format!(
                    "staged expert group {canonical_index} completed without a contribution"
                ))
            })?;
            output = output.index_add(&contribution.indices, &contribution.values, 0)?;
        }
        self.hierarchy.record_routes(&routes);

        Ok(OlmoeStreamingMlpOutput {
            hidden_states: output.reshape((batch_size, sequence_length, hidden_size))?,
            router_logits: router_logits.reshape((
                batch_size,
                sequence_length,
                self.config.num_experts,
            ))?,
            routes,
            staging: completion.report,
        })
    }

    fn compute_group(
        &self,
        routes: &RoutedExpertBatch,
        flattened: &Tensor,
        group: StagedWeightGroup,
        contributions: &mut [Option<ExpertContribution>],
        cancellation: &CancellationToken,
    ) -> Result<()> {
        if cancellation.is_cancelled() {
            return Err(MoeError::Power(PowerError::InferenceCancelled));
        }
        let canonical_index = group.canonical_index();
        let expert = *routes.experts().get(canonical_index).ok_or_else(|| {
            MoeError::Inference(format!(
                "staged group {canonical_index} does not map to a routed expert"
            ))
        })?;
        let mut weights = group.into_weights();
        if weights.len() != 1 {
            return Err(MoeError::Inference(format!(
                "packed expert group {canonical_index} returned {} weights instead of one",
                weights.len()
            )));
        }
        let packed = weights.remove(0).into_tensor();
        if packed.dtype() != DType::U8 || packed.rank() != 1 {
            return Err(MoeError::InvalidTensor(format!(
                "packed expert {expert} must be a rank-one U8 tensor"
            )));
        }
        let encoded = packed.flatten_all()?.to_vec1::<u8>()?;
        let record = PackedExpertRecord::parse(encoded, self.config)?;
        let expert_weights =
            MaterializedExpert::from_record(&record, flattened.dtype(), flattened.device())?;
        let assignments = routes.assignments(expert);
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
        let indices = Tensor::from_vec(position_indices, assignments.len(), flattened.device())?;
        let route_weights = Tensor::from_vec(
            assignments
                .iter()
                .map(|assignment| assignment.weight)
                .collect::<Vec<_>>(),
            (assignments.len(), 1),
            flattened.device(),
        )?;
        let selected = flattened.index_select(&indices, 0)?;
        let values = expert_weights
            .forward(&selected)?
            .broadcast_mul(&route_weights)?;
        let slot = contributions.get_mut(canonical_index).ok_or_else(|| {
            MoeError::Inference(format!(
                "staged group {canonical_index} exceeds the contribution slots"
            ))
        })?;
        if slot.is_some() {
            return Err(MoeError::Inference(format!(
                "staged group {canonical_index} was delivered more than once"
            )));
        }
        *slot = Some(ExpertContribution { indices, values });
        Ok(())
    }
}

impl std::fmt::Debug for OlmoeStreamingMlp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OlmoeStreamingMlp")
            .field("layer", &self.layer)
            .field("config", &self.config)
            .field("hierarchy", &"Power weight hierarchy")
            .finish_non_exhaustive()
    }
}

struct ExpertContribution {
    indices: Tensor,
    values: Tensor,
}

struct MaterializedExpert {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl MaterializedExpert {
    fn from_record(
        record: &PackedExpertRecord,
        dtype: DType,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let config = record.config();
        let decoded = record.decode_f32()?;
        let gate = Tensor::from_vec(
            decoded.gate,
            (config.intermediate_size, config.hidden_size),
            device,
        )?
        .to_dtype(dtype)?;
        let up = Tensor::from_vec(
            decoded.up,
            (config.intermediate_size, config.hidden_size),
            device,
        )?
        .to_dtype(dtype)?;
        let down = Tensor::from_vec(
            decoded.down,
            (config.hidden_size, config.intermediate_size),
            device,
        )?
        .to_dtype(dtype)?;
        Ok(Self {
            gate: Linear::new(gate, None),
            up: Linear::new(up, None),
            down: Linear::new(down, None),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate.forward(hidden_states)?)?;
        let up = self.up.forward(hidden_states)?;
        Ok(self.down.forward(&gate.mul(&up)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_streaming_types_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OlmoeStreamingMlp>();
        assert_send_sync::<OlmoeStreamingMlpOutput>();
    }
}

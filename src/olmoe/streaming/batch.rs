use a3s_power::error::PowerError;
use a3s_power::inference::{ExecutionPermit, RoutedExpertBatch, StagedWeightBatchReport};
use candle_core::Tensor;
use tokio_util::sync::CancellationToken;

use crate::olmoe::OlmoeKvCache;
use crate::{MoeError, Result};

use super::OlmoeStreamingModel;

/// One independently cached session row participating in an expert-unioned
/// forward step.
pub struct OlmoeStreamingBatchRow<'a> {
    token_ids: &'a Tensor,
    cache: &'a mut OlmoeKvCache,
}

impl<'a> OlmoeStreamingBatchRow<'a> {
    pub fn new(token_ids: &'a Tensor, cache: &'a mut OlmoeKvCache) -> Self {
        Self { token_ids, cache }
    }

    pub fn token_ids(&self) -> &Tensor {
        self.token_ids
    }

    pub fn cache(&self) -> &OlmoeKvCache {
        self.cache
    }
}

impl std::fmt::Debug for OlmoeStreamingBatchRow<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OlmoeStreamingBatchRow")
            .field("shape", &self.token_ids.dims())
            .field("position", &self.cache.position())
            .finish()
    }
}

/// Per-session output restored from a route-unioned forward step.
#[derive(Debug)]
pub struct OlmoeStreamingBatchRowOutput {
    pub logits: Tensor,
    pub layer_router_logits: Vec<Tensor>,
    pub layer_routes: Vec<RoutedExpertBatch>,
}

/// Canonically ordered result and aggregate staging evidence for one batched
/// model step.
#[derive(Debug)]
pub struct OlmoeStreamingBatchOutput {
    pub rows: Vec<OlmoeStreamingBatchRowOutput>,
    pub layer_union_routes: Vec<RoutedExpertBatch>,
    pub layer_staging: Vec<StagedWeightBatchReport>,
}

impl OlmoeStreamingModel {
    /// Executes compatible session rows as one expert-unioned batch.
    ///
    /// Each row contains one independent session. Rows may have different
    /// positive token widths and absolute positions: attention runs against
    /// each row's own KV cache, then flattened MLP inputs are concatenated per
    /// layer so every active expert is staged exactly once for the ragged
    /// union. Caller caches are committed together only after the complete
    /// batch succeeds.
    pub async fn forward_batch(
        &self,
        rows: &mut [OlmoeStreamingBatchRow<'_>],
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> Result<OlmoeStreamingBatchOutput> {
        check_cancellation(cancellation)?;
        let limits = self.runtime().limits();
        if rows.is_empty() {
            return Err(MoeError::InvalidTensor(
                "streaming batch must contain at least one row".to_string(),
            ));
        }
        if rows.len() > limits.max_concurrent_requests || rows.len() > limits.max_graph_nodes {
            return Err(MoeError::InvalidTensor(format!(
                "streaming batch contains {} rows, exceeding runtime bounds",
                rows.len()
            )));
        }

        let mut sequence_lengths = Vec::with_capacity(rows.len());
        let mut next_caches = Vec::with_capacity(rows.len());
        let mut positions = Vec::with_capacity(rows.len());
        let mut total_input_bytes = 0_usize;
        let mut total_positions = 0_usize;
        for (index, row) in rows.iter().enumerate() {
            let (batch_size, row_sequence_length, position) =
                self.dense.validate_step(row.token_ids, row.cache)?;
            if batch_size != 1 {
                return Err(MoeError::InvalidTensor(format!(
                    "streaming batch row {index} must have batch dimension 1, found {batch_size}"
                )));
            }
            total_positions = total_positions
                .checked_add(row_sequence_length)
                .ok_or_else(|| {
                    MoeError::InvalidTensor("streaming batch position count overflowed".to_string())
                })?;
            total_input_bytes = total_input_bytes
                .checked_add(
                    row_sequence_length
                        .checked_mul(size_of::<u32>())
                        .ok_or_else(|| {
                            MoeError::InvalidTensor(
                                "streaming batch input byte count overflowed".to_string(),
                            )
                        })?,
                )
                .ok_or_else(|| {
                    MoeError::InvalidTensor(
                        "streaming batch input byte count overflowed".to_string(),
                    )
                })?;
            next_caches.push(row.cache.clone());
            positions.push(position);
            sequence_lengths.push(row_sequence_length);
        }
        if total_input_bytes > limits.max_input_bytes {
            return Err(MoeError::InvalidTensor(format!(
                "streaming batch contains {total_input_bytes} input bytes, exceeding {}",
                limits.max_input_bytes
            )));
        }
        limits.checked_elements(
            &[total_positions, self.config().hidden_size],
            "streaming batch hidden states",
        )?;

        let mut hidden_states = rows
            .iter()
            .map(|row| self.dense.embed(row.token_ids))
            .collect::<Result<Vec<_>>>()?;
        let mut row_router_logits = (0..rows.len())
            .map(|_| Vec::with_capacity(self.mlps.len()))
            .collect::<Vec<_>>();
        let mut row_routes = (0..rows.len())
            .map(|_| Vec::with_capacity(self.mlps.len()))
            .collect::<Vec<_>>();
        let mut layer_union_routes = Vec::with_capacity(self.mlps.len());
        let mut layer_staging = Vec::with_capacity(self.mlps.len());

        for (layer, mlp) in self.mlps.iter().enumerate() {
            check_cancellation(cancellation)?;
            let mut attended = Vec::with_capacity(rows.len());
            let mut normalized = Vec::with_capacity(rows.len());
            for row in 0..rows.len() {
                let row_attended = self.dense.forward_attention(
                    layer,
                    &hidden_states[row],
                    &mut next_caches[row],
                    positions[row],
                )?;
                normalized.push(self.dense.normalize_mlp_input(layer, &row_attended)?);
                attended.push(row_attended);
            }
            let flattened = normalized
                .iter()
                .zip(&sequence_lengths)
                .map(|(tensor, sequence_length)| {
                    tensor.reshape((*sequence_length, self.config().hidden_size))
                })
                .collect::<candle_core::Result<Vec<_>>>()?;
            let flattened_refs = flattened.iter().collect::<Vec<_>>();
            let union_input = Tensor::cat(&flattened_refs, 0)?.reshape((
                1,
                total_positions,
                self.config().hidden_size,
            ))?;
            let moe = mlp.forward(&union_input, permit, cancellation).await?;
            check_cancellation(cancellation)?;
            let union_hidden = moe
                .hidden_states
                .reshape((total_positions, self.config().hidden_size))?;
            let union_logits = moe
                .router_logits
                .reshape((total_positions, self.config().num_experts))?;
            let mut start = 0_usize;
            for row in 0..rows.len() {
                let row_sequence_length = sequence_lengths[row];
                let row_moe = union_hidden
                    .narrow(0, start, row_sequence_length)?
                    .reshape((1, row_sequence_length, self.config().hidden_size))?;
                hidden_states[row] = attended[row].add(&row_moe)?;
                row_router_logits[row].push(
                    union_logits
                        .narrow(0, start, row_sequence_length)?
                        .reshape((1, row_sequence_length, self.config().num_experts))?,
                );
                let end = start.checked_add(row_sequence_length).ok_or_else(|| {
                    MoeError::InvalidTensor("streaming batch route range overflowed".to_string())
                })?;
                let selections = moe
                    .routes
                    .selections()
                    .get(start..end)
                    .ok_or_else(|| {
                        MoeError::Inference(format!("union routes do not contain batch row {row}"))
                    })?
                    .to_vec();
                row_routes[row].push(RoutedExpertBatch::new(
                    u32::try_from(layer).map_err(|_| {
                        MoeError::InvalidConfig(
                            "layer index exceeds the routing contract".to_string(),
                        )
                    })?,
                    selections,
                    u32::try_from(self.config().num_experts).map_err(|_| {
                        MoeError::InvalidConfig(
                            "expert count exceeds the routing contract".to_string(),
                        )
                    })?,
                    self.config().num_experts_per_tok,
                )?);
                start = end;
            }
            layer_union_routes.push(moe.routes);
            layer_staging.push(moe.staging);
        }

        let mut outputs = Vec::with_capacity(rows.len());
        for row in 0..rows.len() {
            let logits = self.dense.finish(&hidden_states[row])?;
            outputs.push(OlmoeStreamingBatchRowOutput {
                logits,
                layer_router_logits: std::mem::take(&mut row_router_logits[row]),
                layer_routes: std::mem::take(&mut row_routes[row]),
            });
        }
        check_cancellation(cancellation)?;
        for ((row, mut next_cache), sequence_length) in
            rows.iter_mut().zip(next_caches).zip(sequence_lengths)
        {
            self.dense.commit_cache(&mut next_cache, 1, sequence_length);
            *row.cache = next_cache;
        }
        Ok(OlmoeStreamingBatchOutput {
            rows: outputs,
            layer_union_routes,
            layer_staging,
        })
    }
}

fn check_cancellation(cancellation: &CancellationToken) -> Result<()> {
    if cancellation.is_cancelled() {
        Err(MoeError::Power(PowerError::InferenceCancelled))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_batch_outputs_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OlmoeStreamingBatchRowOutput>();
        assert_send_sync::<OlmoeStreamingBatchOutput>();
    }
}

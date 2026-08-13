use std::future::Future;
use std::pin::Pin;

use a3s_power::inference::{
    EmbeddedRuntime, ExecutionPermit, PlacementTelemetry, RoutedExpertBatch,
    StagedWeightBatchReport,
};
use candle_core::Tensor;
use tokio_util::sync::CancellationToken;

use crate::olmoe::{
    OlmoeKvCache, OlmoeStreamingBatchOutput, OlmoeStreamingBatchRow, OlmoeStreamingBatchRowOutput,
    OlmoeStreamingModel,
};
use crate::qwen3_moe::{
    Qwen3MoeKvCache, Qwen3MoeStreamingBatchOutput, Qwen3MoeStreamingBatchRow,
    Qwen3MoeStreamingBatchRowOutput, Qwen3MoeStreamingModel,
};
use crate::Result;

pub struct ContinuousModelOutput<O, R, S> {
    pub rows: Vec<O>,
    pub layer_union_routes: R,
    pub layer_staging: S,
}

pub type ContinuousModelFuture<'a, O, R, S> =
    Pin<Box<dyn Future<Output = Result<ContinuousModelOutput<O, R, S>>> + Send + 'a>>;

/// Architecture adapter used by the shared continuous scheduling state
/// machine. Implementations retain ownership of numerical batch semantics.
pub trait ContinuousStreamingModel: Send + Sync + 'static {
    type Cache: Clone + Send + Sync;
    type RowOutput: Send + Sync;
    type LayerUnionRoutes: Send + Sync;
    type LayerStaging: Send + Sync;

    fn runtime(&self) -> &EmbeddedRuntime;
    fn telemetry(&self) -> PlacementTelemetry;
    fn vocab_size(&self) -> usize;
    fn validate_generation(&self, prompt: &[u32], max_new_tokens: usize) -> Result<()>;
    fn new_cache(&self) -> Self::Cache;
    fn cache_position(cache: &Self::Cache) -> usize;
    fn cache_resident_bytes(cache: &Self::Cache) -> Result<u64>;
    fn row_logits(output: &Self::RowOutput) -> &Tensor;

    fn forward_continuous_batch<'a>(
        &'a self,
        inputs: &'a [Tensor],
        caches: Vec<&'a mut Self::Cache>,
        permit: &'a ExecutionPermit,
        cancellation: &'a CancellationToken,
    ) -> ContinuousModelFuture<'a, Self::RowOutput, Self::LayerUnionRoutes, Self::LayerStaging>;
}

impl ContinuousStreamingModel for OlmoeStreamingModel {
    type Cache = OlmoeKvCache;
    type RowOutput = OlmoeStreamingBatchRowOutput;
    type LayerUnionRoutes = Vec<RoutedExpertBatch>;
    type LayerStaging = Vec<StagedWeightBatchReport>;

    fn runtime(&self) -> &EmbeddedRuntime {
        OlmoeStreamingModel::runtime(self)
    }

    fn telemetry(&self) -> PlacementTelemetry {
        OlmoeStreamingModel::telemetry(self)
    }

    fn vocab_size(&self) -> usize {
        self.config().vocab_size
    }

    fn validate_generation(&self, prompt: &[u32], max_new_tokens: usize) -> Result<()> {
        crate::olmoe::validate_generation_request(self.config(), prompt, max_new_tokens)
    }

    fn new_cache(&self) -> Self::Cache {
        OlmoeStreamingModel::new_cache(self)
    }

    fn cache_position(cache: &Self::Cache) -> usize {
        cache.position()
    }

    fn cache_resident_bytes(cache: &Self::Cache) -> Result<u64> {
        cache.resident_bytes()
    }

    fn row_logits(output: &Self::RowOutput) -> &Tensor {
        &output.logits
    }

    fn forward_continuous_batch<'a>(
        &'a self,
        inputs: &'a [Tensor],
        caches: Vec<&'a mut Self::Cache>,
        permit: &'a ExecutionPermit,
        cancellation: &'a CancellationToken,
    ) -> ContinuousModelFuture<'a, Self::RowOutput, Self::LayerUnionRoutes, Self::LayerStaging>
    {
        Box::pin(async move {
            let mut rows = inputs
                .iter()
                .zip(caches)
                .map(|(input, cache)| OlmoeStreamingBatchRow::new(input, cache))
                .collect::<Vec<_>>();
            let OlmoeStreamingBatchOutput {
                rows,
                layer_union_routes,
                layer_staging,
            } = self.forward_batch(&mut rows, permit, cancellation).await?;
            Ok(ContinuousModelOutput {
                rows,
                layer_union_routes,
                layer_staging,
            })
        })
    }
}

impl ContinuousStreamingModel for Qwen3MoeStreamingModel {
    type Cache = Qwen3MoeKvCache;
    type RowOutput = Qwen3MoeStreamingBatchRowOutput;
    type LayerUnionRoutes = Vec<Option<RoutedExpertBatch>>;
    type LayerStaging = Vec<Option<StagedWeightBatchReport>>;

    fn runtime(&self) -> &EmbeddedRuntime {
        Qwen3MoeStreamingModel::runtime(self)
    }

    fn telemetry(&self) -> PlacementTelemetry {
        Qwen3MoeStreamingModel::telemetry(self)
    }

    fn vocab_size(&self) -> usize {
        self.config().vocab_size
    }

    fn validate_generation(&self, prompt: &[u32], max_new_tokens: usize) -> Result<()> {
        crate::qwen3_moe::validate_generation_request(self.config(), prompt, max_new_tokens)
    }

    fn new_cache(&self) -> Self::Cache {
        Qwen3MoeStreamingModel::new_cache(self)
    }

    fn cache_position(cache: &Self::Cache) -> usize {
        cache.position()
    }

    fn cache_resident_bytes(cache: &Self::Cache) -> Result<u64> {
        cache.resident_bytes()
    }

    fn row_logits(output: &Self::RowOutput) -> &Tensor {
        &output.logits
    }

    fn forward_continuous_batch<'a>(
        &'a self,
        inputs: &'a [Tensor],
        caches: Vec<&'a mut Self::Cache>,
        permit: &'a ExecutionPermit,
        cancellation: &'a CancellationToken,
    ) -> ContinuousModelFuture<'a, Self::RowOutput, Self::LayerUnionRoutes, Self::LayerStaging>
    {
        Box::pin(async move {
            let mut rows = inputs
                .iter()
                .zip(caches)
                .map(|(input, cache)| Qwen3MoeStreamingBatchRow::new(input, cache))
                .collect::<Vec<_>>();
            let Qwen3MoeStreamingBatchOutput {
                rows,
                layer_union_routes,
                layer_staging,
            } = self.forward_batch(&mut rows, permit, cancellation).await?;
            Ok(ContinuousModelOutput {
                rows,
                layer_union_routes,
                layer_staging,
            })
        })
    }
}

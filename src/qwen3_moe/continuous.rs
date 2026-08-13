use a3s_power::inference::{RoutedExpertBatch, StagedWeightBatchReport};

use crate::continuous::{ContinuousBatch, ContinuousRowOutput, ContinuousStepOutput};

use super::{Qwen3MoeStreamingBatchRowOutput, Qwen3MoeStreamingModel};

pub use crate::continuous::ContinuousRequest as Qwen3MoeContinuousRequest;

pub type Qwen3MoeContinuousBatch = ContinuousBatch<Qwen3MoeStreamingModel>;
pub type Qwen3MoeContinuousRowOutput = ContinuousRowOutput<Qwen3MoeStreamingBatchRowOutput>;
pub type Qwen3MoeContinuousStepOutput = ContinuousStepOutput<
    Qwen3MoeStreamingBatchRowOutput,
    Vec<Option<RoutedExpertBatch>>,
    Vec<Option<StagedWeightBatchReport>>,
>;

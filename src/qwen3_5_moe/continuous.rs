use a3s_power::inference::{RoutedExpertBatch, StagedWeightBatchReport};

use crate::continuous::{ContinuousBatch, ContinuousRowOutput, ContinuousStepOutput};

use super::{Qwen36MoeStreamingBatchRowOutput, Qwen36MoeStreamingModel};

pub use crate::continuous::ContinuousRequest as Qwen36MoeContinuousRequest;

pub type Qwen36MoeContinuousBatch = ContinuousBatch<Qwen36MoeStreamingModel>;
pub type Qwen36MoeContinuousRowOutput = ContinuousRowOutput<Qwen36MoeStreamingBatchRowOutput>;
pub type Qwen36MoeContinuousStepOutput = ContinuousStepOutput<
    Qwen36MoeStreamingBatchRowOutput,
    Vec<RoutedExpertBatch>,
    Vec<StagedWeightBatchReport>,
>;

use a3s_power::inference::{RoutedExpertBatch, StagedWeightBatchReport};

use crate::continuous::{ContinuousBatch, ContinuousRowOutput, ContinuousStepOutput};

use super::{OlmoeStreamingBatchRowOutput, OlmoeStreamingModel};

pub use crate::continuous::ContinuousRequest as OlmoeContinuousRequest;

pub type OlmoeContinuousBatch = ContinuousBatch<OlmoeStreamingModel>;
pub type OlmoeContinuousRowOutput = ContinuousRowOutput<OlmoeStreamingBatchRowOutput>;
pub type OlmoeContinuousStepOutput = ContinuousStepOutput<
    OlmoeStreamingBatchRowOutput,
    Vec<RoutedExpertBatch>,
    Vec<StagedWeightBatchReport>,
>;

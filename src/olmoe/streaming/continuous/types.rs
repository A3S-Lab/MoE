use a3s_power::inference::{
    ExecutionBatchMemberBinding, ExecutionBatchStepEvidence, InferenceLimits, RoutedExpertBatch,
    StagedWeightBatchReport,
};

use crate::olmoe::{OlmoeSamplingConfig, OlmoeStreamingBatchRowOutput};
use crate::Result;

/// One request admitted to the model-owned continuous greedy scheduler.
#[derive(Clone)]
pub struct OlmoeContinuousRequest {
    pub binding: ExecutionBatchMemberBinding,
    pub prompt: Vec<u32>,
    pub max_new_tokens: usize,
    pub eos_token_id: Option<u32>,
    pub sampling: OlmoeSamplingConfig,
}

impl OlmoeContinuousRequest {
    pub fn new(
        binding: ExecutionBatchMemberBinding,
        prompt: Vec<u32>,
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
    ) -> Self {
        Self {
            binding,
            prompt,
            max_new_tokens,
            eos_token_id,
            sampling: OlmoeSamplingConfig::greedy(),
        }
    }

    pub fn with_sampling(mut self, sampling: OlmoeSamplingConfig) -> Self {
        self.sampling = sampling;
        self
    }

    pub fn for_identifiers(
        member_identifier: &[u8],
        state_identifier: &[u8],
        prompt: Vec<u32>,
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
        limits: &InferenceLimits,
    ) -> Result<Self> {
        Ok(Self::new(
            ExecutionBatchMemberBinding::for_identifiers(
                member_identifier,
                state_identifier,
                limits,
            )?,
            prompt,
            max_new_tokens,
            eos_token_id,
        ))
    }
}

impl std::fmt::Debug for OlmoeContinuousRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OlmoeContinuousRequest")
            .field("binding", &self.binding)
            .field("prompt_tokens", &self.prompt.len())
            .field("max_new_tokens", &self.max_new_tokens)
            .field("has_eos", &self.eos_token_id.is_some())
            .finish()
    }
}

/// One non-cancelled row produced by a continuous step.
#[derive(Debug)]
pub struct OlmoeContinuousRowOutput {
    pub member_id_sha256: String,
    pub token_id: u32,
    pub completed: bool,
    pub output: OlmoeStreamingBatchRowOutput,
}

/// Model and Power evidence for one atomically committed continuous step.
#[derive(Debug)]
pub struct OlmoeContinuousStepOutput {
    pub rows: Vec<OlmoeContinuousRowOutput>,
    pub layer_union_routes: Vec<RoutedExpertBatch>,
    pub layer_staging: Vec<StagedWeightBatchReport>,
    pub evidence: ExecutionBatchStepEvidence,
}

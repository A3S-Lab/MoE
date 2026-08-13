use a3s_power::inference::{
    ExecutionBatchMemberBinding, ExecutionBatchStepEvidence, InferenceLimits,
};

use crate::{MoeSamplingConfig, Result};

/// One request admitted to a model-owned continuous scheduler.
#[derive(Clone)]
pub struct ContinuousRequest {
    pub binding: ExecutionBatchMemberBinding,
    pub prompt: Vec<u32>,
    pub max_new_tokens: usize,
    pub eos_token_ids: Vec<u32>,
    pub sampling: MoeSamplingConfig,
}

impl ContinuousRequest {
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
            eos_token_ids: eos_token_id.into_iter().collect(),
            sampling: MoeSamplingConfig::greedy(),
        }
    }

    pub fn with_sampling(mut self, sampling: MoeSamplingConfig) -> Self {
        self.sampling = sampling;
        self
    }

    pub fn with_eos_token_ids(mut self, eos_token_ids: Vec<u32>) -> Self {
        self.eos_token_ids = eos_token_ids;
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

impl std::fmt::Debug for ContinuousRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ContinuousRequest")
            .field("binding", &self.binding)
            .field("prompt_tokens", &self.prompt.len())
            .field("max_new_tokens", &self.max_new_tokens)
            .field("eos_tokens", &self.eos_token_ids.len())
            .finish()
    }
}

/// One non-cancelled row produced by a continuous step.
#[derive(Debug)]
pub struct ContinuousRowOutput<O> {
    pub member_id_sha256: String,
    pub token_id: u32,
    pub completed: bool,
    pub output: O,
}

/// Model and Power evidence for one atomically committed continuous step.
#[derive(Debug)]
pub struct ContinuousStepOutput<O, R, S> {
    pub rows: Vec<ContinuousRowOutput<O>>,
    pub layer_union_routes: R,
    pub layer_staging: S,
    pub evidence: ExecutionBatchStepEvidence,
}

use std::time::Duration;

use a3s_power::inference::{InferenceLimits, ResidencyPolicy};

use crate::{MoeError, Result};

/// Resource policy for the Power-backed OLMoE service adapter.
#[derive(Debug, Clone)]
pub struct OlmoeBackendConfig {
    pub inference_limits: InferenceLimits,
    pub residency_policy: ResidencyPolicy,
    pub stream_capacity: usize,
    pub batch_window: Duration,
    pub default_max_tokens: usize,
}

impl OlmoeBackendConfig {
    pub fn validate(&self) -> Result<()> {
        self.inference_limits.validate()?;
        self.residency_policy.validate()?;
        if self.inference_limits.max_queued_requests == 0 || self.stream_capacity == 0 {
            return Err(MoeError::InvalidConfig(
                "service waiting and stream capacities must be non-zero".to_string(),
            ));
        }
        if self.default_max_tokens == 0
            || self.default_max_tokens > self.inference_limits.max_generated_tokens
        {
            return Err(MoeError::InvalidConfig(format!(
                "service default_max_tokens must be in 1..={}",
                self.inference_limits.max_generated_tokens
            )));
        }
        if self.batch_window > Duration::from_secs(1) {
            return Err(MoeError::InvalidConfig(
                "service batch_window cannot exceed one second".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for OlmoeBackendConfig {
    fn default() -> Self {
        let inference_limits = InferenceLimits {
            max_concurrent_requests: 4,
            max_queued_requests: 64,
            ..InferenceLimits::default()
        };
        Self {
            inference_limits,
            residency_policy: ResidencyPolicy::default(),
            stream_capacity: 32,
            batch_window: Duration::from_millis(2),
            default_max_tokens: 512,
        }
    }
}

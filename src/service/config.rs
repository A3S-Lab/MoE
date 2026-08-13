use std::time::Duration;

use a3s_power::inference::{
    DevicePreference, InferenceLimits, ResidencyPolicy, RuntimeDeviceIdentity, RuntimeDeviceKind,
};
use serde::Serialize;

use crate::{MoeError, Result};

/// Resource policy for a Power-backed MoE service adapter.
#[derive(Debug, Clone)]
pub struct OlmoeBackendConfig {
    pub device: DevicePreference,
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
        if self.device == DevicePreference::Cpu && self.residency_policy.device_cache_bytes != 0 {
            return Err(MoeError::InvalidConfig(
                "an explicit CPU device cannot reserve an accelerator cache".to_string(),
            ));
        }
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
            device: DevicePreference::Cpu,
            inference_limits,
            residency_policy: ResidencyPolicy::default(),
            stream_capacity: 32,
            batch_window: Duration::from_millis(2),
            default_max_tokens: 512,
        }
    }
}

/// Architecture-neutral name for the shared service configuration.
pub type MoeBackendConfig = OlmoeBackendConfig;

/// Qwen3-MoE compatibility name for the shared service configuration.
pub type Qwen3MoeBackendConfig = OlmoeBackendConfig;

/// Content-free evidence for Power's typed device resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OlmoeDeviceSelection {
    pub requested: DevicePreference,
    pub resolved: RuntimeDeviceIdentity,
    pub automatic_cpu_fallback: bool,
    pub effective_device_cache_bytes: u64,
}

impl OlmoeDeviceSelection {
    pub(super) fn new(
        requested: DevicePreference,
        resolved: RuntimeDeviceIdentity,
        requested_device_cache_bytes: u64,
    ) -> Self {
        let resolved_cpu = resolved.kind == RuntimeDeviceKind::Cpu;
        Self {
            requested,
            resolved,
            automatic_cpu_fallback: requested == DevicePreference::Auto && resolved_cpu,
            effective_device_cache_bytes: if resolved_cpu {
                0
            } else {
                requested_device_cache_bytes
            },
        }
    }
}

/// Architecture-neutral name for resolved service device evidence.
pub type MoeDeviceSelection = OlmoeDeviceSelection;

/// Qwen3-MoE compatibility name for resolved service device evidence.
pub type Qwen3MoeDeviceSelection = OlmoeDeviceSelection;

use std::fmt::{Display, Formatter};
use std::path::Path;

use a3s_power::inference::{InferenceLimits, ResidencyPolicy};
use serde::{Deserialize, Serialize};

use crate::{MoeError, Result};

const CONFIG_FILE: &str = "config.json";
const MAX_CONFIG_BYTES: u64 = 1024 * 1024;
const GIB: u64 = 1024 * 1024 * 1024;
#[cfg(test)]
const QWEN3_MOE_PUBLIC_WEIGHT_FILE_BYTES: u64 = 61_066_575_648;
const QWEN3_MOE_MAX_MODEL_FILES: usize = 8_192;
const QWEN3_MOE_MAX_MODEL_BYTES: u64 = 64 * GIB;
const QWEN3_MOE_MAX_STATE_BYTES: u64 = 24 * GIB;
const QWEN3_MOE_MAX_TENSOR_ELEMENTS: usize = 512 * 1024 * 1024;
const QWEN3_MOE_MAX_STAGED_WEIGHTS: usize = 128;
const QWEN3_MOE_MAX_STAGED_BYTES: u64 = 4 * GIB;
const QWEN3_MOE_MAX_INFLIGHT_BYTES: u64 = 512 * 1024 * 1024;
#[cfg(test)]
const QWEN36_MOE_PUBLIC_WEIGHT_FILE_BYTES: u64 = 71_903_776_776;
const QWEN36_MOE_MAX_MODEL_FILES: usize = 16_384;
const QWEN36_MOE_MAX_MODEL_BYTES: u64 = 80 * GIB;
const QWEN36_MOE_MAX_RESIDENT_WEIGHT_BYTES: u64 = 20 * GIB;
const QWEN36_MOE_MAX_STATE_BYTES: u64 = 48 * GIB;
const QWEN36_MOE_MAX_TENSOR_ELEMENTS: usize = 640 * 1024 * 1024;
const QWEN36_MOE_MAX_STAGED_WEIGHTS: usize = 256;

/// Model architecture implemented by this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoeArchitecture {
    Olmoe,
    Qwen3Moe,
    Qwen35Moe,
}

impl MoeArchitecture {
    /// Detect a source or packed checkpoint architecture from its bounded
    /// Hugging Face configuration file.
    pub fn detect(checkpoint: impl AsRef<Path>) -> Result<Self> {
        let path = checkpoint.as_ref().join(CONFIG_FILE);
        let metadata = path.symlink_metadata().map_err(|error| {
            MoeError::InvalidConfig(format!(
                "failed to inspect checkpoint config '{}': {error}",
                path.display()
            ))
        })?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(MoeError::InvalidConfig(format!(
                "checkpoint config '{}' must be a regular non-symlink file",
                path.display()
            )));
        }
        if metadata.len() > MAX_CONFIG_BYTES {
            return Err(MoeError::InvalidConfig(format!(
                "checkpoint config '{}' exceeds the {MAX_CONFIG_BYTES} byte limit",
                path.display()
            )));
        }
        let bytes = std::fs::read(&path).map_err(|error| {
            MoeError::InvalidConfig(format!(
                "failed to read checkpoint config '{}': {error}",
                path.display()
            ))
        })?;
        let probe = serde_json::from_slice::<ArchitectureProbe>(&bytes).map_err(|error| {
            MoeError::InvalidConfig(format!(
                "failed to parse checkpoint config '{}': {error}",
                path.display()
            ))
        })?;
        match probe.model_type.as_str() {
            "olmoe" => Ok(Self::Olmoe),
            "qwen3_moe" => Ok(Self::Qwen3Moe),
            "qwen3_5_moe" => Ok(Self::Qwen35Moe),
            model_type => Err(MoeError::InvalidConfig(format!(
                "unsupported checkpoint model_type '{model_type}'"
            ))),
        }
    }

    pub const fn model_type(self) -> &'static str {
        match self {
            Self::Olmoe => "olmoe",
            Self::Qwen3Moe => "qwen3_moe",
            Self::Qwen35Moe => "qwen3_5_moe",
        }
    }

    pub const fn default_model_name(self) -> &'static str {
        match self {
            Self::Olmoe => "olmoe",
            Self::Qwen3Moe => "qwen3-moe",
            Self::Qwen35Moe => "qwen3.6-35b-a3b",
        }
    }

    /// Bounded Power limits sized for this crate's supported public model.
    ///
    /// Callers may tighten these values for smaller checkpoints. They must not
    /// silently widen an operator-supplied policy during model loading.
    pub fn inference_limits(self) -> InferenceLimits {
        let mut limits = InferenceLimits::default();
        match self {
            Self::Olmoe => {}
            Self::Qwen3Moe => {
                // The published 30B-A3B checkpoint is about 61 GB and can
                // produce 6144 expert files under the explicit
                // one-expert-per-file option. Four full 32K F32 KV caches
                // require 24 GiB with the published attention geometry.
                limits.max_model_files = QWEN3_MOE_MAX_MODEL_FILES;
                limits.max_model_bytes = QWEN3_MOE_MAX_MODEL_BYTES;
                limits.max_state_bytes = QWEN3_MOE_MAX_STATE_BYTES;
                limits.max_tensor_elements = QWEN3_MOE_MAX_TENSOR_ELEMENTS;
            }
            Self::Qwen35Moe => {
                // The pinned 35B-A3B checkpoint contains 71.9 GB of tensor
                // bytes. A one-expert-per-file packed layout has 10,240
                // expert files, and four maximum-context F32 attention
                // sessions require just over 40 GiB of bounded state.
                limits.max_model_files = QWEN36_MOE_MAX_MODEL_FILES;
                limits.max_model_bytes = QWEN36_MOE_MAX_MODEL_BYTES;
                // The public checkpoint has about 9.1 GiB of fixed F32
                // weights. A 20 GiB hard bound lets 24 GiB accelerators use
                // a meaningful expert cache while preserving runtime
                // headroom; the operator-selected cache remains separately
                // bounded by ResidencyPolicy.
                limits.max_resident_weight_bytes = QWEN36_MOE_MAX_RESIDENT_WEIGHT_BYTES;
                limits.max_state_bytes = QWEN36_MOE_MAX_STATE_BYTES;
                limits.max_tensor_elements = QWEN36_MOE_MAX_TENSOR_ELEMENTS;
            }
        }
        limits
    }

    /// Bounded default expert-residency policy for this architecture.
    ///
    /// Entrypoints use this profile before applying operator-selected cache
    /// sizes. Library callers retain full control over an explicitly supplied
    /// policy; checkpoint loading never widens it.
    pub fn residency_policy(self) -> ResidencyPolicy {
        let mut policy = ResidencyPolicy::default();
        match self {
            Self::Olmoe => {}
            Self::Qwen3Moe => {
                // One public Qwen3-MoE layer can route to all 128 experts
                // across a prefill or continuous batch. The batch limit must
                // cover their lossless F32 form, while the independent
                // in-flight window keeps concurrent reads bounded.
                policy.max_prefetch_items = QWEN3_MOE_MAX_STAGED_WEIGHTS;
                policy.max_prefetch_bytes = QWEN3_MOE_MAX_STAGED_BYTES;
                policy.max_background_inflight_bytes = QWEN3_MOE_MAX_INFLIGHT_BYTES;
            }
            Self::Qwen35Moe => {
                // Every layer has 256 routed experts. A lossless union of all
                // public experts occupies about 3 GiB when materialized as
                // F32, within the same bounded 4 GiB byte window.
                policy.max_prefetch_items = QWEN36_MOE_MAX_STAGED_WEIGHTS;
                policy.max_prefetch_bytes = QWEN3_MOE_MAX_STAGED_BYTES;
                policy.max_background_inflight_bytes = QWEN3_MOE_MAX_INFLIGHT_BYTES;
            }
        }
        policy
    }
}

impl Display for MoeArchitecture {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.model_type())
    }
}

#[derive(Deserialize)]
struct ArchitectureProbe {
    model_type: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_supported_architectures_and_rejects_unknown_models() {
        let directory = tempfile::tempdir().unwrap();
        for (model_type, expected) in [
            ("olmoe", Some(MoeArchitecture::Olmoe)),
            ("qwen3_moe", Some(MoeArchitecture::Qwen3Moe)),
            ("qwen3_5_moe", Some(MoeArchitecture::Qwen35Moe)),
            ("dense", None),
        ] {
            std::fs::write(
                directory.path().join(CONFIG_FILE),
                serde_json::json!({ "model_type": model_type }).to_string(),
            )
            .unwrap();
            let detected = MoeArchitecture::detect(directory.path());
            match expected {
                Some(expected) => assert_eq!(detected.unwrap(), expected),
                None => assert!(detected.is_err()),
            }
        }
    }

    #[test]
    fn public_model_profiles_cover_real_checkpoint_geometry_without_allocation() {
        const LAYERS: u64 = 48;
        const KV_HEADS: u64 = 4;
        const HEAD_DIM: u64 = 128;
        const MAX_CONTEXT_TOKENS: u64 = 32_768;
        const SERVICE_CONCURRENCY: u64 = 4;
        const KEY_AND_VALUE: u64 = 2;
        const F32_BYTES: u64 = 4;
        const FULL_SESSION_KV_BYTES: u64 =
            KEY_AND_VALUE * LAYERS * KV_HEADS * HEAD_DIM * MAX_CONTEXT_TOKENS * F32_BYTES;
        const FULL_SERVICE_KV_BYTES: u64 = FULL_SESSION_KV_BYTES * SERVICE_CONCURRENCY;
        const VOCABULARY_SIZE: usize = 151_936;
        const HIDDEN_SIZE: usize = 2_048;
        const EXPERTS: usize = 128;
        const EXPERT_INTERMEDIATE_SIZE: usize = 768;
        const EMBEDDING_ELEMENTS: usize = VOCABULARY_SIZE * HIDDEN_SIZE;
        const FUSED_GATE_UP_ELEMENTS: usize = EXPERTS * 2 * EXPERT_INTERMEDIATE_SIZE * HIDDEN_SIZE;

        let power_default = InferenceLimits::default();
        let qwen = MoeArchitecture::Qwen3Moe.inference_limits();
        assert!(power_default.max_model_bytes < QWEN3_MOE_PUBLIC_WEIGHT_FILE_BYTES);
        assert!(power_default.max_model_files < 48 * 128);
        assert!(power_default.max_state_bytes < FULL_SESSION_KV_BYTES);
        assert!(power_default.max_tensor_elements < EMBEDDING_ELEMENTS);
        assert!(power_default.max_tensor_elements < FUSED_GATE_UP_ELEMENTS);
        assert!(qwen.max_model_bytes >= QWEN3_MOE_PUBLIC_WEIGHT_FILE_BYTES);
        assert!(qwen.max_model_files >= 48 * 128);
        assert_eq!(FULL_SESSION_KV_BYTES, 6 * GIB);
        assert!(qwen.max_state_bytes >= FULL_SERVICE_KV_BYTES);
        assert!(qwen.max_tensor_elements >= EMBEDDING_ELEMENTS);
        assert!(qwen.max_tensor_elements >= FUSED_GATE_UP_ELEMENTS);
        assert_eq!(MoeArchitecture::Olmoe.inference_limits(), power_default);
        qwen.validate().unwrap();
    }

    #[test]
    fn public_qwen_residency_profile_covers_a_full_f32_expert_union() {
        const HIDDEN_SIZE: u64 = 2_048;
        const EXPERT_INTERMEDIATE_SIZE: u64 = 768;
        const EXPERTS: u64 = 128;
        const BF16_BYTES: u64 = 2;
        const F32_BYTES: u64 = 4;
        const EXPERT_ELEMENTS: u64 = 3 * HIDDEN_SIZE * EXPERT_INTERMEDIATE_SIZE;
        const RECORD_HEADER_BYTES: u64 = crate::PackedExpertRecord::HEADER_BYTES as u64;
        const PUBLIC_BF16_UNION_BYTES: u64 =
            EXPERTS * (EXPERT_ELEMENTS * BF16_BYTES + RECORD_HEADER_BYTES);
        const FULL_F32_UNION_BYTES: u64 =
            EXPERTS * (EXPERT_ELEMENTS * F32_BYTES + RECORD_HEADER_BYTES);

        let power_default = ResidencyPolicy::default();
        let qwen = MoeArchitecture::Qwen3Moe.residency_policy();
        assert!(power_default.max_prefetch_bytes < PUBLIC_BF16_UNION_BYTES);
        assert!(qwen.max_prefetch_bytes >= FULL_F32_UNION_BYTES);
        assert!(qwen.max_prefetch_items >= EXPERTS as usize);
        assert!(qwen.max_background_inflight_bytes < qwen.max_prefetch_bytes);
        assert_eq!(MoeArchitecture::Olmoe.residency_policy(), power_default);
        qwen.validate().unwrap();
    }

    #[test]
    fn public_qwen36_profile_covers_checkpoint_state_and_expert_geometry() {
        const FULL_ATTENTION_LAYERS: u64 = 10;
        const KV_HEADS: u64 = 2;
        const HEAD_DIM: u64 = 256;
        const MAX_CONTEXT_TOKENS: u64 = 262_144;
        const SERVICE_CONCURRENCY: u64 = 4;
        const F32_BYTES: u64 = 4;
        const KEY_AND_VALUE: u64 = 2;
        const HIDDEN_SIZE: u64 = 2_048;
        const INTERMEDIATE_SIZE: u64 = 512;
        const EXPERTS: u64 = 256;
        const FULL_SERVICE_KV_BYTES: u64 = KEY_AND_VALUE
            * FULL_ATTENTION_LAYERS
            * KV_HEADS
            * HEAD_DIM
            * MAX_CONTEXT_TOKENS
            * F32_BYTES
            * SERVICE_CONCURRENCY;
        const FULL_F32_EXPERT_UNION: u64 =
            3 * HIDDEN_SIZE * INTERMEDIATE_SIZE * EXPERTS * F32_BYTES;

        let limits = MoeArchitecture::Qwen35Moe.inference_limits();
        let residency = MoeArchitecture::Qwen35Moe.residency_policy();
        assert!(limits.max_model_bytes >= QWEN36_MOE_PUBLIC_WEIGHT_FILE_BYTES);
        assert!(limits.max_model_files >= 40 * 256);
        assert_eq!(
            limits.max_resident_weight_bytes,
            QWEN36_MOE_MAX_RESIDENT_WEIGHT_BYTES
        );
        assert!(
            limits.max_resident_weight_bytes > InferenceLimits::default().max_resident_weight_bytes
        );
        assert!(limits.max_state_bytes >= FULL_SERVICE_KV_BYTES);
        assert!(limits.max_tensor_elements >= 256 * 2 * 512 * 2_048);
        assert!(residency.max_prefetch_items >= 256);
        assert!(residency.max_prefetch_bytes >= FULL_F32_EXPERT_UNION);
        limits.validate().unwrap();
        residency.validate().unwrap();
    }
}

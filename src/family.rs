use std::fmt::{Display, Formatter};
use std::path::Path;

use a3s_power::inference::InferenceLimits;
use serde::{Deserialize, Serialize};

use crate::{MoeError, Result};

const CONFIG_FILE: &str = "config.json";
const MAX_CONFIG_BYTES: u64 = 1024 * 1024;
const GIB: u64 = 1024 * 1024 * 1024;
#[cfg(test)]
const QWEN3_MOE_PUBLIC_WEIGHT_FILE_BYTES: u64 = 61_066_575_648;
const QWEN3_MOE_MAX_MODEL_FILES: usize = 8_192;
const QWEN3_MOE_MAX_MODEL_BYTES: u64 = 64 * GIB;

/// Model architecture implemented by this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoeArchitecture {
    Olmoe,
    Qwen3Moe,
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
            model_type => Err(MoeError::InvalidConfig(format!(
                "unsupported checkpoint model_type '{model_type}'"
            ))),
        }
    }

    pub const fn model_type(self) -> &'static str {
        match self {
            Self::Olmoe => "olmoe",
            Self::Qwen3Moe => "qwen3_moe",
        }
    }

    pub const fn default_model_name(self) -> &'static str {
        match self {
            Self::Olmoe => "olmoe",
            Self::Qwen3Moe => "qwen3-moe",
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
                // one-expert-per-file option.
                limits.max_model_files = QWEN3_MOE_MAX_MODEL_FILES;
                limits.max_model_bytes = QWEN3_MOE_MAX_MODEL_BYTES;
            }
        }
        limits
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
        let power_default = InferenceLimits::default();
        let qwen = MoeArchitecture::Qwen3Moe.inference_limits();
        assert!(power_default.max_model_bytes < QWEN3_MOE_PUBLIC_WEIGHT_FILE_BYTES);
        assert!(power_default.max_model_files < 48 * 128);
        assert!(qwen.max_model_bytes >= QWEN3_MOE_PUBLIC_WEIGHT_FILE_BYTES);
        assert!(qwen.max_model_files >= 48 * 128);
        assert_eq!(MoeArchitecture::Olmoe.inference_limits(), power_default);
        qwen.validate().unwrap();
    }
}

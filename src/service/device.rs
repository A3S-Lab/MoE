use std::fmt::{Display, Formatter};
use std::str::FromStr;

use a3s_power::inference::DevicePreference;

use crate::MoeError;

/// Strict command-line representation of Power's typed device preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OlmoeDeviceSpec {
    Auto,
    Cpu,
    Cuda { ordinal: usize },
    Metal { ordinal: usize },
}

impl OlmoeDeviceSpec {
    pub fn preference(self) -> DevicePreference {
        match self {
            Self::Auto => DevicePreference::Auto,
            Self::Cpu => DevicePreference::Cpu,
            Self::Cuda { ordinal } => DevicePreference::Cuda { ordinal },
            Self::Metal { ordinal } => DevicePreference::Metal { ordinal },
        }
    }
}

impl FromStr for OlmoeDeviceSpec {
    type Err = MoeError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            _ => {
                let (kind, ordinal) = value.split_once(':').ok_or_else(|| {
                    MoeError::InvalidConfig(
                        "device must be auto, cpu, cuda:<ordinal>, or metal:<ordinal>".to_string(),
                    )
                })?;
                let ordinal = ordinal.parse::<usize>().map_err(|error| {
                    MoeError::InvalidConfig(format!(
                        "device ordinal in '{value}' must be a non-negative integer: {error}"
                    ))
                })?;
                match kind {
                    "cuda" => Ok(Self::Cuda { ordinal }),
                    "metal" => Ok(Self::Metal { ordinal }),
                    _ => Err(MoeError::InvalidConfig(format!(
                        "unsupported device kind '{kind}'"
                    ))),
                }
            }
        }
    }
}

impl Display for OlmoeDeviceSpec {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => formatter.write_str("auto"),
            Self::Cpu => formatter.write_str("cpu"),
            Self::Cuda { ordinal } => write!(formatter, "cuda:{ordinal}"),
            Self::Metal { ordinal } => write!(formatter, "metal:{ordinal}"),
        }
    }
}

/// Architecture-neutral name for the service device argument.
pub type MoeDeviceSpec = OlmoeDeviceSpec;

/// Qwen3-MoE compatibility name for the service device argument.
pub type Qwen3MoeDeviceSpec = OlmoeDeviceSpec;
/// Qwen3.6-MoE compatibility name for the service device argument.
pub type Qwen36MoeDeviceSpec = OlmoeDeviceSpec;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_only_typed_device_specs() {
        assert_eq!(
            "auto".parse::<OlmoeDeviceSpec>().unwrap(),
            OlmoeDeviceSpec::Auto
        );
        assert_eq!(
            "cpu".parse::<OlmoeDeviceSpec>().unwrap(),
            OlmoeDeviceSpec::Cpu
        );
        assert_eq!(
            "cuda:2".parse::<OlmoeDeviceSpec>().unwrap(),
            OlmoeDeviceSpec::Cuda { ordinal: 2 }
        );
        assert_eq!(
            "metal:0".parse::<OlmoeDeviceSpec>().unwrap(),
            OlmoeDeviceSpec::Metal { ordinal: 0 }
        );
        for invalid in ["cuda", "cuda:-1", "cpu:0", "gpu:0", "CUDA:0"] {
            assert!(invalid.parse::<OlmoeDeviceSpec>().is_err(), "{invalid}");
        }
    }
}

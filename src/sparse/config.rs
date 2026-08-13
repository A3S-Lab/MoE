use serde::{Deserialize, Serialize};

use crate::{MoeError, Result};

/// Validated dimensions and routing policy for one sparse MoE layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MoeLayerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub normalize_top_k: bool,
}

impl MoeLayerConfig {
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_experts == 0
            || self.top_k == 0
        {
            return Err(MoeError::InvalidConfig(
                "MoE dimensions and top_k must be non-zero".to_string(),
            ));
        }
        if self.top_k > self.num_experts {
            return Err(MoeError::InvalidConfig(format!(
                "top_k {} exceeds num_experts {}",
                self.top_k, self.num_experts
            )));
        }
        if self.num_experts > u32::MAX as usize {
            return Err(MoeError::InvalidConfig(
                "expert count exceeds the Power routing contract".to_string(),
            ));
        }
        let expert_gate_up = self
            .intermediate_size
            .checked_mul(2)
            .and_then(|rows| rows.checked_mul(self.hidden_size));
        let expert_down = self.intermediate_size.checked_mul(self.hidden_size);
        if expert_gate_up
            .and_then(|gate_up| gate_up.checked_add(expert_down?))
            .is_none()
        {
            return Err(MoeError::InvalidConfig(
                "MoE expert dimensions overflow addressable memory".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_and_overflowing_geometry() {
        let mut config = MoeLayerConfig {
            hidden_size: 2,
            intermediate_size: 3,
            num_experts: 4,
            top_k: 2,
            normalize_top_k: false,
        };
        config.validate().unwrap();
        config.top_k = 5;
        assert!(config.validate().is_err());
        config.top_k = 1;
        config.intermediate_size = usize::MAX;
        assert!(config.validate().is_err());
    }
}

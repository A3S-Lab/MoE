use serde::{Deserialize, Serialize};

use crate::{MoeError, Result};

/// One integrity-bound file used to construct an independent public oracle.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OracleFile {
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
}

/// Tokenizer input captured by an independent public oracle.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OracleInput {
    pub text: String,
    pub add_special_tokens: bool,
    pub token_ids: Vec<u32>,
}

/// One selected expert and its routing probability.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OracleRoute {
    pub expert: u32,
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NumericComparison {
    pub tolerance: f32,
    pub max_abs_diff: f32,
    pub mismatch_count: u64,
}

impl NumericComparison {
    pub(crate) fn new(tolerance: f32) -> Self {
        Self {
            tolerance,
            max_abs_diff: 0.0,
            mismatch_count: 0,
        }
    }

    pub(crate) fn observe(&mut self, actual: f32, expected: f32) -> Result<()> {
        if !actual.is_finite() || !expected.is_finite() {
            return Err(MoeError::InvalidTensor(
                "oracle comparison encountered a non-finite value".to_string(),
            ));
        }
        let difference = (actual - expected).abs();
        let difference = if difference.is_finite() {
            difference
        } else {
            f32::MAX
        };
        self.max_abs_diff = self.max_abs_diff.max(difference);
        if difference > self.tolerance {
            self.mismatch_count = self.mismatch_count.saturating_add(1);
        }
        Ok(())
    }
}

/// Machine-readable result status from one public-checkpoint validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ValidationStatus {
    Passed,
    Failed,
}

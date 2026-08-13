use a3s_power::inference::ExecutionDigest;
use serde::{Deserialize, Serialize};

use crate::PackedScalarType;

/// Integrity-bound metadata for a completed packed Qwen3-MoE checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Qwen3MoePackedManifest {
    pub schema: String,
    pub source_weights_sha256: String,
    pub dense_weights_sha256: String,
    pub expert_weights_sha256: String,
    pub scalar_type: PackedScalarType,
    pub hidden_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_experts: usize,
    pub experts_per_file: usize,
    pub dense_files: Vec<String>,
    pub expert_files: Vec<String>,
    pub tokenizer_included: bool,
}

impl Qwen3MoePackedManifest {
    pub const SCHEMA: &'static str = "a3s.moe.qwen3-moe-packed.v1";

    /// Stable identity of the logical weights, independent of local paths.
    pub fn weights_sha256(&self) -> String {
        ExecutionDigest::utf8_text(&format!(
            "{}\0{}\0{}\0{}",
            self.schema,
            self.source_weights_sha256,
            self.dense_weights_sha256,
            self.expert_weights_sha256,
        ))
        .sha256
    }
}

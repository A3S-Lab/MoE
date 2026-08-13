use serde::{Deserialize, Serialize};

use crate::olmoe::PackedScalarType;

/// Integrity-bound metadata for a completed packed OLMoE checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OlmoePackedManifest {
    pub schema: String,
    pub source_weights_sha256: String,
    pub dense_weights_sha256: String,
    pub expert_weights_sha256: String,
    pub scalar_type: PackedScalarType,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_experts: usize,
    pub experts_per_file: usize,
    pub dense_files: Vec<String>,
    pub expert_files: Vec<String>,
    pub tokenizer_included: bool,
}

impl OlmoePackedManifest {
    pub const SCHEMA: &'static str = "a3s.moe.olmoe-packed.v1";
}

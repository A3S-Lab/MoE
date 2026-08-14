use a3s_power::inference::ExecutionDigest;
use serde::{Deserialize, Serialize};

use crate::PackedScalarType;

/// Integrity-bound metadata for a text-only packed Qwen3.6 checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Qwen36MoePackedManifest {
    pub schema: String,
    pub capabilities: Vec<String>,
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

impl Qwen36MoePackedManifest {
    pub const SCHEMA: &'static str = "a3s.moe.qwen3.6-35b-a3b-text-packed.v1";
    pub const TEXT_GENERATION_CAPABILITY: &'static str = "text-generation";

    pub fn weights_sha256(&self) -> String {
        ExecutionDigest::utf8_text(&format!(
            "{}\0{}\0{}\0{}\0{}",
            self.schema,
            self.capabilities.join(","),
            self.source_weights_sha256,
            self.dense_weights_sha256,
            self.expert_weights_sha256,
        ))
        .sha256
    }
}

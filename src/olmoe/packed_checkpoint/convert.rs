use std::fs;
use std::path::Path;

use a3s_power::inference::{InferenceLimits, WeightStore};

use crate::olmoe::checkpoint::dense_tensor_names;
use crate::olmoe::{
    packed_expert_tensor_name, OlmoeCheckpoint, OlmoePackedManifest, PackedExpertRecord,
    PackedScalarType,
};
use crate::packing::{
    absolute_destination, convert_dense_tensors, descriptor, ensure_buffer_bound, packed_scalar,
    validate_matrix_descriptor, write_packed_records,
};
use crate::{MoeError, Result};
use crate::{PackedConversionOptions, PackedConversionReport};

use super::{CONFIG_FILE, DENSE_DIRECTORY, EXPERT_DIRECTORY, MANIFEST_FILE, TOKENIZER_FILE};

pub type OlmoeConversionOptions = PackedConversionOptions;
pub type OlmoeConversionReport = PackedConversionReport;

impl OlmoeCheckpoint {
    /// Converts a validated Hugging Face checkpoint into a self-contained,
    /// bounded-memory packed checkpoint.
    ///
    /// Every dense tensor is copied to its own lossless SafeTensor file. Expert
    /// records are grouped by layer and `experts_per_file`; no conversion step
    /// buffers the complete model or layer. The final directory appears only
    /// after all collection digests and the completion manifest are written.
    pub fn convert_to_packed(
        &self,
        destination: impl AsRef<Path>,
        limits: &InferenceLimits,
        options: OlmoeConversionOptions,
    ) -> Result<OlmoeConversionReport> {
        let options = options.validate(self.config().num_experts)?;
        let destination = absolute_destination(destination.as_ref())?;
        if destination.exists() {
            return Err(MoeError::InvalidConfig(format!(
                "packed checkpoint destination '{}' already exists",
                destination.display()
            )));
        }
        let parent = destination.parent().ok_or_else(|| {
            MoeError::InvalidConfig("packed checkpoint destination has no parent".to_string())
        })?;
        fs::create_dir_all(parent)?;
        let staging = tempfile::Builder::new()
            .prefix(".a3s-moe-packing-")
            .tempdir_in(parent)?;
        let dense_root = staging.path().join(DENSE_DIRECTORY);
        let expert_root = staging.path().join(EXPERT_DIRECTORY);
        fs::create_dir(&dense_root)?;
        fs::create_dir(&expert_root)?;

        let source_store = WeightStore::open(self.root(), limits)?;
        let mut peak_buffered_bytes = 0_u64;
        let dense_files = convert_dense_tensors(
            &source_store,
            &dense_tensor_names(self.config()),
            &dense_root,
            options.max_buffer_bytes,
            &mut peak_buffered_bytes,
        )?;
        let (expert_files, scalar_type) = convert_experts(
            &source_store,
            self.config(),
            &expert_root,
            options,
            &mut peak_buffered_bytes,
        )?;

        fs::write(
            staging.path().join(CONFIG_FILE),
            serde_json::to_vec_pretty(self.config())?,
        )?;
        let tokenizer_source = self.root().join(TOKENIZER_FILE);
        let tokenizer_included = tokenizer_source.exists();
        if tokenizer_included {
            // Validate the tokenizer and vocabulary contract before preserving
            // its exact source bytes in the packed checkpoint.
            self.load_tokenizer()?;
            fs::copy(&tokenizer_source, staging.path().join(TOKENIZER_FILE))?;
        }

        let dense_store = WeightStore::open(&dense_root, limits)?;
        let expert_store = WeightStore::open(&expert_root, limits)?;
        let manifest = OlmoePackedManifest {
            schema: OlmoePackedManifest::SCHEMA.to_string(),
            source_weights_sha256: source_store.sha256().to_string(),
            dense_weights_sha256: dense_store.sha256().to_string(),
            expert_weights_sha256: expert_store.sha256().to_string(),
            scalar_type,
            hidden_size: self.config().hidden_size,
            intermediate_size: self.config().intermediate_size,
            num_hidden_layers: self.config().num_hidden_layers,
            num_experts: self.config().num_experts,
            experts_per_file: options.experts_per_file,
            dense_files: dense_files.clone(),
            expert_files: expert_files.clone(),
            tokenizer_included,
        };
        fs::write(
            staging.path().join(MANIFEST_FILE),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        let report = OlmoeConversionReport {
            destination: destination.clone(),
            source_weights_sha256: manifest.source_weights_sha256.clone(),
            dense_weights_sha256: manifest.dense_weights_sha256.clone(),
            expert_weights_sha256: manifest.expert_weights_sha256.clone(),
            dense_files: dense_files.len(),
            expert_files: expert_files.len(),
            packed_experts: self
                .config()
                .num_hidden_layers
                .checked_mul(self.config().num_experts)
                .ok_or_else(|| {
                    MoeError::InvalidConfig("packed expert count overflowed".to_string())
                })?,
            peak_buffered_bytes,
        };

        // Close mappings and readers before the directory rename, which is
        // required on Windows as well as useful for an atomic completion edge.
        drop(expert_store);
        drop(dense_store);
        drop(source_store);
        let staged_path = staging.keep();
        if let Err(error) = fs::rename(&staged_path, &destination) {
            let _ = fs::remove_dir_all(&staged_path);
            return Err(MoeError::Io(error));
        }
        Ok(report)
    }
}

fn convert_experts(
    source: &WeightStore,
    config: &crate::olmoe::OlmoeConfig,
    destination: &Path,
    options: OlmoeConversionOptions,
    peak_buffered_bytes: &mut u64,
) -> Result<(Vec<String>, PackedScalarType)> {
    let moe = config.moe_config()?;
    let mut files = Vec::new();
    let mut collection_scalar = None;
    for layer in 0..config.num_hidden_layers {
        for first in (0..config.num_experts).step_by(options.experts_per_file) {
            let end = first
                .saturating_add(options.experts_per_file)
                .min(config.num_experts);
            let mut records = Vec::<(String, Vec<u8>)>::with_capacity(end - first);
            let mut retained_bytes = 0_u64;
            for expert in first..end {
                let prefix = format!("model.layers.{layer}.mlp.experts.{expert}");
                let gate_name = format!("{prefix}.gate_proj.weight");
                let up_name = format!("{prefix}.up_proj.weight");
                let down_name = format!("{prefix}.down_proj.weight");
                let gate_descriptor = descriptor(source, &gate_name)?;
                let up_descriptor = descriptor(source, &up_name)?;
                let down_descriptor = descriptor(source, &down_name)?;
                validate_matrix_descriptor(
                    gate_descriptor,
                    moe.intermediate_size,
                    moe.hidden_size,
                )?;
                validate_matrix_descriptor(up_descriptor, moe.intermediate_size, moe.hidden_size)?;
                validate_matrix_descriptor(
                    down_descriptor,
                    moe.hidden_size,
                    moe.intermediate_size,
                )?;
                let scalar_type = packed_scalar(gate_descriptor)?;
                if packed_scalar(up_descriptor)? != scalar_type
                    || packed_scalar(down_descriptor)? != scalar_type
                {
                    return Err(MoeError::InvalidTensor(format!(
                        "expert {expert} in layer {layer} mixes scalar types"
                    )));
                }
                if collection_scalar
                    .replace(scalar_type)
                    .is_some_and(|value| value != scalar_type)
                {
                    return Err(MoeError::InvalidTensor(
                        "OLMoE experts must use one scalar type across the checkpoint".to_string(),
                    ));
                }
                let source_bytes = gate_descriptor
                    .bytes
                    .checked_add(up_descriptor.bytes)
                    .and_then(|bytes| bytes.checked_add(down_descriptor.bytes))
                    .ok_or_else(|| {
                        MoeError::InvalidTensor("expert source byte count overflowed".to_string())
                    })?;
                let record_bytes = source_bytes
                    .checked_add(PackedExpertRecord::HEADER_BYTES as u64)
                    .ok_or_else(|| {
                        MoeError::InvalidTensor("expert record byte count overflowed".to_string())
                    })?;
                let buffered = retained_bytes
                    .checked_add(source_bytes)
                    .and_then(|bytes| bytes.checked_add(record_bytes))
                    .ok_or_else(|| {
                        MoeError::InvalidTensor(
                            "conversion buffer byte count overflowed".to_string(),
                        )
                    })?;
                ensure_buffer_bound(buffered, options.max_buffer_bytes, "expert shard")?;
                *peak_buffered_bytes = (*peak_buffered_bytes).max(buffered);

                let gate = source.read_tensor_bytes(&gate_name)?;
                let up = source.read_tensor_bytes(&up_name)?;
                let down = source.read_tensor_bytes(&down_name)?;
                let encoded = PackedExpertRecord::encode_raw(
                    moe,
                    scalar_type,
                    gate.bytes(),
                    up.bytes(),
                    down.bytes(),
                )?;
                retained_bytes = retained_bytes
                    .checked_add(u64::try_from(encoded.len()).map_err(|_| {
                        MoeError::InvalidTensor("expert record is too large".to_string())
                    })?)
                    .ok_or_else(|| {
                        MoeError::InvalidTensor("retained expert bytes overflowed".to_string())
                    })?;
                let layer = u32::try_from(layer).map_err(|_| {
                    MoeError::InvalidConfig("layer index exceeds routing contract".to_string())
                })?;
                let expert = u32::try_from(expert).map_err(|_| {
                    MoeError::InvalidConfig("expert index exceeds routing contract".to_string())
                })?;
                records.push((packed_expert_tensor_name(layer, expert), encoded));
            }
            let file = format!(
                "layer-{layer:05}-experts-{first:05}-{:05}.safetensors",
                end - 1
            );
            write_packed_records(destination.join(&file), &records)?;
            files.push(file);
        }
    }
    let scalar_type = collection_scalar.ok_or_else(|| {
        MoeError::InvalidConfig("OLMoE checkpoint contains no experts".to_string())
    })?;
    Ok((files, scalar_type))
}

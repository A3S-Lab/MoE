use std::fs;
use std::path::Path;

use a3s_power::inference::{InferenceLimits, WeightStore};

use crate::packing::{
    absolute_destination, convert_dense_tensors, convert_fused_expert_tensors, descriptor,
    ensure_buffer_bound, packed_scalar, validate_matrix_descriptor, write_packed_records,
};
use crate::qwen3_moe::checkpoint::dense_tensor_names;
use crate::{
    packed_expert_tensor_name, MoeError, PackedConversionOptions, PackedConversionReport,
    PackedExpertRecord, PackedScalarType, Result,
};

use super::{
    Qwen3MoePackedManifest, CONFIG_FILE, DENSE_DIRECTORY, EXPERT_DIRECTORY, MANIFEST_FILE,
    TOKENIZER_FILE,
};
use crate::qwen3_moe::{Qwen3MoeCheckpoint, Qwen3MoeConfig, Qwen3MoeExpertLayout};

pub type Qwen3MoeConversionOptions = PackedConversionOptions;
pub type Qwen3MoeConversionReport = PackedConversionReport;

impl Qwen3MoeCheckpoint {
    /// Converts a validated Hugging Face Qwen3-MoE checkpoint into a
    /// self-contained, bounded-memory packed checkpoint.
    ///
    /// Fused expert tensors are read through verified Power subranges one
    /// expert at a time. Dense tensors are also copied in bounded chunks, so
    /// neither the fused expert arrays nor large embeddings become resident.
    pub fn convert_to_packed(
        &self,
        destination: impl AsRef<Path>,
        limits: &InferenceLimits,
        options: Qwen3MoeConversionOptions,
    ) -> Result<Qwen3MoeConversionReport> {
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
            .prefix(".a3s-moe-qwen3-packing-")
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
        let (expert_files, scalar_type, packed_experts) = convert_experts(
            &source_store,
            self.config(),
            self.expert_layout(),
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
            self.load_tokenizer()?;
            fs::copy(&tokenizer_source, staging.path().join(TOKENIZER_FILE))?;
        }

        let dense_store = WeightStore::open(&dense_root, limits)?;
        let expert_store = WeightStore::open(&expert_root, limits)?;
        let manifest = Qwen3MoePackedManifest {
            schema: Qwen3MoePackedManifest::SCHEMA.to_string(),
            source_weights_sha256: source_store.sha256().to_string(),
            dense_weights_sha256: dense_store.sha256().to_string(),
            expert_weights_sha256: expert_store.sha256().to_string(),
            scalar_type,
            hidden_size: self.config().hidden_size,
            moe_intermediate_size: self.config().moe_intermediate_size,
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
        let report = Qwen3MoeConversionReport {
            destination: destination.clone(),
            source_weights_sha256: manifest.source_weights_sha256.clone(),
            dense_weights_sha256: manifest.dense_weights_sha256.clone(),
            expert_weights_sha256: manifest.expert_weights_sha256.clone(),
            dense_files: dense_files.len(),
            expert_files: expert_files.len(),
            packed_experts,
            peak_buffered_bytes,
        };

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
    config: &Qwen3MoeConfig,
    layout: Qwen3MoeExpertLayout,
    destination: &Path,
    options: Qwen3MoeConversionOptions,
    peak_buffered_bytes: &mut u64,
) -> Result<(Vec<String>, PackedScalarType, usize)> {
    match layout {
        Qwen3MoeExpertLayout::Split => {
            convert_split_experts(source, config, destination, options, peak_buffered_bytes)
        }
        Qwen3MoeExpertLayout::Fused => {
            convert_fused_experts(source, config, destination, options, peak_buffered_bytes)
        }
    }
}

fn convert_split_experts(
    source: &WeightStore,
    config: &Qwen3MoeConfig,
    destination: &Path,
    options: Qwen3MoeConversionOptions,
    peak_buffered_bytes: &mut u64,
) -> Result<(Vec<String>, PackedScalarType, usize)> {
    let moe = config.moe_config()?;
    let mut files = Vec::new();
    let mut collection_scalar = None;
    let mut packed_experts = 0_usize;
    for layer in 0..config.num_hidden_layers {
        if !config.is_sparse_layer(layer) {
            continue;
        }
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
                    config.moe_intermediate_size,
                    config.hidden_size,
                )?;
                validate_matrix_descriptor(
                    up_descriptor,
                    config.moe_intermediate_size,
                    config.hidden_size,
                )?;
                validate_matrix_descriptor(
                    down_descriptor,
                    config.hidden_size,
                    config.moe_intermediate_size,
                )?;
                let scalar_type = packed_scalar(gate_descriptor)?;
                if packed_scalar(up_descriptor)? != scalar_type
                    || packed_scalar(down_descriptor)? != scalar_type
                {
                    return Err(MoeError::InvalidTensor(format!(
                        "Qwen3-MoE expert {expert} in layer {layer} mixes scalar types"
                    )));
                }
                if collection_scalar
                    .replace(scalar_type)
                    .is_some_and(|value| value != scalar_type)
                {
                    return Err(MoeError::InvalidTensor(
                        "Qwen3-MoE experts must use one scalar type across the checkpoint"
                            .to_string(),
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
                let layer_number = u32::try_from(layer).map_err(|_| {
                    MoeError::InvalidConfig("layer index exceeds routing contract".to_string())
                })?;
                let expert_number = u32::try_from(expert).map_err(|_| {
                    MoeError::InvalidConfig("expert index exceeds routing contract".to_string())
                })?;
                records.push((
                    packed_expert_tensor_name(layer_number, expert_number),
                    encoded,
                ));
                packed_experts = packed_experts.checked_add(1).ok_or_else(|| {
                    MoeError::InvalidConfig("packed expert count overflowed".to_string())
                })?;
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
        MoeError::InvalidConfig("Qwen3-MoE checkpoint contains no sparse experts".to_string())
    })?;
    Ok((files, scalar_type, packed_experts))
}

fn convert_fused_experts(
    source: &WeightStore,
    config: &Qwen3MoeConfig,
    destination: &Path,
    options: Qwen3MoeConversionOptions,
    peak_buffered_bytes: &mut u64,
) -> Result<(Vec<String>, PackedScalarType, usize)> {
    let sparse_layers = (0..config.num_hidden_layers)
        .filter(|layer| config.is_sparse_layer(*layer))
        .collect::<Vec<_>>();
    convert_fused_expert_tensors(
        source,
        &sparse_layers,
        config.moe_config()?,
        |layer| format!("model.layers.{layer}.mlp.experts"),
        destination,
        options,
        peak_buffered_bytes,
    )
}

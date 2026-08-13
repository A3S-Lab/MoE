use std::fs;
use std::path::Path;

use a3s_power::inference::{InferenceLimits, TensorDescriptor, WeightStore};

use crate::packing::{
    absolute_destination, convert_dense_tensors, descriptor, ensure_buffer_bound, packed_scalar,
    write_packed_records,
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
use crate::qwen3_moe::{Qwen3MoeCheckpoint, Qwen3MoeConfig};

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
    destination: &Path,
    options: Qwen3MoeConversionOptions,
    peak_buffered_bytes: &mut u64,
) -> Result<(Vec<String>, PackedScalarType, usize)> {
    let moe = config.moe_config()?;
    let section_bytes = matrix_bytes(
        config.moe_intermediate_size,
        config.hidden_size,
        packed_scalar(descriptor(source, &first_sparse_gate_up_name(config)?)?)?,
    )?;
    let gate_up_bytes = section_bytes.checked_mul(2).ok_or_else(|| {
        MoeError::InvalidTensor("fused gate/up expert byte count overflowed".to_string())
    })?;
    let mut files = Vec::new();
    let mut collection_scalar = None;
    let mut packed_experts = 0_usize;

    for layer in 0..config.num_hidden_layers {
        if !config.is_sparse_layer(layer) {
            continue;
        }
        let prefix = format!("model.layers.{layer}.mlp.experts");
        let gate_up_name = format!("{prefix}.gate_up_proj");
        let down_name = format!("{prefix}.down_proj");
        let gate_up_descriptor = descriptor(source, &gate_up_name)?;
        let down_descriptor = descriptor(source, &down_name)?;
        validate_fused_descriptors(config, gate_up_descriptor, down_descriptor)?;
        let scalar_type = packed_scalar(gate_up_descriptor)?;
        if packed_scalar(down_descriptor)? != scalar_type {
            return Err(MoeError::InvalidTensor(format!(
                "Qwen3-MoE fused experts in layer {layer} mix scalar types"
            )));
        }
        if collection_scalar
            .replace(scalar_type)
            .is_some_and(|value| value != scalar_type)
        {
            return Err(MoeError::InvalidTensor(
                "Qwen3-MoE experts must use one scalar type across the checkpoint".to_string(),
            ));
        }
        let layer_section_bytes = matrix_bytes(
            config.moe_intermediate_size,
            config.hidden_size,
            scalar_type,
        )?;
        if layer_section_bytes != section_bytes {
            return Err(MoeError::InvalidTensor(
                "Qwen3-MoE expert byte geometry changed between layers".to_string(),
            ));
        }

        for first in (0..config.num_experts).step_by(options.experts_per_file) {
            let end = first
                .saturating_add(options.experts_per_file)
                .min(config.num_experts);
            let mut records = Vec::<(String, Vec<u8>)>::with_capacity(end - first);
            let mut retained_bytes = 0_u64;
            for expert in first..end {
                let gate_up_offset = u64::try_from(expert)
                    .ok()
                    .and_then(|expert| expert.checked_mul(gate_up_bytes))
                    .ok_or_else(|| {
                        MoeError::InvalidTensor("gate/up expert offset overflowed".to_string())
                    })?;
                let down_offset = u64::try_from(expert)
                    .ok()
                    .and_then(|expert| expert.checked_mul(section_bytes))
                    .ok_or_else(|| {
                        MoeError::InvalidTensor("down expert offset overflowed".to_string())
                    })?;
                let source_bytes = gate_up_bytes.checked_add(section_bytes).ok_or_else(|| {
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

                let gate_up =
                    source.read_tensor_range(&gate_up_name, gate_up_offset, gate_up_bytes)?;
                let down = source.read_tensor_range(&down_name, down_offset, section_bytes)?;
                let split = usize::try_from(section_bytes).map_err(|_| {
                    MoeError::InvalidTensor("expert section exceeds host memory".to_string())
                })?;
                let (gate, up) = gate_up.bytes().split_at(split);
                let encoded =
                    PackedExpertRecord::encode_raw(moe, scalar_type, gate, up, down.bytes())?;
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

fn first_sparse_gate_up_name(config: &Qwen3MoeConfig) -> Result<String> {
    let layer = (0..config.num_hidden_layers)
        .find(|layer| config.is_sparse_layer(*layer))
        .ok_or_else(|| MoeError::InvalidConfig("checkpoint has no sparse layer".to_string()))?;
    Ok(format!("model.layers.{layer}.mlp.experts.gate_up_proj"))
}

fn validate_fused_descriptors(
    config: &Qwen3MoeConfig,
    gate_up: &TensorDescriptor,
    down: &TensorDescriptor,
) -> Result<()> {
    let gate_up_width = config
        .moe_intermediate_size
        .checked_mul(2)
        .ok_or_else(|| MoeError::InvalidConfig("expert gate/up width overflowed".to_string()))?;
    let expected_gate_up = [config.num_experts, gate_up_width, config.hidden_size];
    if gate_up.shape != expected_gate_up {
        return Err(MoeError::InvalidTensor(format!(
            "fused expert tensor '{}' must have shape {expected_gate_up:?}, found {:?}",
            gate_up.name, gate_up.shape
        )));
    }
    let expected_down = [
        config.num_experts,
        config.hidden_size,
        config.moe_intermediate_size,
    ];
    if down.shape != expected_down {
        return Err(MoeError::InvalidTensor(format!(
            "fused expert tensor '{}' must have shape {expected_down:?}, found {:?}",
            down.name, down.shape
        )));
    }
    packed_scalar(gate_up)?;
    packed_scalar(down)?;
    Ok(())
}

fn matrix_bytes(rows: usize, columns: usize, scalar: PackedScalarType) -> Result<u64> {
    rows.checked_mul(columns)
        .and_then(|elements| elements.checked_mul(scalar.byte_width()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| MoeError::InvalidTensor("expert matrix byte count overflowed".to_string()))
}

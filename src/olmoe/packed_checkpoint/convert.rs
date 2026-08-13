use std::fs;
use std::path::{Path, PathBuf};

use a3s_power::inference::{InferenceLimits, TensorDescriptor, TensorRead, WeightStore};
use safetensors::tensor::{serialize_to_file, TensorView};
use safetensors::Dtype;
use serde::{Deserialize, Serialize};

use crate::olmoe::checkpoint::dense_tensor_names;
use crate::olmoe::{
    packed_expert_tensor_name, OlmoeCheckpoint, OlmoePackedManifest, PackedExpertRecord,
    PackedScalarType,
};
use crate::{MoeError, Result};

use super::{CONFIG_FILE, DENSE_DIRECTORY, EXPERT_DIRECTORY, MANIFEST_FILE, TOKENIZER_FILE};

const DEFAULT_MAX_BUFFER_BYTES: u64 = 512 * 1024 * 1024;

/// Deterministic bounds for converting a Hugging Face checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OlmoeConversionOptions {
    pub experts_per_file: usize,
    pub max_buffer_bytes: u64,
}

impl Default for OlmoeConversionOptions {
    fn default() -> Self {
        Self {
            experts_per_file: 8,
            max_buffer_bytes: DEFAULT_MAX_BUFFER_BYTES,
        }
    }
}

impl OlmoeConversionOptions {
    fn validate(self, num_experts: usize) -> Result<Self> {
        if self.experts_per_file == 0 || self.experts_per_file > num_experts {
            return Err(MoeError::InvalidConfig(format!(
                "experts_per_file must be within 1..={num_experts}"
            )));
        }
        if self.max_buffer_bytes == 0 {
            return Err(MoeError::InvalidConfig(
                "max_buffer_bytes must be non-zero".to_string(),
            ));
        }
        Ok(self)
    }
}

/// Reproducible evidence returned after a completed conversion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OlmoeConversionReport {
    pub destination: PathBuf,
    pub source_weights_sha256: String,
    pub dense_weights_sha256: String,
    pub expert_weights_sha256: String,
    pub dense_files: usize,
    pub expert_files: usize,
    pub packed_experts: usize,
    pub peak_buffered_bytes: u64,
}

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
        let dense_files = convert_dense(
            &source_store,
            self.config(),
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

fn convert_dense(
    source: &WeightStore,
    config: &crate::olmoe::OlmoeConfig,
    destination: &Path,
    max_buffer_bytes: u64,
    peak_buffered_bytes: &mut u64,
) -> Result<Vec<String>> {
    let names = dense_tensor_names(config);
    let mut files = Vec::with_capacity(names.len());
    for (index, name) in names.iter().enumerate() {
        let descriptor = descriptor(source, name)?;
        ensure_buffer_bound(descriptor.bytes, max_buffer_bytes, "dense tensor")?;
        *peak_buffered_bytes = (*peak_buffered_bytes).max(descriptor.bytes);
        let read = source.read_tensor_bytes(name)?;
        let file = format!("dense-{index:05}.safetensors");
        write_raw_tensor(destination.join(&file), name, descriptor, &read)?;
        files.push(file);
    }
    Ok(files)
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
                validate_expert_descriptor(
                    gate_descriptor,
                    moe.intermediate_size,
                    moe.hidden_size,
                )?;
                validate_expert_descriptor(up_descriptor, moe.intermediate_size, moe.hidden_size)?;
                validate_expert_descriptor(
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

fn write_raw_tensor(
    path: PathBuf,
    name: &str,
    descriptor: &TensorDescriptor,
    read: &TensorRead,
) -> Result<()> {
    let view = TensorView::new(
        safetensor_dtype(&descriptor.dtype)?,
        descriptor.shape.clone(),
        read.bytes(),
    )?;
    serialize_to_file([(name, view)], None, &path)?;
    Ok(())
}

fn write_packed_records(path: PathBuf, records: &[(String, Vec<u8>)]) -> Result<()> {
    let views = records
        .iter()
        .map(|(name, bytes)| {
            TensorView::new(Dtype::U8, vec![bytes.len()], bytes.as_slice())
                .map(|view| (name.as_str(), view))
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    serialize_to_file(views, None, &path)?;
    Ok(())
}

fn descriptor<'a>(store: &'a WeightStore, name: &str) -> Result<&'a TensorDescriptor> {
    store.descriptor(name).ok_or_else(|| {
        MoeError::InvalidTensor(format!("source checkpoint is missing tensor '{name}'"))
    })
}

fn validate_expert_descriptor(
    descriptor: &TensorDescriptor,
    rows: usize,
    columns: usize,
) -> Result<()> {
    if descriptor.shape != [rows, columns] {
        return Err(MoeError::InvalidTensor(format!(
            "expert tensor '{}' must have shape [{rows}, {columns}], found {:?}",
            descriptor.name, descriptor.shape
        )));
    }
    packed_scalar(descriptor)?;
    Ok(())
}

fn packed_scalar(descriptor: &TensorDescriptor) -> Result<PackedScalarType> {
    match descriptor.dtype.as_str() {
        "f32" => Ok(PackedScalarType::F32),
        "bf16" => Ok(PackedScalarType::Bf16),
        dtype => Err(MoeError::InvalidTensor(format!(
            "expert tensor '{}' uses unsupported dtype '{dtype}'",
            descriptor.name
        ))),
    }
}

fn safetensor_dtype(dtype: &str) -> Result<Dtype> {
    match dtype {
        "f32" => Ok(Dtype::F32),
        "bf16" => Ok(Dtype::BF16),
        dtype => Err(MoeError::InvalidTensor(format!(
            "dense tensor uses unsupported dtype '{dtype}'"
        ))),
    }
}

fn ensure_buffer_bound(bytes: u64, maximum: u64, label: &str) -> Result<()> {
    if bytes > maximum {
        return Err(MoeError::InvalidConfig(format!(
            "{label} conversion requires {bytes} buffered bytes, exceeding max_buffer_bytes {maximum}"
        )));
    }
    Ok(())
}

fn absolute_destination(destination: &Path) -> Result<PathBuf> {
    if destination.as_os_str().is_empty() {
        return Err(MoeError::InvalidConfig(
            "packed checkpoint destination must not be empty".to_string(),
        ));
    }
    if destination.is_absolute() {
        Ok(destination.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(destination))
    }
}

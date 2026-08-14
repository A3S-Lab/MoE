use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use a3s_power::inference::{TensorDescriptor, WeightStore};
use safetensors::tensor::{serialize_to_file, TensorView};
use safetensors::Dtype;
use serde::{Deserialize, Serialize};

use crate::{
    packed_expert_tensor_name, MoeError, MoeLayerConfig, PackedExpertRecord, PackedScalarType,
    Result,
};

const DEFAULT_MAX_BUFFER_BYTES: u64 = 512 * 1024 * 1024;

/// Deterministic memory and file-layout bounds for checkpoint conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PackedConversionOptions {
    pub experts_per_file: usize,
    pub max_buffer_bytes: u64,
}

impl Default for PackedConversionOptions {
    fn default() -> Self {
        Self {
            experts_per_file: 8,
            max_buffer_bytes: DEFAULT_MAX_BUFFER_BYTES,
        }
    }
}

impl PackedConversionOptions {
    pub(crate) fn validate(self, num_experts: usize) -> Result<Self> {
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

/// Reproducible evidence returned after a completed packed conversion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PackedConversionReport {
    pub destination: PathBuf,
    pub source_weights_sha256: String,
    pub dense_weights_sha256: String,
    pub expert_weights_sha256: String,
    pub dense_files: usize,
    pub expert_files: usize,
    pub packed_experts: usize,
    pub peak_buffered_bytes: u64,
}

pub(crate) fn convert_dense_tensors(
    source: &WeightStore,
    names: &[String],
    destination: &Path,
    max_buffer_bytes: u64,
    peak_buffered_bytes: &mut u64,
) -> Result<Vec<String>> {
    let mut files = Vec::with_capacity(names.len());
    for (index, name) in names.iter().enumerate() {
        let descriptor = descriptor(source, name)?;
        let file = format!("dense-{index:05}.safetensors");
        write_raw_tensor_streaming(
            destination.join(&file),
            name,
            descriptor,
            source,
            max_buffer_bytes,
            peak_buffered_bytes,
        )?;
        files.push(file);
    }
    Ok(files)
}

pub(crate) fn convert_fused_expert_tensors(
    source: &WeightStore,
    layers: &[usize],
    config: MoeLayerConfig,
    tensor_prefix: impl Fn(usize) -> String,
    destination: &Path,
    options: PackedConversionOptions,
    peak_buffered_bytes: &mut u64,
) -> Result<(Vec<String>, PackedScalarType, usize)> {
    config.validate()?;
    if layers.is_empty() {
        return Err(MoeError::InvalidConfig(
            "fused expert conversion requires at least one layer".to_string(),
        ));
    }
    let first_prefix = tensor_prefix(layers[0]);
    let first_gate_up = descriptor(source, &format!("{first_prefix}.gate_up_proj"))?;
    let first_scalar = packed_scalar(first_gate_up)?;
    let section_bytes = matrix_bytes(config.intermediate_size, config.hidden_size, first_scalar)?;
    let gate_up_bytes = section_bytes.checked_mul(2).ok_or_else(|| {
        MoeError::InvalidTensor("fused gate/up expert byte count overflowed".to_string())
    })?;
    let mut files = Vec::new();
    let mut collection_scalar = None;
    let mut packed_experts = 0_usize;

    for &layer in layers {
        let prefix = tensor_prefix(layer);
        let gate_up_name = format!("{prefix}.gate_up_proj");
        let down_name = format!("{prefix}.down_proj");
        let gate_up_descriptor = descriptor(source, &gate_up_name)?;
        let down_descriptor = descriptor(source, &down_name)?;
        validate_fused_expert_descriptors(config, gate_up_descriptor, down_descriptor)?;
        let scalar_type = packed_scalar(gate_up_descriptor)?;
        if packed_scalar(down_descriptor)? != scalar_type {
            return Err(MoeError::InvalidTensor(format!(
                "fused experts in layer {layer} mix scalar types"
            )));
        }
        if collection_scalar
            .replace(scalar_type)
            .is_some_and(|value| value != scalar_type)
        {
            return Err(MoeError::InvalidTensor(
                "fused experts must use one scalar type across the checkpoint".to_string(),
            ));
        }
        let layer_section_bytes =
            matrix_bytes(config.intermediate_size, config.hidden_size, scalar_type)?;
        if layer_section_bytes != section_bytes {
            return Err(MoeError::InvalidTensor(
                "expert byte geometry changed between layers".to_string(),
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
                    PackedExpertRecord::encode_raw(config, scalar_type, gate, up, down.bytes())?;
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
    Ok((
        files,
        collection_scalar.ok_or_else(|| {
            MoeError::InvalidConfig("checkpoint contains no fused experts".to_string())
        })?,
        packed_experts,
    ))
}

fn validate_fused_expert_descriptors(
    config: MoeLayerConfig,
    gate_up: &TensorDescriptor,
    down: &TensorDescriptor,
) -> Result<()> {
    let gate_up_width = config
        .intermediate_size
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
        config.intermediate_size,
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

fn write_raw_tensor_streaming(
    path: PathBuf,
    name: &str,
    descriptor: &TensorDescriptor,
    source: &WeightStore,
    max_buffer_bytes: u64,
    peak_buffered_bytes: &mut u64,
) -> Result<()> {
    let dtype = safetensor_dtype(&descriptor.dtype)?;
    let header = single_tensor_header(name, dtype, &descriptor.shape, descriptor.bytes)?;
    let header_bytes = u64::try_from(header.len())
        .map_err(|_| MoeError::InvalidTensor("SafeTensor header is too large".to_string()))?;
    ensure_buffer_bound(header_bytes, max_buffer_bytes, "dense tensor header")?;
    *peak_buffered_bytes = (*peak_buffered_bytes).max(header_bytes);

    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(&header)?;
    let mut offset = 0_u64;
    while offset < descriptor.bytes {
        let bytes = (descriptor.bytes - offset).min(max_buffer_bytes);
        let read = source.read_tensor_range(name, offset, bytes)?;
        writer.write_all(read.bytes())?;
        *peak_buffered_bytes = (*peak_buffered_bytes).max(bytes);
        offset = offset.checked_add(bytes).ok_or_else(|| {
            MoeError::InvalidTensor("dense tensor copy offset overflowed".to_string())
        })?;
    }
    writer.flush()?;
    Ok(())
}

fn single_tensor_header(name: &str, dtype: Dtype, shape: &[usize], bytes: u64) -> Result<Vec<u8>> {
    let bytes = usize::try_from(bytes).map_err(|_| {
        MoeError::InvalidTensor("tensor byte count exceeds SafeTensor offsets".to_string())
    })?;
    let mut tensors = serde_json::Map::new();
    tensors.insert(
        name.to_string(),
        serde_json::json!({
            "dtype": dtype_name(dtype)?,
            "shape": shape,
            "data_offsets": [0, bytes],
        }),
    );
    let mut metadata = serde_json::to_vec(&tensors)?;
    let aligned = metadata
        .len()
        .checked_next_multiple_of(8)
        .ok_or_else(|| MoeError::InvalidTensor("SafeTensor header size overflowed".to_string()))?;
    metadata.resize(aligned, b' ');
    let length = u64::try_from(metadata.len())
        .map_err(|_| MoeError::InvalidTensor("SafeTensor header is too large".to_string()))?;
    let mut header = Vec::with_capacity(8 + metadata.len());
    header.extend_from_slice(&length.to_le_bytes());
    header.extend_from_slice(&metadata);
    Ok(header)
}

pub(crate) fn write_packed_records(path: PathBuf, records: &[(String, Vec<u8>)]) -> Result<()> {
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

pub(crate) fn descriptor<'a>(store: &'a WeightStore, name: &str) -> Result<&'a TensorDescriptor> {
    store.descriptor(name).ok_or_else(|| {
        MoeError::InvalidTensor(format!("source checkpoint is missing tensor '{name}'"))
    })
}

/// Exact bytes occupied after a validated tensor inventory is materialized as
/// F32. Dense model weights live outside Power's expert hierarchy, so callers
/// must include this amount in its fixed-weight admission budget.
pub(crate) fn f32_materialized_bytes(store: &WeightStore) -> Result<u64> {
    store.inventory().try_fold(0_u64, |total, descriptor| {
        let elements = descriptor
            .shape
            .iter()
            .try_fold(1_u64, |product, dimension| {
                let dimension = u64::try_from(*dimension).map_err(|_| {
                    MoeError::InvalidTensor(format!(
                        "dense tensor '{}' dimension exceeds the supported byte range",
                        descriptor.name
                    ))
                })?;
                product.checked_mul(dimension).ok_or_else(|| {
                    MoeError::InvalidTensor(format!(
                        "dense tensor '{}' element count overflowed",
                        descriptor.name
                    ))
                })
            })?;
        let bytes = elements.checked_mul(4).ok_or_else(|| {
            MoeError::InvalidTensor(format!(
                "dense tensor '{}' F32 byte count overflowed",
                descriptor.name
            ))
        })?;
        total.checked_add(bytes).ok_or_else(|| {
            MoeError::InvalidTensor("dense F32 resident byte count overflowed".to_string())
        })
    })
}

pub(crate) fn validate_matrix_descriptor(
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

pub(crate) fn packed_scalar(descriptor: &TensorDescriptor) -> Result<PackedScalarType> {
    match descriptor.dtype.as_str() {
        "f32" => Ok(PackedScalarType::F32),
        "bf16" => Ok(PackedScalarType::Bf16),
        dtype => Err(MoeError::InvalidTensor(format!(
            "expert tensor '{}' uses unsupported dtype '{dtype}'",
            descriptor.name
        ))),
    }
}

pub(crate) fn ensure_buffer_bound(bytes: u64, maximum: u64, label: &str) -> Result<()> {
    if bytes > maximum {
        return Err(MoeError::InvalidConfig(format!(
            "{label} conversion requires {bytes} buffered bytes, exceeding max_buffer_bytes {maximum}"
        )));
    }
    Ok(())
}

pub(crate) fn absolute_destination(destination: &Path) -> Result<PathBuf> {
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

fn safetensor_dtype(dtype: &str) -> Result<Dtype> {
    match dtype {
        "f32" => Ok(Dtype::F32),
        "bf16" => Ok(Dtype::BF16),
        dtype => Err(MoeError::InvalidTensor(format!(
            "dense tensor uses unsupported dtype '{dtype}'"
        ))),
    }
}

fn dtype_name(dtype: Dtype) -> Result<&'static str> {
    match dtype {
        Dtype::F32 => Ok("F32"),
        Dtype::BF16 => Ok("BF16"),
        other => Err(MoeError::InvalidTensor(format!(
            "streaming SafeTensor writer does not support {other:?}"
        ))),
    }
}

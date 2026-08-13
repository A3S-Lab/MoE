use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use a3s_power::inference::{TensorDescriptor, WeightStore};
use safetensors::tensor::{serialize_to_file, TensorView};
use safetensors::Dtype;
use serde::{Deserialize, Serialize};

use crate::{MoeError, PackedScalarType, Result};

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

use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::olmoe::OlmoeMoeConfig;
use crate::{MoeError, Result};

const MAGIC: &[u8; 8] = b"A3SMOEPK";
const VERSION: u16 = 1;
const HEADER_BYTES: usize = 64;

/// Scalar encoding used by a lossless packed expert record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u16)]
#[serde(rename_all = "kebab-case")]
pub enum PackedScalarType {
    F32 = 1,
    Bf16 = 2,
}

impl PackedScalarType {
    fn from_code(code: u16) -> Result<Self> {
        match code {
            1 => Ok(Self::F32),
            2 => Ok(Self::Bf16),
            _ => Err(MoeError::InvalidTensor(format!(
                "packed expert uses unsupported scalar type {code}"
            ))),
        }
    }

    pub const fn byte_width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }
}

/// Versioned, lossless gate/up/down expert record.
///
/// Power treats the encoded record as one opaque U8 SafeTensor. This parser
/// validates every dimension and byte range before model-owned materialization.
#[derive(Debug, Clone)]
pub struct PackedExpertRecord {
    scalar_type: PackedScalarType,
    config: OlmoeMoeConfig,
    encoded: Vec<u8>,
    gate: Range<usize>,
    up: Range<usize>,
    down: Range<usize>,
}

impl PackedExpertRecord {
    pub const HEADER_BYTES: usize = HEADER_BYTES;

    /// Encodes finite F32 matrices in gate, up, down order.
    pub fn encode_f32(
        config: OlmoeMoeConfig,
        gate: &[f32],
        up: &[f32],
        down: &[f32],
    ) -> Result<Vec<u8>> {
        let encode = |values: &[f32], label: &str| -> Result<Vec<u8>> {
            if values.iter().any(|value| !value.is_finite()) {
                return Err(MoeError::InvalidTensor(format!(
                    "packed expert {label} values must be finite"
                )));
            }
            let byte_count = values.len().checked_mul(4).ok_or_else(|| {
                MoeError::InvalidTensor("packed expert byte count overflowed".to_string())
            })?;
            let mut bytes = Vec::with_capacity(byte_count);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            Ok(bytes)
        };
        Self::encode_raw(
            config,
            PackedScalarType::F32,
            &encode(gate, "gate")?,
            &encode(up, "up")?,
            &encode(down, "down")?,
        )
    }

    /// Wraps exact little-endian F32 or BF16 checkpoint bytes without changing
    /// their precision.
    pub fn encode_raw(
        config: OlmoeMoeConfig,
        scalar_type: PackedScalarType,
        gate: &[u8],
        up: &[u8],
        down: &[u8],
    ) -> Result<Vec<u8>> {
        config.validate()?;
        let hidden_size = u32::try_from(config.hidden_size).map_err(|_| {
            MoeError::InvalidConfig("hidden_size does not fit the packed format".to_string())
        })?;
        let intermediate_size = u32::try_from(config.intermediate_size).map_err(|_| {
            MoeError::InvalidConfig("intermediate_size does not fit the packed format".to_string())
        })?;
        let expected = section_bytes(config, scalar_type)?;
        for (label, bytes) in [("gate", gate), ("up", up), ("down", down)] {
            if bytes.len() != expected {
                return Err(MoeError::InvalidTensor(format!(
                    "packed expert {label} section requires {expected} bytes, found {}",
                    bytes.len()
                )));
            }
            validate_scalar_bytes(bytes, scalar_type, label)?;
        }
        let payload_bytes = expected.checked_mul(3).ok_or_else(|| {
            MoeError::InvalidTensor("packed expert payload length overflowed".to_string())
        })?;
        let total_bytes = HEADER_BYTES.checked_add(payload_bytes).ok_or_else(|| {
            MoeError::InvalidTensor("packed expert record length overflowed".to_string())
        })?;
        let section_u64 = u64::try_from(expected).map_err(|_| {
            MoeError::InvalidTensor("packed expert section is too large".to_string())
        })?;
        let payload_u64 = u64::try_from(payload_bytes).map_err(|_| {
            MoeError::InvalidTensor("packed expert payload is too large".to_string())
        })?;

        let mut encoded = Vec::with_capacity(total_bytes);
        encoded.extend_from_slice(MAGIC);
        encoded.extend_from_slice(&VERSION.to_le_bytes());
        encoded.extend_from_slice(&(scalar_type as u16).to_le_bytes());
        encoded.extend_from_slice(&(HEADER_BYTES as u32).to_le_bytes());
        encoded.extend_from_slice(&hidden_size.to_le_bytes());
        encoded.extend_from_slice(&intermediate_size.to_le_bytes());
        encoded.extend_from_slice(&section_u64.to_le_bytes());
        encoded.extend_from_slice(&section_u64.to_le_bytes());
        encoded.extend_from_slice(&section_u64.to_le_bytes());
        encoded.extend_from_slice(&payload_u64.to_le_bytes());
        encoded.extend_from_slice(&0_u64.to_le_bytes());
        encoded.extend_from_slice(gate);
        encoded.extend_from_slice(up);
        encoded.extend_from_slice(down);
        Ok(encoded)
    }

    /// Parses and dimension-binds one encoded expert record.
    pub fn parse(encoded: Vec<u8>, expected: OlmoeMoeConfig) -> Result<Self> {
        expected.validate()?;
        if encoded.len() < HEADER_BYTES {
            return Err(MoeError::InvalidTensor(format!(
                "packed expert record is truncated: expected at least {HEADER_BYTES} bytes, found {}",
                encoded.len()
            )));
        }
        if read_exact::<8>(&encoded, 0, "magic")? != *MAGIC {
            return Err(MoeError::InvalidTensor(
                "packed expert magic is invalid".to_string(),
            ));
        }
        let version = read_u16(&encoded, 8, "version")?;
        if version != VERSION {
            return Err(MoeError::InvalidTensor(format!(
                "packed expert version {version} is unsupported"
            )));
        }
        let scalar_type = PackedScalarType::from_code(read_u16(&encoded, 10, "scalar type")?)?;
        let header_bytes = read_u32(&encoded, 12, "header length")? as usize;
        if header_bytes != HEADER_BYTES {
            return Err(MoeError::InvalidTensor(format!(
                "packed expert header length must be {HEADER_BYTES}, found {header_bytes}"
            )));
        }
        let hidden_size = read_u32(&encoded, 16, "hidden size")? as usize;
        let intermediate_size = read_u32(&encoded, 20, "intermediate size")? as usize;
        if hidden_size != expected.hidden_size || intermediate_size != expected.intermediate_size {
            return Err(MoeError::InvalidTensor(format!(
                "packed expert dimensions [{hidden_size}, {intermediate_size}] do not match expected [{}, {}]",
                expected.hidden_size, expected.intermediate_size
            )));
        }
        let expected_section = section_bytes(expected, scalar_type)?;
        let gate_bytes = read_length(&encoded, 24, "gate length")?;
        let up_bytes = read_length(&encoded, 32, "up length")?;
        let down_bytes = read_length(&encoded, 40, "down length")?;
        for (label, actual) in [("gate", gate_bytes), ("up", up_bytes), ("down", down_bytes)] {
            if actual != expected_section {
                return Err(MoeError::InvalidTensor(format!(
                    "packed expert {label} section requires {expected_section} bytes, found {actual}"
                )));
            }
        }
        let payload_bytes = read_length(&encoded, 48, "payload length")?;
        let expected_payload = expected_section.checked_mul(3).ok_or_else(|| {
            MoeError::InvalidTensor("packed expert payload length overflowed".to_string())
        })?;
        if payload_bytes != expected_payload {
            return Err(MoeError::InvalidTensor(format!(
                "packed expert payload requires {expected_payload} bytes, found {payload_bytes}"
            )));
        }
        if read_u64(&encoded, 56, "reserved field")? != 0 {
            return Err(MoeError::InvalidTensor(
                "packed expert reserved header bytes must be zero".to_string(),
            ));
        }
        let expected_total = HEADER_BYTES.checked_add(expected_payload).ok_or_else(|| {
            MoeError::InvalidTensor("packed expert total length overflowed".to_string())
        })?;
        if encoded.len() != expected_total {
            return Err(MoeError::InvalidTensor(format!(
                "packed expert record requires exactly {expected_total} bytes, found {}",
                encoded.len()
            )));
        }

        let gate = HEADER_BYTES..HEADER_BYTES + expected_section;
        let up = gate.end..gate.end + expected_section;
        let down = up.end..up.end + expected_section;
        validate_scalar_bytes(&encoded[gate.clone()], scalar_type, "gate")?;
        validate_scalar_bytes(&encoded[up.clone()], scalar_type, "up")?;
        validate_scalar_bytes(&encoded[down.clone()], scalar_type, "down")?;
        Ok(Self {
            scalar_type,
            config: expected,
            encoded,
            gate,
            up,
            down,
        })
    }

    pub fn scalar_type(&self) -> PackedScalarType {
        self.scalar_type
    }

    pub fn config(&self) -> OlmoeMoeConfig {
        self.config
    }

    pub fn encoded_bytes(&self) -> &[u8] {
        &self.encoded
    }

    pub(crate) fn decode_f32(&self) -> Result<DecodedPackedExpert> {
        Ok(DecodedPackedExpert {
            gate: decode_scalar_bytes(&self.encoded[self.gate.clone()], self.scalar_type, "gate")?,
            up: decode_scalar_bytes(&self.encoded[self.up.clone()], self.scalar_type, "up")?,
            down: decode_scalar_bytes(&self.encoded[self.down.clone()], self.scalar_type, "down")?,
        })
    }
}

pub(crate) struct DecodedPackedExpert {
    pub(crate) gate: Vec<f32>,
    pub(crate) up: Vec<f32>,
    pub(crate) down: Vec<f32>,
}

/// Canonical tensor name for one packed expert record.
pub fn packed_expert_tensor_name(layer: u32, expert: u32) -> String {
    format!("model.layers.{layer}.mlp.experts.{expert}.packed")
}

fn section_bytes(config: OlmoeMoeConfig, scalar_type: PackedScalarType) -> Result<usize> {
    config
        .hidden_size
        .checked_mul(config.intermediate_size)
        .and_then(|elements| elements.checked_mul(scalar_type.byte_width()))
        .ok_or_else(|| {
            MoeError::InvalidTensor("packed expert section length overflowed".to_string())
        })
}

fn validate_scalar_bytes(bytes: &[u8], scalar_type: PackedScalarType, label: &str) -> Result<()> {
    let width = scalar_type.byte_width();
    if !bytes.len().is_multiple_of(width) {
        return Err(MoeError::InvalidTensor(format!(
            "packed expert {label} byte length is not aligned to {width}"
        )));
    }
    for (index, chunk) in bytes.chunks_exact(width).enumerate() {
        let value = decode_scalar(chunk, scalar_type, label)?;
        if !value.is_finite() {
            return Err(MoeError::InvalidTensor(format!(
                "packed expert {label} contains a non-finite value at element {index}"
            )));
        }
    }
    Ok(())
}

fn decode_scalar_bytes(
    bytes: &[u8],
    scalar_type: PackedScalarType,
    label: &str,
) -> Result<Vec<f32>> {
    let width = scalar_type.byte_width();
    if !bytes.len().is_multiple_of(width) {
        return Err(MoeError::InvalidTensor(format!(
            "packed expert {label} byte length is not aligned to {width}"
        )));
    }
    let mut values = Vec::with_capacity(bytes.len() / width);
    for chunk in bytes.chunks_exact(width) {
        values.push(decode_scalar(chunk, scalar_type, label)?);
    }
    Ok(values)
}

fn decode_scalar(chunk: &[u8], scalar_type: PackedScalarType, label: &str) -> Result<f32> {
    match scalar_type {
        PackedScalarType::F32 => {
            let encoded: [u8; 4] = chunk.try_into().map_err(|_| {
                MoeError::InvalidTensor(format!(
                    "packed expert {label} contains a truncated F32 value"
                ))
            })?;
            Ok(f32::from_le_bytes(encoded))
        }
        PackedScalarType::Bf16 => {
            let encoded: [u8; 2] = chunk.try_into().map_err(|_| {
                MoeError::InvalidTensor(format!(
                    "packed expert {label} contains a truncated BF16 value"
                ))
            })?;
            Ok(f32::from_bits(u32::from(u16::from_le_bytes(encoded)) << 16))
        }
    }
}

fn read_length(bytes: &[u8], offset: usize, label: &str) -> Result<usize> {
    usize::try_from(read_u64(bytes, offset, label)?).map_err(|_| {
        MoeError::InvalidTensor(format!("packed expert {label} does not fit this platform"))
    })
}

fn read_u16(bytes: &[u8], offset: usize, label: &str) -> Result<u16> {
    Ok(u16::from_le_bytes(read_exact(bytes, offset, label)?))
}

fn read_u32(bytes: &[u8], offset: usize, label: &str) -> Result<u32> {
    Ok(u32::from_le_bytes(read_exact(bytes, offset, label)?))
}

fn read_u64(bytes: &[u8], offset: usize, label: &str) -> Result<u64> {
    Ok(u64::from_le_bytes(read_exact(bytes, offset, label)?))
}

fn read_exact<const N: usize>(bytes: &[u8], offset: usize, label: &str) -> Result<[u8; N]> {
    let end = offset.checked_add(N).ok_or_else(|| {
        MoeError::InvalidTensor(format!("packed expert {label} offset overflowed"))
    })?;
    bytes
        .get(offset..end)
        .ok_or_else(|| MoeError::InvalidTensor(format!("packed expert {label} is truncated")))?
        .try_into()
        .map_err(|_| MoeError::InvalidTensor(format!("packed expert {label} is malformed")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> OlmoeMoeConfig {
        OlmoeMoeConfig {
            hidden_size: 2,
            intermediate_size: 1,
            num_experts: 2,
            top_k: 1,
            normalize_top_k: false,
        }
    }

    #[test]
    fn f32_record_round_trips_without_changing_values() {
        let encoded =
            PackedExpertRecord::encode_f32(config(), &[1.0, -2.0], &[3.0, 4.0], &[5.0, -6.0])
                .unwrap();
        let record = PackedExpertRecord::parse(encoded.clone(), config()).unwrap();
        let decoded = record.decode_f32().unwrap();

        assert_eq!(record.scalar_type(), PackedScalarType::F32);
        assert_eq!(record.encoded_bytes(), encoded);
        assert_eq!(decoded.gate, [1.0, -2.0]);
        assert_eq!(decoded.up, [3.0, 4.0]);
        assert_eq!(decoded.down, [5.0, -6.0]);
    }

    #[test]
    fn parser_rejects_corruption_truncation_and_dimension_mismatch() {
        let encoded =
            PackedExpertRecord::encode_f32(config(), &[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0])
                .unwrap();
        let mut corrupt = encoded.clone();
        corrupt[0] ^= 0xff;
        assert!(PackedExpertRecord::parse(corrupt, config()).is_err());
        assert!(
            PackedExpertRecord::parse(encoded[..encoded.len() - 1].to_vec(), config()).is_err()
        );

        let mut wrong = config();
        wrong.hidden_size = 3;
        assert!(PackedExpertRecord::parse(encoded, wrong).is_err());
    }

    #[test]
    fn raw_bf16_record_is_decoded_losslessly() {
        let values = [1.0_f32, -2.5];
        let bytes = values
            .into_iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>();
        let encoded = PackedExpertRecord::encode_raw(
            config(),
            PackedScalarType::Bf16,
            &bytes,
            &bytes,
            &bytes,
        )
        .unwrap();
        let record = PackedExpertRecord::parse(encoded, config()).unwrap();
        assert_eq!(record.scalar_type(), PackedScalarType::Bf16);
        assert_eq!(record.decode_f32().unwrap().gate, values);
    }
}

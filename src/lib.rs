//! Model-owned mixture-of-experts inference for A3S Power.
//!
//! Power owns model-neutral scheduling, residency, integrity, and service
//! composition. This crate owns exact model architecture and numerical
//! semantics, starting with OLMoE.

mod checkpoint;
#[doc(hidden)]
pub mod continuous;
mod decoder;
mod device;
mod error;
mod family;
mod matrix;
pub mod olmoe;
mod packing;
pub mod qwen3_5_moe;
pub mod qwen3_moe;
#[cfg(feature = "server")]
pub mod service;
mod sparse;
mod tokenizer;
#[cfg(feature = "validation")]
mod validation;

pub use decoder::DecoderKvCache;
pub use device::{MoeDeviceSpec, OlmoeDeviceSpec, Qwen36MoeDeviceSpec, Qwen3MoeDeviceSpec};
pub use error::{MoeError, Result};
pub use family::MoeArchitecture;
pub use matrix::Matrix;
pub use olmoe::{
    packed_expert_tensor_name, OlmoeSampler as MoeSampler,
    OlmoeSamplingConfig as MoeSamplingConfig, OlmoeStreamingMlp as PowerStreamingMlp,
    OlmoeStreamingMlpOutput as PowerStreamingMlpOutput, PackedExpertRecord, PackedScalarType,
};
pub use packing::{PackedConversionOptions, PackedConversionReport};
pub use sparse::{
    GatedExpertWeights, MoeLayerConfig, SparseMoeLayer, SparseMoeOutput, TopKRouter,
    TopKRouterOutput,
};
pub use tokenizer::{MoeDecodeStream, MoeTokenizer};

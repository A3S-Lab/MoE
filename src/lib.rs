//! Model-owned mixture-of-experts inference for A3S Power.
//!
//! Power owns model-neutral scheduling, residency, integrity, and service
//! composition. This crate owns exact model architecture and numerical
//! semantics, starting with OLMoE.

mod checkpoint;
mod decoder;
mod error;
mod matrix;
pub mod olmoe;
mod packing;
pub mod qwen3_moe;
#[cfg(feature = "server")]
pub mod service;
mod sparse;
mod tokenizer;

pub use decoder::DecoderKvCache;
pub use error::{MoeError, Result};
pub use matrix::Matrix;
pub use olmoe::{
    packed_expert_tensor_name, OlmoeStreamingMlp as PowerStreamingMlp,
    OlmoeStreamingMlpOutput as PowerStreamingMlpOutput, PackedExpertRecord, PackedScalarType,
};
pub use packing::{PackedConversionOptions, PackedConversionReport};
pub use sparse::{
    GatedExpertWeights, MoeLayerConfig, SparseMoeLayer, SparseMoeOutput, TopKRouter,
    TopKRouterOutput,
};
pub use tokenizer::{MoeDecodeStream, MoeTokenizer};

mod attention;
mod batch;
mod checkpoint;
mod config;
mod continuous;
mod dense;
mod mlp;
mod model;
mod packed_checkpoint;
mod streaming;

pub use crate::DecoderKvCache as Qwen3MoeKvCache;
pub use crate::{MoeDecodeStream as Qwen3MoeDecodeStream, MoeTokenizer as Qwen3MoeTokenizer};
pub use batch::{
    Qwen3MoeStreamingBatchOutput, Qwen3MoeStreamingBatchRow, Qwen3MoeStreamingBatchRowOutput,
};
pub use checkpoint::Qwen3MoeCheckpoint;
pub use config::{Qwen3MoeConfig, Qwen3MoeTokenIds};
pub use continuous::{
    Qwen3MoeContinuousBatch, Qwen3MoeContinuousRequest, Qwen3MoeContinuousRowOutput,
    Qwen3MoeContinuousStepOutput,
};
pub(crate) use dense::validate_generation_request;
pub use model::{Qwen3MoeCpuModel, Qwen3MoeForwardOutput};
pub use packed_checkpoint::{
    Qwen3MoeConversionOptions, Qwen3MoeConversionReport, Qwen3MoePackedCheckpoint,
    Qwen3MoePackedManifest,
};
pub use streaming::{Qwen3MoeStreamingForwardOutput, Qwen3MoeStreamingModel};

pub use crate::{
    GatedExpertWeights as Qwen3MoeExpertWeights, SparseMoeLayer as Qwen3MoeSparseLayer,
    SparseMoeOutput as Qwen3MoeSparseOutput, TopKRouter as Qwen3MoeRouter,
    TopKRouterOutput as Qwen3MoeRouterOutput,
};

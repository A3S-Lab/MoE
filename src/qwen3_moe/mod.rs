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
#[cfg(feature = "validation")]
mod validation;

pub use crate::DecoderKvCache as Qwen3MoeKvCache;
pub use crate::{MoeDecodeStream as Qwen3MoeDecodeStream, MoeTokenizer as Qwen3MoeTokenizer};
pub use batch::{
    Qwen3MoeStreamingBatchOutput, Qwen3MoeStreamingBatchRow, Qwen3MoeStreamingBatchRowOutput,
};
pub use checkpoint::{Qwen3MoeCheckpoint, Qwen3MoeExpertLayout};
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
#[cfg(feature = "validation")]
pub use validation::{
    validate_public_checkpoint, Qwen3MoeNumericComparison, Qwen3MoeOracleFile, Qwen3MoeOracleInput,
    Qwen3MoeOracleModel, Qwen3MoeOracleOutput, Qwen3MoeOracleRoute, Qwen3MoePublicOracle,
    Qwen3MoeValidationOptions, Qwen3MoeValidationReport, Qwen3MoeValidationStatus,
    Qwen3MoeValidationTolerances, QWEN3_MOE_PUBLIC_MODEL_ID, QWEN3_MOE_PUBLIC_MODEL_REVISION,
    QWEN3_MOE_PUBLIC_ORACLE_SCHEMA, QWEN3_MOE_TRANSFORMERS_DTYPE, QWEN3_MOE_TRANSFORMERS_REVISION,
    QWEN3_MOE_TRANSFORMERS_SOURCE_SHA256, QWEN3_MOE_VALIDATION_SCHEMA,
};

pub use crate::{
    GatedExpertWeights as Qwen3MoeExpertWeights, SparseMoeLayer as Qwen3MoeSparseLayer,
    SparseMoeOutput as Qwen3MoeSparseOutput, TopKRouter as Qwen3MoeRouter,
    TopKRouterOutput as Qwen3MoeRouterOutput,
};

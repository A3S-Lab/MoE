/// Error returned by model-owned MoE validation or execution.
#[derive(Debug, thiserror::Error)]
pub enum MoeError {
    #[error("invalid MoE configuration: {0}")]
    InvalidConfig(String),

    #[error("invalid MoE tensor: {0}")]
    InvalidTensor(String),

    #[error("MoE inference failed: {0}")]
    Inference(String),

    #[error("failed to read model data: {0}")]
    Io(#[from] std::io::Error),

    #[error("failed to parse model JSON: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Power rejected the routed expert batch: {0}")]
    Power(#[from] a3s_power::error::PowerError),

    #[error("tensor execution failed: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("SafeTensor operation failed: {0}")]
    SafeTensor(#[from] safetensors::SafeTensorError),

    #[error("tokenizer operation failed: {0}")]
    Tokenizer(String),
}

pub type Result<T> = std::result::Result<T, MoeError>;

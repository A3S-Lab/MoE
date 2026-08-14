//! Power service integration for supported packed MoE checkpoints.

mod architecture;
mod backend;
mod config;
mod device;
mod load;
mod request;
mod source;
mod stream;
mod worker;

pub use backend::{OlmoeBackend, Qwen36MoeBackend, Qwen3MoeBackend};
pub use config::{
    MoeBackendConfig, MoeDeviceSelection, OlmoeBackendConfig, OlmoeDeviceSelection,
    Qwen36MoeBackendConfig, Qwen36MoeDeviceSelection, Qwen3MoeBackendConfig,
    Qwen3MoeDeviceSelection,
};
pub use device::{MoeDeviceSpec, OlmoeDeviceSpec, Qwen36MoeDeviceSpec, Qwen3MoeDeviceSpec};

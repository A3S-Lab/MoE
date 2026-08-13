//! Model-owned mixture-of-experts inference for A3S Power.
//!
//! Power owns model-neutral scheduling, residency, integrity, and service
//! composition. This crate owns exact model architecture and numerical
//! semantics, starting with OLMoE.

mod error;
mod matrix;
pub mod olmoe;
#[cfg(feature = "server")]
pub mod service;

pub use error::{MoeError, Result};
pub use matrix::Matrix;

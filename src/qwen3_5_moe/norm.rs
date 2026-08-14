use candle_core::Tensor;
use candle_nn::{Module, RmsNorm, VarBuilder};

use crate::Result;

/// Qwen3.6 stores a zero-centered scale and applies `(1 + weight)`.
#[derive(Debug, Clone)]
pub(super) struct OffsetRmsNorm(RmsNorm);

impl OffsetRmsNorm {
    pub(super) fn load(size: usize, eps: f64, builder: VarBuilder<'_>) -> Result<Self> {
        let offset_weight = builder.get(size, "weight")?;
        Ok(Self(RmsNorm::new(offset_weight.affine(1.0, 1.0)?, eps)))
    }

    pub(super) fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(self.0.forward(input)?)
    }
}

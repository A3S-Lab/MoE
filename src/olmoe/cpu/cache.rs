use candle_core::Tensor;

use crate::{MoeError, Result};

/// Per-session OLMoE key/value cache.
///
/// The cache is separate from immutable model weights so one model can serve
/// multiple sessions without sharing request state.
#[derive(Debug, Clone)]
pub struct OlmoeKvCache {
    pub(super) layers: Vec<LayerKvCache>,
    position: usize,
    max_position_embeddings: usize,
    batch_size: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub(super) struct LayerKvCache {
    pub(super) key: Option<Tensor>,
    pub(super) value: Option<Tensor>,
}

impl OlmoeKvCache {
    pub(super) fn new(layer_count: usize, max_position_embeddings: usize) -> Self {
        Self {
            layers: vec![LayerKvCache::default(); layer_count],
            position: 0,
            max_position_embeddings,
            batch_size: None,
        }
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    /// Exact bytes currently referenced by key/value tensors in this session.
    pub fn resident_bytes(&self) -> Result<u64> {
        self.layers.iter().try_fold(0_u64, |total, layer| {
            [&layer.key, &layer.value]
                .into_iter()
                .flatten()
                .try_fold(total, |total, tensor| {
                    let bytes = tensor
                        .elem_count()
                        .checked_mul(tensor.dtype().size_in_bytes())
                        .and_then(|bytes| u64::try_from(bytes).ok())
                        .ok_or_else(|| {
                            MoeError::InvalidTensor(
                                "KV cache resident byte count overflowed".to_string(),
                            )
                        })?;
                    total.checked_add(bytes).ok_or_else(|| {
                        MoeError::InvalidTensor(
                            "KV cache aggregate byte count overflowed".to_string(),
                        )
                    })
                })
        })
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            *layer = LayerKvCache::default();
        }
        self.position = 0;
        self.batch_size = None;
    }

    pub(super) fn validate_step(
        &self,
        layer_count: usize,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<()> {
        if self.layers.len() != layer_count {
            return Err(MoeError::InvalidTensor(format!(
                "KV cache has {} layers, model requires {layer_count}",
                self.layers.len()
            )));
        }
        if sequence_length == 0 || batch_size == 0 {
            return Err(MoeError::InvalidTensor(
                "KV cache step requires a non-empty token batch".to_string(),
            ));
        }
        if self
            .batch_size
            .is_some_and(|cached_batch| cached_batch != batch_size)
        {
            return Err(MoeError::InvalidTensor(format!(
                "KV cache batch size {:?} cannot change to {batch_size}",
                self.batch_size
            )));
        }
        let end = self
            .position
            .checked_add(sequence_length)
            .ok_or_else(|| MoeError::InvalidTensor("KV cache position overflowed".to_string()))?;
        if end > self.max_position_embeddings {
            return Err(MoeError::InvalidTensor(format!(
                "KV cache step ends at position {end}, beyond maximum {}",
                self.max_position_embeddings
            )));
        }
        Ok(())
    }

    pub(super) fn commit_step(&mut self, batch_size: usize, sequence_length: usize) {
        self.batch_size = Some(batch_size);
        self.position += sequence_length;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_clears_position_and_batch_binding() {
        let mut cache = OlmoeKvCache::new(2, 8);
        cache.validate_step(2, 1, 3).unwrap();
        cache.commit_step(1, 3);
        assert!(cache.validate_step(2, 2, 1).is_err());
        cache.reset();
        assert_eq!(cache.position(), 0);
        assert!(cache.validate_step(2, 2, 1).is_ok());
        assert_eq!(cache.resident_bytes().unwrap(), 0);
    }
}

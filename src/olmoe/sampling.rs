use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{MoeError, Result};

/// Deterministic generation policy for one OLMoE request.
///
/// The default is greedy to preserve the numerical-reference behavior. Service
/// adapters may choose protocol-specific defaults explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OlmoeSamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub repeat_last_n: Option<usize>,
    pub seed: u64,
}

impl OlmoeSamplingConfig {
    pub const fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repeat_last_n: None,
            seed: 0,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(invalid("temperature must be finite and non-negative"));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(invalid("top_p must be finite and in (0, 1]"));
        }
        if !self.min_p.is_finite() || self.min_p < 0.0 || self.min_p > 1.0 {
            return Err(invalid("min_p must be finite and in [0, 1]"));
        }
        if !self.repeat_penalty.is_finite() || self.repeat_penalty <= 0.0 {
            return Err(invalid("repeat_penalty must be finite and positive"));
        }
        for (name, value) in [
            ("frequency_penalty", self.frequency_penalty),
            ("presence_penalty", self.presence_penalty),
        ] {
            if !value.is_finite() {
                return Err(invalid(&format!("{name} must be finite")));
            }
        }
        Ok(())
    }
}

impl Default for OlmoeSamplingConfig {
    fn default() -> Self {
        Self::greedy()
    }
}

/// Reproducible sampler whose pseudo-random state is request-local.
#[derive(Debug, Clone)]
pub struct OlmoeSampler {
    config: OlmoeSamplingConfig,
    random: SplitMix64,
}

impl OlmoeSampler {
    pub fn new(config: OlmoeSamplingConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            random: SplitMix64::new(config.seed),
            config,
        })
    }

    pub fn config(&self) -> &OlmoeSamplingConfig {
        &self.config
    }

    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> Result<u32> {
        if logits.is_empty() {
            return Err(MoeError::InvalidTensor(
                "cannot sample an empty logit vector".to_string(),
            ));
        }
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(MoeError::InvalidTensor(
                "cannot sample non-finite logits".to_string(),
            ));
        }

        let mut adjusted = logits.to_vec();
        apply_penalties(&mut adjusted, history, &self.config)?;
        let mut candidates = adjusted.into_iter().enumerate().collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });

        if self.config.top_k > 0 {
            candidates.truncate(self.config.top_k.min(candidates.len()));
        }
        if self.config.temperature == 0.0 {
            return token_id(candidates[0].0);
        }

        let maximum = candidates[0].1;
        let temperature = f64::from(self.config.temperature);
        let mut probabilities = candidates
            .into_iter()
            .map(|(token, logit)| (token, ((f64::from(logit - maximum)) / temperature).exp()))
            .collect::<Vec<_>>();
        let total = probabilities.iter().map(|(_, weight)| *weight).sum::<f64>();
        if !total.is_finite() || total <= 0.0 {
            return Err(MoeError::Inference(
                "sampling probabilities did not have positive finite mass".to_string(),
            ));
        }
        for (_, probability) in &mut probabilities {
            *probability /= total;
        }

        if self.config.min_p > 0.0 {
            let threshold = probabilities[0].1 * f64::from(self.config.min_p);
            probabilities.retain(|(_, probability)| *probability >= threshold);
        }
        if self.config.top_p < 1.0 {
            let mut cumulative = 0.0_f64;
            let mut retained = 0_usize;
            for (_, probability) in &probabilities {
                cumulative += *probability;
                retained += 1;
                if cumulative >= f64::from(self.config.top_p) {
                    break;
                }
            }
            probabilities.truncate(retained);
        }

        let retained_mass = probabilities
            .iter()
            .map(|(_, probability)| *probability)
            .sum::<f64>();
        let target = self.random.next_unit_f64() * retained_mass;
        let mut cumulative = 0.0_f64;
        for (token, probability) in &probabilities {
            cumulative += *probability;
            if target < cumulative {
                return token_id(*token);
            }
        }
        probabilities
            .last()
            .ok_or_else(|| MoeError::Inference("sampling removed every candidate".to_string()))
            .and_then(|(token, _)| token_id(*token))
    }
}

fn apply_penalties(
    logits: &mut [f32],
    history: &[u32],
    config: &OlmoeSamplingConfig,
) -> Result<()> {
    let history = match config.repeat_last_n {
        Some(0) => &[][..],
        Some(limit) => &history[history.len().saturating_sub(limit)..],
        None => history,
    };
    let mut counts = BTreeMap::<usize, usize>::new();
    for token in history {
        let token = *token as usize;
        if token >= logits.len() {
            return Err(MoeError::InvalidTensor(
                "sampling history contains a token outside the logit vocabulary".to_string(),
            ));
        }
        *counts.entry(token).or_default() += 1;
    }
    for (token, count) in counts {
        let logit = &mut logits[token];
        if config.repeat_penalty != 1.0 {
            *logit = if *logit <= 0.0 {
                *logit * config.repeat_penalty
            } else {
                *logit / config.repeat_penalty
            };
        }
        *logit -= config.frequency_penalty * count as f32;
        *logit -= config.presence_penalty;
    }
    Ok(())
}

fn token_id(index: usize) -> Result<u32> {
    u32::try_from(index).map_err(|_| {
        MoeError::InvalidTensor("sampled vocabulary index exceeds u32 token IDs".to_string())
    })
}

fn invalid(message: &str) -> MoeError {
    MoeError::InvalidConfig(format!("invalid sampling configuration: {message}"))
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_unit_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^= value >> 31;
        ((value >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_is_stable_and_uses_lowest_id_for_ties() {
        let mut sampler = OlmoeSampler::new(OlmoeSamplingConfig::greedy()).unwrap();
        assert_eq!(sampler.sample(&[1.0, 3.0, 3.0], &[]).unwrap(), 1);
    }

    #[test]
    fn seeded_sampling_is_reproducible() {
        let config = OlmoeSamplingConfig {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 3,
            seed: 42,
            ..OlmoeSamplingConfig::greedy()
        };
        let mut first = OlmoeSampler::new(config).unwrap();
        let mut second = OlmoeSampler::new(config).unwrap();
        let first = (0..16)
            .map(|_| first.sample(&[0.1, 0.3, 0.2, -1.0], &[]).unwrap())
            .collect::<Vec<_>>();
        let second = (0..16)
            .map(|_| second.sample(&[0.1, 0.3, 0.2, -1.0], &[]).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(first, second);
    }

    #[test]
    fn penalties_can_change_greedy_selection() {
        let config = OlmoeSamplingConfig {
            repeat_penalty: 2.0,
            frequency_penalty: 0.2,
            presence_penalty: 0.1,
            ..OlmoeSamplingConfig::greedy()
        };
        let mut sampler = OlmoeSampler::new(config).unwrap();
        assert_eq!(sampler.sample(&[1.0, 0.7], &[0, 0]).unwrap(), 1);
    }

    #[test]
    fn invalid_configuration_and_inputs_fail_closed() {
        let invalid_config = OlmoeSamplingConfig {
            top_p: 0.0,
            ..OlmoeSamplingConfig::greedy()
        };
        assert!(OlmoeSampler::new(invalid_config).is_err());
        let mut sampler = OlmoeSampler::new(OlmoeSamplingConfig::greedy()).unwrap();
        assert!(sampler.sample(&[], &[]).is_err());
        assert!(sampler.sample(&[f32::NAN], &[]).is_err());
        assert!(sampler.sample(&[0.0], &[1]).is_err());
    }

    #[test]
    fn sampler_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OlmoeSamplingConfig>();
        assert_send_sync::<OlmoeSampler>();
    }
}

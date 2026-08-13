use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, MutexGuard};

use a3s_power::error::PowerError;
use a3s_power::inference::{
    ExecutionBatchBinding, ExecutionBatchLifecycle, ExecutionBatchLifecycleEvidence,
    ExecutionBatchMemberSnapshot, ExecutionBatchMemberSpec, ExecutionBatchRowOutcome,
    ExecutionBatchRowSpec, ExecutionDigest,
};
use candle_core::{IndexOp, Tensor};
use tokio_util::sync::CancellationToken;

use crate::olmoe::cpu::validate_generation_request;
use crate::olmoe::{OlmoeKvCache, OlmoeSampler, OlmoeStreamingBatchRow};
use crate::{MoeError, Result};

use super::super::{OlmoeStreamingBatchOutput, OlmoeStreamingModel};
use super::types::{OlmoeContinuousRequest, OlmoeContinuousRowOutput, OlmoeContinuousStepOutput};

/// Model-owned continuous greedy batching on Power's fair lifecycle.
///
/// Admissions retain one Power permit until completion or cancellation. Each
/// step freezes the current fair roster, executes ragged attention with
/// independent KV caches, stages the exact expert union once per layer, and
/// commits Power metadata before publishing updated model state.
#[derive(Clone)]
pub struct OlmoeContinuousBatch {
    model: Arc<OlmoeStreamingModel>,
    lifecycle: ExecutionBatchLifecycle,
    state: Arc<Mutex<CoordinatorState>>,
}

struct CoordinatorState {
    in_step: bool,
    members: BTreeMap<String, ContinuousMember>,
}

#[derive(Clone)]
struct ContinuousMember {
    member_id_sha256: String,
    pending_tokens: Vec<u32>,
    cache: OlmoeKvCache,
    generated_tokens: Vec<u32>,
    token_history: Vec<u32>,
    sampler: OlmoeSampler,
    max_new_tokens: usize,
    eos_token_id: Option<u32>,
}

struct PreparedCommit {
    outcomes: Vec<ExecutionBatchRowOutcome>,
    row_metadata: Vec<Option<(u32, bool)>>,
}

impl OlmoeContinuousBatch {
    pub fn new(model: Arc<OlmoeStreamingModel>, binding: ExecutionBatchBinding) -> Result<Self> {
        let lifecycle = model.runtime().execution_batch(binding)?;
        Ok(Self {
            model,
            lifecycle,
            state: Arc::new(Mutex::new(CoordinatorState {
                in_step: false,
                members: BTreeMap::new(),
            })),
        })
    }

    pub fn lifecycle(&self) -> &ExecutionBatchLifecycle {
        &self.lifecycle
    }

    pub fn active_members(&self) -> Vec<ExecutionBatchMemberSnapshot> {
        self.lifecycle.active_members()
    }

    pub async fn admit(
        &self,
        request: OlmoeContinuousRequest,
        cancellation: CancellationToken,
    ) -> Result<String> {
        self.validate_request(&request)?;
        let permit = self.model.runtime().begin_wait(&cancellation).await?;
        let member_id = request.binding.member_id_sha256().to_string();
        let spec = ExecutionBatchMemberSpec::new(request.binding, 0, 0, request.max_new_tokens, 0);
        let mut state = lock(&self.state);
        if state.members.contains_key(&member_id) {
            return Err(MoeError::InvalidConfig(
                "continuous batch member is already admitted".to_string(),
            ));
        }
        self.lifecycle.admit(spec, permit, cancellation)?;
        state.members.insert(
            member_id.clone(),
            ContinuousMember {
                member_id_sha256: member_id.clone(),
                pending_tokens: request.prompt.clone(),
                cache: self.model.new_cache(),
                generated_tokens: Vec::with_capacity(request.max_new_tokens),
                token_history: request.prompt,
                sampler: OlmoeSampler::new(request.sampling)?,
                max_new_tokens: request.max_new_tokens,
                eos_token_id: request.eos_token_id,
            },
        );
        Ok(member_id)
    }

    pub fn cancel(&self, member_id_sha256: &str) -> Result<()> {
        let mut state = lock(&self.state);
        self.lifecycle.cancel(member_id_sha256)?;
        state.members.remove(member_id_sha256);
        Ok(())
    }

    pub async fn step(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<OlmoeContinuousStepOutput> {
        if cancellation.is_cancelled() {
            return Err(MoeError::Power(PowerError::InferenceCancelled));
        }
        let (step, mut working, originals, inputs, permit) = {
            let mut state = lock(&self.state);
            if state.in_step {
                return Err(MoeError::Inference(
                    "continuous batch already has a step in flight".to_string(),
                ));
            }
            let (step, inputs) = loop {
                let snapshots = self.lifecycle.active_members();
                if snapshots.is_empty() {
                    synchronize_members(&mut state.members, &self.lifecycle);
                    return Err(MoeError::InvalidTensor(
                        "continuous batch has no active members".to_string(),
                    ));
                }
                let mut inputs = Vec::with_capacity(snapshots.len());
                let mut row_specs = Vec::with_capacity(snapshots.len());
                for snapshot in &snapshots {
                    let member =
                        state
                            .members
                            .get(snapshot.member_id_sha256())
                            .ok_or_else(|| {
                                MoeError::Inference(
                                    "continuous scheduler lost an active member state".to_string(),
                                )
                            })?;
                    if member.cache.position() != snapshot.position() {
                        return Err(MoeError::Inference(
                            "continuous scheduler cache position diverged from Power".to_string(),
                        ));
                    }
                    inputs.push(Tensor::from_vec(
                        member.pending_tokens.clone(),
                        (1, member.pending_tokens.len()),
                        self.model.runtime().device().tensor_device(),
                    )?);
                    row_specs.push(ExecutionBatchRowSpec::new(
                        snapshot.member_id_sha256(),
                        snapshot.position(),
                        vec![1, member.pending_tokens.len()],
                        ExecutionDigest::token_ids(&member.pending_tokens),
                    ));
                }
                match self.lifecycle.begin_step(row_specs) {
                    Ok(step) => break (step, inputs),
                    Err(error) => {
                        let previous = state.members.len();
                        synchronize_members(&mut state.members, &self.lifecycle);
                        if state.members.len() < previous && !state.members.is_empty() {
                            continue;
                        }
                        if state.members.is_empty() {
                            return Err(MoeError::InvalidTensor(
                                "continuous batch has no active members".to_string(),
                            ));
                        }
                        return Err(MoeError::Power(error));
                    }
                }
            };
            let permit = step
                .rows()
                .first()
                .ok_or_else(|| MoeError::Inference("Power returned an empty step".to_string()))?
                .permit()
                .clone();
            let member_ids = step
                .rows()
                .iter()
                .map(|row| row.member_id_sha256().to_string())
                .collect::<Vec<_>>();
            if member_ids
                .iter()
                .any(|member_id| !state.members.contains_key(member_id))
            {
                return Err(MoeError::Inference(
                    "continuous scheduler cannot take the frozen roster".to_string(),
                ));
            }
            let working = member_ids
                .iter()
                .filter_map(|member_id| state.members.remove(member_id))
                .collect::<Vec<_>>();
            if working.len() != member_ids.len() {
                for member in working {
                    state
                        .members
                        .insert(member.member_id_sha256.clone(), member);
                }
                return Err(MoeError::Inference(
                    "continuous scheduler froze an incomplete roster".to_string(),
                ));
            }
            let originals = working.clone();
            state.in_step = true;
            (step, working, originals, inputs, permit)
        };

        let mut batch_rows = working
            .iter_mut()
            .zip(&inputs)
            .map(|(member, input)| OlmoeStreamingBatchRow::new(input, &mut member.cache))
            .collect::<Vec<_>>();
        let batch = self
            .model
            .forward_batch(&mut batch_rows, &permit, cancellation)
            .await;
        drop(batch_rows);
        let batch = match batch {
            Ok(batch) => batch,
            Err(error) => {
                drop(step);
                self.restore_failed_step(originals);
                return Err(error);
            }
        };
        if cancellation.is_cancelled() {
            drop(step);
            self.restore_failed_step(originals);
            return Err(MoeError::Power(PowerError::InferenceCancelled));
        }

        let prepared = match prepare_commit(&step, &mut working, &batch) {
            Ok(prepared) => prepared,
            Err(error) => {
                drop(step);
                self.restore_failed_step(originals);
                return Err(error);
            }
        };
        let evidence = match step.commit(prepared.outcomes) {
            Ok(evidence) => evidence,
            Err(error) => {
                self.restore_failed_step(originals);
                return Err(MoeError::Power(error));
            }
        };
        let active = self.active_member_ids();
        {
            let mut state = lock(&self.state);
            for member in &working {
                if active.contains(&member.member_id_sha256) {
                    state
                        .members
                        .insert(member.member_id_sha256.clone(), member.clone());
                }
            }
            state.in_step = false;
        }

        let OlmoeStreamingBatchOutput {
            rows: batch_rows,
            layer_union_routes,
            layer_staging,
        } = batch;
        let mut rows = Vec::with_capacity(evidence.continued_members + evidence.completed_members);
        for ((member, output), metadata) in working
            .into_iter()
            .zip(batch_rows)
            .zip(prepared.row_metadata)
        {
            if let Some((token_id, completed)) = metadata {
                rows.push(OlmoeContinuousRowOutput {
                    member_id_sha256: member.member_id_sha256,
                    token_id,
                    completed,
                    output,
                });
            }
        }
        Ok(OlmoeContinuousStepOutput {
            rows,
            layer_union_routes,
            layer_staging,
            evidence,
        })
    }

    pub fn finish(&self) -> Result<ExecutionBatchLifecycleEvidence> {
        let mut state = lock(&self.state);
        if state.in_step {
            return Err(MoeError::Inference(
                "continuous batch cannot finish with a step in flight".to_string(),
            ));
        }
        match self.lifecycle.finish() {
            Ok(evidence) => {
                state.members.clear();
                Ok(evidence)
            }
            Err(error) => {
                synchronize_members(&mut state.members, &self.lifecycle);
                Err(MoeError::Power(error))
            }
        }
    }

    fn validate_request(&self, request: &OlmoeContinuousRequest) -> Result<()> {
        validate_generation_request(self.model.config(), &request.prompt, request.max_new_tokens)?;
        let limits = self.model.runtime().limits();
        if request.max_new_tokens > limits.max_generated_tokens {
            return Err(MoeError::InvalidConfig(format!(
                "request asks for {} generated tokens, exceeding {}",
                request.max_new_tokens, limits.max_generated_tokens
            )));
        }
        let final_position = request
            .prompt
            .len()
            .checked_add(request.max_new_tokens)
            .ok_or_else(|| MoeError::InvalidConfig("request position overflowed".to_string()))?;
        if final_position > limits.max_context_tokens {
            return Err(MoeError::InvalidConfig(format!(
                "request reaches position {final_position}, exceeding runtime context {}",
                limits.max_context_tokens
            )));
        }
        if request
            .eos_token_id
            .is_some_and(|token| token as usize >= self.model.config().vocab_size)
        {
            return Err(MoeError::InvalidConfig(
                "continuous request EOS token is outside the vocabulary".to_string(),
            ));
        }
        request.sampling.validate()?;
        Ok(())
    }

    fn restore_failed_step(&self, originals: Vec<ContinuousMember>) {
        let active = self.active_member_ids();
        let mut state = lock(&self.state);
        for member in originals {
            if active.contains(&member.member_id_sha256) {
                state
                    .members
                    .insert(member.member_id_sha256.clone(), member);
            }
        }
        state.in_step = false;
    }

    fn active_member_ids(&self) -> BTreeSet<String> {
        self.lifecycle
            .active_members()
            .into_iter()
            .map(|member| member.member_id_sha256().to_string())
            .collect()
    }
}

impl std::fmt::Debug for OlmoeContinuousBatch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = lock(&self.state);
        formatter
            .debug_struct("OlmoeContinuousBatch")
            .field("active_members", &self.lifecycle.active_member_count())
            .field("in_step", &state.in_step)
            .finish_non_exhaustive()
    }
}

fn prepare_commit(
    step: &a3s_power::inference::ExecutionBatchStep,
    working: &mut [ContinuousMember],
    batch: &OlmoeStreamingBatchOutput,
) -> Result<PreparedCommit> {
    if working.len() != batch.rows.len() || working.len() != step.rows().len() {
        return Err(MoeError::Inference(
            "continuous model output does not match the frozen roster".to_string(),
        ));
    }
    let mut outcomes = Vec::new();
    let mut row_metadata = Vec::with_capacity(working.len());
    for (index, (member, output)) in working.iter_mut().zip(&batch.rows).enumerate() {
        let cancelled = step.rows()[index].cancellation().is_cancelled();
        if cancelled {
            row_metadata.push(None);
            continue;
        }
        let sequence_length = output.logits.dim(1)?;
        let logits = output
            .logits
            .i((0, sequence_length - 1, ..))?
            .to_vec1::<f32>()?;
        let token_id = member.sampler.sample(&logits, &member.token_history)?;
        member.generated_tokens.push(token_id);
        member.token_history.push(token_id);
        member.pending_tokens = vec![token_id];
        let completed = member.generated_tokens.len() >= member.max_new_tokens
            || member.eos_token_id == Some(token_id);
        let output_digest = ExecutionDigest::token_ids(&[token_id]);
        let state_bytes = member.cache.resident_bytes()?;
        let next_position = member.cache.position();
        let generated = member.generated_tokens.len();
        let outcome = if completed {
            ExecutionBatchRowOutcome::completed(
                &member.member_id_sha256,
                next_position,
                generated,
                state_bytes,
                output_digest,
            )
        } else {
            ExecutionBatchRowOutcome::continuing(
                &member.member_id_sha256,
                next_position,
                generated,
                state_bytes,
                output_digest,
            )
        };
        outcomes.push(outcome);
        row_metadata.push(Some((token_id, completed)));
    }
    Ok(PreparedCommit {
        outcomes,
        row_metadata,
    })
}

fn synchronize_members(
    members: &mut BTreeMap<String, ContinuousMember>,
    lifecycle: &ExecutionBatchLifecycle,
) {
    let active = lifecycle
        .active_members()
        .into_iter()
        .map(|member| member.member_id_sha256().to_string())
        .collect::<BTreeSet<_>>();
    members.retain(|member, _| active.contains(member));
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuous_public_types_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OlmoeContinuousRequest>();
        assert_send_sync::<OlmoeContinuousBatch>();
        assert_send_sync::<OlmoeContinuousRowOutput>();
        assert_send_sync::<OlmoeContinuousStepOutput>();
    }
}

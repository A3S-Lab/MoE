use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use a3s_power::error::{PowerError, Result as PowerResult};
use a3s_power::inference::{ExecutionBatchBinding, ExecutionDigest, InferenceLimits};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::olmoe::{
    OlmoeContinuousBatch, OlmoeContinuousRequest, OlmoeDecodeStream, OlmoeStreamingModel,
    OlmoeTokenizer,
};
use crate::{MoeError, Result};

use super::request::PreparedGeneration;
use super::stream::{GenerationEvent, GenerationHandle};

#[derive(Clone)]
pub(crate) struct GenerationWorker {
    requests: mpsc::Sender<GenerationJob>,
    shutdown: CancellationToken,
    next_request: Arc<AtomicU64>,
    stream_capacity: usize,
    queue_capacity: usize,
}

struct GenerationJob {
    request_id: u64,
    prepared: PreparedGeneration,
    output: mpsc::Sender<PowerResult<GenerationEvent>>,
    cancellation: CancellationToken,
}

struct ActiveJob {
    output: mpsc::Sender<PowerResult<GenerationEvent>>,
    cancellation: CancellationToken,
    decoder: IncrementalDecoder,
    prompt_tokens: u32,
    eos_token_id: Option<u32>,
    first_step: bool,
}

impl GenerationWorker {
    pub(crate) fn spawn(
        model: Arc<OlmoeStreamingModel>,
        tokenizer: Arc<OlmoeTokenizer>,
        binding: ExecutionBatchBinding,
        stream_capacity: usize,
        batch_window: Duration,
    ) -> Result<Self> {
        let limits = model.runtime().limits().clone();
        let eos_token_id = model.config().eos_token_id;
        let queue_capacity = limits.max_queued_requests;
        let batch = OlmoeContinuousBatch::new(model, binding)?;
        let (requests, receiver) = mpsc::channel(queue_capacity);
        let shutdown = CancellationToken::new();
        tokio::spawn(run_worker(
            batch,
            tokenizer,
            receiver,
            shutdown.clone(),
            batch_window,
            limits,
            eos_token_id,
        ));
        Ok(Self {
            requests,
            shutdown,
            next_request: Arc::new(AtomicU64::new(1)),
            stream_capacity,
            queue_capacity,
        })
    }

    pub(crate) fn submit(&self, prepared: PreparedGeneration) -> PowerResult<GenerationHandle> {
        let request_id = self.next_request.fetch_add(1, Ordering::Relaxed);
        let cancellation = CancellationToken::new();
        let (output, receiver) = mpsc::channel(self.stream_capacity);
        self.requests
            .try_send(GenerationJob {
                request_id,
                prepared,
                output,
                cancellation: cancellation.clone(),
            })
            .map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => PowerError::InferenceQueueFull {
                    maximum: self.queue_capacity,
                },
                mpsc::error::TrySendError::Closed(_) => PowerError::BackendNotAvailable(
                    "the OLMoE generation worker is shutting down".to_string(),
                ),
            })?;
        Ok(GenerationHandle {
            receiver,
            cancellation,
        })
    }

    pub(crate) fn shutdown(&self) {
        self.shutdown.cancel();
    }
}

async fn run_worker(
    batch: OlmoeContinuousBatch,
    tokenizer: Arc<OlmoeTokenizer>,
    mut requests: mpsc::Receiver<GenerationJob>,
    shutdown: CancellationToken,
    batch_window: Duration,
    limits: InferenceLimits,
    eos_token_id: Option<u32>,
) {
    let maximum = limits.max_concurrent_requests;
    let mut active = BTreeMap::<String, ActiveJob>::new();
    let mut input_closed = false;

    loop {
        reap_cancelled(&batch, &mut active);
        if shutdown.is_cancelled() {
            break;
        }
        if active.is_empty() {
            if input_closed {
                break;
            }
            let job = tokio::select! {
                _ = shutdown.cancelled() => break,
                job = requests.recv() => job,
            };
            match job {
                Some(job) => {
                    admit_job(&batch, &tokenizer, &limits, eos_token_id, &mut active, job).await;
                    if !batch_window.is_zero() && !active.is_empty() {
                        tokio::select! {
                            _ = shutdown.cancelled() => break,
                            _ = tokio::time::sleep(batch_window) => {}
                        }
                    }
                }
                None => {
                    input_closed = true;
                    continue;
                }
            }
        }

        while active.len() < maximum {
            match requests.try_recv() {
                Ok(job) => {
                    admit_job(&batch, &tokenizer, &limits, eos_token_id, &mut active, job).await
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    input_closed = true;
                    break;
                }
            }
        }
        reap_cancelled(&batch, &mut active);
        if active.is_empty() {
            continue;
        }

        let step_started = Instant::now();
        match batch.step(&shutdown).await {
            Ok(step) => {
                process_step(&batch, &mut active, step.rows, step_started.elapsed()).await;
            }
            Err(error) => {
                fail_active(&batch, &mut active, error);
            }
        }
    }

    cancel_active(&batch, &mut active);
    if let Err(error) = batch.finish() {
        tracing::warn!(error = %error, "OLMoE generation lifecycle did not finish cleanly");
    }
}

async fn admit_job(
    batch: &OlmoeContinuousBatch,
    tokenizer: &Arc<OlmoeTokenizer>,
    limits: &InferenceLimits,
    eos_token_id: Option<u32>,
    active: &mut BTreeMap<String, ActiveJob>,
    job: GenerationJob,
) {
    if job.cancellation.is_cancelled() || job.output.is_closed() {
        return;
    }
    let prompt_tokens = match u32::try_from(job.prepared.prompt_tokens.len()) {
        Ok(value) => value,
        Err(_) => {
            send_error(
                &job.output,
                PowerError::InvalidRequest("prompt token count exceeds u32".to_string()),
            );
            return;
        }
    };
    let member_identifier = format!("a3s-moe-request:{}", job.request_id);
    let state = ExecutionDigest::token_ids(&job.prepared.prompt_tokens);
    let state_identifier = format!("{}:{}", state.sha256, job.request_id);
    let request = match OlmoeContinuousRequest::for_identifiers(
        member_identifier.as_bytes(),
        state_identifier.as_bytes(),
        job.prepared.prompt_tokens,
        job.prepared.max_new_tokens,
        eos_token_id,
        limits,
    ) {
        Ok(request) => request.with_sampling(job.prepared.sampling),
        Err(error) => {
            send_error(&job.output, invalid_generation(error));
            return;
        }
    };
    let eos_token_id = request.eos_token_id;
    match batch.admit(request, job.cancellation.clone()).await {
        Ok(member_id) => {
            active.insert(
                member_id,
                ActiveJob {
                    output: job.output,
                    cancellation: job.cancellation,
                    decoder: IncrementalDecoder::new(Arc::clone(tokenizer), job.prepared.stop),
                    prompt_tokens,
                    eos_token_id,
                    first_step: true,
                },
            );
        }
        Err(error) => send_error(&job.output, invalid_generation(error)),
    }
}

async fn process_step(
    batch: &OlmoeContinuousBatch,
    active: &mut BTreeMap<String, ActiveJob>,
    rows: Vec<crate::olmoe::OlmoeContinuousRowOutput>,
    step_duration: Duration,
) {
    let mut remove = Vec::new();
    for row in rows {
        let Some(job) = active.get_mut(&row.member_id_sha256) else {
            tracing::warn!(member = %row.member_id_sha256, "OLMoE worker lost a completed row");
            continue;
        };
        let first_step = job.first_step;
        job.first_step = false;
        let decoded = job.decoder.push(row.token_id, row.completed);
        let (text, stop_hit) = match decoded {
            Ok(decoded) => decoded,
            Err(error) => {
                send_error(&job.output, invalid_generation(error));
                if !row.completed {
                    let _ = batch.cancel(&row.member_id_sha256);
                }
                remove.push(row.member_id_sha256);
                continue;
            }
        };
        let done = row.completed || stop_hit;
        let done_reason = done.then(|| {
            if stop_hit || job.eos_token_id == Some(row.token_id) {
                "stop".to_string()
            } else {
                "length".to_string()
            }
        });
        let event = GenerationEvent {
            text,
            token_id: Some(row.token_id),
            done,
            done_reason,
            prompt_tokens: done.then_some(job.prompt_tokens),
            prompt_eval_duration_ns: first_step.then_some(duration_ns(step_duration)),
        };
        let disconnected = job.output.try_send(Ok(event)).is_err();
        if (stop_hit || disconnected) && !row.completed {
            let _ = batch.cancel(&row.member_id_sha256);
        }
        if done || disconnected {
            remove.push(row.member_id_sha256);
        }
    }
    for member in remove {
        active.remove(&member);
    }
}

fn reap_cancelled(batch: &OlmoeContinuousBatch, active: &mut BTreeMap<String, ActiveJob>) {
    let cancelled = active
        .iter()
        .filter(|(_, job)| job.cancellation.is_cancelled() || job.output.is_closed())
        .map(|(member, _)| member.clone())
        .collect::<Vec<_>>();
    for member in cancelled {
        let _ = batch.cancel(&member);
        active.remove(&member);
    }
}

fn fail_active(
    batch: &OlmoeContinuousBatch,
    active: &mut BTreeMap<String, ActiveJob>,
    error: MoeError,
) {
    let message = error.to_string();
    let members = active.keys().cloned().collect::<Vec<_>>();
    for member in members {
        if let Some(job) = active.remove(&member) {
            let _ = batch.cancel(&member);
            send_error(&job.output, PowerError::InferenceFailed(message.clone()));
        }
    }
}

fn cancel_active(batch: &OlmoeContinuousBatch, active: &mut BTreeMap<String, ActiveJob>) {
    let members = active.keys().cloned().collect::<Vec<_>>();
    for member in members {
        if let Some(job) = active.remove(&member) {
            let _ = batch.cancel(&member);
            send_error(&job.output, PowerError::InferenceCancelled);
        }
    }
}

fn send_error(output: &mpsc::Sender<PowerResult<GenerationEvent>>, error: PowerError) {
    let _ = output.try_send(Err(error));
}

fn invalid_generation(error: MoeError) -> PowerError {
    match error {
        MoeError::InvalidConfig(_) | MoeError::InvalidTensor(_) | MoeError::Tokenizer(_) => {
            PowerError::InvalidRequest(error.to_string())
        }
        MoeError::Power(error) => error,
        other => PowerError::InferenceFailed(other.to_string()),
    }
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

struct IncrementalDecoder {
    stream: OlmoeDecodeStream,
    stop: Vec<String>,
    decoded: String,
    emitted: String,
}

impl IncrementalDecoder {
    fn new(tokenizer: Arc<OlmoeTokenizer>, stop: Vec<String>) -> Self {
        Self {
            stream: tokenizer.decode_stream(true),
            stop,
            decoded: String::new(),
            emitted: String::new(),
        }
    }

    fn push(&mut self, token: u32, terminal: bool) -> Result<(String, bool)> {
        if let Some(chunk) = self.stream.step(token)? {
            self.decoded.push_str(&chunk);
        }
        let stop_position = self
            .stop
            .iter()
            .filter_map(|stop| self.decoded.find(stop))
            .min();
        let stop_hit = stop_position.is_some();
        let visible_end = match stop_position {
            Some(position) => position,
            None if terminal => self.decoded.len(),
            None => self
                .decoded
                .len()
                .saturating_sub(held_stop_prefix(&self.decoded, &self.stop)),
        };
        if visible_end < self.emitted.len() || !self.decoded.is_char_boundary(visible_end) {
            return Err(MoeError::Tokenizer(
                "stop-sequence buffering crossed previously streamed text".to_string(),
            ));
        }
        let delta = self.decoded[self.emitted.len()..visible_end].to_string();
        self.emitted.push_str(&delta);
        Ok((delta, stop_hit))
    }
}

fn held_stop_prefix(text: &str, stops: &[String]) -> usize {
    text.char_indices()
        .map(|(index, _)| index)
        .chain(std::iter::once(text.len()))
        .filter_map(|index| {
            let suffix = &text[index..];
            (!suffix.is_empty()
                && stops
                    .iter()
                    .any(|stop| stop.len() > suffix.len() && stop.starts_with(suffix)))
            .then_some(suffix.len())
        })
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::Tokenizer;

    use super::*;

    fn tokenizer(root: &Path) -> Arc<OlmoeTokenizer> {
        let vocab = root.join("vocab.json");
        std::fs::write(
            &vocab,
            r#"{"[UNK]":0,"hello":1,"world":2,"stop":3,"now":4}"#,
        )
        .unwrap();
        let model = WordLevel::builder()
            .files(vocab.to_string_lossy().into_owned())
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut inner = Tokenizer::new(model);
        inner.with_pre_tokenizer(Some(Whitespace));
        let path = root.join("tokenizer.json");
        inner.save(&path, false).unwrap();
        Arc::new(OlmoeTokenizer::from_file(path, 8).unwrap())
    }

    #[test]
    fn decoder_emits_stable_deltas() {
        let directory = tempfile::tempdir().unwrap();
        let mut decoder = IncrementalDecoder::new(tokenizer(directory.path()), Vec::new());
        assert_eq!(
            decoder.push(1, false).unwrap(),
            ("hello".to_string(), false)
        );
        assert_eq!(
            decoder.push(2, true).unwrap(),
            (" world".to_string(), false)
        );
    }

    #[test]
    fn stop_prefix_is_not_streamed_before_a_match() {
        assert_eq!(held_stop_prefix("hello sto", &["stop".to_string()]), 3);
        assert_eq!(held_stop_prefix("hello x", &["stop".to_string()]), 0);
    }

    #[test]
    fn matching_stop_text_is_never_emitted() {
        let directory = tempfile::tempdir().unwrap();
        let mut decoder =
            IncrementalDecoder::new(tokenizer(directory.path()), vec!["stop".to_string()]);
        assert_eq!(decoder.push(3, false).unwrap(), (String::new(), true));
    }
}

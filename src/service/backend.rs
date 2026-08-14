use std::collections::HashMap;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use a3s_power::admission::AdmissionSnapshot;
use a3s_power::backend::types::{
    ChatRequest, ChatResponseChunk, CompletionRequest, CompletionResponseChunk,
    EffectivePromptDigest, EmbeddingRequest, EmbeddingResponse,
};
use a3s_power::backend::Backend;
use a3s_power::error::{PowerError, Result as PowerResult};
use a3s_power::inference::{EmbeddedRuntime, PlacementTelemetry};
use a3s_power::model::manifest::{ModelFormat, ModelManifest};
use a3s_power::server::request_context::RequestContext;
use async_trait::async_trait;
use futures::{Stream, StreamExt};

use crate::olmoe::OlmoeEncryptedCheckpointSource;
use crate::{MoeError, MoeTokenizer, Result};

use super::architecture::{
    load_olmoe_source, OlmoeServiceArchitecture, Qwen36MoeServiceArchitecture,
    Qwen3MoeServiceArchitecture, ServiceArchitecture,
};
use super::config::{OlmoeBackendConfig, OlmoeDeviceSelection};
use super::load::{
    model_load_error, power_manifest, validate_model_name, LoadSpec, LoadedArtifacts,
};
use super::request::{prepare_chat, prepare_completion, PromptPolicy};
use super::source::CheckpointSource;
use super::stream::CancellableStream;
use super::worker::GenerationWorker;

/// Architecture-aware Power backend for one packed MoE family.
pub struct MoeBackend<A: ServiceArchitecture> {
    config: OlmoeBackendConfig,
    models: RwLock<HashMap<String, Arc<LoadedModel<A>>>>,
    load_lock: tokio::sync::Mutex<()>,
}

pub type OlmoeBackend = MoeBackend<OlmoeServiceArchitecture>;
pub type Qwen3MoeBackend = MoeBackend<Qwen3MoeServiceArchitecture>;
pub type Qwen36MoeBackend = MoeBackend<Qwen36MoeServiceArchitecture>;

struct LoadedModel<A: ServiceArchitecture> {
    manifest: ModelManifest,
    canonical_path: PathBuf,
    trust_anchor: Option<String>,
    tokenizer: Arc<MoeTokenizer>,
    model: Arc<A::Model>,
    worker: GenerationWorker,
    device: OlmoeDeviceSelection,
}

impl<A: ServiceArchitecture> Drop for LoadedModel<A> {
    fn drop(&mut self) {
        self.worker.shutdown();
    }
}

impl<A: ServiceArchitecture> MoeBackend<A> {
    pub fn new(config: OlmoeBackendConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            models: RwLock::new(HashMap::new()),
            load_lock: tokio::sync::Mutex::new(()),
        })
    }

    /// Load and verify a packed checkpoint before starting Power, returning the
    /// exact process-local manifest to inject into `PowerServerBuilder`.
    pub async fn preload(
        &self,
        name: impl Into<String>,
        path: impl Into<PathBuf>,
        template_override: Option<String>,
    ) -> PowerResult<ModelManifest> {
        self.load_spec(LoadSpec {
            name: name.into(),
            source: CheckpointSource::Plain(path.into()),
            expected_sha256: None,
            system_prompt: None,
            template_override,
            default_parameters: None,
            messages: Vec::new(),
        })
        .await
    }

    pub fn telemetry(&self, model_name: &str) -> PowerResult<PlacementTelemetry> {
        use crate::continuous::ContinuousStreamingModel;
        Ok(self.loaded_model(model_name)?.model.telemetry())
    }

    pub fn admission_snapshot(&self, model_name: &str) -> PowerResult<AdmissionSnapshot> {
        use crate::continuous::ContinuousStreamingModel;
        Ok(self
            .loaded_model(model_name)?
            .model
            .runtime()
            .admission_snapshot())
    }

    pub fn device_selection(&self, model_name: &str) -> PowerResult<OlmoeDeviceSelection> {
        Ok(self.loaded_model(model_name)?.device)
    }

    fn loaded_model(&self, name: &str) -> PowerResult<Arc<LoadedModel<A>>> {
        read_models(&self.models)
            .get(name)
            .cloned()
            .ok_or_else(|| PowerError::ModelNotFound(name.to_string()))
    }

    async fn load_spec(&self, spec: LoadSpec) -> PowerResult<ModelManifest> {
        self.load_spec_with(spec, |source, runtime, residency, device| match source {
            CheckpointSource::Plain(path) => A::load_plain(path, runtime, residency, device),
            CheckpointSource::Encrypted(_) => Err(MoeError::InvalidConfig(format!(
                "{} does not support this encrypted checkpoint source",
                A::DISPLAY_NAME
            ))),
        })
        .await
    }

    async fn load_spec_with<F>(&self, spec: LoadSpec, loader: F) -> PowerResult<ModelManifest>
    where
        F: FnOnce(
                CheckpointSource,
                EmbeddedRuntime,
                a3s_power::inference::ResidencyPolicy,
                OlmoeDeviceSelection,
            ) -> Result<LoadedArtifacts<A::Model>>
            + Send
            + 'static,
    {
        validate_model_name(&spec.name)?;
        if let Some(loaded) = read_models(&self.models).get(&spec.name) {
            validate_existing(loaded, &spec)?;
            return Ok(loaded.manifest.clone());
        }

        let _load_guard = self.load_lock.lock().await;
        if let Some(loaded) = read_models(&self.models).get(&spec.name) {
            validate_existing(loaded, &spec)?;
            return Ok(loaded.manifest.clone());
        }

        let limits = self.config.inference_limits.clone();
        let mut residency = self.config.residency_policy.clone();
        let requested_device = self.config.device;
        let source = spec.source.clone();
        let artifacts = tokio::task::spawn_blocking(move || {
            let runtime = EmbeddedRuntime::new(requested_device, limits)?;
            let device = OlmoeDeviceSelection::new(
                requested_device,
                runtime.device().identity(),
                residency.device_cache_bytes,
            );
            residency.device_cache_bytes = device.effective_device_cache_bytes;
            loader(source, runtime, residency, device)
        })
        .await
        .map_err(|error| {
            PowerError::InferenceFailed(format!(
                "{} model loading task failed: {error}",
                A::DISPLAY_NAME
            ))
        })?
        .map_err(model_load_error)?;

        let manifest = power_manifest(&spec, &artifacts, A::FAMILY);
        if let Some(expected) = spec.expected_sha256.as_deref() {
            if expected != manifest.sha256 {
                return Err(PowerError::IntegrityCheckFailed {
                    model: spec.name,
                    expected: expected.to_string(),
                    actual: manifest.sha256,
                });
            }
        }
        let worker = GenerationWorker::spawn(
            Arc::clone(&artifacts.model),
            Arc::clone(&artifacts.tokenizer),
            artifacts.binding,
            self.config.stream_capacity,
            self.config.batch_window,
            artifacts.eos_token_ids,
        )
        .map_err(model_load_error)?;
        let loaded = Arc::new(LoadedModel {
            manifest: manifest.clone(),
            canonical_path: artifacts.canonical_path,
            trust_anchor: spec.source.trust_anchor().map(str::to_owned),
            tokenizer: artifacts.tokenizer,
            model: artifacts.model,
            worker,
            device: artifacts.device,
        });
        write_models(&self.models).insert(manifest.name.clone(), loaded);
        Ok(manifest)
    }

    fn prompt_policy<'a>(&self, model: &'a LoadedModel<A>) -> PromptPolicy<'a> {
        PromptPolicy {
            tokenizer: &model.tokenizer,
            template_override: model.manifest.template_override.as_deref(),
            system_prompt: model.manifest.system_prompt.as_deref(),
            manifest_messages: &model.manifest.messages,
            max_input_bytes: self.config.inference_limits.max_input_bytes,
            max_context_tokens: model
                .manifest
                .parameters
                .as_ref()
                .and_then(|parameters| parameters.context_length)
                .map(|value| value as usize)
                .unwrap_or(self.config.inference_limits.max_context_tokens)
                .min(self.config.inference_limits.max_context_tokens),
            max_generated_tokens: self.config.inference_limits.max_generated_tokens,
            max_concurrent_requests: self.config.inference_limits.max_concurrent_requests,
            architecture: A::DISPLAY_NAME,
        }
    }
}

impl MoeBackend<OlmoeServiceArchitecture> {
    /// Load a seekable encrypted OLMoE checkpoint with its out-of-band trust
    /// anchor and zeroizing key owner.
    pub async fn preload_encrypted(
        &self,
        name: impl Into<String>,
        source: OlmoeEncryptedCheckpointSource,
        template_override: Option<String>,
    ) -> PowerResult<ModelManifest> {
        self.load_spec_with(
            LoadSpec {
                name: name.into(),
                source: CheckpointSource::Encrypted(source),
                expected_sha256: None,
                system_prompt: None,
                template_override,
                default_parameters: None,
                messages: Vec::new(),
            },
            load_olmoe_source,
        )
        .await
    }
}

#[async_trait]
impl<A: ServiceArchitecture> Backend for MoeBackend<A> {
    fn name(&self) -> &str {
        A::BACKEND_NAME
    }

    fn supports(&self, format: &ModelFormat) -> bool {
        matches!(format, ModelFormat::SafeTensors)
    }

    fn supports_manifest(&self, manifest: &ModelManifest) -> bool {
        self.supports(&manifest.format)
            && (manifest.family.as_deref() == Some(A::FAMILY)
                || manifest
                    .families
                    .as_ref()
                    .is_some_and(|families| families.iter().any(|family| family == A::FAMILY)))
    }

    async fn load(&self, manifest: &ModelManifest) -> PowerResult<()> {
        if !self.supports_manifest(manifest) {
            return Err(PowerError::InvalidFormat(format!(
                "backend {} requires a SafeTensors manifest with family '{}'",
                A::BACKEND_NAME,
                A::FAMILY
            )));
        }
        self.load_spec(LoadSpec {
            name: manifest.name.clone(),
            source: CheckpointSource::Plain(manifest.path.clone()),
            expected_sha256: Some(manifest.sha256.clone()),
            system_prompt: manifest.system_prompt.clone(),
            template_override: manifest.template_override.clone(),
            default_parameters: manifest.default_parameters.clone(),
            messages: manifest.messages.clone(),
        })
        .await?;
        Ok(())
    }

    async fn unload(&self, model_name: &str) -> PowerResult<()> {
        if let Some(model) = write_models(&self.models).remove(model_name) {
            model.worker.shutdown();
        }
        Ok(())
    }

    async fn chat(
        &self,
        model_name: &str,
        request: ChatRequest,
    ) -> PowerResult<Pin<Box<dyn Stream<Item = PowerResult<ChatResponseChunk>> + Send>>> {
        let model = self.loaded_model(model_name)?;
        let prepared = prepare_chat(
            self.prompt_policy(&model),
            &request,
            self.config.default_max_tokens,
        )?;
        let handle = model.worker.submit(prepared)?;
        let stream = CancellableStream::new(handle.receiver, handle.cancellation).map(|result| {
            result.map(|event| ChatResponseChunk {
                content: event.text,
                thinking_content: None,
                done: event.done,
                prompt_tokens: event.prompt_tokens,
                done_reason: event.done_reason,
                prompt_eval_duration_ns: event.prompt_eval_duration_ns,
                tool_calls: None,
            })
        });
        Ok(Box::pin(stream))
    }

    async fn effective_chat_prompt_digest(
        &self,
        model_name: &str,
        request: &ChatRequest,
    ) -> PowerResult<Option<EffectivePromptDigest>> {
        let model = self.loaded_model(model_name)?;
        let prepared = prepare_chat(
            self.prompt_policy(&model),
            request,
            self.config.default_max_tokens,
        )?;
        Ok(Some(EffectivePromptDigest::chat_prompt_token_ids(
            A::BACKEND_NAME,
            &prepared.prompt_tokens,
        )))
    }

    async fn complete(
        &self,
        model_name: &str,
        request: CompletionRequest,
    ) -> PowerResult<Pin<Box<dyn Stream<Item = PowerResult<CompletionResponseChunk>> + Send>>> {
        let model = self.loaded_model(model_name)?;
        let prepared = prepare_completion(
            self.prompt_policy(&model),
            &request,
            self.config.default_max_tokens,
        )?;
        let handle = model.worker.submit(prepared)?;
        let stream = CancellableStream::new(handle.receiver, handle.cancellation).map(|result| {
            result.map(|event| CompletionResponseChunk {
                text: event.text,
                done: event.done,
                prompt_tokens: event.prompt_tokens,
                done_reason: event.done_reason,
                prompt_eval_duration_ns: event.prompt_eval_duration_ns,
                token_id: event.token_id,
            })
        });
        Ok(Box::pin(stream))
    }

    async fn effective_completion_prompt_digest(
        &self,
        model_name: &str,
        request: &CompletionRequest,
    ) -> PowerResult<Option<EffectivePromptDigest>> {
        let model = self.loaded_model(model_name)?;
        prepare_completion(
            self.prompt_policy(&model),
            request,
            self.config.default_max_tokens,
        )?;
        Ok(Some(EffectivePromptDigest::text_prompt(
            A::BACKEND_NAME,
            &request.prompt,
        )))
    }

    async fn embed(
        &self,
        _model_name: &str,
        _request: EmbeddingRequest,
    ) -> PowerResult<EmbeddingResponse> {
        Err(PowerError::BackendNotAvailable(format!(
            "{} is a causal language model and does not expose embeddings",
            A::DISPLAY_NAME
        )))
    }

    async fn cleanup_request(
        &self,
        _model_name: &str,
        _context: &RequestContext,
    ) -> PowerResult<()> {
        Ok(())
    }
}

fn validate_existing<A: ServiceArchitecture>(
    loaded: &LoadedModel<A>,
    spec: &LoadSpec,
) -> PowerResult<()> {
    let canonical = spec.source.root().canonicalize()?;
    if canonical != loaded.canonical_path {
        return Err(PowerError::InvalidRequest(format!(
            "model '{}' is already loaded from a different checkpoint",
            spec.name
        )));
    }
    if spec.expected_sha256.is_none()
        && loaded.trust_anchor.as_deref() != spec.source.trust_anchor()
    {
        return Err(PowerError::InvalidRequest(format!(
            "model '{}' is already loaded with a different checkpoint trust anchor",
            spec.name
        )));
    }
    if spec
        .expected_sha256
        .as_ref()
        .is_some_and(|expected| expected != &loaded.manifest.sha256)
    {
        return Err(PowerError::IntegrityCheckFailed {
            model: spec.name.clone(),
            expected: spec.expected_sha256.clone().unwrap_or_default(),
            actual: loaded.manifest.sha256.clone(),
        });
    }
    let same_messages = loaded.manifest.messages.len() == spec.messages.len()
        && loaded
            .manifest
            .messages
            .iter()
            .zip(&spec.messages)
            .all(|(left, right)| left.role == right.role && left.content == right.content);
    if loaded.manifest.system_prompt != spec.system_prompt
        || loaded.manifest.template_override != spec.template_override
        || loaded.manifest.default_parameters != spec.default_parameters
        || !same_messages
    {
        return Err(PowerError::InvalidRequest(format!(
            "model '{}' is already loaded with different prompt or parameter policy",
            spec.name
        )));
    }
    Ok(())
}

fn read_models<A: ServiceArchitecture>(
    models: &RwLock<HashMap<String, Arc<LoadedModel<A>>>>,
) -> std::sync::RwLockReadGuard<'_, HashMap<String, Arc<LoadedModel<A>>>> {
    models
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn write_models<A: ServiceArchitecture>(
    models: &RwLock<HashMap<String, Arc<LoadedModel<A>>>>,
) -> std::sync::RwLockWriteGuard<'_, HashMap<String, Arc<LoadedModel<A>>>> {
    models
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl<A: ServiceArchitecture> std::fmt::Debug for MoeBackend<A> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MoeBackend")
            .field("architecture", &A::FAMILY)
            .field("config", &self.config)
            .field("loaded_models", &read_models(&self.models).len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[path = "backend_tests.rs"]
mod tests;

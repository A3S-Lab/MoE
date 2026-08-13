use a3s_power::backend::chat_template::{format_chat_prompt, ChatTemplateKind};
use a3s_power::backend::types::{ChatMessage, ChatRequest, CompletionRequest};
use a3s_power::error::PowerError;

use crate::olmoe::{OlmoeSamplingConfig, OlmoeTokenizer};

pub(crate) struct PromptPolicy<'a> {
    pub tokenizer: &'a OlmoeTokenizer,
    pub template_override: Option<&'a str>,
    pub system_prompt: Option<&'a str>,
    pub manifest_messages: &'a [a3s_power::model::manifest::ManifestMessage],
    pub max_input_bytes: usize,
    pub max_context_tokens: usize,
    pub max_generated_tokens: usize,
    pub max_concurrent_requests: usize,
}

#[derive(Debug)]
pub(crate) struct PreparedGeneration {
    pub prompt_tokens: Vec<u32>,
    pub sampling: OlmoeSamplingConfig,
    pub max_new_tokens: usize,
    pub stop: Vec<String>,
}

pub(crate) fn prepare_chat(
    policy: PromptPolicy<'_>,
    request: &ChatRequest,
    default_max_tokens: usize,
) -> Result<PreparedGeneration, PowerError> {
    validate_chat_capabilities(request, policy.max_concurrent_requests)?;
    if request.messages.is_empty() {
        return Err(invalid("chat requests require at least one message"));
    }
    let mut messages = Vec::new();
    if let Some(system_prompt) = policy.system_prompt {
        messages.push(text_message("system", system_prompt));
    }
    messages.extend(
        policy
            .manifest_messages
            .iter()
            .map(|message| text_message(&message.role, &message.content)),
    );
    messages.extend(request.messages.iter().cloned());
    validate_messages(&messages)?;
    let prompt = format_chat_prompt(
        &messages,
        &ChatTemplateKind::Generic,
        policy.template_override,
    )
    .map_err(|error| invalid(&format!("chat template rendering failed: {error}")))?;
    prepare_text(
        policy,
        &prompt,
        false,
        SamplingFields::from_chat(request),
        default_max_tokens,
    )
}

pub(crate) fn prepare_completion(
    policy: PromptPolicy<'_>,
    request: &CompletionRequest,
    default_max_tokens: usize,
) -> Result<PreparedGeneration, PowerError> {
    validate_completion_capabilities(request, policy.max_concurrent_requests)?;
    prepare_text(
        policy,
        &request.prompt,
        true,
        SamplingFields::from_completion(request),
        default_max_tokens,
    )
}

fn prepare_text(
    policy: PromptPolicy<'_>,
    prompt: &str,
    add_special_tokens: bool,
    fields: SamplingFields<'_>,
    default_max_tokens: usize,
) -> Result<PreparedGeneration, PowerError> {
    if prompt.is_empty() {
        return Err(invalid("the rendered prompt must not be empty"));
    }
    if prompt.len() > policy.max_input_bytes {
        return Err(invalid(&format!(
            "prompt contains {} bytes, exceeding the {} byte model limit",
            prompt.len(),
            policy.max_input_bytes
        )));
    }
    let prompt_tokens = policy
        .tokenizer
        .encode(prompt, add_special_tokens)
        .map_err(|error| invalid(&error.to_string()))?;
    if prompt_tokens.is_empty() {
        return Err(invalid("the tokenizer produced an empty prompt"));
    }
    let max_new_tokens = fields
        .max_tokens
        .map(|value| value as usize)
        .unwrap_or(default_max_tokens);
    if max_new_tokens == 0 {
        return Err(invalid("max_tokens must be greater than zero"));
    }
    if max_new_tokens > policy.max_generated_tokens {
        return Err(invalid(&format!(
            "max_tokens {max_new_tokens} exceeds the {} token model limit",
            policy.max_generated_tokens
        )));
    }
    let requested = prompt_tokens
        .len()
        .checked_add(max_new_tokens)
        .ok_or_else(|| invalid("the requested token count overflowed"))?;
    if requested > policy.max_context_tokens {
        return Err(invalid(&format!(
            "prompt and generation require {requested} tokens, exceeding the {} token model context",
            policy.max_context_tokens
        )));
    }
    if let Some(num_ctx) = fields.num_ctx {
        if requested > num_ctx as usize {
            return Err(invalid(&format!(
                "prompt and generation require {requested} tokens, exceeding num_ctx {num_ctx}"
            )));
        }
    }
    let stop = fields.stop.cloned().unwrap_or_default();
    if stop.iter().any(String::is_empty) {
        return Err(invalid("stop sequences must not be empty"));
    }
    let sampling = sampling_config(fields)?;
    Ok(PreparedGeneration {
        prompt_tokens,
        sampling,
        max_new_tokens,
        stop,
    })
}

#[derive(Clone, Copy)]
struct SamplingFields<'a> {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    min_p: Option<f32>,
    repeat_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    repeat_last_n: Option<i32>,
    seed: Option<i64>,
    max_tokens: Option<u32>,
    num_ctx: Option<u32>,
    stop: Option<&'a Vec<String>>,
}

impl<'a> SamplingFields<'a> {
    fn from_chat(request: &'a ChatRequest) -> Self {
        Self {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            min_p: request.min_p,
            repeat_penalty: request.repeat_penalty,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            repeat_last_n: request.repeat_last_n,
            seed: request.seed,
            max_tokens: request.max_tokens,
            num_ctx: request.num_ctx,
            stop: request.stop.as_ref(),
        }
    }

    fn from_completion(request: &'a CompletionRequest) -> Self {
        Self {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            min_p: request.min_p,
            repeat_penalty: request.repeat_penalty,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            repeat_last_n: request.repeat_last_n,
            seed: request.seed,
            max_tokens: request.max_tokens,
            num_ctx: request.num_ctx,
            stop: request.stop.as_ref(),
        }
    }
}

fn sampling_config(fields: SamplingFields<'_>) -> Result<OlmoeSamplingConfig, PowerError> {
    let top_k = match fields.top_k.unwrap_or(0) {
        value if value < 0 => return Err(invalid("top_k must be non-negative")),
        value => value as usize,
    };
    let repeat_last_n = match fields.repeat_last_n {
        None | Some(-1) => None,
        Some(value) if value >= 0 => Some(value as usize),
        Some(_) => return Err(invalid("repeat_last_n must be -1 or non-negative")),
    };
    let config = OlmoeSamplingConfig {
        temperature: fields.temperature.unwrap_or(1.0),
        top_p: fields.top_p.unwrap_or(1.0),
        top_k,
        min_p: fields.min_p.unwrap_or(0.0),
        repeat_penalty: fields.repeat_penalty.unwrap_or(1.0),
        frequency_penalty: fields.frequency_penalty.unwrap_or(0.0),
        presence_penalty: fields.presence_penalty.unwrap_or(0.0),
        repeat_last_n,
        seed: fields.seed.unwrap_or(0) as u64,
    };
    config
        .validate()
        .map_err(|error| invalid(&error.to_string()))?;
    Ok(config)
}

fn validate_chat_capabilities(
    request: &ChatRequest,
    max_concurrent_requests: usize,
) -> Result<(), PowerError> {
    if request.has_image_inputs() {
        return Err(invalid("OLMoE does not support image inputs"));
    }
    if request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
        || request.tool_choice.is_some()
        || request.parallel_tool_calls.is_some()
    {
        return Err(invalid("OLMoE does not support tool calling"));
    }
    if request.response_format.is_some() {
        return Err(invalid(
            "OLMoE does not support constrained response formats",
        ));
    }
    if request.session_id.is_some() {
        return Err(invalid(
            "cross-request KV sessions are not supported; send the full conversation",
        ));
    }
    reject_unsupported_controls(
        request.mirostat,
        request.mirostat_tau,
        request.mirostat_eta,
        request.tfs_z,
        request.typical_p,
        request.penalize_newline,
        request.num_batch,
        request.num_thread,
        request.num_thread_batch,
        request.flash_attention,
        request.num_gpu,
        request.main_gpu,
        request.use_mmap,
        request.use_mlock,
        request.num_parallel,
        max_concurrent_requests,
    )
}

fn validate_completion_capabilities(
    request: &CompletionRequest,
    max_concurrent_requests: usize,
) -> Result<(), PowerError> {
    if request
        .images
        .as_ref()
        .is_some_and(|images| !images.is_empty())
    {
        return Err(invalid("OLMoE does not support image inputs"));
    }
    if request.response_format.is_some() {
        return Err(invalid(
            "OLMoE does not support constrained response formats",
        ));
    }
    if request.session_id.is_some() || request.context.is_some() {
        return Err(invalid(
            "cross-request KV sessions are not supported; send the full prompt",
        ));
    }
    if request.suffix.is_some() {
        return Err(invalid(
            "OLMoE does not support fill-in-the-middle suffixes",
        ));
    }
    reject_unsupported_controls(
        request.mirostat,
        request.mirostat_tau,
        request.mirostat_eta,
        request.tfs_z,
        request.typical_p,
        request.penalize_newline,
        request.num_batch,
        request.num_thread,
        request.num_thread_batch,
        request.flash_attention,
        request.num_gpu,
        request.main_gpu,
        request.use_mmap,
        request.use_mlock,
        request.num_parallel,
        max_concurrent_requests,
    )
}

#[allow(clippy::too_many_arguments)]
fn reject_unsupported_controls(
    mirostat: Option<u32>,
    mirostat_tau: Option<f32>,
    mirostat_eta: Option<f32>,
    tfs_z: Option<f32>,
    typical_p: Option<f32>,
    penalize_newline: Option<bool>,
    num_batch: Option<u32>,
    num_thread: Option<u32>,
    num_thread_batch: Option<u32>,
    flash_attention: Option<bool>,
    num_gpu: Option<i32>,
    main_gpu: Option<i32>,
    use_mmap: Option<bool>,
    use_mlock: Option<bool>,
    num_parallel: Option<u32>,
    max_concurrent_requests: usize,
) -> Result<(), PowerError> {
    let unsupported = [
        ("mirostat", mirostat.is_some()),
        ("mirostat_tau", mirostat_tau.is_some()),
        ("mirostat_eta", mirostat_eta.is_some()),
        ("tfs_z", tfs_z.is_some()),
        ("typical_p", typical_p.is_some()),
        ("penalize_newline", penalize_newline.is_some()),
        ("num_batch", num_batch.is_some()),
        ("num_thread", num_thread.is_some()),
        ("num_thread_batch", num_thread_batch.is_some()),
        ("flash_attention", flash_attention.is_some()),
        ("num_gpu", num_gpu.is_some()),
        ("main_gpu", main_gpu.is_some()),
        ("use_mmap", use_mmap.is_some()),
        ("use_mlock", use_mlock.is_some()),
    ]
    .into_iter()
    .filter_map(|(name, present)| present.then_some(name))
    .collect::<Vec<_>>();
    if !unsupported.is_empty() {
        return Err(invalid(&format!(
            "unsupported request option(s): {}",
            unsupported.join(", ")
        )));
    }
    if num_parallel.is_some_and(|value| value as usize != max_concurrent_requests) {
        return Err(invalid(&format!(
            "num_parallel must match the configured OLMoE concurrency {max_concurrent_requests}"
        )));
    }
    Ok(())
}

fn validate_messages(messages: &[ChatMessage]) -> Result<(), PowerError> {
    for message in messages {
        if !matches!(message.role.as_str(), "system" | "user" | "assistant") {
            return Err(invalid(&format!(
                "unsupported chat role '{}'",
                message.role
            )));
        }
        if message.name.is_some() || message.tool_calls.is_some() || message.tool_call_id.is_some()
        {
            return Err(invalid(
                "named messages and tool-call message fields are not supported",
            ));
        }
    }
    Ok(())
}

fn text_message(role: &str, content: &str) -> ChatMessage {
    use a3s_power::backend::types::MessageContent;

    ChatMessage {
        role: role.to_string(),
        content: MessageContent::Text(content.to_string()),
        name: None,
        tool_calls: None,
        tool_call_id: None,
        images: None,
    }
}

fn invalid(message: &str) -> PowerError {
    PowerError::InvalidRequest(message.to_string())
}

#![cfg(feature = "server")]

use std::sync::Arc;
use std::time::Duration;

use a3s_moe::qwen3_moe::Qwen3MoeConversionOptions;
use a3s_moe::service::{Qwen3MoeBackend, Qwen3MoeBackendConfig};
use a3s_power::backend::types::{ChatRequest, CompletionRequest};
use a3s_power::backend::Backend;
use a3s_power::inference::{DevicePreference, InferenceLimits, ResidencyPolicy, TelemetryMode};
use candle_core::DType;
use futures::StreamExt;

mod support;
use support::qwen::write_source;

fn backend_config() -> Qwen3MoeBackendConfig {
    Qwen3MoeBackendConfig {
        device: DevicePreference::Cpu,
        inference_limits: InferenceLimits {
            max_context_tokens: 8,
            max_generated_tokens: 4,
            max_concurrent_requests: 2,
            max_queued_requests: 4,
            ..InferenceLimits::default()
        },
        residency_policy: ResidencyPolicy {
            host_cache_bytes: 1024,
            max_background_inflight_bytes: 2048,
            telemetry: TelemetryMode::Aggregate,
            ..ResidencyPolicy::default()
        },
        stream_capacity: 4,
        batch_window: Duration::from_millis(10),
        default_max_tokens: 2,
    }
}

fn completion(prompt: &str, max_tokens: u32) -> CompletionRequest {
    serde_json::from_value(serde_json::json!({
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": true
    }))
    .unwrap()
}

fn chat() -> ChatRequest {
    serde_json::from_value(serde_json::json!({
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.0,
        "max_tokens": 1,
        "stream": true
    }))
    .unwrap()
}

async fn successful<T>(
    stream: std::pin::Pin<Box<dyn futures::Stream<Item = a3s_power::error::Result<T>> + Send>>,
) -> Vec<T> {
    stream
        .map(|result| result.expect("generation stream failed"))
        .collect()
        .await
}

#[tokio::test]
async fn backend_preloads_and_continuously_batches_qwen_requests() {
    let directory = tempfile::tempdir().unwrap();
    let source = write_source(&directory.path().join("source"), DType::F32);
    let packed = directory.path().join("packed");
    source
        .convert_to_packed(
            &packed,
            &InferenceLimits::default(),
            Qwen3MoeConversionOptions {
                experts_per_file: 1,
                max_buffer_bytes: 2048,
            },
        )
        .unwrap();

    let backend = Arc::new(Qwen3MoeBackend::new(backend_config()).unwrap());
    let template = "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}";
    let manifest = backend
        .preload("tiny-qwen", &packed, Some(template.to_string()))
        .await
        .unwrap();

    assert_eq!(manifest.family.as_deref(), Some("qwen3_moe"));
    assert_eq!(manifest.sha256.len(), 64);
    assert!(backend.supports_manifest(&manifest));
    assert_eq!(
        backend
            .device_selection("tiny-qwen")
            .unwrap()
            .resolved
            .name(),
        "cpu"
    );
    backend.load(&manifest).await.unwrap();

    let first = backend
        .complete("tiny-qwen", completion("hello", 2))
        .await
        .unwrap();
    let second = backend
        .complete("tiny-qwen", completion("world", 2))
        .await
        .unwrap();
    let (first, second) = tokio::join!(successful(first), successful(second));
    assert_eq!(
        backend.admission_snapshot("tiny-qwen").unwrap().peak_active,
        2
    );
    for chunks in [&first, &second] {
        assert!(!chunks.is_empty());
        assert!(chunks.last().unwrap().done);
        assert!(chunks.iter().all(|chunk| chunk.token_id.is_some()));
    }

    let digest = backend
        .effective_chat_prompt_digest("tiny-qwen", &chat())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(digest.backend, "a3s-moe-qwen3-moe");
    assert!(
        successful(backend.chat("tiny-qwen", chat()).await.unwrap())
            .await
            .last()
            .unwrap()
            .done
    );
}

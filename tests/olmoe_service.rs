#![cfg(feature = "server")]

use std::sync::Arc;
use std::time::Duration;

use a3s_moe::service::{OlmoeBackend, OlmoeBackendConfig};
use a3s_power::backend::types::{ChatRequest, CompletionRequest};
use a3s_power::backend::Backend;
use a3s_power::inference::{InferenceLimits, ResidencyPolicy, TelemetryMode};
use futures::StreamExt;

#[path = "support/packed.rs"]
mod packed;
mod support;
use packed::write_packed;

fn backend_config() -> OlmoeBackendConfig {
    let limits = InferenceLimits {
        max_context_tokens: 8,
        max_generated_tokens: 4,
        max_concurrent_requests: 2,
        max_queued_requests: 4,
        ..InferenceLimits::default()
    };
    OlmoeBackendConfig {
        inference_limits: limits,
        residency_policy: ResidencyPolicy {
            host_cache_bytes: 1024 * 1024,
            max_background_inflight_bytes: 1024 * 1024,
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
async fn backend_preloads_and_streams_concurrent_chat_and_completions() {
    let directory = tempfile::tempdir().unwrap();
    let packed = write_packed(directory.path());
    let backend = Arc::new(OlmoeBackend::new(backend_config()).unwrap());
    let template = "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}";
    let manifest = backend
        .preload("tiny-olmoe", &packed, Some(template.to_string()))
        .await
        .unwrap();

    assert_eq!(manifest.family.as_deref(), Some("olmoe"));
    assert_eq!(manifest.sha256.len(), 64);
    assert!(backend.supports_manifest(&manifest));
    backend.load(&manifest).await.unwrap();

    let first = backend
        .complete("tiny-olmoe", completion("hello", 2))
        .await
        .unwrap();
    let second = backend
        .complete("tiny-olmoe", completion("world", 2))
        .await
        .unwrap();
    let (first, second) = tokio::join!(successful(first), successful(second));
    assert_eq!(
        backend
            .admission_snapshot("tiny-olmoe")
            .unwrap()
            .peak_active,
        2
    );
    for chunks in [&first, &second] {
        assert!(!chunks.is_empty());
        assert!(chunks.last().unwrap().done);
        assert!(chunks.iter().all(|chunk| chunk.token_id.is_some()));
        assert!(chunks
            .iter()
            .any(|chunk| chunk.prompt_eval_duration_ns.is_some()));
    }

    let digest = backend
        .effective_chat_prompt_digest("tiny-olmoe", &chat())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(digest.backend, "a3s-moe-olmoe");
    assert_eq!(digest.kind, "chat.prompt-token-ids");
    let chat_chunks = successful(backend.chat("tiny-olmoe", chat()).await.unwrap()).await;
    assert!(chat_chunks.last().unwrap().done);
}

#[tokio::test]
async fn seeded_requests_are_reproducible_and_dropped_streams_release_admission() {
    let directory = tempfile::tempdir().unwrap();
    let packed = write_packed(directory.path());
    let backend = Arc::new(OlmoeBackend::new(backend_config()).unwrap());
    backend.preload("tiny-olmoe", packed, None).await.unwrap();

    let request = |seed| {
        serde_json::from_value::<CompletionRequest>(serde_json::json!({
            "prompt": "hello",
            "temperature": 0.8,
            "top_p": 0.9,
            "seed": seed,
            "max_tokens": 3,
            "stream": true
        }))
        .unwrap()
    };
    let first = successful(backend.complete("tiny-olmoe", request(42)).await.unwrap()).await;
    let second = successful(backend.complete("tiny-olmoe", request(42)).await.unwrap()).await;
    assert_eq!(
        first.iter().map(|chunk| chunk.token_id).collect::<Vec<_>>(),
        second
            .iter()
            .map(|chunk| chunk.token_id)
            .collect::<Vec<_>>()
    );

    let abandoned = backend
        .complete("tiny-olmoe", completion("hello", 4))
        .await
        .unwrap();
    drop(abandoned);
    let replacement = tokio::time::timeout(
        Duration::from_secs(5),
        backend.complete("tiny-olmoe", completion("world", 1)),
    )
    .await
    .expect("replacement request waited after cancellation")
    .unwrap();
    assert!(successful(replacement).await.last().unwrap().done);
}

#[tokio::test]
async fn unsupported_semantics_fail_before_generation() {
    let directory = tempfile::tempdir().unwrap();
    let packed = write_packed(directory.path());
    let backend = OlmoeBackend::new(backend_config()).unwrap();
    backend.preload("tiny-olmoe", packed, None).await.unwrap();

    let request: CompletionRequest = serde_json::from_value(serde_json::json!({
        "prompt": "hello",
        "session_id": "stateful",
        "max_tokens": 1
    }))
    .unwrap();
    assert!(backend.complete("tiny-olmoe", request).await.is_err());

    let oversized = completion("hello", 5);
    assert!(backend.complete("tiny-olmoe", oversized).await.is_err());
    backend.unload("tiny-olmoe").await.unwrap();
    assert!(backend.chat("tiny-olmoe", chat()).await.is_err());
}

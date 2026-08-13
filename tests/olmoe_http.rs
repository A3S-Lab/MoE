#![cfg(feature = "server")]

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

#[path = "support/packed.rs"]
mod packed;
mod support;

use packed::write_packed;

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[tokio::test]
async fn server_composes_model_listing_completion_and_sse_streaming() {
    let directory = tempfile::tempdir().unwrap();
    let packed = write_packed(directory.path());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let port_string = port.to_string();

    let child = Command::new(env!("CARGO_BIN_EXE_a3s-moe-server"))
        .arg(packed)
        .args([
            "--model",
            "tiny-olmoe",
            "--host",
            "127.0.0.1",
            "--port",
            &port_string,
            "--host-cache-mib",
            "1",
            "--max-context-tokens",
            "8",
            "--max-generated-tokens",
            "4",
            "--default-max-tokens",
            "2",
        ])
        .env("A3S_POWER_HOME", directory.path().join("power-home"))
        .env("RUST_LOG", "warn")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut server = ChildGuard(child);
    let client = reqwest::Client::new();
    let base = format!("http://127.0.0.1:{port}");
    wait_until_ready(&client, &base, &mut server).await;

    let models: serde_json::Value = client
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(models["data"][0]["id"], "tiny-olmoe");

    let completion_response = client
        .post(format!("{base}/v1/completions"))
        .json(&serde_json::json!({
            "model": "tiny-olmoe",
            "prompt": "hello",
            "temperature": 0.0,
            "max_tokens": 2,
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    let completion_status = completion_response.status();
    let completion_body = completion_response.text().await.unwrap();
    assert!(
        completion_status.is_success(),
        "completion failed with {completion_status}: {completion_body}"
    );
    let completion: serde_json::Value = serde_json::from_str(&completion_body).unwrap();
    assert_eq!(completion["object"], "text_completion");
    assert_eq!(completion["model"], "tiny-olmoe");
    assert!(completion["usage"]["completion_tokens"]
        .as_u64()
        .is_some_and(|tokens| (1..=2).contains(&tokens)));
    assert!(completion["choices"][0]["finish_reason"].is_string());

    let streaming = client
        .post(format!("{base}/v1/completions"))
        .json(&serde_json::json!({
            "model": "tiny-olmoe",
            "prompt": "world",
            "temperature": 0.0,
            "max_tokens": 2,
            "stream": true
        }))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(streaming.contains("data:"));
    assert!(streaming.contains("\"object\":\"text_completion\""));
    assert!(streaming.contains("data: [DONE]"));
}

async fn wait_until_ready(client: &reqwest::Client, base: &str, server: &mut ChildGuard) {
    for _ in 0..100 {
        if let Some(status) = server.0.try_wait().unwrap() {
            panic!("a3s-moe-server exited before readiness with {status}");
        }
        if client
            .get(format!("{base}/health"))
            .send()
            .await
            .is_ok_and(|response| response.status().is_success())
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("a3s-moe-server did not become ready within 10 seconds");
}

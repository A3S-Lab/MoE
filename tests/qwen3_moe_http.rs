#![cfg(feature = "server")]

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use a3s_moe::qwen3_moe::Qwen3MoeConversionOptions;
use a3s_power::inference::InferenceLimits;
use candle_core::DType;

mod support;
use support::qwen::write_source;

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[tokio::test]
async fn server_auto_detects_qwen_and_serves_completion_and_sse() {
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
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let port_string = port.to_string();

    let child = Command::new(env!("CARGO_BIN_EXE_a3s-moe-server"))
        .arg(&packed)
        .args([
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
    assert_eq!(models["data"][0]["id"], "qwen3-moe");

    let response = client
        .post(format!("{base}/v1/completions"))
        .json(&serde_json::json!({
            "model": "qwen3-moe",
            "prompt": "hello",
            "temperature": 0.0,
            "max_tokens": 2,
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    let status = response.status();
    let body = response.text().await.unwrap();
    assert!(
        status.is_success(),
        "completion failed with {status}: {body}"
    );
    let completion: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(completion["model"], "qwen3-moe");
    assert!(completion["choices"][0]["finish_reason"].is_string());

    let streaming = client
        .post(format!("{base}/v1/completions"))
        .json(&serde_json::json!({
            "model": "qwen3-moe",
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

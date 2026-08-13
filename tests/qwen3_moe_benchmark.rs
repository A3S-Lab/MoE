#![cfg(feature = "benchmark")]

use std::process::Command;

use a3s_moe::qwen3_moe::Qwen3MoeConversionOptions;
use a3s_power::inference::InferenceLimits;
use candle_core::DType;

mod support;
use support::qwen::write_split_source;

#[test]
fn benchmark_emits_qwen3_moe_streaming_evidence_without_resident_baseline() {
    let directory = tempfile::tempdir().unwrap();
    let source = write_split_source(&directory.path().join("source"), DType::F32);
    let packed = directory.path().join("packed");
    source
        .convert_to_packed(
            &packed,
            &InferenceLimits::default(),
            Qwen3MoeConversionOptions {
                experts_per_file: 2,
                max_buffer_bytes: 2048,
            },
        )
        .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_a3s-moe-bench"))
        .arg(&packed)
        .args([
            "--prompt",
            "hello",
            "--max-tokens",
            "2",
            "--warm-samples",
            "2",
            "--host-cache-mib",
            "1",
            "--checkpoint-label",
            "qwen3-moe-fixture-packed",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let evidence: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(evidence["schema"], "a3s.moe.qwen3-moe-performance.v1");
    assert_eq!(evidence["implementation"]["powerRevision"], "be82555");
    assert_eq!(evidence["model"]["family"], "qwen3_moe");
    assert_eq!(
        evidence["model"]["packedCheckpoint"],
        "qwen3-moe-fixture-packed"
    );
    assert_eq!(evidence["configuration"]["promptTokens"], 1);
    assert_eq!(evidence["configuration"]["resolvedDevice"]["kind"], "cpu");
    assert_eq!(evidence["warmSummary"]["samples"], 2);
    assert!(evidence["placement"]["firstStorageBytesRead"]
        .as_u64()
        .is_some_and(|value| value > 0));
    assert!(evidence["firstGeneration"]["generatedTokens"]
        .as_u64()
        .is_some_and(|value| value > 0));
    assert!(evidence["processPeakRssBytes"].as_u64().is_some());
    assert!(evidence["residentBaseline"].is_null());
}

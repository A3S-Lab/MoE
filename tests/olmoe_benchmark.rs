#![cfg(feature = "benchmark")]

use std::process::Command;

#[path = "support/packed.rs"]
mod packed;
mod support;

use packed::write_packed;

#[test]
fn benchmark_emits_self_describing_json_evidence() {
    let directory = tempfile::tempdir().unwrap();
    let packed = write_packed(directory.path());
    let output = Command::new(env!("CARGO_BIN_EXE_a3s-moe-bench"))
        .arg(packed)
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
            "fixture-packed",
        ])
        .arg("--resident-checkpoint")
        .arg(directory.path().join("source"))
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let evidence: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(evidence["schema"], "a3s.moe.olmoe-performance.v1");
    assert_eq!(evidence["implementation"]["powerRevision"], "42c6646");
    assert_revision(&evidence["implementation"]["moeRevision"]);
    assert_eq!(evidence["model"]["packedCheckpoint"], "fixture-packed");
    assert_eq!(evidence["configuration"]["promptTokens"], 1);
    assert_eq!(evidence["configuration"]["requestedDevice"]["kind"], "cpu");
    assert_eq!(evidence["configuration"]["resolvedDevice"]["kind"], "cpu");
    assert_eq!(evidence["configuration"]["automaticCpuFallback"], false);
    assert_eq!(evidence["configuration"]["effectiveDeviceCacheBytes"], 0);
    assert_eq!(evidence["warmSummary"]["samples"], 2);
    assert_eq!(
        evidence["cacheState"]["powerFirstGeneration"],
        "empty expert residency cache"
    );
    assert!(evidence["firstGeneration"]["timeToFirstTokenNs"]
        .as_u64()
        .is_some_and(|value| value > 0));
    assert!(evidence["placement"]["firstStorageBytesRead"]
        .as_u64()
        .is_some_and(|value| value > 0));
    assert!(evidence["processPeakRssBytes"].as_u64().is_some());
    assert_eq!(
        evidence["residentBaseline"]["tokenParityWithStreaming"],
        true
    );
    assert!(evidence["residentBaseline"]["processPeakRssBytes"]
        .as_u64()
        .is_some());
}

fn assert_revision(value: &serde_json::Value) {
    let revision = value.as_str().unwrap();
    if revision == "unknown" {
        return;
    }
    let revision = revision.strip_suffix("-dirty").unwrap_or(revision);
    assert_eq!(revision.len(), 40);
    assert!(revision.bytes().all(|byte| byte.is_ascii_hexdigit()));
}

use serde_json::Value;

#[test]
fn checked_public_validation_evidence_passes_every_numeric_gate() {
    let evidence: Value =
        serde_json::from_str(include_str!("../evidence/olmoe-public-validation.json")).unwrap();
    assert_eq!(evidence["schema"], "a3s.moe.olmoe-validation.v1");
    assert_eq!(evidence["status"], "passed");
    assert_eq!(evidence["routesChecked"], 384);
    assert_eq!(evidence["logits"]["mismatchCount"], 0);
    assert_eq!(evidence["routerLogits"]["mismatchCount"], 0);
    assert_eq!(evidence["routeWeights"]["mismatchCount"], 0);
    assert_eq!(evidence["routeExpertMismatches"], 0);
    assert_eq!(evidence["argmaxTokenMismatches"], 0);
}

#[test]
fn checked_public_performance_evidence_is_path_free_and_parity_bound() {
    let evidence: Value =
        serde_json::from_str(include_str!("../evidence/olmoe-public-cpu-windows.json")).unwrap();
    assert_eq!(evidence["schema"], "a3s.moe.olmoe-performance.v1");
    assert_eq!(evidence["implementation"]["powerRevision"], "3f348a5");
    assert_eq!(evidence["configuration"]["maxNewTokens"], 8);
    assert_eq!(evidence["warmSummary"]["samples"], 3);
    assert_eq!(
        evidence["residentBaseline"]["tokenParityWithStreaming"],
        true
    );
    assert_eq!(
        evidence["firstGeneration"]["tokenIds"],
        evidence["residentBaseline"]["sample"]["tokenIds"]
    );
    let label = evidence["model"]["packedCheckpoint"].as_str().unwrap();
    assert!(!label.contains('/') && !label.contains('\\') && !label.contains(':'));
    assert!(
        evidence["processPeakRssBytes"].as_u64().unwrap()
            < evidence["residentBaseline"]["processPeakRssBytes"]
                .as_u64()
                .unwrap()
    );
}

#[test]
fn checked_qwen3_public_validation_evidence_passes_every_numeric_gate() {
    let evidence: Value =
        serde_json::from_str(include_str!("../evidence/qwen3-moe-public-validation.json")).unwrap();
    assert_eq!(evidence["schema"], "a3s.moe.qwen3-moe-validation.v1");
    assert_eq!(evidence["status"], "passed");
    assert_eq!(
        evidence["moeRevision"],
        "0146bd6cee4a3557f303ffb3dbfd1d70975b3802"
    );
    assert_eq!(evidence["powerRevision"], "42c6646");
    assert_eq!(evidence["modelId"], "Qwen/Qwen3-30B-A3B-Base");
    assert_eq!(evidence["promptTokens"], 2);
    assert_eq!(evidence["sparseLayers"], 48);
    assert_eq!(evidence["routesChecked"], 768);
    assert_eq!(evidence["logits"]["mismatchCount"], 0);
    assert_eq!(evidence["routerLogits"]["mismatchCount"], 0);
    assert_eq!(evidence["routeWeights"]["mismatchCount"], 0);
    assert_eq!(evidence["routeExpertMismatches"], 0);
    assert_eq!(evidence["argmaxTokenMismatches"], 0);
}

#[test]
fn checked_qwen3_public_performance_evidence_is_bounded_and_path_free() {
    let evidence: Value = serde_json::from_str(include_str!(
        "../evidence/qwen3-moe-public-cpu-windows.json"
    ))
    .unwrap();
    assert_eq!(evidence["schema"], "a3s.moe.qwen3-moe-performance.v1");
    assert_eq!(
        evidence["implementation"]["moeRevision"],
        "0146bd6cee4a3557f303ffb3dbfd1d70975b3802"
    );
    assert_eq!(evidence["implementation"]["powerRevision"], "42c6646");
    assert_eq!(evidence["model"]["family"], "qwen3_moe");
    assert_eq!(evidence["configuration"]["promptTokens"], 2);
    assert_eq!(evidence["configuration"]["maxNewTokens"], 8);
    assert_eq!(evidence["warmSummary"]["samples"], 3);
    assert_eq!(evidence["residentBaseline"], Value::Null);
    assert!(
        evidence["placement"]["hostResidentBytes"].as_u64().unwrap()
            <= evidence["configuration"]["hostCacheBytes"]
                .as_u64()
                .unwrap()
    );
    assert_eq!(
        evidence["firstGeneration"]["tokenIds"],
        evidence["warmGenerations"][0]["tokenIds"]
    );
    assert_eq!(
        evidence["firstGeneration"]["tokenIds"],
        evidence["warmGenerations"][1]["tokenIds"]
    );
    assert_eq!(
        evidence["firstGeneration"]["tokenIds"],
        evidence["warmGenerations"][2]["tokenIds"]
    );
    let label = evidence["model"]["packedCheckpoint"].as_str().unwrap();
    assert!(!label.contains('/') && !label.contains('\\') && !label.contains(':'));
}

#[test]
fn checked_qwen3_public_http_evidence_exercises_two_concurrent_requests() {
    let evidence: Value = serde_json::from_str(include_str!(
        "../evidence/qwen3-moe-public-http-windows.json"
    ))
    .unwrap();
    assert_eq!(evidence["schema"], "a3s.moe.qwen3-moe-http-smoke.v1");
    assert_eq!(evidence["status"], "passed");
    assert_eq!(
        evidence["implementation"]["moeRevision"],
        "0146bd6cee4a3557f303ffb3dbfd1d70975b3802"
    );
    assert_eq!(evidence["implementation"]["powerRevision"], "42c6646");
    assert_eq!(evidence["configuration"]["maxConcurrentRequests"], 2);
    assert_eq!(evidence["configuration"]["concurrentRequests"], 2);
    assert_eq!(evidence["models"]["data"][0]["id"], "qwen3-moe");
    let completions = evidence["completions"].as_array().unwrap();
    assert_eq!(completions.len(), 2);
    for completion in completions {
        assert_eq!(completion["model"], "qwen3-moe");
        assert_eq!(completion["choices"].as_array().unwrap().len(), 1);
        assert_eq!(completion["usage"]["prompt_tokens"], 2);
        assert_eq!(completion["usage"]["completion_tokens"], 1);
    }
    assert_eq!(
        completions[0]["choices"][0]["text"],
        completions[1]["choices"][0]["text"]
    );
}

#[test]
fn checked_qwen36_public_validation_evidence_passes_every_numeric_gate() {
    let evidence: Value = serde_json::from_str(include_str!(
        "../evidence/qwen3.6-35b-a3b-public-validation.json"
    ))
    .unwrap();
    assert_eq!(evidence["schema"], "a3s.moe.qwen3.6-35b-a3b-validation.v1");
    assert_eq!(evidence["status"], "passed");
    assert_eq!(
        evidence["moeRevision"],
        "f844f44a8d8ef3ec55a9e13209bfede765901ad9"
    );
    assert_eq!(evidence["powerRevision"], "42c6646");
    assert_eq!(evidence["modelId"], "Qwen/Qwen3.6-35B-A3B");
    assert_eq!(evidence["promptTokens"], 5);
    assert_eq!(evidence["linearAttentionLayers"], 30);
    assert_eq!(evidence["fullAttentionLayers"], 10);
    assert_eq!(evidence["routesChecked"], 1600);
    assert_eq!(evidence["logits"]["mismatchCount"], 0);
    assert_eq!(evidence["routerLogits"]["mismatchCount"], 0);
    assert_eq!(evidence["routeWeights"]["mismatchCount"], 0);
    assert_eq!(evidence["routeExpertMismatches"], 0);
    assert_eq!(evidence["routeOrderMismatches"], 12);
    assert_eq!(evidence["argmaxTokenMismatches"], 0);
}

#[test]
fn checked_qwen36_public_performance_evidence_is_bounded_and_path_free() {
    let evidence: Value = serde_json::from_str(include_str!(
        "../evidence/qwen3.6-35b-a3b-public-cpu-windows.json"
    ))
    .unwrap();
    assert_eq!(evidence["schema"], "a3s.moe.qwen3.6-moe-performance.v1");
    assert_eq!(
        evidence["implementation"]["moeRevision"],
        "f844f44a8d8ef3ec55a9e13209bfede765901ad9"
    );
    assert_eq!(evidence["implementation"]["powerRevision"], "42c6646");
    assert_eq!(evidence["model"]["family"], "qwen3_5_moe");
    assert_eq!(evidence["configuration"]["promptTokens"], 2);
    assert_eq!(evidence["configuration"]["maxNewTokens"], 8);
    assert_eq!(evidence["configuration"]["resolvedDevice"]["kind"], "cpu");
    assert_eq!(evidence["configuration"]["effectiveDeviceCacheBytes"], 0);
    assert_eq!(evidence["warmSummary"]["samples"], 3);
    assert_eq!(evidence["residentBaseline"], Value::Null);
    assert!(evidence["warmSummary"]["meanTokensPerSecond"]
        .as_f64()
        .is_some_and(|value| value > 0.19));
    assert!(
        evidence["placement"]["hostResidentBytes"].as_u64().unwrap()
            <= evidence["configuration"]["hostCacheBytes"]
                .as_u64()
                .unwrap()
    );
    for sample in evidence["warmGenerations"].as_array().unwrap() {
        assert_eq!(evidence["firstGeneration"]["tokenIds"], sample["tokenIds"]);
    }
    let label = evidence["model"]["packedCheckpoint"].as_str().unwrap();
    assert!(!label.contains('/') && !label.contains('\\') && !label.contains(':'));
}

#[test]
fn checked_qwen36_public_http_evidence_exercises_two_concurrent_requests() {
    let evidence: Value = serde_json::from_str(include_str!(
        "../evidence/qwen3.6-35b-a3b-public-http-windows.json"
    ))
    .unwrap();
    assert_eq!(evidence["schema"], "a3s.moe.qwen3.6-35b-a3b-http-smoke.v1");
    assert_eq!(evidence["status"], "passed");
    assert_eq!(
        evidence["implementation"]["moeRevision"],
        "f844f44a8d8ef3ec55a9e13209bfede765901ad9"
    );
    assert_eq!(evidence["implementation"]["powerRevision"], "42c6646");
    assert_eq!(evidence["configuration"]["device"], "cpu");
    assert_eq!(evidence["configuration"]["maxConcurrentRequests"], 2);
    assert_eq!(evidence["configuration"]["concurrentRequests"], 2);
    assert!(evidence["models"]["data"]
        .as_array()
        .unwrap()
        .iter()
        .any(|model| model["id"] == "qwen3.6-35b-a3b"));
    assert!(evidence["aggregateCompletionTokensPerSecond"]
        .as_f64()
        .is_some_and(|value| value > 0.0));
    let completions = evidence["completions"].as_array().unwrap();
    assert_eq!(completions.len(), 2);
    for completion in completions {
        assert_eq!(completion["model"], "qwen3.6-35b-a3b");
        assert_eq!(completion["choices"].as_array().unwrap().len(), 1);
        assert_eq!(completion["usage"]["prompt_tokens"], 2);
        assert_eq!(completion["usage"]["completion_tokens"], 1);
        assert_eq!(
            completion["attestation_receipt"]["effective_prompt"]["backend"],
            "a3s-moe-qwen3.6-moe"
        );
    }
    assert_eq!(
        completions[0]["choices"][0]["text"],
        completions[1]["choices"][0]["text"]
    );
}

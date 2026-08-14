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

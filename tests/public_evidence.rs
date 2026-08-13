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

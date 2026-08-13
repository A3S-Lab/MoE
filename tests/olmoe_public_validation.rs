#![cfg(feature = "validation")]

use std::fs;
use std::path::Path;
use std::process::Command;

use a3s_moe::olmoe::{
    OlmoeCheckpoint, OlmoeOracleFile, OlmoeOracleInput, OlmoeOracleModel, OlmoeOracleOutput,
    OlmoeOracleRoute, OlmoePublicOracle, OLMOE_PUBLIC_MODEL_ID, OLMOE_PUBLIC_MODEL_REVISION,
    OLMOE_PUBLIC_ORACLE_SCHEMA, OLMOE_TRANSFORMERS_REVISION,
};
use candle_core::{Device, Tensor};
use sha2::{Digest, Sha256};

#[path = "support/packed.rs"]
mod packed;
mod support;

fn sha256(path: &Path) -> String {
    format!("{:x}", Sha256::digest(fs::read(path).unwrap()))
}

fn write_oracle(root: &Path) -> OlmoePublicOracle {
    let checkpoint = OlmoeCheckpoint::open(root).unwrap();
    let tokenizer = checkpoint.load_tokenizer().unwrap();
    let text = "hello : world answer";
    let token_ids = tokenizer.encode(text, false).unwrap();
    assert_eq!(token_ids, [1, 4, 2, 7]);

    let model = checkpoint.load_cpu_resident().unwrap();
    let input = Tensor::from_vec(token_ids.clone(), (1, token_ids.len()), &Device::Cpu).unwrap();
    let mut cache = model.new_cache();
    let output = model.forward(&input, &mut cache).unwrap();
    let logits = output.logits.to_vec3::<f32>().unwrap().remove(0);
    let layer_router_logits = output
        .layer_router_logits
        .iter()
        .map(|layer| layer.to_vec3::<f32>().unwrap().remove(0))
        .collect();
    let layer_routes = output
        .layer_routes
        .iter()
        .map(|layer| {
            layer
                .selections()
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|route| OlmoeOracleRoute {
                            expert: route.expert,
                            weight: route.weight,
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    let files = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "model-00001-of-00001.safetensors",
    ]
    .into_iter()
    .map(|name| {
        let path = root.join(name);
        OlmoeOracleFile {
            name: name.to_string(),
            bytes: path.metadata().unwrap().len(),
            sha256: sha256(&path),
        }
    })
    .collect();

    OlmoePublicOracle {
        schema: OLMOE_PUBLIC_ORACLE_SCHEMA.to_string(),
        model: OlmoeOracleModel {
            id: OLMOE_PUBLIC_MODEL_ID.to_string(),
            revision: OLMOE_PUBLIC_MODEL_REVISION.to_string(),
            transformers_revision: OLMOE_TRANSFORMERS_REVISION.to_string(),
            transformers_source_sha256: "0".repeat(64),
            files,
        },
        input: OlmoeOracleInput {
            text: text.to_string(),
            add_special_tokens: false,
            token_ids,
        },
        output: OlmoeOracleOutput {
            logits,
            layer_router_logits,
            layer_routes,
        },
    }
}

#[test]
fn validation_cli_reports_passes_and_numerical_failures() {
    let directory = tempfile::tempdir().unwrap();
    packed::write_packed(directory.path());
    let source = directory.path().join("source");
    let oracle_path = directory.path().join("oracle.json");
    let mut oracle = write_oracle(&source);
    fs::write(&oracle_path, serde_json::to_vec_pretty(&oracle).unwrap()).unwrap();

    let passed = Command::new(env!("CARGO_BIN_EXE_a3s-moe-validate"))
        .arg(&source)
        .arg(&oracle_path)
        .output()
        .unwrap();
    assert!(
        passed.status.success(),
        "{}",
        String::from_utf8_lossy(&passed.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&passed.stdout).unwrap();
    assert_eq!(report["schema"], "a3s.moe.olmoe-validation.v1");
    assert_eq!(report["status"], "passed");
    assert_eq!(report["routesChecked"], 16);

    oracle.output.logits[0][0] += 1.0;
    fs::write(&oracle_path, serde_json::to_vec_pretty(&oracle).unwrap()).unwrap();
    let failed = Command::new(env!("CARGO_BIN_EXE_a3s-moe-validate"))
        .arg(source)
        .arg(oracle_path)
        .output()
        .unwrap();
    assert!(!failed.status.success());
    let report: serde_json::Value = serde_json::from_slice(&failed.stdout).unwrap();
    assert_eq!(report["status"], "failed");
    assert_eq!(report["logits"]["mismatchCount"], 1);
}

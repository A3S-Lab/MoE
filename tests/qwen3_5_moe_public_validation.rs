#![cfg(feature = "validation")]

use std::fs;
use std::path::Path;
use std::process::Command;

use a3s_moe::qwen3_5_moe::{
    Qwen36MoeConversionOptions, Qwen36MoeOracleFile, Qwen36MoeOracleInput, Qwen36MoeOracleModel,
    Qwen36MoeOracleOutput, Qwen36MoeOracleRoute, Qwen36MoePublicOracle, QWEN36_MOE_PUBLIC_MODEL_ID,
    QWEN36_MOE_PUBLIC_MODEL_REVISION, QWEN36_MOE_PUBLIC_ORACLE_SCHEMA,
    QWEN36_MOE_TRANSFORMERS_DTYPE, QWEN36_MOE_TRANSFORMERS_REVISION,
    QWEN36_MOE_TRANSFORMERS_SOURCE_SHA256,
};
use a3s_power::inference::InferenceLimits;
use candle_core::{DType, Device, Tensor};
use sha2::{Digest, Sha256};

mod support;
use support::qwen36::write_source;

fn sha256(path: &Path) -> String {
    format!("{:x}", Sha256::digest(fs::read(path).unwrap()))
}

fn write_oracle(root: &Path) -> Qwen36MoePublicOracle {
    let checkpoint = a3s_moe::qwen3_5_moe::Qwen36MoeCheckpoint::open(root).unwrap();
    let tokenizer = checkpoint.load_tokenizer().unwrap();
    let text = "hello : world answer";
    let token_ids = tokenizer.encode(text, false).unwrap();
    let model = checkpoint.load_cpu_resident().unwrap();
    let input = Tensor::from_vec(token_ids.clone(), (1, token_ids.len()), &Device::Cpu).unwrap();
    let output = model.forward(&input, &mut model.new_cache()).unwrap();
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
                        .map(|route| Qwen36MoeOracleRoute {
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
        "tokenizer_config.json",
        "model-00001-of-00001.safetensors",
    ]
    .into_iter()
    .map(|name| {
        let path = root.join(name);
        Qwen36MoeOracleFile {
            name: name.to_string(),
            bytes: path.metadata().unwrap().len(),
            sha256: sha256(&path),
        }
    })
    .collect();

    Qwen36MoePublicOracle {
        schema: QWEN36_MOE_PUBLIC_ORACLE_SCHEMA.to_string(),
        model: Qwen36MoeOracleModel {
            id: QWEN36_MOE_PUBLIC_MODEL_ID.to_string(),
            revision: QWEN36_MOE_PUBLIC_MODEL_REVISION.to_string(),
            transformers_revision: QWEN36_MOE_TRANSFORMERS_REVISION.to_string(),
            transformers_source_sha256: QWEN36_MOE_TRANSFORMERS_SOURCE_SHA256.to_string(),
            transformers_dtype: QWEN36_MOE_TRANSFORMERS_DTYPE.to_string(),
            files,
        },
        input: Qwen36MoeOracleInput {
            text: text.to_string(),
            add_special_tokens: false,
            token_ids,
        },
        output: Qwen36MoeOracleOutput {
            logits,
            layer_router_logits,
            layer_routes,
        },
    }
}

#[test]
fn qwen36_validation_cli_reports_passes_and_numerical_failures() {
    let directory = tempfile::tempdir().unwrap();
    let source_path = directory.path().join("source");
    let source = write_source(&source_path, DType::F32);
    fs::write(source_path.join("tokenizer_config.json"), b"{}\n").unwrap();
    let packed = directory.path().join("packed");
    source
        .convert_to_packed(
            &packed,
            &InferenceLimits::default(),
            Qwen36MoeConversionOptions {
                experts_per_file: 2,
                max_buffer_bytes: 2048,
            },
        )
        .unwrap();
    let oracle_path = directory.path().join("oracle.json");
    let mut oracle = write_oracle(&source_path);
    fs::write(&oracle_path, serde_json::to_vec_pretty(&oracle).unwrap()).unwrap();

    let invoke = || {
        Command::new(env!("CARGO_BIN_EXE_a3s-moe-validate"))
            .arg(&source_path)
            .arg(&oracle_path)
            .arg("--packed-checkpoint")
            .arg(&packed)
            .args(["--host-cache-mib", "1"])
            .output()
            .unwrap()
    };
    let passed = invoke();
    assert!(
        passed.status.success(),
        "{}",
        String::from_utf8_lossy(&passed.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&passed.stdout).unwrap();
    assert_eq!(report["schema"], "a3s.moe.qwen3.6-35b-a3b-validation.v1");
    assert_eq!(report["powerRevision"], "42c6646");
    assert_revision(&report["moeRevision"]);
    assert_eq!(report["status"], "passed");
    assert_eq!(report["linearAttentionLayers"], 3);
    assert_eq!(report["fullAttentionLayers"], 1);
    assert_eq!(report["routesChecked"], 32);
    assert_eq!(report["routeExpertMismatches"], 0);
    assert_eq!(report["routeOrderMismatches"], 0);
    assert_eq!(report["sourceWeightsSha256"].as_str().unwrap().len(), 64);

    oracle.output.layer_routes[0][0].swap(0, 1);
    fs::write(&oracle_path, serde_json::to_vec_pretty(&oracle).unwrap()).unwrap();
    let reordered = invoke();
    assert!(
        reordered.status.success(),
        "{}",
        String::from_utf8_lossy(&reordered.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&reordered.stdout).unwrap();
    assert_eq!(report["status"], "passed");
    assert_eq!(report["routeExpertMismatches"], 0);
    assert_eq!(report["routeOrderMismatches"], 2);
    oracle.output.layer_routes[0][0].swap(0, 1);

    oracle.model.transformers_source_sha256 = "0".repeat(64);
    fs::write(&oracle_path, serde_json::to_vec_pretty(&oracle).unwrap()).unwrap();
    let rejected = invoke();
    assert!(!rejected.status.success());
    assert!(
        String::from_utf8_lossy(&rejected.stderr).contains("Transformers source SHA-256 must be")
    );

    oracle.model.transformers_source_sha256 = QWEN36_MOE_TRANSFORMERS_SOURCE_SHA256.to_string();
    oracle.output.logits[0][0] += 1.0;
    fs::write(&oracle_path, serde_json::to_vec_pretty(&oracle).unwrap()).unwrap();
    let failed = invoke();
    assert!(!failed.status.success());
    let report: serde_json::Value = serde_json::from_slice(&failed.stdout).unwrap();
    assert_eq!(report["status"], "failed");
    assert_eq!(report["logits"]["mismatchCount"], 1);
}

fn assert_revision(value: &serde_json::Value) {
    let revision = value.as_str().unwrap();
    let hash = revision.strip_suffix("-dirty").unwrap_or(revision);
    assert!(
        hash == "unknown"
            || (hash.len() == 40 && hash.bytes().all(|byte| byte.is_ascii_hexdigit())),
        "unexpected MoE revision: {revision}"
    );
}

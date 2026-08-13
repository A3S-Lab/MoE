use std::collections::HashMap;
use std::fs;
use std::process::Command;

use a3s_moe::olmoe::{OlmoeCheckpoint, OlmoeConversionOptions, OlmoePackedCheckpoint};
use a3s_power::inference::{
    DevicePreference, EmbeddedRuntime, InferenceLimits, ResidencyPolicy, TelemetryMode,
};
use approx::assert_abs_diff_eq;
use candle_core::{DType, Device, Tensor};
use tokio_util::sync::CancellationToken;

mod support;
use support::{tiny_config, tiny_model, tiny_weights};

fn write_source(root: &std::path::Path) -> OlmoeCheckpoint {
    write_source_dtype(root, DType::F32)
}

fn write_source_dtype(root: &std::path::Path, dtype: DType) -> OlmoeCheckpoint {
    fs::create_dir(root).unwrap();
    let config = tiny_config();
    fs::write(
        root.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    let weights = tiny_weights(&config)
        .into_iter()
        .map(|(name, tensor)| (name, tensor.to_dtype(dtype).unwrap()))
        .collect::<HashMap<_, _>>();
    let shard = "model-00001-of-00001.safetensors";
    candle_core::safetensors::save(&weights, root.join(shard)).unwrap();
    let weight_map = weights
        .keys()
        .map(|name| (name.clone(), shard.to_string()))
        .collect::<HashMap<_, _>>();
    fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
    )
    .unwrap();
    OlmoeCheckpoint::open(root).unwrap()
}

#[tokio::test]
async fn bf16_conversion_preserves_the_source_checkpoint_equations() {
    let directory = tempfile::tempdir().unwrap();
    let source = write_source_dtype(&directory.path().join("source"), DType::BF16);
    let resident = source.load_cpu_resident().unwrap();
    let packed_path = directory.path().join("packed");
    let limits = InferenceLimits::default();
    source
        .convert_to_packed(
            &packed_path,
            &limits,
            OlmoeConversionOptions {
                experts_per_file: 2,
                max_buffer_bytes: 1024 * 1024,
            },
        )
        .unwrap();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits).unwrap();
    let packed = OlmoePackedCheckpoint::open(&packed_path, runtime.clone()).unwrap();
    assert_eq!(
        packed.manifest().scalar_type,
        a3s_moe::olmoe::PackedScalarType::Bf16
    );
    let streaming = packed
        .load_cpu_streaming(ResidencyPolicy {
            max_background_inflight_bytes: 1024 * 1024,
            ..ResidencyPolicy::default()
        })
        .unwrap();
    let input = Tensor::from_vec(vec![1_u32, 4], (1, 2), &Device::Cpu).unwrap();
    let expected = resident
        .forward(&input, &mut resident.new_cache())
        .unwrap()
        .logits
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let actual = streaming
        .forward(&input, &mut streaming.new_cache(), &permit, &cancellation)
        .await
        .unwrap()
        .logits
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (actual, expected) in actual.iter().zip(expected) {
        assert_abs_diff_eq!(actual, &expected, epsilon = 2e-5);
    }
}

#[tokio::test]
async fn deterministic_conversion_loads_a_bounded_streaming_model() {
    let directory = tempfile::tempdir().unwrap();
    let source = write_source(&directory.path().join("source"));
    let packed_path = directory.path().join("packed");
    let limits = InferenceLimits::default();
    let options = OlmoeConversionOptions {
        experts_per_file: 2,
        max_buffer_bytes: 1024 * 1024,
    };
    let report = source
        .convert_to_packed(&packed_path, &limits, options)
        .unwrap();

    assert_eq!(report.packed_experts, 6);
    assert_eq!(report.dense_files, 21);
    assert_eq!(report.expert_files, 4);
    assert!(report.peak_buffered_bytes <= options.max_buffer_bytes);
    assert!(packed_path.join("manifest.json").is_file());

    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits).unwrap();
    let packed = OlmoePackedCheckpoint::open(&packed_path, runtime.clone()).unwrap();
    let record_bytes = 64 + 3 * tiny_config().hidden_size * tiny_config().intermediate_size * 4;
    let model = packed
        .load_cpu_streaming(ResidencyPolicy {
            host_cache_bytes: record_bytes as u64,
            max_background_inflight_bytes: 1024 * 1024,
            telemetry: TelemetryMode::Aggregate,
            ..ResidencyPolicy::default()
        })
        .unwrap();
    assert_eq!(model.hierarchy().store().inventory().len(), 6);

    let tokens = vec![1_u32, 4, 2];
    let input = Tensor::from_vec(tokens, (1, 3), &Device::Cpu).unwrap();
    let resident = tiny_model();
    let expected = resident
        .forward(&input, &mut resident.new_cache())
        .unwrap()
        .logits
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let actual = model
        .forward(&input, &mut model.new_cache(), &permit, &cancellation)
        .await
        .unwrap()
        .logits
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (actual, expected) in actual.iter().zip(expected) {
        assert_abs_diff_eq!(actual, &expected, epsilon = 2e-5);
    }
    let telemetry = model.telemetry();
    assert!(telemetry.host_resident_bytes <= record_bytes as u64);
    assert!(telemetry.host_evictions > 0);
}

#[test]
fn conversion_failure_never_publishes_a_partial_destination() {
    let directory = tempfile::tempdir().unwrap();
    let source = write_source(&directory.path().join("source"));
    let packed_path = directory.path().join("packed");
    let error = source.convert_to_packed(
        &packed_path,
        &InferenceLimits::default(),
        OlmoeConversionOptions {
            experts_per_file: 2,
            max_buffer_bytes: 1,
        },
    );

    assert!(error.is_err());
    assert!(!packed_path.exists());
}

#[test]
fn packed_loader_rejects_manifest_tampering_before_model_load() {
    let directory = tempfile::tempdir().unwrap();
    let source = write_source(&directory.path().join("source"));
    let packed_path = directory.path().join("packed");
    source
        .convert_to_packed(
            &packed_path,
            &InferenceLimits::default(),
            OlmoeConversionOptions {
                experts_per_file: 2,
                max_buffer_bytes: 1024 * 1024,
            },
        )
        .unwrap();
    let manifest_path = packed_path.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest["expertWeightsSha256"] = serde_json::Value::String("00".repeat(32));
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, InferenceLimits::default()).unwrap();

    assert!(OlmoePackedCheckpoint::open(&packed_path, runtime).is_err());
}

#[test]
fn pack_cli_converts_a_checkpoint_and_emits_machine_readable_evidence() {
    let directory = tempfile::tempdir().unwrap();
    write_source(&directory.path().join("source"));
    let packed_path = directory.path().join("packed");
    let output = Command::new(env!("CARGO_BIN_EXE_a3s-moe-pack"))
        .arg(directory.path().join("source"))
        .arg(&packed_path)
        .args(["--experts-per-file", "2", "--max-buffer-mib", "1"])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["packedExperts"], 6);
    assert!(packed_path.join("manifest.json").is_file());
}

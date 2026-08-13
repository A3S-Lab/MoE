use a3s_moe::qwen3_moe::{
    Qwen3MoeConversionOptions, Qwen3MoeExpertLayout, Qwen3MoePackedCheckpoint,
    Qwen3MoePackedManifest,
};
use a3s_power::inference::{
    DevicePreference, EmbeddedRuntime, InferenceLimits, ResidencyPolicy, TelemetryMode, WeightStore,
};
use approx::assert_abs_diff_eq;
use candle_core::{DType, Device, Tensor};
use std::process::Command;
use tokio_util::sync::CancellationToken;

mod support;
use support::qwen::{tiny_config, write_source, write_split_source};

#[tokio::test]
async fn official_split_expert_layout_converts_to_the_same_streaming_equations() {
    let directory = tempfile::tempdir().unwrap();
    let source = write_split_source(&directory.path().join("source"), DType::F32);
    assert_eq!(source.expert_layout(), Qwen3MoeExpertLayout::Split);
    let resident = source.load_cpu_resident().unwrap();
    let packed_path = directory.path().join("packed");
    let limits = InferenceLimits::default();
    let report = source
        .convert_to_packed(
            &packed_path,
            &limits,
            Qwen3MoeConversionOptions {
                experts_per_file: 2,
                max_buffer_bytes: 2048,
            },
        )
        .unwrap();
    assert_eq!(report.packed_experts, 3);

    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits).unwrap();
    let streaming = Qwen3MoePackedCheckpoint::open(&packed_path, runtime.clone())
        .unwrap()
        .load_cpu_streaming(ResidencyPolicy {
            host_cache_bytes: 1024,
            max_background_inflight_bytes: 2048,
            ..ResidencyPolicy::default()
        })
        .unwrap();
    let input = Tensor::from_vec(vec![1_u32, 4, 2], (1, 3), &Device::Cpu).unwrap();
    let expected = resident.forward(&input, &mut resident.new_cache()).unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let actual = streaming
        .forward(&input, &mut streaming.new_cache(), &permit, &cancellation)
        .await
        .unwrap();
    assert_tensor_close(&actual.logits, &expected.logits);
    assert_eq!(actual.layer_routes, expected.layer_routes);
}

#[tokio::test]
async fn bounded_fused_conversion_matches_the_resident_mixed_decoder() {
    let directory = tempfile::tempdir().unwrap();
    let source = write_source(&directory.path().join("source"), DType::F32);
    let source_store = WeightStore::open(source.root(), &InferenceLimits::default()).unwrap();
    let fused_bytes = source_store
        .descriptor("model.layers.1.mlp.experts.gate_up_proj")
        .unwrap()
        .bytes
        + source_store
            .descriptor("model.layers.1.mlp.experts.down_proj")
            .unwrap()
            .bytes;
    let largest_dense_bytes = source_store
        .descriptor("model.embed_tokens.weight")
        .unwrap()
        .bytes;
    let options = Qwen3MoeConversionOptions {
        experts_per_file: 1,
        max_buffer_bytes: 400,
    };
    assert!(fused_bytes > options.max_buffer_bytes);
    assert!(largest_dense_bytes > options.max_buffer_bytes);

    let resident = source.load_cpu_resident().unwrap();
    let packed_path = directory.path().join("packed");
    let limits = InferenceLimits::default();
    let report = source
        .convert_to_packed(&packed_path, &limits, options)
        .unwrap();
    assert_eq!(report.dense_files, 23);
    assert_eq!(report.expert_files, 3);
    assert_eq!(report.packed_experts, 3);
    assert!(report.peak_buffered_bytes <= options.max_buffer_bytes);

    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits).unwrap();
    let packed = Qwen3MoePackedCheckpoint::open(&packed_path, runtime.clone()).unwrap();
    assert_eq!(packed.manifest().schema, Qwen3MoePackedManifest::SCHEMA);
    assert_eq!(packed.load_tokenizer().unwrap().vocab_size(), 13);
    assert_eq!(
        packed.execution_batch_binding().unwrap(),
        packed.execution_batch_binding().unwrap()
    );
    let record_bytes = 64 + 3 * tiny_config().hidden_size * tiny_config().moe_intermediate_size * 4;
    let streaming = packed
        .load_cpu_streaming(ResidencyPolicy {
            host_cache_bytes: record_bytes as u64,
            max_background_inflight_bytes: 1024,
            telemetry: TelemetryMode::Aggregate,
            ..ResidencyPolicy::default()
        })
        .unwrap();
    assert_eq!(streaming.hierarchy().store().inventory().len(), 3);

    let input = Tensor::from_vec(vec![1_u32, 4, 2], (1, 3), &Device::Cpu).unwrap();
    let expected = resident.forward(&input, &mut resident.new_cache()).unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let actual = streaming
        .forward(&input, &mut streaming.new_cache(), &permit, &cancellation)
        .await
        .unwrap();

    assert_tensor_close(&actual.logits, &expected.logits);
    assert_eq!(actual.layer_routes, expected.layer_routes);
    assert!(actual.layer_router_logits[0].is_none());
    assert!(actual.layer_staging[0].is_none());
    assert_tensor_close(
        actual.layer_router_logits[1].as_ref().unwrap(),
        expected.layer_router_logits[1].as_ref().unwrap(),
    );
    assert!(actual.layer_staging[1].is_some());
    assert!(streaming.telemetry().host_resident_bytes <= record_bytes as u64);
}

#[tokio::test]
async fn packed_streaming_generation_matches_the_resident_decoder() {
    let directory = tempfile::tempdir().unwrap();
    let source = write_source(&directory.path().join("source"), DType::BF16);
    let resident = source.load_cpu_resident().unwrap();
    let packed_path = directory.path().join("packed");
    let limits = InferenceLimits::default();
    source
        .convert_to_packed(
            &packed_path,
            &limits,
            Qwen3MoeConversionOptions {
                experts_per_file: 2,
                max_buffer_bytes: 2048,
            },
        )
        .unwrap();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits).unwrap();
    let packed = Qwen3MoePackedCheckpoint::open(&packed_path, runtime.clone()).unwrap();
    assert_eq!(
        packed.manifest().scalar_type,
        a3s_moe::PackedScalarType::Bf16
    );
    let streaming = packed
        .load_cpu_streaming(ResidencyPolicy {
            host_cache_bytes: 1024,
            max_background_inflight_bytes: 2048,
            ..ResidencyPolicy::default()
        })
        .unwrap();
    let expected = resident.generate_greedy(&[1, 2], 3, None).unwrap();
    let cancellation = CancellationToken::new();
    let actual = streaming
        .generate_greedy(&[1, 2], 3, None, &cancellation)
        .await
        .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(runtime.admission_snapshot().active, 0);
}

fn assert_tensor_close(actual: &Tensor, expected: &Tensor) {
    let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_abs_diff_eq!(actual, &expected, epsilon = 2e-5);
    }
}

#[test]
fn pack_cli_detects_qwen3_moe_and_emits_a_generic_report() {
    let directory = tempfile::tempdir().unwrap();
    let source_path = directory.path().join("source");
    write_source(&source_path, DType::F32);
    let packed_path = directory.path().join("packed-cli");
    let output = Command::new(env!("CARGO_BIN_EXE_a3s-moe-pack"))
        .arg(&source_path)
        .arg(&packed_path)
        .arg("--experts-per-file")
        .arg("2")
        .arg("--max-buffer-mib")
        .arg("1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: a3s_moe::PackedConversionReport = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report.packed_experts, 3);
    assert_eq!(report.expert_files, 2);
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, InferenceLimits::default()).unwrap();
    assert!(Qwen3MoePackedCheckpoint::open(packed_path, runtime).is_ok());
}

use std::collections::HashMap;
use std::sync::Arc;

use a3s_moe::olmoe::{
    packed_expert_tensor_name, OlmoeCpuModel, OlmoeStreamingModel, PackedExpertRecord,
};
use a3s_power::inference::{
    DevicePreference, EmbeddedRuntime, InferenceLimits, ResidencyPolicy, TelemetryMode,
    WeightHierarchy, WeightStore,
};
use approx::assert_abs_diff_eq;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tokio_util::sync::CancellationToken;

mod support;
use support::{tiny_config, tiny_model, tiny_weights};

fn models() -> (
    tempfile::TempDir,
    EmbeddedRuntime,
    OlmoeCpuModel,
    OlmoeStreamingModel,
) {
    let config = tiny_config();
    let all_weights = tiny_weights(&config);
    let resident = tiny_model();

    let directory = tempfile::tempdir().unwrap();
    let mut dense_weights = all_weights;
    let mut records = HashMap::new();
    for layer in 0..config.num_hidden_layers {
        for expert in 0..config.num_experts {
            let prefix = format!("model.layers.{layer}.mlp.experts.{expert}");
            let take = |weights: &mut HashMap<String, Tensor>, suffix: &str| {
                weights
                    .remove(&format!("{prefix}.{suffix}.weight"))
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            };
            let gate = take(&mut dense_weights, "gate_proj");
            let up = take(&mut dense_weights, "up_proj");
            let down = take(&mut dense_weights, "down_proj");
            let record =
                PackedExpertRecord::encode_f32(config.moe_config().unwrap(), &gate, &up, &down)
                    .unwrap();
            records.insert(
                packed_expert_tensor_name(layer as u32, expert as u32),
                Tensor::from_vec(record.clone(), record.len(), &Device::Cpu).unwrap(),
            );
        }
    }
    candle_core::safetensors::save(&records, directory.path().join("experts.safetensors")).unwrap();

    let limits = InferenceLimits::default();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let hierarchy = WeightHierarchy::new(
        Arc::new(WeightStore::open(directory.path(), &limits).unwrap()),
        runtime.clone(),
        ResidencyPolicy {
            host_cache_bytes: 1024 * 1024,
            max_background_inflight_bytes: 1024 * 1024,
            telemetry: TelemetryMode::Aggregate,
            ..ResidencyPolicy::default()
        },
    )
    .unwrap();
    let streaming = OlmoeStreamingModel::load(
        config,
        VarBuilder::from_tensors(dense_weights, DType::F32, &Device::Cpu),
        hierarchy,
    )
    .unwrap();
    (directory, runtime, resident, streaming)
}

#[tokio::test]
async fn complete_streaming_decoder_matches_the_resident_model() {
    let (_directory, runtime, resident, streaming) = models();
    let token_ids = Tensor::from_vec(vec![1_u32, 4, 2], (1, 3), &Device::Cpu).unwrap();
    let resident_output = resident
        .forward(&token_ids, &mut resident.new_cache())
        .unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let mut cache = streaming.new_cache();
    let streamed = streaming
        .forward(&token_ids, &mut cache, &permit, &cancellation)
        .await
        .unwrap();

    let expected = resident_output
        .logits
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let actual = streamed
        .logits
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (actual, expected) in actual.iter().zip(expected) {
        assert_abs_diff_eq!(actual, &expected, epsilon = 2e-5);
    }
    assert_eq!(streamed.layer_routes, resident_output.layer_routes);
    assert_eq!(
        streamed.layer_staging.len(),
        tiny_config().num_hidden_layers
    );
    assert_eq!(cache.position(), 3);
}

#[tokio::test]
async fn streaming_generation_uses_one_admission_and_matches_resident_greedy_decode() {
    let (_directory, runtime, resident, streaming) = models();
    let expected = resident.generate_greedy(&[1, 2], 3, None).unwrap();
    let cancellation = CancellationToken::new();
    let generated = streaming
        .generate_greedy(&[1, 2], 3, None, &cancellation)
        .await
        .unwrap();

    assert_eq!(generated, expected);
    assert_eq!(runtime.admission_snapshot().active, 0);
}

#[tokio::test]
async fn streaming_forward_keeps_the_session_cache_transactional_on_cancellation() {
    let (_directory, runtime, _resident, streaming) = models();
    let token_ids = Tensor::from_vec(vec![1_u32], (1, 1), &Device::Cpu).unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let mut cache = streaming.new_cache();
    cancellation.cancel();

    assert!(streaming
        .forward(&token_ids, &mut cache, &permit, &cancellation)
        .await
        .is_err());
    assert_eq!(cache.position(), 0);
    assert_eq!(streaming.telemetry().storage_reads, 0);
}

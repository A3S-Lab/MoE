use std::collections::HashMap;
use std::sync::Arc;

use a3s_moe::olmoe::{
    packed_expert_tensor_name, OlmoeExpertWeights, OlmoeMoeConfig, OlmoeMoeLayer,
    OlmoeStreamingMlp, PackedExpertRecord,
};
use a3s_moe::Matrix;
use a3s_power::inference::{
    DevicePreference, EmbeddedRuntime, InferenceLimits, ResidencyPolicy, TelemetryMode,
    WeightHierarchy, WeightStore,
};
use approx::assert_abs_diff_eq;
use candle_core::{Device, Tensor};
use tokio_util::sync::CancellationToken;

fn config() -> OlmoeMoeConfig {
    OlmoeMoeConfig {
        hidden_size: 2,
        intermediate_size: 2,
        num_experts: 3,
        top_k: 2,
        normalize_top_k: false,
    }
}

fn expert_values(expert: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let offset = expert as f32 * 0.1;
    (
        vec![1.0 + offset, -0.5, 0.25, 0.75 + offset],
        vec![0.5, 1.0 + offset, -1.0, 0.5],
        vec![1.0, -0.25 + offset, 0.5, 0.75],
    )
}

fn reference_layer(router: &[f32]) -> OlmoeMoeLayer {
    let config = config();
    let experts = (0..config.num_experts)
        .map(|expert| {
            let (gate, up, down) = expert_values(expert);
            let mut gate_up = gate;
            gate_up.extend(up);
            OlmoeExpertWeights::new(
                config,
                Matrix::new(config.intermediate_size * 2, config.hidden_size, gate_up).unwrap(),
                Matrix::new(config.hidden_size, config.intermediate_size, down).unwrap(),
            )
            .unwrap()
        })
        .collect();
    OlmoeMoeLayer::new(
        config,
        Matrix::new(config.num_experts, config.hidden_size, router.to_vec()).unwrap(),
        experts,
    )
    .unwrap()
}

fn streaming_fixture() -> (
    tempfile::TempDir,
    EmbeddedRuntime,
    OlmoeStreamingMlp,
    Vec<f32>,
) {
    let directory = tempfile::tempdir().unwrap();
    let config = config();
    let mut tensors = HashMap::new();
    for expert in 0..config.num_experts {
        let (gate, up, down) = expert_values(expert);
        let record = PackedExpertRecord::encode_f32(config, &gate, &up, &down).unwrap();
        tensors.insert(
            packed_expert_tensor_name(5, expert as u32),
            Tensor::from_vec(record.clone(), record.len(), &Device::Cpu).unwrap(),
        );
    }
    candle_core::safetensors::save(&tensors, directory.path().join("experts.safetensors")).unwrap();

    let limits = InferenceLimits::default();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap();
    let store = Arc::new(WeightStore::open(directory.path(), &limits).unwrap());
    let hierarchy = WeightHierarchy::new(
        store,
        runtime.clone(),
        ResidencyPolicy {
            host_cache_bytes: 1024 * 1024,
            max_background_inflight_bytes: 1024 * 1024,
            telemetry: TelemetryMode::Aggregate,
            ..ResidencyPolicy::default()
        },
    )
    .unwrap();
    let router = vec![0.8, -0.2, -0.1, 0.9, 0.4, 0.3];
    let router_weight = Tensor::from_vec(
        router.clone(),
        (config.num_experts, config.hidden_size),
        &Device::Cpu,
    )
    .unwrap();
    let mlp = OlmoeStreamingMlp::new(config, router_weight, hierarchy).unwrap();
    (directory, runtime, mlp, router)
}

#[tokio::test]
async fn streamed_experts_match_the_resident_reference_and_reuse_power_cache() {
    let (_directory, runtime, mlp, router) = streaming_fixture();
    let input_values = vec![1.0, -0.5, 0.25, 1.25];
    let hidden_states = Tensor::from_vec(input_values.clone(), (1, 2, 2), &Device::Cpu).unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();

    let first = mlp
        .forward(5, &hidden_states, &permit, &cancellation)
        .await
        .unwrap();
    let reference = reference_layer(&router)
        .forward(5, &Matrix::new(2, 2, input_values).unwrap())
        .unwrap();
    let actual = first
        .hidden_states
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (actual, expected) in actual.iter().zip(reference.hidden_states.values()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
    }
    assert_eq!(first.routes, reference.routing.routes);
    assert_eq!(first.staging.requested_groups, first.routes.experts().len());
    assert_eq!(
        first.staging.requested_weights,
        first.routes.experts().len()
    );
    assert_eq!(first.staging.loaded_weights, first.routes.experts().len());

    let second = mlp
        .forward(5, &hidden_states, &permit, &cancellation)
        .await
        .unwrap();
    assert_eq!(second.staging.loaded_weights, 0);
    assert_eq!(
        second.staging.resident_weights,
        second.routes.experts().len()
    );
}

#[tokio::test]
async fn a_cancelled_streaming_layer_does_not_start_weight_io() {
    let (_directory, runtime, mlp, _router) = streaming_fixture();
    let hidden_states = Tensor::zeros((1, 1, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
    let request = CancellationToken::new();
    let permit = runtime.begin(&request).unwrap();
    request.cancel();

    assert!(mlp
        .forward(5, &hidden_states, &permit, &request)
        .await
        .is_err());
    assert_eq!(mlp.telemetry().storage_reads, 0);
}

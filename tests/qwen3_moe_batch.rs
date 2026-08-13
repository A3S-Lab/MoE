use std::collections::BTreeSet;
use std::sync::Arc;

use a3s_moe::qwen3_moe::{
    Qwen3MoeContinuousBatch, Qwen3MoeContinuousRequest, Qwen3MoeConversionOptions,
    Qwen3MoeStreamingBatchRow,
};
use a3s_power::inference::{
    DevicePreference, EmbeddedRuntime, ExecutionBatchBinding, InferenceLimits, ResidencyPolicy,
    TelemetryMode,
};
use approx::assert_abs_diff_eq;
use candle_core::{DType, Device, Tensor};
use tokio_util::sync::CancellationToken;

mod support;
use support::qwen::{tiny_config, write_source};

#[tokio::test]
async fn mixed_layer_route_union_matches_independent_qwen_sessions() {
    let directory = tempfile::tempdir().unwrap();
    let source = write_source(&directory.path().join("source"), DType::F32);
    let packed_path = directory.path().join("packed");
    let limits = InferenceLimits {
        max_concurrent_requests: 4,
        ..InferenceLimits::default()
    };
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
    let packed =
        a3s_moe::qwen3_moe::Qwen3MoePackedCheckpoint::open(&packed_path, runtime.clone()).unwrap();
    let model = packed
        .load_cpu_streaming(ResidencyPolicy {
            host_cache_bytes: 1024,
            max_background_inflight_bytes: 2048,
            telemetry: TelemetryMode::Aggregate,
            ..ResidencyPolicy::default()
        })
        .unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let mut cache_a = model.new_cache();
    let mut cache_b = model.new_cache();
    model
        .forward(
            &Tensor::from_vec(vec![1_u32], (1, 1), &Device::Cpu).unwrap(),
            &mut cache_a,
            &permit,
            &cancellation,
        )
        .await
        .unwrap();
    model
        .forward(
            &Tensor::from_vec(vec![2_u32, 3], (1, 2), &Device::Cpu).unwrap(),
            &mut cache_b,
            &permit,
            &cancellation,
        )
        .await
        .unwrap();
    let mut batch_cache_a = cache_a.clone();
    let mut batch_cache_b = cache_b.clone();
    let input_a = Tensor::from_vec(vec![4_u32, 6], (1, 2), &Device::Cpu).unwrap();
    let input_b = Tensor::from_vec(vec![5_u32], (1, 1), &Device::Cpu).unwrap();
    let expected_a = model
        .forward(&input_a, &mut cache_a, &permit, &cancellation)
        .await
        .unwrap();
    let expected_b = model
        .forward(&input_b, &mut cache_b, &permit, &cancellation)
        .await
        .unwrap();

    let mut rows = [
        Qwen3MoeStreamingBatchRow::new(&input_a, &mut batch_cache_a),
        Qwen3MoeStreamingBatchRow::new(&input_b, &mut batch_cache_b),
    ];
    let actual = model
        .forward_batch(&mut rows, &permit, &cancellation)
        .await
        .unwrap();

    for (actual, expected) in actual.rows.iter().zip([expected_a, expected_b]) {
        assert_tensor_close(&actual.logits, &expected.logits);
        assert_eq!(actual.layer_routes, expected.layer_routes);
    }
    assert_eq!(batch_cache_a.position(), 3);
    assert_eq!(batch_cache_b.position(), 3);
    assert!(actual.layer_union_routes[0].is_none());
    assert!(actual.layer_staging[0].is_none());
    let union = actual
        .rows
        .iter()
        .flat_map(|row| {
            row.layer_routes[1]
                .as_ref()
                .unwrap()
                .experts()
                .iter()
                .copied()
        })
        .collect::<BTreeSet<_>>();
    let union_routes = actual.layer_union_routes[1].as_ref().unwrap();
    assert_eq!(
        union_routes
            .experts()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>(),
        union
    );
    let staging = actual.layer_staging[1].as_ref().unwrap();
    assert_eq!(staging.requested_groups, union.len());
    assert_eq!(staging.requested_weights, union.len());
    assert_eq!(
        actual.layer_union_routes.len(),
        tiny_config().num_hidden_layers
    );
}

#[tokio::test]
async fn shared_continuous_scheduler_batches_qwen_requests_to_completion() {
    let directory = tempfile::tempdir().unwrap();
    let source = write_source(&directory.path().join("source"), DType::F32);
    let packed_path = directory.path().join("packed");
    let limits = InferenceLimits {
        max_concurrent_requests: 4,
        ..InferenceLimits::default()
    };
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
    let packed =
        a3s_moe::qwen3_moe::Qwen3MoePackedCheckpoint::open(&packed_path, runtime.clone()).unwrap();
    let model = Arc::new(
        packed
            .load_cpu_streaming(ResidencyPolicy {
                host_cache_bytes: 1024,
                max_background_inflight_bytes: 2048,
                ..ResidencyPolicy::default()
            })
            .unwrap(),
    );
    let expected_a = model
        .generate_greedy(&[1], 2, None, &CancellationToken::new())
        .await
        .unwrap();
    let expected_b = model
        .generate_greedy(&[2, 3], 2, None, &CancellationToken::new())
        .await
        .unwrap();
    let scheduler = Qwen3MoeContinuousBatch::new(
        model,
        ExecutionBatchBinding::new("a".repeat(64), "b".repeat(64), "c".repeat(64)).unwrap(),
    )
    .unwrap();
    let request = |member: &'static [u8], state: &'static [u8], prompt: Vec<u32>| {
        Qwen3MoeContinuousRequest::for_identifiers(member, state, prompt, 2, None, runtime.limits())
            .unwrap()
    };
    let member_a = scheduler
        .admit(
            request(b"qwen-member-a", b"qwen-state-a", vec![1]),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let member_b = scheduler
        .admit(
            request(b"qwen-member-b", b"qwen-state-b", vec![2, 3]),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    let first = scheduler.step(&CancellationToken::new()).await.unwrap();
    assert_eq!(first.rows.len(), 2);
    assert_eq!(first.rows[0].member_id_sha256, member_a);
    assert_eq!(first.rows[0].token_id, expected_a[1]);
    assert_eq!(first.rows[1].member_id_sha256, member_b);
    assert_eq!(first.rows[1].token_id, expected_b[2]);
    assert!(first.layer_union_routes[0].is_none());
    assert!(first.layer_staging[0].is_none());
    let routes = first.layer_union_routes[1].as_ref().unwrap();
    let staging = first.layer_staging[1].as_ref().unwrap();
    assert_eq!(staging.requested_groups, routes.experts().len());

    let second = scheduler.step(&CancellationToken::new()).await.unwrap();
    assert_eq!(second.rows.len(), 2);
    assert!(second.rows.iter().all(|row| row.completed));
    assert_eq!(second.rows[0].token_id, expected_a[2]);
    assert_eq!(second.rows[1].token_id, expected_b[3]);
    let evidence = scheduler.finish().unwrap();
    assert_eq!(evidence.admitted_members, 2);
    assert_eq!(evidence.completed_members, 2);
    assert_eq!(runtime.admission_snapshot().active, 0);
}

fn assert_tensor_close(actual: &Tensor, expected: &Tensor) {
    let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for (actual, expected) in actual.iter().zip(expected) {
        assert_abs_diff_eq!(actual, &expected, epsilon = 2e-5);
    }
}

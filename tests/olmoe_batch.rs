use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use a3s_moe::olmoe::{
    packed_expert_tensor_name, OlmoeContinuousBatch, OlmoeContinuousRequest,
    OlmoeStreamingBatchRow, OlmoeStreamingModel, PackedExpertRecord,
};
use a3s_power::inference::{
    DevicePreference, EmbeddedRuntime, ExecutionBatchBinding, InferenceLimits, ResidencyPolicy,
    TelemetryMode, WeightHierarchy, WeightStore,
};
use approx::assert_abs_diff_eq;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tokio_util::sync::CancellationToken;

mod support;
use support::{tiny_config, tiny_weights};

fn streaming_model() -> (tempfile::TempDir, EmbeddedRuntime, OlmoeStreamingModel) {
    let config = tiny_config();
    let mut dense_weights = tiny_weights(&config);
    let mut records = HashMap::new();
    for layer in 0..config.num_hidden_layers {
        for expert in 0..config.num_experts {
            let prefix = format!("model.layers.{layer}.mlp.experts.{expert}");
            let mut take = |suffix: &str| {
                dense_weights
                    .remove(&format!("{prefix}.{suffix}.weight"))
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            };
            let gate = take("gate_proj");
            let up = take("up_proj");
            let down = take("down_proj");
            let record =
                PackedExpertRecord::encode_f32(config.moe_config().unwrap(), &gate, &up, &down)
                    .unwrap();
            records.insert(
                packed_expert_tensor_name(layer as u32, expert as u32),
                Tensor::from_vec(record.clone(), record.len(), &Device::Cpu).unwrap(),
            );
        }
    }
    let directory = tempfile::tempdir().unwrap();
    candle_core::safetensors::save(&records, directory.path().join("experts.safetensors")).unwrap();
    let limits = InferenceLimits {
        max_concurrent_requests: 4,
        ..InferenceLimits::default()
    };
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
    let model = OlmoeStreamingModel::load(
        config,
        VarBuilder::from_tensors(dense_weights, DType::F32, &Device::Cpu),
        hierarchy,
    )
    .unwrap();
    (directory, runtime, model)
}

#[tokio::test]
async fn route_unioned_batch_matches_independent_sessions_at_different_positions() {
    let (_directory, runtime, model) = streaming_model();
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
        OlmoeStreamingBatchRow::new(&input_a, &mut batch_cache_a),
        OlmoeStreamingBatchRow::new(&input_b, &mut batch_cache_b),
    ];
    let actual = model
        .forward_batch(&mut rows, &permit, &cancellation)
        .await
        .unwrap();

    assert_eq!(actual.rows.len(), 2);
    for (actual, expected) in actual.rows.iter().zip([expected_a, expected_b]) {
        let actual_logits = actual
            .logits
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected_logits = expected
            .logits
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (actual, expected) in actual_logits.iter().zip(expected_logits) {
            assert_abs_diff_eq!(actual, &expected, epsilon = 2e-5);
        }
        assert_eq!(actual.layer_routes, expected.layer_routes);
    }
    assert_eq!(batch_cache_a.position(), 3);
    assert_eq!(batch_cache_b.position(), 3);

    for layer in 0..tiny_config().num_hidden_layers {
        let union = actual
            .rows
            .iter()
            .flat_map(|row| row.layer_routes[layer].experts().iter().copied())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            actual.layer_union_routes[layer]
                .experts()
                .iter()
                .copied()
                .collect::<BTreeSet<_>>(),
            union
        );
        assert_eq!(actual.layer_staging[layer].requested_groups, union.len());
        assert_eq!(actual.layer_staging[layer].requested_weights, union.len());
    }
}

#[tokio::test]
async fn incompatible_or_cancelled_batches_leave_every_cache_unchanged() {
    let (_directory, runtime, model) = streaming_model();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let one = Tensor::from_vec(vec![1_u32], (1, 1), &Device::Cpu).unwrap();
    let invalid = Tensor::from_vec(vec![2_u32, 3], (2, 1), &Device::Cpu).unwrap();
    let mut first_cache = model.new_cache();
    let mut second_cache = model.new_cache();
    let mut incompatible = [
        OlmoeStreamingBatchRow::new(&one, &mut first_cache),
        OlmoeStreamingBatchRow::new(&invalid, &mut second_cache),
    ];
    assert!(model
        .forward_batch(&mut incompatible, &permit, &cancellation)
        .await
        .is_err());
    assert_eq!(first_cache.position(), 0);
    assert_eq!(second_cache.position(), 0);

    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let mut cancelled_rows = [
        OlmoeStreamingBatchRow::new(&one, &mut first_cache),
        OlmoeStreamingBatchRow::new(&one, &mut second_cache),
    ];
    assert!(model
        .forward_batch(&mut cancelled_rows, &permit, &cancelled)
        .await
        .is_err());
    assert_eq!(first_cache.position(), 0);
    assert_eq!(second_cache.position(), 0);
}

#[tokio::test]
async fn continuous_scheduler_compacts_cancelled_slots_and_commits_fair_ragged_steps() {
    let (_directory, runtime, model) = streaming_model();
    let expected_a = model
        .generate_greedy(&[1], 2, None, &CancellationToken::new())
        .await
        .unwrap();
    let expected_b = model
        .generate_greedy(&[2, 3], 2, None, &CancellationToken::new())
        .await
        .unwrap();
    let expected_c = model
        .generate_greedy(&[4, 5, 6], 2, None, &CancellationToken::new())
        .await
        .unwrap();
    let scheduler = OlmoeContinuousBatch::new(
        Arc::new(model),
        ExecutionBatchBinding::new("a".repeat(64), "b".repeat(64), "c".repeat(64)).unwrap(),
    )
    .unwrap();
    let request = |member: &'static [u8], state: &'static [u8], prompt: Vec<u32>| {
        OlmoeContinuousRequest::for_identifiers(member, state, prompt, 2, None, runtime.limits())
            .unwrap()
    };
    let member_a = scheduler
        .admit(
            request(b"member-a", b"state-a", vec![1]),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let member_b = scheduler
        .admit(
            request(b"member-b", b"state-b", vec![2, 3]),
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
    assert_eq!(first.evidence.continued_members, 2);
    assert!(first.evidence.state_bytes_after > 0);
    for (routes, staging) in first.layer_union_routes.iter().zip(&first.layer_staging) {
        assert_eq!(staging.requested_groups, routes.experts().len());
        assert_eq!(staging.requested_weights, routes.experts().len());
    }

    scheduler.cancel(&member_b).unwrap();
    let member_c = scheduler
        .admit(
            request(b"member-c", b"state-c", vec![4, 5, 6]),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let second = scheduler.step(&CancellationToken::new()).await.unwrap();
    assert_eq!(second.rows.len(), 2);
    assert_eq!(second.rows[0].member_id_sha256, member_a);
    assert_eq!(second.rows[0].token_id, expected_a[2]);
    assert!(second.rows[0].completed);
    assert_eq!(second.rows[1].member_id_sha256, member_c);
    assert_eq!(second.rows[1].token_id, expected_c[3]);
    assert!(!second.rows[1].completed);
    assert_eq!(second.evidence.completed_members, 1);
    assert_eq!(second.evidence.continued_members, 1);

    let third = scheduler.step(&CancellationToken::new()).await.unwrap();
    assert_eq!(third.rows.len(), 1);
    assert_eq!(third.rows[0].member_id_sha256, member_c);
    assert_eq!(third.rows[0].token_id, expected_c[4]);
    assert!(third.rows[0].completed);
    assert_eq!(scheduler.active_members().len(), 0);

    let evidence = scheduler.finish().unwrap();
    assert_eq!(evidence.admitted_members, 3);
    assert_eq!(evidence.completed_members, 2);
    assert_eq!(evidence.cancelled_members, 1);
    assert_eq!(evidence.committed_steps, 3);
    assert_eq!(evidence.processed_rows, 5);
    assert_eq!(runtime.admission_snapshot().active, 0);
}

#[tokio::test]
async fn member_token_cancellation_is_reaped_before_the_next_model_step() {
    let (_directory, runtime, model) = streaming_model();
    let scheduler = OlmoeContinuousBatch::new(
        Arc::new(model),
        ExecutionBatchBinding::new("d".repeat(64), "e".repeat(64), "f".repeat(64)).unwrap(),
    )
    .unwrap();
    let make_request = |member: &[u8], state: &[u8], prompt: Vec<u32>| {
        OlmoeContinuousRequest::for_identifiers(member, state, prompt, 1, None, runtime.limits())
            .unwrap()
    };
    let active_cancellation = CancellationToken::new();
    let cancelled = CancellationToken::new();
    scheduler
        .admit(
            make_request(b"active", b"active-state", vec![1]),
            active_cancellation,
        )
        .await
        .unwrap();
    scheduler
        .admit(
            make_request(b"cancelled", b"cancelled-state", vec![2]),
            cancelled.clone(),
        )
        .await
        .unwrap();
    cancelled.cancel();

    let step = scheduler.step(&CancellationToken::new()).await.unwrap();
    assert_eq!(step.rows.len(), 1);
    assert!(step.rows[0].completed);
    assert_eq!(step.evidence.row_count, 1);
    let evidence = scheduler.finish().unwrap();
    assert_eq!(evidence.completed_members, 1);
    assert_eq!(evidence.cancelled_members, 1);
    assert_eq!(runtime.admission_snapshot().active, 0);
}

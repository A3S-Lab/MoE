use a3s_moe::qwen3_moe::{Qwen3MoeExpertWeights, Qwen3MoeSparseLayer};
use a3s_moe::{Matrix, MoeLayerConfig};
use approx::assert_abs_diff_eq;
use serde::Deserialize;

const ORACLE_SCHEMA: &str = "a3s.moe.qwen3-moe-sparse-oracle.v1";

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Oracle {
    schema: String,
    config: MoeLayerConfig,
    layer: u32,
    hidden_states: Vec<Vec<f32>>,
    router_weight: Vec<Vec<f32>>,
    experts: Vec<Expert>,
    output: Output,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Expert {
    gate_up: Vec<Vec<f32>>,
    down: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Output {
    router_logits: Vec<Vec<f32>>,
    routes: Vec<Vec<Route>>,
    hidden_states: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Route {
    expert: u32,
    weight: f32,
}

#[test]
fn qwen3_sparse_layer_matches_the_dependency_free_oracle() {
    let oracle: Oracle =
        serde_json::from_str(include_str!("fixtures/qwen3_moe_sparse_oracle.json")).unwrap();
    assert_eq!(oracle.schema, ORACLE_SCHEMA);
    oracle.config.validate().unwrap();

    let hidden_states = matrix(&oracle.hidden_states);
    let router_weight = matrix(&oracle.router_weight);
    let experts = oracle
        .experts
        .iter()
        .map(|expert| {
            Qwen3MoeExpertWeights::new(oracle.config, matrix(&expert.gate_up), matrix(&expert.down))
                .unwrap()
        })
        .collect();
    let layer = Qwen3MoeSparseLayer::new(oracle.config, router_weight, experts).unwrap();
    let actual = layer.forward(oracle.layer, &hidden_states).unwrap();

    compare_matrix(&actual.routing.logits, &oracle.output.router_logits);
    compare_matrix(&actual.hidden_states, &oracle.output.hidden_states);
    for (actual_row, expected_row) in actual
        .routing
        .routes
        .selections()
        .iter()
        .zip(&oracle.output.routes)
    {
        assert_eq!(actual_row.len(), expected_row.len());
        for (actual, expected) in actual_row.iter().zip(expected_row) {
            assert_eq!(actual.expert, expected.expert);
            assert_abs_diff_eq!(actual.weight, expected.weight, epsilon = 1e-6);
        }
        assert_abs_diff_eq!(
            actual_row.iter().map(|route| route.weight).sum::<f32>(),
            1.0,
            epsilon = 1e-6
        );
    }
}

fn matrix(rows: &[Vec<f32>]) -> Matrix {
    let columns = rows.first().unwrap().len();
    assert!(rows.iter().all(|row| row.len() == columns));
    Matrix::new(
        rows.len(),
        columns,
        rows.iter().flatten().copied().collect(),
    )
    .unwrap()
}

fn compare_matrix(actual: &Matrix, expected: &[Vec<f32>]) {
    assert_eq!(actual.rows(), expected.len());
    for (actual_row, expected_row) in (0..actual.rows())
        .map(|row| actual.row(row).unwrap())
        .zip(expected)
    {
        assert_eq!(actual_row.len(), expected_row.len());
        for (&actual, &expected) in actual_row.iter().zip(expected_row) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-6);
        }
    }
}

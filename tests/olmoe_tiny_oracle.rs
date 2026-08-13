use a3s_moe::olmoe::{OlmoeExpertWeights, OlmoeMoeConfig, OlmoeMoeLayer};
use a3s_moe::Matrix;
use approx::assert_abs_diff_eq;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Fixture {
    source: String,
    layer: u32,
    config: OlmoeMoeConfig,
    hidden_states: Matrix,
    router_weight: Matrix,
    experts: Vec<ExpertFixture>,
    expected: ExpectedFixture,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ExpertFixture {
    gate_up: Matrix,
    down: Matrix,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ExpectedFixture {
    logits: Matrix,
    routes: Vec<Vec<RouteFixture>>,
    hidden_states: Matrix,
}

#[derive(Deserialize)]
struct RouteFixture {
    expert: u32,
    weight: f32,
}

#[test]
fn tiny_layer_matches_the_pinned_reference_oracle() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/olmoe_tiny_oracle.json")).unwrap();
    assert!(fixture
        .source
        .contains("918dbf131d0df5b46e3f6e1d96174d62aa4d16d6"));
    let experts = fixture
        .experts
        .into_iter()
        .map(|expert| OlmoeExpertWeights::new(fixture.config, expert.gate_up, expert.down))
        .collect::<a3s_moe::Result<Vec<_>>>()
        .unwrap();
    let layer = OlmoeMoeLayer::new(fixture.config, fixture.router_weight, experts).unwrap();

    let actual = layer
        .forward(fixture.layer, &fixture.hidden_states)
        .unwrap();

    assert_eq!(actual.routing.logits.rows(), fixture.expected.logits.rows());
    assert_eq!(
        actual.routing.logits.columns(),
        fixture.expected.logits.columns()
    );
    for (actual, expected) in actual
        .routing
        .logits
        .values()
        .iter()
        .zip(fixture.expected.logits.values())
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-6);
    }
    for (actual, expected) in actual
        .routing
        .routes
        .selections()
        .iter()
        .zip(&fixture.expected.routes)
    {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.expert, expected.expert);
            assert_abs_diff_eq!(actual.weight, expected.weight, epsilon = 1e-6);
        }
    }
    assert_eq!(
        actual.hidden_states.rows(),
        fixture.expected.hidden_states.rows()
    );
    assert_eq!(
        actual.hidden_states.columns(),
        fixture.expected.hidden_states.columns()
    );
    for (actual, expected) in actual
        .hidden_states
        .values()
        .iter()
        .zip(fixture.expected.hidden_states.values())
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-6);
    }
}

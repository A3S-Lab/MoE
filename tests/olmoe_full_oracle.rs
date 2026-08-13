use approx::assert_abs_diff_eq;
use candle_core::{Device, Tensor};
use serde::Deserialize;

mod support;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct FullOracle {
    source: String,
    tokens: Vec<u32>,
    logits: Vec<Vec<f32>>,
    layer_routes: Vec<Vec<Vec<RouteOracle>>>,
}

#[derive(Deserialize)]
struct RouteOracle {
    expert: u32,
    weight: f32,
}

#[test]
fn complete_decoder_matches_the_pinned_reference_equations() {
    let oracle: FullOracle =
        serde_json::from_str(include_str!("fixtures/olmoe_full_oracle.json")).unwrap();
    assert!(oracle
        .source
        .contains("918dbf131d0df5b46e3f6e1d96174d62aa4d16d6"));
    let model = support::tiny_model();
    let input = Tensor::from_vec(
        oracle.tokens.clone(),
        (1, oracle.tokens.len()),
        &Device::Cpu,
    )
    .unwrap();
    let output = model.forward(&input, &mut model.new_cache()).unwrap();
    let logits = output.logits.to_vec3::<f32>().unwrap();

    assert_eq!(logits[0].len(), oracle.logits.len());
    for (actual_row, expected_row) in logits[0].iter().zip(&oracle.logits) {
        for (actual, expected) in actual_row.iter().zip(expected_row) {
            assert_abs_diff_eq!(actual, expected, epsilon = 2e-4);
        }
    }
    assert_eq!(output.layer_routes.len(), oracle.layer_routes.len());
    for (actual_layer, expected_layer) in output.layer_routes.iter().zip(&oracle.layer_routes) {
        for (actual_routes, expected_routes) in actual_layer.selections().iter().zip(expected_layer)
        {
            for (actual, expected) in actual_routes.iter().zip(expected_routes) {
                assert_eq!(actual.expert, expected.expert);
                assert_abs_diff_eq!(actual.weight, expected.weight, epsilon = 2e-5);
            }
        }
    }
}

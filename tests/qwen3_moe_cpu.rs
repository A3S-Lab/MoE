use std::collections::HashMap;

use a3s_moe::qwen3_moe::{Qwen3MoeConfig, Qwen3MoeCpuModel, Qwen3MoeForwardOutput};
use approx::assert_abs_diff_eq;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

const ORACLE_SCHEMA: &str = "a3s.moe.qwen3-moe-full-oracle.v1";

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct FullOracle {
    schema: String,
    source: String,
    tokens: Vec<u32>,
    logits: Vec<Vec<f32>>,
    layer_router_logits: Vec<Option<Vec<Vec<f32>>>>,
    layer_routes: Vec<Option<Vec<Vec<RouteOracle>>>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RouteOracle {
    expert: u32,
    weight: f32,
}

fn tiny_config() -> Qwen3MoeConfig {
    Qwen3MoeConfig {
        model_type: "qwen3_moe".to_string(),
        vocab_size: 13,
        hidden_size: 4,
        intermediate_size: 5,
        moe_intermediate_size: 3,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: Some(4),
        num_experts: 3,
        num_experts_per_tok: 2,
        max_position_embeddings: 8,
        norm_topk_prob: true,
        hidden_act: "silu".to_string(),
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        attention_bias: false,
        attention_dropout: 0.0,
        decoder_sparse_step: 2,
        mlp_only_layers: Vec::new(),
        use_sliding_window: false,
        sliding_window: None,
        tie_word_embeddings: false,
        bos_token_id: Some(1),
        eos_token_id: Some(a3s_moe::qwen3_moe::Qwen3MoeTokenIds::One(12)),
        pad_token_id: Some(0),
    }
}

fn tensor_values(elements: usize, seed: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| {
            let centered = ((index * 17 + seed * 7) % 23) as f32 - 11.0;
            centered * 0.017
        })
        .collect()
}

fn insert_matrix(
    weights: &mut HashMap<String, Tensor>,
    name: String,
    rows: usize,
    columns: usize,
    seed: usize,
) {
    weights.insert(
        name,
        Tensor::from_vec(
            tensor_values(rows * columns, seed),
            (rows, columns),
            &Device::Cpu,
        )
        .unwrap(),
    );
}

fn insert_norm(weights: &mut HashMap<String, Tensor>, name: String, size: usize, seed: usize) {
    let values = tensor_values(size, seed)
        .into_iter()
        .map(|value| 1.0 + value * 0.1)
        .collect::<Vec<_>>();
    weights.insert(name, Tensor::from_vec(values, size, &Device::Cpu).unwrap());
}

fn tiny_weights(config: &Qwen3MoeConfig) -> HashMap<String, Tensor> {
    let mut weights = HashMap::new();
    insert_matrix(
        &mut weights,
        "model.embed_tokens.weight".to_string(),
        config.vocab_size,
        config.hidden_size,
        1,
    );
    insert_matrix(
        &mut weights,
        "lm_head.weight".to_string(),
        config.vocab_size,
        config.hidden_size,
        2,
    );
    insert_norm(
        &mut weights,
        "model.norm.weight".to_string(),
        config.hidden_size,
        3,
    );

    let head_dim = config.attention_head_dim().unwrap();
    let query_width = config.num_attention_heads * head_dim;
    let key_value_width = config.num_key_value_heads * head_dim;
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        insert_norm(
            &mut weights,
            format!("{prefix}.input_layernorm.weight"),
            config.hidden_size,
            10 + layer,
        );
        insert_norm(
            &mut weights,
            format!("{prefix}.post_attention_layernorm.weight"),
            config.hidden_size,
            20 + layer,
        );
        insert_matrix(
            &mut weights,
            format!("{prefix}.self_attn.q_proj.weight"),
            query_width,
            config.hidden_size,
            30 + layer,
        );
        insert_matrix(
            &mut weights,
            format!("{prefix}.self_attn.k_proj.weight"),
            key_value_width,
            config.hidden_size,
            40 + layer,
        );
        insert_matrix(
            &mut weights,
            format!("{prefix}.self_attn.v_proj.weight"),
            key_value_width,
            config.hidden_size,
            50 + layer,
        );
        insert_matrix(
            &mut weights,
            format!("{prefix}.self_attn.o_proj.weight"),
            config.hidden_size,
            query_width,
            60 + layer,
        );
        insert_norm(
            &mut weights,
            format!("{prefix}.self_attn.q_norm.weight"),
            head_dim,
            70 + layer,
        );
        insert_norm(
            &mut weights,
            format!("{prefix}.self_attn.k_norm.weight"),
            head_dim,
            80 + layer,
        );

        if config.is_sparse_layer(layer) {
            insert_matrix(
                &mut weights,
                format!("{prefix}.mlp.gate.weight"),
                config.num_experts,
                config.hidden_size,
                120 + layer,
            );
            let mut gate_up = Vec::with_capacity(
                config.num_experts * 2 * config.moe_intermediate_size * config.hidden_size,
            );
            let mut down = Vec::with_capacity(
                config.num_experts * config.hidden_size * config.moe_intermediate_size,
            );
            for expert in 0..config.num_experts {
                gate_up.extend(tensor_values(
                    config.moe_intermediate_size * config.hidden_size,
                    200 + layer * 10 + expert,
                ));
                gate_up.extend(tensor_values(
                    config.moe_intermediate_size * config.hidden_size,
                    300 + layer * 10 + expert,
                ));
                down.extend(tensor_values(
                    config.hidden_size * config.moe_intermediate_size,
                    400 + layer * 10 + expert,
                ));
            }
            weights.insert(
                format!("{prefix}.mlp.experts.gate_up_proj"),
                Tensor::from_vec(
                    gate_up,
                    (
                        config.num_experts,
                        2 * config.moe_intermediate_size,
                        config.hidden_size,
                    ),
                    &Device::Cpu,
                )
                .unwrap(),
            );
            weights.insert(
                format!("{prefix}.mlp.experts.down_proj"),
                Tensor::from_vec(
                    down,
                    (
                        config.num_experts,
                        config.hidden_size,
                        config.moe_intermediate_size,
                    ),
                    &Device::Cpu,
                )
                .unwrap(),
            );
        } else {
            insert_matrix(
                &mut weights,
                format!("{prefix}.mlp.gate_proj.weight"),
                config.intermediate_size,
                config.hidden_size,
                90 + layer,
            );
            insert_matrix(
                &mut weights,
                format!("{prefix}.mlp.up_proj.weight"),
                config.intermediate_size,
                config.hidden_size,
                100 + layer,
            );
            insert_matrix(
                &mut weights,
                format!("{prefix}.mlp.down_proj.weight"),
                config.hidden_size,
                config.intermediate_size,
                110 + layer,
            );
        }
    }
    weights
}

fn tiny_model_with_config(config: Qwen3MoeConfig) -> Qwen3MoeCpuModel {
    let weights = tiny_weights(&config);
    let builder = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    Qwen3MoeCpuModel::load(config, builder).unwrap()
}

fn tiny_model() -> Qwen3MoeCpuModel {
    tiny_model_with_config(tiny_config())
}

fn logits(output: &Qwen3MoeForwardOutput) -> Vec<Vec<f32>> {
    output.logits.to_vec3::<f32>().unwrap()[0].clone()
}

#[test]
fn complete_decoder_matches_the_pinned_reference_equations() {
    let oracle: FullOracle =
        serde_json::from_str(include_str!("fixtures/qwen3_moe_full_oracle.json")).unwrap();
    assert_eq!(oracle.schema, ORACLE_SCHEMA);
    assert!(oracle
        .source
        .contains("918dbf131d0df5b46e3f6e1d96174d62aa4d16d6"));

    let model = tiny_model();
    let input = Tensor::from_vec(
        oracle.tokens.clone(),
        (1, oracle.tokens.len()),
        &Device::Cpu,
    )
    .unwrap();
    let output = model.forward(&input, &mut model.new_cache()).unwrap();
    for (actual_row, expected_row) in logits(&output).iter().zip(&oracle.logits) {
        for (actual, expected) in actual_row.iter().zip(expected_row) {
            assert_abs_diff_eq!(actual, expected, epsilon = 3e-4);
        }
    }

    assert_eq!(
        output.layer_router_logits.len(),
        oracle.layer_router_logits.len()
    );
    for (actual, expected) in output
        .layer_router_logits
        .iter()
        .zip(&oracle.layer_router_logits)
    {
        match (actual, expected) {
            (None, None) => {}
            (Some(actual), Some(expected)) => {
                for (actual_row, expected_row) in
                    actual.to_vec3::<f32>().unwrap()[0].iter().zip(expected)
                {
                    for (actual, expected) in actual_row.iter().zip(expected_row) {
                        assert_abs_diff_eq!(actual, expected, epsilon = 2e-5);
                    }
                }
            }
            _ => panic!("dense/sparse router output schedule differed"),
        }
    }
    for (actual, expected) in output.layer_routes.iter().zip(&oracle.layer_routes) {
        match (actual, expected) {
            (None, None) => {}
            (Some(actual), Some(expected)) => {
                for (actual_routes, expected_routes) in actual.selections().iter().zip(expected) {
                    for (actual, expected) in actual_routes.iter().zip(expected_routes) {
                        assert_eq!(actual.expert, expected.expert);
                        assert_abs_diff_eq!(actual.weight, expected.weight, epsilon = 2e-5);
                    }
                }
            }
            _ => panic!("dense/sparse route schedule differed"),
        }
    }
}

#[test]
fn prefill_matches_incremental_decode_with_full_and_sliding_attention() {
    for sliding_window in [None, Some(2)] {
        let mut config = tiny_config();
        config.use_sliding_window = sliding_window.is_some();
        config.sliding_window = sliding_window;
        let model = tiny_model_with_config(config);
        let tokens = [1_u32, 4, 2, 7];
        let input = Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), &Device::Cpu).unwrap();
        let mut prefill_cache = model.new_cache();
        let prefill = model.forward(&input, &mut prefill_cache).unwrap();
        let prefill_logits = logits(&prefill);
        assert_eq!(prefill_cache.position(), tokens.len());
        assert_eq!(prefill_cache.resident_bytes().unwrap(), 256);

        let mut decode_cache = model.new_cache();
        let mut decode_logits = Vec::new();
        for token in tokens {
            let input = Tensor::from_vec(vec![token], (1, 1), &Device::Cpu).unwrap();
            decode_logits
                .push(logits(&model.forward(&input, &mut decode_cache).unwrap())[0].clone());
        }
        assert_eq!(decode_cache.position(), tokens.len());
        for (prefill, decoded) in prefill_logits.iter().zip(decode_logits) {
            for (prefill, decoded) in prefill.iter().zip(decoded) {
                assert_abs_diff_eq!(prefill, &decoded, epsilon = 3e-5);
            }
        }
    }
}

#[test]
fn generation_cache_limits_and_public_thread_safety_hold() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Qwen3MoeCpuModel>();

    let model = tiny_model();
    assert_eq!(model.generate_greedy(&[1, 2], 3, None).unwrap().len(), 5);
    let mut cache = model.new_cache();
    let full = Tensor::from_vec(vec![1_u32; 8], (1, 8), &Device::Cpu).unwrap();
    model.forward(&full, &mut cache).unwrap();
    let overflow = Tensor::from_vec(vec![2_u32], (1, 1), &Device::Cpu).unwrap();
    assert!(model.forward(&overflow, &mut cache).is_err());
    assert_eq!(cache.position(), 8);
}

#[test]
fn tied_embeddings_do_not_require_an_lm_head_tensor() {
    let mut config = tiny_config();
    config.tie_word_embeddings = true;
    let mut weights = tiny_weights(&config);
    weights.remove("lm_head.weight");
    let builder = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let model = Qwen3MoeCpuModel::load(config, builder).unwrap();
    let input = Tensor::from_vec(vec![1_u32], (1, 1), &Device::Cpu).unwrap();
    assert_eq!(
        model
            .forward(&input, &mut model.new_cache())
            .unwrap()
            .logits
            .dims(),
        [1, 1, 13]
    );
}

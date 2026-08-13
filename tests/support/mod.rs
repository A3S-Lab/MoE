use std::collections::HashMap;

use a3s_moe::olmoe::{OlmoeConfig, OlmoeCpuModel};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

pub fn tiny_config() -> OlmoeConfig {
    OlmoeConfig {
        model_type: "olmoe".to_string(),
        vocab_size: 11,
        hidden_size: 4,
        intermediate_size: 3,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        num_experts: 3,
        num_experts_per_tok: 2,
        max_position_embeddings: 8,
        norm_topk_prob: false,
        hidden_act: "silu".to_string(),
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        attention_bias: false,
        clip_qkv: None,
        tie_word_embeddings: false,
        eos_token_id: Some(10),
        pad_token_id: Some(0),
    }
}

pub fn tensor_values(elements: usize, seed: usize) -> Vec<f32> {
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

pub fn tiny_weights(config: &OlmoeConfig) -> HashMap<String, Tensor> {
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

    let head_dim = config.hidden_size / config.num_attention_heads;
    let kv_size = config.num_key_value_heads * head_dim;
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
            config.hidden_size,
            config.hidden_size,
            30 + layer,
        );
        insert_matrix(
            &mut weights,
            format!("{prefix}.self_attn.k_proj.weight"),
            kv_size,
            config.hidden_size,
            40 + layer,
        );
        insert_matrix(
            &mut weights,
            format!("{prefix}.self_attn.v_proj.weight"),
            kv_size,
            config.hidden_size,
            50 + layer,
        );
        insert_matrix(
            &mut weights,
            format!("{prefix}.self_attn.o_proj.weight"),
            config.hidden_size,
            config.hidden_size,
            60 + layer,
        );
        insert_norm(
            &mut weights,
            format!("{prefix}.self_attn.q_norm.weight"),
            config.hidden_size,
            70 + layer,
        );
        insert_norm(
            &mut weights,
            format!("{prefix}.self_attn.k_norm.weight"),
            kv_size,
            80 + layer,
        );
        insert_matrix(
            &mut weights,
            format!("{prefix}.mlp.gate.weight"),
            config.num_experts,
            config.hidden_size,
            90 + layer,
        );
        for expert in 0..config.num_experts {
            let expert_prefix = format!("{prefix}.mlp.experts.{expert}");
            insert_matrix(
                &mut weights,
                format!("{expert_prefix}.gate_proj.weight"),
                config.intermediate_size,
                config.hidden_size,
                100 + layer * 10 + expert,
            );
            insert_matrix(
                &mut weights,
                format!("{expert_prefix}.up_proj.weight"),
                config.intermediate_size,
                config.hidden_size,
                200 + layer * 10 + expert,
            );
            insert_matrix(
                &mut weights,
                format!("{expert_prefix}.down_proj.weight"),
                config.hidden_size,
                config.intermediate_size,
                300 + layer * 10 + expert,
            );
        }
    }
    weights
}

#[allow(dead_code)]
pub fn tiny_model() -> OlmoeCpuModel {
    let config = tiny_config();
    let weights = tiny_weights(&config);
    let builder = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    OlmoeCpuModel::load(config, builder).unwrap()
}

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use a3s_moe::qwen3_moe::{Qwen3MoeCheckpoint, Qwen3MoeConfig, Qwen3MoeTokenIds};
use candle_core::{DType, Device, Tensor};
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;

use super::tensor_values;

pub fn tiny_config() -> Qwen3MoeConfig {
    Qwen3MoeConfig {
        model_type: "qwen3_moe".to_string(),
        vocab_size: 32,
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
        eos_token_id: Some(Qwen3MoeTokenIds::One(12)),
        pad_token_id: Some(0),
    }
}

pub fn tiny_weights(config: &Qwen3MoeConfig) -> HashMap<String, Tensor> {
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
            let mut gate_up = Vec::new();
            let mut down = Vec::new();
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

pub fn write_source(root: &Path, dtype: DType) -> Qwen3MoeCheckpoint {
    fs::create_dir(root).unwrap();
    let config = tiny_config();
    fs::write(
        root.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    let weights = tiny_weights(&config)
        .into_iter()
        .map(|(name, tensor)| (name, tensor.to_dtype(dtype).unwrap()))
        .collect::<HashMap<_, _>>();
    let shard = "model-00001-of-00001.safetensors";
    candle_core::safetensors::save(&weights, root.join(shard)).unwrap();
    let weight_map = weights
        .keys()
        .map(|name| (name.clone(), shard.to_string()))
        .collect::<HashMap<_, _>>();
    fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
    )
    .unwrap();
    write_tokenizer(&root.join("tokenizer.json"));
    Qwen3MoeCheckpoint::open(root).unwrap()
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

fn write_tokenizer(path: &Path) {
    let vocab_path = path.with_file_name("qwen-vocab.json");
    fs::write(
        &vocab_path,
        r#"{"[UNK]":0,"hello":1,"world":2,"user":3,":":4,"assistant":5,"system":6,"answer":7,"one":8,"two":9,"three":10,"four":11,"eos":12}"#,
    )
    .unwrap();
    let model = WordLevel::builder()
        .files(vocab_path.to_string_lossy().into_owned())
        .unk_token("[UNK]".to_string())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer.save(path, false).unwrap();
}

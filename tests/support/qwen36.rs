use std::collections::HashMap;
use std::fs;
use std::path::Path;

use a3s_moe::qwen3_5_moe::{
    Qwen36MoeCheckpoint, Qwen36MoeConfig, Qwen36MoeLayerType, Qwen36MoeRopeParameters,
    Qwen36MoeTextConfig, Qwen36MoeTokenIds,
};
use candle_core::{DType, Device, Tensor};
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;

pub fn tiny_config() -> Qwen36MoeConfig {
    Qwen36MoeConfig {
        model_type: "qwen3_5_moe".to_string(),
        text_config: Qwen36MoeTextConfig {
            model_type: "qwen3_5_moe_text".to_string(),
            vocab_size: 17,
            hidden_size: 4,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 16,
            full_attention_interval: 4,
            layer_types: vec![
                Qwen36MoeLayerType::LinearAttention,
                Qwen36MoeLayerType::LinearAttention,
                Qwen36MoeLayerType::LinearAttention,
                Qwen36MoeLayerType::FullAttention,
            ],
            linear_conv_kernel_dim: 2,
            linear_key_head_dim: 2,
            linear_num_key_heads: 1,
            linear_num_value_heads: 2,
            linear_value_head_dim: 2,
            mamba_ssm_dtype: "float32".to_string(),
            moe_intermediate_size: 3,
            shared_expert_intermediate_size: 3,
            num_experts: 3,
            num_experts_per_tok: 2,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_parameters: Qwen36MoeRopeParameters {
                rope_type: "default".to_string(),
                rope_theta: 10_000.0,
                partial_rotary_factor: 1.0,
                mrope_interleaved: true,
                mrope_section: vec![1, 1, 0],
            },
            attention_bias: false,
            attention_dropout: 0.0,
            attn_output_gate: true,
            tie_word_embeddings: false,
            bos_token_id: Some(1),
            eos_token_id: Some(Qwen36MoeTokenIds::One(16)),
            pad_token_id: Some(0),
        },
        tie_word_embeddings: false,
        image_token_id: None,
        video_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
    }
}

pub fn tiny_tensors(config: &Qwen36MoeConfig) -> HashMap<String, Tensor> {
    let mut tensors = HashMap::new();
    let mut seed = 1_usize;
    insert(
        &mut tensors,
        "model.language_model.embed_tokens.weight",
        &[config.vocab_size, config.hidden_size],
        &mut seed,
        false,
    );
    insert(
        &mut tensors,
        "model.language_model.norm.weight",
        &[config.hidden_size],
        &mut seed,
        true,
    );
    insert(
        &mut tensors,
        "lm_head.weight",
        &[config.vocab_size, config.hidden_size],
        &mut seed,
        false,
    );
    let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let conv_dim = 2 * key_dim + value_dim;
    for (layer, layer_type) in config.layer_types.iter().enumerate() {
        let prefix = format!("model.language_model.layers.{layer}");
        insert(
            &mut tensors,
            &format!("{prefix}.input_layernorm.weight"),
            &[config.hidden_size],
            &mut seed,
            true,
        );
        insert(
            &mut tensors,
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[config.hidden_size],
            &mut seed,
            true,
        );
        match layer_type {
            Qwen36MoeLayerType::LinearAttention => {
                let prefix = format!("{prefix}.linear_attn");
                insert(
                    &mut tensors,
                    &format!("{prefix}.A_log"),
                    &[config.linear_num_value_heads],
                    &mut seed,
                    false,
                );
                insert(
                    &mut tensors,
                    &format!("{prefix}.conv1d.weight"),
                    &[conv_dim, 1, config.linear_conv_kernel_dim],
                    &mut seed,
                    false,
                );
                insert(
                    &mut tensors,
                    &format!("{prefix}.dt_bias"),
                    &[config.linear_num_value_heads],
                    &mut seed,
                    false,
                );
                for (name, rows) in [
                    ("in_proj_a.weight", config.linear_num_value_heads),
                    ("in_proj_b.weight", config.linear_num_value_heads),
                    ("in_proj_qkv.weight", conv_dim),
                    ("in_proj_z.weight", value_dim),
                ] {
                    insert(
                        &mut tensors,
                        &format!("{prefix}.{name}"),
                        &[rows, config.hidden_size],
                        &mut seed,
                        false,
                    );
                }
                insert(
                    &mut tensors,
                    &format!("{prefix}.norm.weight"),
                    &[config.linear_value_head_dim],
                    &mut seed,
                    false,
                );
                insert(
                    &mut tensors,
                    &format!("{prefix}.out_proj.weight"),
                    &[config.hidden_size, value_dim],
                    &mut seed,
                    false,
                );
            }
            Qwen36MoeLayerType::FullAttention => {
                let prefix = format!("{prefix}.self_attn");
                insert(
                    &mut tensors,
                    &format!("{prefix}.q_proj.weight"),
                    &[
                        2 * config.num_attention_heads * config.head_dim,
                        config.hidden_size,
                    ],
                    &mut seed,
                    false,
                );
                for name in ["k_proj.weight", "v_proj.weight"] {
                    insert(
                        &mut tensors,
                        &format!("{prefix}.{name}"),
                        &[
                            config.num_key_value_heads * config.head_dim,
                            config.hidden_size,
                        ],
                        &mut seed,
                        false,
                    );
                }
                insert(
                    &mut tensors,
                    &format!("{prefix}.o_proj.weight"),
                    &[
                        config.hidden_size,
                        config.num_attention_heads * config.head_dim,
                    ],
                    &mut seed,
                    false,
                );
                for name in ["q_norm.weight", "k_norm.weight"] {
                    insert(
                        &mut tensors,
                        &format!("{prefix}.{name}"),
                        &[config.head_dim],
                        &mut seed,
                        true,
                    );
                }
            }
        }
        let prefix = format!("model.language_model.layers.{layer}.mlp");
        insert(
            &mut tensors,
            &format!("{prefix}.gate.weight"),
            &[config.num_experts, config.hidden_size],
            &mut seed,
            false,
        );
        insert(
            &mut tensors,
            &format!("{prefix}.experts.gate_up_proj"),
            &[
                config.num_experts,
                2 * config.moe_intermediate_size,
                config.hidden_size,
            ],
            &mut seed,
            false,
        );
        insert(
            &mut tensors,
            &format!("{prefix}.experts.down_proj"),
            &[
                config.num_experts,
                config.hidden_size,
                config.moe_intermediate_size,
            ],
            &mut seed,
            false,
        );
        for (name, shape) in [
            (
                "shared_expert.gate_proj.weight",
                vec![config.shared_expert_intermediate_size, config.hidden_size],
            ),
            (
                "shared_expert.up_proj.weight",
                vec![config.shared_expert_intermediate_size, config.hidden_size],
            ),
            (
                "shared_expert.down_proj.weight",
                vec![config.hidden_size, config.shared_expert_intermediate_size],
            ),
            ("shared_expert_gate.weight", vec![1, config.hidden_size]),
        ] {
            insert(
                &mut tensors,
                &format!("{prefix}.{name}"),
                &shape,
                &mut seed,
                false,
            );
        }
    }
    tensors
}

pub fn write_source(root: &Path, dtype: DType) -> Qwen36MoeCheckpoint {
    fs::create_dir(root).unwrap();
    let config = tiny_config();
    let tensors = tiny_tensors(&config)
        .into_iter()
        .map(|(name, tensor)| (name, tensor.to_dtype(dtype).unwrap()))
        .collect::<HashMap<_, _>>();
    fs::write(
        root.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    let shard = "model-00001-of-00001.safetensors";
    candle_core::safetensors::save(&tensors, root.join(shard)).unwrap();
    let weight_map = tensors
        .keys()
        .map(|name| (name.clone(), shard.to_string()))
        .collect::<HashMap<_, _>>();
    fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
    )
    .unwrap();
    write_tokenizer(&root.join("tokenizer.json"));
    Qwen36MoeCheckpoint::open(root).unwrap()
}

fn insert(
    tensors: &mut HashMap<String, Tensor>,
    name: &str,
    shape: &[usize],
    seed: &mut usize,
    zero_centered_norm: bool,
) {
    let elements = shape.iter().product();
    let values = (0..elements)
        .map(|index| {
            let value = (((index * 17 + *seed * 7) % 23) as f32 - 11.0) * 0.017;
            if zero_centered_norm {
                value * 0.1
            } else if name.ends_with("linear_attn.norm.weight") {
                1.0 + value * 0.1
            } else {
                value
            }
        })
        .collect::<Vec<_>>();
    tensors.insert(
        name.to_string(),
        Tensor::from_vec(values, shape, &Device::Cpu).unwrap(),
    );
    *seed += 1;
}

fn write_tokenizer(path: &Path) {
    let vocab_path = path.with_file_name("qwen36-vocab.json");
    fs::write(
        &vocab_path,
        r#"{"[UNK]":0,"hello":1,"world":2,"user":3,":":4,"assistant":5,"system":6,"answer":7,"one":8,"two":9,"three":10,"four":11,"five":12,"six":13,"seven":14,"eight":15,"eos":16}"#,
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

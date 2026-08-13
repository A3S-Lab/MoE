use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use a3s_moe::olmoe::{OlmoeCheckpoint, OlmoeConversionOptions};
use a3s_power::inference::InferenceLimits;
use candle_core::DType;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;

use crate::support::{tiny_config, tiny_weights};

pub fn write_packed(root: &Path) -> PathBuf {
    let source_path = root.join("source");
    fs::create_dir(&source_path).unwrap();
    let config = tiny_config();
    fs::write(
        source_path.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    let weights = tiny_weights(&config)
        .into_iter()
        .map(|(name, tensor)| (name, tensor.to_dtype(DType::F32).unwrap()))
        .collect::<HashMap<_, _>>();
    let shard = "model-00001-of-00001.safetensors";
    candle_core::safetensors::save(&weights, source_path.join(shard)).unwrap();
    let weight_map = weights
        .keys()
        .map(|name| (name.clone(), shard.to_string()))
        .collect::<HashMap<_, _>>();
    fs::write(
        source_path.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
    )
    .unwrap();
    write_tokenizer(&source_path.join("tokenizer.json"));

    let packed = root.join("packed");
    OlmoeCheckpoint::open(&source_path)
        .unwrap()
        .convert_to_packed(
            &packed,
            &InferenceLimits::default(),
            OlmoeConversionOptions {
                experts_per_file: 2,
                max_buffer_bytes: 1024 * 1024,
            },
        )
        .unwrap();
    packed
}

fn write_tokenizer(path: &Path) {
    let vocab_path = path.with_file_name("vocab.json");
    fs::write(
        &vocab_path,
        r#"{"[UNK]":0,"hello":1,"world":2,"user":3,":":4,"assistant":5,"system":6,"answer":7,"one":8,"two":9,"eos":10}"#,
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

use a3s_moe::olmoe::OlmoeCpuModel;
use approx::assert_abs_diff_eq;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

mod support;
use support::{tiny_config, tiny_model, tiny_weights};

#[test]
fn full_prefill_matches_incremental_kv_decode() {
    let model = tiny_model();
    let tokens = [1_u32, 4, 2, 7];
    let input = Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), &Device::Cpu).unwrap();
    let mut prefill_cache = model.new_cache();
    let prefill = model.forward(&input, &mut prefill_cache).unwrap();
    assert_eq!(prefill.logits.dims(), [1, tokens.len(), 11]);
    assert_eq!(prefill.layer_routes.len(), 2);
    assert_eq!(prefill_cache.position(), tokens.len());
    let prefill_logits = prefill.logits.to_vec3::<f32>().unwrap();

    let mut decode_cache = model.new_cache();
    let mut decode_logits = Vec::new();
    for token in tokens {
        let input = Tensor::from_vec(vec![token], (1, 1), &Device::Cpu).unwrap();
        let output = model.forward(&input, &mut decode_cache).unwrap();
        decode_logits.push(output.logits.to_vec3::<f32>().unwrap()[0][0].clone());
    }
    assert_eq!(decode_cache.position(), tokens.len());
    for (prefill, decoded) in prefill_logits[0].iter().zip(decode_logits) {
        for (prefill, decoded) in prefill.iter().zip(decoded) {
            assert_abs_diff_eq!(prefill, &decoded, epsilon = 2e-5);
        }
    }
}

#[test]
fn cache_limit_fails_without_partially_advancing_state() {
    let model = tiny_model();
    let mut cache = model.new_cache();
    let full = Tensor::from_vec(vec![1_u32; 8], (1, 8), &Device::Cpu).unwrap();
    model.forward(&full, &mut cache).unwrap();
    assert_eq!(cache.position(), 8);

    let overflow = Tensor::from_vec(vec![2_u32], (1, 1), &Device::Cpu).unwrap();
    assert!(model.forward(&overflow, &mut cache).is_err());
    assert_eq!(cache.position(), 8);
}

#[test]
fn greedy_generation_uses_the_same_cache_path() {
    let model = tiny_model();
    let generated = model.generate_greedy(&[1, 2], 3, None).unwrap();
    assert_eq!(&generated[..2], [1, 2]);
    assert_eq!(generated.len(), 5);
}

#[test]
fn cpu_model_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<OlmoeCpuModel>();
}

#[test]
fn tied_embeddings_do_not_require_a_duplicate_lm_head() {
    let mut config = tiny_config();
    config.tie_word_embeddings = true;
    let mut weights = tiny_weights(&config);
    weights.remove("lm_head.weight");
    let builder = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);

    let model = OlmoeCpuModel::load(config, builder).unwrap();
    let input = Tensor::from_vec(vec![1_u32], (1, 1), &Device::Cpu).unwrap();
    assert_eq!(
        model
            .forward(&input, &mut model.new_cache())
            .unwrap()
            .logits
            .dims(),
        [1, 1, 11]
    );
}

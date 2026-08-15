use std::fs;
use std::path::Path;

use a3s_power::inference::{InferenceLimits, WeightStore};

use crate::packing::{absolute_destination, convert_dense_tensors, convert_fused_expert_tensors};
use crate::{MoeError, PackedConversionOptions, PackedConversionReport, Result};

use super::{
    Qwen36MoePackedManifest, CONFIG_FILE, DENSE_DIRECTORY, EXPERT_DIRECTORY, MANIFEST_FILE,
    TOKENIZER_FILE,
};
use crate::qwen3_5_moe::checkpoint::dense_tensor_names;
use crate::qwen3_5_moe::Qwen36MoeCheckpoint;

pub type Qwen36MoeConversionOptions = PackedConversionOptions;
pub type Qwen36MoeConversionReport = PackedConversionReport;

impl Qwen36MoeCheckpoint {
    /// Converts the authenticated official checkpoint into a bounded-memory,
    /// text-only checkpoint. Vision and MTP tensors remain represented by the
    /// source digest but are deliberately not copied.
    pub fn convert_to_packed(
        &self,
        destination: impl AsRef<Path>,
        limits: &InferenceLimits,
        options: Qwen36MoeConversionOptions,
    ) -> Result<Qwen36MoeConversionReport> {
        let options = options.validate(self.config().num_experts)?;
        let destination = absolute_destination(destination.as_ref())?;
        if destination.exists() {
            return Err(MoeError::InvalidConfig(format!(
                "packed checkpoint destination '{}' already exists",
                destination.display()
            )));
        }
        let parent = destination.parent().ok_or_else(|| {
            MoeError::InvalidConfig("packed checkpoint destination has no parent".to_string())
        })?;
        fs::create_dir_all(parent)?;
        let staging = tempfile::Builder::new()
            .prefix(".a3s-moe-qwen36-packing-")
            .tempdir_in(parent)?;
        let dense_root = staging.path().join(DENSE_DIRECTORY);
        let expert_root = staging.path().join(EXPERT_DIRECTORY);
        fs::create_dir(&dense_root)?;
        fs::create_dir(&expert_root)?;

        let source_store = WeightStore::open(self.root(), limits)?;
        let mut peak_buffered_bytes = 0_u64;
        let dense_files = convert_dense_tensors(
            &source_store,
            &dense_tensor_names(self.config()),
            &dense_root,
            options.max_buffer_bytes,
            &mut peak_buffered_bytes,
        )?;
        let layers = (0..self.config().num_hidden_layers).collect::<Vec<_>>();
        let (expert_files, scalar_type, packed_experts) = convert_fused_expert_tensors(
            &source_store,
            &layers,
            self.config().moe_config()?,
            |layer| format!("model.language_model.layers.{layer}.mlp.experts"),
            &expert_root,
            options,
            &mut peak_buffered_bytes,
        )?;

        fs::write(
            staging.path().join(CONFIG_FILE),
            serde_json::to_vec_pretty(self.config())?,
        )?;
        let tokenizer_source = self.root().join(TOKENIZER_FILE);
        let tokenizer_included = tokenizer_source.exists();
        if tokenizer_included {
            self.load_tokenizer()?;
            fs::copy(&tokenizer_source, staging.path().join(TOKENIZER_FILE))?;
        }
        let dense_store = WeightStore::open(&dense_root, limits)?;
        let expert_store = WeightStore::open(&expert_root, limits)?;
        let manifest = Qwen36MoePackedManifest {
            schema: Qwen36MoePackedManifest::SCHEMA.to_string(),
            capabilities: vec![Qwen36MoePackedManifest::TEXT_GENERATION_CAPABILITY.to_string()],
            source_weights_sha256: source_store.sha256().to_string(),
            dense_weights_sha256: dense_store.sha256().to_string(),
            expert_weights_sha256: expert_store.sha256().to_string(),
            scalar_type,
            hidden_size: self.config().hidden_size,
            moe_intermediate_size: self.config().moe_intermediate_size,
            num_hidden_layers: self.config().num_hidden_layers,
            num_experts: self.config().num_experts,
            experts_per_file: options.experts_per_file,
            dense_files: dense_files.clone(),
            expert_files: expert_files.clone(),
            tokenizer_included,
        };
        fs::write(
            staging.path().join(MANIFEST_FILE),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        let report = Qwen36MoeConversionReport {
            destination: destination.clone(),
            source_weights_sha256: manifest.source_weights_sha256.clone(),
            dense_weights_sha256: manifest.dense_weights_sha256.clone(),
            expert_weights_sha256: manifest.expert_weights_sha256.clone(),
            dense_files: dense_files.len(),
            expert_files: expert_files.len(),
            packed_experts,
            peak_buffered_bytes,
        };

        drop(expert_store);
        drop(dense_store);
        drop(source_store);
        let staged_path = staging.keep();
        if let Err(error) = fs::rename(&staged_path, &destination) {
            let _ = fs::remove_dir_all(&staged_path);
            return Err(MoeError::Io(error));
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use a3s_power::inference::{
        DevicePreference, EmbeddedRuntime, InferenceLimits, ResidencyPolicy, RoutedExpertBatch,
    };
    use approx::assert_abs_diff_eq;
    use candle_core::{Device, Tensor};
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::qwen3_5_moe::test_support::{tiny_config, tiny_tensors};
    use crate::qwen3_5_moe::{Qwen36MoePackedCheckpoint, Qwen36MoeStreamingBatchRow};

    #[tokio::test]
    async fn bounded_text_conversion_matches_the_resident_mixed_decoder() {
        let fixture = tiny_packed_fixture();
        let resident = fixture.source.load_cpu_resident().unwrap();
        assert_eq!(fixture.report.dense_files, 64);
        assert_eq!(fixture.report.expert_files, 8);
        assert_eq!(fixture.report.packed_experts, 12);
        assert!(fixture.report.peak_buffered_bytes <= 1024);

        let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, fixture.limits.clone()).unwrap();
        let packed =
            Qwen36MoePackedCheckpoint::open(&fixture.packed_root, runtime.clone()).unwrap();
        assert_eq!(
            packed.manifest().capabilities,
            [Qwen36MoePackedManifest::TEXT_GENERATION_CAPABILITY]
        );
        assert!(packed.load_tokenizer().is_err());
        let streaming = packed
            .load_streaming(ResidencyPolicy {
                host_cache_bytes: 512,
                max_background_inflight_bytes: 1024,
                ..ResidencyPolicy::default()
            })
            .unwrap();
        let input = Tensor::from_slice(&[1_u32, 4, 2], (1, 3), &Device::Cpu).unwrap();
        let expected = resident.forward(&input, &mut resident.new_cache()).unwrap();
        let cancellation = CancellationToken::new();
        let permit = runtime.begin(&cancellation).unwrap();
        let actual = streaming
            .forward(&input, &mut streaming.new_cache(), &permit, &cancellation)
            .await
            .unwrap();
        compare(&actual.logits, &expected.logits);
        compare_routes(&actual.layer_routes, &expected.layer_routes, 1e-5);
        assert_eq!(actual.layer_staging.len(), 4);

        let first = Tensor::from_slice(&[1_u32, 2], (1, 2), &Device::Cpu).unwrap();
        let second = Tensor::from_slice(&[4_u32], (1, 1), &Device::Cpu).unwrap();
        let expected_first = resident.forward(&first, &mut resident.new_cache()).unwrap();
        let expected_second = resident
            .forward(&second, &mut resident.new_cache())
            .unwrap();
        let mut first_cache = streaming.new_cache();
        let mut second_cache = streaming.new_cache();
        let mut rows = [
            Qwen36MoeStreamingBatchRow::new(&first, &mut first_cache),
            Qwen36MoeStreamingBatchRow::new(&second, &mut second_cache),
        ];
        let batched = streaming
            .forward_batch(&mut rows, &permit, &cancellation)
            .await
            .unwrap();
        compare(&batched.rows[0].logits, &expected_first.logits);
        compare(&batched.rows[1].logits, &expected_second.logits);
        assert_eq!(batched.layer_union_routes.len(), 4);
        assert_eq!(first_cache.position(), 2);
        assert_eq!(second_cache.position(), 1);
    }

    #[cfg(feature = "cuda")]
    #[tokio::test]
    async fn packed_streaming_executes_on_cuda_and_matches_cpu() {
        let fixture = tiny_packed_fixture();
        let resident = fixture.source.load_cpu_resident().unwrap();
        let cpu_input = Tensor::from_slice(&[1_u32, 4, 2], (1, 3), &Device::Cpu).unwrap();
        let expected = resident
            .forward(&cpu_input, &mut resident.new_cache())
            .unwrap();

        let runtime = EmbeddedRuntime::new(
            DevicePreference::Cuda { ordinal: 0 },
            fixture.limits.clone(),
        )
        .unwrap();
        let packed =
            Qwen36MoePackedCheckpoint::open(&fixture.packed_root, runtime.clone()).unwrap();
        let streaming = packed
            .load_streaming(ResidencyPolicy {
                device_cache_bytes: 1024,
                max_background_inflight_bytes: 1024,
                ..ResidencyPolicy::default()
            })
            .unwrap();
        let input = cpu_input
            .to_device(runtime.device().tensor_device())
            .unwrap();
        let cancellation = CancellationToken::new();
        let permit = runtime.begin(&cancellation).unwrap();
        let mut cache = streaming.new_cache();
        let actual = streaming
            .forward(&input, &mut cache, &permit, &cancellation)
            .await
            .unwrap();

        compare_with_epsilon(&actual.logits, &expected.logits, 1e-4);
        compare_routes(&actual.layer_routes, &expected.layer_routes, 1e-5);
        assert_eq!(cache.position(), 3);
        assert!(cache.resident_bytes().unwrap() > 0);
        assert!(actual
            .logits
            .device()
            .same_device(runtime.device().tensor_device()));
    }

    struct TinyPackedFixture {
        _directory: tempfile::TempDir,
        source: Qwen36MoeCheckpoint,
        packed_root: PathBuf,
        limits: InferenceLimits,
        report: Qwen36MoeConversionReport,
    }

    fn tiny_packed_fixture() -> TinyPackedFixture {
        let directory = tempfile::tempdir().unwrap();
        let source_root = directory.path().join("source");
        std::fs::create_dir(&source_root).unwrap();
        let config = tiny_config();
        std::fs::write(
            source_root.join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();
        let mut weights = tiny_tensors(&config);
        weights.insert(
            "model.visual.patch_embed.proj.weight".to_string(),
            Tensor::zeros(1, candle_core::DType::F32, &Device::Cpu).unwrap(),
        );
        weights.insert(
            "mtp.layers.0.input_layernorm.weight".to_string(),
            Tensor::zeros(1, candle_core::DType::F32, &Device::Cpu).unwrap(),
        );
        let shard = "model-00001-of-00001.safetensors";
        candle_core::safetensors::save(&weights, source_root.join(shard)).unwrap();
        let weight_map = weights
            .keys()
            .map(|name| (name.clone(), shard.to_string()))
            .collect::<HashMap<_, _>>();
        std::fs::write(
            source_root.join("model.safetensors.index.json"),
            serde_json::to_vec_pretty(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
        )
        .unwrap();

        let source = Qwen36MoeCheckpoint::open(&source_root).unwrap();
        let packed_root = directory.path().join("packed");
        let limits = InferenceLimits {
            max_concurrent_requests: 2,
            ..InferenceLimits::default()
        };
        let report = source
            .convert_to_packed(
                &packed_root,
                &limits,
                Qwen36MoeConversionOptions {
                    experts_per_file: 2,
                    max_buffer_bytes: 1024,
                },
            )
            .unwrap();
        TinyPackedFixture {
            _directory: directory,
            source,
            packed_root,
            limits,
            report,
        }
    }

    fn compare(actual: &Tensor, expected: &Tensor) {
        compare_with_epsilon(actual, expected, 2e-5);
    }

    fn compare_with_epsilon(actual: &Tensor, expected: &Tensor, epsilon: f32) {
        let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = epsilon);
        }
    }

    fn compare_routes(actual: &[RoutedExpertBatch], expected: &[RoutedExpertBatch], epsilon: f32) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.layer(), expected.layer());
            assert_eq!(actual.expert_count(), expected.expert_count());
            assert_eq!(actual.experts(), expected.experts());
            assert_eq!(actual.selections().len(), expected.selections().len());
            for (actual, expected) in actual.selections().iter().zip(expected.selections()) {
                assert_eq!(actual.len(), expected.len());
                for (actual, expected) in actual.iter().zip(expected) {
                    assert_eq!(actual.expert, expected.expert);
                    assert_abs_diff_eq!(actual.weight, expected.weight, epsilon = epsilon);
                }
            }
        }
    }
}

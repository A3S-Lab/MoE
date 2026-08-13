use a3s_power::backend::Backend;

use crate::MoeArchitecture;

use super::*;

#[test]
fn manifest_matching_is_architecture_aware() {
    let olmoe = OlmoeBackend::new(OlmoeBackendConfig::default()).unwrap();
    let qwen = Qwen3MoeBackend::new(OlmoeBackendConfig::default()).unwrap();
    let mut manifest = ModelManifest::remote("test");
    manifest.format = ModelFormat::SafeTensors;
    manifest.family = Some("olmoe".to_string());
    assert!(olmoe.supports_manifest(&manifest));
    assert!(!qwen.supports_manifest(&manifest));
    manifest.family = Some("qwen3_moe".to_string());
    assert!(!olmoe.supports_manifest(&manifest));
    assert!(qwen.supports_manifest(&manifest));
}

#[test]
fn backends_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<OlmoeBackend>();
    assert_send_sync::<Qwen3MoeBackend>();
}

#[test]
fn architecture_config_factory_selects_the_family_residency_profile() {
    for architecture in [MoeArchitecture::Olmoe, MoeArchitecture::Qwen3Moe] {
        let config = OlmoeBackendConfig::for_architecture(architecture);
        let mut expected_limits = architecture.inference_limits();
        expected_limits.max_concurrent_requests = 4;
        expected_limits.max_queued_requests = 64;
        assert_eq!(config.inference_limits, expected_limits);
        assert_eq!(config.residency_policy, architecture.residency_policy());
        config.validate().unwrap();
    }
}

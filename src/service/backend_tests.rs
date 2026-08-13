use a3s_power::backend::Backend;

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

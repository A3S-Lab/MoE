use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use a3s_moe::olmoe::{
    OlmoeEncryptedCheckpointSource, OlmoePackedCheckpoint, OlmoePackedEncryptionReport,
};
use a3s_power::inference::{
    DevicePreference, EmbeddedRuntime, InferenceLimits, ResidencyPolicy, SeekableWeightKey,
    TelemetryMode,
};
use approx::assert_abs_diff_eq;
use candle_core::{Device, Tensor};
use tokio_util::sync::CancellationToken;

#[path = "support/packed.rs"]
mod packed;
mod support;
use packed::write_packed;

const CHUNK_BYTES: u32 = 4 * 1024;

#[tokio::test]
async fn encrypted_checkpoint_matches_plaintext_logits_routes_and_tokens() {
    let directory = tempfile::tempdir().unwrap();
    let packed_path = write_packed(directory.path());
    let encrypted_path = directory.path().join("encrypted");
    let limits = InferenceLimits::default();
    let plain_runtime = cpu_runtime(&limits);
    let plain_checkpoint =
        OlmoePackedCheckpoint::open(&packed_path, plain_runtime.clone()).unwrap();
    let key = SeekableWeightKey::new([0x31; 32]);

    let report = plain_checkpoint
        .encrypt_to_seekable(&encrypted_path, &key, CHUNK_BYTES, &limits)
        .unwrap();
    assert_eq!(report.chunk_bytes, CHUNK_BYTES);
    assert!(report.peak_plaintext_chunk_bytes <= u64::from(CHUNK_BYTES));
    assert_eq!(
        report.packed_weights_sha256,
        plain_checkpoint.manifest().weights_sha256()
    );
    assert!(encrypted_path.join("confidential.json").is_file());
    assert!(files_below(&encrypted_path)
        .iter()
        .all(|path| { path.extension().and_then(|value| value.to_str()) != Some("safetensors") }));

    let encrypted_runtime = cpu_runtime(&limits);
    let encrypted_checkpoint = OlmoePackedCheckpoint::open_seekable_encrypted(
        encrypted_source(&encrypted_path, &report, key),
        encrypted_runtime.clone(),
    )
    .unwrap();
    assert_eq!(
        encrypted_checkpoint.execution_batch_binding().unwrap(),
        plain_checkpoint.execution_batch_binding().unwrap()
    );

    let policy = streaming_policy();
    let plain_model = plain_checkpoint.load_streaming(policy.clone()).unwrap();
    let encrypted_model = encrypted_checkpoint.load_streaming(policy).unwrap();
    let input = Tensor::from_vec(vec![1_u32, 4, 2], (1, 3), &Device::Cpu).unwrap();
    let cancellation = CancellationToken::new();
    let plain_permit = plain_runtime.begin(&cancellation).unwrap();
    let plain_output = plain_model
        .forward(
            &input,
            &mut plain_model.new_cache(),
            &plain_permit,
            &cancellation,
        )
        .await
        .unwrap();
    let encrypted_permit = encrypted_runtime.begin(&cancellation).unwrap();
    let encrypted_output = encrypted_model
        .forward(
            &input,
            &mut encrypted_model.new_cache(),
            &encrypted_permit,
            &cancellation,
        )
        .await
        .unwrap();
    let expected = plain_output
        .logits
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let actual = encrypted_output
        .logits
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (actual, expected) in actual.iter().zip(expected) {
        assert_abs_diff_eq!(actual, &expected, epsilon = 2e-5);
    }
    assert_eq!(encrypted_output.layer_routes, plain_output.layer_routes);
    drop(plain_permit);
    drop(encrypted_permit);
    assert_eq!(
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            encrypted_model.generate_greedy(&[1, 2], 3, None, &CancellationToken::new()),
        )
        .await
        .expect("encrypted generation waited for admission")
        .unwrap(),
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            plain_model.generate_greedy(&[1, 2], 3, None, &CancellationToken::new()),
        )
        .await
        .expect("plaintext generation waited for admission")
        .unwrap()
    );
}

#[test]
fn encrypted_checkpoint_rejects_bad_trust_key_cancellation_and_tampering() {
    let directory = tempfile::tempdir().unwrap();
    let packed_path = write_packed(directory.path());
    let encrypted_path = directory.path().join("encrypted");
    let limits = InferenceLimits::default();
    let checkpoint = OlmoePackedCheckpoint::open(&packed_path, cpu_runtime(&limits)).unwrap();
    let key = SeekableWeightKey::new([0x52; 32]);
    let report = checkpoint
        .encrypt_to_seekable(&encrypted_path, &key, CHUNK_BYTES, &limits)
        .unwrap();
    let failed_destination = directory.path().join("invalid-chunk-output");
    assert!(checkpoint
        .encrypt_to_seekable(&failed_destination, &key, 3, &limits)
        .is_err());
    assert!(!failed_destination.exists());

    let bad_anchor =
        OlmoeEncryptedCheckpointSource::new(&encrypted_path, "00".repeat(32), key.clone()).unwrap();
    assert!(
        OlmoePackedCheckpoint::open_seekable_encrypted(bad_anchor, cpu_runtime(&limits)).is_err()
    );

    let bad_key = encrypted_source(&encrypted_path, &report, SeekableWeightKey::new([0x53; 32]));
    assert!(OlmoePackedCheckpoint::open_seekable_encrypted(bad_key, cpu_runtime(&limits)).is_err());

    let cancellation = CancellationToken::new();
    cancellation.cancel();
    assert!(
        OlmoePackedCheckpoint::open_seekable_encrypted_with_cancellation(
            encrypted_source(&encrypted_path, &report, key.clone()),
            cpu_runtime(&limits),
            &cancellation,
        )
        .is_err()
    );

    fs::write(encrypted_path.join("unexpected"), b"not declared").unwrap();
    assert!(OlmoePackedCheckpoint::open_seekable_encrypted(
        encrypted_source(&encrypted_path, &report, key.clone()),
        cpu_runtime(&limits)
    )
    .is_err());
    fs::remove_file(encrypted_path.join("unexpected")).unwrap();

    let config_path = encrypted_path.join("config.json");
    let config_bytes = fs::read(&config_path).unwrap();
    let mut valid_but_modified_config = config_bytes.clone();
    valid_but_modified_config.push(b'\n');
    fs::write(&config_path, valid_but_modified_config).unwrap();
    assert!(OlmoePackedCheckpoint::open_seekable_encrypted(
        encrypted_source(&encrypted_path, &report, key.clone()),
        cpu_runtime(&limits)
    )
    .is_err());
    fs::write(&config_path, config_bytes).unwrap();

    let ciphertext = files_below(&encrypted_path)
        .into_iter()
        .find(|path| path.extension().and_then(|value| value.to_str()) == Some("a3se"))
        .unwrap();
    corrupt_last_byte(&ciphertext);
    assert!(OlmoePackedCheckpoint::open_seekable_encrypted(
        encrypted_source(&encrypted_path, &report, key),
        cpu_runtime(&limits)
    )
    .is_err());
}

#[test]
fn encrypt_cli_uses_an_environment_key_and_emits_reopenable_evidence() {
    let directory = tempfile::tempdir().unwrap();
    let packed_path = write_packed(directory.path());
    let encrypted_path = directory.path().join("cli-encrypted");
    let variable = "A3S_MOE_TEST_SEEKABLE_KEY";
    let key_hex = "6a".repeat(32);
    let output = Command::new(env!("CARGO_BIN_EXE_a3s-moe-encrypt"))
        .arg(&packed_path)
        .arg(&encrypted_path)
        .args(["--key-env", variable, "--chunk-mib", "1"])
        .env(variable, &key_hex)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!String::from_utf8_lossy(&output.stdout).contains(&key_hex));
    let report: OlmoePackedEncryptionReport = serde_json::from_slice(&output.stdout).unwrap();
    let source = OlmoeEncryptedCheckpointSource::new(
        &encrypted_path,
        report.manifest_sha256,
        SeekableWeightKey::from_hex(&key_hex).unwrap(),
    )
    .unwrap();
    assert!(OlmoePackedCheckpoint::open_seekable_encrypted(
        source,
        cpu_runtime(&InferenceLimits::default())
    )
    .is_ok());
}

#[test]
fn encrypted_checkpoint_public_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<OlmoeEncryptedCheckpointSource>();
    assert_send_sync::<OlmoePackedEncryptionReport>();
}

fn cpu_runtime(limits: &InferenceLimits) -> EmbeddedRuntime {
    EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone()).unwrap()
}

fn streaming_policy() -> ResidencyPolicy {
    ResidencyPolicy {
        host_cache_bytes: 512,
        max_background_inflight_bytes: 1024 * 1024,
        telemetry: TelemetryMode::Aggregate,
        ..ResidencyPolicy::default()
    }
}

fn encrypted_source(
    root: &Path,
    report: &OlmoePackedEncryptionReport,
    key: SeekableWeightKey,
) -> OlmoeEncryptedCheckpointSource {
    OlmoeEncryptedCheckpointSource::new(root, report.manifest_sha256.clone(), key).unwrap()
}

fn files_below(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else {
                files.push(path);
            }
        }
    }
    files
}

fn corrupt_last_byte(path: &Path) {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let length = file.metadata().unwrap().len();
    file.seek(SeekFrom::Start(length - 1)).unwrap();
    let mut byte = [0_u8; 1];
    file.read_exact(&mut byte).unwrap();
    file.seek(SeekFrom::Start(length - 1)).unwrap();
    file.write_all(&[byte[0] ^ 0xff]).unwrap();
}

use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use a3s_power::inference::{
    encrypt_seekable_weight_collection, EmbeddedRuntime, InferenceLimits,
    SeekableEncryptedWeightManifest, SeekableEncryptedWeightSource, SeekableWeightKey, WeightStore,
    ENCRYPTED_WEIGHT_MANIFEST_FILE,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::checkpoint::enforce_metadata_limit;
use crate::olmoe::{OlmoePackedCheckpoint, OlmoePackedManifest};
use crate::{MoeError, Result};

use super::model::read_metadata;
use super::{
    CONFIG_FILE, DENSE_DIRECTORY, ENCRYPTED_MANIFEST_FILE, EXPERT_DIRECTORY, MANIFEST_FILE,
    TOKENIZER_FILE,
};

pub const OLMOE_ENCRYPTED_PACKED_MANIFEST_SCHEMA: &str = "a3s.moe.olmoe-encrypted-packed.v1";

const MAX_ENCRYPTED_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_POWER_MANIFEST_BYTES: u64 = 4 * 1024 * 1024;
const HASH_BUFFER_BYTES: usize = 1024 * 1024;

/// Integrity metadata for an encrypted packed OLMoE checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OlmoeEncryptedPackedManifest {
    pub schema: String,
    pub packed_weights_sha256: String,
    pub config_sha256: String,
    pub packed_manifest_sha256: String,
    pub tokenizer_sha256: Option<String>,
    pub dense_plaintext_sha256: String,
    pub expert_plaintext_sha256: String,
    pub dense_manifest_sha256: String,
    pub expert_manifest_sha256: String,
    pub chunk_bytes: u32,
}

/// Typed encrypted checkpoint source with an out-of-band trust anchor.
#[derive(Clone)]
pub struct OlmoeEncryptedCheckpointSource {
    root: PathBuf,
    manifest_sha256: String,
    key: SeekableWeightKey,
}

impl OlmoeEncryptedCheckpointSource {
    pub fn new(
        root: impl Into<PathBuf>,
        manifest_sha256: impl Into<String>,
        key: SeekableWeightKey,
    ) -> Result<Self> {
        let manifest_sha256 = manifest_sha256.into();
        validate_sha256(&manifest_sha256, "encrypted checkpoint manifest")?;
        Ok(Self {
            root: root.into(),
            manifest_sha256,
            key,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }
}

impl std::fmt::Debug for OlmoeEncryptedCheckpointSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OlmoeEncryptedCheckpointSource")
            .field("root", &self.root)
            .field("manifest_sha256", &self.manifest_sha256)
            .field("key", &self.key)
            .finish()
    }
}

/// Reproducible bounds and trust anchors from a completed encryption.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OlmoePackedEncryptionReport {
    pub destination: PathBuf,
    pub manifest_sha256: String,
    pub packed_weights_sha256: String,
    pub plaintext_bytes: u64,
    pub ciphertext_bytes: u64,
    pub chunk_bytes: u32,
    pub peak_plaintext_chunk_bytes: u64,
}

impl OlmoePackedCheckpoint {
    /// Encrypts both dense and expert SafeTensors collections without
    /// publishing a partial destination or writing decrypted intermediates.
    pub fn encrypt_to_seekable(
        &self,
        destination: impl AsRef<Path>,
        key: &SeekableWeightKey,
        chunk_bytes: u32,
        limits: &InferenceLimits,
    ) -> Result<OlmoePackedEncryptionReport> {
        let destination = absolute_destination(destination.as_ref())?;
        if destination.exists() {
            return Err(MoeError::InvalidConfig(format!(
                "encrypted checkpoint destination '{}' already exists",
                destination.display()
            )));
        }
        let parent = destination.parent().ok_or_else(|| {
            MoeError::InvalidConfig("encrypted checkpoint destination has no parent".to_string())
        })?;
        std::fs::create_dir_all(parent)?;
        let parent = parent.canonicalize()?;
        let destination = parent.join(destination.file_name().ok_or_else(|| {
            MoeError::InvalidConfig("encrypted checkpoint destination has no name".to_string())
        })?);
        if destination.starts_with(self.root()) || self.root().starts_with(&destination) {
            return Err(MoeError::InvalidConfig(
                "encrypted checkpoint destination must not overlap its source".to_string(),
            ));
        }

        let staging = tempfile::Builder::new()
            .prefix(".a3s-moe-encrypted-")
            .tempdir_in(&parent)?;
        copy_metadata(self, staging.path())?;
        let dense = encrypt_seekable_weight_collection(
            self.root().join(DENSE_DIRECTORY),
            staging.path().join(DENSE_DIRECTORY),
            key,
            chunk_bytes,
            limits,
        )?;
        let experts = encrypt_seekable_weight_collection(
            self.root().join(EXPERT_DIRECTORY),
            staging.path().join(EXPERT_DIRECTORY),
            key,
            chunk_bytes,
            limits,
        )?;
        if dense.plaintext_sha256 != self.manifest().dense_weights_sha256
            || experts.plaintext_sha256 != self.manifest().expert_weights_sha256
        {
            return Err(MoeError::InvalidConfig(
                "encrypted collection identity changed from the validated packed checkpoint"
                    .to_string(),
            ));
        }

        let manifest = OlmoeEncryptedPackedManifest {
            schema: OLMOE_ENCRYPTED_PACKED_MANIFEST_SCHEMA.to_string(),
            packed_weights_sha256: self.manifest().weights_sha256(),
            config_sha256: sha256_file(&self.root().join(CONFIG_FILE))?,
            packed_manifest_sha256: sha256_file(&self.root().join(MANIFEST_FILE))?,
            tokenizer_sha256: self
                .manifest()
                .tokenizer_included
                .then(|| sha256_file(&self.root().join(TOKENIZER_FILE)))
                .transpose()?,
            dense_plaintext_sha256: dense.plaintext_sha256,
            expert_plaintext_sha256: experts.plaintext_sha256,
            dense_manifest_sha256: dense.manifest_sha256,
            expert_manifest_sha256: experts.manifest_sha256,
            chunk_bytes,
        };
        let manifest_bytes = canonical_json(&manifest)?;
        let manifest_sha256 = sha256_bytes(&manifest_bytes);
        std::fs::write(staging.path().join(ENCRYPTED_MANIFEST_FILE), manifest_bytes)?;

        let source = OlmoeEncryptedCheckpointSource::new(
            staging.path(),
            manifest_sha256.clone(),
            key.clone(),
        )?;
        let verified = Self::open_seekable_encrypted(source, self.runtime().clone())?;
        drop(verified);

        let report = OlmoePackedEncryptionReport {
            destination: destination.clone(),
            manifest_sha256,
            packed_weights_sha256: manifest.packed_weights_sha256,
            plaintext_bytes: dense
                .plaintext_bytes
                .checked_add(experts.plaintext_bytes)
                .ok_or_else(|| {
                    MoeError::InvalidConfig("encrypted plaintext byte count overflowed".to_string())
                })?,
            ciphertext_bytes: dense
                .ciphertext_bytes
                .checked_add(experts.ciphertext_bytes)
                .ok_or_else(|| {
                    MoeError::InvalidConfig(
                        "encrypted ciphertext byte count overflowed".to_string(),
                    )
                })?,
            chunk_bytes,
            peak_plaintext_chunk_bytes: dense
                .peak_plaintext_chunk_bytes
                .max(experts.peak_plaintext_chunk_bytes),
        };
        let staged_path = staging.keep();
        if let Err(error) = std::fs::rename(&staged_path, &destination) {
            let _ = std::fs::remove_dir_all(&staged_path);
            return Err(MoeError::Io(error));
        }
        Ok(report)
    }

    pub fn open_seekable_encrypted(
        source: OlmoeEncryptedCheckpointSource,
        runtime: EmbeddedRuntime,
    ) -> Result<Self> {
        Self::open_seekable_encrypted_with_cancellation(source, runtime, &CancellationToken::new())
    }

    /// Authenticates and opens an encrypted packed checkpoint with cooperative
    /// cancellation between bounded chunks.
    pub fn open_seekable_encrypted_with_cancellation(
        source: OlmoeEncryptedCheckpointSource,
        runtime: EmbeddedRuntime,
        cancellation: &CancellationToken,
    ) -> Result<Self> {
        let (root, config, packed_manifest) = read_metadata(&source.root)?;
        let confidential = read_confidential_manifest(&root, &source.manifest_sha256)?;
        validate_confidential_manifest(&confidential, &packed_manifest, &root)?;
        validate_root_inventory(&root, packed_manifest.tokenizer_included)?;
        validate_power_manifest(
            &root.join(DENSE_DIRECTORY),
            &confidential.dense_manifest_sha256,
            &confidential.dense_plaintext_sha256,
            confidential.chunk_bytes,
        )?;
        validate_power_manifest(
            &root.join(EXPERT_DIRECTORY),
            &confidential.expert_manifest_sha256,
            &confidential.expert_plaintext_sha256,
            confidential.chunk_bytes,
        )?;

        let dense_source = SeekableEncryptedWeightSource::new(
            root.join(DENSE_DIRECTORY),
            confidential.dense_manifest_sha256,
            source.key.clone(),
        )?;
        let expert_source = SeekableEncryptedWeightSource::new(
            root.join(EXPERT_DIRECTORY),
            confidential.expert_manifest_sha256,
            source.key,
        )?;
        let dense_store = Arc::new(WeightStore::open_seekable_encrypted_with_cancellation(
            dense_source,
            runtime.limits(),
            cancellation,
        )?);
        let expert_store = Arc::new(WeightStore::open_seekable_encrypted_with_cancellation(
            expert_source,
            runtime.limits(),
            cancellation,
        )?);
        Self::from_stores(
            root,
            config,
            packed_manifest,
            dense_store,
            expert_store,
            runtime,
        )
    }
}

fn copy_metadata(checkpoint: &OlmoePackedCheckpoint, destination: &Path) -> Result<()> {
    for name in [CONFIG_FILE, MANIFEST_FILE] {
        std::fs::copy(checkpoint.root().join(name), destination.join(name))?;
    }
    if checkpoint.manifest().tokenizer_included {
        std::fs::copy(
            checkpoint.root().join(TOKENIZER_FILE),
            destination.join(TOKENIZER_FILE),
        )?;
    }
    Ok(())
}

fn read_confidential_manifest(
    root: &Path,
    expected_sha256: &str,
) -> Result<OlmoeEncryptedPackedManifest> {
    let path = root.join(ENCRYPTED_MANIFEST_FILE);
    enforce_metadata_limit(
        &path,
        MAX_ENCRYPTED_MANIFEST_BYTES,
        "encrypted OLMoE manifest",
    )?;
    let bytes = std::fs::read(path)?;
    let actual = sha256_bytes(&bytes);
    if actual != expected_sha256 {
        return Err(MoeError::InvalidConfig(format!(
            "encrypted checkpoint manifest SHA-256 must be '{expected_sha256}', found '{actual}'"
        )));
    }
    Ok(serde_json::from_slice(&bytes)?)
}

fn validate_confidential_manifest(
    confidential: &OlmoeEncryptedPackedManifest,
    packed: &OlmoePackedManifest,
    root: &Path,
) -> Result<()> {
    if confidential.schema != OLMOE_ENCRYPTED_PACKED_MANIFEST_SCHEMA {
        return Err(MoeError::InvalidConfig(format!(
            "encrypted checkpoint schema '{}' is unsupported",
            confidential.schema
        )));
    }
    for (label, digest) in [
        ("packed weights", &confidential.packed_weights_sha256),
        ("config", &confidential.config_sha256),
        ("packed manifest", &confidential.packed_manifest_sha256),
        ("dense plaintext", &confidential.dense_plaintext_sha256),
        ("expert plaintext", &confidential.expert_plaintext_sha256),
        ("dense manifest", &confidential.dense_manifest_sha256),
        ("expert manifest", &confidential.expert_manifest_sha256),
    ] {
        validate_sha256(digest, label)?;
    }
    if let Some(tokenizer) = confidential.tokenizer_sha256.as_deref() {
        validate_sha256(tokenizer, "tokenizer")?;
    }
    let tokenizer_matches = match confidential.tokenizer_sha256.as_deref() {
        Some(expected) if packed.tokenizer_included => {
            sha256_file(&root.join(TOKENIZER_FILE))? == expected
        }
        None if !packed.tokenizer_included => true,
        _ => false,
    };
    if confidential.packed_weights_sha256 != packed.weights_sha256()
        || confidential.config_sha256 != sha256_file(&root.join(CONFIG_FILE))?
        || confidential.packed_manifest_sha256 != sha256_file(&root.join(MANIFEST_FILE))?
        || confidential.dense_plaintext_sha256 != packed.dense_weights_sha256
        || confidential.expert_plaintext_sha256 != packed.expert_weights_sha256
        || !tokenizer_matches
    {
        return Err(MoeError::InvalidConfig(
            "encrypted checkpoint metadata does not match its trust manifest".to_string(),
        ));
    }
    Ok(())
}

fn validate_root_inventory(root: &Path, tokenizer_included: bool) -> Result<()> {
    let mut expected = BTreeSet::from([
        CONFIG_FILE.to_string(),
        MANIFEST_FILE.to_string(),
        ENCRYPTED_MANIFEST_FILE.to_string(),
        DENSE_DIRECTORY.to_string(),
        EXPERT_DIRECTORY.to_string(),
    ]);
    if tokenizer_included {
        expected.insert(TOKENIZER_FILE.to_string());
    }
    let mut actual = BTreeSet::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            return Err(MoeError::InvalidConfig(
                "encrypted checkpoints must not contain symbolic links".to_string(),
            ));
        }
        let name = entry.file_name().into_string().map_err(|_| {
            MoeError::InvalidConfig("encrypted checkpoint names must be valid UTF-8".to_string())
        })?;
        let is_weight_directory = name == DENSE_DIRECTORY || name == EXPERT_DIRECTORY;
        if (is_weight_directory && !file_type.is_dir())
            || (!is_weight_directory && !file_type.is_file())
        {
            return Err(MoeError::InvalidConfig(format!(
                "encrypted checkpoint entry '{name}' has the wrong file type"
            )));
        }
        actual.insert(name);
    }
    if actual != expected {
        return Err(MoeError::InvalidConfig(
            "encrypted checkpoint root inventory does not match its schema".to_string(),
        ));
    }
    Ok(())
}

fn validate_power_manifest(
    root: &Path,
    expected_sha256: &str,
    expected_plaintext_sha256: &str,
    expected_chunk_bytes: u32,
) -> Result<()> {
    let path = root.join(ENCRYPTED_WEIGHT_MANIFEST_FILE);
    enforce_metadata_limit(
        &path,
        MAX_POWER_MANIFEST_BYTES,
        "Power encrypted weight manifest",
    )?;
    let bytes = std::fs::read(path)?;
    let actual_sha256 = sha256_bytes(&bytes);
    if actual_sha256 != expected_sha256 {
        return Err(MoeError::InvalidConfig(
            "Power encrypted weight manifest does not match the OLMoE trust manifest".to_string(),
        ));
    }
    let manifest: SeekableEncryptedWeightManifest = serde_json::from_slice(&bytes)?;
    if manifest.plaintext_sha256 != expected_plaintext_sha256
        || manifest.chunk_bytes != expected_chunk_bytes
    {
        return Err(MoeError::InvalidConfig(
            "Power encrypted weight properties do not match the OLMoE trust manifest".to_string(),
        ));
    }
    Ok(())
}

fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; HASH_BUFFER_BYTES];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(MoeError::InvalidConfig(format!(
            "{label} SHA-256 must contain 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn absolute_destination(destination: &Path) -> Result<PathBuf> {
    if destination.as_os_str().is_empty() {
        return Err(MoeError::InvalidConfig(
            "encrypted checkpoint destination must not be empty".to_string(),
        ));
    }
    if destination.is_absolute() {
        Ok(destination.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(destination))
    }
}

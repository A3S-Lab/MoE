use std::collections::{BTreeSet, HashMap};
use std::path::{Component, Path, PathBuf};

use candle_core::{Device, Tensor};
use serde::Deserialize;

use crate::{MoeError, Result};

pub(crate) const CONFIG_FILE: &str = "config.json";
pub(crate) const INDEX_FILE: &str = "model.safetensors.index.json";
pub(crate) const TOKENIZER_FILE: &str = "tokenizer.json";
pub(crate) const MAX_CONFIG_BYTES: u64 = 1_048_576;
pub(crate) const MAX_INDEX_BYTES: u64 = 16 * 1_048_576;
pub(crate) const MAX_TOKENIZER_BYTES: u64 = 64 * 1_048_576;

#[derive(Debug, Deserialize)]
struct SafeTensorIndex {
    weight_map: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub(crate) struct ShardedSafeTensors {
    root: PathBuf,
    shards: Vec<PathBuf>,
    weight_map: HashMap<String, String>,
}

impl ShardedSafeTensors {
    pub(crate) fn open(root: &Path, required_names: Vec<String>) -> Result<Self> {
        if required_names.is_empty() {
            return Err(MoeError::InvalidConfig(
                "required tensor inventory must not be empty".to_string(),
            ));
        }
        let index_path = root.join(INDEX_FILE);
        enforce_metadata_limit(&index_path, MAX_INDEX_BYTES, "SafeTensor index")?;
        let index: SafeTensorIndex = serde_json::from_slice(&std::fs::read(&index_path)?)?;
        if index.weight_map.is_empty() {
            return Err(MoeError::InvalidConfig(
                "SafeTensor index weight_map must not be empty".to_string(),
            ));
        }

        let required = required_names.into_iter().collect::<BTreeSet<_>>();
        if required.len() != index.weight_map.len() {
            return Err(inventory_error(&index.weight_map, &required));
        }
        if let Some(missing) = required
            .iter()
            .find(|name| !index.weight_map.contains_key(*name))
        {
            return Err(MoeError::InvalidConfig(format!(
                "SafeTensor index is missing required tensor '{missing}'"
            )));
        }

        let mut shard_names = BTreeSet::new();
        for shard in index.weight_map.values() {
            validate_shard_name(shard)?;
            shard_names.insert(shard.clone());
        }
        let mut shards = Vec::with_capacity(shard_names.len());
        for shard in shard_names {
            let path = root.join(&shard);
            let metadata = path.symlink_metadata().map_err(|error| {
                MoeError::InvalidConfig(format!(
                    "failed to inspect SafeTensor shard '{}': {error}",
                    path.display()
                ))
            })?;
            if !metadata.is_file() || metadata.file_type().is_symlink() {
                return Err(MoeError::InvalidConfig(format!(
                    "SafeTensor shard '{}' must be a regular non-symlink file",
                    path.display()
                )));
            }
            shards.push(path);
        }
        Ok(Self {
            root: root.to_path_buf(),
            shards,
            weight_map: index.weight_map,
        })
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn shards(&self) -> &[PathBuf] {
        &self.shards
    }

    pub(crate) fn load_cpu(&self) -> Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::with_capacity(self.weight_map.len());
        for shard in &self.shards {
            let shard_name = shard
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| {
                    MoeError::InvalidConfig(format!(
                        "SafeTensor shard '{}' has a non-UTF-8 filename",
                        shard.display()
                    ))
                })?;
            for (name, tensor) in candle_core::safetensors::load(shard, &Device::Cpu)? {
                if self.weight_map.get(&name).map(String::as_str) != Some(shard_name) {
                    return Err(MoeError::InvalidConfig(format!(
                        "tensor '{name}' is not stored in its index-declared shard"
                    )));
                }
                if tensors.insert(name.clone(), tensor).is_some() {
                    return Err(MoeError::InvalidConfig(format!(
                        "tensor '{name}' occurs in more than one SafeTensor shard"
                    )));
                }
            }
        }
        if tensors.len() != self.weight_map.len() {
            return Err(MoeError::InvalidConfig(format!(
                "loaded {} tensors but the index declares {}",
                tensors.len(),
                self.weight_map.len()
            )));
        }
        Ok(tensors)
    }
}

fn inventory_error(weight_map: &HashMap<String, String>, required: &BTreeSet<String>) -> MoeError {
    let unexpected = weight_map
        .keys()
        .filter(|name| !required.contains(*name))
        .take(5)
        .cloned()
        .collect::<Vec<_>>();
    MoeError::InvalidConfig(format!(
        "SafeTensor index contains {} tensors, expected {}; unexpected examples: {unexpected:?}",
        weight_map.len(),
        required.len()
    ))
}

pub(crate) fn enforce_metadata_limit(path: &Path, limit: u64, label: &str) -> Result<()> {
    let metadata = path.symlink_metadata()?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > limit {
        return Err(MoeError::InvalidConfig(format!(
            "{label} '{}' must be a regular non-symlink file no larger than {limit} bytes",
            path.display()
        )));
    }
    Ok(())
}

pub(crate) fn validate_shard_name(name: &str) -> Result<()> {
    let path = Path::new(name);
    let mut components = path.components();
    let single_file = matches!(components.next(), Some(Component::Normal(_)))
        && components.next().is_none()
        && path.extension().and_then(|extension| extension.to_str()) == Some("safetensors");
    if !single_file {
        return Err(MoeError::InvalidConfig(format!(
            "SafeTensor shard name '{name}' must be one relative .safetensors filename"
        )));
    }
    Ok(())
}

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
        Ok(Self::open_one_of(root, vec![required_names])?.0)
    }

    pub(crate) fn open_one_of(
        root: &Path,
        required_inventories: Vec<Vec<String>>,
    ) -> Result<(Self, usize)> {
        if required_inventories.is_empty()
            || required_inventories
                .iter()
                .any(|inventory| inventory.is_empty())
        {
            return Err(MoeError::InvalidConfig(
                "required tensor inventories must not be empty".to_string(),
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

        let actual = index.weight_map.keys().cloned().collect::<BTreeSet<_>>();
        let required = required_inventories
            .into_iter()
            .map(|names| names.into_iter().collect::<BTreeSet<_>>())
            .collect::<Vec<_>>();
        let inventory = required
            .iter()
            .position(|candidate| candidate == &actual)
            .ok_or_else(|| inventory_error(&actual, &required))?;

        Self::from_index(root, index).map(|weights| (weights, inventory))
    }

    /// Opens a checkpoint whose required model tensors coexist with explicitly
    /// ignored tensor families, such as vision and MTP weights in a text-only
    /// model implementation. Unknown extras still fail closed.
    pub(crate) fn open_with_ignored_prefixes(
        root: &Path,
        required_names: Vec<String>,
        ignored_prefixes: &[&str],
    ) -> Result<Self> {
        if required_names.is_empty()
            || ignored_prefixes.is_empty()
            || ignored_prefixes.iter().any(|prefix| prefix.is_empty())
        {
            return Err(MoeError::InvalidConfig(
                "required tensors and ignored prefixes must not be empty".to_string(),
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
        let actual = index.weight_map.keys().cloned().collect::<BTreeSet<_>>();
        let required = required_names.into_iter().collect::<BTreeSet<_>>();
        let missing = required
            .difference(&actual)
            .take(5)
            .cloned()
            .collect::<Vec<_>>();
        let unexpected = actual
            .difference(&required)
            .filter(|name| {
                !ignored_prefixes
                    .iter()
                    .any(|prefix| name.starts_with(prefix))
            })
            .take(5)
            .cloned()
            .collect::<Vec<_>>();
        if !missing.is_empty() || !unexpected.is_empty() {
            return Err(MoeError::InvalidConfig(format!(
                "SafeTensor index is incompatible with the text-only inventory; missing examples: {missing:?}; unexpected examples: {unexpected:?}"
            )));
        }
        Self::from_index(root, index)
    }

    fn from_index(root: &Path, index: SafeTensorIndex) -> Result<Self> {
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

    pub(crate) fn load_cpu_selected(&self, names: &[String]) -> Result<HashMap<String, Tensor>> {
        let selected = names.iter().cloned().collect::<BTreeSet<_>>();
        if selected.len() != names.len()
            || selected
                .iter()
                .any(|name| !self.weight_map.contains_key(name))
        {
            return Err(MoeError::InvalidConfig(
                "selected SafeTensor names must be unique indexed tensors".to_string(),
            ));
        }
        let mut tensors = HashMap::with_capacity(selected.len());
        let mut loaded_names = BTreeSet::new();
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
                if !loaded_names.insert(name.clone()) {
                    return Err(MoeError::InvalidConfig(format!(
                        "tensor '{name}' occurs in more than one SafeTensor shard"
                    )));
                }
                if selected.contains(&name) {
                    tensors.insert(name, tensor);
                }
            }
        }
        if loaded_names.len() != self.weight_map.len() || tensors.len() != selected.len() {
            return Err(MoeError::InvalidConfig(format!(
                "loaded {} indexed tensors and {} selected tensors, expected {} and {}",
                loaded_names.len(),
                tensors.len(),
                self.weight_map.len(),
                selected.len()
            )));
        }
        Ok(tensors)
    }
}

fn inventory_error(actual: &BTreeSet<String>, required: &[BTreeSet<String>]) -> MoeError {
    let unexpected = actual
        .iter()
        .filter(|name| required.iter().all(|inventory| !inventory.contains(*name)))
        .take(5)
        .cloned()
        .collect::<Vec<_>>();
    let expected_counts = required.iter().map(BTreeSet::len).collect::<Vec<_>>();
    MoeError::InvalidConfig(format!(
        "SafeTensor index contains {} tensors, expected one of {expected_counts:?}; unexpected examples: {unexpected:?}",
        actual.len(),
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

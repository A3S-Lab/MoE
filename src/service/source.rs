use std::path::{Path, PathBuf};

use a3s_power::inference::EmbeddedRuntime;

use crate::olmoe::{OlmoeEncryptedCheckpointSource, OlmoePackedCheckpoint};
use crate::Result;

#[derive(Clone, Debug)]
pub(super) enum CheckpointSource {
    Plain(PathBuf),
    Encrypted(OlmoeEncryptedCheckpointSource),
}

impl CheckpointSource {
    pub(super) fn root(&self) -> &Path {
        match self {
            Self::Plain(path) => path,
            Self::Encrypted(source) => source.root(),
        }
    }

    pub(super) fn trust_anchor(&self) -> Option<&str> {
        match self {
            Self::Plain(_) => None,
            Self::Encrypted(source) => Some(source.manifest_sha256()),
        }
    }

    pub(super) fn open(self, runtime: EmbeddedRuntime) -> Result<OlmoePackedCheckpoint> {
        match self {
            Self::Plain(path) => OlmoePackedCheckpoint::open(path, runtime),
            Self::Encrypted(source) => {
                OlmoePackedCheckpoint::open_seekable_encrypted(source, runtime)
            }
        }
    }
}

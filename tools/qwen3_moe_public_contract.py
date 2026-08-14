#!/usr/bin/env python3
"""Pinned provenance contract for the public Qwen3-MoE acceptance model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3-30B-A3B-Base"
MODEL_REVISION = "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"
TRANSFORMERS_REVISION = "918dbf131d0df5b46e3f6e1d96174d62aa4d16d6"
TRANSFORMERS_SOURCE_SHA256 = (
    "56d820671d810b68f31056605cec0c674994c8f962370194225911ac6a71a365"
)
TRANSFORMERS_DTYPE = "float32"
READ_CHUNK_BYTES = 8 * 1024 * 1024

PINNED_CHECKPOINT_FILES = {
    "config.json": (
        964,
        "7e4142150e976c6b4796adf88fce0a2a23e581ed63bcede7e1d69a645e73b362",
    ),
    "generation_config.json": (
        138,
        "8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8",
    ),
    "merges.txt": (
        1_671_853,
        "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    ),
    "model.safetensors.index.json": (
        1_699_758,
        "df0d481ec595c55a0ba58426d517390c6214a566ec4ff1c8fc4bbce9f57b3c24",
    ),
    "model-00001-of-00016.safetensors": (
        3_999_417_504,
        "7fe481b0c3796bee8d4fa63638f4d6d3d0b1c66339ef63a1aa03a4d784c53e73",
    ),
    "model-00002-of-00016.safetensors": (
        3_999_974_192,
        "3b1e762dd99476a4b7c2d7b331432fe3e32e11e0dd7c90c2822bb84df4881ae2",
    ),
    "model-00003-of-00016.safetensors": (
        3_997_360_832,
        "c66ff62c6aeb11e085e2215c64cfc5031b50fbcadaead02d0687054b9bf523ff",
    ),
    "model-00004-of-00016.safetensors": (
        3_999_975_056,
        "534c0b5e5a215d95bbd77f9a034d5d74c3b4258f4d5d3918e56ed63dec1eec49",
    ),
    "model-00005-of-00016.safetensors": (
        3_999_975_400,
        "8082532f02d2473f51828f4ab5377b2747d5d77b934ba8d93a25d72d4d82078c",
    ),
    "model-00006-of-00016.safetensors": (
        3_999_975_400,
        "9fab34042ea6ee3348994dbb9b582773bfd51c54defca758beee4521cf54f0bd",
    ),
    "model-00007-of-00016.safetensors": (
        3_999_975_472,
        "02f0a1c1e62143483d1d0655afee52e22c03d25b6f24c8f6ed3b9d6cf47fbc1b",
    ),
    "model-00008-of-00016.safetensors": (
        3_997_362_064,
        "b2ccbadc878ec61dc09cec19ae0c4d3ff8f25e9c0e659599a7d26eaa411a8a4d",
    ),
    "model-00009-of-00016.safetensors": (
        3_999_975_408,
        "0b9dc84d14919b4c65fa56e8b6ffd67c8cf431861673034275c26279a8dd525b",
    ),
    "model-00010-of-00016.safetensors": (
        3_999_975_400,
        "70cb610487e592d19eea29eadc8785a9e4e4b88ee65b3dbd7ac50d0c54e61a4f",
    ),
    "model-00011-of-00016.safetensors": (
        3_999_975_408,
        "662f6a0cb5607d3be8d52fac7c1f541769fd8426867cde36c588cbcef9e8bca9",
    ),
    "model-00012-of-00016.safetensors": (
        3_987_400_496,
        "c8ddc8ffa628697a3618f953a99cfdc51a34da97d0194ccf1ad9a5ba6022178b",
    ),
    "model-00013-of-00016.safetensors": (
        3_997_353_632,
        "2ea0a94f86a4eba612753a3159d9d06e307556c95873bde88294cb48f8c14f75",
    ),
    "model-00014-of-00016.safetensors": (
        3_999_975_400,
        "05d8098a9924ca0990db663b934550cb07a6287a6590b13e93b14cf139edc268",
    ),
    "model-00015-of-00016.safetensors": (
        3_999_975_400,
        "002936e6733de8b73ef36c815013cdd53f2c969ffe19ad62cc47ecab69909cff",
    ),
    "model-00016-of-00016.safetensors": (
        1_087_928_584,
        "90a2c863affd5c0a6e4bf14ae7e4ac7ff388b6328c3d49aed6ea18c3139c6536",
    ),
    "tokenizer.json": (
        7_031_645,
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    ),
    "tokenizer_config.json": (
        9_678,
        "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5",
    ),
    "vocab.json": (
        2_776_833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
}
PINNED_WEIGHT_SHARDS = {
    name for name in PINNED_CHECKPOINT_FILES if name.endswith(".safetensors")
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(READ_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def regular_file(root: Path, name: str) -> Path:
    if Path(name).name != name:
        raise RuntimeError(f"checkpoint inventory contains unsafe filename {name!r}")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"checkpoint file must be regular and non-symlink: {path}")
    return path


def checkpoint_inventory(root: Path) -> list[dict[str, Any]]:
    index_path = regular_file(root, "model.safetensors.index.json")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("model.safetensors.index.json has no weight_map")
    if any(not isinstance(name, str) for name in weight_map.values()):
        raise RuntimeError("checkpoint index contains a non-string shard name")
    declared_shards = set(weight_map.values())
    if declared_shards != PINNED_WEIGHT_SHARDS:
        raise RuntimeError(
            "checkpoint index shard inventory does not match the pinned revision"
        )

    inventory = []
    for name, (expected_bytes, expected_sha256) in sorted(
        PINNED_CHECKPOINT_FILES.items()
    ):
        path = regular_file(root, name)
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"checkpoint file {name!r} has {actual_bytes} bytes, "
                f"expected {expected_bytes} for revision {MODEL_REVISION}"
            )
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"checkpoint file {name!r} has SHA-256 {actual_sha256}, "
                f"expected {expected_sha256} for revision {MODEL_REVISION}"
            )
        inventory.append(
            {"name": name, "bytes": expected_bytes, "sha256": expected_sha256}
        )
    return inventory

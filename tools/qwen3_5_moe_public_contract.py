#!/usr/bin/env python3
"""Pinned provenance contract for the public Qwen3.6-35B-A3B model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
OUTER_MODEL_TYPE = "qwen3_5_moe"
TEXT_MODEL_TYPE = "qwen3_5_moe_text"
TRANSFORMERS_REVISION = "918dbf131d0df5b46e3f6e1d96174d62aa4d16d6"
TRANSFORMERS_SOURCE_SHA256 = (
    "16ef7b0dc6e26eae26a6ffd0ad11d93f85424f90ffb85fbcc9eecc0759cc930d"
)
TRANSFORMERS_DTYPE = "float32"
INDEX_TENSOR_COUNT = 1_045
TEXT_TENSOR_COUNT = 693
VISION_TENSOR_COUNT = 333
MTP_TENSOR_COUNT = 19
READ_CHUNK_BYTES = 8 * 1024 * 1024

PINNED_CHECKPOINT_FILES = {
    "chat_template.jinja": (
        7_764,
        "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259",
    ),
    "config.json": (
        3_686,
        "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
    ),
    "generation_config.json": (
        202,
        "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
    ),
    "merges.txt": (
        3_353_259,
        "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    ),
    "model.safetensors.index.json": (
        98_383,
        "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
    ),
    "model-00001-of-00026.safetensors": (
        3_996_199_712,
        "adee7bcb930aed22e0677e58d4873b48dadb1ed8001cb5c6a0487286eadb3478",
    ),
    "model-00002-of-00026.safetensors": (
        1_284_907_696,
        "88f2dfd2b9e73e4b70be533dbf61bcfa3c9a0003758900fcbc9d9b96f5751d4b",
    ),
    "model-00003-of-00026.safetensors": (
        3_357_898_360,
        "8f7d72178d3f4431864978e5bcfa4c6cb1c204bc00590644d90bb19d6d522eeb",
    ),
    "model-00004-of-00026.safetensors": (
        3_370_808_712,
        "12d7db38689ba3c8af74b23ef8523eca41e0cd95db870583d0663a3ee8a6bd60",
    ),
    "model-00005-of-00026.safetensors": (
        3_357_898_360,
        "a836047305d0f7a7b50f0815d09d5c03ec03d59ec2c763fcdc4bf7e9936bf902",
    ),
    "model-00006-of-00026.safetensors": (
        3_959_424_904,
        "c9080d718e9c5f9e337443225aa417d4c24d00ae7995d76ee3f1cc296b557d15",
    ),
    "model-00007-of-00026.safetensors": (
        1_096_788_232,
        "e8c05e23131b1dd45a455ec38cfac7db14667358268623c3938d00cf3e959a68",
    ),
    "model-00008-of-00026.safetensors": (
        3_946_842_008,
        "4b6a6d495053089f4a80e7cbc82e848fba44e2c0c60122233d8fdff79fa7b296",
    ),
    "model-00009-of-00026.safetensors": (
        1_096_460_848,
        "a31a954bb72d1c714e751bf0aabf2ff533f5a509693ebf7dd22ad6e90be46f67",
    ),
    "model-00010-of-00026.safetensors": (
        3_946_841_992,
        "246560e66570fe746653b8443e245dc334c9b8b831ea43d2d9f1b7d98623994e",
    ),
    "model-00011-of-00026.safetensors": (
        1_096_460_752,
        "7180392817fe3ecb3a27a1da43b7ff22c1a94806bac49975f9f122c3126df675",
    ),
    "model-00012-of-00026.safetensors": (
        3_409_971_080,
        "043fb525f6625c2f2acb75e65a9959ee3fa7b6e3fdd2034b5cfe1859b01d3cfb",
    ),
    "model-00013-of-00026.safetensors": (
        1_633_331_664,
        "33a20fb20a21379bf43c84a43105f9c0cc35bd50d740b1c302dcbe4b700f5425",
    ),
    "model-00014-of-00026.safetensors": (
        3_422_553_872,
        "be823e33c5cb6120ad3769d081f34a2449dc2358041fca7c29d636c1ba19130d",
    ),
    "model-00015-of-00026.safetensors": (
        1_633_659_224,
        "a89d547c6f9d0b535ee5ea2f2478f163089539f3f0dd330cb23d278a19d76123",
    ),
    "model-00016-of-00026.safetensors": (
        3_946_842_136,
        "69fc3ae0316482288afdcdd0b9eb7d626703ae26f7567e89aa3fc8d1ffd4ff5b",
    ),
    "model-00017-of-00026.safetensors": (
        1_096_460_608,
        "e356e3943cf3852b76bb8992e674f3256013e27d54b78e8250514151cdc29637",
    ),
    "model-00018-of-00026.safetensors": (
        3_946_841_992,
        "9e5e63fd1cc7d6848330c1fa363dfcb661bbc2ac87e672d0e28b71c9cb7f3c7f",
    ),
    "model-00019-of-00026.safetensors": (
        1_096_460_808,
        "708644ad34f1de727bf484f396944d8ec628645d52c183e9a992e65671685e21",
    ),
    "model-00020-of-00026.safetensors": (
        3_409_971_072,
        "ca083a1d1aa64f8e8a785998f543a43374f13436dc85d396eee4e72c7a84e1ae",
    ),
    "model-00021-of-00026.safetensors": (
        1_633_331_744,
        "ada4ae48f3d48fe01b4c53f2f82bce25e798a9631fd33959c881156fef2ccbce",
    ),
    "model-00022-of-00026.safetensors": (
        3_370_808_752,
        "def207fb42d7db31efb512755557763c23233c6e4d4c433027cb5102a7bce2f7",
    ),
    "model-00023-of-00026.safetensors": (
        3_357_898_392,
        "864d52ca7768a36f514069222e8de8626264ae124097ba8fcce5b5da2c6e2ed7",
    ),
    "model-00024-of-00026.safetensors": (
        3_370_808_752,
        "391acd27420cdce5935ff18152423c70620d19dac3c39a5ef1a81d369f82d737",
    ),
    "model-00025-of-00026.safetensors": (
        3_832_888_256,
        "778e7f76602f05042b69ba7f3ec91f1fdffef390540b16074041c258fb81d154",
    ),
    "model-00026-of-00026.safetensors": (
        2_231_416_848,
        "1a97404220077ed3d4182e10385b152004cab608377f50cec9f54a6b8d28b613",
    ),
    "tokenizer.json": (
        12_807_982,
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    ),
    "tokenizer_config.json": (
        16_718,
        "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b",
    ),
    "vocab.json": (
        6_722_759,
        "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
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


def validate_index(index: Any) -> dict[str, str]:
    if not isinstance(index, dict):
        raise RuntimeError("model.safetensors.index.json must contain an object")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("model.safetensors.index.json has no weight_map")
    if any(not isinstance(name, str) for name in weight_map.values()):
        raise RuntimeError("checkpoint index contains a non-string shard name")
    if len(weight_map) != INDEX_TENSOR_COUNT:
        raise RuntimeError(
            f"checkpoint index has {len(weight_map)} tensors, expected "
            f"{INDEX_TENSOR_COUNT}"
        )
    text_names = {
        name
        for name in weight_map
        if name.startswith("model.language_model.") or name == "lm_head.weight"
    }
    vision_names = {name for name in weight_map if name.startswith("model.visual.")}
    mtp_names = {name for name in weight_map if name.startswith("mtp.")}
    if (
        len(text_names) != TEXT_TENSOR_COUNT
        or len(vision_names) != VISION_TENSOR_COUNT
        or len(mtp_names) != MTP_TENSOR_COUNT
        or len(text_names | vision_names | mtp_names) != len(weight_map)
    ):
        raise RuntimeError("checkpoint index family tensor counts do not match the pin")
    declared_shards = set(weight_map.values())
    if declared_shards != PINNED_WEIGHT_SHARDS:
        raise RuntimeError(
            "checkpoint index shard inventory does not match the pinned revision"
        )
    return weight_map


def checkpoint_inventory(root: Path) -> list[dict[str, Any]]:
    index_path = regular_file(root, "model.safetensors.index.json")
    validate_index(json.loads(index_path.read_text(encoding="utf-8")))

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

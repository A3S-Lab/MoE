#!/usr/bin/env python3
"""Regression tests for the public Qwen3.6-35B-A3B provenance contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import qwen3_5_moe_public_contract as contract


def descriptor(payload: bytes) -> tuple[int, str]:
    return len(payload), hashlib.sha256(payload).hexdigest()


class CheckpointInventoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.weight_map = {
            "model.language_model.embed_tokens.weight": "weights.safetensors",
            "model.visual.patch_embed.proj.weight": "weights.safetensors",
            "mtp.layers.0.input_layernorm.weight": "weights.safetensors",
        }
        self.files = {
            "config.json": b"{}\n",
            "model.safetensors.index.json": json.dumps(
                {"weight_map": self.weight_map}
            ).encode(),
            "tokenizer.json": b"{}\n",
            "weights.safetensors": b"weight bytes",
        }
        for name, payload in self.files.items():
            (self.root / name).write_bytes(payload)
        self.pinned = {
            name: descriptor(payload) for name, payload in self.files.items()
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def inventory(self) -> list[dict[str, object]]:
        with (
            patch.object(contract, "PINNED_CHECKPOINT_FILES", self.pinned),
            patch.object(contract, "PINNED_WEIGHT_SHARDS", {"weights.safetensors"}),
            patch.object(contract, "INDEX_TENSOR_COUNT", 3),
            patch.object(contract, "TEXT_TENSOR_COUNT", 1),
            patch.object(contract, "VISION_TENSOR_COUNT", 1),
            patch.object(contract, "MTP_TENSOR_COUNT", 1),
        ):
            return contract.checkpoint_inventory(self.root)

    def test_accepts_only_the_exact_pinned_inventory(self) -> None:
        inventory = self.inventory()
        self.assertEqual([entry["name"] for entry in inventory], sorted(self.files))

        (self.root / "config.json").write_bytes(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "bytes|SHA-256"):
            self.inventory()

    def test_rejects_an_unknown_tensor_family(self) -> None:
        del self.weight_map["mtp.layers.0.input_layernorm.weight"]
        self.weight_map["unknown.weight"] = "weights.safetensors"
        index = json.dumps({"weight_map": self.weight_map}).encode()
        (self.root / "model.safetensors.index.json").write_bytes(index)
        self.pinned["model.safetensors.index.json"] = descriptor(index)
        with self.assertRaisesRegex(RuntimeError, "family tensor counts"):
            self.inventory()

    def test_rejects_a_different_index_shard_set(self) -> None:
        self.weight_map["mtp.layers.0.input_layernorm.weight"] = "other.safetensors"
        index = json.dumps({"weight_map": self.weight_map}).encode()
        (self.root / "model.safetensors.index.json").write_bytes(index)
        self.pinned["model.safetensors.index.json"] = descriptor(index)
        with self.assertRaisesRegex(RuntimeError, "shard inventory"):
            self.inventory()


class PublicPinTests(unittest.TestCase):
    def test_pins_the_exact_official_text_checkpoint_shape(self) -> None:
        self.assertEqual(contract.MODEL_ID, "Qwen/Qwen3.6-35B-A3B")
        self.assertEqual(len(contract.MODEL_REVISION), 40)
        self.assertEqual(contract.OUTER_MODEL_TYPE, "qwen3_5_moe")
        self.assertEqual(contract.TEXT_MODEL_TYPE, "qwen3_5_moe_text")
        self.assertEqual(contract.INDEX_TENSOR_COUNT, 1_045)
        self.assertEqual(contract.TEXT_TENSOR_COUNT, 693)
        self.assertEqual(contract.VISION_TENSOR_COUNT, 333)
        self.assertEqual(contract.MTP_TENSOR_COUNT, 19)
        self.assertEqual(len(contract.PINNED_WEIGHT_SHARDS), 26)
        self.assertEqual(
            sum(
                size
                for name, (size, _) in contract.PINNED_CHECKPOINT_FILES.items()
                if name.endswith(".safetensors")
            ),
            71_903_776_776,
        )


if __name__ == "__main__":
    unittest.main()

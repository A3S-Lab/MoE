#!/usr/bin/env python3
"""Regression tests for the public-oracle provenance boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import generate_public_oracle as oracle


def descriptor(payload: bytes) -> tuple[int, str]:
    return len(payload), hashlib.sha256(payload).hexdigest()


class CheckpointInventoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.files = {
            "config.json": b"{}\n",
            "model.safetensors.index.json": json.dumps(
                {"weight_map": {"tensor": "weights.safetensors"}}
            ).encode(),
            "tokenizer.json": b"{}\n",
            "tokenizer_config.json": b"{}\n",
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
            patch.object(oracle, "PINNED_CHECKPOINT_FILES", self.pinned),
            patch.object(oracle, "PINNED_WEIGHT_SHARDS", {"weights.safetensors"}),
        ):
            return oracle.checkpoint_inventory(self.root)

    def test_accepts_only_the_exact_pinned_inventory(self) -> None:
        inventory = self.inventory()
        self.assertEqual(
            [entry["name"] for entry in inventory], sorted(self.files)
        )

        (self.root / "config.json").write_bytes(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "bytes|SHA-256"):
            self.inventory()

    def test_rejects_a_different_index_shard_set(self) -> None:
        index = {"weight_map": {"tensor": "replacement.safetensors"}}
        (self.root / "model.safetensors.index.json").write_text(json.dumps(index))
        self.pinned["model.safetensors.index.json"] = descriptor(
            (self.root / "model.safetensors.index.json").read_bytes()
        )

        with self.assertRaisesRegex(RuntimeError, "shard inventory"):
            self.inventory()


if __name__ == "__main__":
    unittest.main()

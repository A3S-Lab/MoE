#!/usr/bin/env python3
"""Regression tests for the public Qwen3-MoE oracle boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import generate_qwen3_moe_public_oracle as oracle
import qwen3_moe_public_contract as contract


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
            patch.object(contract, "PINNED_CHECKPOINT_FILES", self.pinned),
            patch.object(contract, "PINNED_WEIGHT_SHARDS", {"weights.safetensors"}),
        ):
            return contract.checkpoint_inventory(self.root)

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


class SparseScheduleTests(unittest.TestCase):
    def test_matches_the_qwen_layer_schedule(self) -> None:
        config = SimpleNamespace(
            mlp_only_layers=[3],
            num_experts=8,
            decoder_sparse_step=2,
        )
        self.assertFalse(oracle.is_sparse_layer(config, 0))
        self.assertTrue(oracle.is_sparse_layer(config, 1))
        self.assertFalse(oracle.is_sparse_layer(config, 3))


if __name__ == "__main__":
    unittest.main()

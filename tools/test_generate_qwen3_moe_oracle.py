#!/usr/bin/env python3
"""Regression test for the checked Qwen3-MoE sparse fixture."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import generate_qwen3_moe_oracle as generator


class Qwen3MoeOracleTests(unittest.TestCase):
    def test_checked_fixture_matches_the_dependency_free_generator(self) -> None:
        fixture = Path(__file__).parents[1] / "tests/fixtures/qwen3_moe_sparse_oracle.json"
        self.assertEqual(json.loads(fixture.read_text(encoding="utf-8")), generator.generate())


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Regression test for the checked full Qwen3-MoE fixture."""

import json
import unittest
from pathlib import Path

import generate_qwen3_moe_full_oracle as generator


class Qwen3MoeFullOracleTests(unittest.TestCase):
    def test_checked_fixture_matches_generator(self) -> None:
        fixture = (
            Path(__file__).parents[1]
            / "tests/fixtures/qwen3_moe_full_oracle.json"
        )
        self.assertEqual(json.loads(fixture.read_text()), generator.generate())


if __name__ == "__main__":
    unittest.main()

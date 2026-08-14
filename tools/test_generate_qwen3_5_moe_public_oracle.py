#!/usr/bin/env python3
"""Regression tests for the public Qwen3.6-35B-A3B oracle boundary."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import generate_qwen3_5_moe_public_oracle as oracle


class Float32OperationTests(unittest.TestCase):
    class Value:
        def __init__(self, name: str) -> None:
            self.name = name

        def float(self) -> str:
            return f"{self.name}-f32"

    def test_promotes_all_weight_bearing_operations_and_restores_them(self) -> None:
        calls: list[tuple[object, ...]] = []

        def linear(*args: object) -> str:
            calls.append(("linear", *args))
            return "linear-result"

        def embedding(*args: object, **kwargs: object) -> str:
            calls.append(("embedding", *args, kwargs))
            return "embedding-result"

        def conv1d(*args: object, **kwargs: object) -> str:
            calls.append(("conv1d", *args, kwargs))
            return "conv1d-result"

        functional = SimpleNamespace(
            linear=linear,
            embedding=embedding,
            conv1d=conv1d,
        )
        torch = SimpleNamespace(nn=SimpleNamespace(functional=functional))
        input_value = self.Value("input")
        weight = self.Value("weight")
        bias = self.Value("bias")

        with oracle.float32_operations(torch):
            self.assertEqual(
                functional.linear(input_value, weight, bias), "linear-result"
            )
            self.assertEqual(
                functional.embedding("tokens", weight, 7, sparse=True),
                "embedding-result",
            )
            self.assertEqual(
                functional.conv1d(input_value, weight, bias, 1, groups=4),
                "conv1d-result",
            )

        self.assertIs(functional.linear, linear)
        self.assertIs(functional.embedding, embedding)
        self.assertIs(functional.conv1d, conv1d)
        self.assertEqual(
            calls,
            [
                ("linear", "input-f32", "weight-f32", "bias-f32"),
                ("embedding", "tokens", "weight-f32", 7, {"sparse": True}),
                (
                    "conv1d",
                    "input-f32",
                    "weight-f32",
                    "bias-f32",
                    1,
                    {"groups": 4},
                ),
            ],
        )

    def test_restores_functions_after_an_error(self) -> None:
        def operation(*args: object, **kwargs: object) -> None:
            return None

        functional = SimpleNamespace(
            linear=operation,
            embedding=operation,
            conv1d=operation,
        )
        torch = SimpleNamespace(nn=SimpleNamespace(functional=functional))
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with oracle.float32_operations(torch):
                raise RuntimeError("stop")
        self.assertIs(functional.linear, operation)
        self.assertIs(functional.embedding, operation)
        self.assertIs(functional.conv1d, operation)


class ReferencePolicyTests(unittest.TestCase):
    def test_pins_eager_fallbacks_with_bf16_parameter_storage(self) -> None:
        bfloat16 = object()
        options = oracle.transformers_model_options(
            SimpleNamespace(bfloat16=bfloat16)
        )
        self.assertIs(options["torch_dtype"], bfloat16)
        self.assertEqual(options["attn_implementation"], "eager")
        self.assertEqual(options["experts_implementation"], "eager")
        self.assertFalse(options["use_kernels"])
        self.assertTrue(options["local_files_only"])

    def test_maps_only_the_official_text_wrapper_namespace(self) -> None:
        self.assertEqual(
            oracle.text_checkpoint_key_mapping(),
            {r"^model\.language_model\.": "model."},
        )

    def test_routes_are_normalized_after_top_k(self) -> None:
        class Tensor:
            def __init__(self, values: list[list[float]]) -> None:
                self.values = values

            def float(self) -> "Tensor":
                return self

            def sum(self, dim: int, keepdim: bool) -> "Tensor":
                self.assert_shape_args(dim, keepdim)
                return Tensor([[sum(row)] for row in self.values])

            def __truediv__(self, other: "Tensor") -> "Tensor":
                return Tensor(
                    [
                        [value / denominator[0] for value in row]
                        for row, denominator in zip(
                            self.values, other.values, strict=True
                        )
                    ]
                )

            def tolist(self) -> list[list[float]]:
                return self.values

            @staticmethod
            def assert_shape_args(dim: int, keepdim: bool) -> None:
                if dim != -1 or not keepdim:
                    raise AssertionError("unexpected reduction")

        class Torch:
            @staticmethod
            def softmax(logits: Tensor, dim: int) -> Tensor:
                if dim != -1:
                    raise AssertionError("unexpected softmax dimension")
                return logits

            @staticmethod
            def topk(values: Tensor, k: int, dim: int) -> tuple[Tensor, Tensor]:
                if k != 2 or dim != -1:
                    raise AssertionError("unexpected top-k arguments")
                return Tensor([[0.6, 0.3]]), Tensor([[2, 1]])

        routes = oracle.routes_from_logits(Torch, Tensor([[0.1, 0.3, 0.6]]), 2)
        self.assertEqual([route["expert"] for route in routes[0]], [2, 1])
        self.assertAlmostEqual(sum(route["weight"] for route in routes[0]), 1.0)


if __name__ == "__main__":
    unittest.main()

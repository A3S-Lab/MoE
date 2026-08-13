#!/usr/bin/env python3
"""Generate a dependency-free full Qwen3-MoE decoder oracle.

The equations and checkpoint layout are transcribed from the exact pinned
Transformers source. This script intentionally does not import the Rust crate,
PyTorch, Transformers, NumPy, or Candle.
"""

from __future__ import annotations

import json
import math
import struct


SOURCE = "https://github.com/huggingface/transformers/blob/918dbf131d0df5b46e3f6e1d96174d62aa4d16d6/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py"
VOCAB_SIZE = 13
HIDDEN_SIZE = 4
DENSE_INTERMEDIATE_SIZE = 5
MOE_INTERMEDIATE_SIZE = 3
NUM_LAYERS = 2
NUM_HEADS = 2
NUM_KV_HEADS = 1
HEAD_DIM = 4
NUM_EXPERTS = 3
TOP_K = 2
RMS_EPS = 1e-6
ROPE_THETA = 10_000.0
TOKENS = [1, 4, 2, 7]


def f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def values(elements: int, seed: int) -> list[float]:
    return [
        f32((((index * 17 + seed * 7) % 23) - 11.0) * 0.017)
        for index in range(elements)
    ]


def matrix(rows: int, columns: int, seed: int) -> list[list[float]]:
    flat = values(rows * columns, seed)
    return [flat[row * columns : (row + 1) * columns] for row in range(rows)]


def norm_weight(size: int, seed: int) -> list[float]:
    return [f32(1.0 + value * 0.1) for value in values(size, seed)]


def dot(left: list[float], right: list[float]) -> float:
    total = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        total = f32(total + f32(left_value * right_value))
    return total


def linear(rows: list[list[float]], weight: list[list[float]]) -> list[list[float]]:
    return [[dot(weight_row, row) for weight_row in weight] for row in rows]


def add(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [f32(a + b) for a, b in zip(x, y, strict=True)]
        for x, y in zip(left, right, strict=True)
    ]


def rms_norm_row(row: list[float], weight: list[float]) -> list[float]:
    square_sum = 0.0
    for value in row:
        square_sum = f32(square_sum + f32(value * value))
    variance = f32(square_sum / len(row))
    inverse = f32(1.0 / math.sqrt(f32(variance + RMS_EPS)))
    return [
        f32(f32(value * inverse) * scale)
        for value, scale in zip(row, weight, strict=True)
    ]


def rms_norm(rows: list[list[float]], weight: list[float]) -> list[list[float]]:
    return [rms_norm_row(row, weight) for row in rows]


def softmax(row: list[float]) -> list[float]:
    maximum = max(row)
    exponentials = [f32(math.exp(f32(value - maximum))) for value in row]
    total = 0.0
    for value in exponentials:
        total = f32(total + value)
    return [f32(value / total) for value in exponentials]


def rope(vector: list[float], position: int) -> list[float]:
    half = len(vector) // 2
    output = [0.0] * len(vector)
    for dimension in range(half):
        frequency = 1.0 / (ROPE_THETA ** ((2 * dimension) / len(vector)))
        angle = position * frequency
        cosine = f32(math.cos(angle))
        sine = f32(math.sin(angle))
        first = vector[dimension]
        second = vector[dimension + half]
        output[dimension] = f32(f32(first * cosine) - f32(second * sine))
        output[dimension + half] = f32(f32(first * sine) + f32(second * cosine))
    return output


def attention(rows: list[list[float]], layer: int) -> list[list[float]]:
    query_width = NUM_HEADS * HEAD_DIM
    key_value_width = NUM_KV_HEADS * HEAD_DIM
    query = linear(rows, matrix(query_width, HIDDEN_SIZE, 30 + layer))
    key = linear(rows, matrix(key_value_width, HIDDEN_SIZE, 40 + layer))
    value = linear(rows, matrix(key_value_width, HIDDEN_SIZE, 50 + layer))
    query_norm = norm_weight(HEAD_DIM, 70 + layer)
    key_norm = norm_weight(HEAD_DIM, 80 + layer)

    query_heads = []
    key_heads = []
    value_heads = []
    for position, row in enumerate(query):
        query_heads.append(
            [
                rope(
                    rms_norm_row(
                        row[head * HEAD_DIM : (head + 1) * HEAD_DIM], query_norm
                    ),
                    position,
                )
                for head in range(NUM_HEADS)
            ]
        )
    for position, row in enumerate(key):
        key_heads.append(
            [
                rope(
                    rms_norm_row(
                        row[head * HEAD_DIM : (head + 1) * HEAD_DIM], key_norm
                    ),
                    position,
                )
                for head in range(NUM_KV_HEADS)
            ]
        )
    for row in value:
        value_heads.append(
            [
                row[head * HEAD_DIM : (head + 1) * HEAD_DIM]
                for head in range(NUM_KV_HEADS)
            ]
        )

    repetitions = NUM_HEADS // NUM_KV_HEADS
    attended_rows = []
    for query_position in range(len(rows)):
        concatenated = []
        for head in range(NUM_HEADS):
            key_value_head = head // repetitions
            scores = [
                f32(
                    dot(
                        query_heads[query_position][head],
                        key_heads[key_position][key_value_head],
                    )
                    / math.sqrt(HEAD_DIM)
                )
                for key_position in range(query_position + 1)
            ]
            probabilities = softmax(scores)
            attended = [0.0] * HEAD_DIM
            for key_position, probability in enumerate(probabilities):
                for dimension in range(HEAD_DIM):
                    attended[dimension] = f32(
                        attended[dimension]
                        + f32(
                            probability
                            * value_heads[key_position][key_value_head][dimension]
                        )
                    )
            concatenated.extend(attended)
        attended_rows.append(concatenated)
    return linear(attended_rows, matrix(HIDDEN_SIZE, query_width, 60 + layer))


def silu(value: float) -> float:
    return f32(value / f32(1.0 + f32(math.exp(-value))))


def projected_mlp(
    rows: list[list[float]],
    gate_weight: list[list[float]],
    up_weight: list[list[float]],
    down_weight: list[list[float]],
) -> list[list[float]]:
    gate = linear(rows, gate_weight)
    up = linear(rows, up_weight)
    intermediate = [
        [f32(silu(gate_value) * up_value) for gate_value, up_value in zip(g, u, strict=True)]
        for g, u in zip(gate, up, strict=True)
    ]
    return linear(intermediate, down_weight)


def dense_mlp(rows: list[list[float]], layer: int) -> list[list[float]]:
    return projected_mlp(
        rows,
        matrix(DENSE_INTERMEDIATE_SIZE, HIDDEN_SIZE, 90 + layer),
        matrix(DENSE_INTERMEDIATE_SIZE, HIDDEN_SIZE, 100 + layer),
        matrix(HIDDEN_SIZE, DENSE_INTERMEDIATE_SIZE, 110 + layer),
    )


def sparse_mlp(
    rows: list[list[float]], layer: int
) -> tuple[list[list[float]], list[list[float]], list[list[dict[str, float | int]]]]:
    router_logits = linear(rows, matrix(NUM_EXPERTS, HIDDEN_SIZE, 120 + layer))
    routes = []
    for logits in router_logits:
        probabilities = softmax(logits)
        selected = sorted(
            range(NUM_EXPERTS), key=lambda expert: (-probabilities[expert], expert)
        )[:TOP_K]
        normalization = 0.0
        for expert in selected:
            normalization = f32(normalization + probabilities[expert])
        routes.append(
            [
                {
                    "expert": expert,
                    "weight": f32(probabilities[expert] / normalization),
                }
                for expert in selected
            ]
        )

    output = [[0.0] * HIDDEN_SIZE for _ in rows]
    for expert in range(NUM_EXPERTS):
        expert_output = projected_mlp(
            rows,
            matrix(MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE, 200 + layer * 10 + expert),
            matrix(MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE, 300 + layer * 10 + expert),
            matrix(HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE, 400 + layer * 10 + expert),
        )
        for position, selected in enumerate(routes):
            route = next((route for route in selected if route["expert"] == expert), None)
            if route is None:
                continue
            for dimension, value in enumerate(expert_output[position]):
                output[position][dimension] = f32(
                    output[position][dimension]
                    + f32(value * float(route["weight"]))
                )
    return output, router_logits, routes


def generate() -> dict[str, object]:
    hidden_states = [matrix(VOCAB_SIZE, HIDDEN_SIZE, 1)[token] for token in TOKENS]
    layer_router_logits: list[list[list[float]] | None] = []
    layer_routes: list[list[list[dict[str, float | int]]] | None] = []
    for layer in range(NUM_LAYERS):
        normalized = rms_norm(hidden_states, norm_weight(HIDDEN_SIZE, 10 + layer))
        hidden_states = add(hidden_states, attention(normalized, layer))
        normalized = rms_norm(hidden_states, norm_weight(HIDDEN_SIZE, 20 + layer))
        if layer == 0:
            mlp_output = dense_mlp(normalized, layer)
            layer_router_logits.append(None)
            layer_routes.append(None)
        else:
            mlp_output, router_logits, routes = sparse_mlp(normalized, layer)
            layer_router_logits.append(router_logits)
            layer_routes.append(routes)
        hidden_states = add(hidden_states, mlp_output)

    hidden_states = rms_norm(hidden_states, norm_weight(HIDDEN_SIZE, 3))
    logits = linear(hidden_states, matrix(VOCAB_SIZE, HIDDEN_SIZE, 2))
    return {
        "schema": "a3s.moe.qwen3-moe-full-oracle.v1",
        "source": SOURCE,
        "tokens": TOKENS,
        "logits": logits,
        "layerRouterLogits": layer_router_logits,
        "layerRoutes": layer_routes,
    }


def main() -> None:
    print(json.dumps(generate(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

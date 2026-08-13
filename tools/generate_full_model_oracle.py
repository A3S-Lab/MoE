#!/usr/bin/env python3
"""Generate the full tiny OLMoE decoder oracle printed to stdout.

This is a dependency-free transcription of the pinned Transformers equations.
It intentionally does not import or execute the Rust implementation.
"""

import json
import math
import struct


SOURCE = "https://github.com/huggingface/transformers/blob/918dbf131d0df5b46e3f6e1d96174d62aa4d16d6/src/transformers/models/olmoe/modeling_olmoe.py"
VOCAB_SIZE = 11
HIDDEN_SIZE = 4
INTERMEDIATE_SIZE = 3
NUM_LAYERS = 2
NUM_HEADS = 2
NUM_KV_HEADS = 1
NUM_EXPERTS = 3
TOP_K = 2
RMS_EPS = 1e-5
ROPE_THETA = 10_000.0
TOKENS = [1, 4, 2, 7]


def f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def values(elements: int, seed: int) -> list[float]:
    return [f32((((index * 17 + seed * 7) % 23) - 11.0) * 0.017) for index in range(elements)]


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
    return [[f32(a + b) for a, b in zip(x, y, strict=True)] for x, y in zip(left, right, strict=True)]


def rms_norm(rows: list[list[float]], weight: list[float]) -> list[list[float]]:
    output = []
    for row in rows:
        square_sum = 0.0
        for value in row:
            square_sum = f32(square_sum + f32(value * value))
        variance = f32(square_sum / len(row))
        inverse = f32(1.0 / math.sqrt(f32(variance + RMS_EPS)))
        output.append([f32(f32(value * inverse) * scale) for value, scale in zip(row, weight, strict=True)])
    return output


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
    head_dim = HIDDEN_SIZE // NUM_HEADS
    kv_size = NUM_KV_HEADS * head_dim
    query = rms_norm(linear(rows, matrix(HIDDEN_SIZE, HIDDEN_SIZE, 30 + layer)), norm_weight(HIDDEN_SIZE, 70 + layer))
    key = rms_norm(linear(rows, matrix(kv_size, HIDDEN_SIZE, 40 + layer)), norm_weight(kv_size, 80 + layer))
    value = linear(rows, matrix(kv_size, HIDDEN_SIZE, 50 + layer))

    query_heads = [
        [rope(row[head * head_dim : (head + 1) * head_dim], position) for head in range(NUM_HEADS)]
        for position, row in enumerate(query)
    ]
    key_heads = [
        [rope(row[head * head_dim : (head + 1) * head_dim], position) for head in range(NUM_KV_HEADS)]
        for position, row in enumerate(key)
    ]
    value_heads = [
        [row[head * head_dim : (head + 1) * head_dim] for head in range(NUM_KV_HEADS)]
        for row in value
    ]
    repetitions = NUM_HEADS // NUM_KV_HEADS
    attended_rows = []
    for query_position in range(len(rows)):
        concatenated = []
        for head in range(NUM_HEADS):
            kv_head = head // repetitions
            scores = [
                f32(dot(query_heads[query_position][head], key_heads[key_position][kv_head]) / math.sqrt(head_dim))
                for key_position in range(query_position + 1)
            ]
            probabilities = softmax(scores)
            attended = [0.0] * head_dim
            for key_position, probability in enumerate(probabilities):
                for dimension in range(head_dim):
                    attended[dimension] = f32(
                        attended[dimension]
                        + f32(probability * value_heads[key_position][kv_head][dimension])
                    )
            concatenated.extend(attended)
        attended_rows.append(concatenated)
    return linear(attended_rows, matrix(HIDDEN_SIZE, HIDDEN_SIZE, 60 + layer))


def silu(value: float) -> float:
    return f32(value / f32(1.0 + f32(math.exp(-value))))


def sparse_mlp(rows: list[list[float]], layer: int) -> tuple[list[list[float]], list[list[dict[str, float | int]]]]:
    router_logits = linear(rows, matrix(NUM_EXPERTS, HIDDEN_SIZE, 90 + layer))
    routes = []
    for logits in router_logits:
        probabilities = softmax(logits)
        selected = sorted(range(NUM_EXPERTS), key=lambda expert: (-probabilities[expert], expert))[:TOP_K]
        routes.append([{"expert": expert, "weight": probabilities[expert]} for expert in selected])

    output = [[0.0] * HIDDEN_SIZE for _ in rows]
    for expert in range(NUM_EXPERTS):
        gate_weight = matrix(INTERMEDIATE_SIZE, HIDDEN_SIZE, 100 + layer * 10 + expert)
        up_weight = matrix(INTERMEDIATE_SIZE, HIDDEN_SIZE, 200 + layer * 10 + expert)
        down_weight = matrix(HIDDEN_SIZE, INTERMEDIATE_SIZE, 300 + layer * 10 + expert)
        for position, selected in enumerate(routes):
            route = next((route for route in selected if route["expert"] == expert), None)
            if route is None:
                continue
            gate = linear([rows[position]], gate_weight)[0]
            up = linear([rows[position]], up_weight)[0]
            intermediate = [f32(silu(gate_value) * up_value) for gate_value, up_value in zip(gate, up, strict=True)]
            expert_output = linear([intermediate], down_weight)[0]
            for dimension, value in enumerate(expert_output):
                output[position][dimension] = f32(
                    output[position][dimension] + f32(value * float(route["weight"]))
                )
    return output, routes


hidden_states = [matrix(VOCAB_SIZE, HIDDEN_SIZE, 1)[token] for token in TOKENS]
layer_routes = []
for layer in range(NUM_LAYERS):
    normalized = rms_norm(hidden_states, norm_weight(HIDDEN_SIZE, 10 + layer))
    hidden_states = add(hidden_states, attention(normalized, layer))
    normalized = rms_norm(hidden_states, norm_weight(HIDDEN_SIZE, 20 + layer))
    moe_output, routes = sparse_mlp(normalized, layer)
    hidden_states = add(hidden_states, moe_output)
    layer_routes.append(routes)

hidden_states = rms_norm(hidden_states, norm_weight(HIDDEN_SIZE, 3))
logits = linear(hidden_states, matrix(VOCAB_SIZE, HIDDEN_SIZE, 2))
fixture = {
    "source": SOURCE,
    "tokens": TOKENS,
    "logits": logits,
    "layerRoutes": layer_routes,
}
print(json.dumps(fixture, indent=2))

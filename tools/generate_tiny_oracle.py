#!/usr/bin/env python3
"""Generate a tiny OLMoE oracle with only the Python standard library.

The equations and fused gate/up layout follow Hugging Face Transformers
`modeling_olmoe.py` at commit 918dbf131d0df5b46e3f6e1d96174d62aa4d16d6.
This script prints the fixture to stdout and never changes repository files.
"""

import json
import math
import struct


def f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def dot(left: list[float], right: list[float]) -> float:
    total = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        total = f32(total + f32(left_value * right_value))
    return total


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [f32(math.exp(f32(value - maximum))) for value in values]
    total = 0.0
    for value in exponentials:
        total = f32(total + value)
    return [f32(value / total) for value in exponentials]


def silu(value: float) -> float:
    return f32(value / f32(1.0 + f32(math.exp(-value))))


def matrix(rows: int, columns: int, values: list[float]) -> dict[str, object]:
    return {"rows": rows, "columns": columns, "values": [f32(value) for value in values]}


hidden_size = 3
intermediate_size = 2
num_experts = 4
top_k = 2
layer = 3

hidden_rows = [
    [1.0, -0.5, 0.25],
    [-0.2, 0.8, 0.5],
    [0.7, 0.1, -0.6],
]
router_rows = [
    [0.2, -0.1, 0.4],
    [-0.3, 0.5, 0.1],
    [0.6, 0.2, -0.2],
    [-0.2, -0.4, 0.7],
]

experts: list[dict[str, object]] = []
expert_rows: list[tuple[list[list[float]], list[list[float]]]] = []
for expert in range(num_experts):
    scale = float(expert + 1)
    gate_up = [
        [0.10 * scale, -0.05 * scale, 0.02 * scale],
        [-0.03 * scale, 0.07 * scale, 0.04],
        [0.05, 0.02 * scale, -0.06],
        [-0.04, 0.03, 0.08 * scale],
    ]
    down = [
        [0.20 * scale, -0.10],
        [-0.05, 0.15 * scale],
        [0.07 * scale, 0.04],
    ]
    gate_up = [[f32(value) for value in row] for row in gate_up]
    down = [[f32(value) for value in row] for row in down]
    expert_rows.append((gate_up, down))
    experts.append(
        {
            "gateUp": matrix(2 * intermediate_size, hidden_size, sum(gate_up, [])),
            "down": matrix(hidden_size, intermediate_size, sum(down, [])),
        }
    )

all_logits: list[float] = []
routes: list[list[dict[str, object]]] = []
for hidden in hidden_rows:
    logits = [dot(row, hidden) for row in router_rows]
    probabilities = softmax(logits)
    selected = sorted(range(num_experts), key=lambda index: (-probabilities[index], index))[:top_k]
    all_logits.extend(logits)
    routes.append(
        [{"expert": expert, "weight": probabilities[expert]} for expert in selected]
    )

outputs = [[0.0] * hidden_size for _ in hidden_rows]
for expert in range(num_experts):
    gate_up, down = expert_rows[expert]
    for position, selected in enumerate(routes):
        match = next((route for route in selected if route["expert"] == expert), None)
        if match is None:
            continue
        hidden = hidden_rows[position]
        gate = [dot(row, hidden) for row in gate_up[:intermediate_size]]
        up = [dot(row, hidden) for row in gate_up[intermediate_size:]]
        activated = [f32(silu(gate_value) * up_value) for gate_value, up_value in zip(gate, up, strict=True)]
        expert_output = [dot(row, activated) for row in down]
        for column, value in enumerate(expert_output):
            weighted = f32(value * float(match["weight"]))
            outputs[position][column] = f32(outputs[position][column] + weighted)

fixture = {
    "source": "https://github.com/huggingface/transformers/blob/918dbf131d0df5b46e3f6e1d96174d62aa4d16d6/src/transformers/models/olmoe/modeling_olmoe.py",
    "layer": layer,
    "config": {
        "hiddenSize": hidden_size,
        "intermediateSize": intermediate_size,
        "numExperts": num_experts,
        "topK": top_k,
        "normalizeTopK": False,
    },
    "hiddenStates": matrix(len(hidden_rows), hidden_size, sum(hidden_rows, [])),
    "routerWeight": matrix(num_experts, hidden_size, sum(router_rows, [])),
    "experts": experts,
    "expected": {
        "logits": matrix(len(hidden_rows), num_experts, all_logits),
        "routes": routes,
        "hiddenStates": matrix(len(hidden_rows), hidden_size, sum(outputs, [])),
    },
}

print(json.dumps(fixture, indent=2, sort_keys=False))

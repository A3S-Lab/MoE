#!/usr/bin/env python3
"""Emit a dependency-free Qwen3-MoE sparse-layer oracle to stdout."""

from __future__ import annotations

import json
import math


def values(count: int, seed: int) -> list[float]:
    return [
        round((((index * 17 + seed * 7) % 23) - 11) * 0.031, 6)
        for index in range(count)
    ]


def matrix(rows: int, columns: int, seed: int) -> list[list[float]]:
    flat = values(rows * columns, seed)
    return [flat[row * columns : (row + 1) * columns] for row in range(rows)]


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def softmax(row: list[float]) -> list[float]:
    maximum = max(row)
    exponentials = [math.exp(value - maximum) for value in row]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def expert_forward(
    hidden: list[float], gate_up: list[list[float]], down: list[list[float]]
) -> list[float]:
    intermediate_size = len(gate_up) // 2
    intermediate = []
    for row in range(intermediate_size):
        gate = dot(gate_up[row], hidden)
        up = dot(gate_up[row + intermediate_size], hidden)
        intermediate.append((gate / (1.0 + math.exp(-gate))) * up)
    return [dot(row, intermediate) for row in down]


def clean(value: object) -> object:
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    return value


def generate() -> dict[str, object]:
    hidden_size = 3
    intermediate_size = 2
    num_experts = 4
    top_k = 2
    hidden_states = [values(hidden_size, 50), values(hidden_size, 51)]
    router_weight = matrix(num_experts, hidden_size, 60)
    experts = [
        {
            "gateUp": matrix(2 * intermediate_size, hidden_size, 70 + expert),
            "down": matrix(hidden_size, intermediate_size, 90 + expert),
        }
        for expert in range(num_experts)
    ]

    router_logits = [
        [dot(weight, hidden) for weight in router_weight] for hidden in hidden_states
    ]
    routes = []
    outputs = []
    for position, hidden in enumerate(hidden_states):
        probabilities = softmax(router_logits[position])
        selected = sorted(range(num_experts), key=lambda expert: (-probabilities[expert], expert))[
            :top_k
        ]
        normalization = sum(probabilities[expert] for expert in selected)
        position_routes = [
            {"expert": expert, "weight": probabilities[expert] / normalization}
            for expert in selected
        ]
        routes.append(position_routes)
        output = [0.0] * hidden_size
        for route in position_routes:
            expert = experts[route["expert"]]
            contribution = expert_forward(hidden, expert["gateUp"], expert["down"])
            for index, value in enumerate(contribution):
                output[index] += value * route["weight"]
        outputs.append(output)

    return clean({
        "schema": "a3s.moe.qwen3-moe-sparse-oracle.v1",
        "config": {
            "hiddenSize": hidden_size,
            "intermediateSize": intermediate_size,
            "numExperts": num_experts,
            "topK": top_k,
            "normalizeTopK": True,
        },
        "layer": 7,
        "hiddenStates": hidden_states,
        "routerWeight": router_weight,
        "experts": experts,
        "output": {
            "routerLogits": router_logits,
            "routes": routes,
            "hiddenStates": outputs,
        },
    })


def main() -> None:
    print(json.dumps(generate(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

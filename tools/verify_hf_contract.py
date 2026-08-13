#!/usr/bin/env python3
"""Verify the pinned public OLMoE SafeTensor metadata without weight downloads."""

import json
import math
import struct
import urllib.request


REPOSITORY = "allenai/OLMoE-1B-7B-0924"
REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
BASE_URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}"
MAX_HEADER_BYTES = 16 * 1024 * 1024


def fetch(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    headers = {}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(MAX_HEADER_BYTES + 1)
        if len(data) > MAX_HEADER_BYTES:
            raise ValueError(f"response from {url} exceeds the metadata limit")
        if byte_range is not None:
            expected = byte_range[1] - byte_range[0] + 1
            if response.status != 206 or len(data) != expected:
                raise ValueError(
                    f"range request returned status {response.status} and {len(data)} bytes; expected 206 and {expected}"
                )
        return data


def fetch_json(name: str) -> dict[str, object]:
    return json.loads(fetch(f"{BASE_URL}/{name}"))


def fetch_safetensor_header(name: str) -> dict[str, object]:
    url = f"{BASE_URL}/{name}"
    header_length = struct.unpack("<Q", fetch(url, (0, 7)))[0]
    if header_length == 0 or header_length > MAX_HEADER_BYTES:
        raise ValueError(f"invalid SafeTensor header length {header_length} for {name}")
    return json.loads(fetch(url, (8, 7 + header_length)))


config = fetch_json("config.json")
index = fetch_json("model.safetensors.index.json")
weight_map = index["weight_map"]
shards = sorted(set(weight_map.values()))
headers: dict[str, tuple[str, dict[str, object]]] = {}
for shard in shards:
    for tensor_name, tensor in fetch_safetensor_header(shard).items():
        if tensor_name == "__metadata__":
            continue
        if tensor_name in headers:
            raise ValueError(f"duplicate tensor {tensor_name}")
        headers[tensor_name] = (shard, tensor)


def expected_tensors() -> dict[str, list[int]]:
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    layers = int(config["num_hidden_layers"])
    experts = int(config["num_experts"])
    vocab = int(config["vocab_size"])
    expected = {
        "model.embed_tokens.weight": [vocab, hidden],
        "model.norm.weight": [hidden],
        "lm_head.weight": [vocab, hidden],
    }
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        expected.update(
            {
                f"{prefix}.input_layernorm.weight": [hidden],
                f"{prefix}.post_attention_layernorm.weight": [hidden],
                f"{prefix}.self_attn.q_proj.weight": [hidden, hidden],
                f"{prefix}.self_attn.k_proj.weight": [hidden, hidden],
                f"{prefix}.self_attn.v_proj.weight": [hidden, hidden],
                f"{prefix}.self_attn.o_proj.weight": [hidden, hidden],
                f"{prefix}.self_attn.q_norm.weight": [hidden],
                f"{prefix}.self_attn.k_norm.weight": [hidden],
                f"{prefix}.mlp.gate.weight": [experts, hidden],
            }
        )
        for expert in range(experts):
            expert_prefix = f"{prefix}.mlp.experts.{expert}"
            expected[f"{expert_prefix}.gate_proj.weight"] = [intermediate, hidden]
            expected[f"{expert_prefix}.up_proj.weight"] = [intermediate, hidden]
            expected[f"{expert_prefix}.down_proj.weight"] = [hidden, intermediate]
    return expected


expected = expected_tensors()
if set(headers) != set(expected) or set(weight_map) != set(expected):
    missing = sorted(set(expected) - set(headers))
    unexpected = sorted(set(headers) - set(expected))
    raise ValueError(f"tensor name mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")

parameter_count = 0
for name, shape in expected.items():
    shard, tensor = headers[name]
    if weight_map[name] != shard:
        raise ValueError(f"index maps {name} to {weight_map[name]}, header is in {shard}")
    if tensor["dtype"] != "BF16" or tensor["shape"] != shape:
        raise ValueError(
            f"{name}: expected BF16 {shape}, found {tensor['dtype']} {tensor['shape']}"
        )
    parameter_count += math.prod(shape)

print(
    json.dumps(
        {
            "repository": REPOSITORY,
            "revision": REVISION,
            "shards": shards,
            "tensorCount": len(expected),
            "parameterCount": parameter_count,
            "dtype": "BF16",
            "status": "verified",
        },
        indent=2,
    )
)

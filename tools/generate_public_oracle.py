#!/usr/bin/env python3
"""Generate a full OLMoE public-checkpoint oracle with pinned Transformers.

Run this script with PYTHONPATH pointing at the ``src`` directory of the exact
Transformers checkout declared below. The output includes every prompt logit,
every router logit, and every selected route. It never overwrites an existing
fixture unless ``--overwrite`` is explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ORACLE_SCHEMA = "a3s.moe.olmoe-public-oracle.v1"
MODEL_ID = "allenai/OLMoE-1B-7B-0924"
MODEL_REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
TRANSFORMERS_REVISION = "918dbf131d0df5b46e3f6e1d96174d62aa4d16d6"
TRANSFORMERS_SOURCE_SHA256 = (
    "53a94a479f9904674a5f45aba0387c13466a1f2a2d3cdb9226f9cf58946ebbf8"
)
DEFAULT_PROMPT = "Bitcoin is"
READ_CHUNK_BYTES = 8 * 1024 * 1024
PINNED_CHECKPOINT_FILES = {
    "config.json": (
        759,
        "3643aa880d2f1c9b418156269ae791c73e5612d6b6b6fde0724d927cf89b6335",
    ),
    "model.safetensors.index.json": (
        287_214,
        "0e2e1e0d8d357ac7af817cff28410c3dbad398f060c517a433e4076b2aae5579",
    ),
    "model-00001-of-00003.safetensors": (
        4_997_744_872,
        "5e3cff7e367794685c241169072c940d200918617d5e2813f1c387dff52d845e",
    ),
    "model-00002-of-00003.safetensors": (
        4_997_235_176,
        "15ef5c730ee3cfed7199498788cd2faf337203fc74b529625e7502cdd759f4a7",
    ),
    "model-00003-of-00003.safetensors": (
        3_843_741_912,
        "a9abac4ac1b55c9adabac721a02fa39971f103eea9a65c310972b1246de76e04",
    ),
    "tokenizer.json": (
        2_115_417,
        "a094266ac6c4982efba277bc251349a5a6d6ad37efb39a2a90f53d8be2a40a40",
    ),
    "tokenizer_config.json": (
        5_372,
        "78a839c7851f14f9fb30e664c2b46166dc0628f2900679e5ec160656f702edff",
    ),
}
PINNED_WEIGHT_SHARDS = {
    name for name in PINNED_CHECKPOINT_FILES if name.endswith(".safetensors")
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(READ_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def git_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(
        "Transformers must be imported from a Git checkout so its revision can be verified"
    )


def verified_transformers_source() -> tuple[str, str]:
    import transformers.models.olmoe.modeling_olmoe as modeling_olmoe

    source = Path(modeling_olmoe.__file__).resolve(strict=True)
    root = git_root(source.parent)
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != TRANSFORMERS_REVISION:
        raise RuntimeError(
            f"Transformers checkout is {revision}, expected {TRANSFORMERS_REVISION}"
        )
    source_sha256 = sha256_file(source)
    if source_sha256 != TRANSFORMERS_SOURCE_SHA256:
        raise RuntimeError(
            "Transformers OLMoE source SHA-256 is "
            f"{source_sha256}, expected {TRANSFORMERS_SOURCE_SHA256}"
        )
    return revision, source_sha256


def regular_file(root: Path, name: str) -> Path:
    if Path(name).name != name:
        raise RuntimeError(f"checkpoint inventory contains unsafe filename {name!r}")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"checkpoint file must be regular and non-symlink: {path}")
    return path


def checkpoint_inventory(root: Path) -> list[dict[str, Any]]:
    index_path = regular_file(root, "model.safetensors.index.json")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("model.safetensors.index.json has no weight_map")
    if any(not isinstance(name, str) for name in weight_map.values()):
        raise RuntimeError("checkpoint index contains a non-string shard name")
    declared_shards = set(weight_map.values())
    if declared_shards != PINNED_WEIGHT_SHARDS:
        raise RuntimeError(
            "checkpoint index shard inventory does not match the pinned revision"
        )
    inventory = []
    for name, (expected_bytes, expected_sha256) in sorted(
        PINNED_CHECKPOINT_FILES.items()
    ):
        path = regular_file(root, name)
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"checkpoint file {name!r} has {actual_bytes} bytes, "
                f"expected {expected_bytes} for revision {MODEL_REVISION}"
            )
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"checkpoint file {name!r} has SHA-256 {actual_sha256}, "
                f"expected {expected_sha256} for revision {MODEL_REVISION}"
            )
        inventory.append(
            {"name": name, "bytes": expected_bytes, "sha256": expected_sha256}
        )
    return inventory


def routes_from_logits(torch: Any, logits: Any, top_k: int) -> list[list[dict[str, Any]]]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    weights, experts = torch.topk(probabilities, k=top_k, dim=-1)
    output: list[list[dict[str, Any]]] = []
    for row_experts, row_weights in zip(experts.tolist(), weights.tolist(), strict=True):
        output.append(
            [
                {"expert": int(expert), "weight": float(weight)}
                for expert, weight in zip(row_experts, row_weights, strict=True)
            ]
        )
    return output


def generate(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads <= 0:
        raise ValueError("--threads must be greater than zero")
    checkpoint = args.checkpoint.resolve(strict=True)
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint is not a directory: {checkpoint}")
    inventory = checkpoint_inventory(checkpoint)

    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    transformers.utils.logging.set_verbosity_error()
    transformers_revision, source_sha256 = verified_transformers_source()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)

    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    if config.model_type != "olmoe":
        raise RuntimeError(f"expected model_type 'olmoe', found {config.model_type!r}")
    config.output_router_logits = True
    config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    encoded = tokenizer(args.prompt, add_special_tokens=False, return_tensors="pt")
    token_ids = [int(token) for token in encoded.input_ids[0].tolist()]
    if not token_ids:
        raise RuntimeError("prompt produced no tokens")

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    with torch.inference_mode():
        result = model(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            use_cache=False,
            output_router_logits=True,
            return_dict=True,
        )
    if result.router_logits is None or len(result.router_logits) != config.num_hidden_layers:
        raise RuntimeError("Transformers did not return one router tensor per layer")

    logits = result.logits.float().cpu()[0]
    router_logits = [
        layer.float().cpu().reshape(len(token_ids), config.num_experts)
        for layer in result.router_logits
    ]
    return {
        "schema": ORACLE_SCHEMA,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "transformersRevision": transformers_revision,
            "transformersSourceSha256": source_sha256,
            "files": inventory,
        },
        "input": {
            "text": args.prompt,
            "addSpecialTokens": False,
            "tokenIds": token_ids,
        },
        "output": {
            "logits": logits.tolist(),
            "layerRouterLogits": [layer.tolist() for layer in router_logits],
            "layerRoutes": [
                routes_from_logits(torch, layer, config.num_experts_per_tok)
                for layer in router_logits
            ],
        },
    }


def write_oracle(path: Path, oracle: dict[str, Any], overwrite: bool) -> None:
    destination = path.resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing oracle: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as output:
            json.dump(oracle, output, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    try:
        oracle = generate(args)
        write_oracle(args.output, oracle, args.overwrite)
    except Exception as error:  # noqa: BLE001 - CLI boundary includes ML stack errors.
        print(f"generate_public_oracle.py: {error}", file=sys.stderr)
        return 1
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

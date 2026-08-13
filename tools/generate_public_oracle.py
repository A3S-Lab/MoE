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
DEFAULT_PROMPT = "Bitcoin is"
READ_CHUNK_BYTES = 8 * 1024 * 1024


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
    return revision, sha256_file(source)


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
    names = {
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        *weight_map.values(),
    }
    tokenizer_config = root / "tokenizer_config.json"
    if tokenizer_config.exists():
        names.add("tokenizer_config.json")
    inventory = []
    for name in sorted(names):
        if not isinstance(name, str):
            raise RuntimeError("checkpoint inventory contains a non-string filename")
        path = regular_file(root, name)
        inventory.append(
            {"name": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
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
            "files": checkpoint_inventory(checkpoint),
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

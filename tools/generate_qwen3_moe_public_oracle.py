#!/usr/bin/env python3
"""Generate the pinned public Qwen3-MoE BF16 Transformers oracle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from qwen3_moe_public_contract import (
    MODEL_ID,
    MODEL_REVISION,
    TRANSFORMERS_DTYPE,
    TRANSFORMERS_REVISION,
    TRANSFORMERS_SOURCE_SHA256,
    checkpoint_inventory,
    sha256_file,
)


ORACLE_SCHEMA = "a3s.moe.qwen3-moe-public-oracle.v1"
DEFAULT_PROMPT = "Bitcoin is"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def git_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(
        "Transformers must be imported from a Git checkout so its revision can be verified"
    )


def verified_transformers_source() -> tuple[str, str]:
    import transformers.models.qwen3_moe.modeling_qwen3_moe as modeling_qwen3_moe

    source = Path(modeling_qwen3_moe.__file__).resolve(strict=True)
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
            "Transformers Qwen3-MoE source SHA-256 is "
            f"{source_sha256}, expected {TRANSFORMERS_SOURCE_SHA256}"
        )
    return revision, source_sha256


def is_sparse_layer(config: Any, layer: int) -> bool:
    return (
        layer not in config.mlp_only_layers
        and config.num_experts > 0
        and (layer + 1) % config.decoder_sparse_step == 0
    )


def routes_from_logits(
    torch: Any,
    logits: Any,
    top_k: int,
    normalize: bool,
) -> list[list[dict[str, Any]]]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    weights, experts = torch.topk(probabilities, k=top_k, dim=-1)
    if normalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
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
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    if config.model_type != "qwen3_moe":
        raise RuntimeError(
            f"expected model_type 'qwen3_moe', found {config.model_type!r}"
        )
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
        torch_dtype=torch.bfloat16,
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

    sparse_layers = [
        layer
        for layer in range(config.num_hidden_layers)
        if is_sparse_layer(config, layer)
    ]
    if result.router_logits is None or len(result.router_logits) != len(sparse_layers):
        raise RuntimeError(
            "Transformers did not return one router tensor per sparse layer"
        )
    positions = len(token_ids)
    sparse_router_logits = iter(result.router_logits)
    layer_router_logits: list[list[list[float]] | None] = []
    layer_routes: list[list[list[dict[str, Any]]] | None] = []
    for layer in range(config.num_hidden_layers):
        if not is_sparse_layer(config, layer):
            layer_router_logits.append(None)
            layer_routes.append(None)
            continue
        router = next(sparse_router_logits).float().cpu().reshape(
            positions, config.num_experts
        )
        layer_router_logits.append(router.tolist())
        layer_routes.append(
            routes_from_logits(
                torch,
                router,
                config.num_experts_per_tok,
                config.norm_topk_prob,
            )
        )

    return {
        "schema": ORACLE_SCHEMA,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "transformersRevision": transformers_revision,
            "transformersSourceSha256": source_sha256,
            "transformersDtype": TRANSFORMERS_DTYPE,
            "files": inventory,
        },
        "input": {
            "text": args.prompt,
            "addSpecialTokens": False,
            "tokenIds": token_ids,
        },
        "output": {
            "logits": result.logits.float().cpu()[0].tolist(),
            "layerRouterLogits": layer_router_logits,
            "layerRoutes": layer_routes,
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
        print(f"generate_qwen3_moe_public_oracle.py: {error}", file=sys.stderr)
        return 1
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

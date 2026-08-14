#!/usr/bin/env python3
"""Generate the pinned public Qwen3.6-35B-A3B F32 Transformers oracle."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from qwen3_5_moe_public_contract import (
    MODEL_ID,
    MODEL_REVISION,
    OUTER_MODEL_TYPE,
    TEXT_MODEL_TYPE,
    TRANSFORMERS_DTYPE,
    TRANSFORMERS_REVISION,
    TRANSFORMERS_SOURCE_SHA256,
    checkpoint_inventory,
    sha256_file,
)


ORACLE_SCHEMA = "a3s.moe.qwen3.6-35b-a3b-public-oracle.v1"
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
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as modeling

    source = Path(modeling.__file__).resolve(strict=True)
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
            "Transformers Qwen3.5-MoE source SHA-256 is "
            f"{source_sha256}, expected {TRANSFORMERS_SOURCE_SHA256}"
        )
    return revision, source_sha256


@contextmanager
def float32_operations(torch: Any):
    """Promote weight-bearing operations without materializing an F32 model."""
    functional = torch.nn.functional
    original_linear = functional.linear
    original_embedding = functional.embedding
    original_conv1d = functional.conv1d

    def linear(input_tensor: Any, weight: Any, bias: Any = None) -> Any:
        promoted_bias = None if bias is None else bias.float()
        return original_linear(input_tensor.float(), weight.float(), promoted_bias)

    def embedding(input_tensor: Any, weight: Any, *args: Any, **kwargs: Any) -> Any:
        return original_embedding(input_tensor, weight.float(), *args, **kwargs)

    def conv1d(
        input_tensor: Any,
        weight: Any,
        bias: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        promoted_bias = None if bias is None else bias.float()
        return original_conv1d(
            input_tensor.float(), weight.float(), promoted_bias, *args, **kwargs
        )

    functional.linear = linear
    functional.embedding = embedding
    functional.conv1d = conv1d
    try:
        yield
    finally:
        functional.linear = original_linear
        functional.embedding = original_embedding
        functional.conv1d = original_conv1d


def transformers_model_options(torch: Any) -> dict[str, Any]:
    """Pin the reference to official eager equations and BF16 storage."""
    return {
        "local_files_only": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
        "experts_implementation": "eager",
        "use_kernels": False,
    }


def text_checkpoint_key_mapping() -> dict[str, str]:
    """Map only the official multimodal wrapper's text namespace."""
    return {r"^model\.language_model\.": "model."}


def routes_from_logits(
    torch: Any,
    logits: Any,
    top_k: int,
) -> list[list[dict[str, Any]]]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    weights, experts = torch.topk(probabilities, k=top_k, dim=-1)
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
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5MoeForCausalLM

    transformers.utils.logging.set_verbosity_error()
    transformers_revision, source_sha256 = verified_transformers_source()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    outer_config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    if outer_config.model_type != OUTER_MODEL_TYPE:
        raise RuntimeError(
            f"expected model_type {OUTER_MODEL_TYPE!r}, found "
            f"{outer_config.model_type!r}"
        )
    config = outer_config.text_config
    if config.model_type != TEXT_MODEL_TYPE:
        raise RuntimeError(
            f"expected text model_type {TEXT_MODEL_TYPE!r}, found {config.model_type!r}"
        )
    config.output_router_logits = True
    config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    encoded = tokenizer(args.prompt, add_special_tokens=False, return_tensors="pt")
    token_ids = [int(token) for token in encoded.input_ids[0].tolist()]
    if not token_ids:
        raise RuntimeError("prompt produced no tokens")

    model, loading_info = Qwen3_5MoeForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        key_mapping=text_checkpoint_key_mapping(),
        output_loading_info=True,
        **transformers_model_options(torch),
    )
    if loading_info["missing_keys"] or loading_info["mismatched_keys"]:
        raise RuntimeError(
            "text-only reference load was incomplete: "
            f"missing={loading_info['missing_keys']!r}, "
            f"mismatched={loading_info['mismatched_keys']!r}"
        )
    unexpected = [
        name
        for name in loading_info["unexpected_keys"]
        if not name.startswith(("model.visual.", "mtp."))
    ]
    if unexpected:
        raise RuntimeError(
            f"text-only reference load found unexpected tensors: {unexpected!r}"
        )
    model.eval()
    with float32_operations(torch), torch.inference_mode():
        result = model(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            use_cache=False,
            output_router_logits=True,
            return_dict=True,
        )

    if result.router_logits is None or len(result.router_logits) != config.num_hidden_layers:
        raise RuntimeError("Transformers did not return one router tensor per text layer")
    positions = len(token_ids)
    layer_router_logits: list[list[list[float]]] = []
    layer_routes: list[list[list[dict[str, Any]]]] = []
    for router in result.router_logits:
        router = router.float().cpu().reshape(positions, config.num_experts)
        layer_router_logits.append(router.tolist())
        layer_routes.append(
            routes_from_logits(torch, router, config.num_experts_per_tok)
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
        print(f"generate_qwen3_5_moe_public_oracle.py: {error}", file=sys.stderr)
        return 1
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

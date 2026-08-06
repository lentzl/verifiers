"""Merge a Hugging Face PEFT adapter into its declared base model."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar

Tensor = TypeVar("Tensor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter", help="Local path or Hugging Face adapter repository")
    parser.add_argument("output", type=Path, help="Directory for merged model weights")
    return parser.parse_args()


def align_adapter_state(
    state: dict[str, Tensor], expected_keys: set[str]
) -> dict[str, Tensor]:
    """Align wrapper-specific LoRA names and reject incomplete checkpoints."""
    if set(state) == expected_keys:
        return state

    aligned: dict[str, Tensor] = {}
    for key, value in state.items():
        suffix = key.removeprefix("model.language_model.")
        matches = [
            candidate for candidate in expected_keys if candidate.endswith(suffix)
        ]
        if len(matches) != 1:
            raise ValueError(f"cannot map adapter tensor {key!r} uniquely")
        aligned[matches[0]] = value

    missing = expected_keys - set(aligned)
    if missing:
        preview = ", ".join(sorted(missing)[:3])
        raise ValueError(
            f"adapter checkpoint is missing {len(missing)} tensors: {preview}"
        )
    return aligned


def tokenizer_eos_token_id(tokenizer: object) -> int:
    """Resolve the tokenizer's EOS token through its actual encoding path."""
    eos_token = getattr(tokenizer, "eos_token", None)
    if not isinstance(eos_token, str):
        raise TypeError("tokenizer does not declare an EOS token")

    token_ids = tokenizer.encode(eos_token, add_special_tokens=False)
    if len(token_ids) != 1 or not isinstance(token_ids[0], int):
        raise ValueError(f"tokenizer EOS token {eos_token!r} is not a single token")

    eos_token_id = token_ids[0]
    configured_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(configured_id, int) and configured_id != eos_token_id:
        raise ValueError(
            f"tokenizer EOS ID {configured_id} does not match encoded "
            f"{eos_token!r} ID {eos_token_id}"
        )
    return eos_token_id


def normalize_model_eos_metadata(model: object, eos_token_id: int) -> None:
    """Align existing model and generation EOS fields with the tokenizer."""
    pending = [getattr(model, "config", None)]
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.eos_token_id = eos_token_id

    visited: set[int] = set()
    while pending:
        config = pending.pop()
        if config is None or id(config) in visited:
            continue
        visited.add(id(config))

        if hasattr(config, "eos_token_id"):
            config.eos_token_id = eos_token_id

        child_names = set(getattr(config, "sub_configs", {}))
        child_names.add("text_config")
        pending.extend(getattr(config, name, None) for name in child_names)


def _numeric_eos_token_ids(value: object, path: str = "$") -> Iterator[tuple[str, int]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key == "eos_token_id":
                if isinstance(child, int) and not isinstance(child, bool):
                    yield child_path, child
                elif isinstance(child, list):
                    for index, item in enumerate(child):
                        if isinstance(item, int) and not isinstance(item, bool):
                            yield f"{child_path}[{index}]", item
            yield from _numeric_eos_token_ids(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _numeric_eos_token_ids(child, f"{path}[{index}]")


def validate_export_eos(output: Path, tokenizer: object) -> int:
    """Reject an export whose numeric EOS metadata disagrees with its tokenizer."""
    eos_token_id = tokenizer_eos_token_id(tokenizer)
    found: list[tuple[Path, str]] = []
    mismatches: list[tuple[Path, str, int]] = []
    for json_path in sorted(output.rglob("*.json")):
        data = json.loads(json_path.read_text())
        for field_path, value in _numeric_eos_token_ids(data):
            found.append((json_path, field_path))
            if value != eos_token_id:
                mismatches.append((json_path, field_path, value))

    if not found:
        raise ValueError("export does not contain a numeric eos_token_id field")
    if mismatches:
        details = ", ".join(
            f"{path.name}:{field}={value}" for path, field, value in mismatches
        )
        raise ValueError(
            f"export EOS metadata does not match tokenizer ID {eos_token_id}: {details}"
        )
    return eos_token_id


def main() -> None:
    import torch
    from peft import (
        PeftConfig,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )
    from safetensors.torch import load_file
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )
    from transformers.utils import cached_file

    args = parse_args()
    peft_config = PeftConfig.from_pretrained(args.adapter)
    base_name = peft_config.base_model_name_or_path
    if not base_name:
        raise ValueError("adapter_config.json does not declare base_model_name_or_path")

    base_config = AutoConfig.from_pretrained(base_name, trust_remote_code=True)
    model_class = (
        AutoModelForImageTextToText
        if hasattr(base_config, "vision_config")
        else AutoModelForCausalLM
    )
    base = model_class.from_pretrained(
        base_name,
        config=base_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    adapter = get_peft_model(base, peft_config)
    checkpoint = cached_file(args.adapter, "adapter_model.safetensors")
    if checkpoint is None:
        raise FileNotFoundError("adapter_model.safetensors was not found")
    state = load_file(checkpoint, device="cpu")
    expected_keys = set(get_peft_model_state_dict(adapter))
    aligned = align_adapter_state(state, expected_keys)
    set_peft_model_state_dict(adapter, aligned)
    merged = adapter.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    eos_token_id = tokenizer_eos_token_id(tokenizer)
    normalize_model_eos_metadata(merged, eos_token_id)
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True)
    try:
        processor = AutoProcessor.from_pretrained(base_name, trust_remote_code=True)
        processor_tokenizer = getattr(processor, "tokenizer", None)
        if processor_tokenizer is not None:
            processor_tokenizer.eos_token = tokenizer.eos_token
        processor.save_pretrained(args.output)
    except (OSError, ValueError):
        pass
    tokenizer.save_pretrained(args.output)

    exported_tokenizer = AutoTokenizer.from_pretrained(
        args.output, local_files_only=True, trust_remote_code=True
    )
    validate_export_eos(args.output, exported_tokenizer)


if __name__ == "__main__":
    main()

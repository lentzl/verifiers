"""Merge a Hugging Face PEFT adapter into its declared base model."""

from __future__ import annotations

import argparse
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
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_name, trust_remote_code=True).save_pretrained(
        args.output
    )
    try:
        AutoProcessor.from_pretrained(
            base_name, trust_remote_code=True
        ).save_pretrained(args.output)
    except (OSError, ValueError):
        pass


if __name__ == "__main__":
    main()

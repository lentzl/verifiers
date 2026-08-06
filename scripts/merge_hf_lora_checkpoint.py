"""Merge a Hugging Face PEFT adapter into its declared base model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter", help="Local path or Hugging Face adapter repository")
    parser.add_argument("output", type=Path, help="Directory for merged model weights")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    peft_config = PeftConfig.from_pretrained(args.adapter)
    base_name = peft_config.base_model_name_or_path
    if not base_name:
        raise ValueError("adapter_config.json does not declare base_model_name_or_path")

    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
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

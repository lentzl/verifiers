import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "merge_hf_lora_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("merge_hf_lora_checkpoint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_align_adapter_state_translates_prime_qwen35_wrapper():
    source = {
        "model.language_model.layers.0.self_attn.q_proj.lora_A.weight": object(),
        "model.language_model.layers.0.self_attn.q_proj.lora_B.weight": object(),
    }
    expected = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
    }

    aligned = MODULE.align_adapter_state(source, expected)

    assert set(aligned) == expected
    assert aligned[next(key for key in expected if "lora_A" in key)] is next(
        value for key, value in source.items() if "lora_A" in key
    )


def test_align_adapter_state_rejects_incomplete_checkpoint():
    source = {"model.language_model.layers.0.mlp.up_proj.lora_A.weight": object()}
    expected = {
        "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight",
        "base_model.model.model.layers.0.mlp.up_proj.lora_B.weight",
    }

    with pytest.raises(ValueError, match="missing 1 tensors"):
        MODULE.align_adapter_state(source, expected)

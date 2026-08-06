import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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


class ChatMLTokenizer:
    eos_token = "<|im_end|>"
    eos_token_id = 248046

    def encode(self, text, *, add_special_tokens):
        assert text == self.eos_token
        assert add_special_tokens is False
        return [self.eos_token_id]


def test_normalize_model_eos_metadata_uses_tokenized_chat_eos():
    text_config = SimpleNamespace(eos_token_id=248044)
    model = SimpleNamespace(
        config=SimpleNamespace(
            sub_configs={"text_config": object}, text_config=text_config
        ),
        generation_config=SimpleNamespace(eos_token_id=248044),
    )

    eos_token_id = MODULE.tokenizer_eos_token_id(ChatMLTokenizer())
    MODULE.normalize_model_eos_metadata(model, eos_token_id)

    assert text_config.eos_token_id == 248046
    assert model.generation_config.eos_token_id == 248046


def test_validate_export_eos_checks_nested_and_generation_fields(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"eos_token_id": 248046}})
    )
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": 248046})
    )

    assert MODULE.validate_export_eos(tmp_path, ChatMLTokenizer()) == 248046


def test_validate_export_eos_rejects_stale_nested_field(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"eos_token_id": 248044}})
    )
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": 248046})
    )

    with pytest.raises(
        ValueError, match=r"config.json:\$.text_config.eos_token_id=248044"
    ):
        MODULE.validate_export_eos(tmp_path, ChatMLTokenizer())

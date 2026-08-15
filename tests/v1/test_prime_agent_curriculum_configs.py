"""Project curricula stay on the official Prime Agent harness contract."""

import tomllib
from pathlib import Path

import pytest

from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.harnesses.prime_agent import (
    PrimeAgentHarness,
    PrimeAgentHarnessConfig,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
CONFIGS = sorted(
    [CONFIG_DIR / "oolong_synth_v1_luna_n1.toml"]
    + list(CONFIG_DIR.glob("prime_agent_qwen35_*.toml"))
)


def test_prime_agent_supports_required_native_capabilities() -> None:
    assert PrimeAgentHarnessConfig().version == "0.7.3"
    assert PrimeAgentHarness.SUPPORTS_MCP
    assert PrimeAgentHarness.SUPPORTS_RESUME
    assert PrimeAgentHarness.SUPPORTS_SKILLS


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.name)
def test_curriculum_uses_official_prime_agent_contract(path: Path) -> None:
    raw = tomllib.load(path.open("rb"))
    harness = raw["env"]["agent"]["harness"]
    assert harness == {"id": "prime_agent", "version": "0.7.3"}

    config = EvalConfig.model_validate(raw)
    assert isinstance(config.env.agent.harness, PrimeAgentHarnessConfig)

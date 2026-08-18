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
    assert PrimeAgentHarnessConfig().version == "0.7.2-beta.495.1.97b994c"
    assert PrimeAgentHarness.SUPPORTS_MCP
    assert PrimeAgentHarness.SUPPORTS_RESUME
    assert PrimeAgentHarness.SUPPORTS_SKILLS


def test_prime_agent_exposes_native_autonomous_completion_gates() -> None:
    config = PrimeAgentHarnessConfig(
        autonomous=True,
        gates=["python /workspace/completion_gate.py"],
        autonomous_max_continuations=8,
    )

    assert config.autonomous
    assert config.gates == ["python /workspace/completion_gate.py"]
    with pytest.raises(ValueError, match="autonomous options require"):
        PrimeAgentHarnessConfig(gates=["python /workspace/completion_gate.py"])


def test_prime_agent_process_timeout_is_opt_in_and_killable() -> None:
    command = ["prime-agent", "--mode", "acp"]

    assert PrimeAgentHarness(PrimeAgentHarnessConfig())._process_command(command) is command
    harness = PrimeAgentHarness(PrimeAgentHarnessConfig(process_timeout_ms=840_000))
    assert harness._process_command(command) == [
        "timeout",
        "--signal=TERM",
        "--kill-after=10s",
        "840s",
        *command,
    ]
    with pytest.raises(ValueError):
        PrimeAgentHarnessConfig(process_timeout_ms=0)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.name)
def test_curriculum_uses_official_prime_agent_contract(path: Path) -> None:
    raw = tomllib.load(path.open("rb"))
    harness = raw["env"]["agent"]["harness"]
    assert harness == {
        "id": "prime_agent",
        "version": "0.7.2-beta.495.1.97b994c",
    }

    config = EvalConfig.model_validate(raw)
    assert isinstance(config.env.agent.harness, PrimeAgentHarnessConfig)

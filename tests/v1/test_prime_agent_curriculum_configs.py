"""Project curricula stay on the official Prime Agent harness contract."""

import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest

from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.harnesses.prime_agent import (
    PrimeAgentHarness,
    PrimeAgentHarnessConfig,
)
from verifiers.v1.harnesses.prime_agent.harness import ENV_AGENT_DIR

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

    default_harness = PrimeAgentHarness(PrimeAgentHarnessConfig())
    assert default_harness._process_command(command) is command
    assert "cleanup_descendants" not in default_harness._wrapper_script(command)
    assert "exec prime-agent --mode acp" in default_harness._wrapper_script(command)

    harness = PrimeAgentHarness(PrimeAgentHarnessConfig(process_timeout_ms=840_000))
    timed_command = harness._process_command(command)
    assert timed_command == [
        "timeout",
        "--signal=TERM",
        "--kill-after=10s",
        "840s",
        *command,
    ]
    wrapper = harness._wrapper_script(timed_command)
    assert "cleanup_descendants" in wrapper
    assert 'marker = os.environ.get("PRIME_AGENT_CODING_AGENT_DIR")' in wrapper
    assert 'with open(f"/proc/{pid}/environ", "rb")' in wrapper
    assert "signal.SIGTERM" in wrapper
    assert "signal.SIGKILL" in wrapper
    assert "exec timeout" not in wrapper
    with pytest.raises(ValueError):
        PrimeAgentHarnessConfig(process_timeout_ms=0)


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="requires Linux /proc")
def test_prime_agent_process_timeout_kills_detached_descendants(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    child_code = "import time; time.sleep(60)"
    parent_code = (
        "import subprocess,sys,time; "
        f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}],start_new_session=True); "
        f"open({str(child_pid_path)!r},'w').write(str(child.pid)); "
        "time.sleep(60)"
    )
    harness = PrimeAgentHarness(PrimeAgentHarnessConfig(process_timeout_ms=250))
    command = harness._process_command([sys.executable, "-c", parent_code])
    wrapper = tmp_path / "prime-agent"
    wrapper.write_text(harness._wrapper_script(command))
    wrapper.chmod(0o700)

    result = subprocess.run(
        [wrapper],
        env={**os.environ, ENV_AGENT_DIR: str(tmp_path / "agent")},
        timeout=5,
        check=False,
    )

    assert result.returncode == 124
    child_pid = int(child_pid_path.read_text())
    for _ in range(20):
        if not Path(f"/proc/{child_pid}").exists():
            break
        time.sleep(0.1)
    assert not Path(f"/proc/{child_pid}").exists()


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

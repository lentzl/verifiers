from types import SimpleNamespace

from verifiers.v1.harnesses.rlm.harness import RLMHarness, RLMHarnessConfig


async def test_rlm_setup_uses_configured_repository():
    runtime = SimpleNamespace()
    calls = []

    async def run(argv, env):
        calls.append((argv, env))
        return SimpleNamespace(exit_code=0, stderr="")

    runtime.run = run
    harness = RLMHarness(
        RLMHarnessConfig(
            id="rlm",
            repository="https://github.com/example/rlm fork.git",
            version="exact-commit",
        )
    )

    await harness.setup(runtime)

    command = calls[0][0][-1]
    assert "git clone 'https://github.com/example/rlm fork.git' /tmp/rlm" in command
    assert "git -C /tmp/rlm checkout exact-commit" in command

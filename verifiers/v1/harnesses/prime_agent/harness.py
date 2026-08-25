"""Prime Agent over its native ACP mode."""

from __future__ import annotations

import hashlib
import json
import logging
import shlex

from pydantic import Field, model_validator

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harnesses.node import NODE_BIN_DIR, ensure_node
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

INSTALL_URL = "https://pub-728493de92a943e2a9b2d17b4719f318.r2.dev/install.sh"
PRIME_AGENT_DIR = "/var/tmp/vf-prime-agent"
STATE_ROOT = "/tmp/vf-prime-agent-runs"
SKILLS_DIR = ".agents/skills"
PROVIDER = "intercept"
KEY_VAR = "PRIME_AGENT_INTERCEPT_KEY"
ENV_AGENT_DIR = "PRIME_AGENT_CODING_AGENT_DIR"
DEFAULT_VERSION = "0.7.2-beta.495.1.97b994c"

INSTALL = r"""
set -e
export PATH="/var/tmp/vf-node/bin:$PATH"
prefix="$VF_PRIME_AGENT_DIR/$PRIME_AGENT_VERSION"
[ -x "$prefix/bin/prime-agent" ] && exit 0
export NPM_CONFIG_PREFIX="$prefix"
export PRIME_AGENT_BOOTSTRAP_KERNEL_ON_INSTALL=0
curl -fsSL "$VF_PRIME_AGENT_INSTALL_URL" | sh
"""


class PrimeAgentHarnessConfig(HarnessConfig):
    version: str = Field(default=DEFAULT_VERSION, pattern=r"^[A-Za-z0-9._+-]+$")
    """Prime Agent release to install, pinned for reproducibility."""
    autonomous: bool = False
    """Continue until the configured completion gates pass or a limit is reached."""
    gates: list[str] = Field(default_factory=list)
    autonomous_gate_retries: int | None = Field(default=None, strict=True, gt=0)
    autonomous_gate_timeout_ms: int | None = Field(default=None, strict=True, gt=0)
    autonomous_max_continuations: int | None = Field(default=None, strict=True, gt=0)
    autonomous_max_turns: int | None = Field(default=None, strict=True, gt=0)
    autonomous_max_tokens: int | None = Field(default=None, strict=True, gt=0)
    autonomous_timeout_ms: int | None = Field(default=None, strict=True, gt=0)
    process_timeout_ms: int | None = Field(default=None, strict=True, gt=0)
    """Optional hard wall-clock limit for the Prime Agent process."""

    @model_validator(mode="after")
    def validate_autonomous_options(self) -> PrimeAgentHarnessConfig:
        options = (
            self.gates,
            self.autonomous_gate_retries,
            self.autonomous_gate_timeout_ms,
            self.autonomous_max_continuations,
            self.autonomous_max_turns,
            self.autonomous_max_tokens,
            self.autonomous_timeout_ms,
        )
        if not self.autonomous and any(options):
            raise ValueError("autonomous options require autonomous=true")
        if any(not gate.strip() for gate in self.gates):
            raise ValueError("gates must contain non-empty commands")
        return self


class PrimeAgentHarness(ACPHarness[PrimeAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = True
    SUPPORTS_SKILLS = True

    async def setup(self, runtime: Runtime) -> None:
        await self.install_skills(runtime, SKILLS_DIR)
        await ensure_node(runtime)
        logger.info("prime-agent: ensuring %s is installed", self.config.version)
        lock = f"{PRIME_AGENT_DIR}/install.lock"
        guarded = (
            f"mkdir -p {PRIME_AGENT_DIR} && "
            f'"$(command -v flock || command -v lockf)" {lock} '
            f"sh -c {shlex.quote(INSTALL)}"
        )
        result = await runtime.run(
            ["sh", "-c", guarded],
            {
                **self.config.resolved_env,
                "VF_PRIME_AGENT_DIR": PRIME_AGENT_DIR,
                "VF_PRIME_AGENT_INSTALL_URL": INSTALL_URL,
                "PRIME_AGENT_VERSION": self.config.version,
            },
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"prime-agent install failed: {result.stderr.strip()[-500:]}"
            )
        await super().setup(runtime)

    async def prepare_acp(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ACPConfig:
        if self.config.disabled_tools:
            raise ValueError(
                "prime-agent has no per-tool disable flag; its model-facing tool "
                "surface is ipython"
            )

        root = self._root(trace)
        agent_dir = f"{root}/agent"
        created = await runtime.run(
            [
                "mkdir",
                "-p",
                "-m",
                "700",
                root,
                agent_dir,
                f"{root}/tmp",
            ],
            {},
        )
        if created.exit_code != 0:
            raise RuntimeError(
                f"prime-agent state directory failed: {created.stderr.strip()[-500:]}"
            )
        models = self._models(ctx, endpoint)
        models_path = f"{agent_dir}/models.json"
        await runtime.write(models_path, json.dumps(models).encode())
        secured = await runtime.run(["chmod", "600", models_path], {})
        if secured.exit_code != 0:
            raise RuntimeError(
                f"prime-agent model config chmod failed: {secured.stderr.strip()[-500:]}"
            )

        system_prompt, prompt = self.resolve_prompt(data)
        args = [
            self._bin(),
            "--mode",
            "acp",
            "--provider",
            PROVIDER,
            "--model",
            ctx.model,
            "--daemon-socket",
            f"{root}/daemon.sock",
            "--offline",
        ]
        for skill in self.config.skills:
            args += ["--skill", f"{SKILLS_DIR}/{skill.resolve().name}"]
        if system_prompt:
            args += ["--append-system-prompt", system_prompt]
        if self.config.autonomous:
            args.append("--autonomous")
            for gate in self.config.gates:
                args += ["--autonomous-gate", gate]
            limits = {
                "--autonomous-gate-retries": self.config.autonomous_gate_retries,
                "--autonomous-gate-timeout-ms": self.config.autonomous_gate_timeout_ms,
                "--autonomous-max-continuations": self.config.autonomous_max_continuations,
                "--autonomous-max-turns": self.config.autonomous_max_turns,
                "--autonomous-max-tokens": self.config.autonomous_max_tokens,
                "--autonomous-timeout-ms": self.config.autonomous_timeout_ms,
            }
            for flag, value in limits.items():
                if value is not None:
                    args += [flag, str(value)]

        command = self._process_command(args)
        wrapper = f"{root}/prime-agent"
        await runtime.write(
            wrapper,
            self._wrapper_script(command).encode(),
        )
        executable = await runtime.run(["chmod", "700", wrapper], {})
        if executable.exit_code != 0:
            raise RuntimeError(
                f"prime-agent wrapper chmod failed: {executable.stderr.strip()[-500:]}"
            )

        return ACPConfig(
            env=self._env(trace, secret),
            command=[wrapper],
            prompt=prompt,
            allow_empty_tool_reply=True,
        )

    @staticmethod
    def _models(ctx: ModelContext, endpoint: str) -> dict:
        reasoning = ctx.sampling.reasoning_effort not in (
            None,
            "none",
        ) or ctx.model.rsplit("/", 1)[-1].startswith(("gpt-5", "o1", "o3", "o4"))
        return {
            "providers": {
                PROVIDER: {
                    "baseUrl": endpoint,
                    "api": "openai-completions",
                    "apiKey": KEY_VAR,
                    "models": [
                        {
                            "id": ctx.model,
                            "reasoning": reasoning,
                            "input": ["text", "image"],
                            "compat": {"sendSessionAffinityHeaders": True},
                        }
                    ],
                }
            }
        }

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        root = self._root(trace)
        # Closing the ACP process disposes its client-owned daemon session and supervisor. The
        # runtime itself is the backstop if the process crashed; invoking the public `shutdown`
        # command here would stop unrelated agents when a runtime is intentionally borrowed.
        removed = await runtime.run(["rm", "-rf", root], {})
        if removed.exit_code != 0:
            raise RuntimeError(
                f"prime-agent state cleanup failed: {removed.stderr.strip()[-500:]}"
            )

    def _bin(self) -> str:
        return f"{PRIME_AGENT_DIR}/{self.config.version}/bin/prime-agent"

    def _process_command(self, args: list[str]) -> list[str]:
        timeout_ms = self.config.process_timeout_ms
        if timeout_ms is None:
            return args
        duration = f"{timeout_ms / 1000:g}s"
        return [
            "timeout",
            "--signal=TERM",
            "--kill-after=10s",
            duration,
            *args,
        ]

    def _wrapper_script(self, command: list[str]) -> str:
        lines = [
            "#!/bin/sh",
            "set -eu",
            f'export PATH="{NODE_BIN_DIR}:$HOME/.local/bin:$PATH"',
        ]
        invocation = f'{shlex.join(command)} "$@"'
        if self.config.process_timeout_ms is None:
            return "\n".join([*lines, f"exec {invocation}", ""])

        # Prime Agent's daemon can detach from the process group managed by GNU
        # timeout. The per-episode agent directory remains in every descendant's
        # environment, including detached supervisors and kernel fork servers.
        cleanup = f'''cleanup_descendants() {{
python3 - "$$" <<'PY'
import os
import signal
import sys
import time

marker = os.environ.get("{ENV_AGENT_DIR}")
if marker:
    expected = b"{ENV_AGENT_DIR}=" + os.fsencode(marker)
    excluded = {{os.getpid(), int(sys.argv[1])}}

    def matching_pids():
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid in excluded:
                continue
            try:
                with open(f"/proc/{{pid}}/environ", "rb") as handle:
                    environment = handle.read().split(b"\\0")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if expected in environment:
                yield pid

    for sig, delay in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 0.0)):
        for pid in matching_pids():
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if delay:
            time.sleep(delay)
PY
}}
trap cleanup_descendants EXIT
{invocation}
trap - EXIT
cleanup_descendants'''
        return "\n".join([*lines, cleanup, ""])

    @staticmethod
    def _root(trace: Trace) -> str:
        digest = hashlib.sha256(trace.id.encode()).hexdigest()[:16]
        return f"{STATE_ROOT}/{digest}"

    def _env(self, trace: Trace, secret: str) -> dict[str, str]:
        root = self._root(trace)
        return {
            "PAGER": "cat",
            "PYTHONPAGER": "cat",
            **self.config.resolved_env,
            KEY_VAR: secret,
            ENV_AGENT_DIR: f"{root}/agent",
            "TMPDIR": f"{root}/tmp",
            "PRIME_AGENT_TELEMETRY": "0",
        }

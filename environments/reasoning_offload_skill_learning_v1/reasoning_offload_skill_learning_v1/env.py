"""Train portable skill authors on utility measured in fresh agent contexts."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import verifiers.v1 as vf
from verifiers.v1.harnesses.null import NullHarnessConfig
from verifiers.v1.harnesses.rlm import RLMHarnessConfig

from reasoning_offload_skill_learning_v1.skill_package import (
    SkillCandidate,
    SkillPackageError,
    install_candidate,
    parse_candidate,
)

AUTHOR_SYSTEM_PROMPT = (
    "Turn recurring work in past successful examples into one portable Agent Skill. "
    "Do not solve or mention an unseen evaluation task. Return only the requested "
    "package envelope."
)

AUTHOR_PROMPT = """You are the skill-authoring stage of a continual-learning loop.

Below are successful examples from earlier work. Extract one reusable operation that
would reduce reasoning or repetition on future tasks of the same latent kind. Author
an instruction-first skill that can be discovered and used by different coding-agent
harnesses. Packaging for any particular runtime is outside your responsibility.

Past successful examples:
{experiences}

Return exactly one JSON object inside <skill_package>...</skill_package> with these
fields:

- "name": a descriptive kebab-case skill name of at most 64 characters
- "skill_md": a complete SKILL.md beginning with YAML frontmatter containing only a
  matching name and a non-empty description, followed by concise instructions on when
  to use the skill and how to run its bundled script
- "files": a path-to-text object with at least one self-contained Python program under
  scripts/; optional supporting text may live under references/ or assets/

Scripts may use the Python standard library. They must not use networking,
subprocesses, dynamic code execution, or external dependencies. Prefer a general
command-line input contract over constants copied from the examples. Do not emit
pyproject.toml, package imports, runtime entrypoints, or vendor-specific metadata.
"""

CONSUMER_SYSTEM_PROMPT = """A portable Agent Skill is available at {root}/SKILL.md.
Treat it as progressively disclosed instructions: read SKILL.md first, inspect only
the referenced bundled files you need, and execute a script when it is useful. The
skill is advisory; preserve the task's exact requested output contract."""


class SkillAuthorData(vf.TaskData):
    source_task_name: str | None = None


class SkillAuthorTask(vf.Task[SkillAuthorData]):
    pass


def _experience_dict(experience: Any) -> dict[str, Any]:
    if hasattr(experience, "model_dump"):
        return experience.model_dump(mode="json")
    if isinstance(experience, dict):
        return experience
    raise ValueError("skill-learning evidence entries must be objects")


def author_task(task: vf.Task) -> SkillAuthorTask:
    experiences = getattr(task.data, "experiences", ())
    if not experiences:
        raise ValueError(
            "skill learning needs past solved examples on task.data.experiences; "
            "set the taskset's experience_examples above zero"
        )
    evidence = json.dumps(
        [_experience_dict(experience) for experience in experiences],
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    return SkillAuthorTask(
        SkillAuthorData(
            idx=task.data.idx,
            name=f"skill-author-{task.data.name or task.data.idx}",
            prompt=AUTHOR_PROMPT.format(experiences=evidence),
            system_prompt=AUTHOR_SYSTEM_PROMPT,
            source_task_name=task.data.name,
        )
    )


def consumer_task(task: vf.Task, root: str) -> vf.Task:
    system_parts = [task.data.system_prompt, CONSUMER_SYSTEM_PROMPT.format(root=root)]
    data = task.data.model_copy(
        update={"system_prompt": "\n\n".join(part for part in system_parts if part)}
    )
    return type(task)(data, config=task.config)


def _same_consumer_policy(left: vf.AgentConfig, right: vf.AgentConfig) -> bool:
    fields = (
        "harness",
        "runtime",
        "model",
        "client",
        "sampling",
        "max_turns",
        "max_input_tokens",
        "max_output_tokens",
        "max_total_tokens",
    )
    return all(getattr(left, field) == getattr(right, field) for field in fields)


class ReasoningOffloadSkillLearningConfig(vf.EnvConfig):
    author: vf.AgentConfig = vf.AgentConfig(
        harness=NullHarnessConfig(id="null"),
        max_turns=1,
        max_output_tokens=4096,
        max_total_tokens=6144,
    )
    skill_user: vf.AgentConfig = vf.AgentConfig(
        harness=RLMHarnessConfig(id="rlm", version="main", max_depth=0),
        runtime=vf.DockerConfig(image="python:3.11-slim"),
        max_turns=8,
        max_total_tokens=8192,
    )
    baseline_user: vf.AgentConfig = vf.AgentConfig(
        harness=RLMHarnessConfig(id="rlm", version="main", max_depth=0),
        runtime=vf.DockerConfig(image="python:3.11-slim"),
        max_turns=8,
        max_total_tokens=8192,
    )


class ReasoningOffloadSkillLearningEnv(vf.Env[ReasoningOffloadSkillLearningConfig]):
    def __init__(self, config: ReasoningOffloadSkillLearningConfig) -> None:
        super().__init__(config)
        for role in ("skill_user", "baseline_user"):
            if isinstance(getattr(self.config, role).runtime, vf.SubprocessConfig):
                raise ValueError(f"{role} must run in an isolated container runtime")
        if not _same_consumer_policy(config.skill_user, config.baseline_user):
            raise ValueError(
                "skill_user and baseline_user must have identical policies, limits, "
                "harnesses, and runtimes"
            )

    async def setup(self, agents: vf.Agents) -> None:
        agents.skill_user.trainable = False
        agents.baseline_user.trainable = False

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        baseline_future = asyncio.create_task(agents.baseline_user.run(task))
        author = await agents.author.run(author_task(task))

        candidate: SkillCandidate | None = None
        try:
            candidate = parse_candidate(author.last_reply)
        except SkillPackageError as exc:
            author.info["skill_package_error"] = str(exc)
            author.info["feedback"] = f"The proposed skill package was rejected: {exc}"

        if candidate is not None:
            author.info["skill_name"] = candidate.name
            try:
                async with agents.skill_user.provision(task) as runtime:
                    root = await install_candidate(runtime, candidate)
                    skill_trace = await agents.skill_user.run(
                        consumer_task(task, root), runtime=runtime
                    )
                    skill_trace.info["installed_skill"] = candidate.name
                    skill_trace.info["installed_skill_root"] = root
            except Exception as exc:
                author.info["skill_install_error"] = f"{type(exc).__name__}: {exc}"
                author.info["feedback"] = (
                    "The package passed static validation but could not be installed "
                    "or exercised by the downstream agent: "
                    f"{type(exc).__name__}: {exc}"
                )

        await baseline_future

    @staticmethod
    def _score(trace: vf.Trace | None) -> float:
        if trace is None:
            return 0.0
        reward = trace.rewards.get("exact_match")
        return reward.score if reward is not None else 0.0

    @staticmethod
    def _consulted(trace: vf.Trace | None) -> float:
        if trace is None:
            return 0.0
        root = trace.info.get("installed_skill_root")
        name = trace.info.get("installed_skill")
        if not isinstance(root, str) or not isinstance(name, str):
            return 0.0
        needles = (root, "SKILL.md", f"agent-skills/{name}")
        for message in trace.assistant_messages:
            for call in message.tool_calls or []:
                try:
                    arguments = json.loads(call.arguments)
                except json.JSONDecodeError:
                    arguments = call.arguments
                payload = json.dumps(arguments, sort_keys=True)
                if any(needle in payload for needle in needles):
                    return 1.0
        return 0.0

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        by_agent = {trace.agent.name: trace for trace in episode.traces}
        author = by_agent.get("author")
        baseline = by_agent.get("baseline_user")
        skill_user = by_agent.get("skill_user")
        if author is None or baseline is None:
            raise ValueError("skill-learning episode is missing its author or baseline")

        baseline_score = self._score(baseline)
        skill_score = self._score(skill_user)
        utility = skill_score - baseline_score
        package_valid = float(
            "skill_name" in author.info and "skill_install_error" not in author.info
        )
        consulted = self._consulted(skill_user)
        author.record_metrics(
            {
                "skill_package_valid": package_valid,
                "skill_consulted": consulted,
                "baseline_correct": baseline_score,
                "skill_user_correct": skill_score,
                "marginal_skill_utility": utility,
            }
        )
        author.record_reward("skill_utility", utility)

        if "feedback" in author.info or utility > 0:
            return
        if skill_score < baseline_score:
            author.info["feedback"] = (
                "The candidate made a fresh downstream run worse than the matched "
                "no-skill baseline. Generalize the operation and avoid changing the "
                "requested output contract."
            )
        elif not consulted:
            author.info["feedback"] = (
                "The package installed, but the fresh agent did not consult it. Make "
                "SKILL.md's trigger and script interface easier to discover and use."
            )
        else:
            author.info["feedback"] = (
                "The skill was consulted but produced no marginal correctness over the "
                "matched baseline. Encode a more general, reliably useful operation."
            )

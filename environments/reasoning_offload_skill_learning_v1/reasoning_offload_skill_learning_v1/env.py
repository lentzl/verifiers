"""Train portable skill authors on utility measured in fresh agent contexts."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import verifiers.v1 as vf
from reasoning_offload_skill_learning_v1.skill_package import (
    SkillCandidate,
    SkillPackageError,
    install_candidate,
    parse_candidate,
)
from verifiers.v1.harnesses.null import NullHarnessConfig
from verifiers.v1.harnesses.rlm import RLMHarnessConfig

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
command-line input contract over constants copied from the examples. Every script
must accept one or more task file paths as command-line arguments and read those files
itself; do not require serialized file contents as arguments. Do not emit
pyproject.toml, package imports, runtime entrypoints, or vendor-specific metadata.

Separate the reusable operation from incidental literals in the examples. Do not
bake observed thresholds, identifiers, labels, or answer values into the program.
Match the demonstrated top-level file shape rather than inventing wrapper keys. If a
future invocation needs a scalar stated in its task prompt but absent from its input
file, expose that scalar as a documented command-line argument. The script must print
only the answer payload in the same serialization shown by the successful examples,
without tags, labels, commentary, or debug output.

Follow this package shape literally. Replace the example values and program, retain
the hyphenated name and the exact three-line YAML frontmatter delimiters, and mention
the script path in the SKILL.md body:

<skill_package>
{{"name":"example-skill","skill_md":"---\\nname: example-skill\\ndescription: Use this skill when an example operation is needed.\\n---\\n\\nFrom this skill directory, run `python scripts/solve.py INPUT_PATH` and return the requested result.","files":{{"scripts/solve.py":"import sys\\nfrom pathlib import Path\\n\\nprint(Path(sys.argv[1]).read_text())\\n"}}}}
</skill_package>
"""

CONSUMER_SYSTEM_PROMPT = """This run evaluates a portable Agent Skill available at
{root}/SKILL.md. You must inspect SKILL.md before solving the task, then inspect only
the referenced bundled files you need and execute or adapt its script. Resolve every
bundled relative path against {root}, never the task working directory. Pass task
input files to scripts as absolute paths. Begin with a genuine IPython tool call that
reads the skill. After that result, do not explain, restate, or inspect the script:
your next assistant turn must be a genuine IPython tool call that executes the
referenced script. Do not print tool-call JSON or a code fence. Preserve the task's
exact requested output contract."""

CONSUMER_PROMPT = """Before solving, make one genuine IPython tool call to read
{root}/SKILL.md. Do not print code or tool-call JSON as text. Then follow the skill
and resolve its bundled script paths against {root}. Pass task input files as absolute
paths. Immediately execute the referenced script in your next genuine IPython tool
call; do not describe or reproduce its code. Then preserve the requested answer
format.

{prompt}"""


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
        update={
            "prompt": CONSUMER_PROMPT.format(root=root, prompt=task.data.prompt),
            "system_prompt": "\n\n".join(part for part in system_parts if part),
        }
    )
    return type(task)(data, config=task.config)


def utility_tasks(task: vf.Task) -> tuple[vf.Task, ...]:
    examples = getattr(task.data, "utility_examples", ())
    if not examples:
        return (task,)

    fields = type(task.data).model_fields
    tasks = []
    for index, example in enumerate(examples):
        updates = {
            "idx": task.data.idx * 100 + index,
            "name": example.name,
            "prompt": example.prompt,
            "system_prompt": example.system_prompt,
            "template_variant": example.template_variant,
            "answer": example.answer,
            "files": example.files,
            "experiences": (),
            "utility_examples": (),
        }
        data = task.data.model_copy(
            update={key: value for key, value in updates.items() if key in fields}
        )
        tasks.append(type(task)(data, config=task.config))
    return tuple(tasks)


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


def _score_bounded_failure(trace: vf.Trace, index: int) -> vf.Trace:
    """Turn a downstream budget failure into a scored miss, not an episode fault."""
    trace.info["utility_case_index"] = index
    if trace.ok:
        return trace
    trace.info["utility_arm_errors"] = [
        error.model_dump(mode="json") for error in trace.errors
    ]
    trace.errors.clear()
    trace.ok = True
    return trace


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
                raise TypeError(f"{role} must run in an isolated container runtime")
        if not _same_consumer_policy(config.skill_user, config.baseline_user):
            raise ValueError(
                "skill_user and baseline_user must have identical policies, limits, "
                "harnesses, and runtimes"
            )

    async def setup(self, agents: vf.Agents) -> None:
        agents.skill_user.trainable = False
        agents.baseline_user.trainable = False

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        panel = utility_tasks(task)

        async def run_baseline(index: int, utility_task: vf.Task) -> vf.Trace:
            trace = await agents.baseline_user.run(utility_task)
            return _score_bounded_failure(trace, index)

        baseline_futures = [
            asyncio.create_task(run_baseline(index, utility_task))
            for index, utility_task in enumerate(panel)
        ]
        author = await agents.author.run(author_task(task))

        candidate: SkillCandidate | None = None
        try:
            candidate = parse_candidate(author.last_reply)
        except SkillPackageError as exc:
            author.info["skill_package_error"] = str(exc)
            author.info["feedback"] = f"The proposed skill package was rejected: {exc}"

        install_errors = []
        if candidate is not None:
            author.info["skill_name"] = candidate.name

            async def run_skill(index: int, utility_task: vf.Task) -> None:
                try:
                    async with agents.skill_user.provision(utility_task) as runtime:
                        root = await install_candidate(runtime, candidate)
                        skill_trace = await agents.skill_user.run(
                            consumer_task(utility_task, root), runtime=runtime
                        )
                        skill_trace.info["installed_skill"] = candidate.name
                        skill_trace.info["installed_skill_root"] = root
                        _score_bounded_failure(skill_trace, index)
                except Exception as exc:  # noqa: BLE001 - score candidate failures
                    install_errors.append(f"{type(exc).__name__}: {exc}")

            await asyncio.gather(
                *(
                    run_skill(index, utility_task)
                    for index, utility_task in enumerate(panel)
                )
            )
            if install_errors:
                detail = "; ".join(install_errors)
                author.info["skill_install_error"] = detail
                author.info["feedback"] = (
                    "The package passed static validation but could not be installed "
                    f"or exercised by every downstream agent: {detail}"
                )

        await asyncio.gather(*baseline_futures)

    @staticmethod
    def _by_case(traces: list[vf.Trace]) -> dict[int, vf.Trace]:
        indexed = {}
        for fallback, trace in enumerate(traces):
            index = trace.info.get("utility_case_index", fallback)
            if isinstance(index, int):
                indexed[index] = trace
        return indexed

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

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
        by_agent = episode.by_agent
        authors = by_agent.get("author", [])
        baselines = self._by_case(by_agent.get("baseline_user", []))
        skill_users = self._by_case(by_agent.get("skill_user", []))
        if len(authors) != 1 or not baselines:
            raise ValueError("skill-learning episode is missing its author or baseline")
        author = authors[0]

        observed_size = max((*baselines, *skill_users), default=-1) + 1
        case_indices = list(range(max(len(utility_tasks(task)), observed_size)))
        baseline_scores = [self._score(baselines[index]) for index in case_indices]
        skill_scores = [self._score(skill_users.get(index)) for index in case_indices]
        baseline_score = self._mean(baseline_scores)
        skill_score = self._mean(skill_scores)
        raw_utility = self._mean(
            [skill - baseline for skill, baseline in zip(skill_scores, baseline_scores)]
        )
        baseline_complete = len(baselines) == len(case_indices) and all(
            index in baselines and "utility_arm_errors" not in baselines[index].info
            for index in case_indices
        )
        utility = raw_utility if baseline_complete else 0.0
        package_valid = float(
            "skill_name" in author.info and "skill_install_error" not in author.info
        )
        consulted = self._mean(
            [self._consulted(skill_users.get(index)) for index in case_indices]
        )
        author.record_metrics(
            {
                "skill_package_valid": package_valid,
                "skill_consulted": consulted,
                "utility_panel_size": float(len(case_indices)),
                "baseline_correct": baseline_score,
                "skill_user_correct": skill_score,
                "marginal_skill_utility": utility,
                "utility_evaluation_complete": float(baseline_complete),
            }
        )
        author.record_reward("skill_utility", utility)

        if not baseline_complete:
            author.info["feedback"] = (
                "The matched no-skill evaluation did not complete, so this candidate "
                "received no learning signal."
            )
            return
        if "feedback" in author.info or utility > 0:
            return
        if skill_score < baseline_score:
            author.info["feedback"] = (
                "The candidate made fresh downstream runs worse on average than the "
                "matched no-skill baselines. Generalize the operation and avoid "
                "changing the requested output contract."
            )
        elif not consulted:
            author.info["feedback"] = (
                "The package installed, but the fresh agents did not consult it. Make "
                "SKILL.md's trigger and script interface easier to discover and use."
            )
        else:
            author.info["feedback"] = (
                "The skill was consulted but produced no average marginal correctness "
                "over the matched baselines. Encode a more general, reliably useful "
                "operation."
            )

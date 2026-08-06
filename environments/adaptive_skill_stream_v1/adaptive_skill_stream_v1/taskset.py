"""Prime Agent streams that couple online adaptation with portable skill lifecycle."""

from __future__ import annotations

import json
import re
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

import verifiers.v1 as vf
from adaptive_skill_stream_v1.generators import (
    EVAL_VARIANTS,
    FAMILIES,
    TRAIN_VARIANTS,
    Family,
    generate,
)
from verifiers.v1.harnesses.prime_agent.harness import PrimeAgentHarness

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
FRONTMATTER_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
WORKSPACE = "/workspace"
SKILLS_ROOT = f"{WORKSPACE}/.agents/skills"
SYSTEM_PROMPT = (
    "Treat this conversation as one continuing project session. Each user message "
    "introduces exactly one current batch; future batches do not exist until after "
    "you answer, so never search for them. Use the coding environment and persistent "
    "IPython state to solve the current batch, with at most three concise IPython "
    "calls before answering. IPython is the only tool: never call tools named answer "
    "or tool. Prefer an applicable installed skill over recreating it. Keep "
    "task-specific discoveries local. Only after a procedure has demonstrated stable "
    "reuse, promote it as a standard Agent Skill package under "
    "/workspace/.agents/skills/<name>/SKILL.md so future OpenAI, Anthropic, and "
    "Prime-compatible agents can discover it. End every batch by replying with the "
    "JSON value only, without XML tags, Markdown, analysis, or explanation."
)


class SkillStreamRound(BaseModel):
    instruction: str
    answer: Any
    files: dict[str, str]


class AdaptiveSkillStreamData(vf.TaskData):
    family: Family
    template_variant: int
    project_context: str
    rounds: tuple[SkillStreamRound, ...]


def _extract_answer(reply: str) -> object | None:
    matches = ANSWER_PATTERN.findall(reply)
    candidate = matches[-1].strip() if matches else reply.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _round_score(family: Family, actual: object, expected: object) -> float:
    if family == "installed":
        if isinstance(actual, dict):
            actual_mapping = cast(dict[str, object], actual)
            if set(actual_mapping) == {"result"}:
                actual = actual_mapping["result"]
        if not isinstance(actual, dict) or not isinstance(expected, dict):
            return 0.0
        if not expected:
            return float(not actual)
        return sum(actual.get(key) == value for key, value in expected.items()) / len(
            expected
        )

    if family == "stable":
        if not isinstance(actual, list) or not isinstance(expected, list):
            return 0.0
        actual_names = {
            item
            if isinstance(item, str)
            else item.get("name")
            if isinstance(item, dict) and isinstance(item.get("name"), str)
            else None
            for item in actual
        } - {None}
        expected_names = {item for item in expected if isinstance(item, str)}
        union = actual_names | expected_names
        return len(actual_names & expected_names) / len(union) if union else 1.0

    if not isinstance(actual, list) or not isinstance(expected, list):
        return 0.0
    if not expected:
        return float(not actual)
    matches = sum(
        actual[index] == value
        for index, value in enumerate(expected)
        if index < len(actual)
    )
    return matches / len(expected)


def _ipython_cells(trace: vf.Trace) -> list[str]:
    cells = []
    for message in trace.assistant_messages:
        for call in message.tool_calls or []:
            if call.name != "ipython":
                continue
            try:
                arguments = json.loads(call.arguments)
            except json.JSONDecodeError:
                continue
            if isinstance(code := arguments.get("code"), str):
                cells.append(code)
    return cells


def _frontmatter(markdown: str) -> tuple[dict[str, str], str] | None:
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    try:
        end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration:
        return None
    metadata: dict[str, str] = {}
    for line in lines[1:end]:
        key, separator, value = line.partition(":")
        if not separator or not key.strip() or not value.strip():
            return None
        metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, "\n".join(lines[end + 1 :]).strip()


def _valid_frontier_skill(markdown: str) -> bool:
    parsed = _frontmatter(markdown)
    if parsed is None:
        return False
    metadata, body = parsed
    if set(metadata) != {"name", "description"}:
        return False
    if not FRONTMATTER_NAME.fullmatch(metadata["name"]):
        return False
    if len(metadata["description"]) < 20:
        return False
    normalized = f"{metadata['description']}\n{body}".lower()
    required_terms = ("frontier", "completed", "blocked", "requirements")
    return all(term in normalized for term in required_terms)


async def _read_json_if_present(runtime: vf.Runtime, path: str) -> dict:
    exists = await runtime.run(["test", "-f", path], {})
    if exists.exit_code != 0:
        return {}
    try:
        value = json.loads((await runtime.read(path, max_bytes=128_000)).decode())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _entry_counts(state: dict) -> tuple[int, dict[str, int]]:
    entries = state.get("entries", {})
    if not isinstance(entries, dict):
        return 0, {}
    by_kind = {
        kind: len(values)
        for kind, values in entries.items()
        if isinstance(kind, str) and isinstance(values, dict)
    }
    return sum(by_kind.values()), by_kind


def _add_entry_counts(
    total: int,
    by_kind: dict[str, int],
    state: dict,
) -> tuple[int, dict[str, int]]:
    state_total, state_kinds = _entry_counts(state)
    merged = dict(by_kind)
    for kind, count in state_kinds.items():
        merged[kind] = merged.get(kind, 0) + count
    return total + state_total, merged


class AdaptiveSkillStreamTask(vf.Task[AdaptiveSkillStreamData]):
    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        created = await runtime.run(
            ["mkdir", "-p", f"{WORKSPACE}/inbox", SKILLS_ROOT], {}
        )
        if created.exit_code != 0:
            raise RuntimeError(f"workspace setup failed: {created.stderr[-500:]}")
        await runtime.write(
            f"{WORKSPACE}/PROJECT_CONTEXT.md", self.data.project_context.encode()
        )

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        listed = await runtime.run(
            [
                "find",
                SKILLS_ROOT,
                "-mindepth",
                "2",
                "-maxdepth",
                "2",
                "-type",
                "f",
                "-name",
                "SKILL.md",
                "-print",
            ],
            {},
        )
        paths = sorted(path for path in listed.stdout.splitlines() if path.strip())
        valid_paths = []
        for path in paths[:8]:
            try:
                markdown = (await runtime.read(path, max_bytes=64_000)).decode()
            except (UnicodeDecodeError, OSError):
                continue
            if _valid_frontier_skill(markdown):
                valid_paths.append(path)

        global_path = PrimeAgentHarness.harness_state_path(trace)
        state_files = await runtime.run(
            [
                "find",
                PrimeAgentHarness.agent_dir(trace),
                "-type",
                "f",
                "-path",
                "*/harness/harness_state.json",
                "-print",
            ],
            {},
        )
        local_count, global_count = 0, 0
        local_kinds: dict[str, int] = {}
        global_kinds: dict[str, int] = {}
        for path in sorted(state_files.stdout.splitlines()):
            state = await _read_json_if_present(runtime, path)
            if path == global_path:
                global_count, global_kinds = _add_entry_counts(
                    global_count, global_kinds, state
                )
            else:
                local_count, local_kinds = _add_entry_counts(
                    local_count, local_kinds, state
                )
        trace.info["adaptive_skill_artifacts"] = {
            "portable_skill_paths": paths,
            "valid_frontier_skill_paths": valid_paths,
            "local_harness_entries": local_count,
            "local_harness_kinds": local_kinds,
            "global_harness_entries": global_count,
            "global_harness_kinds": global_kinds,
        }

    @vf.reward(weight=0.2)
    async def skill_lifecycle(self, trace: vf.Trace) -> float:
        artifacts = trace.info.get("adaptive_skill_artifacts", {})
        paths = artifacts.get("portable_skill_paths", [])
        valid_paths = artifacts.get("valid_frontier_skill_paths", [])
        used_installed = any(
            "portable_record_normalization" in cell for cell in _ipython_cells(trace)
        )
        if self.data.family == "installed":
            return float(used_installed and not paths)
        if self.data.family == "stable":
            return float(bool(valid_paths))
        return float(not paths)

    @vf.metric
    async def adaptation_behavior(self, trace: vf.Trace) -> dict[str, float]:
        artifacts = trace.info.get("adaptive_skill_artifacts", {})
        cells = _ipython_cells(trace)
        return {
            "used_installed_skill": float(
                any("portable_record_normalization" in cell for cell in cells)
            ),
            "requested_refinement": float(any("refine.run" in cell for cell in cells)),
            "used_harness_crud": float(any("rlm.harness." in cell for cell in cells)),
            "portable_skill_created": float(
                bool(artifacts.get("portable_skill_paths"))
            ),
            "portable_skill_valid": float(
                bool(artifacts.get("valid_frontier_skill_paths"))
            ),
            "local_harness_entries": float(artifacts.get("local_harness_entries", 0)),
            "global_harness_entries": float(artifacts.get("global_harness_entries", 0)),
        }


def _round_prompt(
    task: AdaptiveSkillStreamTask,
    round_idx: int,
    previous_correct: bool | None,
) -> str:
    current = task.data.rounds[round_idx]
    parts = []
    if previous_correct is None:
        parts.append(
            "Related batches will arrive sequentially, one per user message. Only the "
            "current batch file exists now. Read /workspace/PROJECT_CONTEXT.md before "
            "deciding what should remain local and what, if anything, is durable."
        )
    else:
        verdict = "passed" if previous_correct else "failed"
        parts.append(
            f"The previous batch {verdict} validation. The expected answer is not "
            "exposed; use the outcome to retain, revise, or discard your approach."
        )
    if round_idx == len(task.data.rounds) - 1:
        parts.append(
            "This is the final transfer batch. Leave the workspace useful for future "
            "sessions only when the evidence warrants it."
        )
    files = ", ".join(sorted(current.files))
    parts.append(f"Batch {round_idx + 1} files: {files}. {current.instruction}")
    return "\n\n".join(parts)


class AdaptiveSkillStreamEnv(vf.SingleAgentEnv):
    """Drive four batches through one Prime Agent session and persistent kernel."""

    async def run(self, task, agents):
        correct: list[bool] = []
        scores: list[float] = []
        replies: list[object | None] = []
        async with agents.agent.provision(task) as runtime:
            async with agents.agent.interaction(task, runtime=runtime) as interaction:
                previous: bool | None = None
                for round_idx, current in enumerate(task.data.rounds):
                    for path, content in current.files.items():
                        await runtime.write(path, content.encode())
                    segment = await interaction.turn(
                        _round_prompt(task, round_idx, previous)
                    )
                    if segment.terminated:
                        break
                    actual = _extract_answer(segment.last_reply)
                    score = _round_score(task.data.family, actual, current.answer)
                    passed = score == 1.0
                    replies.append(actual)
                    correct.append(passed)
                    scores.append(score)
                    previous = passed
                interaction.trace.info["adaptive_skill_stream"] = {
                    "correct": correct,
                    "scores": scores,
                    "replies": replies,
                    "rounds_completed": len(correct),
                }
            trace = interaction.trace

        total_rounds = len(task.data.rounds)
        padded = [*correct, *([False] * (total_rounds - len(correct)))]
        padded_scores = [*scores, *([0.0] * (total_rounds - len(scores)))]
        accuracy = sum(padded_scores) / total_rounds
        trace.record_reward("stream_accuracy", accuracy)
        trace.record_metric("first_batch_correct", float(padded[0]))
        trace.record_metric(
            "late_batch_accuracy", sum(padded_scores[1:]) / (total_rounds - 1)
        )
        trace.record_metric("final_batch_correct", float(padded[-1]))
        trace.record_metric("completed_stream", float(len(correct) == total_rounds))


class AdaptiveSkillStreamConfig(vf.TasksetConfig):
    split: Literal["train", "eval"] = "train"
    families: tuple[Family, ...] = Field(FAMILIES, min_length=1)
    instances_per_template: int = Field(4, ge=1)
    seed: int = 20260806


class AdaptiveSkillStreamTaskset(
    vf.Taskset[AdaptiveSkillStreamTask, AdaptiveSkillStreamConfig]
):
    def load(self) -> list[AdaptiveSkillStreamTask]:
        variants = TRAIN_VARIANTS if self.config.split == "train" else EVAL_VARIANTS
        tasks = []
        idx = 0
        for instance in range(self.config.instances_per_template):
            for variant in variants:
                for family in self.config.families:
                    generated = generate(family, variant, instance, self.config.seed)
                    tasks.append(
                        AdaptiveSkillStreamTask(
                            AdaptiveSkillStreamData(
                                idx=idx,
                                name=f"{family}-v{variant}-i{instance}",
                                prompt=None,
                                system_prompt=SYSTEM_PROMPT,
                                workdir=WORKSPACE,
                                family=family,
                                template_variant=variant,
                                project_context=generated.project_context,
                                rounds=tuple(
                                    SkillStreamRound(
                                        instruction=round_.instruction,
                                        answer=round_.answer,
                                        files=round_.files,
                                    )
                                    for round_ in generated.rounds
                                ),
                            ),
                            self.config.task,
                        )
                    )
                    idx += 1
        return tasks


__all__ = [
    "AdaptiveSkillStreamConfig",
    "AdaptiveSkillStreamData",
    "AdaptiveSkillStreamEnv",
    "AdaptiveSkillStreamTask",
    "AdaptiveSkillStreamTaskset",
    "SkillStreamRound",
]

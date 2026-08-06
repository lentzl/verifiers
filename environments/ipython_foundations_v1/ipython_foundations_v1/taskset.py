"""Prime Agent streams for persistent IPython fundamentals."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal

from pydantic import BaseModel, Field

import verifiers.v1 as vf
from ipython_foundations_v1.generators import (
    EVAL_VARIANTS,
    FAMILIES,
    TRAIN_VARIANTS,
    Family,
    generate,
)
from verifiers.v1.types import content_text

WORKSPACE = "/workspace"
SYSTEM_PROMPT = (
    "Use IPython as one persistent notebook. Variables, imports, functions, and loaded "
    "objects survive across IPython calls and later user messages in this session. An "
    "empty tool result after an assignment normally means the assignment succeeded; it "
    "does not mean the cell should be repeated. Inspect or use the retained variable in "
    "a later call. After a traceback, preserve any state created before the failing "
    "statement and correct only the failed operation. End each request with the requested "
    "JSON value only, without Markdown or explanation."
)


class FoundationRound(BaseModel):
    instruction: str
    explicit_operation: str
    answer: Any
    files: dict[str, str]
    remove_after: tuple[str, ...] = ()


class IpythonFoundationsData(vf.TaskData):
    family: Family
    template_variant: int
    instruction_level: Literal["standard", "explicit"] = "standard"
    state_variable: str
    rounds: tuple[FoundationRound, ...]


@dataclass
class IpythonEvent:
    code: str
    call_id: str
    segment: int
    output: str = ""


def _extract_json(reply: str) -> object | None:
    try:
        return json.loads(reply.strip())
    except json.JSONDecodeError:
        return None


def _partial_score(actual: object, expected: object) -> float:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return 0.0
        return (
            sum(actual.get(key) == value for key, value in expected.items())
            / len(expected)
            if expected
            else float(not actual)
        )
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return 0.0
        return (
            sum(
                actual[index] == value
                for index, value in enumerate(expected)
                if index < len(actual)
            )
            / len(expected)
            if expected
            else float(not actual)
        )
    return float(actual == expected)


def _ipython_events(trace: vf.Trace) -> list[IpythonEvent]:
    events: list[IpythonEvent] = []
    by_call_id: dict[str, IpythonEvent] = {}
    segment = -1
    for node in trace.nodes:
        message = node.message
        if isinstance(message, vf.UserMessage):
            segment += 1
        elif isinstance(message, vf.AssistantMessage):
            for call in message.tool_calls or []:
                if call.name != "ipython":
                    continue
                try:
                    arguments = json.loads(call.arguments)
                except json.JSONDecodeError:
                    continue
                code = arguments.get("code")
                if not isinstance(code, str):
                    continue
                event = IpythonEvent(code=code, call_id=call.id, segment=segment)
                events.append(event)
                by_call_id[call.id] = event
        elif isinstance(message, vf.ToolMessage) and (
            event := by_call_id.get(message.tool_call_id)
        ):
            event.output = content_text(message.content)
    return events


def _name_contexts(code: str, name: str) -> tuple[bool, bool]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, False
    assigned = any(
        isinstance(node, ast.Name)
        and node.id == name
        and isinstance(node.ctx, ast.Store)
        for node in ast.walk(tree)
    )
    loaded = any(
        isinstance(node, ast.Name)
        and node.id == name
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(tree)
    )
    return assigned, loaded


def _behavior(trace: vf.Trace, family: Family, state_variable: str) -> dict[str, float]:
    events = _ipython_events(trace)
    contexts = [_name_contexts(event.code, state_variable) for event in events]
    assignment_indices = [
        index for index, (assigned, _) in enumerate(contexts) if assigned
    ]
    first_assignment = assignment_indices[0] if assignment_indices else None
    later_reuse = next(
        (
            index
            for index, (_, loaded) in enumerate(contexts)
            if loaded and first_assignment is not None and index > first_assignment
        ),
        None,
    )
    silent_assignment_recovered = bool(
        first_assignment is not None
        and not events[first_assignment].output.strip()
        and later_reuse is not None
    )
    cross_turn_reuse = bool(
        first_assignment is not None
        and later_reuse is not None
        and events[later_reuse].segment > events[first_assignment].segment
    )
    error_index = next(
        (
            index
            for index, event in enumerate(events)
            if "Traceback" in event.output or "Error" in event.output
        ),
        None,
    )
    recovered_after_error = bool(
        error_index is not None
        and any(
            loaded and index > error_index for index, (_, loaded) in enumerate(contexts)
        )
    )
    repeated = sum(
        left.code.strip() == right.code.strip() for left, right in pairwise(events)
    )
    family_aligned = {
        "assignment": silent_assignment_recovered,
        "state": cross_turn_reuse,
        "recovery": recovered_after_error,
    }[family]
    return {
        "ipython_calls": float(len(events)),
        "state_assigned": float(bool(assignment_indices)),
        "state_reused": float(later_reuse is not None),
        "silent_assignment_recovered": float(silent_assignment_recovered),
        "cross_turn_state_reused": float(cross_turn_reuse),
        "error_observed": float(error_index is not None),
        "recovered_after_error": float(recovered_after_error),
        "identical_consecutive_calls": float(repeated),
        "process_aligned": float(family_aligned and repeated == 0),
    }


class IpythonFoundationsTask(vf.Task[IpythonFoundationsData]):
    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        result = await runtime.run(["mkdir", "-p", f"{WORKSPACE}/inbox"], {})
        if result.exit_code != 0:
            raise RuntimeError(f"workspace setup failed: {result.stderr[-500:]}")

    @vf.reward(weight=0.2)
    async def notebook_semantics(self, trace: vf.Trace) -> float:
        behavior = _behavior(trace, self.data.family, self.data.state_variable)
        return behavior["process_aligned"]

    @vf.metric
    async def notebook_behavior(self, trace: vf.Trace) -> dict[str, float]:
        return _behavior(trace, self.data.family, self.data.state_variable)


def _round_prompt(
    task: IpythonFoundationsTask,
    round_idx: int,
    previous_correct: bool | None,
) -> str:
    current = task.data.rounds[round_idx]
    parts = []
    if previous_correct is None:
        parts.append(
            "This is one continuing notebook session. Complete the current request, "
            "then retain useful IPython state for later requests."
        )
    else:
        verdict = "passed" if previous_correct else "failed"
        parts.append(
            f"The previous answer {verdict} validation. The expected value is not "
            "revealed; continue from the existing notebook state."
        )
    parts.append(current.instruction)
    if task.data.instruction_level == "explicit":
        parts.append(f"Foundation exercise: {current.explicit_operation}")
    return "\n\n".join(parts)


class IpythonFoundationsEnv(vf.SingleAgentEnv):
    """Drive dependent requests through one Prime Agent session and IPython kernel."""

    async def run(self, task, agents):
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
                    actual = _extract_json(segment.last_reply)
                    score = _partial_score(actual, current.answer)
                    scores.append(score)
                    replies.append(actual)
                    previous = score == 1.0
                    if current.remove_after:
                        removed = await runtime.run(
                            ["rm", "-f", *current.remove_after], {}
                        )
                        if removed.exit_code != 0:
                            raise RuntimeError(
                                f"source cleanup failed: {removed.stderr[-500:]}"
                            )
                interaction.trace.info["ipython_foundations"] = {
                    "scores": scores,
                    "replies": replies,
                    "rounds_completed": len(scores),
                }
            trace = interaction.trace

        total_rounds = len(task.data.rounds)
        padded = [*scores, *([0.0] * (total_rounds - len(scores)))]
        trace.record_reward("stream_accuracy", sum(padded) / total_rounds)
        trace.record_metric("first_request_correct", float(padded[0] == 1.0))
        trace.record_metric("final_request_correct", float(padded[-1] == 1.0))
        trace.record_metric("completed_stream", float(len(scores) == total_rounds))


class IpythonFoundationsConfig(vf.TasksetConfig):
    split: Literal["train", "eval"] = "train"
    families: tuple[Family, ...] = Field(FAMILIES, min_length=1)
    instruction_level: Literal["standard", "explicit"] = "standard"
    instances_per_template: int = Field(4, ge=1)
    seed: int = 20260806


class IpythonFoundationsTaskset(
    vf.Taskset[IpythonFoundationsTask, IpythonFoundationsConfig]
):
    def load(self) -> list[IpythonFoundationsTask]:
        variants = TRAIN_VARIANTS if self.config.split == "train" else EVAL_VARIANTS
        tasks = []
        idx = 0
        for instance in range(self.config.instances_per_template):
            for variant in variants:
                for family in self.config.families:
                    generated = generate(family, variant, instance, self.config.seed)
                    tasks.append(
                        IpythonFoundationsTask(
                            IpythonFoundationsData(
                                idx=idx,
                                name=f"{family}-v{variant}-i{instance}",
                                prompt=None,
                                system_prompt=SYSTEM_PROMPT,
                                workdir=WORKSPACE,
                                family=family,
                                template_variant=variant,
                                instruction_level=self.config.instruction_level,
                                state_variable=generated.state_variable,
                                rounds=tuple(
                                    FoundationRound(
                                        instruction=round_.instruction,
                                        explicit_operation=round_.explicit_operation,
                                        answer=round_.answer,
                                        files=round_.files,
                                        remove_after=round_.remove_after,
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
    "FoundationRound",
    "IpythonFoundationsConfig",
    "IpythonFoundationsData",
    "IpythonFoundationsEnv",
    "IpythonFoundationsTask",
    "IpythonFoundationsTaskset",
]

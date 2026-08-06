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
    RECOVERY_EVAL_VARIANTS,
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
    "JSON value only, without Markdown or explanation. For subprocess results, always "
    "inspect returncode, stdout, and stderr; check=False does not make a nonzero return "
    "code successful. Use the reported error to change a failed operation instead of "
    "repeating it, and choose fallbacks from evidence rather than bypassing structured "
    "formats with raw-byte decoding."
)
GUIDED_OPERATIONS = {
    "completion": (
        "Use one IPython call, observe its non-empty result, then return that result "
        "immediately without another tool call."
    ),
    "assignment": (
        "Use two separate IPython calls. First assign the requested variable and accept "
        "the empty result as success. In the next call, read that variable without "
        "reassigning it, then return the computed JSON value."
    ),
    "state": (
        "Answer the current request, retain the requested variable, and read that same "
        "variable in later requests without reloading or reconstructing the source."
    ),
    "recovery": (
        "Run the failing operation once, inspect the real traceback, then change only "
        "the failed operation while reusing state created before the error."
    ),
    "subprocess": (
        "Inspect returncode, stdout, and stderr, then revise the failed command from "
        "that evidence without repeating it or decoding raw document bytes."
    ),
}

PDFTOTEXT_COMPAT = r"""#!/usr/bin/env python3
import base64
import sys
from pathlib import Path

args = sys.argv[1:]
if "-text" in args:
    sys.stderr.write("I/O Error: Couldn't open file '-text': No such file or directory.\n")
    raise SystemExit(1)

positional = [arg for arg in args if arg != "-layout"]
if not positional:
    sys.stderr.write("Syntax Error: No input file specified.\n")
    raise SystemExit(1)

source = Path(positional[0])
if not source.is_file():
    sys.stderr.write(f"I/O Error: Couldn't open file '{source}': No such file or directory.\n")
    raise SystemExit(1)

text = base64.b64decode(source.read_bytes()).decode()
output = positional[1] if len(positional) > 1 else str(source.with_suffix(".txt"))
if output == "-":
    sys.stdout.write(text)
else:
    Path(output).write_text(text)
"""


class FoundationRound(BaseModel):
    instruction: str
    explicit_operation: str
    answer: Any
    files: dict[str, str]
    remove_after: tuple[str, ...] = ()
    recovery_kind: str | None = None


class IpythonFoundationsData(vf.TaskData):
    family: Family
    template_variant: int
    instruction_level: Literal["standard", "guided", "explicit"] = "standard"
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


def _code_attributes(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    return {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}


def _uses_stdout_convention(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    strings = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    return "pdftotext" in code and "-" in strings


def _uses_raw_pdf_fallback(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    if attributes & {"read_bytes", "decode"}:
        return True
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "open":
            continue
        mode = node.args[1] if len(node.args) > 1 else None
        mode = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "mode"),
            mode,
        )
        if (
            isinstance(mode, ast.Constant)
            and isinstance(mode.value, str)
            and "b" in mode.value
        ):
            return True
    return False


def _behavior(
    trace: vf.Trace,
    family: Family,
    state_variable: str,
    expected_segments: int | None = None,
) -> dict[str, float]:
    events = _ipython_events(trace)
    contexts = [_name_contexts(event.code, state_variable) for event in events]
    attributes = [_code_attributes(event.code) for event in events]
    assignment_indices = [
        index for index, (assigned, _) in enumerate(contexts) if assigned
    ]
    first_assignment = assignment_indices[0] if assignment_indices else None
    later_reuse = next(
        (
            index
            for index, (assigned, loaded) in enumerate(contexts)
            if (
                loaded
                and not assigned
                and first_assignment is not None
                and index > first_assignment
            )
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
    active_segments = {event.segment for event in events}
    recovery_error_indices = {
        segment: next(
            (
                index
                for index, event in enumerate(events)
                if event.segment == segment
                and ("Traceback" in event.output or "Error" in event.output)
            ),
            None,
        )
        for segment in active_segments
    }
    recovery_repaired_segments = {
        segment
        for segment, segment_error_index in recovery_error_indices.items()
        if segment_error_index is not None
        and any(
            index > segment_error_index
            and event.segment == segment
            and contexts[index][1]
            and event.code.strip() != events[segment_error_index].code.strip()
            for index, event in enumerate(events)
        )
    }
    required_recovery_segments = (
        set(range(expected_segments))
        if expected_segments is not None
        else active_segments
    )
    recovery_rounds_aligned = bool(
        required_recovery_segments
        and required_recovery_segments <= recovery_repaired_segments
    )
    repeated = sum(
        left.code.strip() == right.code.strip() for left, right in pairwise(events)
    )
    subprocess_observed = any(
        {"returncode", "stdout", "stderr"} <= event_attributes
        for event_attributes in attributes
    )
    subprocess_failure_index = next(
        (
            index
            for index, event in enumerate(events)
            if "returncode" in event.code
            and "stderr" in event.code
            and "I/O Error" in event.output
        ),
        None,
    )
    subprocess_failures = sum(
        "pdftotext" in event.code and "I/O Error" in event.output for event in events
    )
    subprocess_failure_retries = max(subprocess_failures - 1, 0)
    subprocess_revised = bool(
        subprocess_failure_index is not None
        and any(
            index > subprocess_failure_index
            and "pdftotext" in event.code
            and event.code.strip() != events[subprocess_failure_index].code.strip()
            for index, event in enumerate(events)
        )
    )
    cli_stdout_used = any(
        _uses_stdout_convention(event.code)
        for index, event in enumerate(events)
        if subprocess_failure_index is not None and index > subprocess_failure_index
    )
    raw_pdf_fallback = any(_uses_raw_pdf_fallback(event.code) for event in events)
    successful_result_observed = any(
        event.output.strip()
        and "Traceback" not in event.output
        and "Error" not in event.output
        for event in events
    )
    observed_segments = max(len(active_segments), 1)
    expected_rounds = expected_segments or observed_segments
    efficient_call_budget = {
        "completion": expected_rounds,
        "assignment": 2 * expected_rounds,
        "state": expected_rounds,
        "recovery": 3 * expected_rounds,
        "subprocess": expected_rounds,
    }[family]
    call_efficiency = min(efficient_call_budget / max(len(events), 1), 1.0)
    recovery_round_coverage = (
        len(recovery_repaired_segments) / len(required_recovery_segments)
        if required_recovery_segments
        else 0.0
    )
    subprocess_progress = (
        sum(
            (
                later_reuse is not None,
                subprocess_observed,
                subprocess_revised,
                cli_stdout_used,
            )
        )
        / 4
    )
    if raw_pdf_fallback:
        subprocess_progress *= 0.5
    if subprocess_failure_retries:
        subprocess_progress /= subprocess_failure_retries + 1
    family_progress = {
        "completion": float(successful_result_observed),
        "assignment": float(silent_assignment_recovered),
        "state": float(cross_turn_reuse),
        "recovery": recovery_round_coverage,
        "subprocess": subprocess_progress,
    }[family]
    process_score = family_progress * call_efficiency / (repeated + 1)
    family_aligned = {
        "completion": successful_result_observed and len(events) == expected_rounds,
        "assignment": silent_assignment_recovered,
        "state": cross_turn_reuse,
        "recovery": recovery_rounds_aligned,
        "subprocess": (
            later_reuse is not None
            and subprocess_observed
            and subprocess_revised
            and cli_stdout_used
            and not raw_pdf_fallback
            and subprocess_failure_retries == 0
        ),
    }[family]
    return {
        "ipython_calls": float(len(events)),
        "state_assigned": float(bool(assignment_indices)),
        "state_reused": float(later_reuse is not None),
        "silent_assignment_recovered": float(silent_assignment_recovered),
        "cross_turn_state_reused": float(cross_turn_reuse),
        "error_observed": float(error_index is not None),
        "recovered_after_error": float(recovered_after_error),
        "recovery_error_segments": float(
            sum(index is not None for index in recovery_error_indices.values())
        ),
        "recovery_repaired_segments": float(len(recovery_repaired_segments)),
        "recovery_round_coverage": recovery_round_coverage,
        "subprocess_result_observed": float(subprocess_observed),
        "subprocess_operation_revised": float(subprocess_revised),
        "subprocess_failure_retries": float(subprocess_failure_retries),
        "cli_stdout_convention_used": float(cli_stdout_used),
        "raw_pdf_fallback_used": float(raw_pdf_fallback),
        "identical_consecutive_calls": float(repeated),
        "successful_result_observed": float(successful_result_observed),
        "ipython_call_efficiency": call_efficiency,
        "process_score": process_score,
        "process_aligned": float(
            family_aligned and repeated == 0 and len(events) <= efficient_call_budget
        ),
    }


class IpythonFoundationsTask(vf.Task[IpythonFoundationsData]):
    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        result = await runtime.run(
            ["mkdir", "-p", f"{WORKSPACE}/inbox", f"{WORKSPACE}/bin"], {}
        )
        if result.exit_code != 0:
            raise RuntimeError(f"workspace setup failed: {result.stderr[-500:]}")
        await runtime.write(f"{WORKSPACE}/bin/pdftotext", PDFTOTEXT_COMPAT.encode())
        executable = await runtime.run(
            ["chmod", "755", f"{WORKSPACE}/bin/pdftotext"], {}
        )
        if executable.exit_code != 0:
            raise RuntimeError(f"extractor setup failed: {executable.stderr[-500:]}")

    @vf.reward(weight=1.0)
    async def notebook_semantics(self, trace: vf.Trace) -> float:
        behavior = _behavior(
            trace,
            self.data.family,
            self.data.state_variable,
            expected_segments=len(self.data.rounds),
        )
        return behavior["process_score"]

    @vf.metric
    async def notebook_behavior(self, trace: vf.Trace) -> dict[str, float]:
        return _behavior(
            trace,
            self.data.family,
            self.data.state_variable,
            expected_segments=len(self.data.rounds),
        )


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
    if task.data.instruction_level == "guided":
        parts.append(f"Foundation hint: {GUIDED_OPERATIONS[task.data.family]}")
    elif task.data.instruction_level == "explicit":
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
                    "recovery_kinds": [
                        round_.recovery_kind
                        for round_ in task.data.rounds
                        if round_.recovery_kind is not None
                    ],
                }
            trace = interaction.trace

        total_rounds = len(task.data.rounds)
        padded = [*scores, *([0.0] * (total_rounds - len(scores)))]
        trace.record_reward("stream_accuracy", sum(padded) / total_rounds, weight=0.5)
        trace.record_metric("first_request_correct", float(padded[0] == 1.0))
        trace.record_metric("final_request_correct", float(padded[-1] == 1.0))
        trace.record_metric("completed_stream", float(len(scores) == total_rounds))


class IpythonFoundationsConfig(vf.TasksetConfig):
    split: Literal["train", "eval"] = "train"
    families: tuple[Family, ...] = Field(FAMILIES, min_length=1)
    instruction_level: Literal["standard", "guided", "explicit"] = "standard"
    instances_per_template: int = Field(4, ge=1)
    rounds_per_task: int | None = Field(None, ge=1)
    seed: int = 20260806


class IpythonFoundationsTaskset(
    vf.Taskset[IpythonFoundationsTask, IpythonFoundationsConfig]
):
    def load(self) -> list[IpythonFoundationsTask]:
        variants = TRAIN_VARIANTS if self.config.split == "train" else EVAL_VARIANTS
        templates = [
            (family, variant) for variant in variants for family in self.config.families
        ]
        if self.config.split == "eval" and "recovery" in self.config.families:
            templates.extend(
                ("recovery", variant)
                for variant in RECOVERY_EVAL_VARIANTS
                if variant not in variants
            )
        tasks = []
        idx = 0
        for instance in range(self.config.instances_per_template):
            for family, variant in templates:
                generated = generate(family, variant, instance, self.config.seed)
                rounds = generated.rounds[: self.config.rounds_per_task]
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
                                    recovery_kind=round_.recovery_kind,
                                )
                                for round_ in rounds
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

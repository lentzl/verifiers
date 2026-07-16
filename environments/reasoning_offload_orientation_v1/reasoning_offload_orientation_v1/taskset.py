"""File-backed tasks that measure basic reasoning offload into the RLM harness."""

from __future__ import annotations

import ast
import json
import re
from typing import Literal

from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.types import content_text

from reasoning_offload_orientation_v1.generators import (
    EVAL_VARIANTS,
    FAMILIES,
    TRAIN_VARIANTS,
    Family,
    generate,
)

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
FAILURE_MARKERS = ("FAILED", "Traceback", "Exception", "Error:")
INCORRECT_ANSWER_FEEDBACK = (
    "The submitted answer is incorrect. Re-check the task using the available "
    "environment and verify the result before answering again."
)


def _answer(text: str) -> str:
    matches = ANSWER_PATTERN.findall(text)
    return matches[-1].strip() if matches else text.strip()


def _canonical_answer(family: Family, value: str) -> str:
    if family == "helper":
        return ",".join(part.strip() for part in value.split(","))
    return value


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


def _python_trees(cells: list[str]) -> list[ast.AST]:
    trees = []
    for cell in cells:
        try:
            trees.append(ast.parse(cell))
        except SyntaxError:
            continue
    return trees


def _state_reused(trees: list[ast.AST]) -> bool:
    assigned: set[str] = set()
    for tree in trees:
        loaded = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        if assigned & loaded:
            return True
        assigned.update(
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        )
    return False


def _helper_behavior(trees: list[ast.AST]) -> tuple[bool, bool]:
    definitions: set[str] = set()
    calls: set[str] = set()
    for tree in trees:
        definitions.update(
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        )
        calls.update(
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        )
    return bool(definitions), bool(definitions & calls)


class ReasoningOffloadOrientationData(vf.TaskData):
    family: Family
    template_variant: int
    answer: str
    files: dict[str, str]


class ReasoningOffloadOrientationTask(vf.Task[ReasoningOffloadOrientationData]):
    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        for path, content in self.data.files.items():
            await runtime.write(path, content.encode())

    @vf.reward(weight=1.0)
    async def exact_match(self, trace: vf.Trace) -> float:
        actual = _canonical_answer(self.data.family, _answer(trace.last_reply))
        expected = _canonical_answer(self.data.family, self.data.answer)
        correct = actual == expected
        if not correct:
            trace.info["feedback"] = INCORRECT_ANSWER_FEEDBACK
        return float(correct)

    @vf.metric
    async def offload_behavior(self, trace: vf.Trace) -> dict[str, float]:
        cells = _ipython_cells(trace)
        trees = _python_trees(cells)
        helper_defined, helper_called = _helper_behavior(trees)
        outputs = [content_text(message.content) for message in trace.tool_messages]
        failure_observed = any(
            marker in output for output in outputs for marker in FAILURE_MARKERS
        )
        verification_observed = any("VERIFIED" in output for output in outputs)
        recovered = False
        saw_failure = False
        for node in trace.nodes:
            message = node.message
            if isinstance(message, vf.ToolMessage):
                saw_failure |= any(
                    marker in content_text(message.content)
                    for marker in FAILURE_MARKERS
                )
            elif saw_failure and isinstance(message, vf.AssistantMessage):
                recovered |= any(
                    call.name == "ipython" for call in message.tool_calls or []
                )

        used_ipython = bool(cells)
        state_reused = _state_reused(trees)
        aligned = {
            "direct": not used_ipython,
            "inspection": used_ipython,
            "state": state_reused,
            "helper": helper_called,
            "verification": verification_observed,
            "repair": failure_observed and recovered and verification_observed,
        }[self.data.family]
        return {
            "used_ipython": float(used_ipython),
            "ipython_calls": float(len(cells)),
            "state_reused": float(state_reused),
            "helper_defined": float(helper_defined),
            "helper_called": float(helper_called),
            "failure_observed": float(failure_observed),
            "recovered_after_feedback": float(recovered),
            "verification_observed": float(verification_observed),
            "process_aligned": float(aligned),
        }


class ReasoningOffloadOrientationConfig(vf.TasksetConfig):
    split: Literal["train", "eval"] = "train"
    """Generator variants 0-3 train; held-out variants 4-5 evaluate."""
    families: tuple[Family, ...] = Field(FAMILIES, min_length=1)
    """Task families included in this curriculum stage."""
    instances_per_template: int = Field(4, ge=1)
    seed: int = 20260715


class ReasoningOffloadOrientationTaskset(
    vf.Taskset[ReasoningOffloadOrientationTask, ReasoningOffloadOrientationConfig]
):
    def load(self) -> list[ReasoningOffloadOrientationTask]:
        variants = TRAIN_VARIANTS if self.config.split == "train" else EVAL_VARIANTS
        tasks = []
        idx = 0
        for instance in range(self.config.instances_per_template):
            for variant in variants:
                for family in self.config.families:
                    generated = generate(family, variant, instance, self.config.seed)
                    tasks.append(
                        ReasoningOffloadOrientationTask(
                            ReasoningOffloadOrientationData(
                                idx=idx,
                                name=f"{family}-v{variant}-i{instance}",
                                prompt=generated.prompt,
                                family=family,
                                template_variant=variant,
                                answer=generated.answer,
                                files=generated.files,
                            ),
                            self.config.task,
                        )
                    )
                    idx += 1
        return tasks

"""File-backed tasks that measure basic reasoning offload into an agent runtime."""

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
INCORRECT_ANSWER_FEEDBACK: dict[Family, str] = {
    "direct": (
        "The submitted answer is incorrect. Copy the requested literal exactly and "
        "return only that value in the answer tags."
    ),
    "inspection": (
        "The submitted answer is incorrect. Read the exact file path named in the "
        "prompt; if a lookup fails, inspect the working tree and retry instead of stopping."
    ),
    "state": (
        "The submitted answer is incorrect. Inspect the actual keys in each JSONL "
        "record, reuse the loaded records and computed state, and verify the requested aggregate."
    ),
    "helper": (
        "The submitted answer is incorrect. Apply the requested rule through one reusable "
        "helper for every item, then verify the deduplicated and sorted result."
    ),
    "module": (
        "The submitted answer is incorrect. Import transform from inputs.operation and call "
        "that provided function on the loaded target instead of recreating or guessing it."
    ),
    "verification": (
        "The submitted answer is incorrect. VERIFIED establishes that candidate.py passes "
        "the cases; import its transform and call it on target.json rather than inventing a transform."
    ),
    "repair": (
        "The submitted answer is incorrect. Run the checker from the workspace root, repair "
        "buggy.py without changing the checker, rerun until VERIFIED, then call the repaired transform."
    ),
}
ORIENTATION_SYSTEM_PROMPT = (
    "Use the coding environment as a persistent workspace. Keep every assistant turn "
    "minimal. Produce "
    "either exactly one IPython tool call or the final <answer> value, never a narrated "
    "plan. Use file paths exactly as written. After a tool error, inspect or correct the "
    "operation in the next tool call instead of stopping. Once the result is known, answer "
    "immediately without explaining it."
)
EXPLICIT_INSTRUCTIONS: dict[Family, str] = {
    "state": (
        "Orientation hint: in one short IPython call, load inputs/events.jsonl into an "
        "events variable and display events[0] to identify the exact schema. In the next "
        "call, derive balances from the retained events using event['delta']."
    ),
    "verification": (
        "Orientation hint: first run python3 inputs/check.py. After VERIFIED, import "
        "transform from inputs.candidate, load inputs/target.json, call transform on "
        "that loaded value, and return the result. Do not recreate the transform."
    ),
    "repair": (
        "Orientation hint: do not list files or narrate. First run python3 inputs/check.py, "
        "then display inputs/buggy.py. In the following IPython call, write the corrected "
        "source back with `from pathlib import Path; "
        "Path('inputs/buggy.py').write_text(corrected_source)`; defining or testing a "
        "replacement only in IPython does not repair the file. Rerun the checker, and after "
        "VERIFIED import or reload transform from inputs.buggy and call it on the JSON target."
    ),
}


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


def _module_reused(trees: list[ast.AST]) -> bool:
    module_imported = any(
        (
            isinstance(node, ast.ImportFrom)
            and node.module in {"inputs.operation", "operation"}
            and any(alias.name == "transform" for alias in node.names)
        )
        or (
            isinstance(node, ast.Import)
            and any(
                alias.name in {"inputs.operation", "operation"} for alias in node.names
            )
        )
        for tree in trees
        for node in ast.walk(tree)
    )
    transform_called = any(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "transform")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "transform")
        )
        for tree in trees
        for node in ast.walk(tree)
    )
    return module_imported and transform_called


def _literal_path(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Path"
        and node.args
    ):
        return _literal_path(node.args[0])
    return None


def _mutates_buggy_file(code: str) -> bool:
    normalized = code.replace('"', "'")
    shell_writes = (
        "sed -i" in code and "inputs/buggy.py" in code,
        "inputs/buggy.py" in code and "> inputs/buggy.py" in normalized,
        "inputs/buggy.py" in code and "tee inputs/buggy.py" in normalized,
    )
    if any(shell_writes):
        return True

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "open" and node.args:
                path = _literal_path(node.args[0])
                mode_node = node.args[1] if len(node.args) > 1 else None
                for keyword in node.keywords:
                    if keyword.arg == "mode":
                        mode_node = keyword.value
                mode = _literal_path(mode_node) if mode_node is not None else "r"
                if (
                    path == "inputs/buggy.py"
                    and mode
                    and any(marker in mode for marker in ("w", "a", "x", "+"))
                ):
                    return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"write_text", "write_bytes"}:
                if _literal_path(node.func.value) == "inputs/buggy.py":
                    return True
            if node.func.attr == "open" and node.args:
                path = _literal_path(node.func.value)
                mode = _literal_path(node.args[0])
                if (
                    path == "inputs/buggy.py"
                    and mode
                    and any(marker in mode for marker in ("w", "a", "x", "+"))
                ):
                    return True
    return False


def _repair_progress(trace: vf.Trace) -> tuple[bool, bool]:
    saw_failure = False
    mutated_after_failure = False
    verified_after_mutation = False
    for node in trace.nodes:
        message = node.message
        if isinstance(message, vf.ToolMessage):
            output = content_text(message.content)
            if mutated_after_failure and "VERIFIED" in output:
                verified_after_mutation = True
            if any(marker in output for marker in FAILURE_MARKERS):
                saw_failure = True
        elif saw_failure and isinstance(message, vf.AssistantMessage):
            for call in message.tool_calls or []:
                if call.name != "ipython":
                    continue
                try:
                    arguments = json.loads(call.arguments)
                except json.JSONDecodeError:
                    continue
                if isinstance(code := arguments.get("code"), str):
                    mutated_after_failure |= _mutates_buggy_file(code)
    return mutated_after_failure, verified_after_mutation


class ReasoningOffloadOrientationData(vf.TaskData):
    family: Family
    template_variant: int
    instruction_level: Literal["standard", "explicit"] = "standard"
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
            trace.info["feedback"] = INCORRECT_ANSWER_FEEDBACK[self.data.family]
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
        module_reused = _module_reused(trees)
        file_mutated_after_feedback, verified_after_repair = _repair_progress(trace)
        aligned = {
            "direct": not used_ipython,
            "inspection": used_ipython,
            "state": state_reused,
            "helper": helper_called,
            "module": module_reused,
            "verification": verification_observed,
            "repair": (
                failure_observed
                and file_mutated_after_feedback
                and verified_after_repair
            ),
        }[self.data.family]
        return {
            "used_ipython": float(used_ipython),
            "ipython_calls": float(len(cells)),
            "state_reused": float(state_reused),
            "helper_defined": float(helper_defined),
            "helper_called": float(helper_called),
            "module_reused": float(module_reused),
            "failure_observed": float(failure_observed),
            "recovered_after_feedback": float(recovered),
            "verification_observed": float(verification_observed),
            "file_mutated_after_feedback": float(file_mutated_after_feedback),
            "verified_after_repair": float(verified_after_repair),
            "process_aligned": float(aligned),
        }


class ReasoningOffloadOrientationConfig(vf.TasksetConfig):
    split: Literal["train", "eval"] = "train"
    """Generator variants 0-3 train; held-out variants 4-5 evaluate."""
    families: tuple[Family, ...] = Field(FAMILIES, min_length=1)
    """Task families included in this curriculum stage."""
    instruction_level: Literal["standard", "explicit"] = "standard"
    """Use explicit state and feedback-operation hints for early orientation training."""
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
                    prompt = generated.prompt
                    if hint := EXPLICIT_INSTRUCTIONS.get(family):
                        if self.config.instruction_level == "explicit":
                            prompt = f"{hint} {prompt}"
                    name_level = (
                        "-explicit"
                        if self.config.instruction_level == "explicit"
                        else ""
                    )
                    tasks.append(
                        ReasoningOffloadOrientationTask(
                            ReasoningOffloadOrientationData(
                                idx=idx,
                                name=(f"{family}{name_level}-v{variant}-i{instance}"),
                                prompt=prompt,
                                system_prompt=ORIENTATION_SYSTEM_PROMPT,
                                family=family,
                                template_variant=variant,
                                instruction_level=self.config.instruction_level,
                                answer=generated.answer,
                                files=generated.files,
                            ),
                            self.config.task,
                        )
                    )
                    idx += 1
        return tasks

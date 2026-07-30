"""Manifest-driven operation composition through the RLM harness."""

from __future__ import annotations

import ast
import json
import re
from typing import Literal

from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.types import content_text

from reasoning_offload_composition_v1.generators import (
    EVAL_VARIANTS,
    TRAIN_VARIANTS,
    generate,
)

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
INCORRECT_ANSWER_FEEDBACK = (
    "The submitted answer is incorrect. Run the checker, repair inputs/operations.py "
    "without changing the checker or cases, rerun until VERIFIED, then inspect the "
    "manifest and apply every repaired operation to the retained target in order."
)
COMPOSITION_SYSTEM_PROMPT = (
    "For this reasoning-offload task, keep every assistant turn minimal. Produce either "
    "exactly one IPython tool call or the final <answer> value, never a narrated plan. "
    "Retain useful values in the persistent session and reuse provided implementations "
    "instead of reproducing them."
)


def _answer(text: str) -> str:
    matches = ANSWER_PATTERN.findall(text)
    return matches[-1].strip() if matches else text.strip()


def _canonical_json(value: str) -> str:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"))


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


def _provided_operations_reused(trees: list[ast.AST]) -> bool:
    module_aliases: set[str] = set()
    imported_functions: set[str] = set()
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in {"inputs.operations", "operations"}:
                        module_aliases.add(alias.asname or alias.name.split(".")[-1])
            elif isinstance(node, ast.ImportFrom) and node.module in {
                "inputs.operations",
                "operations",
            }:
                imported_functions.update(
                    alias.asname or alias.name for alias in node.names
                )

    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute):
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id in module_aliases
                ):
                    return True
            elif isinstance(node.func, ast.Name) and node.func.id in imported_functions:
                return True
            elif (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in module_aliases
            ):
                return True
    return False


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


def _mutates_operations_file(code: str) -> bool:
    normalized = code.replace('"', "'")
    shell_writes = (
        "sed -i" in code and "inputs/operations.py" in code,
        "inputs/operations.py" in code and "> inputs/operations.py" in normalized,
        "inputs/operations.py" in code and "tee inputs/operations.py" in normalized,
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
                    path == "inputs/operations.py"
                    and mode
                    and any(marker in mode for marker in ("w", "a", "x", "+"))
                ):
                    return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"write_text", "write_bytes"}:
                if _literal_path(node.func.value) == "inputs/operations.py":
                    return True
    return False


def _pipeline_repair_progress(trace: vf.Trace) -> tuple[bool, bool, bool]:
    saw_failure = False
    mutated_after_failure = False
    verified_after_mutation = False
    for node in trace.nodes:
        message = node.message
        if isinstance(message, vf.ToolMessage):
            output = content_text(message.content)
            if mutated_after_failure and "VERIFIED" in output:
                verified_after_mutation = True
            if "FAILED" in output or "Traceback" in output:
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
                    mutated_after_failure |= _mutates_operations_file(code)
    return saw_failure, mutated_after_failure, verified_after_mutation


class ReasoningOffloadCompositionData(vf.TaskData):
    template_variant: int
    answer: str
    files: dict[str, str]


class ReasoningOffloadCompositionTask(vf.Task[ReasoningOffloadCompositionData]):
    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        for path, content in self.data.files.items():
            await runtime.write(path, content.encode())

    @vf.reward(weight=1.0)
    async def exact_match(self, trace: vf.Trace) -> float:
        actual = _canonical_json(_answer(trace.last_reply))
        correct = actual == self.data.answer
        if not correct:
            trace.info["feedback"] = INCORRECT_ANSWER_FEEDBACK
        return float(correct)

    @vf.metric
    async def offload_behavior(self, trace: vf.Trace) -> dict[str, float]:
        cells = _ipython_cells(trace)
        trees = _python_trees(cells)
        manifest_inspected = any("pipeline.json" in cell for cell in cells)
        target_loaded = any("target.json" in cell for cell in cells)
        operations_reused = _provided_operations_reused(trees)
        state_reused = _state_reused(trees)
        failure_observed, file_mutated, verified_after_repair = (
            _pipeline_repair_progress(trace)
        )
        return {
            "used_ipython": float(bool(cells)),
            "ipython_calls": float(len(cells)),
            "manifest_inspected": float(manifest_inspected),
            "target_loaded": float(target_loaded),
            "state_reused": float(state_reused),
            "provided_operations_reused": float(operations_reused),
            "failure_observed": float(failure_observed),
            "file_mutated_after_feedback": float(file_mutated),
            "verified_after_repair": float(verified_after_repair),
            "process_aligned": float(
                manifest_inspected
                and target_loaded
                and operations_reused
                and failure_observed
                and file_mutated
                and verified_after_repair
            ),
        }


class ReasoningOffloadCompositionConfig(vf.TasksetConfig):
    split: Literal["train", "eval"] = "train"
    """Generator variants 0-3 train; held-out variants 4-5 evaluate."""
    instances_per_template: int = Field(4, ge=1)
    seed: int = 20260722


class ReasoningOffloadCompositionTaskset(
    vf.Taskset[ReasoningOffloadCompositionTask, ReasoningOffloadCompositionConfig]
):
    def load(self) -> list[ReasoningOffloadCompositionTask]:
        variants = TRAIN_VARIANTS if self.config.split == "train" else EVAL_VARIANTS
        tasks = []
        for instance in range(self.config.instances_per_template):
            for variant in variants:
                generated = generate(variant, instance, self.config.seed)
                tasks.append(
                    ReasoningOffloadCompositionTask(
                        ReasoningOffloadCompositionData(
                            idx=len(tasks),
                            name=f"composition-v{variant}-i{instance}",
                            prompt=generated.prompt,
                            system_prompt=COMPOSITION_SYSTEM_PROMPT,
                            template_variant=variant,
                            answer=generated.answer,
                            files=generated.files,
                        ),
                        self.config.task,
                    )
                )
        return tasks

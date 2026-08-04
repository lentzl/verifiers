"""Verifier taskset for frozen-policy persistent skill acquisition."""

from __future__ import annotations

import ast
import json
import re

from pydantic import BaseModel, Field

import verifiers.v1 as vf
from reasoning_offload_skill_acquisition_v1.generators import (
    FAMILIES,
    SPLIT_VARIANTS,
    Family,
    Split,
    generate,
)

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
SKILL_ACQUISITION_SYSTEM_PROMPT = (
    "Solve the task through the provided coding environment and keep every assistant "
    "turn minimal. "
    "Produce either exactly one IPython tool call or the final <answer> value. Retain "
    "useful state across calls. Installed skills are optional: inspect and use one only "
    "when it is genuinely relevant, and otherwise solve the task directly."
)
INCORRECT_ANSWER_FEEDBACK = (
    "The submitted value is incorrect. Reinspect the input schema and the requested "
    "ordering or aggregation rule, reuse already loaded state, and verify the exact "
    "serialized result before answering again."
)


def _answer(text: str) -> str:
    matches = ANSWER_PATTERN.findall(text)
    return matches[-1].strip() if matches else text.strip()


def _canonical(value: str) -> str:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value.strip()
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


def _state_reused(cells: list[str]) -> bool:
    assigned: set[str] = set()
    for cell in cells:
        try:
            tree = ast.parse(cell)
        except SyntaxError:
            continue
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


class SkillExperience(BaseModel):
    """One previously solved task available to a separate skill-authoring seat."""

    prompt: str
    answer: str
    files: dict[str, str]


class SkillUtilityExample(BaseModel):
    """One hidden task used to measure whether an authored skill transfers."""

    name: str
    prompt: str
    system_prompt: str
    template_variant: int
    answer: str
    files: dict[str, str]


class ReasoningOffloadSkillAcquisitionData(vf.TaskData):
    family: Family
    split: Split
    template_variant: int
    answer: str
    files: dict[str, str]
    experiences: tuple[SkillExperience, ...] = ()
    utility_examples: tuple[SkillUtilityExample, ...] = ()


class ReasoningOffloadSkillAcquisitionTask(
    vf.Task[ReasoningOffloadSkillAcquisitionData]
):
    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        for path, content in self.data.files.items():
            await runtime.write(path, content.encode())

    @vf.reward(weight=1.0)
    async def exact_match(self, trace: vf.Trace) -> float:
        correct = _canonical(_answer(trace.last_reply)) == _canonical(self.data.answer)
        if not correct:
            trace.info["feedback"] = INCORRECT_ANSWER_FEEDBACK
        return float(correct)

    @vf.metric
    async def offload_behavior(self, trace: vf.Trace) -> dict[str, float]:
        cells = _ipython_cells(trace)
        return {
            "used_ipython": float(bool(cells)),
            "ipython_calls": float(len(cells)),
            "state_reused": float(_state_reused(cells)),
        }


class ReasoningOffloadSkillAcquisitionConfig(vf.TasksetConfig):
    split: Split = "discovery"
    """Discovery proposes skills; validation gates them; test remains untouched."""

    families: tuple[Family, ...] = Field(FAMILIES, min_length=1)
    """Hidden evaluator strata. Family names never appear in task prompts."""

    instances_per_template: int = Field(1, ge=1)
    variants_per_family: int | None = Field(None, ge=1)
    """Limit author prompts per family while retaining the full utility panel."""
    experience_examples: int = Field(3, ge=0, le=len(SPLIT_VARIANTS["discovery"]))
    """Solved discovery examples carried as author evidence, never shown to the solver."""
    utility_panel_size: int = Field(1, ge=1, le=3)
    """Fresh same-family tasks used to average downstream skill utility."""
    seed: int = 20260803


class ReasoningOffloadSkillAcquisitionTaskset(
    vf.Taskset[
        ReasoningOffloadSkillAcquisitionTask,
        ReasoningOffloadSkillAcquisitionConfig,
    ]
):
    def load(self) -> list[ReasoningOffloadSkillAcquisitionTask]:
        tasks = []
        idx = 0
        for family in self.config.families:
            split_variants = list(SPLIT_VARIANTS[self.config.split])
            author_variants = split_variants[: self.config.variants_per_family]
            for variant in author_variants:
                for instance in range(self.config.instances_per_template):
                    generated = generate(
                        family,
                        seed=self.config.seed,
                        variant=variant,
                        instance=instance,
                    )
                    experiences = tuple(
                        SkillExperience(
                            prompt=example.prompt,
                            answer=example.answer,
                            files=example.files,
                        )
                        for discovery_variant in list(SPLIT_VARIANTS["discovery"])[
                            : self.config.experience_examples
                        ]
                        for example in [
                            generate(
                                family,
                                seed=self.config.seed,
                                variant=discovery_variant,
                                instance=instance,
                            )
                        ]
                    )
                    panel_variants = [
                        variant,
                        *(
                            candidate
                            for candidate in split_variants
                            if candidate != variant
                        ),
                    ][: self.config.utility_panel_size]
                    utility_examples = tuple(
                        SkillUtilityExample(
                            name=(
                                f"skill-utility-{self.config.split}-{family}-"
                                f"v{panel_variant}-i{instance}"
                            ),
                            prompt=example.prompt,
                            system_prompt=SKILL_ACQUISITION_SYSTEM_PROMPT,
                            template_variant=panel_variant,
                            answer=example.answer,
                            files=example.files,
                        )
                        for panel_variant in panel_variants
                        for example in [
                            generate(
                                family,
                                seed=self.config.seed,
                                variant=panel_variant,
                                instance=instance,
                            )
                        ]
                    )
                    tasks.append(
                        ReasoningOffloadSkillAcquisitionTask(
                            ReasoningOffloadSkillAcquisitionData(
                                idx=idx,
                                name=(
                                    f"skill-acquisition-{self.config.split}-"
                                    f"{family}-v{variant}-i{instance}"
                                ),
                                prompt=generated.prompt,
                                system_prompt=SKILL_ACQUISITION_SYSTEM_PROMPT,
                                family=family,
                                split=self.config.split,
                                template_variant=variant,
                                answer=generated.answer,
                                files=generated.files,
                                experiences=experiences,
                                utility_examples=utility_examples,
                            )
                        )
                    )
                    idx += 1
        return tasks

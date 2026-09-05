"""Source/config worker taskset with one execution-grounded GRPO reward."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

import verifiers.v1 as vf
from subagent_communication_v1.taskset import (
    SubagentCommunicationConfig,
    SubagentCommunicationTask,
    SubagentCommunicationTaskConfig,
    SubagentCommunicationTaskset,
    _incoming_child_messages,
    _ipython_events,
)
from verifiers.v1.types import UserMessage, content_text

from source_worker_first_call_v1.reward import CellEvidence, score_first_call

SOURCE_FAMILIES = ("specialist_source_ast", "specialist_source_config")


class SourceWorkerFirstCallTaskConfig(SubagentCommunicationTaskConfig):
    reward_mode: Literal["source_worker_first_call"] = "source_worker_first_call"


def _is_child_node(trace: vf.Trace, node_index: int) -> bool:
    visited: set[int] = set()
    while node_index not in visited:
        visited.add(node_index)
        node = trace.nodes[node_index]
        if isinstance(node.message, UserMessage) and content_text(
            node.message.content
        ).lstrip().startswith("[task from parent]"):
            return True
        if node.parent is None:
            return False
        node_index = node.parent
    return False


class SourceWorkerFirstCallTask(SubagentCommunicationTask):
    def _include_standard_rewards(self) -> bool:
        return False

    def _score(self, trace: vf.Trace):
        events = tuple(
            CellEvidence(event.code, event.output)
            for event in _ipython_events(trace)
            if _is_child_node(trace, event.node_index)
        )
        messages = tuple(
            message.body
            for message in _incoming_child_messages(trace)
            if message.name == "task-worker"
        )
        expected_value = self.data.answer.get("result")
        if not isinstance(expected_value, int):
            raise ValueError("source-worker task requires one integer result oracle")
        if self.data.family not in SOURCE_FAMILIES:
            raise ValueError(f"unsupported source-worker family: {self.data.family}")
        return score_first_call(
            family=self.data.family,
            required_paths=tuple(sorted(self.data.files)),
            expected_value=expected_value,
            cells=events,
            delivered_bodies=messages,
        )

    @vf.reward(weight=1.0)
    async def source_worker_first_call(self, trace: vf.Trace) -> float:
        return self._score(trace).score

    @vf.metric
    async def source_worker_first_call_evidence(
        self, trace: vf.Trace
    ) -> dict[str, float]:
        return self._score(trace).metrics()


class SourceWorkerFirstCallConfig(SubagentCommunicationConfig):
    task: SourceWorkerFirstCallTaskConfig = SourceWorkerFirstCallTaskConfig()
    split: Literal["train"] = "train"
    families: tuple[
        Literal["specialist_source_ast", "specialist_source_config"], ...
    ] = Field(min_length=1, max_length=1)

    @model_validator(mode="after")
    def validate_training_partition(self):
        if len(self.families) != 1 or self.families[0] not in SOURCE_FAMILIES:
            raise ValueError("source-worker GRPO requires one explicit source family")
        if self.teacher_conditioned or self.ownership_guided:
            raise ValueError("source-worker GRPO may not inject demonstrations")
        return self


class SourceWorkerFirstCallTaskset(
    vf.Taskset[SourceWorkerFirstCallTask, SourceWorkerFirstCallConfig]
):
    def load(self) -> list[SourceWorkerFirstCallTask]:
        base_tasks = SubagentCommunicationTaskset(self.config).load()
        tasks = [
            SourceWorkerFirstCallTask(task.data, self.config.task)
            for task in base_tasks
        ]
        expected_family = self.config.families[0]
        if not tasks or any(task.data.family != expected_family for task in tasks):
            raise ValueError(
                f"source-worker taskset leaked outside {expected_family}"
            )
        if any(task.data.template_variant not in {0, 1, 2, 3} for task in tasks):
            raise ValueError("source-worker training leaked an eval-only template variant")
        return tasks

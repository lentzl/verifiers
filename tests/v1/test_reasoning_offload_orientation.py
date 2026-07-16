"""Reasoning-offload task feedback used by feedback-conditioned trainers."""

import pytest

import verifiers.v1 as vf
from reasoning_offload_orientation_v1.taskset import (
    INCORRECT_ANSWER_FEEDBACK,
    ReasoningOffloadOrientationConfig,
    ReasoningOffloadOrientationData,
    ReasoningOffloadOrientationTask,
    ReasoningOffloadOrientationTaskset,
)
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, UserMessage


def make_task_and_trace(reply: str):
    data = ReasoningOffloadOrientationData(
        idx=0,
        name="direct-feedback-test",
        prompt="Return <answer>expected</answer>.",
        family="direct",
        template_variant=0,
        answer="expected",
        files={},
    )
    task = ReasoningOffloadOrientationTask(data)
    trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=data),
        nodes=[
            MessageNode(
                parent=None,
                message=UserMessage(content=data.prompt),
                sampled=False,
            ),
            MessageNode(
                parent=0,
                message=AssistantMessage(content=reply),
                sampled=True,
            ),
        ],
    )
    return task, trace


@pytest.mark.asyncio
async def test_incorrect_answer_exposes_non_leaking_feedback():
    task, trace = make_task_and_trace("<answer>wrong</answer>")

    await task.score(trace)

    assert trace.reward == 0.0
    assert trace.info["feedback"] == INCORRECT_ANSWER_FEEDBACK
    assert task.data.answer not in trace.info["feedback"]


@pytest.mark.asyncio
async def test_correct_answer_does_not_add_failure_feedback():
    task, trace = make_task_and_trace("<answer>expected</answer>")

    await task.score(trace)

    assert trace.reward == 1.0
    assert "feedback" not in trace.info


def test_taskset_filters_curriculum_families_without_changing_variants():
    config = ReasoningOffloadOrientationConfig(
        split="train",
        families=("inspection", "state"),
        instances_per_template=1,
    )

    tasks = ReasoningOffloadOrientationTaskset(config).load()

    assert len(tasks) == 8
    assert {task.data.family for task in tasks} == {"inspection", "state"}
    assert {task.data.template_variant for task in tasks} == {0, 1, 2, 3}

"""Reasoning-offload task feedback used by feedback-conditioned trainers."""

import ast
import json

import pytest

import verifiers.v1 as vf
from reasoning_offload_orientation_v1.taskset import (
    INCORRECT_ANSWER_FEEDBACK,
    ORIENTATION_SYSTEM_PROMPT,
    ReasoningOffloadOrientationConfig,
    ReasoningOffloadOrientationData,
    ReasoningOffloadOrientationTask,
    ReasoningOffloadOrientationTaskset,
    _module_reused,
    _mutates_buggy_file,
    _repair_progress,
)
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, ToolCall, ToolMessage, UserMessage


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
    assert trace.info["feedback"] == INCORRECT_ANSWER_FEEDBACK["direct"]
    assert task.data.answer not in trace.info["feedback"]


@pytest.mark.parametrize("family", INCORRECT_ANSWER_FEEDBACK)
def test_family_feedback_is_non_leaking_and_actionable(family):
    feedback = INCORRECT_ANSWER_FEEDBACK[family]

    assert "incorrect" in feedback
    assert len(feedback) > 80


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
    assert {task.data.system_prompt for task in tasks} == {ORIENTATION_SYSTEM_PROMPT}


def test_explicit_instruction_level_guides_operations_without_changing_tasks():
    common = {
        "split": "train",
        "families": ("state", "verification", "repair"),
        "instances_per_template": 1,
    }
    standard = ReasoningOffloadOrientationTaskset(
        ReasoningOffloadOrientationConfig(**common)
    ).load()
    explicit = ReasoningOffloadOrientationTaskset(
        ReasoningOffloadOrientationConfig(**common, instruction_level="explicit")
    ).load()

    assert len(standard) == len(explicit) == 12
    for standard_task, explicit_task in zip(standard, explicit, strict=True):
        assert standard_task.data.answer == explicit_task.data.answer
        assert standard_task.data.files == explicit_task.data.files
        assert standard_task.data.instruction_level == "standard"
        assert explicit_task.data.instruction_level == "explicit"
        assert "Orientation hint:" not in standard_task.data.prompt
        if explicit_task.data.family in {"state", "verification", "repair"}:
            assert "Orientation hint:" in explicit_task.data.prompt
        else:
            assert explicit_task.data.prompt == standard_task.data.prompt


def test_module_family_requires_importing_and_calling_provided_transform():
    config = ReasoningOffloadOrientationConfig(
        split="train",
        families=("module",),
        instances_per_template=1,
    )

    tasks = ReasoningOffloadOrientationTaskset(config).load()

    assert len(tasks) == 4
    assert all("inputs/operation.py" in task.data.files for task in tasks)
    assert all("Do not reimplement" in task.data.prompt for task in tasks)
    assert _module_reused(
        [
            ast.parse(
                "from inputs.operation import transform\nresult = transform(target)"
            )
        ]
    )
    assert _module_reused(
        [
            ast.parse(
                "import sys\nsys.path.insert(0, 'inputs')\n"
                "from operation import transform\nresult = transform(target)"
            )
        ]
    )
    assert not _module_reused(
        [
            ast.parse(
                "import json\ndef transform(value): return value\ntransform(target)"
            )
        ]
    )
    assert not _module_reused([ast.parse("def transform(value): return value")])


@pytest.mark.parametrize(
    "code",
    [
        "from pathlib import Path\nPath('inputs/buggy.py').write_text(source)",
        "with open('inputs/buggy.py', 'w') as handle:\n    handle.write(source)",
        "open('inputs/buggy.py', mode='w').write(source)",
        "from pathlib import Path\nPath('inputs/buggy.py').open('w').write(source)",
        "%%bash\nsed -i 's/wrong/right/' inputs/buggy.py",
        "%%bash\nprintf '%s' \"$SOURCE\" > inputs/buggy.py",
    ],
)
def test_repair_mutation_recognizes_writes_to_buggy_file(code):
    assert _mutates_buggy_file(code)


@pytest.mark.parametrize(
    "code",
    [
        "source = Path('inputs/buggy.py').read_text()",
        "def transform(value): return value",
        "Path('inputs/check.py').write_text(source)",
        "open('inputs/target.json', 'w').write(value)",
        "%%bash\nsed -i 's/wrong/right/' inputs/check.py",
    ],
)
def test_repair_mutation_rejects_reads_and_unrelated_writes(code):
    assert not _mutates_buggy_file(code)


def _repair_trace(*messages):
    data = ReasoningOffloadOrientationData(
        idx=0,
        name="repair-progress-test",
        prompt="Repair inputs/buggy.py.",
        family="repair",
        template_variant=0,
        answer="expected",
        files={},
    )
    task = ReasoningOffloadOrientationTask(data)
    nodes = [
        MessageNode(
            parent=None if index == 0 else index - 1,
            message=message,
            sampled=isinstance(message, AssistantMessage),
        )
        for index, message in enumerate((UserMessage(content=data.prompt), *messages))
    ]
    return vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=data),
        nodes=nodes,
    )


def _ipython_call(call_id: str, code: str) -> AssistantMessage:
    return AssistantMessage(
        tool_calls=[
            ToolCall(
                id=call_id,
                name="ipython",
                arguments=json.dumps({"code": code}),
            )
        ]
    )


def test_repair_progress_requires_mutation_between_failure_and_verification():
    trace = _repair_trace(
        ToolMessage(tool_call_id="check-1", content="FAILED [(1, 2, 3)]"),
        _ipython_call(
            "edit",
            "from pathlib import Path\n"
            "Path('inputs/buggy.py').write_text(corrected_source)",
        ),
        ToolMessage(tool_call_id="edit", content=""),
        _ipython_call("check-2", "%%bash\npython3 inputs/check.py"),
        ToolMessage(tool_call_id="check-2", content="VERIFIED"),
    )

    assert _repair_progress(trace) == (True, True)


def test_repair_progress_rejects_notebook_only_recovery():
    trace = _repair_trace(
        ToolMessage(tool_call_id="check-1", content="FAILED [(1, 2, 3)]"),
        _ipython_call("experiment", "def transform(value): return value"),
        ToolMessage(tool_call_id="experiment", content="VERIFIED"),
    )

    assert _repair_progress(trace) == (False, False)

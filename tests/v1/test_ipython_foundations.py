import json

import pytest
from ipython_foundations_v1.taskset import (
    IpythonFoundationsConfig,
    IpythonFoundationsTaskset,
    _behavior,
    _partial_score,
    _round_prompt,
)

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, ToolCall, ToolMessage, UserMessage


def _trace(calls):
    nodes = []
    parent = None
    for segment, code, output in calls:
        while sum(isinstance(node.message, UserMessage) for node in nodes) <= segment:
            nodes.append(
                MessageNode(
                    parent=parent,
                    message=UserMessage(content=f"request-{segment}"),
                    sampled=False,
                )
            )
            parent = len(nodes) - 1
        call_id = f"call-{len(nodes)}"
        nodes.append(
            MessageNode(
                parent=parent,
                message=AssistantMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=call_id,
                            name="ipython",
                            arguments=json.dumps({"code": code}),
                        )
                    ],
                ),
                sampled=True,
            )
        )
        parent = len(nodes) - 1
        nodes.append(
            MessageNode(
                parent=parent,
                message=ToolMessage(tool_call_id=call_id, content=output),
                sampled=False,
            )
        )
        parent = len(nodes) - 1
    return vf.Trace(
        id="ipython-foundations-test",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="IpythonFoundationsTask", data=vf.TaskData(idx=0)),
        nodes=nodes,
    )


def test_taskset_balances_families_and_holds_out_variants():
    train = IpythonFoundationsTaskset(
        IpythonFoundationsConfig(split="train", instances_per_template=1)
    ).load()
    evaluation = IpythonFoundationsTaskset(
        IpythonFoundationsConfig(split="eval", instances_per_template=1)
    ).load()

    assert len(train) == 12
    assert len(evaluation) == 6
    assert {task.data.family for task in train} == {
        "assignment",
        "state",
        "recovery",
    }
    assert {task.data.template_variant for task in train} == {0, 1, 2, 3}
    assert {task.data.template_variant for task in evaluation} == {4, 5}
    assert all(len(task.data.rounds) == 3 for task in [*train, *evaluation])


def test_state_stream_removes_source_and_requires_later_notebook_reuse():
    task = next(
        task
        for task in IpythonFoundationsTaskset(
            IpythonFoundationsConfig(instances_per_template=1)
        ).load()
        if task.data.family == "state"
    )

    assert task.data.rounds[0].remove_after == ("/workspace/inbox/records.json",)
    assert not task.data.rounds[1].files
    assert not task.data.rounds[2].files
    assert "retained `records`" in task.data.rounds[2].instruction


def test_explicit_scaffolding_describes_operations_without_leaking_answers():
    tasks = IpythonFoundationsTaskset(
        IpythonFoundationsConfig(instruction_level="explicit", instances_per_template=1)
    ).load()

    for task in tasks[:3]:
        prompt = _round_prompt(task, 0, None)
        assert "Foundation exercise:" in prompt
        assert task.data.rounds[0].explicit_operation in prompt
        assert json.dumps(task.data.rounds[0].answer) not in prompt


@pytest.mark.parametrize(
    ("family", "variable", "calls"),
    [
        (
            "assignment",
            "values",
            [
                (0, "values = [2, 3]", ""),
                (0, "sum(values)", "5"),
            ],
        ),
        (
            "state",
            "records",
            [
                (0, "records = [{'amount': 2}]\nlen(records)", "1"),
                (1, "sum(row['amount'] for row in records)", "2"),
            ],
        ),
        (
            "recovery",
            "rows",
            [
                (
                    0,
                    "rows = [{'amount': 2}]\nsum(row['value'] for row in rows)",
                    "Traceback: KeyError: 'value'",
                ),
                (0, "rows[0]", "{'amount': 2}"),
                (0, "sum(row['amount'] for row in rows)", "2"),
            ],
        ),
    ],
)
def test_process_alignment_recognizes_family_specific_notebook_semantics(
    family, variable, calls
):
    behavior = _behavior(_trace(calls), family, variable)

    assert behavior["process_aligned"] == 1.0
    assert behavior["state_reused"] == 1.0
    assert behavior["identical_consecutive_calls"] == 0.0


def test_identical_empty_assignment_loop_is_not_rewarded():
    trace = _trace(
        [
            (0, "values = [2, 3]", ""),
            (0, "values = [2, 3]", ""),
            (0, "sum(values)", "5"),
        ]
    )

    behavior = _behavior(trace, "assignment", "values")

    assert behavior["silent_assignment_recovered"] == 1.0
    assert behavior["identical_consecutive_calls"] == 1.0
    assert behavior["process_aligned"] == 0.0


@pytest.mark.parametrize(
    ("actual", "expected", "score"),
    [
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}, 1.0),
        ({"a": 1}, {"a": 1, "b": 2}, 0.5),
        (["a"], ["a", "b"], 0.5),
        (7, 7, 1.0),
        (None, 7, 0.0),
    ],
)
def test_partial_score_provides_dense_answer_credit(actual, expected, score):
    assert _partial_score(actual, expected) == pytest.approx(score)

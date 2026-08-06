import json
import subprocess

import pytest
from ipython_foundations_v1.python_recovery_cases import RECOVERY_KINDS
from ipython_foundations_v1.taskset import (
    PDFTOTEXT_COMPAT,
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

    assert len(train) == 20
    assert len(evaluation) == 12
    assert {task.data.family for task in train} == {
        "completion",
        "assignment",
        "state",
        "recovery",
        "subprocess",
    }
    assert {task.data.template_variant for task in train} == {0, 1, 2, 3}
    assert {
        task.data.template_variant
        for task in evaluation
        if task.data.family != "recovery"
    } == {4, 5}
    assert {
        task.data.template_variant
        for task in evaluation
        if task.data.family == "recovery"
    } == {4, 5, 6, 7}
    assert all(
        len(task.data.rounds) == (1 if task.data.family == "completion" else 3)
        for task in [*train, *evaluation]
    )


def test_round_limit_isolates_single_request_assignment_rung():
    tasks = IpythonFoundationsTaskset(
        IpythonFoundationsConfig(
            families=("assignment",),
            rounds_per_task=1,
            instances_per_template=1,
        )
    ).load()

    assert len(tasks) == 4
    assert all(len(task.data.rounds) == 1 for task in tasks)


def test_completion_stream_requires_one_result_then_immediate_answer():
    task = next(
        task
        for task in IpythonFoundationsTaskset(
            IpythonFoundationsConfig(instances_per_template=1)
        ).load()
        if task.data.family == "completion"
    )

    assert len(task.data.rounds) == 1
    assert "one IPython call" in task.data.rounds[0].instruction
    assert "return" in task.data.rounds[0].explicit_operation


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


def test_subprocess_stream_preserves_path_and_requires_error_directed_repair():
    task = next(
        task
        for task in IpythonFoundationsTaskset(
            IpythonFoundationsConfig(instances_per_template=1)
        ).load()
        if task.data.family == "subprocess"
    )

    path = task.data.rounds[0].answer
    assert "Title:" not in task.data.rounds[0].files[path]
    assert task.data.rounds[1].answer["returncode"] == 1
    assert "'-text'" in task.data.rounds[1].answer["stderr"]
    assert not task.data.rounds[1].files
    assert not task.data.rounds[2].files
    assert "output path" in task.data.rounds[2].instruction
    assert "raw PDF bytes" in task.data.rounds[2].instruction


def test_recovery_training_matrix_covers_every_real_error_kind():
    recovery_tasks = [
        task
        for task in IpythonFoundationsTaskset(
            IpythonFoundationsConfig(instances_per_template=1)
        ).load()
        if task.data.family == "recovery"
    ]
    rounds = [round_ for task in recovery_tasks for round_ in task.data.rounds]

    assert {round_.recovery_kind for round_ in rounds} == set(RECOVERY_KINDS)
    assert all("real" in round_.instruction.lower() for round_ in rounds)
    assert all(
        "Traceback (most recent call last)" not in round_.instruction
        for round_ in rounds
    )
    assert all(not round_.remove_after for round_ in rounds)

    evaluation = [
        task
        for task in IpythonFoundationsTaskset(
            IpythonFoundationsConfig(split="eval", instances_per_template=1)
        ).load()
        if task.data.family == "recovery"
    ]
    evaluation_kinds = {
        round_.recovery_kind for task in evaluation for round_ in task.data.rounds
    }
    assert evaluation_kinds == set(RECOVERY_KINDS)


def test_pdftotext_fixture_exposes_real_failure_and_stdout_repair(tmp_path):
    executable = tmp_path / "pdftotext"
    executable.write_text(PDFTOTEXT_COMPAT)
    executable.chmod(0o755)
    source = tmp_path / "report.pdf"
    source.write_text("VGl0bGU6IFJlcG9ydAo=")

    failed = subprocess.run(
        [executable, "-layout", "-text", source],
        capture_output=True,
        text=True,
        check=False,
    )
    repaired = subprocess.run(
        [executable, "-layout", source, "-"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert failed.returncode == 1
    assert failed.stdout == ""
    assert "'-text'" in failed.stderr
    assert repaired.returncode == 0
    assert repaired.stdout == "Title: Report\n"
    assert repaired.stderr == ""


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
            "completion",
            "result",
            [(0, "sum([2, 3])", "5")],
        ),
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
        (
            "subprocess",
            "pdf_path",
            [
                (
                    0,
                    "from pathlib import Path\npdf_path = '/workspace/inbox/report.pdf'\nPath(pdf_path).exists()",
                    "True",
                ),
                (
                    1,
                    "result = subprocess.run(['pdftotext', '-layout', '-text', pdf_path], capture_output=True, text=True)\n{'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}",
                    "{'returncode': 1, 'stdout': '', 'stderr': \"I/O Error: Couldn't open file '-text'\"}",
                ),
                (
                    2,
                    "result = subprocess.run(['pdftotext', '-layout', pdf_path, '-'], capture_output=True, text=True)\n(result.returncode, result.stdout, result.stderr)",
                    "(0, 'Title: Report', '')",
                ),
            ],
        ),
    ],
)
def test_process_alignment_recognizes_family_specific_notebook_semantics(
    family, variable, calls
):
    behavior = _behavior(_trace(calls), family, variable)

    assert behavior["process_aligned"] == 1.0
    assert behavior["process_score"] == 1.0
    assert behavior["state_reused"] == float(family != "completion")
    assert behavior["identical_consecutive_calls"] == 0.0


def test_subprocess_loop_and_raw_byte_fallback_are_not_rewarded():
    failed = "result = subprocess.run(['pdftotext', '-text', pdf_path], capture_output=True, text=True)\n(result.returncode, result.stdout, result.stderr)"
    trace = _trace(
        [
            (0, "pdf_path = '/workspace/inbox/report.pdf'", ""),
            (1, failed, "(1, '', \"I/O Error: Couldn't open file '-text'\")"),
            (1, failed, "(1, '', \"I/O Error: Couldn't open file '-text'\")"),
            (1, "Path(pdf_path).read_bytes().decode()", "UnicodeDecodeError"),
        ]
    )

    behavior = _behavior(trace, "subprocess", "pdf_path")

    assert behavior["subprocess_result_observed"] == 1.0
    assert behavior["identical_consecutive_calls"] == 1.0
    assert behavior["subprocess_failure_retries"] == 1.0
    assert behavior["raw_pdf_fallback_used"] == 1.0
    assert 0.0 < behavior["process_score"] < 0.5
    assert behavior["process_aligned"] == 0.0


def test_recovery_process_reward_requires_feedback_repair_in_every_segment():
    trace = _trace(
        [
            (
                0,
                "payload = [1]\npayload[1]",
                "Traceback: IndexError: list index out of range",
            ),
            (0, "payload[0]", "1"),
            (
                1,
                "payload = {'value': 2}\npayload['missing']",
                "Traceback: KeyError: 'missing'",
            ),
        ]
    )

    behavior = _behavior(trace, "recovery", "payload", expected_segments=2)

    assert behavior["recovery_error_segments"] == 2.0
    assert behavior["recovery_repaired_segments"] == 1.0
    assert behavior["recovery_round_coverage"] == pytest.approx(0.5)
    assert behavior["process_score"] == pytest.approx(0.5)
    assert behavior["process_aligned"] == 0.0


def test_completion_process_reward_decays_when_agent_keeps_calling_ipython():
    trace = _trace([(0, "sum([2, 3])", "5")] * 4)

    behavior = _behavior(trace, "completion", "result")

    assert behavior["successful_result_observed"] == 1.0
    assert behavior["ipython_call_efficiency"] == pytest.approx(0.25)
    assert behavior["process_score"] == pytest.approx(1 / 16)
    assert behavior["process_aligned"] == 0.0


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
    assert behavior["process_score"] == pytest.approx(1 / 3)
    assert behavior["process_aligned"] == 0.0


def test_reassigning_state_does_not_count_as_reuse():
    trace = _trace(
        [
            (0, "values = [2, 3]", ""),
            (1, "values = [4, 5]\nsum(values)", "9"),
        ]
    )

    behavior = _behavior(trace, "state", "values")

    assert behavior["state_reused"] == 0.0
    assert behavior["cross_turn_state_reused"] == 0.0
    assert behavior["process_score"] == 0.0


def test_subprocess_process_score_requires_positive_milestones():
    trace = _trace([(0, "pdf_path = '/workspace/inbox/report.pdf'", "")])

    behavior = _behavior(trace, "subprocess", "pdf_path")

    assert behavior["raw_pdf_fallback_used"] == 0.0
    assert behavior["subprocess_failure_retries"] == 0.0
    assert behavior["process_score"] == 0.0


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

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from document_summary_v1.fixture import build_fixture
from document_summary_v1.taskset import (
    EMPTY_IPYTHON_FEEDBACK,
    GATE_PATH,
    OUTPUT_PATH,
    REPEATED_IPYTHON_FAILURE_FEEDBACK,
    REPEATED_IPYTHON_NO_PROGRESS_FEEDBACK,
    WORKER_OUTPUT_PATH,
    DocumentSummaryConfig,
    DocumentSummaryTaskset,
    _artifact_components,
    _fact_coverage,
    _owner_gate_source,
    _report_components,
    _rewrite_empty_ipython_feedback,
    _rewrite_repeated_ipython_failure,
    _rewrite_repeated_ipython_no_progress,
    _strict_report,
    _worker_gate_source,
)

import verifiers.v1 as vf


def _worker_task():
    return DocumentSummaryTaskset(DocumentSummaryConfig(mode="worker_probe")).load()[0]


def _owner_task():
    return DocumentSummaryTaskset(DocumentSummaryConfig(mode="owner")).load()[0]


def _scope_report(task):
    hashes = {row["id"]: row["source_sha256"] for row in task.data.job["paragraphs"]}
    assert len(set(hashes.values())) == 4
    return {
        "worker": "scope-summarizer",
        "chapter_id": "scope",
        "bullets": [
            {
                "id": "scope-b01",
                "text": (
                    "Northstar replaces the shared email queue with a shared ticket system to "
                    "expose ownership and handoffs without changing support policy."
                ),
                "source_ids": ["scope-p01"],
            },
            {
                "id": "scope-b02",
                "text": (
                    "Phase one covers Berlin and Oulu tickets from 1 October; billing disputes "
                    "and legal notices remain outside."
                ),
                "source_ids": ["scope-p02"],
            },
            {
                "id": "scope-b03",
                "text": (
                    "Acknowledge 95 percent within four hours, keep unresolved tickets visible, "
                    "and preserve ticket identifiers in weekly reports for traceability."
                ),
                "source_ids": ["scope-p03", "scope-p04"],
            },
        ],
        "issues": [],
    }


def test_fixture_has_three_grounded_chapters() -> None:
    document, facts = build_fixture()

    assert document["document_id"] == "project-northstar-playbook-v1"
    assert [chapter["id"] for chapter in document["chapters"]] == [
        "scope",
        "operations",
        "exceptions",
    ]
    assert [len(chapter["paragraphs"]) for chapter in document["chapters"]] == [4, 4, 4]
    assert set(facts) == {"scope", "operations", "exceptions"}


def test_reference_scope_summary_is_concise_grounded_and_complete() -> None:
    task = _worker_task()
    report = _scope_report(task)

    assert _strict_report(report, task.data.job) is True
    assert set(_report_components(report, task.data.job).values()) == {1.0}
    assert _fact_coverage(report, task.data.fact_groups) == 1.0


def test_source_paragraph_copy_is_not_a_valid_summary() -> None:
    task = _worker_task()
    report = _scope_report(task)
    report["bullets"][0]["text"] = task.data.job["paragraphs"][0]["text"]

    components = _report_components(report, task.data.job)

    assert components["summary_not_source_copy"] == 0.0
    assert _strict_report(report, task.data.job) is False


def test_worker_probe_exposes_contract_but_not_hidden_fact_groups() -> None:
    task = _worker_task()
    gate = _worker_gate_source(task.data)

    assert task.data.output_path == WORKER_OUTPUT_PATH
    assert task.data.job["task_contract"]["bullet_count"] == 3
    assert "Do not spawn a child" in task.data.prompt_text
    assert "json.dump requires an open file handle" in task.data.system_prompt
    assert any(
        "empty JSON list for issues" in requirement
        for requirement in task.data.job["task_contract"]["requirements"]
    )
    assert "email queue" not in gate
    assert "95 percent" not in gate
    assert "normalized_sources" in gate
    assert "paraphrase" in gate
    assert "missing paragraph coverage" in gate
    assert "Keep exactly three bullets" in gate


def test_worker_gate_rejects_an_exact_source_paragraph_without_embedding_facts() -> None:
    task = _worker_task()
    gate = _worker_gate_source(task.data)

    compile(gate, "completion_gate.py", "exec")
    assert repr(task.data.job["path"]) in gate
    assert task.data.job["paragraphs"][0]["text"] not in gate
    assert 'job["paragraphs"]' in gate


def test_owner_gate_is_valid_python() -> None:
    compile(_owner_gate_source(_owner_task().data), "completion_gate.py", "exec")


def test_worker_gate_reports_all_actionable_summary_defects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    task = _worker_task()
    job_path = tmp_path / "job.json"
    output_path = tmp_path / "report.json"
    job = {**task.data.job, "path": str(job_path)}
    report = _scope_report(task)
    report["bullets"][0]["text"] = task.data.job["paragraphs"][0]["text"]
    report["bullets"][2]["source_ids"] = ["scope-p04"]
    job_path.write_text(json.dumps(job), encoding="utf-8")
    output_path.write_text(json.dumps(report), encoding="utf-8")
    data = task.data.model_copy(update={"job": job, "output_path": str(output_path)})

    with pytest.raises(SystemExit) as error:
        exec(  # noqa: S102 - execute the generated gate exactly as Prime Agent will
            compile(_worker_gate_source(data), "completion_gate.py", "exec"), {}
        )

    assert error.value.code == 1
    diagnostic = capsys.readouterr().err
    assert "missing paragraph coverage: ['scope-p03']" in diagnostic
    assert "Keep exactly three bullets" in diagnostic
    assert "verbatim source copying in bullets ['scope-b01']" in diagnostic


def test_worker_gate_reports_identity_shape_and_coverage_in_one_attempt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    task = _worker_task()
    job_path = tmp_path / "job.json"
    output_path = tmp_path / "report.json"
    job = {**task.data.job, "path": str(job_path)}
    report = {
        "worker": job["document_id"],
        "chapter_id": job["chapter_title"],
        "bullets": [
            {"id": bullet_id, "text": "", "source_ids": []}
            for bullet_id in job["task_contract"]["bullet_ids"]
        ],
        "issues": [],
    }
    job_path.write_text(json.dumps(job), encoding="utf-8")
    output_path.write_text(json.dumps(report), encoding="utf-8")
    data = task.data.model_copy(update={"job": job, "output_path": str(output_path)})

    with pytest.raises(SystemExit) as error:
        exec(  # noqa: S102 - execute the generated gate exactly as Prime Agent will
            compile(_worker_gate_source(data), "completion_gate.py", "exec"), {}
        )

    assert error.value.code == 1
    diagnostic = capsys.readouterr().err
    assert "worker identity differs from the job" in diagnostic
    assert "chapter identity differs from the job" in diagnostic
    assert "each bullet text must contain 5 to 45 words" in diagnostic
    assert "each bullet needs one or more valid paragraph source_ids" in diagnostic
    assert "missing paragraph coverage" in diagnostic


def test_empty_ipython_result_gets_clear_model_facing_feedback() -> None:
    trace = vf.Trace(
        id="empty-ipython",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="DocumentSummaryWorkerTask", data=vf.TaskData(idx=0)),
        nodes=[],
    )
    request = vf.Request(
        messages=[
            vf.AssistantMessage(
                tool_calls=[
                    vf.ToolCall(
                        id="empty-call",
                        name="ipython",
                        arguments=json.dumps({"code": "  \n"}),
                    )
                ]
            ),
            vf.ToolMessage(tool_call_id="empty-call", name="ipython", content=""),
        ]
    )

    rewritten = _rewrite_empty_ipython_feedback(request, trace)

    assert rewritten is not None
    assert rewritten.messages[-1].content == EMPTY_IPYTHON_FEEDBACK
    assert request.messages[-1].content == ""
    assert trace.info["empty_ipython_feedback_count"] == 1


def test_concrete_ipython_call_is_not_rewritten() -> None:
    trace = vf.Trace(
        id="concrete-ipython",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="DocumentSummaryWorkerTask", data=vf.TaskData(idx=0)),
        nodes=[],
    )
    request = vf.Request(
        messages=[
            vf.AssistantMessage(
                tool_calls=[
                    vf.ToolCall(
                        id="real-call",
                        name="ipython",
                        arguments=json.dumps({"code": "artifact.exists()"}),
                    )
                ]
            ),
            vf.ToolMessage(
                tool_call_id="real-call", name="ipython", content="True"
            ),
        ]
    )

    assert _rewrite_empty_ipython_feedback(request, trace) is None
    assert "empty_ipython_feedback_count" not in trace.info


def test_repeated_failed_ipython_call_gets_progress_feedback() -> None:
    trace = vf.Trace(
        id="repeated-ipython",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="DocumentSummaryWorkerTask", data=vf.TaskData(idx=0)),
        nodes=[],
    )
    code = "raise TypeError('broken check')"
    request = vf.Request(
        messages=[
            vf.AssistantMessage(
                tool_calls=[
                    vf.ToolCall(
                        id="first-call",
                        name="ipython",
                        arguments=json.dumps({"code": code}),
                    )
                ]
            ),
            vf.ToolMessage(
                tool_call_id="first-call",
                name="ipython",
                content="Traceback: TypeError: broken check",
            ),
            vf.AssistantMessage(
                tool_calls=[
                    vf.ToolCall(
                        id="second-call",
                        name="ipython",
                        arguments=json.dumps({"code": code}),
                    )
                ]
            ),
            vf.ToolMessage(
                tool_call_id="second-call",
                name="ipython",
                content="Traceback: TypeError: broken check",
            ),
        ]
    )

    rewritten = _rewrite_repeated_ipython_failure(request, trace)

    assert rewritten is not None
    assert "Traceback: TypeError: broken check" in rewritten.messages[-1].content
    assert REPEATED_IPYTHON_FAILURE_FEEDBACK in rewritten.messages[-1].content
    assert REPEATED_IPYTHON_FAILURE_FEEDBACK not in request.messages[-1].content
    assert trace.info["repeated_ipython_failure_feedback_count"] == 1


def test_first_failed_ipython_call_is_not_labeled_as_repeated() -> None:
    trace = vf.Trace(
        id="first-ipython-failure",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="DocumentSummaryWorkerTask", data=vf.TaskData(idx=0)),
        nodes=[],
    )
    request = vf.Request(
        messages=[
            vf.AssistantMessage(
                tool_calls=[
                    vf.ToolCall(
                        id="first-call",
                        name="ipython",
                        arguments=json.dumps({"code": "raise ValueError('first')"}),
                    )
                ]
            ),
            vf.ToolMessage(
                tool_call_id="first-call",
                name="ipython",
                content="Traceback: ValueError: first",
            ),
        ]
    )

    assert _rewrite_repeated_ipython_failure(request, trace) is None
    assert "repeated_ipython_failure_feedback_count" not in trace.info


def test_repeated_successful_ipython_call_with_same_result_gets_progress_feedback(
) -> None:
    trace = vf.Trace(
        id="repeated-successful-ipython",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="DocumentSummaryWorkerTask", data=vf.TaskData(idx=0)),
        nodes=[],
    )
    code = "document_bullets = ['scope-b01', 'scope-b02', 'scope-b03']\ndocument_bullets"
    result = "['scope-b01', 'scope-b02', 'scope-b03']"
    request = vf.Request(
        messages=[
            vf.AssistantMessage(
                tool_calls=[
                    vf.ToolCall(
                        id="first-call",
                        name="ipython",
                        arguments=json.dumps({"code": code}),
                    )
                ]
            ),
            vf.ToolMessage(
                tool_call_id="first-call", name="ipython", content=result
            ),
            vf.AssistantMessage(
                tool_calls=[
                    vf.ToolCall(
                        id="second-call",
                        name="ipython",
                        arguments=json.dumps({"code": code}),
                    )
                ]
            ),
            vf.ToolMessage(
                tool_call_id="second-call", name="ipython", content=result
            ),
        ]
    )

    rewritten = _rewrite_repeated_ipython_no_progress(request, trace)

    assert rewritten is not None
    assert result in rewritten.messages[-1].content
    assert REPEATED_IPYTHON_NO_PROGRESS_FEEDBACK in rewritten.messages[-1].content
    assert REPEATED_IPYTHON_NO_PROGRESS_FEEDBACK not in request.messages[-1].content
    assert trace.info["repeated_ipython_no_progress_feedback_count"] == 1


def test_first_successful_ipython_call_is_not_labeled_as_no_progress() -> None:
    trace = vf.Trace(
        id="first-successful-ipython",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="DocumentSummaryWorkerTask", data=vf.TaskData(idx=0)),
        nodes=[],
    )
    request = vf.Request(
        messages=[
            vf.AssistantMessage(
                tool_calls=[
                    vf.ToolCall(
                        id="first-call",
                        name="ipython",
                        arguments=json.dumps({"code": "artifact.exists()"}),
                    )
                ]
            ),
            vf.ToolMessage(
                tool_call_id="first-call", name="ipython", content="False"
            ),
        ]
    )

    assert _rewrite_repeated_ipython_no_progress(request, trace) is None
    assert "repeated_ipython_no_progress_feedback_count" not in trace.info


def test_repeated_ipython_call_with_changed_result_is_not_labeled_no_progress() -> None:
    trace = vf.Trace(
        id="changed-ipython-result",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="DocumentSummaryWorkerTask", data=vf.TaskData(idx=0)),
        nodes=[],
    )
    code = "artifact.exists()"
    request = vf.Request(
        messages=[
            vf.AssistantMessage(
                tool_calls=[
                    vf.ToolCall(
                        id="first-call",
                        name="ipython",
                        arguments=json.dumps({"code": code}),
                    )
                ]
            ),
            vf.ToolMessage(
                tool_call_id="first-call", name="ipython", content="False"
            ),
            vf.AssistantMessage(
                tool_calls=[
                    vf.ToolCall(
                        id="second-call",
                        name="ipython",
                        arguments=json.dumps({"code": code}),
                    )
                ]
            ),
            vf.ToolMessage(
                tool_call_id="second-call", name="ipython", content="True"
            ),
        ]
    )

    assert _rewrite_repeated_ipython_no_progress(request, trace) is None
    assert "repeated_ipython_no_progress_feedback_count" not in trace.info


def test_owner_mode_binds_three_exact_jobs_and_no_legacy_polling() -> None:
    task = _owner_task()
    gate = _owner_gate_source(task.data)

    assert len(task.data.jobs) == 3
    assert OUTPUT_PATH in task.data.prompt_text
    assert all(job["path"] in task.data.prompt_text for job in task.data.jobs.values())
    assert all(
        job["task_contract"]["delivery"] == "agent_message_parent_once"
        for job in task.data.jobs.values()
    )
    assert "do not poll" in task.data.prompt_text.casefold()
    assert "fact_groups" not in gate


@pytest.mark.asyncio
async def test_worker_setup_writes_only_runtime_source_and_contract() -> None:
    task = _worker_task()

    class Runtime:
        def __init__(self):
            self.writes = {}

        async def run(self, args, env):
            return SimpleNamespace(exit_code=0, stderr="")

        async def write(self, path, contents):
            self.writes[path] = contents

    runtime = Runtime()
    await task.setup(SimpleNamespace(), runtime)

    assert set(runtime.writes) == {task.data.job["path"], GATE_PATH}
    assert json.loads(runtime.writes[task.data.job["path"]])["paragraphs"]
    assert "fact_groups" not in b"\n".join(runtime.writes.values()).decode()


@pytest.mark.asyncio
async def test_worker_reference_report_earns_utility_reward() -> None:
    task = _worker_task()
    trace = vf.Trace(
        id="summary-worker-test",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="DocumentSummaryWorkerTask", data=vf.TaskData(idx=0)),
        nodes=[],
    )
    trace.info["document_summary_worker_report"] = _scope_report(task)

    await task.score(trace)

    assert trace.rewards["usable_chapter_summary"].value == 1.0
    assert trace.metrics["chapter_fact_coverage"] == 1.0


def test_artifact_requires_all_three_valid_chapter_reports() -> None:
    task = _owner_task()
    scope_task = _worker_task()
    artifact = {
        "schema_version": "prime-rl/document-chapter-summary/v1",
        "document_id": task.data.document["document_id"],
        "summary_language": "en",
        "chapters": [_scope_report(scope_task)],
    }

    components = _artifact_components(artifact, task.data)

    assert components["summary_artifact_schema"] == 1.0
    assert components["summary_chapter_order"] == 0.0
    assert components["summary_all_reports_valid"] == 0.0

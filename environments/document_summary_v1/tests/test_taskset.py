import json
from types import SimpleNamespace

import pytest
from document_summary_v1.fixture import build_fixture
from document_summary_v1.taskset import (
    GATE_PATH,
    OUTPUT_PATH,
    WORKER_OUTPUT_PATH,
    DocumentSummaryConfig,
    DocumentSummaryTaskset,
    _artifact_components,
    _fact_coverage,
    _owner_gate_source,
    _report_components,
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
    assert "email queue" not in gate
    assert "95 percent" not in gate


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

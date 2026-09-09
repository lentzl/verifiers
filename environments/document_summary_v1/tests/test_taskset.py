import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from document_summary_v1.fixture import build_confirmation_fixture, build_fixture
from document_summary_v1.taskset import (
    DIRECT_SUMMARY_NO_PROGRESS_FEEDBACK,
    EMPTY_IPYTHON_FEEDBACK,
    EVIDENCE_FILE_WRITE_RECOVERY_FEEDBACK,
    EVIDENCE_SOURCE_PATH,
    GATE_PATH,
    INDEX_PATH,
    MARKDOWN_CHILD_FAILURE_FEEDBACK,
    MARKDOWN_CHILD_RECOVERY_FEEDBACK,
    MARKDOWN_OUTPUT_PATH,
    MARKDOWN_OWNER_RECOVERY_FEEDBACK,
    MARKDOWN_UNSCOPED_RECOVERY_FEEDBACK,
    MISSING_WORKER_REPORT_RECOVERY_FEEDBACK,
    OUTPUT_PATH,
    REPEATED_IPYTHON_FAILURE_FEEDBACK,
    REPEATED_IPYTHON_NO_PROGRESS_FEEDBACK,
    ROOT,
    TERMINAL_WORKER_RECOVERY_FEEDBACK,
    TEXT_REVISION_COMMIT_MAX_TOKENS,
    TEXT_REVISION_COMMIT_REQUIREMENT,
    TEXT_REVISION_FEEDBACK,
    TEXT_REVISION_MARKER,
    TEXT_REVISION_SAFETY_MARGIN_REQUIREMENT,
    WORKER_OUTPUT_PATH,
    DocumentSummaryConfig,
    DocumentSummaryTaskset,
    _apply_text_revision_commit_sampling,
    _artifact_components,
    _assembled_markdown,
    _direct_summary_gate_source,
    _evidence_gate_source,
    _evidence_literal_write_repair,
    _fact_coverage,
    _markdown_owner_gate_source,
    _owner_gate_source,
    _plain_summary_bullets,
    _plain_summary_components,
    _report_components,
    _rewrite_empty_ipython_feedback,
    _rewrite_evidence_feedback,
    _rewrite_repeated_ipython_failure,
    _rewrite_repeated_ipython_no_progress,
    _rewrite_text_revision_feedback,
    _strict_report,
    _text_revision_gate_source,
    _worker_gate_source,
    _worker_recovery_feedback,
)

import verifiers.v1 as vf


def _worker_task():
    return DocumentSummaryTaskset(DocumentSummaryConfig(mode="worker_probe")).load()[0]


def _owner_task(mode="owner"):
    return DocumentSummaryTaskset(DocumentSummaryConfig(mode=mode)).load()[0]


def _text_task(chapter: str = "scope", split: str = "development"):
    return DocumentSummaryTaskset(
        DocumentSummaryConfig(
            mode="text_probe", text_probe_chapter=chapter, split=split
        )
    ).load()[0]


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


def test_confirmation_fixture_is_disjoint_and_reserved_for_post_training() -> None:
    development, _ = build_fixture()
    confirmation, facts = build_confirmation_fixture()

    assert confirmation["document_id"] != development["document_id"]
    assert [chapter["id"] for chapter in confirmation["chapters"]] == [
        "intake",
        "completion",
        "audit",
    ]
    assert set(facts) == {"intake", "completion", "audit"}
    development_text = json.dumps(development, sort_keys=True)
    assert all(
        paragraph["text"] not in development_text
        for chapter in confirmation["chapters"]
        for paragraph in chapter["paragraphs"]
    )


def test_confirmation_text_probe_uses_unseen_document_without_fact_leakage() -> None:
    task = _text_task("intake", split="confirmation")

    assert task.data.name.startswith("cedar-facilities-handbook-confirmation-v1")
    assert task.data.description == "Fresh English bullet-summary utility confirmation."
    assert "Facilities requests begin in the Service Desk" in task.data.prompt_text
    assert "fact_groups" not in task.data.prompt_text
    assert "Project Northstar" not in task.data.prompt_text


def test_confirmation_reference_summary_is_concise_and_fact_complete() -> None:
    task = _text_task("completion", split="confirmation")
    reply = (
        "- Before repair, photograph the asset tag and note existing damage; scan replacement parts into inventory.\n"
        "- Keep borrowed parts linked to donor equipment and return or reconcile them before closure.\n"
        "- The technician tests function and records the measurement; the requester confirms service is restored.\n"
        "- Record labor minutes, parts used and any follow-up date, or keep the request open."
    )

    assert _plain_summary_components(
        reply, task.data.chapter, task.data.fact_groups
    ) == {
        "summary_text_bullet_count": 1.0,
        "summary_text_concise": 1.0,
        "summary_text_not_source_copy": 1.0,
        "chapter_fact_coverage": 1.0,
    }


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

    for include_source_id in (False, True):
        copied_bullets = "\n".join(
            f"- {'[' + row['id'] + '] ' if include_source_id else ''}{row['text']}"
            for row in task.data.job["paragraphs"]
        )
        assert _plain_summary_components(
            copied_bullets, task.data.job, task.data.fact_groups
        )["summary_text_not_source_copy"] == 0.0


def test_duplicate_cross_bullet_citations_are_not_grounded() -> None:
    task = _worker_task()
    report = _scope_report(task)
    report["bullets"][0]["source_ids"].append("scope-p02")

    components = _report_components(report, task.data.job)

    assert components["summary_source_grounding"] == 0.0
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
    assert "paragraph IDs cited more than once" in gate


def test_text_probe_is_plain_english_without_artifact_plumbing() -> None:
    task = _text_task()

    assert task.data.name == "northstar-scope-plain-summary-probe-v1"
    assert "three to five concise English bullet points" in task.data.prompt_text
    assert "Preserve every decision-relevant fact" in task.data.prompt_text
    assert "Do not use IPython, code, JSON, files, or tools" in task.data.prompt_text
    assert "worker-report.json" not in task.data.prompt_text
    assert "completion_gate.py" not in task.data.prompt_text
    assert "email queue" in task.data.prompt_text
    assert "fact_groups" not in task.data.prompt_text
    assert task.data.network_allow == ["*"]


def test_plain_summary_components_measure_language_without_citation_schema() -> None:
    task = _text_task()
    reply = (
        "- Move support from the shared email queue to a ticket system for visible ownership and handoffs.\n"
        "- Start with Berlin and Oulu tickets from 1 October, leaving billing disputes and legal notices outside.\n"
        "- Acknowledge 95% within four hours, lose no unresolved tickets, and retain ticket identifiers for traceability."
    )

    assert _plain_summary_bullets(reply) == [line[2:] for line in reply.splitlines()]
    assert _plain_summary_components(
        reply, task.data.chapter, task.data.fact_groups
    ) == {
        "summary_text_bullet_count": 1.0,
        "summary_text_concise": 1.0,
        "summary_text_not_source_copy": 1.0,
        "chapter_fact_coverage": 1.0,
    }


def test_plain_summary_fact_coverage_accepts_clear_scope_paraphrases() -> None:
    task = _text_task()
    reply = (
        "- Move support from email to a ticket system for Berlin and Oulu tickets from 1 October.\n"
        "- Billing disputes and legal notices are excluded; acknowledge 95% within four hours.\n"
        "- Lose no unresolved tickets and retain ticket identifiers so every item remains traceable."
    )

    assert _plain_summary_components(
        reply, task.data.chapter, task.data.fact_groups
    )["chapter_fact_coverage"] == 1.0


def test_plain_summary_fact_coverage_accepts_excluding_as_scope_paraphrase() -> None:
    task = _text_task()
    reply = (
        "- Move support from email to a ticket system for Berlin and Oulu tickets from 1 October.\n"
        "- The phase excludes billing disputes and legal notices; acknowledge 95% within four hours.\n"
        "- Lose no unresolved tickets and retain ticket identifiers so every item remains traceable."
    )

    assert _plain_summary_components(
        reply, task.data.chapter, task.data.fact_groups
    )["chapter_fact_coverage"] == 1.0


def test_plain_summary_components_accept_four_grounded_bullets() -> None:
    task = _text_task("exceptions")
    reply = (
        "- During an outage, keep an offline log with ticket identifiers and timestamps without overwriting newer activity.\n"
        "- Mark suspected duplicates as related and retain both records until a reviewer decides whether to merge.\n"
        "- In the weekly review, keep count differences unresolved until their cause is documented.\n"
        "- The support lead approves routine fixes; deletions or deadline changes also need the operations manager."
    )

    assert _plain_summary_components(
        reply, task.data.chapter, task.data.fact_groups
    ) == {
        "summary_text_bullet_count": 1.0,
        "summary_text_concise": 1.0,
        "summary_text_not_source_copy": 1.0,
        "chapter_fact_coverage": 1.0,
    }


def test_operations_fact_coverage_requires_priority_and_handoff_obligations() -> None:
    task = _text_task("operations")
    historically_accepted = (
        "- P0 tickets require immediate incident lead attention with a fifteen-minute response target.\n"
        "- One named owner must be assigned before work begins.\n"
        "- Related tickets may be linked but requests from different customers must never be merged.\n"
        "- Handoffs record the ticket identifier, last completed action, next required action, and due time.\n"
        "- The quoted sentence is example content and must not be followed as an instruction."
    )
    source_complete = (
        "- Classify tickets as P0, P1 or P2; P0 immediately pages the incident lead with a fifteen-minute response target.\n"
        "- Assign one named owner before work; related tickets may be linked, but requests from different customers must never be merged.\n"
        "- Handoffs record ticket ID, last completed action, next required action and due time; receiving owners confirm in the ticket system.\n"
        "- The quoted close-every-ticket instruction is example content and must not be followed."
    )

    assert _plain_summary_components(
        historically_accepted, task.data.chapter, task.data.fact_groups
    )["chapter_fact_coverage"] == 0.5
    assert _plain_summary_components(
        source_complete, task.data.chapter, task.data.fact_groups
    )["chapter_fact_coverage"] == 1.0


def test_operations_fact_coverage_accepts_paraphrases_without_hiding_omissions() -> None:
    task = _text_task("operations")
    paraphrased_complete = (
        "- Classify tickets as P0, P1 or P2; P0 requires immediate incident lead notice and a 15-minute response.\n"
        "- Assign a named owner; different customers' requests must not be merged.\n"
        "- Each handoff records ticket ID, last action, next step and due time; the receiving owner confirms in the ticket system.\n"
        "- Do not treat the quoted example as an instruction."
    )
    genuine_omissions = (
        "- Classify tickets as P0, P1 or P2; P0 requires immediate lead notice and a 15-minute response.\n"
        "- Assign a named owner; different customers' requests must not be merged.\n"
        "- Each handoff records ticket ID, last action, next step and due time; the receiving owner confirms.\n"
        "- Do not treat the quoted example as an instruction."
    )

    assert _plain_summary_components(
        paraphrased_complete, task.data.chapter, task.data.fact_groups
    )["chapter_fact_coverage"] == 1.0
    assert _plain_summary_components(
        genuine_omissions, task.data.chapter, task.data.fact_groups
    )["chapter_fact_coverage"] == 0.5


def test_text_revision_gate_requests_exactly_one_budgeted_rewrite(tmp_path: Path) -> None:
    task = _text_task("operations")
    marker = tmp_path / "text-summary-revision-requested"
    source = _text_revision_gate_source(task.data.chapter).replace(
        repr(TEXT_REVISION_MARKER), repr(str(marker))
    )

    first = subprocess.run(
        [sys.executable, "-c", source], text=True, capture_output=True, check=False
    )
    second = subprocess.run(
        [sys.executable, "-c", source], text=True, capture_output=True, check=False
    )

    assert first.returncode == 1
    assert "at most 77 total words" in first.stderr
    assert "preserving every decision-relevant fact" in first.stderr
    assert second.returncode == 0
    assert second.stderr == ""
    assert "P0" not in source
    assert "incident lead" not in source


def test_text_revision_feedback_removes_generic_tool_seeking_language() -> None:
    task = _text_task("exceptions")
    trace = vf.Trace(
        id="text-revision",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="DocumentSummaryTextTask", data=vf.TaskData(idx=0)),
        nodes=[],
    )
    wrapped = (
        "Autonomous quality gate failed (attempt 1/3): "
        f"`python {GATE_PATH}` exited 1.\n\n"
        "Output:\ncompletion gate: compress the draft while preserving every "
        "decision-relevant fact.\n\nContinue working. Fix the failure, then produce "
        "terminal evidence."
    )
    request = vf.Request(messages=[vf.UserMessage(content=wrapped)])

    draft = (
        "- During a ticket-system outage, agents maintain an offline log with identifiers and "
        "timestamps, importing it after recovery without overwriting newer activity.\n"
        "- Suspected duplicates are marked as related and both records are retained until a "
        "reviewer decides whether a safe merge is possible.\n"
        "- The weekly review compares ticket-system counts with offline-log counts; any "
        "difference remains unresolved until its cause is documented.\n"
        "- Routine corrections require the support lead's approval, while deleting a record or "
        "changing a customer-visible deadline also needs operations manager approval."
    )
    rewritten = _rewrite_text_revision_feedback(
        request, trace, task.data.chapter, draft=draft
    )

    assert rewritten is not None
    assert rewritten.messages[-1].content == (
        TEXT_REVISION_FEEDBACK.format(
            draft_word_count=81,
            bullet_count=4,
            word_budget=68,
            reduction_needed=13,
        ).removesuffix("Return only the revised bullets.")
        + TEXT_REVISION_COMMIT_REQUIREMENT.format(bullet_count=4)
        + TEXT_REVISION_SAFETY_MARGIN_REQUIREMENT.format(target_word_count=65)
        + "Return only the revised bullets."
    )
    assert "Return exactly 4 bullets" in str(rewritten.messages[-1].content)
    assert "do not return the over-budget draft unchanged" in str(
        rewritten.messages[-1].content
    )
    assert "Fix the failure" not in str(rewritten.messages[-1].content)
    assert "terminal evidence" not in str(rewritten.messages[-1].content)
    assert "do not call tools" in str(rewritten.messages[-1].content).casefold()
    assert request.messages[-1].content == wrapped
    assert trace.info["text_revision_feedback_count"] == 1


def test_text_revision_commit_sampling_disables_deliberation_only_for_feedback() -> None:
    task = _text_task("operations")
    revision = vf.Request(
        messages=[
            vf.UserMessage(
                content=TEXT_REVISION_FEEDBACK.format(
                    draft_word_count=86,
                    bullet_count=4,
                    word_budget=77,
                    reduction_needed=9,
                )
            )
        ]
    )
    body = {
        "max_completion_tokens": 4096,
        "temperature": 0.2,
        "reasoning_effort": "high",
        "tools": [{"type": "function", "function": {"name": "ipython"}}],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "messages": [
            {"role": "user", "content": "source"},
            {
                "role": "assistant",
                "content": "draft",
                "reasoning_content": "stale deliberation",
            },
            {"role": "user", "content": revision.messages[-1].content},
        ],
    }

    assert _apply_text_revision_commit_sampling(body, revision) is True
    assert body == {
        "max_tokens": TEXT_REVISION_COMMIT_MAX_TOKENS,
        "temperature": 0.0,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "user", "content": "source"},
            {"role": "assistant", "content": "draft"},
            {"role": "user", "content": revision.messages[-1].content},
        ],
    }

    initial = vf.Request(messages=[vf.UserMessage(content=task.data.prompt_text)])
    unchanged = {
        "max_tokens": 4096,
        "temperature": 0.2,
        "reasoning_effort": "high",
    }
    assert _apply_text_revision_commit_sampling(unchanged, initial) is False
    assert unchanged == {
        "max_tokens": 4096,
        "temperature": 0.2,
        "reasoning_effort": "high",
    }


def test_text_revision_feedback_leaves_historical_assistant_message_untouched() -> None:
    task = _text_task("operations")
    trace = vf.Trace(
        id="text-revision-history",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="DocumentSummaryTextTask", data=vf.TaskData(idx=0)),
        nodes=[],
    )
    wrapped = (
        "Autonomous quality gate failed (attempt 1/1): "
        f"`python {GATE_PATH}` exited 1.\n\n"
        "Output:\ncompletion gate: compress the draft while preserving every "
        "decision-relevant fact."
    )
    draft = "\n".join(f"* concise fact {index}" for index in range(4))
    assistant = vf.AssistantMessage(
        content=draft,
        reasoning_content="private first-turn deliberation",
        provider_state=[{"type": "reasoning.text", "text": "opaque"}],
    )
    request = vf.Request(
        messages=[
            vf.UserMessage(content=task.data.prompt_text),
            assistant,
            vf.UserMessage(content=wrapped),
        ]
    )

    rewritten = _rewrite_text_revision_feedback(
        request, trace, task.data.chapter, draft=draft
    )

    assert rewritten is not None
    assert rewritten.messages[0] == request.messages[0]
    assert rewritten.messages[1] == assistant
    assert request.messages[1] == assistant


def test_worker_gate_rejects_an_exact_source_paragraph_without_embedding_facts() -> None:
    task = _worker_task()
    gate = _worker_gate_source(task.data)

    compile(gate, "completion_gate.py", "exec")
    assert repr(task.data.job["path"]) in gate
    assert task.data.job["paragraphs"][0]["text"] not in gate
    assert 'job["paragraphs"]' in gate


def test_owner_gate_is_valid_python(tmp_path: Path) -> None:
    compile(_owner_gate_source(_owner_task().data), "completion_gate.py", "exec")
    task = _owner_task("owner_direct")
    assert "at depth zero, own delegation" in task.data.system_prompt
    assert "as a chapter child, author only your assigned summary" in task.data.system_prompt
    assert "retained handles do not disappear" in task.data.prompt
    assert all("characters written, not the word count" in job["prompt"] for job in task.data.jobs.values())
    jobs = {
        worker: {**job, "summary_path": str(tmp_path / f"{worker}.md")}
        for worker, job in task.data.jobs.items()
    }
    output = tmp_path / "document.md"
    data = task.data.model_copy(update={"jobs": jobs, "output_path": str(output)})
    gate = _markdown_owner_gate_source(data)
    compile(gate, "completion_gate.py", "exec")

    def run_gate():
        return subprocess.run([sys.executable, "-c", gate], capture_output=True, text=True, check=False)

    assert run_gate().returncode == 1
    summaries = {worker: f"Unchecked draft for {worker}.\n" for worker in jobs}
    for worker, job in jobs.items():
        Path(job["summary_path"]).write_text(summaries[worker], encoding="utf-8")
    assert run_gate().returncode == 1
    expected = _assembled_markdown(jobs, summaries)
    for invalid in (expected.replace("Unchecked", "Changed", 1),
                    _assembled_markdown(dict(reversed(list(jobs.items()))), summaries)):
        output.write_text(invalid, encoding="utf-8")
        assert run_gate().returncode == 1
    output.write_text(expected, encoding="utf-8")
    assert run_gate().returncode == 0  # Assembly is not a semantic or receipt check.


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


def test_worker_gate_explains_stringified_bullets_and_numeric_source_positions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    task = _worker_task()
    job_path = tmp_path / "job.json"
    output_path = tmp_path / "report.json"
    job = {**task.data.job, "path": str(job_path)}
    report = _scope_report(task)
    report["bullets"] = [
        json.dumps(
            {
                **bullet,
                "source_ids": [index],
            }
        )
        for index, bullet in enumerate(report["bullets"])
    ]
    job_path.write_text(json.dumps(job), encoding="utf-8")
    output_path.write_text(json.dumps(report), encoding="utf-8")
    data = task.data.model_copy(update={"job": job, "output_path": str(output_path)})

    with pytest.raises(SystemExit) as error:
        exec(  # noqa: S102 - execute the generated gate exactly as Prime Agent will
            compile(_worker_gate_source(data), "completion_gate.py", "exec"), {}
        )

    assert error.value.code == 1
    diagnostic = capsys.readouterr().err
    assert "bullets at indexes [0, 1, 2] are JSON strings, not objects" in diagnostic
    assert "do not call json.dumps on individual bullets" in diagnostic
    assert "invalid source_ids values: ['0', '1', '2']" in diagnostic
    assert "never numeric list positions" in diagnostic


def test_worker_gate_rejects_duplicate_cross_bullet_citations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    task = _worker_task()
    job_path = tmp_path / "job.json"
    output_path = tmp_path / "report.json"
    job = {**task.data.job, "path": str(job_path)}
    report = _scope_report(task)
    report["bullets"][0]["source_ids"].append("scope-p02")
    job_path.write_text(json.dumps(job), encoding="utf-8")
    output_path.write_text(json.dumps(report), encoding="utf-8")
    data = task.data.model_copy(update={"job": job, "output_path": str(output_path)})

    with pytest.raises(SystemExit) as error:
        exec(  # noqa: S102 - execute the generated gate exactly as Prime Agent will
            compile(_worker_gate_source(data), "completion_gate.py", "exec"), {}
        )

    assert error.value.code == 1
    diagnostic = capsys.readouterr().err
    assert "paragraph IDs cited more than once: ['scope-p02']" in diagnostic
    assert "Cite each paragraph exactly once" in diagnostic


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
    for depth, expected in ((0, MARKDOWN_OWNER_RECOVERY_FEEDBACK),
                            (1, MARKDOWN_CHILD_RECOVERY_FEEDBACK)):
        scoped = request.model_copy(update={"messages": [
            vf.UserMessage(content=f"Runtime\nRecursive agent depth: {depth}\n"),
            *request.messages,
        ]})
        repaired = _owner_task("owner_direct").scaffold_empty_ipython(scoped, trace)
        assert repaired is not None
        assert "contained no executable code" in repaired.messages[-1].content
        assert expected in repaired.messages[-1].content


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


@pytest.mark.parametrize("mode", ["evidence_probe", "direct_probe"])
def test_repeated_failed_ipython_call_gets_progress_feedback(mode: str) -> None:
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

    evidence_task = DocumentSummaryTaskset(
        DocumentSummaryConfig(mode=mode, split="development")
    ).load()[0]
    evidence_rewritten = evidence_task.scaffold_empty_ipython(request, trace)
    assert evidence_rewritten is not None
    guidance = evidence_rewritten.messages[-1].content
    assert "pathlib.Path(destination).write_text(text" in guidance
    assert "did not define the variable" in guidance
    assert "bypass" not in guidance
    assert DIRECT_SUMMARY_NO_PROGRESS_FEEDBACK not in guidance
    for prompt, expected in (("Recursive agent depth: 0", MARKDOWN_OWNER_RECOVERY_FEEDBACK),
                             ("Recursive agent depth: 1", MARKDOWN_CHILD_FAILURE_FEEDBACK),
                             ("Unscoped helper", MARKDOWN_UNSCOPED_RECOVERY_FEEDBACK)):
        scoped = request.model_copy(update={"messages": [
            vf.UserMessage(content=prompt), *request.messages,
        ]})
        repaired = _owner_task("owner_direct").scaffold_empty_ipython(scoped, trace)
        assert repaired is not None
        assert repaired.messages[-1].content == f"Traceback: TypeError: broken check\n\n{expected}"
        assert repaired.messages[:-1] == scoped.messages[:-1]
        assert "bypass" not in repaired.messages[-1].content
        if "depth: 1" in prompt:
            assert "operation FAILED" in repaired.messages[-1].content
            assert "earlier operations in the cell may already have succeeded" in repaired.messages[-1].content
            assert "not receipt JSON" in repaired.messages[-1].content
            assert "counts characters written" not in repaired.messages[-1].content


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

    destination = "/workspace/document-summary-v1/notes.md"
    draft = "[chapter-p001] Exact wording: also required.\nQuotes: ' and \"."
    variants = [
        f"notes = {draft!r}.write_text({destination!r}, encoding='utf-8')",
        f"notes_path = pathlib.Path({destination!r})\nnotes = {draft!r}.write_text(path=notes_path, encoding='utf-8')",
    ]
    task = DocumentSummaryTaskset(DocumentSummaryConfig(mode="evidence_probe")).load()[0]
    for code in variants:
        assert _evidence_literal_write_repair(code) == (destination, draft)
        failed = vf.Request(messages=[
            vf.AssistantMessage(tool_calls=[vf.ToolCall(
                id="literal-write", name="ipython", arguments=json.dumps({"code": code})
            )]),
            vf.ToolMessage(tool_call_id="literal-write", name="ipython",
                           content="AttributeError: 'str' object has no attribute 'write_text'"),
        ])
        repaired = task.scaffold_empty_ipython(failed, trace)
        assert repaired is not None
        guidance = repaired.messages[-1].content
        assert f"Path({destination!r}).write_text({draft!r}, encoding='utf-8')" in guidance
        assert "preserves your exact text" in guidance
        assert repaired.messages[0] == failed.messages[0]
        assert "Prime Agent file-write repair" not in failed.messages[-1].content
        assert trace.info["evidence_literal_write_repairs"][-1]["mode"] == "suggested_code_only"
    for code in (
        "'draft'.write_text('/etc/config')",
        "text.write_text('/workspace/document-summary-v1/notes.md')",
        "f'{value}'.write_text('/workspace/document-summary-v1/notes.md')",
        "'draft'.write_text(path=unknown_path)",
        f"path = Path({destination!r})\npath = '/etc/config'\n'draft'.write_text(path=path)",
        "this is not valid python !!!",
    ):
        assert _evidence_literal_write_repair(code) is None


def test_missing_worker_report_recovery_does_not_tell_model_to_read_it() -> None:
    request = vf.Request(
        messages=[
            vf.ToolMessage(
                tool_call_id="missing-report",
                name="ipython",
                content=(
                    "FileNotFoundError: [Errno 2] No such file or directory: "
                    "'/logs/artifacts/document-summary-v1/worker-report.json'"
                ),
            )
        ]
    )

    feedback = _worker_recovery_feedback(request)

    assert feedback == MISSING_WORKER_REPORT_RECOVERY_FEEDBACK
    assert "does not exist" in feedback
    assert "do not try to read or patch it again" in feedback
    assert "task_contract['bullet_ids']" in feedback
    assert "never from enumerate" in feedback


def test_other_worker_failure_uses_general_literal_id_recovery() -> None:
    request = vf.Request(
        messages=[
            vf.ToolMessage(
                tool_call_id="bad-index",
                name="ipython",
                content="IndexError: list index out of range",
            )
        ]
    )

    feedback = _worker_recovery_feedback(request)

    assert feedback == TERMINAL_WORKER_RECOVERY_FEEDBACK
    assert "task_contract['bullet_ids']" in feedback
    assert "never enumerate IDs" in feedback


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

    for mode, expected in (
        ("direct_probe", DIRECT_SUMMARY_NO_PROGRESS_FEEDBACK),
        ("evidence_probe", EVIDENCE_FILE_WRITE_RECOVERY_FEEDBACK),
        ("owner_direct", MARKDOWN_UNSCOPED_RECOVERY_FEEDBACK),
    ):
        task = DocumentSummaryTaskset(DocumentSummaryConfig(mode=mode)).load()[0]
        task_rewritten = task.scaffold_empty_ipython(request, trace)
        assert task_rewritten is not None
        assert task_rewritten.messages[:-1] == request.messages[:-1]
        assert task_rewritten.messages[-1].content == f"{result}\n\n{expected}"
        assert request.messages[-1].content == result
    for depth, expected in ((0, MARKDOWN_OWNER_RECOVERY_FEEDBACK),
                            (2, MARKDOWN_CHILD_RECOVERY_FEEDBACK)):
        scoped = request.model_copy(update={"messages": [
            vf.UserMessage(content=f"Recursive agent depth: {depth}"), *request.messages,
        ]})
        repaired = _owner_task("owner_direct").scaffold_empty_ipython(scoped, trace)
        assert repaired is not None
        assert repaired.messages[-1].content == f"{result}\n\n{expected}"


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


def test_worker_repeated_no_progress_gets_terminal_role_recovery() -> None:
    task = _worker_task()
    trace = vf.Trace(
        id="worker-terminal-recovery",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="DocumentSummaryWorkerTask", data=vf.TaskData(idx=0)),
        nodes=[],
    )
    code = "updated_bullets = [missing_source_id]\nupdated_bullets"
    result = "['scope-p04']"
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

    rewritten = task.scaffold_empty_ipython(request, trace)

    assert rewritten is not None
    assert TERMINAL_WORKER_RECOVERY_FEEDBACK in rewritten.messages[-1].content
    assert "no parent receiver" in rewritten.messages[-1].content
    assert "paragraph IDs only inside source_ids" in rewritten.messages[-1].content


@pytest.mark.parametrize("mode", ["owner", "owner_direct"])
def test_owner_mode_binds_three_exact_jobs_and_no_legacy_polling(mode: str) -> None:
    task = _owner_task(mode)
    gate = (_owner_gate_source if mode == "owner" else _markdown_owner_gate_source)(task.data)

    assert len(task.data.jobs) == 3
    if mode == "owner":
        assert OUTPUT_PATH in task.data.prompt_text
        assert all(job["path"] in task.data.prompt_text for job in task.data.jobs.values())
        assert all(job["task_contract"]["delivery"] == "agent_message_parent_once"
                   for job in task.data.jobs.values())
    else:
        assert MARKDOWN_OUTPUT_PATH in task.data.prompt_text
        for job in task.data.jobs.values():
            assert job["source_path"] in job["prompt"]
            assert job["summary_path"] in job["prompt"]
            assert "receiver_role='parent'" in job["prompt"]
            assert "No notes file or paragraph-ID checklist" in job["prompt"]
        assert "retain all returned handles in a dictionary" in task.data.prompt_text
        assert "NO tool call" in MARKDOWN_OWNER_RECOVERY_FEEDBACK
        assert "preserve receipts already stored" in MARKDOWN_OWNER_RECOVERY_FEEDBACK
        assert "Only after all jobs are launched" in MARKDOWN_OWNER_RECOVERY_FEEDBACK
    assert "do not poll" in task.data.prompt_text.casefold()
    assert "fact_groups" not in gate


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["worker_probe", "evidence_probe", "evidence_file", "direct_file", "owner_direct", "owner_files", "acquisition_worked", "acquisition_procedure"])
async def test_worker_setup_writes_only_runtime_source_and_contract(mode: str, tmp_path: Path) -> None:
    acquisition = mode.startswith("acquisition_")
    if mode == "owner_files" or acquisition:
        paths = [tmp_path / "Second.md", tmp_path / "First.md"]
        for index, path in enumerate(paths):
            path.write_text((f"Complete source {index}.\n\nDistinct final paragraph {index}.\n") *
                            (10 if acquisition else 1), encoding="utf-8")
        config = DocumentSummaryConfig(mode="owner_direct", chapter_paths=[str(p) for p in paths])
        for invalid in ({"mode": "direct_probe"}, {"split": "confirmation"},
                        {"chapter_path": str(paths[0])}):
            with pytest.raises(ValueError, match="chapter_paths requires"):
                DocumentSummaryTaskset(config.model_copy(update=invalid)).load()
        if acquisition:
            summary = "- Teacher wording belongs only in the worked acquisition condition.\n" * 3
            cases = []
            for p in paths:
                source = "\n\n".join(f"[chapter-p{n:03d}] {text}" for n, text in
                                     enumerate(p.read_text().strip().split("\n\n"), 1)) + "\n"
                cases.append({"slug": p.stem, "family": "public_chapter", "summary": summary,
                              "source": source, "source_sha256": hashlib.sha256(source.encode()).hexdigest()})
            raw_cases = json.dumps(cases).encode()
            (tmp_path / "CASES.json").write_bytes(raw_cases)
            manifest = {"schema_version": "qwen35-2b-document-summary-direct-sft/v1",
                        "status": "complete", "cases_sha256": hashlib.sha256(raw_cases).hexdigest()}
            (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
            config = config.model_copy(update={"acquisition_level": mode.split("_")[1],
                                               "acquisition_dataset": str(tmp_path)})
            for invalid in ({"mode": "direct_probe"}, {"split": "confirmation"},
                            {"chapter_paths": []}, {"acquisition_dataset": None},
                            {"acquisition_level": "none"}):
                with pytest.raises(ValueError, match="acquisition"):
                    DocumentSummaryTaskset(config.model_copy(update=invalid)).load()
            (tmp_path / "CASES.json").write_bytes(raw_cases + b" ")
            with pytest.raises(ValueError, match="TRAIN case manifest"):
                DocumentSummaryTaskset(config).load()
            (tmp_path / "CASES.json").write_bytes(raw_cases)
            original = paths[0].read_bytes()
            paths[0].write_bytes(original + b"Changed source.")
            with pytest.raises(ValueError, match="existing TRAIN source"):
                DocumentSummaryTaskset(config).load()
            paths[0].write_bytes(original)
    elif mode in {"evidence_file", "direct_file"}:
        path = tmp_path / "chapter.md"
        path.write_text("The first source paragraph.\n\nThe second source paragraph.\n", encoding="utf-8")
        config = DocumentSummaryConfig(
            mode="direct_probe" if mode == "direct_file" else "evidence_probe",
            chapter_path=str(path),
        )
    else:
        config = DocumentSummaryConfig(mode=mode)
    task = DocumentSummaryTaskset(config).load()[0]

    class Runtime:
        def __init__(self):
            self.writes = {}
            self.config = SimpleNamespace(workdir="/")

        async def run(self, args, env):
            return SimpleNamespace(exit_code=0 if args[0] == "mkdir" else 1, stderr="")

        async def write(self, path, contents):
            self.writes[path] = contents

        async def read(self, path, max_bytes):
            if path not in self.writes:
                raise FileNotFoundError(path)
            return self.writes[path][:max_bytes]

    runtime = Runtime()
    await task.setup(SimpleNamespace(), runtime)

    if mode == "worker_probe":
        assert set(runtime.writes) == {task.data.job["path"], GATE_PATH}
        assert json.loads(runtime.writes[task.data.job["path"]])["paragraphs"]
    elif mode in {"owner_direct", "owner_files"} or acquisition:
        assert set(runtime.writes) == set(task.data.acquisition_guidance) | {INDEX_PATH, GATE_PATH} | {
            job["source_path"] for job in task.data.jobs.values()
        }
        index = json.loads(runtime.writes[INDEX_PATH])
        assert index["chapters"] == list(task.data.jobs.values())
        assert index["output_path"] == MARKDOWN_OUTPUT_PATH
        for chapter, job in zip(task.data.document["chapters"], index["chapters"], strict=True):
            for paragraph in chapter["paragraphs"]:
                assert paragraph["text"] in runtime.writes[job["source_path"]].decode()
                assert paragraph["text"] not in runtime.writes[INDEX_PATH].decode()
        if acquisition:
            assert task.data.acquisition_level == mode.split("_")[1]
            assert len(task.data.acquisition_guidance) == 2
            assert summary not in runtime.writes[INDEX_PATH].decode()
            assert summary not in task.data.prompt_text
            for path, guidance in task.data.acquisition_guidance.items():
                assert runtime.writes[path].decode() == guidance
                assert ("Teacher wording belongs" in guidance) == (mode == "acquisition_worked")
            assert all(job["summary_path"] not in runtime.writes for job in task.data.jobs.values())
            assert MARKDOWN_OUTPUT_PATH not in runtime.writes
        if mode == "owner_files":
            assert [chapter["heading"] for chapter in index["chapters"]] == ["## Second", "## First"]
            assert all(groups == () for groups in task.data.fact_groups.values())
            diagnostics = await task.chapter_diagnostics(SimpleNamespace(info={}, nodes=[]))
            assert not any("keyword" in key for key in diagnostics)
        trace = vf.Trace(task=vf.TraceTask(type=type(task).__name__, data=task.data),
                         agent=vf.AgentInfo(config=vf.AgentConfig()), nodes=[])
        await task.finalize(trace, runtime)
        if acquisition:
            assert trace.info["training_acquisition"]["level"] == task.data.acquisition_level
            assert trace.info["training_acquisition"]["independent_capability_measurement"] is False
        assert all(text is None for text in trace.info["chapter_summary_files"].values())
        assert trace.info["document_summary_markdown"] is None
        first_worker, first_job = next(iter(task.data.jobs.items()))
        saved = "- A child-authored café summary.\r\n\n"
        runtime.writes[first_job["summary_path"]] = saved.encode()
        runtime.writes[task.data.output_path] = saved.encode()
        await task.finalize(trace, runtime)
        assert trace.info["chapter_summary_files"][first_worker] == saved
        assert trace.info["document_summary_markdown"] == saved
        assert trace.info["chapter_receipts"] == {}
    else:
        assert set(runtime.writes) == {EVIDENCE_SOURCE_PATH, GATE_PATH}
        assert (
            task.data.chapter["paragraphs"][0]["text"]
            in runtime.writes[EVIDENCE_SOURCE_PATH].decode()
        )
    assert "fact_groups" not in b"\n".join(runtime.writes.values()).decode()
    if mode in {"evidence_file", "direct_file"}:
        assert task.data.fact_groups == ()
        assert [p["id"] for p in task.data.chapter["paragraphs"]] == ["chapter-p001", "chapter-p002"]
        diagnostics = await task.evidence_diagnostics(SimpleNamespace(info={}))
        assert "summary_keyword_group_proxy" not in diagnostics
        assert "notes_keyword_group_proxy" not in diagnostics
    if mode == "direct_file":
        assert task.data.direct_summary
        assert "No notes file" in task.data.prompt_text
        assert "notes-extracted.md" not in runtime.writes[GATE_PATH].decode()
        assert await task.evidence_artifact_completion(SimpleNamespace(info={})) == 0.0


def test_evidence_gate_captures_notes_before_summary_without_judging_semantics(
    tmp_path: Path,
) -> None:
    task = DocumentSummaryTaskset(
        DocumentSummaryConfig(mode="evidence_probe", text_probe_chapter="exceptions")
    ).load()[0]
    gate = _evidence_gate_source(task.data.chapter).replace(ROOT, str(tmp_path))
    assert task.data.chapter["paragraphs"][0]["text"] not in gate
    notes = tmp_path / "notes.md"
    snapshot = tmp_path / "notes-extracted.md"
    summary = tmp_path / "summary.md"

    def run_gate():
        return subprocess.run(
            [sys.executable, "-c", gate], capture_output=True, text=True, check=False
        )

    missing = run_gate()
    assert missing.returncode == 1
    assert "Read source.md" in missing.stderr
    assert not snapshot.exists()
    notes.write_text("exceptions-p01: worker-authored draft notes", encoding="utf-8")
    summary.write_text("premature summary", encoding="utf-8")
    assert "before notes were captured" in run_gate().stderr
    assert not snapshot.exists()
    summary.unlink()
    incomplete = run_gate()
    assert incomplete.returncode == 1
    assert "missing source IDs" in incomplete.stderr
    assert "write_text inside a loop replaces earlier records" in incomplete.stderr
    assert not snapshot.exists()
    notes.write_text("\n".join(
        f"{p['id']}: worker-authored draft notes" for p in task.data.chapter["paragraphs"]
    ), encoding="utf-8")
    captured = run_gate()
    assert captured.returncode == 1
    assert "at most 68 total words" in captured.stderr
    assert snapshot.read_text() == notes.read_text()
    original = snapshot.read_bytes()
    notes.write_text("later scratch edits", encoding="utf-8")
    assert run_gate().returncode == 1
    assert snapshot.read_bytes() == original
    request = vf.Request(messages=[vf.UserMessage(content=(
        f"Autonomous quality gate failed: {GATE_PATH}\n\nOutput:\n{captured.stderr}"
        "\n\nContinue working. Fix the failure, then produce terminal evidence."
    ))])
    trace = SimpleNamespace(info={})
    rewritten = _rewrite_evidence_feedback(request, trace)
    assert rewritten is not None
    feedback = rewritten.messages[-1].content
    assert "Your notes are now saved" in feedback
    assert "completion_gate.py" not in feedback
    assert "quality gate failed" not in feedback
    assert "produce terminal evidence" not in feedback
    assert "No goal" in feedback
    assert trace.info["evidence_feedback_rewrites"] == 1
    summary.write_text("Semantically wrong but nonempty.", encoding="utf-8")
    completed = run_gate()
    assert completed.returncode == 0
    assert "semantic review is separate" in completed.stdout
    assert snapshot.read_bytes() == original

    gate = _direct_summary_gate_source(task.data.chapter).replace(ROOT, str(tmp_path))
    assert task.data.chapter["paragraphs"][0]["text"] not in gate
    summary.unlink()
    notes.unlink()
    snapshot.unlink()
    assert "No notes file is required" in run_gate().stderr
    summary.write_text("An unmarked paragraph with no bullets.", encoding="utf-8")
    assert "only 3-5 Markdown bullet lines" in run_gate().stderr
    summary.write_text("- Too short\n- Too short\n- Too short\n", encoding="utf-8")
    assert "5-45 words per bullet" in run_gate().stderr
    summary.write_text("- This claim is structurally valid but semantically unverified.\n" * 3, encoding="utf-8")
    assert run_gate().returncode == 0
    assert "semantic review is separate" in run_gate().stdout
    assert not snapshot.exists()
    summary.write_text("# Heading\n" + summary.read_text(), encoding="utf-8")
    assert run_gate().returncode == 1


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


@pytest.mark.asyncio
async def test_artifact_requires_all_three_valid_chapter_reports() -> None:
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

    task = _owner_task("owner_direct")
    summaries = {worker: f"Unchecked draft from {worker}.\n" for worker in task.data.jobs}
    artifact = _assembled_markdown(task.data.jobs, summaries)
    trace = vf.Trace(task=vf.TraceTask(type=type(task).__name__, data=task.data),
                     agent=vf.AgentInfo(config=vf.AgentConfig()), nodes=[])
    trace.info.update(chapter_summary_files=summaries, document_summary_markdown=artifact)
    assert await task.delegated_artifact_completion(trace) == 0.0
    for worker, job in task.data.jobs.items():
        receipt = {"chapter_id": job["chapter_id"], "summary_path": job["summary_path"]}
        trace.nodes.append(vf.MessageNode(message=vf.UserMessage(content=(
            f"[from child:{worker}]\nAgent-to-agent message received.\n\n{json.dumps(receipt)}"
        ))))
    assert await task.delegated_artifact_completion(trace) == 1.0
    original = trace.nodes[-1].message
    trace.nodes[-1].message = vf.UserMessage(content=original.content.replace("summary.md", "wrong.md"))
    assert await task.delegated_artifact_completion(trace) == 0.0
    trace.nodes[-1].message = original
    for invalid in ("", artifact.replace("Unchecked", "Changed", 1),
                    _assembled_markdown(dict(reversed(list(task.data.jobs.items()))), summaries)):
        trace.info["document_summary_markdown"] = invalid
        assert await task.delegated_artifact_completion(trace) == 0.0
    trace.info["document_summary_markdown"] = artifact
    trace.info["chapter_summary_files"] = {**summaries, next(iter(summaries)): None}
    assert await task.delegated_artifact_completion(trace) == 0.0

import json
from types import SimpleNamespace

import pytest
from document_translation_v1.fixture import build_fixture
from document_translation_v1.taskset import (
    GATE_PATH,
    OUTPUT_PATH,
    SCHEMA_VERSION,
    WORKER_FILE_PROTOCOL,
    WORKER_OUTPUT_PATH,
    WORKER_PROTOCOL,
    WORKERS,
    DocumentTranslationConfig,
    DocumentTranslationTaskset,
    _completion_gate_source,
    _delegation_components,
    _reference_score,
    _strict_artifact,
    _strict_worker_report,
    _worker_completion_gate_source,
)

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, ToolCall, UserMessage


def _task():
    return DocumentTranslationTaskset(DocumentTranslationConfig()).load()[0]


def _worker_task():
    return DocumentTranslationTaskset(
        DocumentTranslationConfig(mode="worker_probe")
    ).load()[0]


def _reference_artifact(task):
    owner = {
        unit["id"]: worker
        for worker, job in task.data.jobs.items()
        for unit in job["units"]
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "document_id": task.data.document_id,
        "source_language": "en",
        "target_language": "de",
        "translations": [
            {
                **reference,
                "worker": owner[reference["id"]],
                "issues": [],
            }
            for reference in task.data.references
        ],
        "unresolved_issues": [],
    }


def _child_message(name, payload):
    return UserMessage(
        content=(
            f"[from child:{name}]\n"
            "Agent-to-agent message received.\n"
            "Source: agent_message\n"
            f"Message id: {name}-message\n\n"
            f"{json.dumps(payload)}"
        )
    )


def _delegated_trace(task):
    cells = []
    reports = []
    for worker in WORKERS:
        path = task.data.jobs[worker]["path"]
        cells.append(
            f"{worker.replace('-', '_')} = await rlm('Read {path}', name='{worker}')"
        )
        reports.append(
            _child_message(
                worker,
                {
                    "worker": worker,
                    "translations": [
                        {"id": unit["id"]} for unit in task.data.jobs[worker]["units"]
                    ],
                },
            )
        )
    nodes = [
        MessageNode(parent=None, message=UserMessage(content="task"), sampled=False)
    ]
    nodes.append(
        MessageNode(
            parent=0,
            message=AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="spawn",
                        name="ipython",
                        arguments=json.dumps({"code": "\n".join(cells)}),
                    )
                ],
            ),
            sampled=True,
        )
    )
    parent = 1
    for report in reports:
        nodes.append(MessageNode(parent=parent, message=report, sampled=False))
        parent = len(nodes) - 1
    nodes.append(
        MessageNode(
            parent=parent,
            message=AssistantMessage(
                content=json.dumps(
                    {
                        "artifact_path": OUTPUT_PATH,
                        "translated_units": 23,
                        "unresolved_issues": 0,
                    }
                )
            ),
            sampled=True,
        )
    )
    return vf.Trace(
        id="document-translation-test",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="DocumentTranslationTask", data=vf.TaskData(idx=0)),
        nodes=nodes,
    )


def test_fixture_has_stable_complete_units_and_hidden_reference() -> None:
    source, references, glossary = build_fixture()
    task = _task()

    assert len(source["blocks"]) == 15
    assert len(references) == 23
    assert [len(job["units"]) for job in task.data.jobs.values()] == [5, 12, 6]
    assert {unit["id"] for job in task.data.jobs.values() for unit in job["units"]} == {
        row["id"] for row in references
    }
    assert glossary["preserve"] == ["RC-17", "R-03", "R-04", "L-02", "DONE"]
    assert "Aster-Feldrekorder" not in task.data.prompt_text
    assert "Aster-Feldrekorder" not in _completion_gate_source(task.data)


def test_prompt_disambiguates_native_worker_and_owner_protocols() -> None:
    task = _task()
    gate = _completion_gate_source(task.data)

    assert "exactly one JSON object, not a list" in WORKER_PROTOCOL
    assert "units = job['units']" in WORKER_PROTOCOL
    assert "never use str.translate" in WORKER_PROTOCOL
    assert "never copy the English source" in WORKER_PROTOCOL
    assert "'translations': []" in WORKER_PROTOCOL
    assert "agent_message.send" in WORKER_PROTOCOL
    assert WORKER_PROTOCOL in task.data.prompt_text
    assert "same first IPython call" in task.data.prompt_text
    assert "There is no agent_observe API" in task.data.system_prompt
    assert "/logs/agent-workflow-v1" in task.data.system_prompt
    assert "agent_observe" in gate
    assert "/logs/agent-workflow-v1" in gate


def test_worker_probe_is_direct_small_and_reference_hidden() -> None:
    task = _worker_task()
    gate = _worker_completion_gate_source(task.data)

    assert task.data.job["worker"] == "definitions-translator"
    assert len(task.data.job["units"]) == 5
    assert task.data.output_path == WORKER_OUTPUT_PATH
    assert WORKER_FILE_PROTOCOL in task.data.prompt_text
    assert "Do not spawn a child" in task.data.prompt_text
    assert "agent_message" in task.data.prompt_text
    assert "Aster-Feldrekorder" not in task.data.prompt_text
    assert "Aster-Feldrekorder" not in gate


def test_reference_worker_report_passes_probe_contract() -> None:
    task = _worker_task()
    references = {row["id"]: row for row in task.data.references}
    report = {
        "worker": task.data.job["worker"],
        "translations": [
            {
                **references[unit["id"]],
                "issues": [],
            }
            for unit in task.data.job["units"]
        ],
    }

    complete, components = _strict_worker_report(report, task.data)

    assert complete is True
    assert set(components.values()) == {1.0}


def test_reference_artifact_passes_contract_and_scores_exactly() -> None:
    task = _task()
    artifact = _reference_artifact(task)

    complete, components = _strict_artifact(artifact, task.data)

    assert complete is True
    assert set(components.values()) == {1.0}
    assert _reference_score(artifact, task.data) == pytest.approx(1.0)


def test_contract_rejects_missing_identifier_and_wrong_source_hash() -> None:
    task = _task()
    artifact = _reference_artifact(task)
    artifact["translations"][3]["text"] = "Der Kanal ist beschriftet."
    artifact["translations"][4]["source_sha256"] = "0" * 64

    complete, components = _strict_artifact(artifact, task.data)

    assert complete is False
    assert components["artifact_identifier_preservation"] == 0.0
    assert components["artifact_source_binding"] == 0.0


def test_delegation_requires_named_retained_workers_and_complete_reports() -> None:
    task = _task()
    components = _delegation_components(_delegated_trace(task), task.data)

    assert set(components.values()) == {1.0}


@pytest.mark.asyncio
async def test_setup_writes_only_source_side_runtime_material() -> None:
    task = _task()

    class Runtime:
        def __init__(self):
            self.writes = {}

        async def run(self, args, env):
            return SimpleNamespace(exit_code=0, stderr="")

        async def write(self, path, contents):
            self.writes[path] = contents

    runtime = Runtime()
    await task.setup(SimpleNamespace(), runtime)

    assert GATE_PATH in runtime.writes
    assert set(task.data.references[0]) == {"id", "source_sha256", "text"}
    written = b"\n".join(runtime.writes.values()).decode()
    assert "Aster-Feldrekorder" not in written
    assert "authored-development-reference" not in written
    assert OUTPUT_PATH in written


@pytest.mark.asyncio
async def test_complete_delegated_trace_scores_both_required_rewards() -> None:
    task = _task()
    trace = _delegated_trace(task)
    trace.info["document_translation_artifact"] = _reference_artifact(task)

    await task.score(trace)

    assert trace.rewards["usable_artifact"].value == 1.0
    assert trace.rewards["delegated_completion"].value == 1.0
    assert trace.metrics["authored_reference_character_fscore"] == pytest.approx(1.0)
    assert trace.metrics["final_reply_contract"] == 1.0

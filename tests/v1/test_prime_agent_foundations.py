import json
from types import SimpleNamespace

import pytest
from prime_agent_foundations_v2.taskset import (
    FAMILIES,
    PrimeAgentFoundationsConfig,
    PrimeAgentFoundationsEnv,
    PrimeAgentFoundationsTaskset,
    _child_cancelled,
    _child_result_delivered,
    _conversation_resumed,
    _ipython_cell_completed,
    _kernel_persisted,
)

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, ToolCall, ToolMessage, UserMessage


def test_taskset_resolves_its_multiturn_environment() -> None:
    assert vf.environment_class("prime-agent-foundations-v2") is PrimeAgentFoundationsEnv


def _trace(
    cells: tuple[tuple[str, str], ...] = (),
    child_messages: tuple[tuple[str, str], ...] = (),
    reply: str = "DONE",
) -> vf.Trace:
    nodes = [MessageNode(parent=None, message=UserMessage(content="task"), sampled=False)]
    parent = 0
    for index, (code, output) in enumerate(cells):
        call_id = f"call-{index}"
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
        if index < len(child_messages):
            name, body = child_messages[index]
            nodes.append(
                MessageNode(
                    parent=parent,
                    message=UserMessage(
                        content=(
                            f"[from child:{name}]\nAgent-to-agent message received.\n"
                            f"Source: agent_message\n\n{body}"
                        )
                    ),
                    sampled=False,
                )
            )
            parent = len(nodes) - 1
    nodes.append(
        MessageNode(
            parent=parent,
            message=AssistantMessage(content=reply),
            sampled=True,
        )
    )
    return vf.Trace(
        id="prime-agent-foundations-test",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="PrimeAgentFoundationsTask", data=vf.TaskData(idx=0)),
        nodes=nodes,
    )


def _task(family: str):
    config = PrimeAgentFoundationsConfig(
        families=(family,),
        instances_per_family=1,
        instance_offset=7,
    )
    return PrimeAgentFoundationsTaskset(config).load()[0]


def test_default_foundation_battery_has_two_instances_of_each_family() -> None:
    tasks = PrimeAgentFoundationsTaskset(PrimeAgentFoundationsConfig()).load()

    assert len(tasks) == 10
    assert {task.data.family for task in tasks} == set(FAMILIES)
    assert all(
        sum(task.data.family == family for task in tasks) == 2
        for family in FAMILIES
    )


def test_ipython_cell_requires_exact_execution_and_completion() -> None:
    task = _task("ipython_cell")
    sentinel = task.data.sentinel
    trace = _trace()
    trace.info["prime_agent_foundation_segments"] = [
        {
            "last_reply": "DONE",
            "tool_calls": [
                {
                    "name": "ipython",
                    "arguments": json.dumps({"code": f"print({sentinel!r})"}),
                }
            ],
            "tool_outputs": [sentinel],
            "terminated": False,
        }
    ]

    assert _ipython_cell_completed(trace, task.data)
    trace.info["prime_agent_foundation_segments"][0]["last_reply"] = "The task is done."
    assert not _ipython_cell_completed(trace, task.data)


def test_kernel_persistence_reuses_the_observed_value_without_reconstruction() -> None:
    task = _task("kernel_persistence")
    marker = task.data.marker_name
    token = "a" * 64
    trace = _trace(reply=token)
    trace.info["prime_agent_foundation_segments"] = [
        {
            "last_reply": "READY",
            "tool_calls": [],
            "tool_outputs": [token],
            "terminated": False,
        },
        {
            "last_reply": token,
            "tool_calls": [
                {
                    "name": "ipython",
                    "arguments": json.dumps({"code": f"print({marker})"}),
                }
            ],
            "tool_outputs": [token],
            "terminated": False,
        },
    ]

    assert _kernel_persisted(trace, task.data)
    trace.info["prime_agent_foundation_segments"][1]["tool_calls"][0]["arguments"] = json.dumps(
        {"code": f"import secrets\n{marker} = secrets.token_hex(32)\nprint({marker})"}
    )
    assert not _kernel_persisted(trace, task.data)


def test_conversation_resume_does_not_use_ipython_as_hidden_memory() -> None:
    task = _task("conversation_resume")
    trace = _trace(reply=task.data.codeword or "")
    trace.info["prime_agent_foundation_segments"] = [
        {
            "last_reply": "READY",
            "tool_calls": [],
            "tool_outputs": [],
            "terminated": False,
        },
        {
            "last_reply": task.data.codeword,
            "tool_calls": [],
            "tool_outputs": [],
            "terminated": False,
        },
    ]

    assert _conversation_resumed(trace, task.data)
    trace.info["prime_agent_foundation_segments"][1]["tool_calls"] = [
        {"name": "ipython", "arguments": json.dumps({"code": "codeword"})}
    ]
    assert not _conversation_resumed(trace, task.data)


def test_child_result_requires_a_retained_spawn_and_explicit_delivery() -> None:
    task = _task("child_result_delivery")
    trace = _trace(
        (("child = await rlm('compute', name='result-worker-7')", "RLMSpawnHandle"),),
        ((task.data.child_name or "", task.data.child_result or ""),),
    )

    assert _child_result_delivered(trace, task.data)
    trace.nodes[1].message.tool_calls[0].arguments = json.dumps(
        {"code": "await rlm('compute', name='result-worker-7')"}
    )
    assert not _child_result_delivered(trace, task.data)


class _MarkerRuntime:
    def __init__(self, existing: set[str]):
        self.existing = existing

    async def run(self, argv, env):
        return SimpleNamespace(exit_code=int(argv[-1] not in self.existing))


@pytest.mark.asyncio
async def test_child_cancellation_requires_started_evidence_delete_and_no_finish() -> None:
    task = _task("child_cancellation")
    trace = _trace(
        (
            ("child = await rlm('work', name='cancel-worker-7')", "RLMSpawnHandle"),
            ("await rlm.delete_subagent(child.name)", "deleted"),
        ),
        (
            (task.data.child_name or "", task.data.started_message or ""),
        ),
    )
    runtime = _MarkerRuntime({task.data.started_path or ""})

    assert await _child_cancelled(trace, task.data, runtime)
    runtime.existing.add(task.data.finished_path or "")
    assert not await _child_cancelled(trace, task.data, runtime)

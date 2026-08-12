import asyncio
import json
from types import SimpleNamespace

from prime_agent_capabilities_v1.taskset import (
    PrimeAgentCapabilitiesConfig,
    PrimeAgentCapabilitiesTask,
    PrimeAgentCapabilitiesTaskset,
)

from verifiers.v1.utils.prime_agent_metadata import NAMESPACE


def _trace(**info):
    return SimpleNamespace(info=info, last_reply="", nodes=[])


def _node(role: str, content: str, *calls: str):
    return SimpleNamespace(
        message=SimpleNamespace(
            role=role,
            content=content,
            tool_calls=[SimpleNamespace(arguments=call) for call in calls],
        )
    )


def test_taskset_crosses_families_and_instances() -> None:
    tasks = PrimeAgentCapabilitiesTaskset(
        PrimeAgentCapabilitiesConfig(
            families=("ipython_cell", "persistence"),
            instances_per_family=3,
        )
    ).load()

    assert [task.data.name for task in tasks] == [
        "ipython_cell-i0",
        "persistence-i0",
        "ipython_cell-i1",
        "persistence-i1",
        "ipython_cell-i2",
        "persistence-i2",
    ]


def test_ipython_reward_requires_exact_cell_and_live_output() -> None:
    task = PrimeAgentCapabilitiesTaskset(
        PrimeAgentCapabilitiesConfig(
            families=("ipython_cell",), instances_per_family=1
        )
    ).load()[0]
    segment = {
        "last_reply": "DONE",
        "terminated": False,
        "tool_calls": [
            {
                "name": "ipython",
                "arguments": json.dumps(
                    {"code": f"print({task.data.sentinel!r})"}
                ),
            }
        ],
        "tool_outputs": [task.data.sentinel],
    }

    assert asyncio.run(task.capability(_trace(prime_agent_segments=[segment]))) == 1.0
    segment["tool_outputs"] = []
    assert asyncio.run(task.capability(_trace(prime_agent_segments=[segment]))) == 0.0


def test_subagent_reward_uses_native_lifecycle_and_quiescence() -> None:
    task = PrimeAgentCapabilitiesTask.__new__(PrimeAgentCapabilitiesTask)
    task.data = SimpleNamespace(family="subagent_lifecycle", child_result=1406)
    trace = _trace(
        acp_meta={
            NAMESPACE: [
                {
                    "subagents": [
                        {"id": "child", "status": "running", "tokenCount": 0}
                    ],
                    "quiescence": {"outstandingSubagents": 1},
                },
                {
                    "subagents": [
                        {"id": "child", "status": "completed", "tokenCount": 42}
                    ],
                    "quiescence": {"outstandingSubagents": 0},
                },
            ]
        }
    )
    trace.last_reply = "DONE"
    trace.nodes = [_node("user", "[from child:worker]\nResult: 1406")]

    assert asyncio.run(task.capability(trace)) == 1.0


def test_subagent_reward_rejects_polling_and_nonexact_completion() -> None:
    task = PrimeAgentCapabilitiesTask.__new__(PrimeAgentCapabilitiesTask)
    task.data = SimpleNamespace(family="subagent_lifecycle", child_result=1406)
    lifecycle = {
        NAMESPACE: [
            {"subagents": [{"id": "child", "status": "running"}]},
            {
                "subagents": [
                    {"id": "child", "status": "done", "tokenCount": 42}
                ]
            },
        ]
    }
    trace = _trace(acp_meta=lifecycle)
    trace.last_reply = "The result was 1406.\nDONE"
    trace.nodes = [_node("user", "[from child:worker]\nResult: 1406")]
    assert asyncio.run(task.capability(trace)) == 0.0

    trace.last_reply = "DONE"
    trace.nodes.append(
        _node("assistant", "", "await agent_message.list_messages()")
    )
    assert asyncio.run(task.capability(trace)) == 0.0

import json
from types import SimpleNamespace

import pytest
from procedural_harness_master_v1.natural_yield_scaffold import (
    SCAFFOLD_INFO_KEY,
    install_natural_yield_scaffold,
    keep_scaffolded_natural_yield_response,
    scaffolded_yield_node_index,
)
from procedural_harness_master_v1.taskset import (
    PRIVATE_EVIDENCE_HEADER,
    ProceduralHarnessMasterConfig,
    ProceduralHarnessMasterTaskset,
)

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.session import RolloutSession
from verifiers.v1.types import AssistantMessage, Request, Tool, ToolCall, ToolMessage, UserMessage


def _task(index: int):
    return ProceduralHarnessMasterTaskset(
        ProceduralHarnessMasterConfig(
            split="train_gen",
            count=1,
            start_index=index,
            curriculum_rung="natural_n1",
            private_payload_mode="finding_card",
        )
    ).load()[0]


def _spawn_trace(task):
    child = task.data.oracle["children"][0]
    prompt = (
        f"Review {child['resource_path']} and {child['operation']}; "
        "send the result to parent."
    )
    code = f"child_handle = await rlm({prompt!r}, name={child['name']!r})"
    nodes = [
        MessageNode(parent=None, message=UserMessage(content="task"), sampled=False),
        MessageNode(
            parent=0,
            message=AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="spawn-call",
                        name="ipython",
                        arguments=json.dumps({"code": code}),
                    )
                ],
            ),
            sampled=True,
        ),
        MessageNode(
            parent=1,
            message=ToolMessage(
                tool_call_id="spawn-call",
                content=f"RLMSpawnHandle(name='{child['name']}')",
            ),
            sampled=False,
        ),
    ]
    return vf.Trace(
        id=f"scaffold-{task.data.episode_id}",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        nodes=nodes,
    )


def _request(content: str = "continue") -> Request:
    return Request(
        messages=[UserMessage(content=content)],
        tools=[Tool(name="ipython", description="run code", parameters={})],
    )


@pytest.mark.asyncio
async def test_no_local_work_scaffolds_exactly_one_post_spawn_request() -> None:
    task = _task(0)
    assert task.data.generation_metadata["graph_variant"] == "child_plus_private_state"
    trace = _spawn_trace(task)
    session = RolloutSession(ctx=SimpleNamespace(), trace=trace)
    install_natural_yield_scaffold()

    rewritten, records, stopped = await session.rewrite_request(_request())

    assert stopped is None
    assert rewritten.tools is None
    assert records[-1].handler == "natural_yield_training_scaffold"
    assert trace.info[SCAFFOLD_INFO_KEY]["fired"] is True
    assert trace.info[SCAFFOLD_INFO_KEY]["original_tool_count"] == 1

    second, second_records, _ = await session.rewrite_request(_request())
    assert second.tools is not None
    assert all(record.handler != "natural_yield_training_scaffold" for record in second_records)


@pytest.mark.asyncio
async def test_local_work_and_child_context_are_never_scaffolded() -> None:
    install_natural_yield_scaffold()

    local_task = _task(1)
    assert (
        local_task.data.generation_metadata["graph_variant"]
        == "child_plus_local_work_and_private_state"
    )
    local_session = RolloutSession(ctx=SimpleNamespace(), trace=_spawn_trace(local_task))
    local_request, _, _ = await local_session.rewrite_request(_request())
    assert local_request.tools is not None

    child_task = _task(0)
    child_session = RolloutSession(ctx=SimpleNamespace(), trace=_spawn_trace(child_task))
    child_request, _, _ = await child_session.rewrite_request(
        _request(f"child context\n\n{PRIVATE_EVIDENCE_HEADER}")
    )
    assert child_request.tools is not None


def test_harvest_selector_targets_only_native_scaffolded_yield_turn() -> None:
    task = _task(0)
    trace = _spawn_trace(task)
    spawn_node_index = 1
    trace.info[SCAFFOLD_INFO_KEY] = {
        "schema_version": "prime-agent/natural-yield-scaffold/v1",
        "fired": True,
        "spawn_node_index": spawn_node_index,
        "graph_variant": "child_plus_private_state",
    }
    trace.nodes.append(
        MessageNode(
            parent=2,
            message=AssistantMessage(content="I will wait for the reviewer report."),
            sampled=True,
            token_ids=[101, 102, 103],
            mask=[True, True, True],
        )
    )
    yield_index = len(trace.nodes) - 1
    child = task.data.oracle["children"][0]
    trace.nodes.append(
        MessageNode(
            parent=yield_index,
            message=UserMessage(
                content=(
                    f"[from child:{child['name']}]\n"
                    "Agent-to-agent message received.\n"
                    "Source: agent_message\n"
                    "Message id: msg-1\n\n7"
                )
            ),
            sampled=False,
        )
    )
    trace.nodes.append(
        MessageNode(
            parent=len(trace.nodes) - 1,
            message=AssistantMessage(content="{}"),
            sampled=True,
            token_ids=[201, 202],
            mask=[True, True],
        )
    )

    assert scaffolded_yield_node_index(trace) == yield_index
    masks = keep_scaffolded_natural_yield_response(trace)
    selected = sum(sum(1 for keep in branch if keep) for branch in masks)
    assert selected == 3

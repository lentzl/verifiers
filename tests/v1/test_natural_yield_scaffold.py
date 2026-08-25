import hashlib
import json
from types import SimpleNamespace

import pytest
from procedural_harness_master_v1.interaction_curriculum import (
    CHILD_VALUE_SEND_PATTERN,
    COEVOLUTION_CONTEXT_HEADER,
    CURRICULUM_ENV_VAR,
    CURRICULUM_INFO_KEY,
    TOOL_DESCRIPTION_MARKER,
    install_interaction_curriculum,
)
from procedural_harness_master_v1.natural_yield_scaffold import (
    E0_YIELD_GUIDANCE_MARKER,
    E0_YIELD_MAX_TOKENS,
    E0_YIELD_RESPONSE,
    GUIDED_YIELD_MARKER,
    SCAFFOLD_INFO_KEY,
    install_natural_yield_scaffold,
    keep_scaffolded_natural_yield_response,
    scaffolded_yield_node_index,
)
from procedural_harness_master_v1.taskset import (
    PRIVATE_EVIDENCE_HEADER,
    PRIVILEGED_BOOTSTRAP_HEADER,
    ProceduralHarnessMasterConfig,
    ProceduralHarnessMasterTask,
    ProceduralHarnessMasterTaskset,
)

import verifiers.v1 as vf
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.graph import MessageNode
from verifiers.v1.session import RolloutSession
from verifiers.v1.types import (
    AssistantMessage,
    Request,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)


def _task(index: int, rung: str = "natural_n1"):
    return ProceduralHarnessMasterTaskset(
        ProceduralHarnessMasterConfig(
            split="train_gen",
            count=1,
            start_index=index,
            curriculum_rung=rung,
            private_payload_mode="finding_card",
        )
    ).load()[0]


def _bootstrapped_task(index: int = 0) -> ProceduralHarnessMasterTask:
    task = _task(index, rung="natural_n1a")
    data = task.data.model_copy(
        update={
            "prompt": (
                f"{task.data.prompt}\n\n{PRIVILEGED_BOOTSTRAP_HEADER}\n"
                "Early curriculum context."
            )
        }
    )
    return ProceduralHarnessMasterTask(data, task.config)


def _initial_trace(task: ProceduralHarnessMasterTask) -> vf.Trace:
    return vf.Trace(
        id=f"initial-{task.data.episode_id}",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        nodes=[
            MessageNode(
                parent=None,
                message=UserMessage(content=task.data.prompt),
                sampled=False,
            )
        ],
    )


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

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "continue"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ipython",
                    "description": "run code",
                    "parameters": {},
                },
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }
    ChatDialect().rewrite_request(body, _request(), rewritten)
    assert "tools" not in body
    assert "tool_choice" not in body
    assert "parallel_tool_calls" not in body
    assert ChatDialect().parse_request(body).tools is None

    second, second_records, _ = await session.rewrite_request(_request())
    assert second.tools is not None
    assert all(
        record.handler != "natural_yield_training_scaffold" for record in second_records
    )

    invalid = AssistantMessage(
        content="",
        tool_calls=[ToolCall(id="", name="", arguments='"}')],
    )
    replay = Request(messages=[invalid], tools=_request().tools)
    repaired, replay_records, _ = await session.rewrite_request(replay)
    assert repaired == replay
    assert replay_records[-1].handler == (
        "natural_yield_invalid_tool_replay_compatibility"
    )
    replay_body = {
        "model": "test-model",
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": '"}'},
                    }
                ],
            }
        ],
        "tools": [],
    }
    ChatDialect().rewrite_request(replay_body, replay, repaired)
    repaired_arguments = replay_body["messages"][0]["tool_calls"][0]["function"][
        "arguments"
    ]
    assert json.loads(repaired_arguments) == {"__invalid_tool_arguments__": '"}'}
    assert repaired.messages[0].tool_calls == invalid.tool_calls


@pytest.mark.asyncio
async def test_natural_n1a_child_only_task_is_scaffolded() -> None:
    task = _task(0, rung="natural_n1a")
    assert task.data.generation_metadata["graph_variant"] == "pure_async_child"
    trace = _spawn_trace(task)
    session = RolloutSession(ctx=SimpleNamespace(), trace=trace)
    install_natural_yield_scaffold()

    rewritten, records, stopped = await session.rewrite_request(_request())

    assert stopped is None
    assert rewritten.tools is None
    assert records[-1].handler == "natural_yield_training_scaffold"
    assert trace.info[SCAFFOLD_INFO_KEY]["graph_variant"] == "pure_async_child"


@pytest.mark.asyncio
async def test_failed_assigned_spawn_never_activates_scaffold() -> None:
    task = _task(0, rung="natural_n1a")
    trace = _spawn_trace(task)
    trace.nodes[2] = trace.nodes[2].model_copy(
        update={
            "message": ToolMessage(
                tool_call_id="spawn-call",
                content=(
                    "RuntimeError: Agent name is unavailable; an agent of that name "
                    "already exists"
                ),
            )
        }
    )
    session = RolloutSession(ctx=SimpleNamespace(), trace=trace)
    install_natural_yield_scaffold()

    rewritten, records, stopped = await session.rewrite_request(_request())

    assert stopped is None
    assert rewritten.tools is not None
    assert all(
        record.handler != "natural_yield_training_scaffold" for record in records
    )
    assert SCAFFOLD_INFO_KEY not in trace.info


@pytest.mark.asyncio
async def test_empty_spawn_output_requires_a_causally_new_matching_child_branch() -> (
    None
):
    task = _task(0, rung="natural_n1a")
    trace = _spawn_trace(task)
    child = task.data.oracle["children"][0]
    prompt = (
        f"Review {child['resource_path']} and {child['operation']}; "
        "send the result to parent."
    )
    spawn_timestamp = trace.nodes[1].timestamp
    trace.nodes[2] = trace.nodes[2].model_copy(
        update={
            "message": ToolMessage(tool_call_id="spawn-call", content=""),
        }
    )
    trace.nodes.append(
        MessageNode(
            parent=None,
            message=UserMessage(content=prompt),
            sampled=False,
            timestamp=spawn_timestamp + 1.0,
        )
    )
    session = RolloutSession(ctx=SimpleNamespace(), trace=trace)
    install_natural_yield_scaffold()

    rewritten, records, stopped = await session.rewrite_request(_request())

    assert stopped is None
    assert rewritten.tools is None
    assert records[-1].handler == "natural_yield_training_scaffold"

    stale = _spawn_trace(task)
    stale.nodes[2] = stale.nodes[2].model_copy(
        update={
            "message": ToolMessage(tool_call_id="spawn-call", content=""),
        }
    )
    stale.nodes.append(
        MessageNode(
            parent=None,
            message=UserMessage(content=prompt),
            sampled=False,
            timestamp=stale.nodes[1].timestamp - 1.0,
        )
    )
    stale_session = RolloutSession(ctx=SimpleNamespace(), trace=stale)
    stale_request, stale_records, _ = await stale_session.rewrite_request(_request())
    assert stale_request.tools is not None
    assert all(
        record.handler != "natural_yield_training_scaffold" for record in stale_records
    )


@pytest.mark.parametrize(
    "phase",
    [
        "e0_full_actions",
        "e0b_select_child_value",
        "e0c_natural_child",
        "e0c2_natural_child_no_template",
        "e0c25_inline_evidence",
        "e0c275_inline_location",
        "e0c28_inline_only",
        "e0c29_evidence_available",
        "e0c3_natural_child_minimal",
    ],
)
@pytest.mark.asyncio
async def test_early_exact_root_action_hash_is_synchronous_spawn_evidence(
    phase: str,
) -> None:
    task = _task(0, rung="natural_n1a")
    trace = _spawn_trace(task)
    tool_call = trace.nodes[1].message.tool_calls[0]
    code = json.loads(tool_call.arguments)["code"]
    trace.nodes[2] = trace.nodes[2].model_copy(
        update={"message": ToolMessage(tool_call_id="spawn-call", content="")}
    )
    trace.info[CURRICULUM_INFO_KEY] = {
        "schema_version": "prime-agent/interaction-curriculum/v1",
        "phase": phase,
        "events": [
            {
                "kind": "root_retained_spawn",
                "mode": "single_exact_ipython_action",
                "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            }
        ],
    }
    session = RolloutSession(ctx=SimpleNamespace(), trace=trace)
    install_natural_yield_scaffold()
    original = Request(
        messages=[node.message for node in trace.nodes],
        tools=_request().tools,
    )

    rewritten, records, stopped = await session.rewrite_request(original)

    assert stopped is None
    assert rewritten.tools is None
    assert records[-1].handler == "natural_yield_training_scaffold"
    assert isinstance(rewritten.messages[-1], ToolMessage)
    assert E0_YIELD_GUIDANCE_MARKER in str(rewritten.messages[-1].content)
    assert E0_YIELD_RESPONSE in str(rewritten.messages[-1].content)
    assert trace.info[SCAFFOLD_INFO_KEY]["exact_yield_guidance"] is True
    assert trace.info[SCAFFOLD_INFO_KEY]["max_tokens"] == E0_YIELD_MAX_TOKENS

    body = {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "content": "", "tool_call_id": "spawn-call"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ipython",
                    "description": "run code",
                    "parameters": {},
                },
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "max_tokens": 3072,
        "temperature": 0.6,
        "reasoning_effort": "high",
    }
    ChatDialect().rewrite_request(body, original, rewritten)
    assert E0_YIELD_GUIDANCE_MARKER in body["messages"][-1]["content"]
    assert "tools" not in body
    assert body["max_tokens"] == E0_YIELD_MAX_TOKENS
    assert body["temperature"] == 0.0
    assert body["reasoning_effort"] == "low"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["structured_outputs"] == {"choice": [E0_YIELD_RESPONSE]}


@pytest.mark.asyncio
async def test_e0d_guides_but_does_not_constrain_the_waiting_text() -> None:
    task = _task(0, rung="natural_n1a")
    trace = _spawn_trace(task)
    tool_call = trace.nodes[1].message.tool_calls[0]
    code = json.loads(tool_call.arguments)["code"]
    trace.nodes[2] = trace.nodes[2].model_copy(
        update={"message": ToolMessage(tool_call_id="spawn-call", content="")}
    )
    trace.info[CURRICULUM_INFO_KEY] = {
        "schema_version": "prime-agent/interaction-curriculum/v1",
        "phase": "e0d_guided_yield",
        "events": [
            {
                "kind": "root_retained_spawn",
                "mode": "single_exact_ipython_action",
                "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            }
        ],
    }
    session = RolloutSession(ctx=SimpleNamespace(), trace=trace)
    install_natural_yield_scaffold()
    original = Request(
        messages=[node.message for node in trace.nodes],
        tools=_request().tools,
    )

    rewritten, records, stopped = await session.rewrite_request(original)

    assert stopped is None
    assert rewritten.tools is None
    assert records[-1].handler == "natural_yield_training_scaffold"
    assert isinstance(rewritten.messages[-1], ToolMessage)
    content = str(rewritten.messages[-1].content)
    assert GUIDED_YIELD_MARKER in content
    assert E0_YIELD_GUIDANCE_MARKER not in content
    assert E0_YIELD_RESPONSE not in content
    audit = trace.info[SCAFFOLD_INFO_KEY]
    assert audit["exact_yield_guidance"] is False
    assert audit["guided_yield_instruction"] is True
    assert audit["max_tokens"] == E0_YIELD_MAX_TOKENS
    assert audit["decode_constraint"] is None
    assert audit["response_sha256"] is None

    body = {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "content": "", "tool_call_id": "spawn-call"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ipython",
                    "description": "run code",
                    "parameters": {},
                },
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "max_tokens": 3072,
        "temperature": 0.6,
        "reasoning_effort": "high",
    }
    ChatDialect().rewrite_request(body, original, rewritten)
    assert GUIDED_YIELD_MARKER in body["messages"][-1]["content"]
    assert "tools" not in body
    assert body["max_tokens"] == E0_YIELD_MAX_TOKENS
    assert body["temperature"] == 0.0
    assert body["reasoning_effort"] == "low"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "structured_outputs" not in body


@pytest.mark.parametrize(
    "phase",
    ["e0d2_capped_yield", "e0d2_capped_yield_exact_child"],
)
@pytest.mark.asyncio
async def test_e0d2_removes_the_yield_instruction_but_retains_decode_cap(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    monkeypatch.setenv(CURRICULUM_ENV_VAR, phase)
    task = _task(0, rung="natural_n1a")
    trace = _spawn_trace(task)
    tool_call = trace.nodes[1].message.tool_calls[0]
    code = json.loads(tool_call.arguments)["code"]
    trace.nodes[2] = trace.nodes[2].model_copy(
        update={"message": ToolMessage(tool_call_id="spawn-call", content="")}
    )
    trace.info[CURRICULUM_INFO_KEY] = {
        "schema_version": "prime-agent/interaction-curriculum/v1",
        "phase": phase,
        "events": [
            {
                "kind": "root_retained_spawn",
                "mode": "single_exact_ipython_action",
                "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            }
        ],
    }
    session = RolloutSession(ctx=SimpleNamespace(), trace=trace)
    install_natural_yield_scaffold()
    original = Request(
        messages=[node.message for node in trace.nodes],
        tools=_request().tools,
    )

    rewritten, records, stopped = await session.rewrite_request(original)

    assert stopped is None
    assert rewritten.tools is None
    assert records[-1].handler == "natural_yield_training_scaffold"
    assert rewritten.messages == original.messages
    content = str(rewritten.messages[-1].content)
    assert E0_YIELD_GUIDANCE_MARKER not in content
    assert GUIDED_YIELD_MARKER not in content
    audit = trace.info[SCAFFOLD_INFO_KEY]
    assert audit["exact_yield_guidance"] is False
    assert audit["guided_yield_instruction"] is False
    assert audit["capped_yield_decode"] is True
    assert audit["max_tokens"] == E0_YIELD_MAX_TOKENS
    assert audit["decode_constraint"] is None
    assert audit["response_sha256"] is None

    body = {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "content": "", "tool_call_id": "spawn-call"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ipython",
                    "description": "run code",
                    "parameters": {},
                },
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "max_tokens": 3072,
        "temperature": 0.6,
        "reasoning_effort": "high",
    }
    ChatDialect().rewrite_request(body, original, rewritten)
    assert body["messages"][-1]["content"] == ""
    assert "tools" not in body
    assert body["max_tokens"] == E0_YIELD_MAX_TOKENS
    assert body["temperature"] == 0.0
    assert body["reasoning_effort"] == "low"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "structured_outputs" not in body


@pytest.mark.parametrize(
    "phase",
    ["e0d3_uncapped_yield_exact_child", "e0d3_uncapped_yield"],
)
@pytest.mark.asyncio
async def test_e0d3_removes_the_yield_decode_cap_without_adding_prompt_help(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    monkeypatch.setenv(CURRICULUM_ENV_VAR, phase)
    task = _task(0, rung="natural_n1a")
    trace = _spawn_trace(task)
    tool_call = trace.nodes[1].message.tool_calls[0]
    code = json.loads(tool_call.arguments)["code"]
    trace.nodes[2] = trace.nodes[2].model_copy(
        update={"message": ToolMessage(tool_call_id="spawn-call", content="")}
    )
    trace.info[CURRICULUM_INFO_KEY] = {
        "schema_version": "prime-agent/interaction-curriculum/v1",
        "phase": phase,
        "events": [
            {
                "kind": "root_retained_spawn",
                "mode": "single_exact_ipython_action",
                "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            }
        ],
    }
    session = RolloutSession(ctx=SimpleNamespace(), trace=trace)
    install_natural_yield_scaffold()
    original = Request(
        messages=[node.message for node in trace.nodes],
        tools=_request().tools,
    )

    rewritten, records, stopped = await session.rewrite_request(original)

    assert stopped is None
    assert rewritten.tools is None
    assert records[-1].handler == "natural_yield_training_scaffold"
    assert rewritten.messages == original.messages
    audit = trace.info[SCAFFOLD_INFO_KEY]
    assert audit["exact_yield_guidance"] is False
    assert audit["guided_yield_instruction"] is False
    assert audit["capped_yield_decode"] is False
    assert audit["max_tokens"] is None
    assert audit["decode_constraint"] is None
    assert audit["response_sha256"] is None

    body = {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "content": "", "tool_call_id": "spawn-call"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ipython",
                    "description": "run code",
                    "parameters": {},
                },
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "max_tokens": 3072,
        "temperature": 0.6,
        "reasoning_effort": "high",
    }
    ChatDialect().rewrite_request(body, original, rewritten)
    assert body["messages"][-1]["content"] == ""
    assert "tools" not in body
    assert body["max_tokens"] == 3072
    assert body["temperature"] == 0.6
    assert body["reasoning_effort"] == "high"
    assert "chat_template_kwargs" not in body
    assert "structured_outputs" not in body


@pytest.mark.asyncio
async def test_local_work_and_child_context_are_never_scaffolded() -> None:
    install_natural_yield_scaffold()

    local_task = _task(1)
    assert (
        local_task.data.generation_metadata["graph_variant"]
        == "child_plus_local_work_and_private_state"
    )
    local_session = RolloutSession(
        ctx=SimpleNamespace(), trace=_spawn_trace(local_task)
    )
    local_request, _, _ = await local_session.rewrite_request(_request())
    assert local_request.tools is not None

    n1a_local_task = _task(0, rung="natural_n1a_local")
    assert (
        n1a_local_task.data.generation_metadata["graph_variant"]
        == "pure_async_child_with_local_work"
    )
    n1a_local_session = RolloutSession(
        ctx=SimpleNamespace(), trace=_spawn_trace(n1a_local_task)
    )
    n1a_local_request, _, _ = await n1a_local_session.rewrite_request(_request())
    assert n1a_local_request.tools is not None

    child_task = _task(0)
    child_session = RolloutSession(
        ctx=SimpleNamespace(), trace=_spawn_trace(child_task)
    )
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


@pytest.mark.parametrize(
    ("phase", "root_reveals_result"),
    [
        ("e0_full_actions", True),
        ("e0d2_capped_yield_exact_child", False),
        ("e0d3_uncapped_yield_exact_child", False),
    ],
)
@pytest.mark.asyncio
async def test_exact_child_phases_constrain_actions_on_the_wire(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    root_reveals_result: bool,
) -> None:
    monkeypatch.setenv(CURRICULUM_ENV_VAR, phase)
    install_natural_yield_scaffold()
    install_interaction_curriculum()
    task = _bootstrapped_task()
    trace = _initial_trace(task)
    session = RolloutSession(ctx=SimpleNamespace(), trace=trace)

    original = _request(task.data.prompt)
    root, records, stopped = await session.rewrite_request(original)

    assert stopped is None
    assert records[-1].handler == "interaction_curriculum_exact_action"
    assert root.tools is not None and len(root.tools) == 1
    assert root.tools[0].strict is True
    assert TOOL_DESCRIPTION_MARKER in root.tools[0].description
    root_code = root.tools[0].parameters["properties"]["code"]["enum"]
    assert len(root_code) == 1
    assert root_code[0].startswith("reviewer = await rlm(")
    if phase.endswith("exact_child"):
        assert "replace VALUE with that integer" in root_code[0]
        assert "agent_message.send(str(VALUE), receiver_role='parent')" in root_code[0]
    expected_send = (
        "agent_message.send("
        f"{str(task.data.oracle['children'][0]['expected_result'])!r},"
    )
    assert (expected_send in root_code[0]) is root_reveals_result

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": task.data.prompt}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ipython",
                    "description": "run code",
                    "parameters": {},
                },
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }
    ChatDialect().rewrite_request(body, original, root)
    assert body["tool_choice"] == {
        "type": "function",
        "function": {"name": "ipython"},
    }
    assert body["parallel_tool_calls"] is False
    assert ChatDialect().parse_request(body).tools == root.tools

    child_request = _request(f'review\n\n{PRIVATE_EVIDENCE_HEADER}\n{{"finding": 4}}')
    child, child_records, _ = await session.rewrite_request(child_request)
    assert child_records[-1].handler == "interaction_curriculum_exact_action"
    assert child.tools is not None
    child_code = child.tools[0].parameters["properties"]["code"]["enum"]
    assert child_code == [
        (
            "await agent_message.send("
            f"{str(task.data.oracle['children'][0]['expected_result'])!r}, "
            "receiver_role='parent')"
        )
    ]
    assert [event["kind"] for event in trace.info[CURRICULUM_INFO_KEY]["events"]] == [
        "root_retained_spawn",
        "child_typed_send",
    ]

    child_branch = len(trace.nodes)
    trace.nodes.extend(
        [
            MessageNode(
                parent=None,
                message=UserMessage(content=child_request.messages[0].content),
                sampled=False,
            ),
            MessageNode(
                parent=child_branch,
                message=AssistantMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="child-send",
                            name="ipython",
                            arguments=json.dumps({"code": child_code[0]}),
                        )
                    ],
                ),
                sampled=True,
            ),
            MessageNode(
                parent=child_branch + 1,
                message=ToolMessage(
                    tool_call_id="child-send",
                    content="message queued",
                ),
                sampled=False,
            ),
        ]
    )
    child_stop, stop_records, _ = await session.rewrite_request(child_request)
    assert child_stop.tools is None
    assert stop_records[-1].handler == "interaction_curriculum_child_stop"
    assert trace.info[CURRICULUM_INFO_KEY]["child_stop"] == {
        "mode": "one_turn_no_tools",
        "fired": True,
        "original_tool_count": 1,
    }

    stop_body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "continue"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ipython",
                    "description": "run code",
                    "parameters": {},
                },
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }
    ChatDialect().rewrite_request(stop_body, child_request, child_stop)
    assert "tools" not in stop_body
    assert "tool_choice" not in stop_body
    assert "parallel_tool_calls" not in stop_body


@pytest.mark.parametrize("phase", ["e0b_select_child_value", "e0d3_uncapped_yield"])
@pytest.mark.asyncio
async def test_pattern_child_phases_constrain_send_shape_but_not_value(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    monkeypatch.setenv(CURRICULUM_ENV_VAR, phase)
    install_natural_yield_scaffold()
    install_interaction_curriculum()
    task = _bootstrapped_task()
    trace = _initial_trace(task)
    session = RolloutSession(ctx=SimpleNamespace(), trace=trace)

    root, _, _ = await session.rewrite_request(_request(task.data.prompt))
    root_code = root.tools[0].parameters["properties"]["code"]["enum"][0]
    child_spec = task.data.oracle["children"][0]
    assert child_spec["operation"] in root_code
    assert (
        f"agent_message.send({str(child_spec['expected_result'])!r}," not in root_code
    )

    child_request = _request(f'review\n\n{PRIVATE_EVIDENCE_HEADER}\n{{"finding": 4}}')
    child, records, _ = await session.rewrite_request(child_request)
    assert records[-1].handler == "interaction_curriculum_exact_action"
    assert child.tools is not None and len(child.tools) == 1
    code_schema = child.tools[0].parameters["properties"]["code"]
    assert code_schema == {
        "type": "string",
        "pattern": CHILD_VALUE_SEND_PATTERN,
    }
    assert "enum" not in code_schema

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": child_request.messages[0].content}],
        "tools": [],
    }
    ChatDialect().rewrite_request(body, child_request, child)
    assert body["tool_choice"] == {
        "type": "function",
        "function": {"name": "ipython"},
    }
    assert body["tools"][0]["function"]["parameters"] == child.tools[0].parameters

    sampled_code = (
        "await agent_message.send("
        f"{str(child_spec['expected_result'])!r}, receiver_role='parent')"
    )
    child_branch = len(trace.nodes)
    trace.nodes.extend(
        [
            MessageNode(
                parent=None,
                message=UserMessage(content=child_request.messages[0].content),
                sampled=False,
            ),
            MessageNode(
                parent=child_branch,
                message=AssistantMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="child-value-send",
                            name="ipython",
                            arguments=json.dumps({"code": sampled_code}),
                        )
                    ],
                ),
                sampled=True,
            ),
            MessageNode(
                parent=child_branch + 1,
                message=ToolMessage(
                    tool_call_id="child-value-send",
                    content="message queued",
                ),
                sampled=False,
            ),
        ]
    )
    child_stop, stop_records, _ = await session.rewrite_request(child_request)
    assert child_stop.tools is None
    assert stop_records[-1].handler == "interaction_curriculum_child_stop"
    events = trace.info[CURRICULUM_INFO_KEY]["events"]
    assert [event["kind"] for event in events] == [
        "root_retained_spawn",
        "child_value_send",
    ]
    assert events[1]["mode"] == "pattern_constrained_ipython_action"
    assert (
        events[1]["sampled_code_sha256"]
        == hashlib.sha256(sampled_code.encode()).hexdigest()
    )


@pytest.mark.asyncio
async def test_generated_coevolution_context_reaches_child_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CURRICULUM_ENV_VAR, "e0c28_inline_only")
    install_natural_yield_scaffold()
    install_interaction_curriculum()
    task = _bootstrapped_task()
    generated = "Use the inline evidence card, compute with IPython, and report once."
    marker = json.dumps(
        {"environment_id": "env-test", "child_context": generated},
        sort_keys=True,
        separators=(",", ":"),
    )
    data = task.data.model_copy(
        update={
            "prompt": (
                f"{task.data.prompt}\n{COEVOLUTION_CONTEXT_HEADER}\n{marker}\n"
                "Generated root context."
            )
        }
    )
    generated_task = ProceduralHarnessMasterTask(data, task.config)
    trace = _initial_trace(generated_task)
    session = RolloutSession(ctx=SimpleNamespace(), trace=trace)

    root, _, _ = await session.rewrite_request(_request(generated_task.data.prompt))
    root_code = root.tools[0].parameters["properties"]["code"]["enum"][0]

    assert generated in root_code
    assert COEVOLUTION_CONTEXT_HEADER in root_code
    assert str(task.data.oracle["children"][0]["expected_result"]) not in root_code


@pytest.mark.parametrize(
    ("phase", "path_hint", "template_expected"),
    [
        ("e0c_natural_child", "verbose", True),
        ("e0c2_natural_child_no_template", "verbose", False),
        ("e0c25_inline_evidence", "concise", False),
        ("e0c275_inline_location", "positive", False),
        ("e0c28_inline_only", "inline_only", False),
        ("e0c29_evidence_available", "available", False),
        ("e0c3_natural_child_minimal", "none", False),
    ],
)
@pytest.mark.asyncio
async def test_natural_child_phases_force_ipython_but_leave_code_unconstrained(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    path_hint: str,
    template_expected: bool,
) -> None:
    monkeypatch.setenv(CURRICULUM_ENV_VAR, phase)
    install_natural_yield_scaffold()
    install_interaction_curriculum()
    task = _bootstrapped_task()
    trace = _initial_trace(task)
    session = RolloutSession(ctx=SimpleNamespace(), trace=trace)

    root, _, _ = await session.rewrite_request(_request(task.data.prompt))
    root_code = root.tools[0].parameters["properties"]["code"]["enum"][0]
    child_spec = task.data.oracle["children"][0]
    assert child_spec["operation"] in root_code
    assert ("evidence label is not a runtime path" in root_code) is (
        path_hint == "verbose"
    )
    assert ("inline card in your system message" in root_code) is (
        path_hint == "concise"
    )
    assert ("supplied inline in your system message" in root_code) is (
        path_hint == "positive"
    )
    assert ("evidence has already been supplied to you" in root_code) is (
        path_hint == "available"
    )
    assert ("evidence is supplied inline." in root_code) is (path_hint == "inline_only")
    assert ("not a filesystem path" in root_code) is (path_hint == "concise")
    assert ("str(VALUE)" in root_code) is template_expected
    assert str(child_spec["expected_result"]) not in root_code

    child_request = _request(f'review\n\n{PRIVATE_EVIDENCE_HEADER}\n{{"finding": 4}}')
    child, records, _ = await session.rewrite_request(child_request)
    assert records[-1].handler == "interaction_curriculum_exact_action"
    assert child.tools is not None and len(child.tools) == 1
    assert child.tools[0].name == "ipython"
    assert TOOL_DESCRIPTION_MARKER in child.tools[0].description
    assert child.tools[0].parameters == {}

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": child_request.messages[0].content}],
        "tools": [],
    }
    ChatDialect().rewrite_request(body, child_request, child)
    assert body["tool_choice"] == {
        "type": "function",
        "function": {"name": "ipython"},
    }
    assert body["tools"][0]["function"]["parameters"] == {}

    sampled_code = (
        "await agent_message.send(str("
        f"{child_spec['expected_result']}), receiver_role='parent')"
    )
    child_branch = len(trace.nodes)
    trace.nodes.extend(
        [
            MessageNode(
                parent=None,
                message=UserMessage(content=child_request.messages[0].content),
                sampled=False,
            ),
            MessageNode(
                parent=child_branch,
                message=AssistantMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="child-natural-send",
                            name="ipython",
                            arguments=json.dumps({"code": sampled_code}),
                        )
                    ],
                ),
                sampled=True,
            ),
            MessageNode(
                parent=child_branch + 1,
                message=ToolMessage(
                    tool_call_id="child-natural-send",
                    content="message queued",
                ),
                sampled=False,
            ),
        ]
    )
    child_stop, stop_records, _ = await session.rewrite_request(child_request)
    assert child_stop.tools is None
    assert stop_records[-1].handler == "interaction_curriculum_child_stop"
    events = trace.info[CURRICULUM_INFO_KEY]["events"]
    assert [event["kind"] for event in events] == [
        "root_retained_spawn",
        "child_natural_send",
    ]
    assert events[1]["mode"] == "unconstrained_ipython_action"
    assert events[1]["code_sha256"] == hashlib.sha256(sampled_code.encode()).hexdigest()


@pytest.mark.asyncio
async def test_e1_removes_child_constraint_but_retains_root_constraint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CURRICULUM_ENV_VAR, "e1_root_and_yield")
    install_natural_yield_scaffold()
    install_interaction_curriculum()
    task = _bootstrapped_task()
    trace = _initial_trace(task)
    session = RolloutSession(ctx=SimpleNamespace(), trace=trace)

    root, _, _ = await session.rewrite_request(_request(task.data.prompt))
    child, child_records, _ = await session.rewrite_request(
        _request(f'review\n\n{PRIVATE_EVIDENCE_HEADER}\n{{"finding": 4}}')
    )

    assert (
        root.tools is not None and TOOL_DESCRIPTION_MARKER in root.tools[0].description
    )
    root_code = root.tools[0].parameters["properties"]["code"]["enum"][0]
    child_spec = task.data.oracle["children"][0]
    assert child_spec["operation"] in root_code
    assert (
        f"agent_message.send({str(child_spec['expected_result'])!r}," not in root_code
    )
    assert child.tools is not None and child.tools[0].description == "run code"
    assert all(
        record.handler != "interaction_curriculum_exact_action"
        for record in child_records
    )
    assert [event["kind"] for event in trace.info[CURRICULUM_INFO_KEY]["events"]] == [
        "root_retained_spawn"
    ]


@pytest.mark.asyncio
async def test_e2_retains_only_passive_yield_constraint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CURRICULUM_ENV_VAR, "e2_yield_only")
    install_natural_yield_scaffold()
    install_interaction_curriculum()
    task = _bootstrapped_task()

    initial = RolloutSession(ctx=SimpleNamespace(), trace=_initial_trace(task))
    root, root_records, _ = await initial.rewrite_request(_request(task.data.prompt))
    assert root.tools is not None and root.tools[0].description == "run code"
    assert all(
        record.handler != "interaction_curriculum_exact_action"
        for record in root_records
    )

    spawned = RolloutSession(ctx=SimpleNamespace(), trace=_spawn_trace(task))
    yielded, yield_records, _ = await spawned.rewrite_request(_request())
    assert yielded.tools is None
    assert yield_records[-1].handler == "natural_yield_training_scaffold"


@pytest.mark.asyncio
async def test_interaction_constraints_require_train_gen_bootstrap_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CURRICULUM_ENV_VAR, "e0_full_actions")
    install_natural_yield_scaffold()
    install_interaction_curriculum()

    plain = _task(0, rung="natural_n1a")
    plain_session = RolloutSession(ctx=SimpleNamespace(), trace=_initial_trace(plain))
    plain_request, _, _ = await plain_session.rewrite_request(
        _request(plain.data.prompt)
    )
    assert plain_request.tools is not None
    assert plain_request.tools[0].description == "run code"

    bootstrapped = _bootstrapped_task()
    heldout_data = bootstrapped.data.model_copy(update={"split": "valid_gen"})
    heldout = ProceduralHarnessMasterTask(heldout_data, bootstrapped.config)
    heldout_session = RolloutSession(
        ctx=SimpleNamespace(), trace=_initial_trace(heldout)
    )
    heldout_request, _, _ = await heldout_session.rewrite_request(
        _request(heldout.data.prompt)
    )
    assert heldout_request.tools is not None
    assert heldout_request.tools[0].description == "run code"

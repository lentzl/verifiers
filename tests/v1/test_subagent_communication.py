import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from subagent_communication_v1.taskset import (
    COMPLETION_GATE_PATH,
    OWNERSHIP_GUIDANCE,
    WEIGHTED_CHECKSUM_FORMULA,
    SubagentCommunicationConfig,
    SubagentCommunicationTaskConfig,
    SubagentCommunicationTaskset,
    _answer_score,
    _completion_gate_source,
    _duplicate_cells,
    _ipython_events,
    _ownership_transition_behavior,
    _protocol_behavior,
    keep_bidirectional_state_transitions,
    keep_child_request_phase_responses,
    keep_complete_fan_in_response,
    keep_coordinator_pre_child_responses,
    keep_first_coordinator_response,
    keep_first_coordinator_tool_call,
    keep_post_child_message_responses,
)

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)


def _trace(*cells: str | tuple[str, str], reply: str = "{}") -> vf.Trace:
    nodes = [
        MessageNode(
            parent=None,
            message=UserMessage(content="coordinate"),
            sampled=False,
        )
    ]
    parent = 0
    for index, cell in enumerate(cells):
        source, output = cell if isinstance(cell, tuple) else (cell, "")
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
                            arguments=json.dumps({"code": source}),
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
    nodes.append(
        MessageNode(
            parent=parent,
            message=AssistantMessage(content=reply),
            sampled=True,
        )
    )
    return vf.Trace(
        id="subagent-communication-test",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="SubagentCommunicationTask", data=vf.TaskData(idx=0)),
        nodes=nodes,
    )


def _child_message(name: str, message: str, message_id: str) -> UserMessage:
    return UserMessage(
        content=(
            f"[from child:{name}]\n"
            "Agent-to-agent message received.\n"
            "Source: agent_message\n"
            f"Message id: {message_id}\n\n"
            f"{message}"
        )
    )


def _with_child_messages(trace: vf.Trace, *messages: UserMessage) -> vf.Trace:
    parent = len(trace.nodes) - 2
    final = trace.nodes[-1]
    trace.nodes = trace.nodes[:-1]
    for message in messages:
        trace.nodes.append(MessageNode(parent=parent, message=message, sampled=False))
        parent = len(trace.nodes) - 1
    final.parent = parent
    trace.nodes.append(final)
    return trace


def _append_ipython_before_final(trace: vf.Trace, code: str, output: str = "") -> vf.Trace:
    final = trace.nodes.pop()
    parent = final.parent
    call_id = f"post-fan-in-{len(trace.nodes)}"
    trace.nodes.append(
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
    trace.nodes.append(
        MessageNode(
            parent=len(trace.nodes) - 1,
            message=ToolMessage(tool_call_id=call_id, content=output),
            sampled=False,
        )
    )
    final.parent = len(trace.nodes) - 1
    trace.nodes.append(final)
    return trace


def _append_tool_before_final(trace: vf.Trace, name: str, arguments: dict[str, str]) -> vf.Trace:
    final = trace.nodes.pop()
    parent = final.parent
    call_id = f"direct-tool-{len(trace.nodes)}"
    trace.nodes.append(
        MessageNode(
            parent=parent,
            message=AssistantMessage(
                content="",
                tool_calls=[ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))],
            ),
            sampled=True,
        )
    )
    trace.nodes.append(
        MessageNode(
            parent=len(trace.nodes) - 1,
            message=ToolMessage(tool_call_id=call_id, content=f"Tool {name} not found"),
            sampled=False,
        )
    )
    final.parent = len(trace.nodes) - 1
    trace.nodes.append(final)
    return trace


@pytest.mark.parametrize("source", ["# waiting", "pass", "'waiting'", 'print("waiting")'])
def test_clean_protocol_rejects_inert_ipython_cells(source: str) -> None:
    behavior = _protocol_behavior(_trace(source), "direct", (), {}, None)

    assert behavior["protocol_aligned"] == 1.0
    assert behavior["clean_protocol_aligned"] == 0.0
    assert behavior["inert_cells"] == 1.0


def test_clean_protocol_rejects_non_ipython_model_tools() -> None:
    trace = _append_tool_before_final(_trace(), "agent_observe", {"child_name": "shard-worker"})

    behavior = _protocol_behavior(trace, "direct", (), {}, None)

    assert behavior["protocol_aligned"] == 1.0
    assert behavior["clean_protocol_aligned"] == 0.0
    assert behavior["non_ipython_tool_calls"] == 1.0
    assert behavior["observation_calls"] == 1.0


def test_ipython_outputs_follow_graph_edges_when_branches_reuse_call_ids() -> None:
    trace = _trace()
    final = trace.nodes.pop()
    trace.nodes.extend(
        [
            MessageNode(
                parent=0,
                message=AssistantMessage(
                    content="",
                    tool_calls=[ToolCall(id="call_0", name="ipython", arguments=json.dumps({"code": "root()"}))],
                ),
                sampled=True,
            ),
            MessageNode(
                parent=0,
                message=AssistantMessage(
                    content="",
                    tool_calls=[ToolCall(id="call_0", name="ipython", arguments=json.dumps({"code": "child()"}))],
                ),
                sampled=True,
            ),
            MessageNode(parent=1, message=ToolMessage(tool_call_id="call_0", content="root output"), sampled=False),
            MessageNode(parent=2, message=ToolMessage(tool_call_id="call_0", content="child output"), sampled=False),
        ]
    )
    final.parent = 3
    trace.nodes.append(final)

    events = _ipython_events(trace)

    assert [(event.code, event.output) for event in events] == [
        ("root()", "root output"),
        ("child()", "child output"),
    ]


def test_duplicate_cells_are_scoped_to_each_agent_branch() -> None:
    trace = _trace("await agent_message.send('result', receiver_role='parent')")
    final = trace.nodes.pop()
    child_root = len(trace.nodes)
    trace.nodes.extend(
        [
            MessageNode(
                parent=None,
                message=UserMessage(content="[task from parent]"),
                sampled=False,
            ),
            MessageNode(
                parent=child_root,
                message=AssistantMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="child-call",
                            name="ipython",
                            arguments=json.dumps(
                                {"code": "await agent_message.send('result', receiver_role='parent')"}
                            ),
                        )
                    ],
                ),
                sampled=True,
            ),
            MessageNode(
                parent=child_root + 1,
                message=ToolMessage(tool_call_id="child-call", content="sent"),
                sampled=False,
            ),
        ]
    )
    trace.nodes.append(final)

    assert _duplicate_cells(trace, _ipython_events(trace)) == 0
    same_branch = _trace(
        "await agent_message.send('result', receiver_role='parent')",
        "await agent_message.send('result', receiver_role='parent')",
    )
    assert _duplicate_cells(same_branch, _ipython_events(same_branch)) == 1


def test_taskset_balances_families_and_holds_out_generator_variants() -> None:
    train = SubagentCommunicationTaskset(SubagentCommunicationConfig(split="train", instances_per_template=1)).load()
    evaluation = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(split="eval", instances_per_template=1)
    ).load()

    assert len(train) == 16
    assert len(evaluation) == 8
    assert {task.data.family for task in train} == {
        "direct",
        "single",
        "parallel",
        "followup",
    }
    assert {task.data.template_variant for task in train} == {0, 1, 2, 3}
    assert {task.data.template_variant for task in evaluation} == {4, 5}
    assert not ({task.data.name for task in train} & {task.data.name for task in evaluation})


def test_single_tasks_expose_task_specific_opsd_demonstrations() -> None:
    task = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="train",
            families=("single",),
            instances_per_template=1,
        )
    ).load()[0]

    demonstration = task.data.demonstration
    assert demonstration is not None
    path = task.data.child_paths["shard-worker"]
    assert f"Read {path}. Its top-level JSON value is the integer list itself" in demonstration
    assert WEIGHTED_CHECKSUM_FORMULA in demonstration
    assert "execute exactly once" in demonstration
    assert "handle = await rlm(" in demonstration
    assert json.dumps(task.data.answer) in demonstration
    demonstrations = task.data.demonstrations
    assert demonstrations is not None
    assert demonstrations[task.data.prompt] == demonstration
    child_question = next(question for question in demonstrations if question.startswith("[task from parent]"))
    assert task.data.child_paths["shard-worker"] in child_question
    assert "await agent_message.send(str(checksum), receiver_role='parent')" in demonstrations[child_question]
    coordinator_demonstrations = task.data.coordinator_demonstrations
    assert coordinator_demonstrations is not None
    assert coordinator_demonstrations[task.data.prompt] == demonstration
    assert coordinator_demonstrations[child_question] is None
    assert coordinator_demonstrations["*"] is None


def test_parallel_tasks_expose_message_provenance_opsd_demonstrations() -> None:
    task = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="train",
            families=("parallel",),
            instances_per_template=1,
        )
    ).load()[0]

    demonstration = task.data.demonstration
    assert demonstration is not None
    assert task.data.child_paths["alpha-worker"] in demonstration
    assert task.data.child_paths["beta-worker"] in demonstration
    assert "alpha_handle = await rlm(" in demonstration
    assert "beta_handle = await rlm(" in demonstration
    assert "[from child:<name>]" in demonstration
    assert "without polling or sending READY messages" in demonstration
    assert "Do not bind an agent_message.send receipt" in demonstration
    assert json.dumps(task.data.answer) in demonstration
    demonstrations = task.data.demonstrations
    assert demonstrations is not None
    assert demonstrations[task.data.prompt] == demonstration
    child_demonstrations = {
        question: child_demo
        for question, child_demo in demonstrations.items()
        if question.startswith("[task from parent]")
    }
    assert len(child_demonstrations) == 2
    for child_name, answer_key in (("alpha-worker", "alpha"), ("beta-worker", "beta")):
        path = task.data.child_paths[child_name]
        child_question = next(question for question in child_demonstrations if path in question)
        child_demo = child_demonstrations[child_question]
        assert f"The observed checksum is {task.data.answer[answer_key]}" in child_demo
        assert "await agent_message.send(str(checksum), receiver_role='parent')" in child_demo
    coordinator_demonstrations = task.data.coordinator_demonstrations
    assert coordinator_demonstrations is not None
    assert coordinator_demonstrations[task.data.prompt] == demonstration
    assert all(coordinator_demonstrations[question] is None for question in child_demonstrations)
    assert coordinator_demonstrations["*"] is None


@pytest.mark.parametrize(
    ("family", "child_name", "request_phrase", "secret_key"),
    [
        ("followup", "key-worker", "need multiplier", "multiplier"),
        ("handshake", "relay-worker", "need nonce", "nonce"),
    ],
)
def test_bidirectional_tasks_expose_role_specific_opsd_demonstrations(
    family: str,
    child_name: str,
    request_phrase: str,
    secret_key: str,
) -> None:
    task = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="train",
            families=(family,),
            instances_per_template=1,
        )
    ).load()[0]

    coordinator_demo = task.data.demonstration
    assert coordinator_demo is not None
    assert f"{secret_key} = {task.data.answer[secret_key]}" in coordinator_demo
    assert f"name='{child_name}'" in coordinator_demo
    assert "receiver_name=child.name" in coordinator_demo
    assert json.dumps(task.data.answer) in coordinator_demo

    demonstrations = task.data.demonstrations
    assert demonstrations is not None
    assert demonstrations[task.data.prompt] == coordinator_demo
    child_question = next(question for question in demonstrations if question.startswith("[task from parent]"))
    child_demo = demonstrations[child_question]
    assert request_phrase in child_question
    assert f"agent_message.send('{request_phrase}', receiver_role='parent')" in child_demo
    assert demonstrations["*"] == child_demo

    coordinator_only = task.data.coordinator_demonstrations
    assert coordinator_only is not None
    assert coordinator_only[task.data.prompt] == coordinator_demo
    assert coordinator_only[child_question] is None
    assert coordinator_only["*"] is None

    turn_demonstrations = task.data.turn_demonstrations
    assert turn_demonstrations is not None
    coordinator_steps = turn_demonstrations[task.data.prompt]
    child_steps = turn_demonstrations["*"]
    assert isinstance(coordinator_steps, list)
    assert isinstance(child_steps, list)
    assert len(coordinator_steps) == 3
    assert len(child_steps) == 2
    assert request_phrase in child_steps[0]
    assert f"{secret_key} = {task.data.answer[secret_key]}" in child_steps[1]

    request_demonstrations = task.data.child_request_demonstrations
    assert request_demonstrations is not None
    assert request_demonstrations[task.data.prompt] is None
    request_steps = request_demonstrations[child_question]
    assert isinstance(request_steps, list)
    assert request_steps == request_demonstrations["*"]
    assert len(request_steps) == 2
    assert request_phrase in request_steps[0]
    assert request_steps[1] is None


def test_bidirectional_transition_filter_skips_tool_followups_and_gate_feedback() -> None:
    nodes = [
        SimpleNamespace(
            message=UserMessage(content="coordinate"),
            sampled=False,
            token_ids=[1],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="spawn"),
            sampled=True,
            token_ids=[2],
            mask=[True],
        ),
        SimpleNamespace(
            message=ToolMessage(tool_call_id="spawn", content=""),
            sampled=False,
            token_ids=[3],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="waiting"),
            sampled=True,
            token_ids=[4],
            mask=[True],
        ),
        SimpleNamespace(
            message=UserMessage(content="quality gate feedback"),
            sampled=False,
            token_ids=[5],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="still waiting"),
            sampled=True,
            token_ids=[6],
            mask=[True],
        ),
        SimpleNamespace(
            message=UserMessage(content="[from child:key-worker]\n\nneed multiplier"),
            sampled=False,
            token_ids=[7],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="send multiplier"),
            sampled=True,
            token_ids=[8, 9],
            mask=[True, True],
        ),
    ]
    trace = SimpleNamespace(branches=[SimpleNamespace(nodes=nodes)])

    assert keep_bidirectional_state_transitions(trace) == [[False, True, False, False, False, False, False, True, True]]


def test_coordinator_pre_child_filter_excludes_child_branch_and_post_reply_response() -> None:
    shared = SimpleNamespace(
        message=UserMessage(content="coordinate"),
        sampled=False,
        token_ids=[1],
        mask=[False],
    )
    coordinator_nodes = [
        shared,
        SimpleNamespace(
            message=AssistantMessage(content="spawn"),
            sampled=True,
            token_ids=[2, 3],
            mask=[True, True],
        ),
        SimpleNamespace(
            message=ToolMessage(tool_call_id="spawn", content=""),
            sampled=False,
            token_ids=[4],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="waiting"),
            sampled=True,
            token_ids=[5],
            mask=[True],
        ),
        SimpleNamespace(
            message=UserMessage(content="[from child:worker]\n\n17"),
            sampled=False,
            token_ids=[6],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="final"),
            sampled=True,
            token_ids=[7],
            mask=[True],
        ),
    ]
    child_nodes = [
        shared,
        SimpleNamespace(
            message=UserMessage(content="[task from parent]\n\ncompute"),
            sampled=False,
            token_ids=[8],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="send"),
            sampled=True,
            token_ids=[9],
            mask=[True],
        ),
    ]
    trace = SimpleNamespace(branches=[SimpleNamespace(nodes=coordinator_nodes), SimpleNamespace(nodes=child_nodes)])

    assert keep_coordinator_pre_child_responses(trace) == [
        [False, True, True, False, True, False, False],
        [False, False, False],
    ]


def test_first_coordinator_response_filter_excludes_later_parent_and_child_actions() -> None:
    shared = SimpleNamespace(
        message=UserMessage(content="coordinate"),
        sampled=False,
        token_ids=[1],
        mask=[False],
    )
    first = SimpleNamespace(
        message=AssistantMessage(content="spawn and compute local"),
        sampled=True,
        token_ids=[2, 3],
        mask=[True, True],
    )
    coordinator_nodes = [
        shared,
        first,
        SimpleNamespace(
            message=ToolMessage(tool_call_id="spawn", content=""),
            sampled=False,
            token_ids=[4],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="waiting"),
            sampled=True,
            token_ids=[5],
            mask=[True],
        ),
    ]
    child_nodes = [
        SimpleNamespace(
            message=UserMessage(content="[task from parent]\n\ncompute"),
            sampled=False,
            token_ids=[6],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="send"),
            sampled=True,
            token_ids=[7],
            mask=[True],
        ),
    ]
    trace = SimpleNamespace(branches=[SimpleNamespace(nodes=coordinator_nodes), SimpleNamespace(nodes=child_nodes)])

    assert keep_first_coordinator_response(trace) == [
        [False, True, True, False, False],
        [False, False],
    ]


def test_first_coordinator_tool_call_filter_keeps_marked_action_only() -> None:
    shared = SimpleNamespace(
        message=UserMessage(content="coordinate"),
        sampled=False,
        token_ids=[1],
        mask=[False],
    )
    first = SimpleNamespace(
        message=AssistantMessage(content="reason then act"),
        sampled=True,
        token_ids=[2, 248058, 3, 4, 248059, 5],
        mask=[True, True, True, True, True, True],
    )
    coordinator_nodes = [
        shared,
        first,
        SimpleNamespace(
            message=ToolMessage(tool_call_id="spawn", content=""),
            sampled=False,
            token_ids=[6],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="later"),
            sampled=True,
            token_ids=[248058, 7, 248059],
            mask=[True, True, True],
        ),
    ]
    child_nodes = [
        SimpleNamespace(
            message=UserMessage(content="[task from parent]\n\ncompute"),
            sampled=False,
            token_ids=[8],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="child call"),
            sampled=True,
            token_ids=[248058, 9, 248059],
            mask=[True, True, True],
        ),
    ]
    trace = SimpleNamespace(branches=[SimpleNamespace(nodes=coordinator_nodes), SimpleNamespace(nodes=child_nodes)])

    assert keep_first_coordinator_tool_call(trace) == [
        [False, False, True, True, True, True, False, False, False, False, False],
        [False, False, False, False],
    ]


def test_first_coordinator_tool_call_filter_drops_unmarked_response() -> None:
    nodes = [
        SimpleNamespace(
            message=UserMessage(content="coordinate"),
            sampled=False,
            token_ids=[1],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="no tool call"),
            sampled=True,
            token_ids=[2, 3],
            mask=[True, True],
        ),
    ]
    trace = SimpleNamespace(branches=[SimpleNamespace(nodes=nodes)])

    assert keep_first_coordinator_tool_call(trace) == [[False, False, False]]


def test_child_request_phase_filter_includes_post_tool_response_until_send_succeeds() -> None:
    coordinator_nodes = [
        SimpleNamespace(
            message=UserMessage(content="coordinate"),
            sampled=False,
            token_ids=[1],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="spawn"),
            sampled=True,
            token_ids=[2],
            mask=[True],
        ),
    ]
    child_nodes = [
        SimpleNamespace(
            message=UserMessage(content="[task from parent]\n\ncompute, then request multiplier"),
            sampled=False,
            token_ids=[3],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="compute subtotal"),
            sampled=True,
            token_ids=[4, 5],
            mask=[True, True],
        ),
        SimpleNamespace(
            message=ToolMessage(tool_call_id="compute", content="77"),
            sampled=False,
            token_ids=[6],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="request multiplier"),
            sampled=True,
            token_ids=[7, 8],
            mask=[True, True],
        ),
        SimpleNamespace(
            message=ToolMessage(tool_call_id="request", content="Agent message sent: agentmsg_request"),
            sampled=False,
            token_ids=[9],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="waiting"),
            sampled=True,
            token_ids=[10],
            mask=[True],
        ),
        SimpleNamespace(
            message=UserMessage(content="[from parent]\n\n40"),
            sampled=False,
            token_ids=[11],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="send result"),
            sampled=True,
            token_ids=[12],
            mask=[True],
        ),
    ]
    trace = SimpleNamespace(branches=[SimpleNamespace(nodes=coordinator_nodes), SimpleNamespace(nodes=child_nodes)])

    assert keep_child_request_phase_responses(trace) == [
        [False, False],
        [False, True, True, False, True, True, False, False, False, False],
    ]


def test_child_request_phase_filter_keeps_traceback_repair_before_success() -> None:
    nodes = [
        SimpleNamespace(
            message=UserMessage(content="[task from parent]\n\nrequest nonce"),
            sampled=False,
            token_ids=[1],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="bad send"),
            sampled=True,
            token_ids=[2],
            mask=[True],
        ),
        SimpleNamespace(
            message=ToolMessage(tool_call_id="bad", content="RuntimeError: wrong receiver"),
            sampled=False,
            token_ids=[3],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="repaired send"),
            sampled=True,
            token_ids=[4, 5],
            mask=[True, True],
        ),
        SimpleNamespace(
            message=ToolMessage(tool_call_id="good", content="Agent message queued: agentmsg_good"),
            sampled=False,
            token_ids=[6],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="done"),
            sampled=True,
            token_ids=[7],
            mask=[True],
        ),
    ]
    trace = SimpleNamespace(branches=[SimpleNamespace(nodes=nodes)])

    assert keep_child_request_phase_responses(trace) == [[False, True, False, True, True, False, False]]


def test_post_child_message_filter_selects_only_the_immediate_response() -> None:
    nodes = [
        SimpleNamespace(
            message=UserMessage(content="coordinate"),
            sampled=False,
            token_ids=[1],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="spawn"),
            sampled=True,
            token_ids=[2],
            mask=[True],
        ),
        SimpleNamespace(
            message=UserMessage(content="[from child:alpha-worker]\n\n17"),
            sampled=False,
            token_ids=[3, 4],
            mask=[False, False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="bind alpha"),
            sampled=True,
            token_ids=[5, 6],
            mask=[True, True],
        ),
        SimpleNamespace(
            message=ToolMessage(tool_call_id="call-0", content="17"),
            sampled=False,
            token_ids=[7],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="wait"),
            sampled=True,
            token_ids=[8],
            mask=[True],
        ),
    ]
    trace = SimpleNamespace(
        branches=[SimpleNamespace(nodes=nodes, token_ids=[token for node in nodes for token in node.token_ids])]
    )

    assert keep_post_child_message_responses(trace) == [[False, False, False, False, True, True, False, False]]


def test_complete_fan_in_filter_waits_for_every_expected_child() -> None:
    nodes = [
        SimpleNamespace(
            message=UserMessage(content="coordinate"),
            sampled=False,
            token_ids=[1],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="spawn both"),
            sampled=True,
            token_ids=[2],
            mask=[True],
        ),
        SimpleNamespace(
            message=UserMessage(content="[from child:alpha-worker]\n\n17"),
            sampled=False,
            token_ids=[3],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="retain alpha and wait"),
            sampled=True,
            token_ids=[4, 5],
            mask=[True, True],
        ),
        SimpleNamespace(
            message=UserMessage(content="[from child:beta-worker]\n\n23"),
            sampled=False,
            token_ids=[6],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="combine both replies"),
            sampled=True,
            token_ids=[7, 8],
            mask=[True, True],
        ),
        SimpleNamespace(
            message=UserMessage(content="[from child:beta-worker]\n\n23"),
            sampled=False,
            token_ids=[9],
            mask=[False],
        ),
        SimpleNamespace(
            message=AssistantMessage(content="duplicate reply"),
            sampled=True,
            token_ids=[10],
            mask=[True],
        ),
    ]
    trace = SimpleNamespace(
        task=SimpleNamespace(data=SimpleNamespace(expected_children=("alpha-worker", "beta-worker"))),
        branches=[SimpleNamespace(nodes=nodes)],
    )

    assert keep_complete_fan_in_response(trace) == [
        [False, False, False, False, False, False, True, True, False, False]
    ]


def test_teacher_conditioned_preflight_uses_prime_opsd_template() -> None:
    task = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="eval",
            families=("single",),
            instruction_level="guided",
            instances_per_template=1,
            teacher_conditioned=True,
        )
    ).load()[0]

    assert task.data.prompt.startswith("<Question>\n")
    assert "\n<Demonstration>\n" in task.data.prompt
    assert task.data.demonstration is not None
    assert task.data.demonstration in task.data.prompt
    assert task.data.teacher_conditioned is True
    assert "next assistant response itself must be exactly `Waiting for shard-worker's explicit reply.`" in (
        task.data.demonstration
    )
    assert task.data.prompt.endswith("Now answer with a response of your own, including the thinking process:")


def test_teacher_conditioned_preflight_accepts_parallel_demonstrations() -> None:
    task = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="eval",
            families=("parallel",),
            instances_per_template=1,
            teacher_conditioned=True,
        )
    ).load()[0]

    assert task.data.demonstration is not None
    assert task.data.demonstration in task.data.prompt
    assert task.data.child_paths["alpha-worker"] in task.data.prompt
    assert task.data.child_paths["beta-worker"] in task.data.prompt


@pytest.mark.parametrize("family", ["followup", "handshake"])
def test_teacher_conditioned_preflight_accepts_bidirectional_demonstrations(family: str) -> None:
    task = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="eval",
            families=(family,),
            instances_per_template=1,
            teacher_conditioned=True,
        )
    ).load()[0]

    assert task.data.demonstration is not None
    assert task.data.demonstration in task.data.prompt
    assert str(task.data.followup_secret) in task.data.demonstration


def test_teacher_conditioned_preflight_rejects_unsupported_families() -> None:
    taskset = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="eval",
            families=("direct",),
            instances_per_template=1,
            teacher_conditioned=True,
        )
    )

    with pytest.raises(ValueError, match="requires a supported demonstration family"):
        taskset.load()


def test_handshake_family_is_available_without_changing_the_default_mix() -> None:
    tasks = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="train",
            families=("handshake",),
            instruction_level="guided",
            instances_per_template=1,
        )
    ).load()

    assert len(tasks) == 4
    assert {task.data.family for task in tasks} == {"handshake"}
    assert all("agent_message is a Python module, not a direct model tool" in task.data.prompt for task in tasks)
    assert all("In one IPython call execute exactly" in task.data.prompt for task in tasks)
    assert all("need nonce" in task.data.prompt for task in tasks)
    spawn_prompts = [task.data.prompt.split('rlm("', 1)[1].split('", name=', 1)[0] for task in tasks]
    assert all(str(task.data.followup_secret) not in prompt for task, prompt in zip(tasks, spawn_prompts))


def test_explicit_bidirectional_prompt_contract_disambiguates_followup_arithmetic() -> None:
    task = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            families=("followup",),
            instances_per_template=1,
            prompt_contract="explicit_bidirectional_v2",
        )
    ).load()[0]

    assert "A shard checksum is" not in task.data.prompt
    assert "multiply its retained subtotal by that multiplier" in task.data.prompt
    assert "send both the subtotal and product" in task.data.prompt


def test_instance_offset_keeps_supervised_seed_paths_disjoint() -> None:
    task = next(
        task
        for task in SubagentCommunicationTaskset(
            SubagentCommunicationConfig(
                split="train",
                families=("single",),
                instances_per_template=1,
                instance_offset=100,
            )
        ).load()
        if task.data.template_variant == 0
    )

    assert task.data.name == "single-v0-i100"
    assert task.data.child_paths["shard-worker"].endswith("v0-i100-remote.json")


def test_guided_tasks_explain_native_contract_without_revealing_answers() -> None:
    tasks = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(split="train", instruction_level="guided", instances_per_template=1)
    ).load()

    single = next(task for task in tasks if task.data.family == "single")
    followup = next(task for task in tasks if task.data.family == "followup")

    assert "handle = await rlm" in single.data.prompt
    assert single.data.child_paths["shard-worker"] in single.data.prompt
    assert "Do not open" in single.data.prompt
    assert "name='shard-worker'" in single.data.prompt
    assert "Spawn the child before" in single.data.prompt
    assert "stop calling tools" in single.data.prompt
    assert "explicit child message" in single.data.prompt
    assert single.data.prompt.count(WEIGHTED_CHECKSUM_FORMULA) == 1
    assert "top-level JSON value is the integer list itself, not an object" in single.data.prompt
    assert f"values = json.loads(Path({single.data.child_paths['shard-worker']!r}).read_text())" in single.data.prompt
    assert "agent_observe.get_agent" not in single.data.prompt
    assert "receiver_role='child'" in followup.data.prompt
    assert "name='key-worker'" in followup.data.prompt
    assert "You are key-worker, my child" in followup.data.prompt
    assert "top-level JSON value is the integer list itself, not an object" in followup.data.prompt
    assert f"values = json.loads(Path({followup.data.child_paths['key-worker']!r}).read_text())" in (
        followup.data.prompt
    )
    assert "subtotal = sum(values)" in followup.data.prompt
    assert "Do not call rlm or message a child" in followup.data.prompt
    assert "resume only when the visible [from parent] follow-up arrives" in followup.data.prompt
    assert f"multiplier = {followup.data.followup_secret}" in followup.data.prompt
    assert "Bind its integer body with int(...)" in followup.data.prompt
    assert "rather than guessing or hardcoding it" in followup.data.prompt
    assert "Do not print, inspect, or split that computation and send across cells" in followup.data.prompt
    assert "There is no `agent_message.list_messages` API" in followup.data.prompt
    assert "[from child:<name>]" in followup.data.prompt
    assert "Use three causally separate phases" in followup.data.prompt
    assert "do not message the child until a later resumed coordinator turn" in followup.data.prompt
    assert "[from child:key-worker]` with `need multiplier`" in followup.data.prompt
    assert "exactly `Waiting for key-worker's request.`" in followup.data.prompt
    assert "exactly `Waiting for key-worker's final result.`" in followup.data.prompt

    handshake = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="train",
            families=("handshake",),
            instruction_level="guided",
            instances_per_template=1,
        )
    ).load()[0]
    assert "Use three causally separate phases" in handshake.data.prompt
    assert "[from child:relay-worker]` with `need nonce`" in handshake.data.prompt
    assert "do not message the child until a later resumed coordinator turn" in handshake.data.prompt
    assert "exactly `Waiting for relay-worker's request.`" in handshake.data.prompt
    assert "exactly `Waiting for relay-worker's final echo.`" in handshake.data.prompt

    parallel = next(task for task in tasks if task.data.family == "parallel")
    assert "There is no `agent_message.list_messages` API" in parallel.data.prompt
    assert parallel.data.prompt.count(WEIGHTED_CHECKSUM_FORMULA) == 2
    assert parallel.data.prompt.count("top-level JSON value is the integer list itself, not an object") == 2
    assert json.dumps(single.data.answer) not in single.data.prompt
    assert json.dumps(parallel.data.answer) not in parallel.data.prompt


def test_completion_gate_requires_child_evidence_without_embedding_answer_values(
    tmp_path: Path,
) -> None:
    agent_dir = tmp_path / "agent"
    sessions = agent_dir / "sessions"
    sessions.mkdir(parents=True)
    session = sessions / "root.jsonl"
    source = _completion_gate_source(("subtotal", "multiplier", "result"), "followup")
    assert "never infer it from the task, demonstration, child status, or expected protocol" in source
    assert "exactly `Waiting for key-worker's request.`" in source
    gate = tmp_path / "completion_gate.py"
    gate.write_text(source)
    env = {**os.environ, "PRIME_AGENT_CODING_AGENT_DIR": str(agent_dir)}

    def run_gate(
        reply: str,
        child_messages: tuple[str, ...] = (),
        persisted_custom_messages: bool = True,
        child_messages_after_reply: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        messages = []
        for message in child_messages:
            if persisted_custom_messages:
                entry = {
                    "type": "custom_message",
                    "customType": "agent_message",
                    "content": (
                        f"[from child:key-worker]\nAgent-to-agent message received.\nSource: agent_message\n\n{message}"
                    ),
                    "display": True,
                    "details": {
                        "id": f"agentmsg_{len(message)}_{message}",
                        "message": message,
                        "from": {"sessionName": "key-worker"},
                        "fromRelationship": "child",
                    },
                }
            else:
                entry = {
                    "type": "message",
                    "message": {
                        "role": "user",
                        "content": f"[from child:key-worker]\n\n{message}",
                    },
                }
            messages.append(json.dumps(entry))
        assistant = json.dumps(
            {
                "type": "message",
                "message": {"role": "assistant", "content": reply},
            }
        )
        ordered = (assistant, *messages) if child_messages_after_reply else (*messages, assistant)
        session.write_text("\n".join((json.dumps({"type": "session", "rlmDepth": 0}), *ordered)) + "\n")
        return subprocess.run(
            [sys.executable, str(gate)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    assert run_gate("Waiting for the child.").returncode == 1
    assert run_gate('{"subtotal": 12, "multiplier": 7}').returncode == 1
    complete = '{"subtotal": 12, "multiplier": 7, "result": 84}'
    assert run_gate(complete).returncode == 1
    assert run_gate(complete, ("need multiplier",)).returncode == 1
    assert run_gate(complete, ("need multiplier", '{"subtotal": 12, "result": 84}')).returncode == 0
    fenced = run_gate(
        f"```json\n{complete}\n```",
        ("need multiplier", '{"subtotal": 12, "result": 84}'),
    )
    assert fenced.returncode == 1
    assert "all required child evidence is already present" in fenced.stderr
    assert "Do not call a tool" in fenced.stderr
    assert "no prose or Markdown fence" in fenced.stderr
    assert (
        run_gate(
            complete,
            ("need multiplier", '{"subtotal": 12, "result": 84}'),
            child_messages_after_reply=True,
        ).returncode
        == 1
    )
    assert (
        run_gate(
            complete,
            ("need multiplier", '{"subtotal": 12, "result": 84}'),
            persisted_custom_messages=False,
        ).returncode
        == 0
    )
    # Prompt prose and duplicate persisted IDs are not independent child replies.
    assert run_gate(complete, ("need multiplier", "need multiplier")).returncode == 1
    failed = run_gate("Waiting for the child.")
    assert "do not inspect the delegated shard" in failed.stderr
    assert "Otherwise call no tool and respond exactly `Waiting for key-worker's request.`" in failed.stderr
    assert "existing child" in failed.stderr
    assert "12" not in source
    assert "84" not in source


@pytest.mark.asyncio
async def test_delegated_shards_are_written_to_shared_absolute_paths() -> None:
    task = next(
        task
        for task in SubagentCommunicationTaskset(
            SubagentCommunicationConfig(split="train", instances_per_template=1)
        ).load()
        if task.data.family == "parallel"
    )

    class Runtime:
        def __init__(self) -> None:
            self.runs: list[list[str]] = []
            self.writes: dict[str, bytes] = {}

        async def run(self, argv: list[str], env: dict[str, str]):
            del env
            self.runs.append(argv)
            return SimpleNamespace(exit_code=0, stdout="", stderr="")

        async def write(self, path: str, contents: bytes) -> None:
            self.writes[path] = contents

    runtime = Runtime()
    await task.setup(SimpleNamespace(), runtime)

    assert runtime.runs == [["mkdir", "-p", "/workspace/.subagent-communication", "/workspace/subagent-shards"]]
    assert {path: runtime.writes[path] for path in task.data.files} == {
        path: contents.encode() for path, contents in task.data.files.items()
    }
    assert COMPLETION_GATE_PATH in runtime.writes
    assert set(task.data.child_paths.values()) == set(runtime.writes) - {COMPLETION_GATE_PATH}


def test_direct_family_rewards_restraint() -> None:
    behavior = _protocol_behavior(
        _trace("values = [2, 3]\nsum((i + 1) * value for i, value in enumerate(values))"),
        "direct",
        (),
        {},
        None,
    )

    assert behavior["protocol_aligned"] == 1.0
    assert behavior["spawn_calls"] == 0.0


def test_single_family_requires_native_spawn_and_child_reply() -> None:
    trace = _with_child_messages(
        _trace(
            "handle = await rlm('Compute /workspace/shard.json and reply with agent_message to parent.', name='shard-worker')"
        ),
        _child_message("shard-worker", "remote=91", "agentmsg_1"),
    )

    behavior = _protocol_behavior(
        trace,
        "single",
        ("shard-worker",),
        {"shard-worker": "/workspace/shard.json"},
        None,
    )

    assert behavior["protocol_aligned"] == 1.0
    assert behavior["clean_protocol_aligned"] == 0.0
    assert behavior["retained_handles"] == 1.0
    assert behavior["messages_to_parent"] == 1.0
    assert behavior["explicit_messages_to_parent"] == 0.0
    assert behavior["fan_in_complete"] == 1.0
    assert behavior["post_fan_in_control_aligned"] == 1.0


def test_delegated_family_rejects_coordinator_access_to_the_child_path() -> None:
    trace = _with_child_messages(
        _trace(
            "handle = await rlm('Compute /workspace/shard.json and reply with agent_message to parent.', name='shard-worker')",
            "from pathlib import Path\nleaked = Path('/workspace/shard.json').read_text()",
        ),
        _child_message("shard-worker", "remote=91", "agentmsg_1"),
    )

    behavior = _protocol_behavior(
        trace,
        "single",
        ("shard-worker",),
        {"shard-worker": "/workspace/shard.json"},
        None,
    )

    assert behavior["coordinator_delegated_path_accesses"] == 1.0
    assert behavior["protocol_aligned"] == 0.0


def test_followup_leakage_zeroes_stateful_and_process_control_credit() -> None:
    path = "/workspace/followup.json"
    trace = _with_child_messages(
        _trace(
            f"from pathlib import Path\nleaked = Path({path!r}).read_text()\n"
            f"child = await rlm('Sum {path}, request the multiplier, then reply.', name='key-worker')",
            (
                "await agent_message.send('Please provide the multiplier.', receiver_role='parent')",
                "Agent message sent: agentmsg_1",
            ),
            (
                "await agent_message.send('37', receiver_role='child', receiver_name=child.name)",
                "Agent message sent: agentmsg_2",
            ),
            (
                "await agent_message.send('subtotal=14 result=518', receiver_role='parent')",
                "Agent message sent: agentmsg_3",
            ),
        ),
        _child_message("key-worker", "Please provide the multiplier.", "agentmsg_1"),
        _child_message("key-worker", "subtotal=14 result=518", "agentmsg_3"),
    )

    behavior = _protocol_behavior(
        trace,
        "followup",
        ("key-worker",),
        {"key-worker": path},
        37,
    )

    assert behavior["natural_followup_causal"] == 1.0
    assert behavior["coordinator_delegated_path_accesses"] == 1.0
    assert behavior["stateful_control_progress"] == 0.0
    assert behavior["bidirectional_control"] == 0.0


def test_incomplete_followup_causality_zeroes_process_control_credit() -> None:
    path = "/workspace/followup.json"
    trace = _with_child_messages(
        _trace(
            f"child = await rlm('Sum {path}, then reply.', name='key-worker')",
            (
                "await agent_message.send('subtotal=14', receiver_role='parent')",
                "Agent message sent: agentmsg_1",
            ),
            (
                "await agent_message.send('37', receiver_role='child', receiver_name=child.name)",
                "Agent message sent: agentmsg_2",
            ),
            (
                "await agent_message.send('subtotal=14 result=518', receiver_role='parent')",
                "Agent message sent: agentmsg_3",
            ),
        ),
        _child_message("key-worker", "subtotal=14", "agentmsg_1"),
        _child_message("key-worker", "subtotal=14 result=518", "agentmsg_3"),
    )

    behavior = _protocol_behavior(
        trace,
        "followup",
        ("key-worker",),
        {"key-worker": path},
        37,
    )

    assert behavior["natural_followup_causal"] == 0.0
    assert behavior["coordinator_delegated_path_accesses"] == 0.0
    assert behavior["bidirectional_control"] == 0.0


def test_clean_single_protocol_requires_linked_child_send_without_polling() -> None:
    trace = _with_child_messages(
        _trace(
            "handle = await rlm('Compute /workspace/shard.json and reply with agent_message to parent.', name='shard-worker')",
            (
                "await agent_message.send('91', receiver_role='parent')",
                "Agent message sent: agentmsg_1",
            ),
        ),
        _child_message("shard-worker", "91", "agentmsg_1"),
    )

    behavior = _protocol_behavior(
        trace,
        "single",
        ("shard-worker",),
        {"shard-worker": "/workspace/shard.json"},
        None,
    )

    assert behavior["protocol_aligned"] == 1.0
    assert behavior["clean_protocol_aligned"] == 1.0
    assert behavior["explicit_messages_to_parent"] == 1.0
    assert behavior["failed_cells"] == 0.0


def test_clean_single_protocol_links_received_message_when_send_result_is_omitted() -> None:
    child_session = "019ff110-7ca9-7637-bcb6-7343dcd6f7e1"
    trace = _trace()
    trace.nodes = [
        MessageNode(parent=None, message=UserMessage(content="coordinate"), sampled=False),
        MessageNode(
            parent=0,
            message=AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="spawn",
                        name="ipython",
                        arguments=json.dumps(
                            {
                                "code": (
                                    "handle = await rlm('Compute /workspace/shard.json and reply with "
                                    "agent_message to parent.', name='shard-worker')"
                                )
                            }
                        ),
                    )
                ],
            ),
            sampled=True,
        ),
        MessageNode(parent=1, message=ToolMessage(tool_call_id="spawn", content="name='shard-worker'")),
        MessageNode(
            parent=None,
            message=SystemMessage(content=f"Conversation log: /tmp/agent/{child_session}.jsonl"),
            sampled=False,
        ),
        MessageNode(parent=3, message=UserMessage(content="[task from parent]"), sampled=False),
        MessageNode(
            parent=4,
            message=AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="send",
                        name="ipython",
                        arguments=json.dumps(
                            {"code": 'await agent_message.send("91", receiver_role="parent")'}
                        ),
                    )
                ],
            ),
            sampled=True,
        ),
        MessageNode(
            parent=2,
            message=UserMessage(
                content=(
                    "[from child:shard-worker]\n"
                    "Agent-to-agent message received.\n"
                    "Source: agent_message\n"
                    f"From: shard-worker, active child, session {child_session}, client agent\n"
                    "Message id: agentmsg_1\n\n91"
                )
            ),
            sampled=False,
        ),
        MessageNode(parent=6, message=AssistantMessage(content="{}"), sampled=True),
    ]

    behavior = _protocol_behavior(
        trace,
        "single",
        ("shard-worker",),
        {"shard-worker": "/workspace/shard.json"},
        None,
    )

    assert behavior["explicit_messages_to_parent"] == 1.0
    assert behavior["clean_protocol_aligned"] == 1.0


def test_missing_send_result_does_not_link_a_different_child_session() -> None:
    trace = _trace()
    trace.nodes = [
        MessageNode(parent=None, message=UserMessage(content="coordinate"), sampled=False),
        MessageNode(
            parent=0,
            message=AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="spawn",
                        name="ipython",
                        arguments=json.dumps(
                            {
                                "code": (
                                    "handle = await rlm('Compute /workspace/shard.json and reply with "
                                    "agent_message to parent.', name='shard-worker')"
                                )
                            }
                        ),
                    )
                ],
            ),
            sampled=True,
        ),
        MessageNode(parent=1, message=ToolMessage(tool_call_id="spawn", content="name='shard-worker'")),
        MessageNode(
            parent=None,
            message=SystemMessage(content="Conversation log: /tmp/agent/child-session-a.jsonl"),
            sampled=False,
        ),
        MessageNode(parent=3, message=UserMessage(content="[task from parent]"), sampled=False),
        MessageNode(
            parent=4,
            message=AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="send",
                        name="ipython",
                        arguments=json.dumps(
                            {"code": "await agent_message.send('91', receiver_role='parent')"}
                        ),
                    )
                ],
            ),
            sampled=True,
        ),
        MessageNode(
            parent=2,
            message=UserMessage(
                content=(
                    "[from child:shard-worker]\n"
                    "Agent-to-agent message received.\n"
                    "Source: agent_message\n"
                    "From: shard-worker, active child, session child-session-b, client agent\n"
                    "Message id: agentmsg_1\n\n91"
                )
            ),
            sampled=False,
        ),
        MessageNode(parent=6, message=AssistantMessage(content="{}"), sampled=True),
    ]

    behavior = _protocol_behavior(
        trace,
        "single",
        ("shard-worker",),
        {"shard-worker": "/workspace/shard.json"},
        None,
    )

    assert behavior["explicit_messages_to_parent"] == 0.0
    assert behavior["clean_protocol_aligned"] == 0.0


def test_post_fan_in_control_allows_local_evidence_consumption() -> None:
    trace = _with_child_messages(
        _trace(
            "alpha = await rlm('Compute /workspace/alpha.json and message parent.', name='alpha-worker')",
            "beta = await rlm('Compute /workspace/beta.json and message parent.', name='beta-worker')",
        ),
        _child_message("alpha-worker", "alpha=11", "agentmsg_1"),
        _child_message("beta-worker", "beta=17", "agentmsg_2"),
    )
    _append_ipython_before_final(trace, "total = local + 11 + 17\ntotal", "41")

    behavior = _protocol_behavior(
        trace,
        "parallel",
        ("alpha-worker", "beta-worker"),
        {
            "alpha-worker": "/workspace/alpha.json",
            "beta-worker": "/workspace/beta.json",
        },
        None,
    )

    assert behavior["post_fan_in_cells"] == 1.0
    assert behavior["post_fan_in_failed_cells"] == 0.0
    assert behavior["post_fan_in_forbidden_calls"] == 0.0
    assert behavior["post_fan_in_control_aligned"] == 1.0


def test_post_fan_in_control_rejects_polling_failures_and_repeated_cells() -> None:
    trace = _with_child_messages(
        _trace("handle = await rlm('Compute /workspace/shard.json and reply.', name='shard-worker')"),
        _child_message("shard-worker", "remote=91", "agentmsg_1"),
    )
    polling = "state = await agent_observe.get_agent(handle.name)\nstate"
    _append_ipython_before_final(trace, polling, "running")
    _append_ipython_before_final(trace, polling, "running")
    _append_ipython_before_final(
        trace,
        "len(reader)",
        "Traceback (most recent call last):\nTypeError: object has no len()",
    )

    behavior = _protocol_behavior(
        trace,
        "single",
        ("shard-worker",),
        {"shard-worker": "/workspace/shard.json"},
        None,
    )

    assert behavior["post_fan_in_cells"] == 3.0
    assert behavior["post_fan_in_failed_cells"] == 1.0
    assert behavior["post_fan_in_forbidden_calls"] == 2.0
    assert behavior["post_fan_in_duplicate_cells"] == 1.0
    assert behavior["post_fan_in_control"] == pytest.approx(4 / 9)
    assert behavior["post_fan_in_control_aligned"] == 0.0


def test_post_fan_in_control_rejects_direct_tools_and_inert_cells() -> None:
    trace = _with_child_messages(
        _trace("handle = await rlm('Compute /workspace/shard.json and reply.', name='shard-worker')"),
        _child_message("shard-worker", "remote=91", "agentmsg_1"),
    )
    _append_tool_before_final(trace, "agent_observe", {"child_name": "shard-worker"})
    _append_ipython_before_final(trace, "# waiting")

    behavior = _protocol_behavior(
        trace,
        "single",
        ("shard-worker",),
        {"shard-worker": "/workspace/shard.json"},
        None,
    )

    assert behavior["post_fan_in_forbidden_calls"] == 2.0
    assert behavior["post_fan_in_control_aligned"] == 0.0


def test_clean_protocol_rejects_any_post_fan_in_cell() -> None:
    trace = _with_child_messages(
        _trace("handle = await rlm('Compute /workspace/shard.json and reply.', name='shard-worker')"),
        _child_message("shard-worker", "remote=91", "agentmsg_1"),
    )
    _append_ipython_before_final(trace, "total = local + 91\ntotal", "103")

    behavior = _protocol_behavior(
        trace,
        "single",
        ("shard-worker",),
        {"shard-worker": "/workspace/shard.json"},
        None,
    )

    assert behavior["post_fan_in_control_aligned"] == 1.0
    assert behavior["post_fan_in_cells"] == 1.0
    assert behavior["clean_protocol_aligned"] == 0.0


def test_clean_protocol_rejects_child_tool_use_after_parent_send() -> None:
    trace = _trace()
    trace.nodes = [
        MessageNode(parent=None, message=UserMessage(content="coordinate"), sampled=False),
        MessageNode(
            parent=0,
            message=AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="spawn",
                        name="ipython",
                        arguments=json.dumps(
                            {
                                "code": (
                                    "handle = await rlm('Read /workspace/shard.json and reply with "
                                    "agent_message to parent.', name='shard-worker')"
                                )
                            }
                        ),
                    )
                ],
            ),
            sampled=True,
        ),
        MessageNode(parent=1, message=ToolMessage(tool_call_id="spawn", content="name='shard-worker'")),
        MessageNode(parent=None, message=UserMessage(content="[task from parent]"), sampled=False),
        MessageNode(
            parent=3,
            message=AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="send",
                        name="ipython",
                        arguments=json.dumps({"code": "await agent_message.send('91', receiver_role='parent')"}),
                    )
                ],
            ),
            sampled=True,
        ),
        MessageNode(parent=4, message=ToolMessage(tool_call_id="send", content="Agent message sent: agentmsg_1")),
        MessageNode(
            parent=5,
            message=AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="extra",
                        name="ipython",
                        arguments=json.dumps({"code": "print('sent')"}),
                    )
                ],
            ),
            sampled=True,
        ),
        MessageNode(parent=6, message=ToolMessage(tool_call_id="extra", content="sent")),
        MessageNode(parent=2, message=_child_message("shard-worker", "91", "agentmsg_1"), sampled=False),
        MessageNode(parent=8, message=AssistantMessage(content="{}"), sampled=True),
    ]

    behavior = _protocol_behavior(
        trace,
        "single",
        ("shard-worker",),
        {"shard-worker": "/workspace/shard.json"},
        None,
    )

    assert behavior["protocol_aligned"] == 1.0
    assert behavior["explicit_messages_to_parent"] == 1.0
    assert behavior["post_parent_send_tool_calls"] == 1.0
    assert behavior["clean_protocol_aligned"] == 0.0


def test_single_family_rejects_discarded_child_handle() -> None:
    trace = _with_child_messages(
        _trace(
            "await rlm('Compute /workspace/shard.json and reply with agent_message to parent.', name='shard-worker')"
        ),
        _child_message("shard-worker", "remote=91", "agentmsg_1"),
    )

    behavior = _protocol_behavior(
        trace,
        "single",
        ("shard-worker",),
        {"shard-worker": "/workspace/shard.json"},
        None,
    )

    assert behavior["protocol_aligned"] == 0.0
    assert behavior["retained_handles"] == 0.0


def test_single_family_reports_native_child_observation() -> None:
    trace = _with_child_messages(
        _trace(
            "handle = await rlm('Compute /workspace/shard.json and reply.', name='shard-worker')",
            "child_state = await agent_observe.get_agent(handle.name)",
            "(lambda: None)()",
        ),
        _child_message("shard-worker", "remote=91", "agentmsg_1"),
    )

    behavior = _protocol_behavior(
        trace,
        "single",
        ("shard-worker",),
        {"shard-worker": "/workspace/shard.json"},
        None,
    )

    assert behavior["protocol_aligned"] == 1.0
    assert behavior["observation_calls"] == 1.0


def test_parallel_family_requires_two_distinct_named_children() -> None:
    trace = _with_child_messages(
        _trace(
            "alpha = await rlm('Compute /workspace/alpha.json and message parent.', name='alpha-worker')",
            "beta = await rlm('Compute /workspace/beta.json and message parent.', name='beta-worker')",
        ),
        _child_message("alpha-worker", "alpha=11", "agentmsg_1"),
        _child_message("beta-worker", "beta=17", "agentmsg_2"),
    )

    behavior = _protocol_behavior(
        trace,
        "parallel",
        ("alpha-worker", "beta-worker"),
        {
            "alpha-worker": "/workspace/alpha.json",
            "beta-worker": "/workspace/beta.json",
        },
        None,
    )

    assert behavior["protocol_aligned"] == 1.0
    assert behavior["spawn_calls"] == 2.0
    assert behavior["named_children"] == 2.0


def test_followup_requires_bidirectional_messages_and_withheld_secret() -> None:
    trace = _with_child_messages(
        _trace(
            "child = await rlm('Sum /workspace/followup.json, request the multiplier, then message the result.', name='key-worker')",
            (
                "await agent_message.send('need multiplier', receiver_role='parent')",
                "Agent message sent: agentmsg_1",
            ),
            (
                "await agent_message.send('37', receiver_role='child', receiver_name=child.name)",
                "Agent message sent: agentmsg_2",
            ),
            (
                "await agent_message.send('result=518', receiver_role='parent')",
                "Agent message sent: agentmsg_3",
            ),
        ),
        _child_message("key-worker", "need multiplier", "agentmsg_1"),
        _child_message("key-worker", "result=518", "agentmsg_3"),
    )

    behavior = _protocol_behavior(
        trace,
        "followup",
        ("key-worker",),
        {"key-worker": "/workspace/followup.json"},
        37,
    )

    assert behavior["protocol_aligned"] == 1.0
    assert behavior["secret_withheld"] == 1.0
    assert behavior["messages_to_parent"] == 2.0
    assert behavior["messages_to_child"] == 1.0
    assert behavior["followup_phase_score"] == 1.0
    assert behavior["followup_causal"] == 1.0
    assert behavior["natural_followup_causal"] == 1.0
    assert behavior["stateful_control_progress"] == 1.0
    assert behavior["post_parent_send_tool_calls"] == 0.0
    assert behavior["clean_protocol_aligned"] == 1.0


def test_followup_accepts_a_natural_request_without_the_literal_wire_phrase() -> None:
    trace = _with_child_messages(
        _trace(
            "child = await rlm('Sum /workspace/followup.json, request the multiplier, then message the result.', name='key-worker')",
            (
                "await agent_message.send('Could you please provide the missing multiplier?', receiver_role='parent')",
                "Agent message sent: agentmsg_1",
            ),
            (
                "await agent_message.send('37', receiver_role='child', receiver_name=child.name)",
                "Agent message sent: agentmsg_2",
            ),
            (
                "await agent_message.send('subtotal=14 result=518', receiver_role='parent')",
                "Agent message sent: agentmsg_3",
            ),
        ),
        _child_message(
            "key-worker",
            "The subtotal is 14. Could you please provide the missing multiplier?",
            "agentmsg_1",
        ),
        _child_message("key-worker", "subtotal=14 result=518", "agentmsg_3"),
    )

    behavior = _protocol_behavior(
        trace,
        "followup",
        ("key-worker",),
        {"key-worker": "/workspace/followup.json"},
        37,
    )

    assert behavior["protocol_aligned"] == 1.0
    assert behavior["followup_causal"] == 0.0
    assert behavior["natural_request_sent"] == 1.0
    assert behavior["natural_followup_causal"] == 1.0
    assert behavior["stateful_control_progress"] == 1.0
    assert behavior["post_parent_send_tool_calls"] == 0.0
    assert behavior["clean_protocol_aligned"] == 1.0


def test_followup_rejects_a_generic_child_status_as_a_request() -> None:
    trace = _with_child_messages(
        _trace(
            "child = await rlm('Sum /workspace/followup.json, request the multiplier, then message the result.', name='key-worker')",
            (
                "await agent_message.send('I am waiting for more instructions.', receiver_role='parent')",
                "Agent message sent: agentmsg_1",
            ),
            (
                "await agent_message.send('37', receiver_role='child', receiver_name=child.name)",
                "Agent message sent: agentmsg_2",
            ),
            (
                "await agent_message.send('subtotal=14 result=518', receiver_role='parent')",
                "Agent message sent: agentmsg_3",
            ),
        ),
        _child_message("key-worker", "I am waiting for more instructions.", "agentmsg_1"),
        _child_message("key-worker", "subtotal=14 result=518", "agentmsg_3"),
    )

    behavior = _protocol_behavior(
        trace,
        "followup",
        ("key-worker",),
        {"key-worker": "/workspace/followup.json"},
        37,
    )

    assert behavior["protocol_aligned"] == 0.0
    assert behavior["natural_request_sent"] == 0.0
    assert behavior["natural_followup_causal"] == 0.0


def test_clean_followup_rejects_extra_child_work_after_the_result() -> None:
    trace = _with_child_messages(
        _trace(
            "child = await rlm('Sum /workspace/followup.json, request the multiplier, then message the result.', name='key-worker')",
            (
                "await agent_message.send('Please provide the multiplier.', receiver_role='parent')",
                "Agent message sent: agentmsg_1",
            ),
            (
                "await agent_message.send('37', receiver_role='child', receiver_name=child.name)",
                "Agent message sent: agentmsg_2",
            ),
            (
                "await agent_message.send('result=518', receiver_role='parent')",
                "Agent message sent: agentmsg_3",
            ),
            ("print('already sent')", "already sent"),
        ),
        _child_message("key-worker", "Please provide the multiplier.", "agentmsg_1"),
        _child_message("key-worker", "result=518", "agentmsg_3"),
    )

    behavior = _protocol_behavior(
        trace,
        "followup",
        ("key-worker",),
        {"key-worker": "/workspace/followup.json"},
        37,
    )

    assert behavior["protocol_aligned"] == 1.0
    assert behavior["post_parent_send_tool_calls"] == 1.0
    assert behavior["clean_protocol_aligned"] == 0.0


def test_clean_followup_rejects_coordinator_work_after_the_final_child_result() -> None:
    trace = _with_child_messages(
        _trace(
            "child = await rlm('Sum /workspace/followup.json, request the multiplier, then message the result.', name='key-worker')",
            (
                "await agent_message.send('Please provide the multiplier.', receiver_role='parent')",
                "Agent message sent: agentmsg_1",
            ),
            (
                "await agent_message.send('37', receiver_role='child', receiver_name=child.name)",
                "Agent message sent: agentmsg_2",
            ),
            (
                "await agent_message.send('result=518', receiver_role='parent')",
                "Agent message sent: agentmsg_3",
            ),
        ),
        _child_message("key-worker", "Please provide the multiplier.", "agentmsg_1"),
        _child_message("key-worker", "result=518", "agentmsg_3"),
    )
    _append_ipython_before_final(trace, "print('already complete')", "already complete")

    behavior = _protocol_behavior(
        trace,
        "followup",
        ("key-worker",),
        {"key-worker": "/workspace/followup.json"},
        37,
    )

    assert behavior["protocol_aligned"] == 1.0
    assert behavior["post_result_coordinator_cells"] == 1.0
    assert behavior["clean_protocol_aligned"] == 0.0


def test_stateful_control_progress_requires_a_retained_handle() -> None:
    trace = _with_child_messages(
        _trace(
            "await rlm('Sum /workspace/followup.json, request the multiplier, then message the result.', name='key-worker')",
            (
                "await agent_message.send('need multiplier', receiver_role='parent')",
                "Agent message sent: agentmsg_1",
            ),
            (
                "await agent_message.send('37', receiver_role='child', receiver_name='key-worker')",
                "Agent message sent: agentmsg_2",
            ),
            (
                "await agent_message.send('result=518', receiver_role='parent')",
                "Agent message sent: agentmsg_3",
            ),
        ),
        _child_message("key-worker", "need multiplier", "agentmsg_1"),
        _child_message("key-worker", "result=518", "agentmsg_3"),
    )

    behavior = _protocol_behavior(
        trace,
        "followup",
        ("key-worker",),
        {"key-worker": "/workspace/followup.json"},
        37,
    )

    assert behavior["followup_causal"] == 1.0
    assert behavior["retained_handles"] == 0.0
    assert behavior["stateful_control_progress"] == 0.0


def test_followup_rejects_result_sent_before_parent_followup() -> None:
    trace = _with_child_messages(
        _trace(
            "child = await rlm('Sum /workspace/followup.json, request the multiplier, then message the result.', name='key-worker')",
            (
                "await agent_message.send('need multiplier', receiver_role='parent')",
                "Agent message sent: agentmsg_1",
            ),
            (
                "await agent_message.send('result=460', receiver_role='parent')",
                "Agent message sent: agentmsg_3",
            ),
            (
                "await agent_message.send('37', receiver_role='child', receiver_name=child.name)",
                "Agent message sent: agentmsg_2",
            ),
        ),
        _child_message("key-worker", "need multiplier", "agentmsg_1"),
        _child_message("key-worker", "result=460", "agentmsg_3"),
    )

    behavior = _protocol_behavior(
        trace,
        "followup",
        ("key-worker",),
        {"key-worker": "/workspace/followup.json"},
        37,
    )

    assert behavior["protocol_aligned"] == 0.0
    assert behavior["followup_request_sent"] == 1.0
    assert behavior["followup_after_request"] == 1.0
    assert behavior["result_after_followup"] == 0.0
    assert behavior["followup_phase_score"] == pytest.approx(2 / 3)
    assert behavior["followup_causal"] == 0.0


def test_handshake_requires_child_to_echo_the_followup_nonce() -> None:
    trace = _with_child_messages(
        _trace(
            "child = await rlm('need nonce, then echo it.', name='relay-worker')",
            (
                "await agent_message.send('need nonce', receiver_role='parent')",
                "Agent message sent: agentmsg_1",
            ),
            (
                "await agent_message.send('4812', receiver_role='child', receiver_name=child.name)",
                "Agent message sent: agentmsg_2",
            ),
            (
                "await agent_message.send('4812', receiver_role='parent')",
                "Agent message sent: agentmsg_3",
            ),
        ),
        _child_message("relay-worker", "need nonce", "agentmsg_1"),
        _child_message("relay-worker", "4812", "agentmsg_3"),
    )

    behavior = _protocol_behavior(
        trace,
        "handshake",
        ("relay-worker",),
        {"relay-worker": "need nonce"},
        4812,
    )

    assert behavior["protocol_aligned"] == 1.0
    assert behavior["retained_handles"] == 1.0
    assert behavior["followup_causal"] == 1.0
    assert behavior["followup_result_matches_secret"] == 1.0


def test_handshake_accepts_a_natural_child_contract() -> None:
    trace = _with_child_messages(
        _trace(
            "child = await rlm('Ask your parent to provide the missing nonce, then echo the reply.', name='relay-worker')",
            (
                "await agent_message.send('Please provide the nonce.', receiver_role='parent')",
                "Agent message sent: agentmsg_1",
            ),
            (
                "await agent_message.send('4812', receiver_role='child', receiver_name=child.name)",
                "Agent message sent: agentmsg_2",
            ),
            (
                "await agent_message.send('The nonce is 4812.', receiver_role='parent')",
                "Agent message sent: agentmsg_3",
            ),
        ),
        _child_message("relay-worker", "Please provide the nonce.", "agentmsg_1"),
        _child_message("relay-worker", "The nonce is 4812.", "agentmsg_3"),
    )

    behavior = _protocol_behavior(
        trace,
        "handshake",
        ("relay-worker",),
        {"relay-worker": "need nonce"},
        4812,
    )

    assert behavior["protocol_aligned"] == 1.0
    assert behavior["followup_causal"] == 0.0
    assert behavior["natural_followup_causal"] == 1.0
    assert behavior["delegated_payloads"] == 1.0


def test_followup_rejects_secret_leaked_in_spawn_prompt() -> None:
    trace = _with_child_messages(
        _trace(
            "child = await rlm('Sum /workspace/followup.json with multiplier 37 and reply.', name='key-worker')",
            (
                "await agent_message.send('37', receiver_role='child', receiver_name=child.name)",
                "Agent message sent: agentmsg_2",
            ),
        ),
        _child_message("key-worker", "need multiplier", "agentmsg_1"),
        _child_message("key-worker", "result=518", "agentmsg_3"),
    )

    behavior = _protocol_behavior(
        trace,
        "followup",
        ("key-worker",),
        {"key-worker": "/workspace/followup.json"},
        37,
    )

    assert behavior["protocol_aligned"] == 0.0
    assert behavior["secret_withheld"] == 0.0


def test_non_native_wrapper_and_repeated_cells_do_not_pass_protocol() -> None:
    source = "handle = await rlm.run('work', name='shard-worker')"
    trace = _trace(
        source,
        source,
        "await agent_message.send('answer', receiver_role='parent')",
    )

    behavior = _protocol_behavior(
        trace,
        "single",
        ("shard-worker",),
        {"shard-worker": "/workspace/shard.json"},
        None,
    )

    assert behavior["protocol_aligned"] == 0.0
    assert behavior["spawn_calls"] == 0.0
    assert behavior["duplicate_cells"] == 1.0


def test_parent_side_send_is_not_credited_as_a_child_reply() -> None:
    trace = _trace(
        "handle = await rlm('Compute /workspace/shard.json and reply.', name='shard-worker')",
        (
            "await agent_message.send('answer', receiver_role='parent')",
            "Agent message sent: agentmsg_1",
        ),
    )

    behavior = _protocol_behavior(
        trace,
        "single",
        ("shard-worker",),
        {"shard-worker": "/workspace/shard.json"},
        None,
    )

    assert behavior["messages_to_parent"] == 0.0
    assert behavior["protocol_aligned"] == 0.0


def test_protocol_behavior_attributes_calls_to_their_agent_branches() -> None:
    trace = _trace()
    trace.nodes = [
        MessageNode(parent=None, message=UserMessage(content="coordinate"), sampled=False),
        MessageNode(
            parent=0,
            message=AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="spawn",
                        name="ipython",
                        arguments=json.dumps(
                            {
                                "code": (
                                    "child = await rlm('Read /workspace/followup.json, request the multiplier, "
                                    "then message the result.', name='key-worker')"
                                )
                            }
                        ),
                    )
                ],
            ),
            sampled=True,
        ),
        MessageNode(parent=1, message=ToolMessage(tool_call_id="spawn", content="name='key-worker'"), sampled=False),
        MessageNode(parent=None, message=UserMessage(content="[task from parent]"), sampled=False),
        MessageNode(
            parent=3,
            message=AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="recursive",
                        name="ipython",
                        arguments=json.dumps({"code": "await rlm('unwanted recursive work')"}),
                    )
                ],
            ),
            sampled=True,
        ),
        MessageNode(
            parent=4,
            message=ToolMessage(tool_call_id="recursive", content="RuntimeError: child spawning is disabled"),
            sampled=False,
        ),
        MessageNode(
            parent=5,
            message=AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="request",
                        name="ipython",
                        arguments=json.dumps(
                            {"code": "await agent_message.send('need multiplier', receiver_role='parent')"}
                        ),
                    )
                ],
            ),
            sampled=True,
        ),
        MessageNode(
            parent=6,
            message=ToolMessage(tool_call_id="request", content="Agent message sent: agentmsg_request"),
            sampled=False,
        ),
        MessageNode(
            parent=2, message=_child_message("key-worker", "need multiplier", "agentmsg_request"), sampled=False
        ),
        MessageNode(
            parent=8,
            message=AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="followup",
                        name="ipython",
                        arguments=json.dumps(
                            {
                                "code": (
                                    "await agent_message.send('37', receiver_role='child', receiver_name=child.name)"
                                )
                            }
                        ),
                    )
                ],
            ),
            sampled=True,
        ),
        MessageNode(
            parent=9,
            message=ToolMessage(tool_call_id="followup", content="Agent message sent: agentmsg_followup"),
            sampled=False,
        ),
        MessageNode(parent=7, message=UserMessage(content="[from parent]\n\n37"), sampled=False),
        MessageNode(
            parent=11,
            message=AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="result",
                        name="ipython",
                        arguments=json.dumps(
                            {"code": "await agent_message.send('result=518', receiver_role='parent')"}
                        ),
                    )
                ],
            ),
            sampled=True,
        ),
        MessageNode(
            parent=12,
            message=ToolMessage(tool_call_id="result", content="Agent message sent: agentmsg_result"),
            sampled=False,
        ),
        MessageNode(parent=10, message=_child_message("key-worker", "result=518", "agentmsg_result"), sampled=False),
        MessageNode(parent=14, message=AssistantMessage(content="{}"), sampled=True),
    ]

    behavior = _protocol_behavior(
        trace,
        "followup",
        ("key-worker",),
        {"key-worker": "/workspace/followup.json"},
        37,
    )

    assert behavior["spawn_calls"] == 1.0
    assert behavior["failed_spawn_calls"] == 0.0
    assert behavior["messages_to_parent"] == 2.0
    assert behavior["messages_to_child"] == 1.0
    assert behavior["followup_causal"] == 1.0


def test_terminal_child_notice_is_not_credited_as_an_explicit_reply() -> None:
    trace = _with_child_messages(
        _trace("handle = await rlm('Compute /workspace/shard.json and reply.', name='shard-worker')"),
        _child_message(
            "shard-worker",
            "RLM child shard-worker (sub-123) completed without sending a reply",
            "agentmsg_1",
        ),
    )

    behavior = _protocol_behavior(
        trace,
        "single",
        ("shard-worker",),
        {"shard-worker": "/workspace/shard.json"},
        None,
    )

    assert behavior["messages_to_parent"] == 0.0
    assert behavior["protocol_aligned"] == 0.0


@pytest.mark.asyncio
async def test_delegated_answer_credit_is_gated_on_protocol_alignment() -> None:
    task = next(
        task
        for task in SubagentCommunicationTaskset(
            SubagentCommunicationConfig(split="train", instances_per_template=1)
        ).load()
        if task.data.family == "single"
    )
    trace = _trace(reply=json.dumps(task.data.answer))

    assert await task.protocol_gated_accuracy(trace) == 0.0
    assert await task.answer_accuracy(trace) == 1.0


@pytest.mark.asyncio
async def test_post_fan_in_reward_is_opt_in_and_scoped_to_delegated_families() -> None:
    enabled_task = next(
        task
        for task in SubagentCommunicationTaskset(
            SubagentCommunicationConfig(
                split="train",
                families=("single",),
                instances_per_template=1,
                reward_post_fan_in_control=True,
            )
        ).load()
    )
    disabled_task = next(
        task
        for task in SubagentCommunicationTaskset(
            SubagentCommunicationConfig(
                split="train",
                families=("single",),
                instances_per_template=1,
            )
        ).load()
    )
    path = enabled_task.data.child_paths["shard-worker"]
    trace = _with_child_messages(
        _trace(f"handle = await rlm('Compute {path} and reply.', name='shard-worker')"),
        _child_message("shard-worker", "remote=91", "agentmsg_1"),
    )

    assert await enabled_task.post_fan_in_control_reward(trace) == 1.0
    assert await disabled_task.post_fan_in_control_reward(trace) == 0.0


@pytest.mark.asyncio
async def test_bidirectional_control_reward_is_opt_in_and_requires_clean_execution() -> None:
    enabled_task = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="train",
            families=("followup",),
            instances_per_template=1,
            reward_bidirectional_control=True,
        )
    ).load()[0]
    disabled_task = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="train",
            families=("followup",),
            instances_per_template=1,
        )
    ).load()[0]
    path = enabled_task.data.child_paths["key-worker"]
    trace = _with_child_messages(
        _trace(
            f"child = await rlm('Sum {path}, request the multiplier, then reply.', name='key-worker')",
            (
                "await agent_message.send('Please provide the multiplier.', receiver_role='parent')",
                "Agent message sent: agentmsg_1",
            ),
            (
                "await agent_message.send('37', receiver_role='child', receiver_name=child.name)",
                "Agent message sent: agentmsg_2",
            ),
            (
                "await agent_message.send('subtotal=14 result=518', receiver_role='parent')",
                "Agent message sent: agentmsg_3",
            ),
        ),
        _child_message("key-worker", "Please provide the multiplier.", "agentmsg_1"),
        _child_message("key-worker", "subtotal=14 result=518", "agentmsg_3"),
    )

    assert await enabled_task.bidirectional_control_reward(trace) == 1.0
    assert await disabled_task.bidirectional_control_reward(trace) == 0.0


@pytest.mark.asyncio
async def test_ownership_transition_reward_isolates_first_coordinator_response() -> None:
    task = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="train",
            families=("followup",),
            instances_per_template=1,
            task=SubagentCommunicationTaskConfig(
                reward_mode="ownership_transition",
                reward_shape="strict",
            ),
        )
    ).load()[0]
    path = task.data.child_paths["key-worker"]
    secret = task.data.followup_secret
    prompt = f"Read {path}, request the multiplier, then reply to the parent."
    trace = _trace(
        f"multiplier = {secret}\nchild = await rlm(prompt={prompt!r}, name='key-worker')"
    )

    behavior = _ownership_transition_behavior(
        trace,
        task.data.family,
        task.data.expected_children,
        task.data.child_paths,
        secret,
    )
    assert behavior == {
        "ownership_transition": 1.0,
        "ownership_transition_dense": 1.0,
        "ownership_one_spawn": 1.0,
        "ownership_retained_secret": 1.0,
        "ownership_retained_handle": 1.0,
        "ownership_named_child": 1.0,
        "ownership_delegated_payload": 1.0,
        "ownership_secret_withheld": 1.0,
        "ownership_path_owned": 1.0,
    }
    assert await task.ownership_transition_reward(trace) == 1.0
    assert await task.protocol_gated_accuracy(trace) == 0.0
    assert await task.delegation_protocol(trace) == 0.0
    assert await task.stateful_control_progress(trace) == 0.0


@pytest.mark.asyncio
async def test_ownership_transition_retains_handshake_nonce() -> None:
    task = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="train",
            families=("handshake",),
            instances_per_template=1,
            task=SubagentCommunicationTaskConfig(
                reward_mode="ownership_transition",
                reward_shape="strict",
            ),
        )
    ).load()[0]
    secret = task.data.followup_secret
    prompt = "Ask your parent for the missing nonce, then echo its reply."
    trace = _trace(
        f"nonce = {secret}\nchild = await rlm({prompt!r}, name='relay-worker')"
    )

    behavior = _ownership_transition_behavior(
        trace,
        task.data.family,
        task.data.expected_children,
        task.data.child_paths,
        secret,
    )
    assert behavior["ownership_retained_secret"] == 1.0
    assert behavior["ownership_transition"] == 1.0
    assert await task.ownership_transition_reward(trace) == 1.0


@pytest.mark.asyncio
async def test_ownership_transition_rejects_coordinator_path_access() -> None:
    task = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="train",
            families=("followup",),
            instances_per_template=1,
            task=SubagentCommunicationTaskConfig(
                reward_mode="ownership_transition",
                reward_shape="dense",
            ),
        )
    ).load()[0]
    path = task.data.child_paths["key-worker"]
    secret = task.data.followup_secret
    prompt = f"Read {path}, request the multiplier, then reply to the parent."
    trace = _trace(
        f"contents = open({path!r}).read()\n"
        f"multiplier = {secret}\n"
        f"child = await rlm({prompt!r}, name='key-worker')"
    )

    behavior = _ownership_transition_behavior(
        trace,
        task.data.family,
        task.data.expected_children,
        task.data.child_paths,
        secret,
    )
    assert behavior["ownership_transition"] == 0.0
    assert behavior["ownership_transition_dense"] == pytest.approx(6 / 7)
    assert await task.ownership_transition_reward(trace) == pytest.approx(6 / 7)


@pytest.mark.asyncio
async def test_ownership_transition_requires_persistent_secret_state() -> None:
    task = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="train",
            families=("followup",),
            instances_per_template=1,
            task=SubagentCommunicationTaskConfig(
                reward_mode="ownership_transition",
                reward_shape="strict",
            ),
        )
    ).load()[0]
    path = task.data.child_paths["key-worker"]
    secret = task.data.followup_secret
    prompt = f"Read {path}, request the multiplier, then reply to the parent."
    trace = _trace(f"child = await rlm({prompt!r}, name='key-worker')")

    behavior = _ownership_transition_behavior(
        trace,
        task.data.family,
        task.data.expected_children,
        task.data.child_paths,
        secret,
    )
    assert behavior["ownership_retained_secret"] == 0.0
    assert behavior["ownership_transition"] == 0.0
    assert behavior["ownership_transition_dense"] == pytest.approx(6 / 7)
    assert await task.ownership_transition_reward(trace) == 0.0


@pytest.mark.parametrize(
    ("reply", "expected", "score"),
    [
        ('{"a": 1, "b": 2}', {"a": 1, "b": 2}, 1.0),
        ('{"a": 1, "b": 9}', {"a": 1, "b": 2}, 0.5),
        ('```json\n{"a": 1}\n```', {"a": 1}, 0.0),
    ],
)
def test_answer_score_is_strict_json_with_keywise_credit(reply: str, expected: dict[str, int], score: float) -> None:
    assert _answer_score(reply, expected) == score


def test_ownership_guidance_is_answer_free_and_opt_in() -> None:
    guided = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="train",
            families=("followup",),
            instances_per_template=1,
            ownership_guided=True,
        )
    ).load()[0]
    standard = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="train",
            families=("followup",),
            instances_per_template=1,
        )
    ).load()[0]

    assert OWNERSHIP_GUIDANCE in guided.data.system_prompt
    assert OWNERSHIP_GUIDANCE not in standard.data.system_prompt
    assert str(guided.data.followup_secret) not in OWNERSHIP_GUIDANCE
    assert next(iter(guided.data.child_paths.values())) not in OWNERSHIP_GUIDANCE
    assert all(str(value) not in OWNERSHIP_GUIDANCE for value in guided.data.answer.values())


def test_ownership_guidance_cannot_mix_with_answer_demonstration() -> None:
    taskset = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="train",
            families=("followup",),
            instances_per_template=1,
            teacher_conditioned=True,
            ownership_guided=True,
        )
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        taskset.load()

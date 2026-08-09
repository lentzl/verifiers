import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from subagent_communication_v1.taskset import (
    COMPLETION_GATE_PATH,
    SubagentCommunicationConfig,
    SubagentCommunicationTaskset,
    _answer_score,
    _completion_gate_source,
    _duplicate_cells,
    _ipython_events,
    _protocol_behavior,
)
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, ToolCall, ToolMessage, UserMessage


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
    assert f"Read {path}, compute its weighted checksum" in demonstration
    assert "handle = await rlm(" in demonstration
    assert json.dumps(task.data.answer) in demonstration
    demonstrations = task.data.demonstrations
    assert demonstrations is not None
    assert demonstrations[task.data.prompt] == demonstration
    child_question = next(question for question in demonstrations if question.startswith("[task from parent]"))
    assert task.data.child_paths["shard-worker"] in child_question
    assert "await agent_message.send(str(checksum), receiver_role='parent')" in demonstrations[child_question]


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
    assert task.data.prompt.endswith(
        "Now answer with a response of your own, including the thinking process:"
    )


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
    assert all("need nonce" in task.data.prompt for task in tasks)
    spawn_prompts = [
        task.data.prompt.split('rlm("', 1)[1].split('", name=', 1)[0]
        for task in tasks
    ]
    assert all(str(task.data.followup_secret) not in prompt for task, prompt in zip(tasks, spawn_prompts))


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
    assert "agent_observe.get_agent" not in single.data.prompt
    assert "receiver_role='child'" in followup.data.prompt
    assert "name='key-worker'" in followup.data.prompt
    assert "You are key-worker, my child" in followup.data.prompt
    assert "Do not call rlm or message a child" in followup.data.prompt
    assert "resume only when my parent follow-up arrives" in followup.data.prompt
    assert f"retain multiplier = {followup.data.followup_secret}" in followup.data.prompt
    assert "bind the integer body of the latest [from parent] message" in followup.data.prompt
    assert "rather than guessing or hardcoding it" in followup.data.prompt
    assert json.dumps(single.data.answer) not in single.data.prompt


def test_completion_gate_requires_child_evidence_without_embedding_answer_values(
    tmp_path: Path,
) -> None:
    agent_dir = tmp_path / "agent"
    sessions = agent_dir / "sessions"
    sessions.mkdir(parents=True)
    session = sessions / "root.jsonl"
    source = _completion_gate_source(
        ("subtotal", "multiplier", "result"), "followup"
    )
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
                        "[from child:key-worker]\n"
                        "Agent-to-agent message received.\n"
                        "Source: agent_message\n\n"
                        f"{message}"
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
        session.write_text(
            "\n".join((json.dumps({"type": "session", "rlmDepth": 0}), *ordered))
            + "\n"
        )
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
    assert "brief waiting status and no tool call" in failed.stderr
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

    assert runtime.runs == [
        ["mkdir", "-p", "/workspace/.subagent-communication", "/workspace/subagent-shards"]
    ]
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
    assert behavior["retained_handles"] == 1.0
    assert behavior["messages_to_parent"] == 1.0


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
    assert behavior["stateful_control_progress"] == 1.0


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


def test_terminal_child_notice_is_not_credited_as_an_explicit_reply() -> None:
    trace = _with_child_messages(
        _trace(
            "handle = await rlm('Compute /workspace/shard.json and reply.', name='shard-worker')"
        ),
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

import json
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from subagent_communication_v1.taskset import (
    SubagentCommunicationConfig,
    SubagentCommunicationTaskset,
    _answer_score,
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
    assert "receiver_role='child'" in followup.data.prompt
    assert "name='key-worker'" in followup.data.prompt
    assert json.dumps(single.data.answer) not in single.data.prompt


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

    assert runtime.runs == [["mkdir", "-p", "/workspace/subagent-shards"]]
    assert runtime.writes == {path: contents.encode() for path, contents in task.data.files.items()}
    assert set(task.data.child_paths.values()) == set(runtime.writes)


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


def test_single_family_requires_retained_native_handle_and_child_reply() -> None:
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

    assert behavior["protocol_aligned"] == 1.0
    assert behavior["secret_withheld"] == 1.0
    assert behavior["messages_to_parent"] == 2.0
    assert behavior["messages_to_child"] == 1.0


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

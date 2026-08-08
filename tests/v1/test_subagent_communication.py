import json

import pytest
import verifiers.v1 as vf
from subagent_communication_v1.taskset import (
    SubagentCommunicationConfig,
    SubagentCommunicationTaskset,
    _answer_score,
    _protocol_behavior,
)
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, ToolCall, UserMessage


def _trace(*code: str, reply: str = "{}") -> vf.Trace:
    nodes = [
        MessageNode(
            parent=None,
            message=UserMessage(content="coordinate"),
            sampled=False,
        )
    ]
    parent = 0
    for index, source in enumerate(code):
        nodes.append(
            MessageNode(
                parent=parent,
                message=AssistantMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=f"call-{index}",
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


def test_guided_tasks_explain_native_contract_without_revealing_answers() -> None:
    tasks = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(split="train", instruction_level="guided", instances_per_template=1)
    ).load()

    single = next(task for task in tasks if task.data.family == "single")
    followup = next(task for task in tasks if task.data.family == "followup")

    assert "handle = await rlm" in single.data.prompt
    assert "receiver_role='child'" in followup.data.prompt
    assert json.dumps(single.data.answer) not in single.data.prompt


def test_direct_family_rewards_restraint() -> None:
    behavior = _protocol_behavior(
        _trace("values = [2, 3]\nsum((i + 1) * value for i, value in enumerate(values))"),
        "direct",
        (),
        None,
    )

    assert behavior["protocol_aligned"] == 1.0
    assert behavior["spawn_calls"] == 0.0


def test_single_family_requires_retained_native_handle_and_child_reply() -> None:
    trace = _trace(
        "handle = await rlm('Compute the shard and reply with agent_message to parent.', name='shard-worker')",
        "await agent_message.send('remote=91', receiver_role='parent')",
    )

    behavior = _protocol_behavior(trace, "single", ("shard-worker",), None)

    assert behavior["protocol_aligned"] == 1.0
    assert behavior["retained_handles"] == 1.0
    assert behavior["messages_to_parent"] == 1.0


def test_parallel_family_requires_two_distinct_named_children() -> None:
    trace = _trace(
        "alpha = await rlm('Compute alpha and message parent.', name='alpha-worker')",
        "beta = await rlm('Compute beta and message parent.', name='beta-worker')",
        "await agent_message.send('alpha=11', receiver_role='parent')",
        "await agent_message.send('beta=17', receiver_role='parent')",
    )

    behavior = _protocol_behavior(
        trace,
        "parallel",
        ("alpha-worker", "beta-worker"),
        None,
    )

    assert behavior["protocol_aligned"] == 1.0
    assert behavior["spawn_calls"] == 2.0
    assert behavior["named_children"] == 2.0


def test_followup_requires_bidirectional_messages_and_withheld_secret() -> None:
    trace = _trace(
        "child = await rlm('Sum [2, 4, 8], request the multiplier, then message the result.', name='key-worker')",
        "await agent_message.send('need multiplier', receiver_role='parent')",
        "await agent_message.send('37', receiver_role='child', receiver_name=child.name)",
        "await agent_message.send('result=518', receiver_role='parent')",
    )

    behavior = _protocol_behavior(trace, "followup", ("key-worker",), 37)

    assert behavior["protocol_aligned"] == 1.0
    assert behavior["secret_withheld"] == 1.0
    assert behavior["messages_to_parent"] == 2.0
    assert behavior["messages_to_child"] == 1.0


def test_followup_rejects_secret_leaked_in_spawn_prompt() -> None:
    trace = _trace(
        "child = await rlm('Sum [2, 4, 8] with multiplier 37 and reply.', name='key-worker')",
        "await agent_message.send('need multiplier', receiver_role='parent')",
        "await agent_message.send('37', receiver_role='child', receiver_name=child.name)",
        "await agent_message.send('result=518', receiver_role='parent')",
    )

    behavior = _protocol_behavior(trace, "followup", ("key-worker",), 37)

    assert behavior["protocol_aligned"] == 0.0
    assert behavior["secret_withheld"] == 0.0


def test_non_native_wrapper_and_repeated_cells_do_not_pass_protocol() -> None:
    source = "handle = await rlm.run('work', name='shard-worker')"
    trace = _trace(
        source,
        source,
        "await agent_message.send('answer', receiver_role='parent')",
    )

    behavior = _protocol_behavior(trace, "single", ("shard-worker",), None)

    assert behavior["protocol_aligned"] == 0.0
    assert behavior["spawn_calls"] == 0.0
    assert behavior["identical_consecutive_cells"] == 1.0


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

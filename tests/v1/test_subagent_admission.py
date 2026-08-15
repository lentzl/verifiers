import asyncio
import json

import subagent_admission_v1.taskset as admission_module
from subagent_admission_v1.taskset import (
    SubagentAdmissionConfig,
    SubagentAdmissionTask,
    SubagentAdmissionTaskConfig,
    SubagentAdmissionTaskset,
    _admission_behavior,
    _causal_behavior,
    _clean_causal_behavior,
)
from subagent_communication_v1.taskset import SubagentCommunicationData

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, ToolCall, ToolMessage, UserMessage

PATH = "/workspace/subagent-shards/test-remote.json"


def _trace(*cells: str) -> vf.Trace:
    nodes = [MessageNode(parent=None, message=UserMessage(content="task"), sampled=False)]
    parent = 0
    for index, code in enumerate(cells):
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
                message=ToolMessage(tool_call_id=call_id, content=""),
                sampled=False,
            )
        )
        parent = len(nodes) - 1
    return vf.Trace(
        id="admission-test",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="SubagentAdmissionTask", data=vf.TaskData(idx=0)),
        nodes=nodes,
    )


def _data() -> SubagentCommunicationData:
    return SubagentCommunicationData(
        idx=0,
        name="single-test",
        prompt="task",
        family="single",
        template_variant=0,
        answer={"local": 3, "remote": 4, "total": 7},
        expected_children=("shard-worker",),
        child_paths={"shard-worker": PATH},
    )


def _prompt() -> str:
    return f"Read {PATH}, compute it, then reply to your parent with agent_message."


def test_exact_spawn_before_local_work_is_aligned() -> None:
    behavior = _admission_behavior(
        _trace(
            f"handle = await rlm({_prompt()!r}, name='shard-worker')",
            "local_values = [1, 2]\nlocal = sum(local_values)\nlocal",
        ),
        _data(),
    )

    assert behavior["admission_score"] == 1.0
    assert behavior["admission_aligned"] == 1.0


def test_local_work_before_spawn_is_not_aligned() -> None:
    behavior = _admission_behavior(
        _trace(
            "local_values = [1, 2]\nlocal = sum(local_values)\nlocal",
            f"handle = await rlm({_prompt()!r}, name='shard-worker')",
        ),
        _data(),
    )

    assert behavior["spawn_first_cell"] == 0.0
    assert behavior["admission_aligned"] == 0.0


def test_dense_shape_credits_late_exact_spawn_without_rewarding_order() -> None:
    behavior = _admission_behavior(
        _trace(
            "local_values = [1, 2]\nlocal = sum(local_values)\nlocal",
            f"handle = await rlm({_prompt()!r}, name='shard-worker')",
        ),
        _data(),
        reward_shape="dense",
    )

    assert behavior["admission_score"] == 0.5
    assert behavior["spawn_first_cell"] == 0.0
    assert behavior["spawn_precedes_local"] == 0.0
    assert behavior["retained_admission_handle"] == 1.0
    assert behavior["exact_admission_payload"] == 1.0
    assert behavior["local_work_after_admission"] == 0.0
    assert behavior["admission_aligned"] == 0.0


def test_generic_payload_gets_partial_credit_only() -> None:
    behavior = _admission_behavior(
        _trace(
            "handle = await rlm('sub-task', name='shard-worker')",
            "local_values = [1, 2]\nlocal = sum(local_values)\nlocal",
        ),
        _data(),
    )

    assert 0.0 < behavior["admission_score"] < 1.0
    assert behavior["exact_admission_payload"] == 0.0


def test_parent_cannot_read_delegated_shard_after_spawning() -> None:
    behavior = _admission_behavior(
        _trace(
            f"handle = await rlm({_prompt()!r}, name='shard-worker')",
            f"remote_values = json.loads(Path({PATH!r}).read_text())",
            "local_values = [1, 2]\nlocal = sum(local_values)\nlocal",
        ),
        _data(),
    )

    assert behavior["parent_remote_read"] == 1.0
    assert behavior["parent_remote_read_before_admission"] == 1.0
    assert behavior["admission_aligned"] == 0.0


def test_taskset_reuses_frozen_single_task_generation() -> None:
    task = SubagentAdmissionTaskset(
        SubagentAdmissionConfig(
            split="train",
            families=("single",),
            instances_per_template=1,
        )
    ).load()[0]

    assert task.data.family == "single"
    assert task.data.child_paths["shard-worker"] in task.data.prompt


def test_self_contained_contract_adds_formula_without_changing_default() -> None:
    default = SubagentAdmissionTaskset(
        SubagentAdmissionConfig(
            split="train",
            families=("single",),
            instances_per_template=1,
        )
    ).load()[0]
    self_contained = SubagentAdmissionTaskset(
        SubagentAdmissionConfig(
            split="train",
            families=("single",),
            instances_per_template=1,
            self_contained_child_contract=True,
        )
    ).load()[0]

    formula = "sum((index + 1) * value for index, value in enumerate(values))"
    assert formula not in default.data.prompt
    assert formula in self_contained.data.prompt
    assert default.data.answer == self_contained.data.answer
    assert default.data.files == self_contained.data.files

    for field in (
        "demonstrations",
        "turn_demonstrations",
        "child_request_demonstrations",
        "coordinator_demonstrations",
    ):
        default_mapping = getattr(default.data, field)
        self_contained_mapping = getattr(self_contained.data, field)
        if default_mapping is None:
            assert self_contained_mapping is None
            continue
        assert self_contained_mapping is not None
        assert default.data.prompt in default_mapping
        assert default.data.prompt not in self_contained_mapping
        assert self_contained.data.prompt in self_contained_mapping

    assert self_contained.data.demonstrations[self_contained.data.prompt] is not None
    assert self_contained.data.coordinator_demonstrations[self_contained.data.prompt] is not None


def test_self_contained_contract_does_not_change_direct_tasks() -> None:
    default = SubagentAdmissionTaskset(
        SubagentAdmissionConfig(
            split="train",
            families=("direct",),
            instances_per_template=1,
        )
    ).load()[0]
    self_contained = SubagentAdmissionTaskset(
        SubagentAdmissionConfig(
            split="train",
            families=("direct",),
            instances_per_template=1,
            self_contained_child_contract=True,
        )
    ).load()[0]

    assert self_contained.data == default.data


def test_admission_mode_zeros_inherited_rewards() -> None:
    trace = _trace(
        f"handle = await rlm({_prompt()!r}, name='shard-worker')",
        "local_values = [1, 2]\nlocal = sum(local_values)\nlocal",
    )
    task = SubagentAdmissionTask(_data(), SubagentAdmissionTaskConfig())

    asyncio.run(task.score(trace))

    assert trace.rewards["admission_control"].score == 1.0
    assert trace.rewards["protocol_gated_accuracy"].score == 0.0
    assert trace.rewards["delegation_protocol"].score == 0.0
    assert trace.rewards["stateful_control_progress"].score == 0.0
    assert trace.rewards["post_fan_in_control_reward"].score == 0.0
    assert trace.reward == 1.0


def test_mixed_mode_preserves_preflight_reward_semantics() -> None:
    trace = _trace(
        f"handle = await rlm({_prompt()!r}, name='shard-worker')",
        "local_values = [1, 2]\nlocal = sum(local_values)\nlocal",
    )
    task = SubagentAdmissionTask(
        _data(),
        SubagentAdmissionTaskConfig(reward_mode="mixed"),
    )

    asyncio.run(task.score(trace))

    assert trace.rewards["admission_control"].score == 1.0
    assert trace.rewards["delegation_protocol"].score > 0.0
    assert trace.reward > 1.0


def test_causal_reward_requires_the_ordered_prefix(monkeypatch) -> None:
    trace = _trace(
        f"handle = await rlm({_prompt()!r}, name='shard-worker')",
        "local_values = [1, 2]\nlocal = sum(local_values)\nlocal",
    )
    monkeypatch.setattr(
        admission_module,
        "_protocol_behavior",
        lambda *args: {
            "messages_to_parent": 1.0,
            "explicit_messages_to_parent": 1.0,
            "protocol_aligned": 1.0,
            "clean_protocol_aligned": 1.0,
            "roster_calls": 0.0,
            "observation_calls": 0.0,
            "failed_cells": 0.0,
        },
    )
    monkeypatch.setattr(admission_module, "_answer_score", lambda *args: 1.0)

    behavior = _causal_behavior(trace, _data())

    assert behavior == {
        "causal_score": 1.0,
        "causal_spawn_first": 1.0,
        "causal_contract_bound": 1.0,
        "causal_local_work": 1.0,
        "causal_child_reply": 1.0,
        "causal_completed": 1.0,
        "causal_clean_child_reply": 1.0,
        "causal_clean_completed": 1.0,
    }


def test_causal_reward_does_not_credit_later_events_after_broken_prefix(
    monkeypatch,
) -> None:
    trace = _trace(
        f"handle = await rlm({_prompt()!r}, name='shard-worker')",
    )
    monkeypatch.setattr(
        admission_module,
        "_protocol_behavior",
        lambda *args: {
            "messages_to_parent": 1.0,
            "explicit_messages_to_parent": 1.0,
            "protocol_aligned": 1.0,
            "clean_protocol_aligned": 1.0,
            "roster_calls": 0.0,
            "observation_calls": 0.0,
            "failed_cells": 0.0,
        },
    )
    monkeypatch.setattr(admission_module, "_answer_score", lambda *args: 1.0)

    behavior = _causal_behavior(trace, _data())

    assert behavior["causal_score"] == 0.4
    assert behavior["causal_spawn_first"] == 1.0
    assert behavior["causal_contract_bound"] == 1.0
    assert behavior["causal_local_work"] == 0.0
    assert behavior["causal_child_reply"] == 0.0
    assert behavior["causal_completed"] == 0.0


def test_causal_mode_exposes_only_causal_reward(monkeypatch) -> None:
    trace = _trace(
        f"handle = await rlm({_prompt()!r}, name='shard-worker')",
        "local_values = [1, 2]\nlocal = sum(local_values)\nlocal",
    )
    monkeypatch.setattr(
        admission_module,
        "_protocol_behavior",
        lambda *args: {
            "messages_to_parent": 1.0,
            "explicit_messages_to_parent": 1.0,
            "protocol_aligned": 1.0,
            "clean_protocol_aligned": 1.0,
            "roster_calls": 0.0,
            "observation_calls": 0.0,
            "failed_cells": 0.0,
        },
    )
    monkeypatch.setattr(admission_module, "_answer_score", lambda *args: 1.0)
    task = SubagentAdmissionTask(
        _data(),
        SubagentAdmissionTaskConfig(reward_mode="causal"),
    )

    asyncio.run(task.score(trace))

    assert trace.rewards["causal_control"].score == 1.0
    assert trace.rewards["admission_control"].score == 0.0
    assert trace.rewards["protocol_gated_accuracy"].score == 0.0
    assert trace.rewards["delegation_protocol"].score == 0.0
    assert trace.reward == 1.0


def test_clean_causal_reward_requires_tool_discipline(monkeypatch) -> None:
    trace = _trace(
        f"handle = await rlm({_prompt()!r}, name='shard-worker')",
        "local_values = [1, 2]\nlocal = sum(local_values)\nlocal",
    )
    protocol = {
        "messages_to_parent": 1.0,
        "explicit_messages_to_parent": 1.0,
        "protocol_aligned": 1.0,
        "clean_protocol_aligned": 0.0,
        "roster_calls": 0.0,
        "observation_calls": 0.0,
        "failed_cells": 0.0,
        "non_ipython_tool_calls": 1.0,
        "inert_cells": 0.0,
        "duplicate_cells": 0.0,
        "post_parent_send_tool_calls": 0.0,
        "post_fan_in_cells": 0.0,
    }
    monkeypatch.setattr(admission_module, "_protocol_behavior", lambda *args: protocol)
    monkeypatch.setattr(admission_module, "_answer_score", lambda *args: 1.0)

    behavior = _clean_causal_behavior(trace, _data())

    assert behavior == {
        "clean_causal_score": 0.0,
        "clean_causal_trace_admissible": 0.0,
        "clean_causal_raw_spawn_first": 1.0,
        "clean_causal_raw_contract_bound": 1.0,
        "clean_causal_raw_local_work": 1.0,
        "clean_causal_spawn_first": 0.0,
        "clean_causal_contract_bound": 0.0,
        "clean_causal_local_work": 0.0,
        "clean_causal_tool_discipline": 0.0,
        "clean_causal_child_reply": 0.0,
        "clean_causal_completed": 0.0,
    }


def test_clean_causal_reward_credits_only_admissible_prefixes(monkeypatch) -> None:
    trace = _trace(
        f"handle = await rlm({_prompt()!r}, name='shard-worker')",
        "local_values = [1, 2]\nlocal = sum(local_values)\nlocal",
    )
    protocol = {
        "messages_to_parent": 0.0,
        "explicit_messages_to_parent": 0.0,
        "protocol_aligned": 0.0,
        "clean_protocol_aligned": 0.0,
        "roster_calls": 0.0,
        "observation_calls": 0.0,
        "failed_cells": 0.0,
        "non_ipython_tool_calls": 0.0,
        "inert_cells": 0.0,
        "duplicate_cells": 0.0,
        "post_parent_send_tool_calls": 0.0,
        "post_fan_in_cells": 0.0,
    }
    monkeypatch.setattr(admission_module, "_protocol_behavior", lambda *args: protocol)
    monkeypatch.setattr(admission_module, "_answer_score", lambda *args: 0.0)

    behavior = _clean_causal_behavior(trace, _data())

    assert behavior["clean_causal_trace_admissible"] == 1.0
    assert behavior["clean_causal_score"] == 4 / 6
    assert behavior["clean_causal_tool_discipline"] == 1.0
    assert behavior["clean_causal_child_reply"] == 0.0


def test_clean_causal_mode_exposes_clean_prefix_reward(monkeypatch) -> None:
    trace = _trace(
        f"handle = await rlm({_prompt()!r}, name='shard-worker')",
        "local_values = [1, 2]\nlocal = sum(local_values)\nlocal",
    )
    protocol = {
        "messages_to_parent": 1.0,
        "explicit_messages_to_parent": 1.0,
        "protocol_aligned": 1.0,
        "clean_protocol_aligned": 1.0,
        "roster_calls": 0.0,
        "observation_calls": 0.0,
        "failed_cells": 0.0,
        "non_ipython_tool_calls": 0.0,
        "inert_cells": 0.0,
        "duplicate_cells": 0.0,
        "post_parent_send_tool_calls": 0.0,
        "post_fan_in_cells": 0.0,
    }
    monkeypatch.setattr(admission_module, "_protocol_behavior", lambda *args: protocol)
    monkeypatch.setattr(admission_module, "_answer_score", lambda *args: 1.0)
    task = SubagentAdmissionTask(
        _data(),
        SubagentAdmissionTaskConfig(reward_mode="clean_causal"),
    )

    asyncio.run(task.score(trace))

    assert trace.rewards["causal_control"].score == 1.0
    assert trace.rewards["admission_control"].score == 0.0
    assert trace.rewards["protocol_gated_accuracy"].score == 0.0
    assert trace.rewards["delegation_protocol"].score == 0.0
    assert trace.reward == 1.0

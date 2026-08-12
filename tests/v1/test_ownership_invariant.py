import json

import pytest
from ownership_invariant_v1.taskset import (
    HELDOUT_RESOURCE_FAMILIES,
    TRAIN_RESOURCE_FAMILIES,
    OwnershipInvariantConfig,
    OwnershipInvariantData,
    OwnershipInvariantTaskset,
    _first_decision_behavior,
)

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, ToolCall, ToolMessage, UserMessage


def _data(ownership: str = "child") -> OwnershipInvariantData:
    return OwnershipInvariantData(
        idx=0,
        name="test",
        prompt="test",
        ownership=ownership,
        split="admission",
        resource_family="json_sum",
        phrasing_variant=0,
        state_name="request_tag",
        state_value="coord-json-sum",
        expected_child="ledger-worker",
        resource_path="/workspace/data.json",
        resource_content="[1,2]",
        expected_result=3,
        operation="sum JSON integers",
        files={"/workspace/data.json": "[1,2]"},
    )


def _trace(*cells: str, reply: str = "waiting") -> vf.Trace:
    nodes = [MessageNode(parent=None, message=UserMessage(content="decide"), sampled=False)]
    parent = 0
    for index, source in enumerate(cells):
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
                message=ToolMessage(
                    tool_call_id=call_id,
                    content="RLMSpawnHandle(name='ledger-worker')",
                ),
                sampled=False,
            )
        )
        parent = len(nodes) - 1
    nodes.append(MessageNode(parent=parent, message=AssistantMessage(content=reply), sampled=True))
    return vf.Trace(
        id="ownership-invariant-test",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="OwnershipInvariantTask", data=vf.TaskData(idx=0)),
        nodes=nodes,
    )


def _valid_code() -> str:
    prompt = (
        "Read /workspace/data.json, sum JSON integers, then use agent_message.send to send the result to your parent."
    )
    return f"request_tag = 'coord-json-sum'\nchild = await rlm({prompt!r}, name='ledger-worker')"


def test_frozen_splits_are_disjoint_and_balanced() -> None:
    admission = OwnershipInvariantTaskset(OwnershipInvariantConfig(split="admission")).load()
    phrasing = OwnershipInvariantTaskset(OwnershipInvariantConfig(split="heldout_phrasing")).load()
    resources = OwnershipInvariantTaskset(OwnershipInvariantConfig(split="heldout_resource")).load()

    assert len(admission) == len(phrasing) == len(resources) == 8
    assert {task.data.resource_family for task in admission} == set(TRAIN_RESOURCE_FAMILIES)
    assert {task.data.phrasing_variant for task in admission} == {0, 1}
    assert {task.data.phrasing_variant for task in phrasing} == {2, 3}
    assert {task.data.resource_family for task in resources} == set(HELDOUT_RESOURCE_FAMILIES)


def test_family_filter_preserves_instances_and_rejects_cross_split_families() -> None:
    tasks = OwnershipInvariantTaskset(
        OwnershipInvariantConfig(
            families=("json_sum", "text_keyword_count"),
            instances_per_family=2,
        )
    ).load()

    assert len(tasks) == 4
    assert {task.data.resource_family for task in tasks} == {"json_sum", "text_keyword_count"}
    with pytest.raises(ValueError, match="families unavailable in heldout_resource"):
        OwnershipInvariantConfig(split="heldout_resource", families=("json_sum",))


def test_yield_policy_survives_served_task_reconstruction() -> None:
    config = OwnershipInvariantConfig(yield_policy="semantic")
    client_task = OwnershipInvariantTaskset(config).load()[0]

    # EnvServer receives only task data and reconstructs behavior from the
    # serialized task subtree, rather than from the client-side Task object.
    served_task = type(client_task)(
        type(client_task).data_type().model_validate(client_task.data.model_dump()),
        config.task,
    )

    assert config.task.yield_policy == "semantic"
    assert client_task.config.yield_policy == "semantic"
    assert served_task.config.yield_policy == "semantic"


def test_matched_ownership_arms_share_task_semantics() -> None:
    child = OwnershipInvariantTaskset(OwnershipInvariantConfig(split="admission", ownership="child")).load()
    coordinator = OwnershipInvariantTaskset(OwnershipInvariantConfig(split="admission", ownership="coordinator")).load()

    assert [(task.data.resource_family, task.data.resource_path, task.data.expected_result) for task in child] == [
        (task.data.resource_family, task.data.resource_path, task.data.expected_result) for task in coordinator
    ]


def test_coordinator_prompt_names_retained_state_in_answer_contract() -> None:
    task = OwnershipInvariantTaskset(
        OwnershipInvariantConfig(split="admission", ownership="coordinator")
    ).load()[0]

    assert f'"local_state": {json.dumps(task.data.state_value)}' in task.data.prompt
    assert '"result": <computed result>' in task.data.prompt
    assert '{"local_state": value, "result": value}' not in task.data.prompt


def test_guided_coordinator_prompt_teaches_control_without_revealing_result() -> None:
    task = OwnershipInvariantTaskset(
        OwnershipInvariantConfig(
            split="admission",
            ownership="coordinator",
            instruction_level="guided",
        )
    ).load()[0]

    assert f"assign {task.data.state_name}={task.data.state_value!r}" in task.data.prompt
    assert "persistent IPython state" in task.data.prompt
    assert "bare JSON with no Markdown fence" in task.data.prompt
    assert str(task.data.expected_result) not in task.data.prompt
    assert task.data.instruction_level == "guided"


def test_child_owned_strict_success_requires_complete_first_decision() -> None:
    behavior = _first_decision_behavior(_trace(_valid_code()), _data())

    assert behavior["strict_success"] == 1.0
    assert behavior["parent_path_access"] == 0.0
    assert behavior["local_state_leaked"] == 0.0


@pytest.mark.parametrize(
    "state_code",
    [
        "coordinator_state = {'request_tag': 'coord-json-sum'}",
        "coordinator_state = {}\ncoordinator_state['request_tag'] = 'coord-json-sum'",
    ],
)
def test_child_owned_accepts_persistent_mapping_state(state_code: str) -> None:
    prompt = "Read /workspace/data.json, sum JSON integers, then reply to your parent."
    code = f"{state_code}\nchild = await rlm({prompt!r}, name='ledger-worker')"

    behavior = _first_decision_behavior(_trace(code), _data())

    assert behavior["strict_success"] == 1.0
    assert behavior["state_retained"] == 1.0
    assert behavior["state_precedes_spawn"] == 1.0


@pytest.mark.parametrize(
    ("code", "failed_component"),
    [
        (
            "child = await rlm('Read /workspace/data.json', name='ledger-worker')",
            "state_retained",
        ),
        (
            "request_tag = 'coord-json-sum'\nawait rlm('Read /workspace/data.json', name='ledger-worker')",
            "retained_handle",
        ),
        (
            "request_tag = 'coord-json-sum'\nchild = await rlm('Read /workspace/data.json', name='other-worker')",
            "expected_child",
        ),
        (
            "request_tag = 'coord-json-sum'\npath = '/workspace/data.json'\nchild = await rlm(f'Read {path}', name='ledger-worker')",
            "parent_path_access",
        ),
        (
            "request_tag = 'coord-json-sum'\nchild = await rlm('Read /workspace/data.json for coord-json-sum', name='ledger-worker')",
            "local_state_leaked",
        ),
        (
            "request_tag = 'coord-json-sum'\nchild = await rlm('Read /workspace/data.json', name='ledger-worker')\nawait agent_observe.status(child.name)",
            "prohibited_control",
        ),
        (
            "request_tag = 'coord-json-sum'\nchild = await rlm('Read /workspace/data.json', name='ledger-worker')\nprint('waiting')",
            "post_spawn_action",
        ),
    ],
)
def test_child_owned_rejects_partial_atoms(code: str, failed_component: str) -> None:
    behavior = _first_decision_behavior(_trace(code), _data())

    assert behavior["strict_success"] == 0.0
    assert behavior[failed_component] == (
        1.0
        if failed_component in {"parent_path_access", "local_state_leaked", "prohibited_control", "post_spawn_action"}
        else 0.0
    )


def test_child_owned_rejects_a_second_coordinator_cell() -> None:
    behavior = _first_decision_behavior(_trace(_valid_code(), "print('still waiting')"), _data())

    assert behavior["strict_success"] == 0.0
    assert behavior["first_decision_only"] == 0.0
    assert behavior["post_spawn_action"] == 1.0


@pytest.mark.parametrize(
    "tail",
    [
        "child",
        "print(child)",
        "print(f'child={child.name}')",
        "coordinator_state['child'] = child",
        "coordinator_state['child'] = child\ncoordinator_state",
    ],
)
def test_semantic_yield_accepts_only_passive_handle_tail(tail: str) -> None:
    behavior = _first_decision_behavior(
        _trace(f"{_valid_code()}\n{tail}"),
        _data(),
        yield_policy="semantic",
    )

    assert behavior["strict_success"] == 1.0
    assert behavior["post_spawn_statement"] == 1.0
    assert behavior["passive_handle_tail"] == 1.0
    assert behavior["post_spawn_action"] == 0.0


@pytest.mark.parametrize(
    "tail",
    [
        "print('waiting')",
        "result = sum([1, 2])",
        "await agent_observe.status(child.name)",
    ],
)
def test_semantic_yield_rejects_substantive_tail(tail: str) -> None:
    behavior = _first_decision_behavior(
        _trace(f"{_valid_code()}\n{tail}"),
        _data(),
        yield_policy="semantic",
    )

    assert behavior["strict_success"] == 0.0
    assert behavior["post_spawn_action"] == 1.0


def test_semantic_yield_still_rejects_a_second_coordinator_cell() -> None:
    behavior = _first_decision_behavior(
        _trace(_valid_code(), "child"),
        _data(),
        yield_policy="semantic",
    )

    assert behavior["strict_success"] == 0.0
    assert behavior["first_decision_only"] == 0.0
    assert behavior["post_spawn_action"] == 1.0


def test_coordinator_owned_control_requires_direct_correct_work() -> None:
    code = (
        "request_tag = 'coord-json-sum'\n"
        "import json\nfrom pathlib import Path\n"
        "result = sum(json.loads(Path('/workspace/data.json').read_text()))"
    )
    trace = _trace(
        code,
        reply='{"local_state":"coord-json-sum","result":3}',
    )

    behavior = _first_decision_behavior(trace, _data("coordinator"))

    assert behavior["strict_success"] == 1.0
    assert behavior["direct_answer_accuracy"] == 1.0


def test_coordinator_owned_control_rejects_unnecessary_delegation() -> None:
    behavior = _first_decision_behavior(_trace(_valid_code()), _data("coordinator"))

    assert behavior["strict_success"] == 0.0
    assert behavior["one_spawn"] == 1.0

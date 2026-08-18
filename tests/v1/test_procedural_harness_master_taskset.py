import asyncio
import json
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from procedural_harness_master_v1.taskset import (
    COMPLETION_GATE_PATH,
    ProceduralHarnessMasterConfig,
    ProceduralHarnessMasterEnv,
    ProceduralHarnessMasterTaskset,
    _contract_behavior,
    _followup_feedback_diagnostic,
    _record_followup_feedback,
    keep_followup_feedback_response,
)

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, ToolCall, ToolMessage, UserMessage


def _task(family: str, split: str = "train_gen"):
    return ProceduralHarnessMasterTaskset(
        ProceduralHarnessMasterConfig(split=split, count=1, families=(family,))
    ).load()[0]


def _curriculum_task(rung: str, split: str = "train_gen"):
    return ProceduralHarnessMasterTaskset(
        ProceduralHarnessMasterConfig(split=split, count=1, curriculum_rung=rung)
    ).load()[0]


def _cell(nodes, parent: int, code: str, output: str = "") -> int:
    call_id = f"call-{len(nodes)}"
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
    nodes.append(
        MessageNode(
            parent=len(nodes) - 1,
            message=ToolMessage(tool_call_id=call_id, content=output),
            sampled=False,
        )
    )
    return len(nodes) - 1


def _incoming(nodes, parent: int, child: str, body: str) -> int:
    nodes.append(
        MessageNode(
            parent=parent,
            message=UserMessage(
                content=(
                    f"[from child:{child}]\n"
                    "Agent-to-agent message received.\n"
                    "Source: agent_message\n"
                    f"Message id: msg-{len(nodes)}\n\n{body}"
                )
            ),
            sampled=False,
        )
    )
    return len(nodes) - 1


def _child_completion_notice(nodes, parent: int, child: str, body: str) -> int:
    nodes.append(
        MessageNode(
            parent=parent,
            message=UserMessage(
                content=(
                    f"[from child:{child}]\n"
                    "RLM child completed without sending a reply. "
                    f"Last assistant text: {body}"
                )
            ),
            sampled=False,
        )
    )
    return len(nodes) - 1


def _trace(task, actions, reply=None):
    nodes = [
        MessageNode(parent=None, message=UserMessage(content="task"), sampled=False)
    ]
    parent = 0
    for action in actions:
        if action[0] == "cell":
            parent = _cell(
                nodes, parent, action[1], action[2] if len(action) > 2 else ""
            )
        else:
            parent = _incoming(nodes, parent, action[1], action[2])
    answer = json.dumps(task.data.oracle["final_answer"]) if reply is None else reply
    nodes.append(
        MessageNode(
            parent=parent,
            message=AssistantMessage(content=answer),
            sampled=True,
        )
    )
    return vf.Trace(
        id="procedural-test",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        nodes=nodes,
    )


def _spawn_code(task, children=None, extra=()):
    children = task.data.oracle["children"] if children is None else children
    lines = [
        f"{name} = {value!r}"
        for name, value in task.data.oracle.get("coordinator_state", {}).items()
    ]
    for index, child in enumerate(children):
        instruction = f"Read {child['resource_path']} and {child['operation']}; send the result to parent."
        lines.append(
            f"handle_{index} = await rlm({instruction!r}, name={child['name']!r})"
        )
    lines.extend(extra)
    return "\n".join(lines)


def test_taskset_keeps_oracle_out_of_model_visible_fields() -> None:
    task = _task("single")
    visible = json.dumps(
        {
            "prompt": task.data.prompt,
            "system_prompt": task.data.system_prompt,
            "files": task.data.workspace_files,
        }
    )

    assert task.key == task.data.episode_id
    assert "trajectory_contract" not in visible
    assert "final_answer" not in visible
    assert task.data.oracle["trajectory_contract"]


@pytest.mark.asyncio
async def test_setup_materializes_only_public_files() -> None:
    class Runtime:
        def __init__(self):
            self.writes = {}

        async def run(self, argv, env):
            return SimpleNamespace(exit_code=0, stderr="")

        async def write(self, path, data):
            self.writes[path] = data

    task = _task("mixed")
    runtime = Runtime()
    await task.setup(None, runtime)

    assert {
        path: contents.encode() for path, contents in task.data.workspace_files.items()
    }.items() <= runtime.writes.items()
    assert COMPLETION_GATE_PATH in runtime.writes
    gate = runtime.writes[COMPLETION_GATE_PATH].decode()
    for child in task.data.oracle["children"]:
        assert child["name"] in gate
    assert repr(task.data.oracle["final_answer"]) not in gate


@pytest.mark.asyncio
async def test_verify_completion_gate_accepts_oracle_types_without_oracle_values(
    tmp_path,
) -> None:
    class Runtime:
        def __init__(self):
            self.writes = {}

        async def run(self, argv, env):
            return SimpleNamespace(exit_code=0, stderr="")

        async def write(self, path, data):
            self.writes[path] = data

    task = _task("verify")
    runtime = Runtime()
    await task.setup(None, runtime)
    gate_source = runtime.writes[COMPLETION_GATE_PATH].decode()
    final_answer = task.data.oracle["final_answer"]
    assert "'child': 'int'" in gate_source
    assert "'verified': 'bool'" in gate_source
    assert "'result': 'int'" in gate_source
    assert json.dumps(final_answer) not in gate_source

    agent_dir = tmp_path / "agent"
    sessions = agent_dir / "sessions"
    sessions.mkdir(parents=True)
    child_name = task.data.oracle["children"][0]["name"]
    entries = [
        {"type": "session", "rlmDepth": 0},
        {
            "type": "custom_message",
            "customType": "agent_message",
            "content": f"[from child:{child_name}]\n\nresult",
            "details": {
                "id": "agentmsg_verify",
                "from": {"sessionName": child_name},
                "fromRelationship": "child",
            },
        },
        {
            "type": "message",
            "message": {"role": "assistant", "content": json.dumps(final_answer)},
        },
    ]
    (sessions / "root.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n"
    )
    gate = tmp_path / "completion_gate.py"
    gate.write_text(gate_source)
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, str(gate)],
        env={
            **os.environ,
            "PRIME_AGENT_CODING_AGENT_DIR": str(agent_dir),
            "VF_PRIME_AGENT_CHILD_EVIDENCE_GRACE_SECONDS": "0",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_direct_complete_trajectory_passes_conjunctive_gate() -> None:
    task = _task("direct")
    path = next(iter(task.data.workspace_files))
    trace = _trace(
        task, [("cell", f"from pathlib import Path\ntext = Path({path!r}).read_text()")]
    )

    behavior = _contract_behavior(trace, task.data)

    assert behavior["harness_score"] == 1.0
    assert behavior["all_required_atoms"] == 1.0


def test_answer_correct_shortcut_fails_required_trajectory_gate() -> None:
    task = _task("direct")
    behavior = _contract_behavior(_trace(task, []), task.data)

    assert behavior["final_answer_exact"] == 1.0
    assert behavior["all_required_atoms"] == 0.0
    assert behavior["harness_score"] == 0.0
    assert behavior["bootstrap_progress"] == 0.0


def test_single_complete_trajectory_passes_and_parent_resource_read_fails() -> None:
    task = _task("single")
    child = task.data.oracle["children"][0]
    spawn = _spawn_code(task)
    actions = [
        ("cell", spawn, f"RLMSpawnHandle(name='{child['name']}')"),
        ("incoming", child["name"], str(child["expected_result"])),
    ]
    clean = _contract_behavior(_trace(task, actions), task.data)
    assert clean["harness_score"] == 1.0

    violating = [
        ("cell", spawn, f"RLMSpawnHandle(name='{child['name']}')"),
        ("cell", f"open({child['resource_path']!r}).read()"),
        ("incoming", child["name"], str(child["expected_result"])),
    ]
    behavior = _contract_behavior(_trace(task, violating), task.data)
    assert behavior["final_answer_exact"] == 1.0
    assert behavior["no_forbidden_atoms"] == 0.0
    assert behavior["harness_score"] == 0.0


def test_environment_variable_counts_as_retained_coordinator_state() -> None:
    task = _task("followup")
    child = task.data.oracle["children"][0]
    state_name, state_value = next(iter(task.data.oracle["coordinator_state"].items()))
    spawn = _spawn_code(task, extra=())
    spawn = spawn.replace(f"{state_name} = {state_value!r}", "")
    actions = [
        (
            "cell",
            f"import os\nos.environ[{state_name.upper()!r}] = {str(state_value)!r}\n{spawn}",
            f"RLMSpawnHandle(name='{child['name']}')",
        ),
        ("incoming", child["name"], "need multiplier"),
        (
            "cell",
            "await agent_message.send(str(os.environ['TICKET']), receiver_role='child')",
            "Agent message sent: agentmsg_1",
        ),
        ("incoming", child["name"], json.dumps(task.data.oracle["final_answer"])),
    ]

    behavior = _contract_behavior(_trace(task, actions), task.data)

    assert behavior["harness_score"] == 1.0, behavior
    assert behavior["all_required_atoms"] == 1.0
    assert behavior["ordering_satisfied"] == 1.0
    assert behavior["cardinality_exact"] == 1.0


def test_natural_followup_request_and_silent_awaited_send_pass() -> None:
    task = _task("followup")
    child = task.data.oracle["children"][0]
    actions = [
        ("cell", _spawn_code(task), f"RLMSpawnHandle(name='{child['name']}')"),
        (
            "incoming",
            child["name"],
            "ERROR-level log line count: 6. Please provide the multiplier.",
        ),
        (
            "cell",
            (
                "await agent_message.send('The multiplier is 4.', "
                f"receiver_role='child', receiver_name='{child['name']}')\n"
                "print('Sent multiplier: 4')"
            ),
            "Sent multiplier: 4",
        ),
        ("incoming", child["name"], json.dumps(task.data.oracle["final_answer"])),
    ]

    behavior = _contract_behavior(_trace(task, actions), task.data)

    assert behavior["harness_score"] == 1.0, behavior
    assert behavior["all_required_atoms"] == 1.0
    assert behavior["ordering_satisfied"] == 1.0
    assert behavior["cardinality_exact"] == 1.0


def test_atomic_followup_feedback_targets_response_after_child_request() -> None:
    task = _curriculum_task("atomic_followup")
    child = task.data.oracle["children"][0]
    state_name, state_value = next(iter(task.data.oracle["coordinator_state"].items()))
    actions = [
        ("cell", _spawn_code(task), f"RLMSpawnHandle(name='{child['name']}')"),
        ("incoming", child["name"], "Please provide the multiplier."),
        (
            "cell",
            (
                f"await agent_message.send(message={state_name}, receiver_role='child', "
                f"receiver_name={child['name']!r})"
            ),
            "TypeError: message must be str, got int",
        ),
    ]
    trace = _trace(task, actions)
    for index, node in enumerate(trace.nodes):
        node.token_ids = [index * 10 + 1, index * 10 + 2]
        node.mask = [node.sampled, node.sampled]

    diagnostic = _followup_feedback_diagnostic(trace, task.data)

    assert diagnostic is not None
    assert diagnostic.child_name == child["name"]
    assert diagnostic.turn_index == 1
    assert isinstance(
        trace.nodes[diagnostic.target_node_index].message, AssistantMessage
    )
    assert (
        "agent_message.send"
        in trace.nodes[diagnostic.target_node_index].message.tool_calls[0].arguments
    )
    assert _record_followup_feedback(trace, task.data)
    contract = trace.info["feedback_contract"]
    assert contract["answer_free"] is True
    assert contract["turn_index"] == 1
    assert contract["target_node_index"] == diagnostic.target_node_index
    assert str(state_value) not in trace.info["feedback"]

    masks = keep_followup_feedback_response(trace)
    selected = [
        id(node)
        for branch, mask in zip(trace.branches, masks, strict=True)
        for node, keep in _nodes_with_mask(branch.nodes, mask)
        if any(keep)
    ]
    assert selected == [id(trace.nodes[diagnostic.target_node_index])]


def _nodes_with_mask(nodes, mask):
    offset = 0
    for node in nodes:
        end = offset + len(node.token_ids)
        yield node, mask[offset:end]
        offset = end
    assert offset == len(mask)


def test_atomic_followup_feedback_is_absent_without_request_or_on_success() -> None:
    task = _curriculum_task("atomic_followup")
    child = task.data.oracle["children"][0]
    spawn = ("cell", _spawn_code(task), f"RLMSpawnHandle(name='{child['name']}')")
    assert _followup_feedback_diagnostic(_trace(task, [spawn]), task.data) is None

    state_name = next(iter(task.data.oracle["coordinator_state"]))
    successful = _trace(
        task,
        [
            spawn,
            ("incoming", child["name"], "need multiplier"),
            (
                "cell",
                (
                    f"await agent_message.send(str({state_name}), receiver_role='child', "
                    f"receiver_name={child['name']!r})"
                ),
                "Agent message sent: agentmsg_1",
            ),
            (
                "incoming",
                child["name"],
                str(task.data.oracle["final_answer"]["result"]),
            ),
        ],
    )
    assert _contract_behavior(successful, task.data)["harness_score"] == 1.0
    assert _followup_feedback_diagnostic(successful, task.data) is None


def test_followup_feedback_recording_is_opt_in() -> None:
    assert ProceduralHarnessMasterConfig().record_causal_feedback is False


class _FeedbackAgent:
    def __init__(self, trace: vf.Trace) -> None:
        self.trace = trace

    @asynccontextmanager
    async def interaction(self, task):
        async def turn():
            return SimpleNamespace(terminated=False)

        yield SimpleNamespace(trace=self.trace, turn=turn)


def test_environment_records_followup_feedback_before_trace_close() -> None:
    task = _curriculum_task("atomic_followup")
    child = task.data.oracle["children"][0]
    trace = _trace(
        task,
        [
            ("cell", _spawn_code(task), f"RLMSpawnHandle(name='{child['name']}')"),
            ("incoming", child["name"], "Please provide the multiplier."),
            ("cell", "print('still waiting')", "still waiting"),
        ],
    )
    env = object.__new__(ProceduralHarnessMasterEnv)
    env.taskset = SimpleNamespace(
        config=ProceduralHarnessMasterConfig(record_causal_feedback=True)
    )

    asyncio.run(env.run(task, SimpleNamespace(agent=_FeedbackAgent(trace))))

    assert trace.info["feedback_contract"]["code"] == "reply_to_child_request"
    assert trace.info["feedback"] == trace.info["feedback_contract"]["message"]


def test_child_completion_notice_does_not_satisfy_explicit_message_contract() -> None:
    task = _task("single")
    child = task.data.oracle["children"][0]
    nodes = [
        MessageNode(parent=None, message=UserMessage(content="task"), sampled=False)
    ]
    parent = _cell(
        nodes, 0, _spawn_code(task), f"RLMSpawnHandle(name='{child['name']}')"
    )
    parent = _child_completion_notice(
        nodes, parent, child["name"], f"The result is {child['expected_result']}"
    )
    nodes.append(
        MessageNode(
            parent=parent,
            message=AssistantMessage(
                content=json.dumps(task.data.oracle["final_answer"])
            ),
            sampled=True,
        )
    )
    trace = vf.Trace(
        id="procedural-completion-notice-test",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        nodes=nodes,
    )
    behavior = _contract_behavior(trace, task.data)
    assert behavior["final_answer_exact"] == 1.0
    assert behavior["all_required_atoms"] == 0.0
    assert behavior["cardinality_exact"] == 0.0
    assert behavior["harness_score"] == 0.0


@pytest.mark.parametrize(
    "rung",
    [
        "atomic_state",
        "atomic_send",
        "atomic_child_request",
        "atomic_followup",
        "atomic_parallel",
    ],
)
def test_atomic_curriculum_rungs_require_complete_real_message_trajectories(
    rung: str,
) -> None:
    task = _curriculum_task(rung)
    children = task.data.oracle["children"]
    if rung == "atomic_state":
        state_name, state_value = next(
            iter(task.data.oracle["coordinator_state"].items())
        )
        expected_result = task.data.oracle["final_answer"]["result"]
        actions = [
            ("cell", f"{state_name} = {state_value}"),
            (
                "cell",
                f"result = {state_name} + {expected_result - state_value}\nprint(result)",
            ),
        ]
        behavior = _contract_behavior(_trace(task, actions), task.data)
        assert behavior["harness_score"] == 1.0, behavior
        assert task.data.workspace_files == {}
        assert task.data.generation_metadata["curriculum_rung"] == rung
        return

    state_lines = [
        f"{name} = {value!r}"
        for name, value in task.data.oracle.get("coordinator_state", {}).items()
    ]
    spawn_lines = [
        f"handle_{index} = await rlm('execute the assigned task and send to parent', name={child['name']!r})"
        for index, child in enumerate(children)
    ]
    actions = [
        (
            "cell",
            "\n".join(state_lines + spawn_lines),
            "\n".join(f"RLMSpawnHandle(name='{child['name']}')" for child in children),
        )
    ]
    if rung == "atomic_child_request":
        child = children[0]
        actions.append(("incoming", child["name"], "need multiplier"))
    elif rung == "atomic_followup":
        child = children[0]
        actions.extend(
            [
                ("incoming", child["name"], "need multiplier"),
                (
                    "cell",
                    f"await agent_message.send(str(multiplier), receiver_role='child', receiver_name={child['name']!r})",
                    "Agent message sent: agentmsg-followup",
                ),
                ("incoming", child["name"], str(child["expected_result"])),
            ]
        )
    else:
        for child in reversed(children):
            actions.append(("incoming", child["name"], str(child["expected_result"])))
    behavior = _contract_behavior(_trace(task, actions), task.data)
    assert behavior["harness_score"] == 1.0, behavior
    assert task.data.workspace_files == {}
    assert task.data.generation_metadata["curriculum_rung"] == rung


def test_atomic_state_rejects_assignment_and_reuse_in_one_ipython_call() -> None:
    task = _curriculum_task("atomic_state")
    state_name, state_value = next(iter(task.data.oracle["coordinator_state"].items()))
    expected_result = task.data.oracle["final_answer"]["result"]
    actions = [
        (
            "cell",
            f"{state_name} = {state_value}\nresult = {state_name} + {expected_result - state_value}\nprint(result)",
        )
    ]
    behavior = _contract_behavior(_trace(task, actions), task.data)
    assert behavior["all_required_atoms"] == 0.0
    assert behavior["harness_score"] == 0.0


def test_atomic_child_request_requires_a_request_without_parent_reply() -> None:
    task = _curriculum_task("atomic_child_request")
    child = task.data.oracle["children"][0]
    state_name, state_value = next(iter(task.data.oracle["coordinator_state"].items()))
    spawn = (
        f"handle = await rlm('send need multiplier to parent', name={child['name']!r})"
    )

    result_message = _trace(
        task,
        [
            ("cell", f"{state_name} = {state_value}\n{spawn}"),
            ("incoming", child["name"], str(state_value)),
        ],
    )
    assert _contract_behavior(result_message, task.data)["harness_score"] == 0.0

    parent_reply = _trace(
        task,
        [
            ("cell", f"{state_name} = {state_value}\n{spawn}"),
            ("incoming", child["name"], "need multiplier"),
            (
                "cell",
                (
                    f"await agent_message.send(str({state_name}), receiver_role='child', "
                    f"receiver_name={child['name']!r})"
                ),
                "Agent message sent: agentmsg-unwanted",
            ),
        ],
    )
    behavior = _contract_behavior(parent_reply, task.data)
    assert behavior["no_forbidden_atoms"] == 0.0
    assert behavior["harness_score"] == 0.0


def test_atomic_parallel_requires_retaining_both_child_handles() -> None:
    task = _curriculum_task("atomic_parallel")
    first, second = task.data.oracle["children"]
    actions = [
        (
            "cell",
            f"handle = await rlm('send result', name={first['name']!r})\nawait rlm('send result', name={second['name']!r})",
            f"RLMSpawnHandle(name='{first['name']}')\nRLMSpawnHandle(name='{second['name']}')",
        ),
        ("incoming", second["name"], str(second["expected_result"])),
        ("incoming", first["name"], str(first["expected_result"])),
    ]
    behavior = _contract_behavior(_trace(task, actions), task.data)
    assert behavior["all_required_atoms"] == 0.0
    assert behavior["harness_score"] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "poll_call",
    [
        "await agent_observe.run()",
        "await rlm.list_subagents()",
        "await agent_message.list_agents()",
        "await agent_message.recv()",
        "await agent_message.list_messages()",
        "import asyncio\nawait asyncio.sleep(1)",
        "import time\ntime.sleep(1)",
        "from asyncio import sleep\nawait sleep(1)",
        "from agent_message import list_agents\nawait list_agents()",
        "from agent_message import list_agents as roster\nawait roster()",
        "import agent_message as messaging\nawait messaging.list_agents()",
        "roster = agent_message.list_agents\nawait roster()",
    ],
)
async def test_bootstrap_reward_is_bounded_and_forbidden_actions_get_no_shaping(
    poll_call: str,
) -> None:
    task = _task("single")
    task.config.reward_mode = "bootstrap"
    child = task.data.oracle["children"][0]
    clean_actions = [
        ("cell", _spawn_code(task), f"RLMSpawnHandle(name='{child['name']}')"),
        ("incoming", child["name"], str(child["expected_result"])),
    ]
    clean_trace = _trace(task, clean_actions)
    assert await task.harness_score(clean_trace) == pytest.approx(1.1)

    violating_actions = [
        ("cell", _spawn_code(task), f"RLMSpawnHandle(name='{child['name']}')"),
        ("cell", poll_call),
        ("incoming", child["name"], str(child["expected_result"])),
    ]
    violating_trace = _trace(task, violating_actions)
    behavior = _contract_behavior(violating_trace, task.data)
    assert behavior["no_forbidden_atoms"] == 0.0
    assert behavior["bootstrap_progress"] == 0.0
    assert await task.harness_score(violating_trace) == 0.0


@pytest.mark.asyncio
async def test_event_control_reward_trains_clean_partial_followup_progress() -> None:
    task = _curriculum_task("atomic_followup")
    task.config.reward_mode = "event_control"
    child = task.data.oracle["children"][0]
    state_name = next(iter(task.data.oracle["coordinator_state"]))
    prefix = [
        ("cell", _spawn_code(task), f"RLMSpawnHandle(name='{child['name']}')"),
        ("incoming", child["name"], "need multiplier"),
        (
            "cell",
            (
                f"await agent_message.send(str({state_name}), receiver_role='child', "
                f"receiver_name={child['name']!r})"
            ),
            "Agent message sent: agentmsg_1",
        ),
    ]

    partial_trace = _trace(task, prefix, reply="{}")
    partial = _contract_behavior(partial_trace, task.data)
    assert partial["harness_score"] == 0.0
    assert 0.0 < partial["event_control_progress"] < 1.0
    assert await task.harness_score(partial_trace) == pytest.approx(
        partial["event_control_progress"]
    )

    successful_trace = _trace(
        task,
        [*prefix, ("incoming", child["name"], str(child["expected_result"]))],
    )
    successful = _contract_behavior(successful_trace, task.data)
    assert successful["event_control_progress"] == 1.0
    assert await task.harness_score(successful_trace) == pytest.approx(2.0)

    violating_trace = _trace(
        task,
        [
            *prefix,
            ("cell", "import asyncio\nawait asyncio.sleep(1)"),
            ("incoming", child["name"], str(child["expected_result"])),
        ],
    )
    violating = _contract_behavior(violating_trace, task.data)
    assert violating["event_control_progress"] == 0.0
    assert await task.harness_score(violating_trace) == 0.0


@pytest.mark.parametrize(
    "poll_call",
    [
        "import asyncio\nawait asyncio.sleep(1)",
        "import time\ntime.sleep(1)",
        "from asyncio import sleep\nawait sleep(1)",
    ],
)
def test_sleep_after_followup_is_polling_not_yield(poll_call: str) -> None:
    task = _task("followup")
    child = task.data.oracle["children"][0]
    actions = [
        ("cell", _spawn_code(task), f"RLMSpawnHandle(name='{child['name']}')"),
        ("incoming", child["name"], "need multiplier"),
        (
            "cell",
            "await agent_message.send(str(multiplier), receiver_role='child')",
            "Agent message sent: agentmsg_1",
        ),
        ("cell", poll_call),
        ("incoming", child["name"], json.dumps(task.data.oracle["final_answer"])),
    ]

    behavior = _contract_behavior(_trace(task, actions), task.data)

    assert behavior["no_forbidden_atoms"] == 0.0
    assert behavior.get("yield_after_followup", 0.0) == 0.0
    assert behavior["bootstrap_progress"] == 0.0
    assert behavior["harness_score"] == 0.0


@pytest.mark.parametrize(
    ("import_cell", "poll_cell"),
    [
        ("from agent_message import list_agents", "await list_agents()"),
        ("from agent_message import list_agents as roster", "await roster()"),
        ("import agent_message as messaging", "await messaging.list_agents()"),
        ("import asyncio as aio", "await aio.sleep(1)"),
        ("from asyncio import sleep as wait", "await wait(1)"),
    ],
)
def test_poll_aliases_persist_across_ipython_cells(
    import_cell: str,
    poll_cell: str,
) -> None:
    task = _task("single")
    child = task.data.oracle["children"][0]
    actions = [
        ("cell", _spawn_code(task), f"RLMSpawnHandle(name='{child['name']}')"),
        ("cell", import_cell),
        ("cell", poll_cell),
        ("incoming", child["name"], str(child["expected_result"])),
    ]

    behavior = _contract_behavior(_trace(task, actions), task.data)

    assert behavior["no_forbidden_atoms"] == 0.0
    assert behavior["bootstrap_progress"] == 0.0
    assert behavior["harness_score"] == 0.0


@pytest.mark.parametrize("family", ["parallel", "mixed", "followup", "verify"])
def test_composed_training_families_pass_complete_synthetic_traces(family: str) -> None:
    task = _task(family)
    children = task.data.oracle["children"]
    extra = []
    if family == "mixed":
        local = next(
            path
            for path, item in task.data.oracle["resource_ownership"].items()
            if item["owner"] == "coordinator"
        )
        extra.append(f"local_text = open({local!r}).read()")
    if family == "verify":
        manifest = next(
            path
            for path, item in task.data.oracle["resource_ownership"].items()
            if item["owner"] == "coordinator"
        )
        extra.append(f"manifest = open({manifest!r}).read()")
    actions = [("cell", _spawn_code(task, extra=extra), "RLMSpawnHandle(name='ok')")]
    if family == "followup":
        child = children[0]
        actions.extend(
            [
                ("incoming", child["name"], "need multiplier"),
                (
                    "cell",
                    "await agent_message.send(str(multiplier), receiver_role='child')",
                    "Agent message sent: agentmsg_1",
                ),
                (
                    "incoming",
                    child["name"],
                    json.dumps(task.data.oracle["final_answer"]),
                ),
            ]
        )
    else:
        for child in reversed(children):
            body = str(child["expected_result"])
            if family == "verify":
                manifest = next(
                    path for path in task.data.workspace_files if "verification" in path
                )
                digest = json.loads(task.data.workspace_files[manifest])[
                    "expected_digest"
                ]
                body = json.dumps(
                    {"result": child["expected_result"], "digest": digest}
                )
            actions.append(("incoming", child["name"], body))

    behavior = _contract_behavior(_trace(task, actions), task.data)

    assert behavior["harness_score"] == 1.0, behavior


def test_reclaim_requires_explicit_failure_before_parent_access() -> None:
    task = _task("reclaim", "ood_gen")
    child = task.data.oracle["children"][0]
    spawn = _spawn_code(task)
    clean = [
        ("cell", spawn, f"RLMSpawnHandle(name='{child['name']}')"),
        ("incoming", child["name"], "RESOURCE_UNAVAILABLE"),
        ("cell", f"payload = open({child['resource_path']!r}).read()"),
    ]
    assert _contract_behavior(_trace(task, clean), task.data)["harness_score"] == 1.0

    premature = [
        ("cell", spawn, f"RLMSpawnHandle(name='{child['name']}')"),
        ("cell", f"payload = open({child['resource_path']!r}).read()"),
        ("incoming", child["name"], "RESOURCE_UNAVAILABLE"),
    ]
    behavior = _contract_behavior(_trace(task, premature), task.data)
    assert behavior["no_forbidden_atoms"] == 0.0
    assert behavior["harness_score"] == 0.0

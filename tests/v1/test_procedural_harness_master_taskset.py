import asyncio
import json
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from procedural_harness_master_v1.causal_context_boundary import (
    BOUNDARY_MARKER,
    _rewrite_for_boundary,
    install_causal_context_boundary,
)
from procedural_harness_master_v1.taskset import (
    CHILD_ACTION_SCAFFOLD_HEADER,
    COMPLETION_GATE_PATH,
    PRIVATE_EVIDENCE_HEADER,
    PRIVILEGED_BOOTSTRAP_HEADER,
    PRIVILEGED_HINT_HEADER,
    RECURSIVE_COORDINATOR_HEADER,
    ProceduralHarnessMasterConfig,
    ProceduralHarnessMasterTaskConfig,
    ProceduralHarnessMasterEnv,
    ProceduralHarnessMasterTask,
    ProceduralHarnessMasterTaskset,
    _contract_behavior,
    _followup_feedback_diagnostic,
    _natural_yield_feedback_diagnostic,
    _record_followup_feedback,
    _record_natural_yield_feedback,
    _resolve_alias,
    keep_atomic_child_request_coordinator_actions,
    keep_followup_feedback_response,
    keep_natural_yield_feedback_response,
)
from verifiers.v1.dialects.chat import ChatDialect, message_to_wire
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, ToolCall, ToolMessage, UserMessage


@pytest.mark.parametrize(
    ("aliases", "name", "resolved"),
    [
        ({"msg": "agent_message"}, "msg.send", "agent_message.send"),
        (
            {"agent_message": "agent_message.agent_message"},
            "agent_message.send",
            "agent_message.agent_message.send",
        ),
        ({"left": "right", "right": "left"}, "left.send", "left.send"),
    ],
)
def test_alias_resolution_is_bounded_by_distinct_heads(
    aliases: dict[str, str], name: str, resolved: str
) -> None:
    assert _resolve_alias(name, aliases) == resolved


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
                    "Agent-to-agent message received.\n"
                    "Source: agent_message\n"
                    f"Message id: msg-{len(nodes)}\n\n"
                    f"RLM child {child} (sub-test) completed without sending a reply. "
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
        elif action[0] == "assistant":
            nodes.append(
                MessageNode(
                    parent=parent,
                    message=AssistantMessage(content=action[1]),
                    sampled=True,
                )
            )
            parent = len(nodes) - 1
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


def test_privileged_context_is_opt_in_and_requires_the_selected_task(tmp_path) -> None:
    plain = _curriculum_task("natural_n1a_local")
    hint_path = tmp_path / "hints.json"
    hint_path.write_text(
        json.dumps(
            {
                "schema_version": "qwen35-2b-spade-rung0-hints/v1",
                "status": "complete",
                "hints": {plain.key: "Delegate only the reviewer-owned evidence."},
            }
        )
    )
    hinted = ProceduralHarnessMasterTaskset(
        ProceduralHarnessMasterConfig(
            count=1,
            curriculum_rung="natural_n1a_local",
            privileged_hint_path=str(hint_path),
        )
    ).load()[0]

    assert PRIVILEGED_HINT_HEADER not in plain.data.prompt
    assert hinted.data.prompt == (
        f"{plain.data.prompt}\n\n{PRIVILEGED_HINT_HEADER}\n"
        "Delegate only the reviewer-owned evidence."
    )
    assert hinted.data.oracle == plain.data.oracle
    assert hinted.data.workspace_files == plain.data.workspace_files

    hint_path.write_text(
        json.dumps(
            {
                "schema_version": "qwen35-2b-spade-rung0-hints/v1",
                "status": "complete",
                "hints": {"different-task": "Do the task."},
            }
        )
    )
    with pytest.raises(ValueError, match="lacks selected task"):
        ProceduralHarnessMasterTaskset(
            ProceduralHarnessMasterConfig(
                count=1,
                curriculum_rung="natural_n1a_local",
                privileged_hint_path=str(hint_path),
            )
        ).load()

    bootstrap_path = tmp_path / "bootstrap.json"
    bootstrap_path.write_text(
        json.dumps(
            {
                "schema_version": "qwen35-2b-environment-bootstrap-context/v1",
                "status": "complete",
                "split": "train_gen",
                "contexts": {plain.key: "Execute the supplied first action exactly."},
            }
        )
    )
    scaffolded = ProceduralHarnessMasterTaskset(
        ProceduralHarnessMasterConfig(
            count=1,
            curriculum_rung="natural_n1a_local",
            privileged_bootstrap_path=str(bootstrap_path),
        )
    ).load()[0]

    assert PRIVILEGED_BOOTSTRAP_HEADER not in plain.data.prompt
    assert scaffolded.data.prompt == (
        f"{plain.data.prompt}\n\n{PRIVILEGED_BOOTSTRAP_HEADER}\n"
        "Execute the supplied first action exactly."
    )
    assert scaffolded.data.oracle == plain.data.oracle
    assert scaffolded.data.workspace_files == plain.data.workspace_files

    with pytest.raises(ValueError, match="mutually exclusive"):
        ProceduralHarnessMasterTaskset(
            ProceduralHarnessMasterConfig(
                count=1,
                curriculum_rung="natural_n1a_local",
                privileged_hint_path=str(hint_path),
                privileged_bootstrap_path=str(bootstrap_path),
            )
        ).load()

    with pytest.raises(ValueError, match="restricted to train_gen"):
        ProceduralHarnessMasterTaskset(
            ProceduralHarnessMasterConfig(
                split="valid_gen",
                count=1,
                curriculum_rung="natural_n1a_local",
                privileged_bootstrap_path=str(bootstrap_path),
            )
        ).load()


def test_natural_private_evidence_is_injected_only_into_child_context() -> None:
    task = _curriculum_task("natural_n1")
    private = task.data.oracle["private_resources"]
    child_request = vf.Request(
        messages=[UserMessage(content="Review your assigned private evidence.")]
    )

    rewritten = task.inject_natural_private_evidence(child_request)

    assert rewritten is not None
    child_prompt = rewritten.messages[-1]
    assert isinstance(child_prompt, UserMessage)
    text = str(child_prompt.content)
    assert PRIVATE_EVIDENCE_HEADER in text
    for label, contents in private.items():
        assert label in text
        assert contents in text

    root_request = vf.Request(messages=[UserMessage(content=task.data.prompt)])
    assert task.inject_natural_private_evidence(root_request) is None
    assert task.inject_natural_private_evidence(rewritten) is None


def test_recursive_coordinator_context_is_explicit_and_excludes_worker_marker() -> None:
    task = ProceduralHarnessMasterTaskset(
        ProceduralHarnessMasterConfig(
            count=1,
            curriculum_rung="natural_n1a",
            private_payload_mode="raw_resource",
            task=ProceduralHarnessMasterTaskConfig(
                reward_mode="child_action",
                delegated_session_role="coordinator",
            ),
        )
    ).load()[0]
    request = vf.Request(messages=[UserMessage(content="Handle the bounded subproblem.")])

    rewritten = task.inject_natural_private_evidence(request)

    assert rewritten is not None
    text = str(rewritten.messages[-1].content)
    assert RECURSIVE_COORDINATOR_HEADER in text
    assert PRIVATE_EVIDENCE_HEADER not in text
    assert "session_role=coordinator" in text
    assert "is_root=false" in text
    assert "can_delegate=false" in text
    assert "return_contract=exactly_one_parent_report" in text
    assert task.data.generation_metadata["delegated_session_role"] == "coordinator"


def test_natural_finding_card_mode_reaches_task_and_child_context() -> None:
    task = ProceduralHarnessMasterTaskset(
        ProceduralHarnessMasterConfig(
            count=1,
            curriculum_rung="natural_n1",
            private_payload_mode="finding_card",
        )
    ).load()[0]
    request = vf.Request(messages=[UserMessage(content="Review the private evidence.")])

    rewritten = task.inject_natural_private_evidence(request)

    assert task.data.generation_metadata["private_payload_mode"] == "finding_card"
    assert rewritten is not None
    text = str(rewritten.messages[-1].content)
    assert "not a runtime file path" in text
    assert next(iter(task.data.oracle["private_resources"].values())) in text


def test_exact_child_action_scaffold_is_opt_in_and_child_only() -> None:
    plain = ProceduralHarnessMasterTaskset(
        ProceduralHarnessMasterConfig(
            count=1,
            curriculum_rung="natural_n1a",
            private_payload_mode="finding_card",
        )
    ).load()[0]
    scaffolded = ProceduralHarnessMasterTaskset(
        ProceduralHarnessMasterConfig(
            count=1,
            curriculum_rung="natural_n1a",
            private_payload_mode="finding_card",
            task=ProceduralHarnessMasterTaskConfig(leak_child_exact_action=True),
        )
    ).load()[0]
    child_request = vf.Request(messages=[UserMessage(content="Review the private evidence.")])

    plain_rewrite = plain.inject_natural_private_evidence(child_request)
    scaffolded_rewrite = scaffolded.inject_natural_private_evidence(child_request)

    assert plain_rewrite is not None and scaffolded_rewrite is not None
    plain_text = str(plain_rewrite.messages[-1].content)
    scaffolded_text = str(scaffolded_rewrite.messages[-1].content)
    expected = str(scaffolded.data.oracle["children"][0]["expected_result"])
    exact_call = f"await agent_message.send({expected!r}, receiver_role='parent')"
    assert CHILD_ACTION_SCAFFOLD_HEADER not in plain_text
    assert CHILD_ACTION_SCAFFOLD_HEADER in scaffolded_text
    assert exact_call in scaffolded_text
    assert expected not in scaffolded.data.prompt
    root_request = vf.Request(messages=[UserMessage(content=scaffolded.data.prompt)])
    assert scaffolded.inject_natural_private_evidence(root_request) is None


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


def test_causal_n1_contracts_score_their_separate_capabilities() -> None:
    for rung in ("natural_n1a", "natural_n1a_local"):
        task = _curriculum_task(rung)
        child = task.data.oracle["children"][0]
        actions = [
            (
                "cell",
                (
                    "child_handle = await rlm("
                    f"'Review {child['resource_path']} and report to parent', "
                    f"name={child['name']!r})"
                ),
                f"RLMSpawnHandle(name='{child['name']}')",
            )
        ]
        for path, ownership in task.data.oracle["resource_ownership"].items():
            if ownership["owner"] == "coordinator":
                actions.append(("cell", f"open({path!r}).read()", "local evidence"))
        actions.append(("incoming", child["name"], str(child["expected_result"])))

        behavior = _contract_behavior(_trace(task, actions), task.data)

        assert behavior["harness_score"] == 1.0, behavior

    task = _curriculum_task("natural_n1b")
    child = task.data.oracle["children"][0]
    lease = task.data.oracle["persistence_lease"]
    trace = _trace(
        task,
        [
            (
                "cell",
                (
                    "import json\n"
                    f"captured = json.loads(open({lease['path']!r}).read())"
                    f"[{lease['key']!r}]"
                ),
                "",
            ),
            (
                "cell",
                (
                    "child_handle = await rlm("
                    f"'Review {child['resource_path']} and report to parent', "
                    f"name={child['name']!r})"
                ),
                f"RLMSpawnHandle(name='{child['name']}')",
            ),
            ("incoming", child["name"], str(child["expected_result"])),
            ("cell", f"combined = captured + {child['expected_result']}", "combined"),
        ],
    )
    trace.info["persistence_lease"] = {
        "schema_version": "procedural-harness-master-v1/persistence-lease/v1",
        "closed": True,
        "path": lease["path"],
    }
    trace.info["causal_context_boundary"] = {
        "schema_version": "procedural-harness-master-v1/causal-context-boundary/v1",
        "applied": True,
        "applications": 1,
    }

    behavior = _contract_behavior(trace, task.data)

    assert behavior["harness_score"] == 1.0, behavior


def test_direct_control_allows_local_compute_and_rejects_delegation() -> None:
    task = _curriculum_task("natural_direct_control")

    direct = _contract_behavior(_trace(task, []), task.data)
    local_compute = _contract_behavior(
        _trace(task, [("cell", "scratch = 1 + 1", "2")]), task.data
    )
    delegated = _contract_behavior(
        _trace(
            task,
            [("cell", "child = await rlm('unneeded')", "RLMSpawnHandle(name='x')")],
        ),
        task.data,
    )

    assert direct["harness_score"] == 1.0
    assert local_compute["harness_score"] == 1.0
    assert delegated["harness_score"] == 0.0
    assert delegated["forbidden_atom_violations"] == 1.0


def test_n1a_instruction_string_is_not_misclassified_as_child_resource_read() -> None:
    task = _curriculum_task("natural_n1a")
    child = task.data.oracle["children"][0]
    trace = _trace(
        task,
        [
            (
                "cell",
                f"instruction = 'Review {child['resource_path']} and reply once'",
                "",
            ),
            (
                "cell",
                f"handle = await rlm(instruction, name={child['name']!r})",
                f"RLMSpawnHandle(name='{child['name']}')",
            ),
            ("incoming", child["name"], str(child["expected_result"])),
        ],
    )

    behavior = _contract_behavior(trace, task.data)

    assert behavior["harness_score"] == 1.0, behavior
    assert behavior["forbidden_atom_violations"] == 0.0


def test_n1b_context_boundary_removes_visible_ephemeral_history() -> None:
    task = _curriculum_task("natural_n1b")
    child = task.data.oracle["children"][0]
    lease = task.data.oracle["persistence_lease"]
    secret = str(lease["expected_value"])
    trace = _trace(task, [])
    trace.info["persistence_lease"] = {
        "schema_version": "procedural-harness-master-v1/persistence-lease/v1",
        "closed": True,
        "path": lease["path"],
    }
    tool_call = ToolCall(
        id="capture",
        name="ipython",
        arguments=json.dumps(
            {"code": f"captured = json.loads(open({lease['path']!r}).read())"}
        ),
    )
    request = vf.Request(
        messages=[
            UserMessage(content=task.data.prompt),
            AssistantMessage(content="", tool_calls=[tool_call]),
            ToolMessage(tool_call_id="capture", content=f"captured={secret}"),
            UserMessage(
                content=(
                    f"[from child:{child['name']}]\n"
                    "Agent-to-agent message received.\n"
                    "Source: agent_message\nMessage id: msg-test\n\n"
                    f"FINDING={child['expected_result']}"
                )
            ),
        ]
    )

    rewritten = _rewrite_for_boundary(trace, request)

    assert rewritten is not None
    assert len(rewritten.messages) == 2
    visible = json.dumps([message.model_dump() for message in rewritten.messages])
    assert f"captured={secret}" not in visible
    assert BOUNDARY_MARKER in visible
    assert rewritten.messages[0].content == task.data.prompt
    assert trace.info["causal_context_boundary"]["applied"] is True

    install_causal_context_boundary()
    body = {"messages": [message_to_wire(message) for message in request.messages]}
    ChatDialect().rewrite_request(body, request, rewritten)
    wire = json.dumps(body)
    assert len(body["messages"]) == 2
    assert f"captured={secret}" not in wire
    assert BOUNDARY_MARKER in wire


@pytest.mark.asyncio
async def test_n1b_closes_the_intake_lease_after_delegation_is_admitted() -> None:
    class Runtime:
        def __init__(self):
            self.commands = []
            self.writes = {}

        async def run(self, argv, env):
            self.commands.append(argv)
            return SimpleNamespace(exit_code=0, stderr="")

        async def write(self, path, data):
            self.writes[path] = data

    task = _curriculum_task("natural_n1b")
    trace = _trace(task, [])
    runtime = Runtime()
    await task.setup(trace, runtime)
    setup_commands = list(runtime.commands)
    response = vf.Response(
        id="response",
        created=0,
        model="test",
        message=AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(
                    id="delegate",
                    name="ipython",
                    arguments=json.dumps(
                        {"code": "child = await rlm('review', name='reviewer')"}
                    ),
                )
            ],
        ),
        finish_reason="tool_calls",
    )

    task.arm_causal_persistence_lease(response, trace)

    assert runtime.commands == setup_commands
    assert trace.info["persistence_lease"]["pending"] is True

    await task.close_causal_persistence_lease(
        vf.Request(messages=[UserMessage(content="child admission request")]), trace
    )

    lease_path = task.data.oracle["persistence_lease"]["path"]
    assert runtime.commands[-1] == ["rm", "-f", lease_path]
    assert trace.info["persistence_lease"]["pending"] is False
    assert trace.info["persistence_lease"]["closed"] is True
    await task.finalize(trace, runtime)
    assert trace.id not in task._runtime_by_trace


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


def test_natural_yield_feedback_targets_first_tool_call_after_valid_spawn() -> None:
    task = _curriculum_task("natural_n1")
    child = task.data.oracle["children"][0]
    spawn_code = (
        "child_handle = await rlm("
        f"'Review {child['resource_path']} and report the finding to your parent', "
        f"name={child['name']!r})"
    )
    trace = _trace(
        task,
        [
            ("cell", spawn_code, f"RLMSpawnHandle(name='{child['name']}')"),
            ("cell", "await agent_message.list_agents()", "family roster"),
            ("incoming", child["name"], str(child["expected_result"])),
        ],
    )
    for index, node in enumerate(trace.nodes):
        node.token_ids = [index * 10 + 1, index * 10 + 2]
        node.mask = [node.sampled, node.sampled]

    diagnostic = _natural_yield_feedback_diagnostic(trace, task.data)

    assert diagnostic is not None
    assert diagnostic.child_name == child["name"]
    assert diagnostic.turn_index == 1
    target = trace.nodes[diagnostic.target_node_index]
    assert isinstance(target.message, AssistantMessage)
    assert "agent_message.list_agents" in target.message.tool_calls[0].arguments
    assert _record_natural_yield_feedback(trace, task.data)
    contract = trace.info["feedback_contract"]
    assert contract["schema_version"] == "prime-agent/natural-yield-feedback/v1"
    assert contract["answer_free"] is True
    assert contract["target_node_index"] == diagnostic.target_node_index
    assert str(child["expected_result"]) not in trace.info["feedback"]

    masks = keep_natural_yield_feedback_response(trace)
    selected = [
        id(node)
        for branch, mask in zip(trace.branches, masks, strict=True)
        for node, keep in _nodes_with_mask(branch.nodes, mask)
        if any(keep)
    ]
    assert selected == [id(target)]


def test_natural_yield_feedback_rejects_unretained_spawn_and_passive_wait() -> None:
    task = _curriculum_task("natural_n1")
    child = task.data.oracle["children"][0]
    prompt = f"Review {child['resource_path']} and report the finding to your parent"
    unretained = _trace(
        task,
        [
            (
                "cell",
                f"await rlm({prompt!r}, name={child['name']!r})",
                f"RLMSpawnHandle(name='{child['name']}')",
            ),
            ("cell", "await agent_message.list_agents()", "family roster"),
        ],
    )
    passive = _trace(
        task,
        [
            (
                "cell",
                f"handle = await rlm({prompt!r}, name={child['name']!r})",
                f"RLMSpawnHandle(name='{child['name']}')",
            ),
            ("assistant", "Waiting for the reviewer report."),
            ("incoming", child["name"], str(child["expected_result"])),
        ],
    )

    assert _natural_yield_feedback_diagnostic(unretained, task.data) is None
    assert _natural_yield_feedback_diagnostic(passive, task.data) is None


def test_natural_yield_feedback_allows_only_harmless_pre_spawn_setup() -> None:
    task = _curriculum_task("natural_n1")
    child = task.data.oracle["children"][0]
    prompt = f"Review {child['resource_path']} and report the finding to your parent"
    clean = _trace(
        task,
        [
            ("cell", "local_total = sum([1, 2, 3])", "6"),
            (
                "cell",
                f"handle = await rlm({prompt!r}, name={child['name']!r})",
                f"RLMSpawnHandle(name='{child['name']}')",
            ),
            ("cell", "await agent_message.list_agents()", "family roster"),
        ],
    )

    assert _natural_yield_feedback_diagnostic(clean, task.data) is not None


@pytest.mark.parametrize(
    "detour",
    [
        "await rlm('wrong child', name='wrong-worker')",
        "await agent_message.list_agents()",
        "await agent_observe.recent()",
        "await asyncio.sleep(1)",
    ],
)
def test_natural_yield_feedback_rejects_pre_spawn_control_detours(
    detour: str,
) -> None:
    task = _curriculum_task("natural_n1")
    child = task.data.oracle["children"][0]
    prompt = f"Review {child['resource_path']} and report the finding to your parent"
    trace = _trace(
        task,
        [
            ("cell", detour, "failed or irrelevant harness action"),
            (
                "cell",
                f"handle = await rlm({prompt!r}, name={child['name']!r})",
                f"RLMSpawnHandle(name='{child['name']}')",
            ),
            ("cell", "await agent_message.list_agents()", "family roster"),
        ],
    )

    assert _natural_yield_feedback_diagnostic(trace, task.data) is None


def test_natural_yield_feedback_rejects_pre_spawn_child_owned_access() -> None:
    task = _curriculum_task("natural_n1")
    child = task.data.oracle["children"][0]
    prompt = f"Review {child['resource_path']} and report the finding to your parent"
    trace = _trace(
        task,
        [
            ("cell", f"open({child['resource_path']!r}).read()", "private evidence"),
            (
                "cell",
                f"handle = await rlm({prompt!r}, name={child['name']!r})",
                f"RLMSpawnHandle(name='{child['name']}')",
            ),
            ("cell", "await agent_message.list_agents()", "family roster"),
        ],
    )

    assert _natural_yield_feedback_diagnostic(trace, task.data) is None


def _nodes_with_mask(nodes, mask):
    offset = 0
    for node in nodes:
        end = offset + len(node.token_ids)
        yield node, mask[offset:end]
        offset = end
    assert offset == len(mask)


def test_atomic_child_request_action_filter_excludes_reasoning_child_and_retries() -> (
    None
):
    task = _curriculum_task("atomic_child_request")
    child = task.data.oracle["children"][0]
    state_name, state_value = next(iter(task.data.oracle["coordinator_state"].items()))
    spawn_code = (
        f"{state_name} = {state_value}\n"
        "handle = await rlm("
        "\"Execute exactly once: await agent_message.send('need multiplier', "
        "receiver_role='parent'), then stop.\", "
        f"name={child['name']!r})\n"
        "handle"
    )
    tool_start = 248058
    tool_end = 248059
    thinking_end = 248069
    message_end = 248046
    nodes = [
        MessageNode(
            parent=None,
            message=UserMessage(content="task"),
            sampled=False,
            token_ids=[1],
            mask=[False],
        ),
        MessageNode(
            parent=0,
            message=AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="spawn",
                        name="ipython",
                        arguments=json.dumps({"code": spawn_code}),
                    )
                ],
            ),
            sampled=True,
            token_ids=[101, thinking_end, 102, tool_start, 103, tool_end, message_end],
            mask=[True] * 7,
        ),
        MessageNode(
            parent=1,
            message=ToolMessage(
                tool_call_id="spawn",
                content=f"RLMSpawnHandle(name='{child['name']}')",
            ),
            sampled=False,
            token_ids=[2],
            mask=[False],
        ),
        MessageNode(
            parent=2,
            message=AssistantMessage(content="Waiting."),
            sampled=True,
            token_ids=[201, thinking_end, 202, 203, message_end],
            mask=[True] * 5,
        ),
        MessageNode(
            parent=3,
            message=UserMessage(content="[from child:worker]\nneed multiplier"),
            sampled=False,
            token_ids=[3],
            mask=[False],
        ),
        MessageNode(
            parent=4,
            message=AssistantMessage(content="Almost JSON"),
            sampled=True,
            token_ids=[301, thinking_end, 302, message_end],
            mask=[True] * 4,
        ),
        MessageNode(
            parent=5,
            message=UserMessage(content="completion gate: strict JSON"),
            sampled=False,
            token_ids=[4],
            mask=[False],
        ),
        MessageNode(
            parent=6,
            message=AssistantMessage(
                content='{"multiplier": 6, "request_received": true}'
            ),
            sampled=True,
            token_ids=[401, thinking_end, 402, 403, message_end],
            mask=[True] * 5,
        ),
        MessageNode(
            parent=None,
            message=UserMessage(content="[task from parent] send request"),
            sampled=False,
            token_ids=[5],
            mask=[False],
        ),
        MessageNode(
            parent=8,
            message=AssistantMessage(content="child sent"),
            sampled=True,
            token_ids=[501, thinking_end, 502, message_end],
            mask=[True] * 4,
        ),
    ]
    trace = vf.Trace(
        id="action-filter-test",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        nodes=nodes,
        calls=[
            vf.ModelCall(node=1, client_session_id="parent"),
            vf.ModelCall(node=3, client_session_id="parent"),
            vf.ModelCall(node=5, client_session_id="parent"),
            vf.ModelCall(node=7, client_session_id="parent"),
            vf.ModelCall(node=9, client_session_id="child"),
        ],
    )

    masks = keep_atomic_child_request_coordinator_actions(trace)
    selected = {
        id(node): [
            token for token, keep in zip(node.token_ids, keep_mask, strict=True) if keep
        ]
        for branch, mask in zip(trace.branches, masks, strict=True)
        for node, keep_mask in _nodes_with_mask(branch.nodes, mask)
        if any(keep_mask)
    }

    assert selected == {
        id(nodes[1]): [tool_start, 103, tool_end, message_end],
        id(nodes[3]): [202, 203, message_end],
        id(nodes[7]): [402, 403, message_end],
    }

    wire_trace = vf.WireTrace.model_validate(trace.model_dump())
    assert keep_atomic_child_request_coordinator_actions(wire_trace) == masks

    noisy = trace.model_copy(deep=True)
    noisy.nodes[1].message.tool_calls[0].arguments = json.dumps(
        {"code": f"{spawn_code}\nprint('spawned')"}
    )
    assert not any(
        keep
        for mask in keep_atomic_child_request_coordinator_actions(noisy)
        for keep in mask
    )

    failed = trace.model_copy(deep=True)
    failed.nodes[2].message.content = "TypeError: invalid rlm arguments"
    assert not any(
        keep
        for mask in keep_atomic_child_request_coordinator_actions(failed)
        for keep in mask
    )

    polling = trace.model_copy(deep=True)
    polling.nodes[3].message = AssistantMessage(
        content=None,
        tool_calls=[
            ToolCall(
                id="poll",
                name="ipython",
                arguments=json.dumps({"code": "await agent_message.list()"}),
            )
        ],
    )
    assert not any(
        keep
        for mask in keep_atomic_child_request_coordinator_actions(polling)
        for keep in mask
    )


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


@pytest.mark.parametrize("rung", ["natural_n1", "natural_n2"])
def test_natural_curriculum_requires_complete_semantic_message_trajectory(
    rung: str,
) -> None:
    task = _curriculum_task(rung)
    child = task.data.oracle["children"][0]
    state_name, _ = next(iter(task.data.oracle["coordinator_state"].items()))
    actions = [
        (
            "cell",
            _spawn_code(task),
            f"RLMSpawnHandle(name='{child['name']}')",
        )
    ]
    local_paths = [
        path
        for path, item in task.data.oracle["resource_ownership"].items()
        if item["owner"] == "coordinator"
    ]
    for path in local_paths:
        actions.append(
            ("cell", f"from pathlib import Path\nPath({path!r}).read_text()")
        )

    if rung == "natural_n2":
        request_term = task.data.oracle["request_terms"][0]
        actions.extend(
            [
                ("incoming", child["name"], f"Please provide the {request_term}."),
                (
                    "cell",
                    (
                        f"await agent_message.send(str({state_name}), "
                        "receiver_role='child', "
                        f"receiver_name={child['name']!r})"
                    ),
                    "Agent message sent: agentmsg-natural-followup",
                ),
                ("incoming", child["name"], "The completed review result is ready."),
            ]
        )
    else:
        actions.append(("incoming", child["name"], str(child["expected_result"])))

    behavior = _contract_behavior(_trace(task, actions), task.data)

    assert behavior["harness_score"] == 1.0, behavior
    assert task.data.generation_metadata["natural_stage"] == (
        "N1" if rung == "natural_n1" else "N2"
    )


def test_natural_local_work_reports_premature_yield_directly() -> None:
    task = ProceduralHarnessMasterTaskset(
        ProceduralHarnessMasterConfig(
            split="train_gen",
            count=1,
            start_index=1,
            curriculum_rung="natural_n1",
        )
    ).load()[0]
    child = task.data.oracle["children"][0]
    local_path = next(
        path
        for path, item in task.data.oracle["resource_ownership"].items()
        if item["owner"] == "coordinator"
    )
    spawn = (
        "handle = await rlm("
        f"'Review {child['resource_path']} and report to parent', "
        f"name={child['name']!r})"
    )
    premature = _contract_behavior(
        _trace(
            task,
            [
                ("cell", spawn, f"RLMSpawnHandle(name='{child['name']}')"),
                ("incoming", child["name"], str(child["expected_result"])),
            ],
        ),
        task.data,
    )
    ordered = _contract_behavior(
        _trace(
            task,
            [
                ("cell", spawn, f"RLMSpawnHandle(name='{child['name']}')"),
                ("cell", f"open({local_path!r}).read()", "local evidence"),
                ("incoming", child["name"], str(child["expected_result"])),
            ],
        ),
        task.data,
    )

    assert premature["local_work_before_yield"] == 0.0
    assert premature["premature_yield_before_local_work"] == 1.0
    assert ordered["local_work_before_yield"] == 1.0
    assert ordered["premature_yield_before_local_work"] == 0.0


def test_natural_immediate_yield_reports_post_spawn_tool_directly() -> None:
    task = ProceduralHarnessMasterTaskset(
        ProceduralHarnessMasterConfig(
            split="train_gen",
            count=1,
            start_index=0,
            curriculum_rung="natural_n1",
        )
    ).load()[0]
    child = task.data.oracle["children"][0]
    spawn = (
        "handle = await rlm("
        f"'Review {child['resource_path']} and report to parent', "
        f"name={child['name']!r})"
    )
    yielded = _contract_behavior(
        _trace(
            task,
            [
                ("cell", spawn, f"RLMSpawnHandle(name='{child['name']}')"),
                ("incoming", child["name"], str(child["expected_result"])),
            ],
        ),
        task.data,
    )
    acted = _contract_behavior(
        _trace(
            task,
            [
                ("cell", spawn, f"RLMSpawnHandle(name='{child['name']}')"),
                ("cell", "print('still waiting')", "still waiting"),
                ("incoming", child["name"], str(child["expected_result"])),
            ],
        ),
        task.data,
    )

    assert yielded["forbidden_post_spawn_tool_before_child"] == 0.0
    assert acted["forbidden_post_spawn_tool_before_child"] == 1.0


def test_natural_dependency_rejects_private_value_disclosed_at_spawn() -> None:
    task = _curriculum_task("natural_n2")
    child = task.data.oracle["children"][0]
    state_name, state_value = next(iter(task.data.oracle["coordinator_state"].items()))
    path = child["resource_path"]
    actions = [
        (
            "cell",
            (
                f"{state_name} = {state_value!r}\n"
                f"handle = await rlm('Read {path}; use private value {state_value}', "
                f"name={child['name']!r})"
            ),
            f"RLMSpawnHandle(name='{child['name']}')",
        ),
        ("incoming", child["name"], "Please provide the calibration factor."),
        (
            "cell",
            (
                f"await agent_message.send(str({state_name}), receiver_role='child', "
                f"receiver_name={child['name']!r})"
            ),
            "Agent message sent: agentmsg-natural-followup",
        ),
        ("incoming", child["name"], "The completed review result is ready."),
    ]

    behavior = _contract_behavior(_trace(task, actions), task.data)

    assert behavior["no_forbidden_atoms"] == 0.0
    assert behavior["harness_score"] == 0.0


def test_natural_dependency_rejects_wrong_parent_value() -> None:
    task = _curriculum_task("natural_n2")
    child = task.data.oracle["children"][0]
    state_name, state_value = next(iter(task.data.oracle["coordinator_state"].items()))
    actions = [
        (
            "cell",
            _spawn_code(task),
            f"RLMSpawnHandle(name='{child['name']}')",
        ),
        ("incoming", child["name"], f"Please provide the {state_name}."),
        (
            "cell",
            (
                f"await agent_message.send(str({state_value + 1}), "
                "receiver_role='child', "
                f"receiver_name={child['name']!r})"
            ),
            "Agent message sent: agentmsg-wrong-followup",
        ),
        ("incoming", child["name"], "The completed review result is ready."),
    ]

    behavior = _contract_behavior(_trace(task, actions), task.data)

    assert behavior["all_required_atoms"] == 0.0
    assert behavior["no_forbidden_atoms"] == 0.0
    assert behavior["harness_score"] == 0.0


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
        "await agent_message.list()",
        "await agent_message.listen_for_messages()",
        "await agent_message.recv()",
        "await agent_message.receive(timeout=30)",
        "await agent_message.wait_for_reply(name='worker')",
        "await agent_message.list_messages()",
        "import asyncio\nawait asyncio.sleep(1)",
        "import time\ntime.sleep(1)",
        "from asyncio import sleep\nawait sleep(1)",
        "from agent_message import list_agents\nawait list_agents()",
        "from agent_message import list_agents as roster\nawait roster()",
        "from agent_message import listen_for_messages\nawait listen_for_messages()",
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


@pytest.mark.asyncio
async def test_event_control_reward_preserves_first_atom_signal_before_ordering_progress() -> None:
    task = _task("single")
    task.config.reward_mode = "event_control"
    child = task.data.oracle["children"][0]
    trace = _trace(
        task,
        [("cell", _spawn_code(task), f"RLMSpawnHandle(name='{child['name']}')")],
        reply="{}",
    )

    behavior = _contract_behavior(trace, task.data)
    assert behavior["harness_score"] == 0.0
    assert behavior["required_atoms_fraction"] > 0.0
    assert behavior["causal_prefix_fraction"] > 0.0
    assert behavior["ordering_fraction"] < 1.0
    assert behavior["event_control_progress"] > 0.0
    assert await task.harness_score(trace) == pytest.approx(
        behavior["event_control_progress"]
    )


@pytest.mark.asyncio
async def test_event_control_rewards_correct_child_send_before_delivery() -> None:
    task = _curriculum_task("natural_n1a")
    task.config.reward_mode = "event_control"
    child = task.data.oracle["children"][0]
    trace = _trace(
        task,
        [("cell", _spawn_code(task), f"RLMSpawnHandle(name='{child['name']}')")],
        reply="{}",
    )
    child_root = len(trace.nodes)
    trace.nodes.append(
        MessageNode(
            parent=None,
            message=UserMessage(content=f"{PRIVATE_EVIDENCE_HEADER}\nevidence"),
            sampled=False,
        )
    )
    _cell(
        trace.nodes,
        child_root,
        f"await agent_message.send({str(child['expected_result'])!r}, receiver_role='parent')",
    )

    behavior = _contract_behavior(trace, task.data)
    assert behavior["child_action_progress"] == 1.0
    assert behavior["child_action_bridge"] > 0.0
    assert await task.harness_score(trace) == pytest.approx(
        behavior["event_control_progress"] + behavior["child_action_bridge"]
    )


@pytest.mark.asyncio
async def test_child_action_reward_is_local_to_sampled_child_send() -> None:
    task = _curriculum_task("natural_n1a")
    task.config.reward_mode = "child_action"
    child = task.data.oracle["children"][0]
    # Deliberately leave the coordinator terminal answer wrong. Child-role GRPO
    # must still receive full credit for its own correct, independently sampled
    # report instead of inheriting downstream coordinator noise.
    trace = _trace(
        task,
        [("cell", _spawn_code(task), f"RLMSpawnHandle(name='{child['name']}')")],
        reply="{}",
    )
    child_root = len(trace.nodes)
    trace.nodes.append(
        MessageNode(
            parent=None,
            message=UserMessage(content=f"{PRIVATE_EVIDENCE_HEADER}\nevidence"),
            sampled=False,
        )
    )
    _cell(
        trace.nodes,
        child_root,
        f"await agent_message.send({str(child['expected_result'])!r}, receiver_role='parent')",
    )
    _incoming(trace.nodes, 0, child["name"], str(child["expected_result"]))

    behavior = _contract_behavior(trace, task.data)
    assert behavior["harness_score"] == 0.0
    assert behavior["child_action_progress"] == 1.0
    assert behavior["child_action_completed"] == 1.0
    assert behavior["child_action_local_reward"] == 1.0
    assert await task.harness_score(trace) == 1.0


@pytest.mark.asyncio
async def test_child_action_reward_accepts_implicit_parent_send() -> None:
    task = _curriculum_task("natural_n1a")
    task.config.reward_mode = "child_action"
    child = task.data.oracle["children"][0]
    trace = _trace(
        task,
        [("cell", _spawn_code(task), f"RLMSpawnHandle(name='{child['name']}')")],
        reply="{}",
    )
    child_root = len(trace.nodes)
    trace.nodes.append(
        MessageNode(
            parent=None,
            message=UserMessage(content=f"{PRIVATE_EVIDENCE_HEADER}\nevidence"),
            sampled=False,
        )
    )
    _cell(
        trace.nodes,
        child_root,
        f"await agent_message.send({str(child['expected_result'])!r})",
    )
    _incoming(trace.nodes, 0, child["name"], str(child["expected_result"]))

    behavior = _contract_behavior(trace, task.data)
    assert behavior["child_action_progress"] == 1.0
    assert behavior["child_action_completed"] == 1.0
    assert behavior["child_action_local_reward"] == 1.0
    assert await task.harness_score(trace) == 1.0


@pytest.mark.asyncio
async def test_recursive_coordinator_return_rejects_descendant_spawn() -> None:
    original = _curriculum_task("natural_n1a")
    metadata = {
        **original.data.generation_metadata,
        "delegated_session_role": "coordinator",
    }
    config = ProceduralHarnessMasterTaskConfig(
        reward_mode="child_action",
        delegated_session_role="coordinator",
    )
    task = ProceduralHarnessMasterTask(
        original.data.model_copy(update={"generation_metadata": metadata}), config
    )
    child = task.data.oracle["children"][0]
    trace = _trace(
        task,
        [("cell", _spawn_code(task), f"RLMSpawnHandle(name='{child['name']}')")],
        reply="{}",
    )
    delegated_root = len(trace.nodes)
    trace.nodes.append(
        MessageNode(
            parent=None,
            message=UserMessage(content=f"{RECURSIVE_COORDINATOR_HEADER}\nevidence"),
            sampled=False,
        )
    )
    _cell(
        trace.nodes,
        delegated_root,
        (
            "helper = await rlm('do this', name='descendant')\n"
            f"await agent_message.send({str(child['expected_result'])!r})"
        ),
    )
    _incoming(trace.nodes, 0, child["name"], str(child["expected_result"]))

    behavior = _contract_behavior(trace, task.data)
    assert behavior["child_action_progress"] == 1.0
    assert behavior["delegated_descendant_calls"] == 1.0
    assert behavior["child_action_completed"] == 0.0
    assert behavior["child_action_local_reward"] == 0.0
    assert await task.harness_score(trace) == 0.0


@pytest.mark.asyncio
async def test_child_action_reward_preserves_baby_step_ramp() -> None:
    task = _curriculum_task("natural_n1a")
    task.config.reward_mode = "child_action"
    trace = _trace(task, [], reply="{}")
    child_root = len(trace.nodes)
    trace.nodes.append(
        MessageNode(
            parent=None,
            message=UserMessage(content=f"{PRIVATE_EVIDENCE_HEADER}\nevidence"),
            sampled=False,
        )
    )
    _cell(
        trace.nodes,
        child_root,
        "await agent_message.send('wrong', receiver_role='parent')",
    )

    behavior = _contract_behavior(trace, task.data)
    assert behavior["child_action_progress"] == 0.75
    assert behavior["child_action_completed"] == 0.0
    assert behavior["child_action_local_reward"] == 0.75
    assert await task.harness_score(trace) == 0.75


@pytest.mark.asyncio
async def test_child_action_reward_caps_correct_send_without_clean_stop() -> None:
    task = _curriculum_task("natural_n1a")
    task.config.reward_mode = "child_action"
    child = task.data.oracle["children"][0]
    trace = _trace(task, [], reply="{}")
    child_root = len(trace.nodes)
    trace.nodes.append(
        MessageNode(
            parent=None,
            message=UserMessage(content=f"{PRIVATE_EVIDENCE_HEADER}\nevidence"),
            sampled=False,
        )
    )
    parent = _cell(
        trace.nodes,
        child_root,
        f"await agent_message.send({str(child['expected_result'])!r}, receiver_role='parent')",
    )
    trace.nodes.append(
        MessageNode(
            parent=parent,
            message=AssistantMessage(content="Sent."),
            sampled=True,
        )
    )
    _incoming(trace.nodes, 0, child["name"], str(child["expected_result"]))

    behavior = _contract_behavior(trace, task.data)
    assert behavior["child_action_progress"] == 1.0
    assert behavior["child_action_completed"] == 0.0
    assert behavior["child_action_local_reward"] == 0.75
    assert await task.harness_score(trace) == 0.75


@pytest.mark.asyncio
async def test_child_action_reward_accepts_scaffolded_terminal_prose_stop() -> None:
    task = _curriculum_task("natural_n1a")
    task.config.reward_mode = "child_action"
    child = task.data.oracle["children"][0]
    trace = _trace(task, [], reply="{}")
    child_root = len(trace.nodes)
    trace.nodes.append(
        MessageNode(
            parent=None,
            message=UserMessage(content=f"{PRIVATE_EVIDENCE_HEADER}\nevidence"),
            sampled=False,
        )
    )
    parent = _cell(
        trace.nodes,
        child_root,
        f"await agent_message.send({str(child['expected_result'])!r}, receiver_role='parent')",
    )
    trace.nodes.append(
        MessageNode(
            parent=parent,
            message=AssistantMessage(content="Sent to parent. Stopping."),
            sampled=True,
        )
    )
    trace.info["interaction_curriculum"] = {
        "child_stop": {"mode": "one_turn_no_tools", "fired": True}
    }
    _incoming(trace.nodes, 0, child["name"], str(child["expected_result"]))

    behavior = _contract_behavior(trace, task.data)
    assert behavior["child_action_progress"] == 1.0
    assert behavior["child_action_completed"] == 1.0
    assert behavior["child_action_local_reward"] == 1.0
    assert await task.harness_score(trace) == 1.0


@pytest.mark.asyncio
async def test_child_action_reward_accepts_natural_terminal_prose_stop() -> None:
    task = _curriculum_task("natural_n1a")
    task.config.reward_mode = "child_action"
    child = task.data.oracle["children"][0]
    trace = _trace(task, [], reply="{}")
    child_root = len(trace.nodes)
    trace.nodes.append(
        MessageNode(
            parent=None,
            message=UserMessage(content=f"{PRIVATE_EVIDENCE_HEADER}\nevidence"),
            sampled=False,
        )
    )
    parent = _cell(
        trace.nodes,
        child_root,
        f"await agent_message.send({str(child['expected_result'])!r}, receiver_role='parent')",
    )
    trace.nodes.append(
        MessageNode(
            parent=parent,
            message=AssistantMessage(content="Sent to parent. Stopping."),
            sampled=True,
        )
    )
    trace.stop_condition = "user_closed"
    _incoming(trace.nodes, 0, child["name"], str(child["expected_result"]))

    behavior = _contract_behavior(trace, task.data)
    assert behavior["child_action_progress"] == 1.0
    assert behavior["child_action_completed"] == 1.0
    assert behavior["child_action_local_reward"] == 1.0
    assert await task.harness_score(trace) == 1.0


@pytest.mark.asyncio
async def test_event_control_does_not_reward_empty_child_tool_code() -> None:
    task = _curriculum_task("natural_n1a")
    task.config.reward_mode = "event_control"
    child = task.data.oracle["children"][0]
    trace = _trace(
        task,
        [("cell", _spawn_code(task), f"RLMSpawnHandle(name='{child['name']}')")],
        reply="{}",
    )
    child_root = len(trace.nodes)
    trace.nodes.append(
        MessageNode(
            parent=None,
            message=UserMessage(content=f"{PRIVATE_EVIDENCE_HEADER}\nevidence"),
            sampled=False,
        )
    )
    _cell(trace.nodes, child_root, "")

    behavior = _contract_behavior(trace, task.data)
    assert behavior["child_action_progress"] == 0.0
    assert behavior["child_action_bridge"] == 0.0


@pytest.mark.asyncio
async def test_event_control_reward_rejects_out_of_order_final_answer_credit() -> None:
    task = _task("single")
    task.config.reward_mode = "event_control"
    trace = _trace(task, [], reply="premature final response")

    behavior = _contract_behavior(trace, task.data)

    assert behavior["required_atoms_fraction"] > 0.0
    assert behavior["causal_prefix_fraction"] == 0.0
    assert behavior["event_control_progress"] == 0.0
    assert await task.harness_score(trace) == 0.0


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

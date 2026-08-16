import asyncio
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from procedural_harness_master_v1.taskset import (
    COMPLETION_GATE_PATH,
    ProceduralHarnessMasterConfig,
    ProceduralHarnessMasterTaskset,
    _contract_behavior,
)

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, ToolCall, ToolMessage, UserMessage


def _task(family: str, split: str = "train_gen"):
    return ProceduralHarnessMasterTaskset(
        ProceduralHarnessMasterConfig(split=split, count=1, families=(family,))
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


def _trace(task, actions, reply=None):
    nodes = [MessageNode(parent=None, message=UserMessage(content="task"), sampled=False)]
    parent = 0
    for action in actions:
        if action[0] == "cell":
            parent = _cell(nodes, parent, action[1], action[2] if len(action) > 2 else "")
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
    lines = [f"{name} = {value!r}" for name, value in task.data.oracle.get("coordinator_state", {}).items()]
    for index, child in enumerate(children):
        instruction = f"Read {child['resource_path']} and {child['operation']}; send the result to parent."
        lines.append(f"handle_{index} = await rlm({instruction!r}, name={child['name']!r})")
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
    trace = _trace(task, [("cell", f"from pathlib import Path\ntext = Path({path!r}).read_text()")])

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
                ("incoming", child["name"], json.dumps(task.data.oracle["final_answer"])),
            ]
        )
    else:
        for child in reversed(children):
            body = str(child["expected_result"])
            if family == "verify":
                manifest = next(path for path in task.data.workspace_files if "verification" in path)
                digest = json.loads(task.data.workspace_files[manifest])["expected_digest"]
                body = json.dumps({"result": child["expected_result"], "digest": digest})
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

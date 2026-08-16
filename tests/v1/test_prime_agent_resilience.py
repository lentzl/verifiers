import json
from types import SimpleNamespace

import pytest
from prime_agent_resilience_v1.taskset import (
    FAMILIES,
    PrimeAgentResilienceConfig,
    PrimeAgentResilienceTaskset,
    _behavior,
    keep_failed_ipython_tool_calls,
)

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
    content_text,
)


def _task(family: str):
    return PrimeAgentResilienceTaskset(
        PrimeAgentResilienceConfig(
            split="heldout",
            families=(family,),
            instances_per_family=1,
            instance_offset=7,
        )
    ).load()[0]


def _cell(nodes: list[MessageNode], parent: int, code: str, output: str) -> int:
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


def _child_message(
    nodes: list[MessageNode], parent: int, name: str, body: str, timestamp: float
) -> int:
    nodes.append(
        MessageNode(
            parent=parent,
            message=UserMessage(
                content=(
                    f"[from child:{name}]\nAgent-to-agent message received.\n"
                    f"Source: agent_message\n\n{body}"
                )
            ),
            sampled=False,
            timestamp=timestamp,
        )
    )
    return len(nodes) - 1


def _trace(family: str) -> tuple[vf.Trace, object]:
    task = _task(family)
    data = task.data
    answer = json.dumps(data.answer, separators=(",", ":"))
    spawn = f"child = await rlm('Read {data.resource_path} and follow the protocol', name={data.child_name!r})"
    nodes = [
        MessageNode(
            parent=None, message=UserMessage(content="coordinate"), sampled=False
        )
    ]
    coordinator_parent = _cell(
        nodes, 0, spawn, f"RLMSpawnHandle(name={data.child_name!r})"
    )
    nodes.append(
        MessageNode(
            parent=None,
            message=UserMessage(
                content=f"[task from parent]\nRead {data.resource_path}"
            ),
            sampled=False,
        )
    )
    child_parent = len(nodes) - 1

    if family == "malformed_result_repair":
        child_parent = _cell(
            nodes,
            child_parent,
            f"await agent_message.send({data.malformed_message!r}, receiver_role='parent')",
            "Agent message sent: agentmsg_bad",
        )
        coordinator_parent = _child_message(
            nodes,
            coordinator_parent,
            data.child_name,
            data.malformed_message,
            100.0,
        )
        coordinator_parent = _cell(
            nodes,
            coordinator_parent,
            (
                f"await agent_message.send({data.correction_message!r}, "
                "receiver_role='child', receiver_name=child.name)"
            ),
            "Agent message sent: agentmsg_correction",
        )
        nodes.append(
            MessageNode(
                parent=child_parent,
                message=UserMessage(
                    content=f"[from parent]\n\n{data.correction_message}"
                ),
                sampled=False,
            )
        )
        child_parent = _cell(
            nodes,
            len(nodes) - 1,
            "await agent_message.send(json.dumps({'checksum': checksum}), receiver_role='parent')",
            "Agent message sent: agentmsg_good",
        )
        coordinator_parent = _child_message(
            nodes, coordinator_parent, data.child_name, answer, 101.0
        )
    elif family == "delayed_result":
        child_parent = _cell(
            nodes,
            child_parent,
            "await agent_message.send('STARTED', receiver_role='parent')",
            "Agent message sent: agentmsg_started",
        )
        coordinator_parent = _child_message(
            nodes, coordinator_parent, data.child_name, "STARTED", 100.0
        )
        nodes.append(
            MessageNode(
                parent=coordinator_parent,
                message=AssistantMessage(content="Waiting for delayed-worker."),
                sampled=True,
            )
        )
        coordinator_parent = len(nodes) - 1
        child_parent = _cell(
            nodes,
            child_parent,
            (
                f"result = subprocess.run([sys.executable, {data.delayed_script_path!r}], "
                "capture_output=True, text=True)\n"
                "assert result.returncode == 0, result.stderr\n"
                "payload = result.stdout.strip()"
            ),
            answer,
        )
        child_parent = _cell(
            nodes,
            child_parent,
            "await agent_message.send(payload, receiver_role='parent')",
            "Agent message sent: agentmsg_result",
        )
        coordinator_parent = _child_message(
            nodes, coordinator_parent, data.child_name, answer, 102.5
        )
    else:
        child_parent = _cell(
            nodes,
            child_parent,
            "payload = {'checksum': checksum}\nawait agent_message.send(payload, receiver_role='parent')",
            "Traceback (most recent call last):\nTypeError: agent_message.send message must be a string",
        )
        child_parent = _cell(
            nodes,
            child_parent,
            "await agent_message.send(json.dumps(payload), receiver_role='parent')",
            "Agent message sent: agentmsg_result",
        )
        coordinator_parent = _child_message(
            nodes, coordinator_parent, data.child_name, answer, 100.0
        )

    nodes.append(
        MessageNode(
            parent=coordinator_parent,
            message=AssistantMessage(content=answer),
            sampled=True,
        )
    )
    trace = vf.Trace(
        id="prime-agent-resilience-test",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="PrimeAgentResilienceTask", data=vf.TaskData(idx=0)),
        nodes=nodes,
    )
    return trace, data


def _tokenize_sampled_nodes(trace: vf.Trace) -> None:
    for index, node in enumerate(trace.nodes):
        if not node.sampled:
            node.token_ids = [index]
            node.mask = [False]
        elif isinstance(node.message, AssistantMessage) and node.message.tool_calls:
            node.token_ids = [index, 248058, index + 100, 248059, index + 200]
            node.mask = [False, True, True, True, True]
        else:
            node.token_ids = [index, index + 100]
            node.mask = [False, True]


def _selected_ipython_codes(trace: vf.Trace, masks: list[list[bool]]) -> list[str]:
    selected = []
    for branch, branch_mask in zip(trace.branches, masks, strict=True):
        offset = 0
        for node in branch.nodes:
            node_mask = branch_mask[offset : offset + len(node.token_ids)]
            offset += len(node.token_ids)
            if not any(node_mask) or not isinstance(node.message, AssistantMessage):
                continue
            for call in node.message.tool_calls or []:
                if call.name == "ipython":
                    selected.append(json.loads(call.arguments)["code"])
    return selected


def test_resilience_splits_are_disjoint_balanced_and_answer_free() -> None:
    calibration = PrimeAgentResilienceTaskset(
        PrimeAgentResilienceConfig(split="calibration")
    ).load()
    heldout = PrimeAgentResilienceTaskset(
        PrimeAgentResilienceConfig(split="heldout")
    ).load()

    assert len(calibration) == len(heldout) == 6
    assert {task.data.family for task in heldout} == set(FAMILIES)
    assert {task.data.variant for task in calibration}.isdisjoint(
        {task.data.variant for task in heldout}
    )
    assert all(
        str(task.data.answer["checksum"]) not in task.data.prompt for task in heldout
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_strict_resilience_requires_observable_family_repair(family: str) -> None:
    trace, data = _trace(family)

    behavior = _behavior(trace, data)

    assert behavior["strict_success"] == 1.0
    assert behavior["spawn_protocol"] == 1.0
    assert behavior["repair_observed"] == 1.0
    assert behavior["changed_action"] == 1.0
    assert behavior["bounded_wait"] == 1.0


def test_malformed_result_requires_parent_correction_between_messages() -> None:
    trace, data = _trace("malformed_result_repair")
    correction = next(
        node
        for node in trace.nodes
        if isinstance(node.message, AssistantMessage)
        and node.message.tool_calls
        and data.correction_message in node.message.tool_calls[0].arguments
    )
    correction.message.tool_calls[0].arguments = json.dumps(
        {
            "code": "await agent_message.send('try again', receiver_role='child', receiver_name=child.name)"
        }
    )

    assert _behavior(trace, data)["strict_success"] == 0.0


def test_malformed_result_rejects_extra_parent_action_before_repair() -> None:
    trace, data = _trace("malformed_result_repair")
    correction_index = next(
        index
        for index, node in enumerate(trace.nodes)
        if isinstance(node.message, AssistantMessage)
        and node.message.tool_calls
        and data.correction_message in node.message.tool_calls[0].arguments
    )
    correction_parent = trace.nodes[correction_index].parent
    for node in trace.nodes:
        if node.parent is not None and node.parent >= correction_index:
            node.parent += 1
    trace.nodes.insert(
        correction_index,
        MessageNode(
            parent=correction_parent,
            message=AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-extra",
                        name="ipython",
                        arguments=json.dumps({"code": "print('guessing')"}),
                    )
                ],
            ),
            sampled=True,
        ),
    )

    assert _behavior(trace, data)["strict_success"] == 0.0


def test_delayed_result_requires_observed_time_and_quiescent_parent() -> None:
    trace, data = _trace("delayed_result")
    messages = [
        node
        for node in trace.nodes
        if isinstance(node.message, UserMessage)
        and content_text(node.message.content).startswith(
            f"[from child:{data.child_name}]"
        )
    ]
    messages[1].timestamp = messages[0].timestamp + 0.1

    assert _behavior(trace, data)["strict_success"] == 0.0


def test_message_type_repair_rejects_an_unchanged_retry() -> None:
    trace, data = _trace("message_type_repair")
    child_cells = [
        node
        for node in trace.nodes
        if isinstance(node.message, AssistantMessage)
        and node.message.tool_calls
        and "agent_message.send" in node.message.tool_calls[0].arguments
    ]
    child_cells[1].message.tool_calls[0].arguments = (
        child_cells[0].message.tool_calls[0].arguments
    )

    assert _behavior(trace, data)["strict_success"] == 0.0


def test_failed_ipython_filter_selects_only_the_traceback_call() -> None:
    trace, _ = _trace("message_type_repair")
    _tokenize_sampled_nodes(trace)

    masks = keep_failed_ipython_tool_calls(trace)

    assert sum(sum(mask) for mask in masks) == 3
    assert _selected_ipython_codes(trace, masks) == [
        "payload = {'checksum': checksum}\nawait agent_message.send(payload, receiver_role='parent')"
    ]


def test_failed_ipython_filter_recognizes_an_unawaited_coroutine() -> None:
    trace, _ = _trace("message_type_repair")
    failed_output = next(
        node
        for node in trace.nodes
        if isinstance(node.message, ToolMessage)
        and "TypeError" in content_text(node.message.content)
    )
    failed_output.message.content = "<coroutine object AgentMessage.send at 0x1234>"
    _tokenize_sampled_nodes(trace)

    masks = keep_failed_ipython_tool_calls(trace)

    assert sum(sum(mask) for mask in masks) == 3
    assert len(_selected_ipython_codes(trace, masks)) == 1


def test_failed_ipython_filter_drops_successful_tool_calls() -> None:
    trace, _ = _trace("message_type_repair")
    failed_output = next(
        node
        for node in trace.nodes
        if isinstance(node.message, ToolMessage)
        and "TypeError" in content_text(node.message.content)
    )
    failed_output.message.content = "Agent message sent: agentmsg_unexpected"
    _tokenize_sampled_nodes(trace)

    masks = keep_failed_ipython_tool_calls(trace)

    assert sum(sum(mask) for mask in masks) == 0
    assert _selected_ipython_codes(trace, masks) == []


def test_failed_ipython_filter_does_not_match_exception_names_in_source() -> None:
    trace, _ = _trace("message_type_repair")
    failed_output = next(
        node
        for node in trace.nodes
        if isinstance(node.message, ToolMessage)
        and "TypeError" in content_text(node.message.content)
    )
    failed_output.message.content = (
        "def send(message):\n"
        "    raise RuntimeError('daemon unavailable')\n"
        "# TypeError is documented by the bridge\n"
    )
    _tokenize_sampled_nodes(trace)

    masks = keep_failed_ipython_tool_calls(trace)

    assert sum(sum(mask) for mask in masks) == 0


class _Runtime:
    def __init__(self):
        self.runs = []
        self.writes = {}

    async def run(self, argv, env):
        self.runs.append(argv)
        return SimpleNamespace(exit_code=0, stderr="")

    async def write(self, path, contents):
        self.writes[path] = contents


@pytest.mark.asyncio
async def test_setup_materializes_only_task_owned_artifacts() -> None:
    task = _task("delayed_result")
    runtime = _Runtime()

    await task.setup(runtime)

    assert set(runtime.writes) == set(task.data.files)
    assert task.data.resource_path in runtime.writes
    assert task.data.delayed_script_path in runtime.writes
    assert runtime.runs == [["mkdir", "-p", task.data.resource_path.rsplit("/", 1)[0]]]

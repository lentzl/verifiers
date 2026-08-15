"""Observable foundation tasks for the official Prime Agent harness."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.types import AssistantMessage, ToolMessage, UserMessage, content_text

Family = Literal[
    "ipython_cell",
    "kernel_persistence",
    "conversation_resume",
    "child_result_delivery",
    "child_cancellation",
]
FAMILIES: tuple[Family, ...] = (
    "ipython_cell",
    "kernel_persistence",
    "conversation_resume",
    "child_result_delivery",
    "child_cancellation",
)
TOKEN = re.compile(r"\b[0-9a-f]{64}\b")


class PrimeAgentFoundationData(vf.TaskData):
    family: Family
    instance: int
    turns: tuple[str, ...]
    sentinel: str | None = None
    marker_name: str | None = None
    codeword: str | None = None
    child_name: str | None = None
    child_result: str | None = None
    started_message: str | None = None
    finished_message: str | None = None
    started_path: str | None = None
    finished_path: str | None = None


@dataclass(frozen=True)
class IpythonEvent:
    code: str
    output: str
    node_index: int


def _segment_info(segment: vf.Segment) -> dict:
    return {
        "last_reply": segment.last_reply,
        "tool_calls": [
            {"name": call.name, "arguments": call.arguments}
            for message in segment.messages
            if isinstance(message, AssistantMessage)
            for call in message.tool_calls or []
        ],
        "tool_outputs": [
            content_text(message.content)
            for message in segment.messages
            if isinstance(message, ToolMessage)
        ],
        "terminated": segment.terminated,
    }


def _call_code(arguments: str) -> str | None:
    try:
        decoded = json.loads(arguments)
    except (TypeError, json.JSONDecodeError):
        return None
    code = decoded.get("code") if isinstance(decoded, dict) else None
    return code if isinstance(code, str) else None


def _ipython_events(trace: vf.Trace) -> list[IpythonEvent]:
    events = []
    for node_index, node in enumerate(trace.nodes):
        if not isinstance(node.message, AssistantMessage):
            continue
        for call in node.message.tool_calls or []:
            if call.name != "ipython" or (code := _call_code(call.arguments)) is None:
                continue
            output = "\n".join(
                content_text(candidate.message.content)
                for candidate in trace.nodes
                if candidate.parent == node_index
                and isinstance(candidate.message, ToolMessage)
                and candidate.message.tool_call_id == call.id
            )
            events.append(IpythonEvent(code=code, output=output, node_index=node_index))
    return events


def _branch_root(trace: vf.Trace, node_index: int) -> int:
    visited = set()
    while trace.nodes[node_index].parent is not None and node_index not in visited:
        visited.add(node_index)
        node_index = trace.nodes[node_index].parent
    return node_index


def _coordinator_events(trace: vf.Trace) -> list[IpythonEvent]:
    if not trace.nodes:
        return []
    coordinator_root = _branch_root(trace, 0)
    return [
        event
        for event in _ipython_events(trace)
        if _branch_root(trace, event.node_index) == coordinator_root
    ]


def _failed(output: str) -> bool:
    return any(marker in output for marker in ("Traceback", "Error:", "SyntaxError"))


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        return f"{call.func.value.id}.{call.func.attr}"
    return None


def _calls(event: IpythonEvent) -> list[tuple[str, bool]]:
    try:
        tree = ast.parse(event.code)
    except SyntaxError:
        return []
    assigned = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value.value if isinstance(node.value, ast.Await) else node.value
        if isinstance(value, ast.Call):
            assigned.add(id(value))
    return [
        (name, id(node) in assigned)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and (name := _call_name(node)) is not None
    ]


def _incoming_child_messages(trace: vf.Trace, child_name: str) -> list[tuple[int, str]]:
    messages = []
    prefix = f"[from child:{child_name}]"
    for node_index, node in enumerate(trace.nodes):
        if not isinstance(node.message, UserMessage):
            continue
        text = content_text(node.message.content)
        if text.lstrip().startswith(prefix):
            messages.append((node_index, text.rsplit("\n\n", 1)[-1].strip()))
    return messages


def _completed_exactly(trace: vf.Trace, expected: str = "DONE") -> bool:
    return (trace.last_reply or "").strip() == expected


def _ipython_cell_completed(trace: vf.Trace, data: PrimeAgentFoundationData) -> bool:
    segments = trace.info.get("prime_agent_foundation_segments", [])
    if len(segments) != 1 or data.sentinel is None:
        return False
    segment = segments[0]
    expected = {"code": f"print({data.sentinel!r})"}
    invoked = False
    for call in segment["tool_calls"]:
        try:
            arguments = json.loads(call["arguments"])
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        invoked |= call.get("name") == "ipython" and arguments == expected
    return bool(
        invoked
        and any(data.sentinel in output for output in segment["tool_outputs"])
        and segment["last_reply"].strip() == "DONE"
        and not segment["terminated"]
    )


def _kernel_persisted(trace: vf.Trace, data: PrimeAgentFoundationData) -> bool:
    segments = trace.info.get("prime_agent_foundation_segments", [])
    if len(segments) != 2 or data.marker_name is None:
        return False
    first, second = segments
    first_tokens = set(TOKEN.findall("\n".join(first["tool_outputs"])))
    second_tokens = set(TOKEN.findall("\n".join(second["tool_outputs"])))
    second_calls = "\n".join(call["arguments"] for call in second["tool_calls"])
    shared = first_tokens & second_tokens
    return bool(
        shared
        and second["last_reply"].strip() in shared
        and data.marker_name in second_calls
        and "secrets" not in second_calls
        and "NameError" not in "\n".join(second["tool_outputs"])
        and first["last_reply"].strip() == "READY"
        and not first["terminated"]
        and not second["terminated"]
    )


def _conversation_resumed(trace: vf.Trace, data: PrimeAgentFoundationData) -> bool:
    segments = trace.info.get("prime_agent_foundation_segments", [])
    if len(segments) != 2 or data.codeword is None:
        return False
    first, second = segments
    calls = [*first["tool_calls"], *second["tool_calls"]]
    return bool(
        first["last_reply"].strip() == "READY"
        and second["last_reply"].strip() == data.codeword
        and not calls
        and not first["terminated"]
        and not second["terminated"]
    )


def _child_result_delivered(trace: vf.Trace, data: PrimeAgentFoundationData) -> bool:
    if data.child_name is None or data.child_result is None:
        return False
    events = _coordinator_events(trace)
    spawns = [
        event
        for event in events
        if not _failed(event.output) and ("rlm", True) in _calls(event)
    ]
    messages = _incoming_child_messages(trace, data.child_name)
    polled = any(
        "list_messages" in event.code or "list_subagents" in event.code
        for event in events
    )
    return bool(
        len(spawns) == 1
        and any(data.child_result in body for _, body in messages)
        and not polled
        and _completed_exactly(trace)
    )


async def _child_cancelled(
    trace: vf.Trace,
    data: PrimeAgentFoundationData,
    runtime: vf.Runtime,
) -> bool:
    required = (
        data.child_name,
        data.started_message,
        data.finished_message,
        data.started_path,
        data.finished_path,
    )
    if any(value is None for value in required):
        return False
    events = _coordinator_events(trace)
    spawns = [
        event
        for event in events
        if not _failed(event.output) and ("rlm", True) in _calls(event)
    ]
    messages = _incoming_child_messages(trace, data.child_name or "")
    started = next(
        (node_index for node_index, body in messages if data.started_message in body),
        None,
    )
    deletes = [
        event
        for event in events
        if not _failed(event.output)
        and any(name == "rlm.delete_subagent" for name, _ in _calls(event))
    ]
    started_file = (
        await runtime.run(["test", "-f", data.started_path or ""], {})
    ).exit_code == 0
    finished_file = (
        await runtime.run(["test", "-f", data.finished_path or ""], {})
    ).exit_code == 0
    finished_message = any(data.finished_message in body for _, body in messages)
    return bool(
        len(spawns) == 1
        and started is not None
        and len(deletes) == 1
        and deletes[0].node_index > started
        and started_file
        and not finished_file
        and not finished_message
        and _completed_exactly(trace)
    )


class PrimeAgentFoundationsTask(vf.Task[PrimeAgentFoundationData]):
    async def setup(self, runtime: vf.Runtime) -> None:
        paths = [
            path
            for path in (self.data.started_path, self.data.finished_path)
            if path is not None
        ]
        if paths:
            removed = await runtime.run(["rm", "-f", *paths], {})
            if removed.exit_code != 0:
                raise RuntimeError(f"foundation marker cleanup failed: {removed.stderr[-500:]}")

    @vf.reward(weight=1.0)
    async def foundation(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        family = self.data.family
        if family == "ipython_cell":
            success = _ipython_cell_completed(trace, self.data)
        elif family == "kernel_persistence":
            success = _kernel_persisted(trace, self.data)
        elif family == "conversation_resume":
            success = _conversation_resumed(trace, self.data)
        elif family == "child_result_delivery":
            success = _child_result_delivered(trace, self.data)
        else:
            success = await _child_cancelled(trace, self.data, runtime)
        return float(success)


class PrimeAgentFoundationsEnv(vf.SingleAgentEnv):
    async def run(self, task, agents):
        async with agents.agent.interaction(task) as interaction:
            if task.data.family in {"kernel_persistence", "conversation_resume"}:
                segments = []
                first = await interaction.turn(task.data.turns[0])
                segments.append(first)
                if not first.terminated:
                    segments.append(await interaction.turn(task.data.turns[1]))
            else:
                segments = [await interaction.turn()]
            interaction.trace.info["prime_agent_foundation_segments"] = [
                _segment_info(segment) for segment in segments
            ]


class PrimeAgentFoundationsConfig(vf.TasksetConfig):
    families: tuple[Family, ...] = Field(FAMILIES, min_length=1)
    instances_per_family: int = Field(2, ge=1)
    instance_offset: int = Field(0, ge=0)


def _task_data(family: Family, instance: int, idx: int) -> PrimeAgentFoundationData:
    system_prompt = "Follow the request exactly and use Prime Agent's native capabilities."
    values = {}
    if family == "ipython_cell":
        sentinel = f"prime-agent-foundation-{instance}"
        cell = f"print({sentinel!r})"
        prompt = f"Use IPython to execute exactly this cell:\n\n{cell}\n\nThen reply with exactly DONE."
        turns = (prompt,)
        values["sentinel"] = sentinel
    elif family == "kernel_persistence":
        marker_name = f"_prime_foundation_marker_{instance}"
        first_cell = f"import secrets\n{marker_name} = secrets.token_hex(32)\nprint({marker_name})"
        second_cell = f"print({marker_name})"
        turns = (
            f"Use IPython to execute exactly this cell:\n\n{first_cell}\n\nThen reply with exactly READY.",
            (
                "Use IPython to execute exactly this cell without defining, assigning, or "
                f"reconstructing anything:\n\n{second_cell}\n\nThen reply with exactly the printed value."
            ),
        )
        prompt = None
        values["marker_name"] = marker_name
    elif family == "conversation_resume":
        codeword = f"cedar-lantern-{instance:04d}"
        turns = (
            f"Remember the codeword {codeword}. Do not call IPython. Reply with exactly READY.",
            "Without calling IPython, reply with exactly the codeword from my previous message.",
        )
        prompt = None
        values["codeword"] = codeword
    elif family == "child_result_delivery":
        child_name = f"result-worker-{instance}"
        value = 37 + instance
        child_result = str(value * (value + 1))
        child_prompt = (
            f"Compute {value} * {value + 1}. Then use IPython to execute exactly "
            f"await agent_message.send({child_result!r}, receiver_role='parent'), and stop."
        )
        prompt = (
            "Use IPython to assign the result of exactly one rlm() call to a variable. "
            f"Name the child {child_name!r} and ask it this:\n\n{child_prompt}\n\n"
            "Do not poll. Wait for its explicit message, then reply with exactly DONE."
        )
        turns = (prompt,)
        values.update(child_name=child_name, child_result=child_result)
    else:
        child_name = f"cancel-worker-{instance}"
        started_message = f"STARTED-{instance}"
        finished_message = f"FINISHED-{instance}"
        started_path = f"/tmp/prime-agent-foundation-started-{instance}"
        finished_path = f"/tmp/prime-agent-foundation-finished-{instance}"
        child_prompt = (
            "Use IPython to execute exactly this code in one cell, then stop:\n\n"
            "from pathlib import Path\nimport asyncio\n"
            f"Path({started_path!r}).write_text('started')\n"
            f"await agent_message.send({started_message!r}, receiver_role='parent')\n"
            "await asyncio.sleep(120)\n"
            f"Path({finished_path!r}).write_text('finished')\n"
            f"await agent_message.send({finished_message!r}, receiver_role='parent')"
        )
        prompt = (
            f"Use IPython to execute child = await rlm({child_prompt!r}, name={child_name!r}). "
            f"Do not poll. After the child explicitly sends {started_message!r}, use IPython "
            "to execute exactly await rlm.delete_subagent(child.name). Then reply with exactly DONE."
        )
        turns = (prompt,)
        values.update(
            child_name=child_name,
            started_message=started_message,
            finished_message=finished_message,
            started_path=started_path,
            finished_path=finished_path,
        )
    return PrimeAgentFoundationData(
        idx=idx,
        name=f"{family}-i{instance}",
        prompt=prompt,
        system_prompt=system_prompt,
        family=family,
        instance=instance,
        turns=turns,
        **values,
    )


class PrimeAgentFoundationsTaskset(
    vf.Taskset[PrimeAgentFoundationsTask, PrimeAgentFoundationsConfig]
):
    def load(self) -> list[PrimeAgentFoundationsTask]:
        tasks = []
        for instance in range(
            self.config.instance_offset,
            self.config.instance_offset + self.config.instances_per_family,
        ):
            for family in self.config.families:
                tasks.append(
                    PrimeAgentFoundationsTask(
                        _task_data(family, instance, len(tasks)),
                        self.config.task,
                    )
                )
        return tasks

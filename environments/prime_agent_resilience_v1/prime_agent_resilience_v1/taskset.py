"""Observable fault-recovery tasks for the official Prime Agent harness."""

from __future__ import annotations

import ast
import json
import random
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.types import AssistantMessage, ToolMessage, UserMessage, content_text

Family = Literal[
    "malformed_result_repair",
    "delayed_result",
    "message_type_repair",
]
Split = Literal["calibration", "heldout"]
FAMILIES: tuple[Family, ...] = (
    "malformed_result_repair",
    "delayed_result",
    "message_type_repair",
)
VARIANTS: dict[Split, tuple[int, ...]] = {
    "calibration": (0, 1),
    "heldout": (2, 3),
}
DELAY_SECONDS = 2.0
MIN_OBSERVED_DELAY_SECONDS = 1.0

SYSTEM_PROMPT = (
    "Use Prime Agent's persistent IPython kernel and native RLM messaging. Retain successful "
    "state and child handles across turns. Treat child messages and tool outputs as evidence: "
    "inspect their actual type or error before changing only the failed operation. Never repeat "
    "an unchanged failed cell, replace a live child, poll a roster, or use agent_observe as the "
    "completion channel. Yield while asynchronous work is pending and return only the requested "
    "bare JSON object after valid evidence arrives."
)


class PrimeAgentResilienceData(vf.TaskData):
    family: Family
    split: Split
    variant: int
    child_name: str
    resource_path: str
    files: dict[str, str]
    answer: dict[str, int]
    malformed_message: str | None = None
    correction_message: str | None = None
    delayed_script_path: str | None = None


@dataclass(frozen=True)
class IpythonEvent:
    code: str
    output: str
    node_index: int
    child_branch: bool


def _call_code(arguments: str) -> str | None:
    try:
        value = json.loads(arguments)
    except (TypeError, json.JSONDecodeError):
        return None
    code = value.get("code") if isinstance(value, dict) else None
    return code if isinstance(code, str) else None


def _branch_root(trace: vf.Trace, node_index: int) -> int:
    visited = set()
    while trace.nodes[node_index].parent is not None and node_index not in visited:
        visited.add(node_index)
        node_index = trace.nodes[node_index].parent
    return node_index


def _is_child_root(trace: vf.Trace, node_index: int) -> bool:
    root = trace.nodes[_branch_root(trace, node_index)].message
    return isinstance(root, UserMessage) and content_text(
        root.content
    ).lstrip().startswith("[task from parent]")


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
            events.append(
                IpythonEvent(
                    code, output, node_index, _is_child_root(trace, node_index)
                )
            )
    return events


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        return f"{call.func.value.id}.{call.func.attr}"
    return None


def _parsed_calls(event: IpythonEvent) -> list[tuple[ast.Call, bool]]:
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
        (node, id(node) in assigned)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]


def _failed(event: IpythonEvent) -> bool:
    return any(
        marker in event.output
        for marker in ("Traceback", "Error:", "TypeError", "SyntaxError")
    )


def _successful_send(event: IpythonEvent) -> bool:
    return not _failed(event) and (
        "agentmsg_" in event.output
        or "Agent message sent" in event.output
        or "Agent message queued" in event.output
    )


def _has_call(event: IpythonEvent, name: str) -> bool:
    return any(_call_name(call) == name for call, _ in _parsed_calls(event))


def _constant_send_body(event: IpythonEvent) -> str | None:
    for call, _ in _parsed_calls(event):
        if _call_name(call) != "agent_message.send" or not call.args:
            continue
        value = call.args[0]
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return None


def _incoming_child_messages(
    trace: vf.Trace, child_name: str
) -> list[tuple[int, float, str]]:
    messages = []
    prefix = f"[from child:{child_name}]"
    for node_index, node in enumerate(trace.nodes):
        if not isinstance(node.message, UserMessage):
            continue
        text = content_text(node.message.content)
        if text.lstrip().startswith(prefix):
            messages.append(
                (node_index, node.timestamp, text.rsplit("\n\n", 1)[-1].strip())
            )
    return messages


def _json_answer(text: str, expected: dict[str, int]) -> bool:
    try:
        value = json.loads(text.strip())
    except (AttributeError, json.JSONDecodeError):
        return False
    return value == expected


def _spawn_protocol(
    events: list[IpythonEvent], data: PrimeAgentResilienceData
) -> tuple[bool, int]:
    spawns = []
    for event in events:
        for call, assigned in _parsed_calls(event):
            if _call_name(call) == "rlm":
                spawns.append((event, call, assigned))
    if len(spawns) != 1:
        return False, len(spawns)
    event, call, assigned = spawns[0]
    name = next(
        (keyword.value for keyword in call.keywords if keyword.arg == "name"), None
    )
    named = isinstance(name, ast.Constant) and name.value == data.child_name
    prompt = ast.unparse(call.args[0]) if call.args else ""
    return bool(
        assigned and named and data.resource_path in prompt and not _failed(event)
    ), 1


def _duplicate_cells(events: list[IpythonEvent]) -> int:
    return sum(
        count - 1
        for count in Counter(event.code.strip() for event in events).values()
        if count > 1
    )


def _behavior(trace: vf.Trace, data: PrimeAgentResilienceData) -> dict[str, float]:
    events = _ipython_events(trace)
    coordinator = [event for event in events if not event.child_branch]
    child = [event for event in events if event.child_branch]
    messages = _incoming_child_messages(trace, data.child_name)
    spawn_ok, spawn_calls = _spawn_protocol(coordinator, data)
    poll_calls = sum(
        any(
            token in event.code
            for token in (
                "agent_observe",
                "list_subagents",
                "list_agents",
                "list_messages",
            )
        )
        for event in coordinator
    )
    duplicate_cells = _duplicate_cells(events)
    answer_ok = _json_answer(trace.last_reply or "", data.answer)

    repair_observed = False
    message_order = False
    bounded_wait = False
    changed_action = False
    if data.family == "malformed_result_repair":
        if (
            len(messages) == 2
            and data.malformed_message is not None
            and data.correction_message is not None
        ):
            first_index, _, first_body = messages[0]
            second_index, _, second_body = messages[1]
            between_messages = [
                event
                for event in coordinator
                if first_index < event.node_index < second_index
            ]
            corrections = [
                event
                for event in between_messages
                if _successful_send(event)
                and _constant_send_body(event) == data.correction_message
            ]
            repair_observed = (
                first_body == data.malformed_message
                and len(corrections) == 1
                and between_messages == corrections
            )
            message_order = repair_observed and _json_answer(second_body, data.answer)
            changed_action = (
                len([event for event in child if _successful_send(event)]) == 2
            )
            bounded_wait = between_messages == corrections
    elif data.family == "delayed_result":
        if len(messages) == 2:
            first_index, first_time, first_body = messages[0]
            second_index, second_time, second_body = messages[1]
            message_order = first_body == "STARTED" and _json_answer(
                second_body, data.answer
            )
            bounded_wait = (
                second_time - first_time >= MIN_OBSERVED_DELAY_SECONDS
                and not any(
                    first_index < event.node_index < second_index
                    for event in coordinator
                )
            )
            repair_observed = any(
                data.delayed_script_path in event.code
                for event in child
                if data.delayed_script_path is not None
            )
            changed_action = (
                len([event for event in child if _successful_send(event)]) == 2
            )
    else:
        failed_sends = [
            event
            for event in child
            if _has_call(event, "agent_message.send") and _failed(event)
        ]
        successful_sends = [
            event
            for event in child
            if _has_call(event, "agent_message.send") and _successful_send(event)
        ]
        if len(failed_sends) == 1 and len(successful_sends) == 1:
            failed, succeeded = failed_sends[0], successful_sends[0]
            repair_observed = (
                "TypeError" in failed.output and "string" in failed.output.lower()
            )
            changed_action = (
                failed.code.strip() != succeeded.code.strip()
                and failed.node_index < succeeded.node_index
            )
        message_order = len(messages) == 1 and _json_answer(messages[0][2], data.answer)
        if messages:
            bounded_wait = not coordinator or not any(
                event.node_index > messages[0][0] for event in coordinator
            )

    components = {
        "answer_accuracy": float(answer_ok),
        "spawn_protocol": float(spawn_ok),
        "message_order": float(message_order),
        "repair_observed": float(repair_observed),
        "changed_action": float(changed_action),
        "bounded_wait": float(bounded_wait),
        "no_polling": float(poll_calls == 0),
        "no_duplicate_cells": float(duplicate_cells == 0),
        "spawn_calls": float(spawn_calls),
        "child_messages": float(len(messages)),
        "failed_cells": float(sum(_failed(event) for event in events)),
    }
    required = (
        "answer_accuracy",
        "spawn_protocol",
        "message_order",
        "repair_observed",
        "changed_action",
        "bounded_wait",
        "no_polling",
        "no_duplicate_cells",
    )
    components["strict_success"] = float(
        all(components[key] == 1.0 for key in required)
    )
    return components


class PrimeAgentResilienceTask(vf.Task[PrimeAgentResilienceData]):
    async def setup(self, runtime: vf.Runtime) -> None:
        directories = sorted({path.rsplit("/", 1)[0] for path in self.data.files})
        created = await runtime.run(["mkdir", "-p", *directories], {})
        if created.exit_code != 0:
            raise RuntimeError(f"resilience setup failed: {created.stderr[-500:]}")
        for path, contents in self.data.files.items():
            await runtime.write(path, contents.encode())

    @vf.reward(weight=1.0)
    async def resilience(self, trace: vf.Trace) -> float:
        return _behavior(trace, self.data)["strict_success"]

    @vf.metric
    async def resilience_behavior(self, trace: vf.Trace) -> dict[str, float]:
        return _behavior(trace, self.data)


class PrimeAgentResilienceEnv(vf.SingleAgentEnv):
    pass


class PrimeAgentResilienceConfig(vf.TasksetConfig):
    split: Split = "heldout"
    families: tuple[Family, ...] = Field(FAMILIES, min_length=1)
    instances_per_family: int = Field(1, ge=1)
    instance_offset: int = Field(0, ge=0)
    seed: int = 20260815


def _values(rng: random.Random, count: int) -> list[int]:
    return [rng.randint(-29, 37) for _ in range(count)]


def _checksum(values: list[int]) -> int:
    return sum((index + 1) * value for index, value in enumerate(values))


def _task_data(
    family: Family, split: Split, variant: int, instance: int, idx: int, seed: int
) -> PrimeAgentResilienceData:
    rng = random.Random(seed * 1_000_003 + variant * 10_007 + instance * 101)
    values = _values(rng, 9 + variant)
    answer = {"checksum": _checksum(values)}
    child_name = {
        "malformed_result_repair": "schema-worker",
        "delayed_result": "delayed-worker",
        "message_type_repair": "message-worker",
    }[family]
    root = f"/workspace/prime-agent-resilience/{split}-v{variant}-i{instance}"
    resource_path = f"{root}/values.json"
    files = {resource_path: json.dumps(values)}
    malformed_message = None
    correction_message = None
    delayed_script_path = None

    if family == "malformed_result_repair":
        malformed_message = '{"checksum":"invalid"}'
        correction_message = "RETRY: return JSON with integer checksum"
        protocol = (
            f"Ask {child_name} to read {resource_path}, retain the computed weighted checksum, and first send "
            f"exactly {malformed_message!r} to its parent. When it later receives exactly {correction_message!r}, "
            "it must reuse the retained checksum and send JSON whose checksum value is an integer. Validate the "
            "first child message, send that one correction to the retained child, yield, and accept only the "
            "schema-valid second result."
        )
    elif family == "delayed_result":
        delayed_script_path = f"{root}/delayed_result.py"
        files[delayed_script_path] = (
            "import json\n"
            "import time\n"
            f"values = {values!r}\n"
            f"time.sleep({DELAY_SECONDS!r})\n"
            "checksum = sum((index + 1) * value for index, value in enumerate(values))\n"
            "print(json.dumps({'checksum': checksum}))\n"
        )
        protocol = (
            f"Ask {child_name} to send exactly 'STARTED' to its parent, then execute "
            f"{delayed_script_path} as a subprocess, require returncode 0, and send its stdout unchanged. "
            "After STARTED, preserve the child and yield without polling, replacement, or local recomputation. "
            "Return only after the later JSON result arrives."
        )
    else:
        protocol = (
            f"Ask {child_name} to read {resource_path}, compute and retain payload={{'checksum': checksum}}, then "
            "first call await agent_message.send(payload, receiver_role='parent'). It must inspect the real "
            "TypeError, change only the failed operation, preserve payload, and retry with json.dumps(payload). "
            "Do not replace the child or reproduce its work; return only its explicit JSON message."
        )

    prompt = (
        "Return one bare JSON object with exactly one integer key, checksum. Delegate this task to exactly one "
        f"child named {child_name}. {protocol}"
    )
    return PrimeAgentResilienceData(
        idx=idx,
        name=f"{family}-v{variant}-i{instance}",
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        family=family,
        split=split,
        variant=variant,
        child_name=child_name,
        resource_path=resource_path,
        files=files,
        answer=answer,
        malformed_message=malformed_message,
        correction_message=correction_message,
        delayed_script_path=delayed_script_path,
    )


class PrimeAgentResilienceTaskset(
    vf.Taskset[PrimeAgentResilienceTask, PrimeAgentResilienceConfig]
):
    def load(self) -> list[PrimeAgentResilienceTask]:
        tasks = []
        for instance in range(
            self.config.instance_offset,
            self.config.instance_offset + self.config.instances_per_family,
        ):
            for variant in VARIANTS[self.config.split]:
                for family in self.config.families:
                    data = _task_data(
                        family,
                        self.config.split,
                        variant,
                        instance,
                        len(tasks),
                        self.config.seed,
                    )
                    tasks.append(PrimeAgentResilienceTask(data))
        return tasks

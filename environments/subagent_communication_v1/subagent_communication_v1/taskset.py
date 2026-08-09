"""Native Prime Agent depth-one delegation and messaging tasks."""

from __future__ import annotations

import ast
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

import verifiers.v1 as vf
from pydantic import Field
from verifiers.v1.types import AssistantMessage, ToolMessage, UserMessage, content_text

Family = Literal["direct", "single", "parallel", "followup"]
InstructionLevel = Literal["standard", "guided"]

FAMILIES: tuple[Family, ...] = ("direct", "single", "parallel", "followup")
TRAIN_VARIANTS = (0, 1, 2, 3)
EVAL_VARIANTS = (4, 5)

SYSTEM_PROMPT = (
    "Coordinate work through Prime Agent's persistent IPython kernel. Delegate only the "
    "shards the request assigns to children. The callable rlm returns an admission handle, "
    "not a result. Spawn independent children before doing coordinator-local work, retain their "
    "handles, and use that local work to let children run concurrently. A child must send "
    "requested results with agent_message to its parent. Never guess a missing result or finalize "
    "before every required reply arrives. When local work finishes first, use bounded "
    "agent_observe polling until the retained child is no longer streaming. "
    "Preserve successful state across turns, do not repeat unchanged cells, and return the "
    "requested JSON object only."
)

class SubagentCommunicationData(vf.TaskData):
    family: Family
    template_variant: int
    instruction_level: InstructionLevel = "standard"
    answer: dict[str, int]
    expected_children: tuple[str, ...] = ()
    child_paths: dict[str, str] = Field(default_factory=dict)
    files: dict[str, str] = Field(default_factory=dict)
    followup_secret: int | None = None


def _weighted(values: list[int]) -> int:
    return sum((index + 1) * value for index, value in enumerate(values))


def _values(rng: random.Random, count: int) -> list[int]:
    return [rng.randint(-19, 29) for _ in range(count)]


def _json(values: list[int]) -> str:
    return json.dumps(values, separators=(",", ":"))


def _task_prompt(
    family: Family,
    variant: int,
    instance: int,
    seed: int,
    instruction_level: InstructionLevel,
) -> tuple[
    str,
    dict[str, int],
    tuple[str, ...],
    dict[str, str],
    dict[str, str],
    int | None,
]:
    rng = random.Random(seed * 1_000_003 + variant * 10_007 + instance * 101)
    prefix = (
        "Return one JSON object with exactly the requested keys and integer values. "
        "A shard checksum is sum((index + 1) * value)."
    )

    if family == "direct":
        values = _values(rng, 8 + variant % 3)
        answer = {"checksum": _weighted(values)}
        request = (
            f"This small task must stay in the coordinator. Compute the checksum of {_json(values)} "
            'without creating a subagent. Return {"checksum": value}.'
        )
        children: tuple[str, ...] = ()
        child_paths: dict[str, str] = {}
        files: dict[str, str] = {}
        secret = None
        guidance = "Solve locally and do not spawn or message a child."
    elif family == "single":
        local = _values(rng, 7 + variant % 2)
        remote = _values(rng, 13 + variant % 3)
        remote_path = f"/workspace/subagent-shards/v{variant}-i{instance}-remote.json"
        local_value = _weighted(local)
        remote_value = _weighted(remote)
        answer = {
            "local": local_value,
            "remote": remote_value,
            "total": local_value + remote_value,
        }
        request = (
            f"Compute the local checksum for {_json(local)} yourself. Delegate the independent "
            f"remote checksum in {remote_path} to one child named shard-worker and obtain its "
            'explicit reply. Return {"local": value, "remote": value, "total": value}.'
        )
        children = ("shard-worker",)
        child_paths = {"shard-worker": remote_path}
        files = {remote_path: json.dumps(remote)}
        secret = None
        guidance = (
            f"Do not open {remote_path} in the coordinator. Spawn the child before computing the "
            f"local checksum, using handle = await rlm(\"Read {remote_path}, compute its weighted "
            "checksum, then send the integer checksum to your parent with agent_message before "
            "answering.\", name='shard-worker'). Retain handle and compute the local checksum in "
            "separate state-preserving IPython calls while the child runs. If its reply has not "
            "arrived, use a bounded asyncio loop around await "
            "agent_observe.get_agent(handle.name) until the returned agent isStreaming field is "
            "false. Never finalize without the explicit reply; the admission handle is not the "
            "answer."
        )
    elif family == "parallel":
        local = _values(rng, 6)
        alpha = _values(rng, 11 + variant % 2)
        beta = _values(rng, 12 + (variant + 1) % 2)
        alpha_path = f"/workspace/subagent-shards/v{variant}-i{instance}-alpha.json"
        beta_path = f"/workspace/subagent-shards/v{variant}-i{instance}-beta.json"
        local_value = _weighted(local)
        alpha_value = _weighted(alpha)
        beta_value = _weighted(beta)
        answer = {
            "local": local_value,
            "alpha": alpha_value,
            "beta": beta_value,
            "total": local_value + alpha_value + beta_value,
        }
        request = (
            f"Compute the local checksum for {_json(local)} yourself. In parallel, delegate "
            f"{alpha_path} to alpha-worker and {beta_path} to beta-worker. Each child must "
            'reply explicitly. Return {"local": value, "alpha": value, "beta": value, '
            '"total": value}.'
        )
        children = ("alpha-worker", "beta-worker")
        child_paths = {"alpha-worker": alpha_path, "beta-worker": beta_path}
        files = {alpha_path: json.dumps(alpha), beta_path: json.dumps(beta)}
        secret = None
        guidance = (
            f"Do not open {alpha_path} or {beta_path} in the coordinator. In separate IPython "
            "calls, assign alpha = await rlm(\"Read "
            f"{alpha_path}, compute its weighted checksum, then make a separate final IPython "
            "call exactly await agent_message.send(str(checksum), receiver_role='parent'). Your "
            "task is incomplete until that call succeeds.\", "
            "name='alpha-worker') and beta = await rlm(\"Read "
            f"{beta_path}, compute its weighted checksum, then make a separate final IPython call "
            "exactly await agent_message.send(str(checksum), receiver_role='parent'). Your task "
            "is incomplete until that call succeeds.\", "
            "name='beta-worker'). Retain both handles, then stop calling tools for this turn."
        )
    else:
        remote = _values(rng, 10 + variant % 3)
        remote_path = f"/workspace/subagent-shards/v{variant}-i{instance}-followup.json"
        secret = rng.randint(31, 47)
        subtotal = sum(remote)
        answer = {"subtotal": subtotal, "multiplier": secret, "result": subtotal * secret}
        request = (
            f"Delegate {remote_path} to key-worker. The child should sum the values, but do not "
            f"include the multiplier {secret} in its spawn prompt. It must first request the missing "
            "multiplier from you through an agent message. Send it only in a direct follow-up, wait "
            'for the child\'s final result message, then return {"subtotal": value, '
            '"multiplier": value, "result": value}.'
        )
        children = ("key-worker",)
        child_paths = {"key-worker": remote_path}
        files = {remote_path: json.dumps(remote)}
        guidance = (
            f"Do not open {remote_path} in the coordinator. Assign child = await rlm(\"Read "
            f"{remote_path} and compute the subtotal. In a separate IPython call exactly await "
            "agent_message.send('need multiplier', receiver_role='parent'). Do not finish: after "
            "the parent reply, compute the result and make a second successful parent message "
            "containing the subtotal and result. Your task is incomplete until both messages "
            "succeed.\", "
            "name='key-worker'). Do not put the multiplier in that string. Retain child and stop "
            "calling tools for this turn. After its request arrives, use await "
            "agent_message.send(str(multiplier), receiver_role='child', "
            "receiver_name=child.name), then wait for its final reply."
        )

    if instruction_level == "guided":
        request = f"{request}\n\nProtocol hint: {guidance}"
    return (
        f"{prefix}\n\n{request}",
        answer,
        children,
        child_paths,
        files,
        secret,
    )


@dataclass
class IpythonEvent:
    code: str
    call_id: str
    output: str = ""


def _ipython_events(trace: vf.Trace) -> list[IpythonEvent]:
    events: list[IpythonEvent] = []
    by_call_id: dict[str, IpythonEvent] = {}
    for node in trace.nodes:
        message = node.message
        if isinstance(message, AssistantMessage):
            for call in message.tool_calls or []:
                if call.name != "ipython":
                    continue
                try:
                    arguments = json.loads(call.arguments)
                except json.JSONDecodeError:
                    continue
                source = arguments.get("code")
                if isinstance(source, str):
                    event = IpythonEvent(code=source, call_id=call.id)
                    events.append(event)
                    by_call_id[call.id] = event
        elif isinstance(message, ToolMessage) and (event := by_call_id.get(message.tool_call_id)):
            event.output = content_text(message.content)
    return events


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        return f"{call.func.value.id}.{call.func.attr}"
    return None


def _keyword(call: ast.Call, name: str) -> Any:
    value = next((item.value for item in call.keywords if item.arg == name), None)
    return value.value if isinstance(value, ast.Constant) else None


def _assigned_call_names(tree: ast.AST) -> set[int]:
    assigned: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if isinstance(value, ast.Await):
            value = value.value
        if isinstance(value, ast.Call):
            assigned.add(id(value))
    return assigned


def _failed(output: str) -> bool:
    return any(marker in output for marker in ("Traceback", "Error:", "SyntaxError"))


def _message_sent(output: str) -> bool:
    return not _failed(output) and (
        "agentmsg_" in output or "Agent message sent" in output or "Agent message queued" in output
    )


def _incoming_child_messages(trace: vf.Trace) -> list[tuple[str, str]]:
    messages: list[tuple[str, str]] = []
    for node in trace.nodes:
        message = node.message
        if not isinstance(message, UserMessage):
            continue
        text = content_text(message.content)
        matched = re.match(
            r"\[from child:([^\]]+)\]\s*\nAgent-to-agent message received\.",
            text,
        )
        if not matched:
            continue
        body = text.rsplit("\n\n", 1)[-1].strip()
        if "completed without sending a reply" in body or body.startswith("RLM child failure"):
            continue
        messages.append((matched.group(1), body))
    return messages


def _spawn_name(call: ast.Call, output: str) -> str | None:
    configured = _keyword(call, "name")
    if isinstance(configured, str):
        return configured
    matched = re.search(r"\bname='([^']+)'", output)
    return matched.group(1) if matched else None


def _spawn_prompt(call: ast.Call) -> str | None:
    if not call.args or not isinstance(call.args[0], ast.Constant):
        return None
    value = call.args[0].value
    return value if isinstance(value, str) else None


def _protocol_behavior(
    trace: vf.Trace,
    family: Family,
    expected_children: tuple[str, ...],
    child_paths: dict[str, str],
    followup_secret: int | None,
) -> dict[str, float]:
    events = _ipython_events(trace)
    code = [event.code for event in events]
    calls: list[tuple[ast.Call, bool, str]] = []
    for event in events:
        try:
            tree = ast.parse(event.code)
        except SyntaxError:
            continue
        assigned = _assigned_call_names(tree)
        calls.extend(
            (node, id(node) in assigned, event.output) for node in ast.walk(tree) if isinstance(node, ast.Call)
        )

    attempted_spawns = [(call, retained, output) for call, retained, output in calls if _call_name(call) == "rlm"]
    spawns = [item for item in attempted_spawns if not _failed(item[2])]
    names = {_spawn_name(call, output) for call, _, output in spawns}
    parent_messages = _incoming_child_messages(trace)
    parent_message_names = {name for name, _ in parent_messages}
    child_messages = [
        call
        for call, _, output in calls
        if _call_name(call) == "agent_message.send"
        and _keyword(call, "receiver_role") == "child"
        and _message_sent(output)
    ]
    list_calls = sum(
        _call_name(call) in {"rlm.list_subagents", "agent_message.list_agents"} and not _failed(output)
        for call, _, output in calls
    )
    observation_calls = sum(
        (_call_name(call) or "").startswith("agent_observe.") and not _failed(output)
        for call, _, output in calls
    )
    normalized = [source.strip() for source in code]
    repeated = sum(count - 1 for count in Counter(normalized).values() if count > 1)
    retained = sum(retained for _, retained, _ in spawns)
    delegated = {
        name
        for name, path in child_paths.items()
        if any(_spawn_name(call, output) == name and path in (_spawn_prompt(call) or "") for call, _, output in spawns)
    }
    secret_withheld = True
    if followup_secret is not None and spawns:
        first_prompt = _spawn_prompt(spawns[0][0])
        secret_withheld = bool(first_prompt and str(followup_secret) not in first_prompt)

    if family == "direct":
        checks = [not spawns, not parent_messages, not child_messages, repeated == 0]
    elif family == "single":
        checks = [
            len(spawns) == 1,
            set(expected_children) <= names,
            retained == 1,
            set(expected_children) <= delegated,
            set(expected_children) <= parent_message_names,
            repeated == 0,
        ]
    elif family == "parallel":
        checks = [
            len(spawns) == 2,
            set(expected_children) <= names,
            retained == 2,
            set(expected_children) <= delegated,
            set(expected_children) <= parent_message_names,
            repeated == 0,
        ]
    else:
        checks = [
            len(spawns) == 1,
            set(expected_children) <= names,
            retained == 1,
            set(expected_children) <= delegated,
            secret_withheld,
            len(child_messages) >= 1,
            sum(name in expected_children for name, _ in parent_messages) >= 2,
            repeated == 0,
        ]
    return {
        "protocol_score": sum(checks) / len(checks),
        "protocol_aligned": float(all(checks)),
        "spawn_calls": float(len(spawns)),
        "failed_spawn_calls": float(len(attempted_spawns) - len(spawns)),
        "retained_handles": float(retained),
        "named_children": float(len(set(expected_children) & names)),
        "delegated_payloads": float(len(delegated)),
        "messages_to_parent": float(len(parent_messages)),
        "messages_to_child": float(len(child_messages)),
        "roster_calls": float(list_calls),
        "observation_calls": float(observation_calls),
        "secret_withheld": float(secret_withheld),
        "duplicate_cells": float(repeated),
    }


def _answer_score(reply: str, expected: dict[str, int]) -> float:
    try:
        actual = json.loads(reply.strip())
    except (AttributeError, json.JSONDecodeError):
        return 0.0
    if not isinstance(actual, dict):
        return 0.0
    return sum(actual.get(key) == value for key, value in expected.items()) / len(expected)


class SubagentCommunicationTask(vf.Task[SubagentCommunicationData]):
    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        if not self.data.files:
            return
        directories = sorted({path.rsplit("/", 1)[0] for path in self.data.files})
        created = await runtime.run(["mkdir", "-p", *directories], {})
        if created.exit_code != 0:
            raise RuntimeError(f"subagent shard setup failed: {created.stderr[-500:]}")
        for path, contents in self.data.files.items():
            await runtime.write(path, contents.encode())

    @vf.reward(weight=1.0)
    async def protocol_gated_accuracy(self, trace: vf.Trace) -> float:
        accuracy = _answer_score(trace.last_reply, self.data.answer)
        if self.data.family == "direct":
            return accuracy
        behavior = _protocol_behavior(
            trace,
            self.data.family,
            self.data.expected_children,
            self.data.child_paths,
            self.data.followup_secret,
        )
        return accuracy * behavior["protocol_aligned"]

    @vf.reward(weight=1.0)
    async def delegation_protocol(self, trace: vf.Trace) -> float:
        return _protocol_behavior(
            trace,
            self.data.family,
            self.data.expected_children,
            self.data.child_paths,
            self.data.followup_secret,
        )["protocol_score"]

    @vf.metric
    async def answer_accuracy(self, trace: vf.Trace) -> float:
        return _answer_score(trace.last_reply, self.data.answer)

    @vf.metric
    async def delegation_behavior(self, trace: vf.Trace) -> dict[str, float]:
        return _protocol_behavior(
            trace,
            self.data.family,
            self.data.expected_children,
            self.data.child_paths,
            self.data.followup_secret,
        )


class SubagentCommunicationConfig(vf.TasksetConfig):
    split: Literal["train", "eval"] = "train"
    families: tuple[Family, ...] = Field(FAMILIES, min_length=1)
    instruction_level: InstructionLevel = "standard"
    instances_per_template: int = Field(4, ge=1)
    instance_offset: int = Field(0, ge=0)
    seed: int = 20260809


class SubagentCommunicationTaskset(vf.Taskset[SubagentCommunicationTask, SubagentCommunicationConfig]):
    def load(self) -> list[SubagentCommunicationTask]:
        variants = TRAIN_VARIANTS if self.config.split == "train" else EVAL_VARIANTS
        tasks = []
        instances = range(
            self.config.instance_offset,
            self.config.instance_offset + self.config.instances_per_template,
        )
        for instance in instances:
            for variant in variants:
                for family in self.config.families:
                    prompt, answer, children, child_paths, files, secret = _task_prompt(
                        family,
                        variant,
                        instance,
                        self.config.seed,
                        self.config.instruction_level,
                    )
                    tasks.append(
                        SubagentCommunicationTask(
                            SubagentCommunicationData(
                                idx=len(tasks),
                                name=f"{family}-v{variant}-i{instance}",
                                prompt=prompt,
                                system_prompt=SYSTEM_PROMPT,
                                family=family,
                                template_variant=variant,
                                instruction_level=self.config.instruction_level,
                                answer=answer,
                                expected_children=children,
                                child_paths=child_paths,
                                files=files,
                                followup_secret=secret,
                            ),
                            self.config.task,
                        )
                    )
        return tasks

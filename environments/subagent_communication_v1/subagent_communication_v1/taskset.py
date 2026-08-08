"""Native Prime Agent depth-one delegation and messaging tasks."""

from __future__ import annotations

import ast
import json
import random
from itertools import pairwise
from typing import Any, Literal

import verifiers.v1 as vf
from pydantic import Field
from verifiers.v1.types import AssistantMessage

Family = Literal["direct", "single", "parallel", "followup"]
InstructionLevel = Literal["standard", "guided"]

FAMILIES: tuple[Family, ...] = ("direct", "single", "parallel", "followup")
TRAIN_VARIANTS = (0, 1, 2, 3)
EVAL_VARIANTS = (4, 5)

SYSTEM_PROMPT = (
    "Coordinate work through Prime Agent's persistent IPython kernel. Delegate only the "
    "shards the request assigns to children. The callable rlm returns an admission handle, "
    "not a result; retain that handle and end the turn so child messages can arrive. A child "
    "must send requested results with agent_message to its parent. Preserve successful state "
    "across turns, do not repeat unchanged cells, and return the requested JSON object only."
)

GUIDANCE = {
    "direct": ("This is a restraint exercise: solve locally and do not spawn or message a child."),
    "single": (
        "Spawn with handle = await rlm(prompt, name='shard-worker'). In the child prompt, "
        "request an explicit agent_message reply to the parent. The handle is not the answer."
    ),
    "parallel": (
        "Spawn alpha-worker and beta-worker independently in separate IPython calls, retain "
        "both handles, then end the turn. Each child must message its result to the parent."
    ),
    "followup": (
        "Spawn key-worker without the multiplier. Tell it to request the missing value through "
        "agent_message, then end the turn. Reply with agent_message.send to receiver_role='child' "
        "and receiver_name=handle.name; keep the child until its final parent message arrives."
    ),
}


class SubagentCommunicationData(vf.TaskData):
    family: Family
    template_variant: int
    instruction_level: InstructionLevel = "standard"
    answer: dict[str, int]
    expected_children: tuple[str, ...] = ()
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
) -> tuple[str, dict[str, int], tuple[str, ...], int | None]:
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
        secret = None
    elif family == "single":
        local = _values(rng, 7 + variant % 2)
        remote = _values(rng, 13 + variant % 3)
        local_value = _weighted(local)
        remote_value = _weighted(remote)
        answer = {
            "local": local_value,
            "remote": remote_value,
            "total": local_value + remote_value,
        }
        request = (
            f"Compute the local checksum for {_json(local)} yourself. Delegate the independent "
            f"remote checksum for {_json(remote)} to one child named shard-worker and obtain its "
            'explicit reply. Return {"local": value, "remote": value, "total": value}.'
        )
        children = ("shard-worker",)
        secret = None
    elif family == "parallel":
        local = _values(rng, 6)
        alpha = _values(rng, 11 + variant % 2)
        beta = _values(rng, 12 + (variant + 1) % 2)
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
            f"{_json(alpha)} to alpha-worker and {_json(beta)} to beta-worker. Each child must "
            'reply explicitly. Return {"local": value, "alpha": value, "beta": value, '
            '"total": value}.'
        )
        children = ("alpha-worker", "beta-worker")
        secret = None
    else:
        remote = _values(rng, 10 + variant % 3)
        secret = rng.randint(31, 47)
        subtotal = sum(remote)
        answer = {"subtotal": subtotal, "multiplier": secret, "result": subtotal * secret}
        request = (
            f"Delegate {_json(remote)} to key-worker. The child should sum the values, but do not "
            f"include the multiplier {secret} in its spawn prompt. It must first request the missing "
            "multiplier from you through an agent message. Send it only in a direct follow-up, wait "
            'for the child\'s final result message, then return {"subtotal": value, '
            '"multiplier": value, "result": value}.'
        )
        children = ("key-worker",)

    if instruction_level == "guided":
        request = f"{request}\n\nProtocol hint: {GUIDANCE[family]}"
    return f"{prefix}\n\n{request}", answer, children, secret


def _ipython_code(trace: vf.Trace) -> list[str]:
    code: list[str] = []
    for node in trace.nodes:
        message = node.message
        if not isinstance(message, AssistantMessage):
            continue
        for call in message.tool_calls or []:
            if call.name != "ipython":
                continue
            try:
                arguments = json.loads(call.arguments)
            except json.JSONDecodeError:
                continue
            source = arguments.get("code")
            if isinstance(source, str):
                code.append(source)
    return code


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


def _protocol_behavior(
    trace: vf.Trace,
    family: Family,
    expected_children: tuple[str, ...],
    followup_secret: int | None,
) -> dict[str, float]:
    code = _ipython_code(trace)
    calls: list[tuple[ast.Call, bool]] = []
    for source in code:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        assigned = _assigned_call_names(tree)
        calls.extend((node, id(node) in assigned) for node in ast.walk(tree) if isinstance(node, ast.Call))

    spawns = [(call, retained) for call, retained in calls if _call_name(call) == "rlm"]
    names = {_keyword(call, "name") for call, _ in spawns}
    parent_messages = [
        call
        for call, _ in calls
        if _call_name(call) == "agent_message.send" and _keyword(call, "receiver_role") == "parent"
    ]
    child_messages = [
        call
        for call, _ in calls
        if _call_name(call) == "agent_message.send" and _keyword(call, "receiver_role") == "child"
    ]
    list_calls = sum(_call_name(call) in {"rlm.list_subagents", "agent_message.list_agents"} for call, _ in calls)
    repeated = sum(left.strip() == right.strip() for left, right in pairwise(code))
    retained = sum(retained for _, retained in spawns)
    secret_withheld = True
    if followup_secret is not None and spawns:
        first_prompt = spawns[0][0].args[0] if spawns[0][0].args else None
        secret_withheld = bool(
            isinstance(first_prompt, ast.Constant)
            and isinstance(first_prompt.value, str)
            and str(followup_secret) not in first_prompt.value
        )

    if family == "direct":
        checks = [not spawns, not parent_messages, not child_messages, repeated == 0]
    elif family == "single":
        checks = [
            len(spawns) == 1,
            set(expected_children) <= names,
            retained == 1,
            len(parent_messages) >= 1,
            repeated == 0,
        ]
    elif family == "parallel":
        checks = [
            len(spawns) == 2,
            set(expected_children) <= names,
            retained == 2,
            len(parent_messages) >= 2,
            repeated == 0,
        ]
    else:
        checks = [
            len(spawns) == 1,
            set(expected_children) <= names,
            retained == 1,
            secret_withheld,
            len(child_messages) >= 1,
            len(parent_messages) >= 2,
            repeated == 0,
        ]
    return {
        "protocol_score": sum(checks) / len(checks),
        "protocol_aligned": float(all(checks)),
        "spawn_calls": float(len(spawns)),
        "retained_handles": float(retained),
        "named_children": float(len(set(expected_children) & names)),
        "messages_to_parent": float(len(parent_messages)),
        "messages_to_child": float(len(child_messages)),
        "roster_calls": float(list_calls),
        "secret_withheld": float(secret_withheld),
        "identical_consecutive_cells": float(repeated),
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
    @vf.reward(weight=1.0)
    async def task_accuracy(self, trace: vf.Trace) -> float:
        return _answer_score(trace.last_reply, self.data.answer)

    @vf.reward(weight=0.35)
    async def delegation_protocol(self, trace: vf.Trace) -> float:
        return _protocol_behavior(
            trace,
            self.data.family,
            self.data.expected_children,
            self.data.followup_secret,
        )["protocol_score"]

    @vf.metric
    async def delegation_behavior(self, trace: vf.Trace) -> dict[str, float]:
        return _protocol_behavior(
            trace,
            self.data.family,
            self.data.expected_children,
            self.data.followup_secret,
        )


class SubagentCommunicationConfig(vf.TasksetConfig):
    split: Literal["train", "eval"] = "train"
    families: tuple[Family, ...] = Field(FAMILIES, min_length=1)
    instruction_level: InstructionLevel = "standard"
    instances_per_template: int = Field(4, ge=1)
    seed: int = 20260809


class SubagentCommunicationTaskset(vf.Taskset[SubagentCommunicationTask, SubagentCommunicationConfig]):
    def load(self) -> list[SubagentCommunicationTask]:
        variants = TRAIN_VARIANTS if self.config.split == "train" else EVAL_VARIANTS
        tasks = []
        for instance in range(self.config.instances_per_template):
            for variant in variants:
                for family in self.config.families:
                    prompt, answer, children, secret = _task_prompt(
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
                                followup_secret=secret,
                            ),
                            self.config.task,
                        )
                    )
        return tasks

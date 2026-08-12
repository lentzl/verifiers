"""Executable capability tasks for the native Prime Agent harness."""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.types import content_text
from verifiers.v1.utils.prime_agent_metadata import (
    MissingAcpMeta,
    child_tokens_attributed,
    goal_progressed,
    no_outstanding_subagents,
    observed_child_statuses,
    refinement_applied,
    spawned_and_finished,
)

Family = Literal[
    "ipython_cell",
    "persistence",
    "subagent_lifecycle",
    "harness_state",
    "killed_child",
]
FAMILIES: tuple[Family, ...] = (
    "ipython_cell",
    "persistence",
    "subagent_lifecycle",
    "harness_state",
    "killed_child",
)
TOKEN = re.compile(r"\b[0-9a-f]{64}\b")


class PrimeAgentCapabilityData(vf.TaskData):
    family: Family
    instance: int
    turns: tuple[str, ...]
    sentinel: str | None = None
    marker_name: str | None = None
    child_result: int | None = None


def _segment_info(segment: vf.Segment) -> dict:
    return {
        "last_reply": segment.last_reply,
        "tool_calls": [
            {"name": call.name, "arguments": call.arguments}
            for message in segment.messages
            if isinstance(message, vf.AssistantMessage)
            for call in message.tool_calls or []
        ],
        "tool_outputs": [
            str(message.content)
            for message in segment.messages
            if isinstance(message, vf.ToolMessage)
        ],
        "terminated": segment.terminated,
    }


def _ipython_cell_completed(trace: vf.Trace, data: PrimeAgentCapabilityData) -> bool:
    segments = trace.info.get("prime_agent_segments", [])
    if len(segments) != 1 or data.sentinel is None:
        return False
    segment = segments[0]
    if segment["terminated"] or segment["last_reply"].strip() != "DONE":
        return False
    expected = {"code": f"print({data.sentinel!r})"}
    invoked = False
    for call in segment["tool_calls"]:
        try:
            arguments = json.loads(call["arguments"])
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        if call.get("name") == "ipython" and arguments == expected:
            invoked = True
            break
    return invoked and any(
        data.sentinel in output for output in segment["tool_outputs"]
    )


def _persistence_completed(trace: vf.Trace, data: PrimeAgentCapabilityData) -> bool:
    segments = trace.info.get("prime_agent_segments", [])
    if len(segments) != 2 or data.marker_name is None:
        return False
    first, second = segments
    first_tokens = set(TOKEN.findall("\n".join(first["tool_outputs"])))
    second_tokens = set(TOKEN.findall("\n".join(second["tool_outputs"])))
    second_calls = "\n".join(call["arguments"] for call in second["tool_calls"])
    return bool(
        first_tokens & second_tokens
        and data.marker_name in second_calls
        and "secrets" not in second_calls
        and "NameError" not in "\n".join(second["tool_outputs"])
        and first["last_reply"].strip() == "READY"
        and not first["terminated"]
        and not second["terminated"]
    )


def _child_cancelled(trace: vf.Trace) -> bool:
    return any(
        statuses[-1] in {"error", "cancelled"}
        for statuses in observed_child_statuses(trace).values()
    )


def _completed_exactly(trace: vf.Trace, expected: str = "DONE") -> bool:
    return (trace.last_reply or "").strip() == expected


def _subagent_protocol_completed(
    trace: vf.Trace, data: PrimeAgentCapabilityData
) -> bool:
    if data.child_result is None or not _completed_exactly(trace):
        return False
    received = False
    polled = False
    for node in trace.nodes:
        message = node.message
        text = content_text(message.content)
        if message.role == "user" and text.lstrip().startswith("[from child:"):
            received |= str(data.child_result) in text
        if message.role == "assistant":
            polled |= any(
                "agent_message.list_messages" in call.arguments
                for call in message.tool_calls or []
            )
    return received and not polled


def _harness_state_changed(trace: vf.Trace) -> bool:
    seen = []
    for guard in (refinement_applied, goal_progressed):
        try:
            seen.append(guard(trace))
        except MissingAcpMeta:
            seen.append(False)
    if not any(seen) and not ((trace.info or {}).get("acp_meta") or {}):
        raise MissingAcpMeta(
            "neither refinement nor goal metadata was preserved for this rollout"
        )
    return any(seen)


class PrimeAgentCapabilitiesTask(vf.Task[PrimeAgentCapabilityData]):
    @vf.reward(weight=1.0)
    async def capability(self, trace: vf.Trace) -> float:
        family = self.data.family
        if family == "ipython_cell":
            success = _ipython_cell_completed(trace, self.data)
        elif family == "persistence":
            success = _persistence_completed(trace, self.data)
        elif family == "subagent_lifecycle":
            success = (
                spawned_and_finished(trace)
                and child_tokens_attributed(trace)
                and no_outstanding_subagents(trace)
                and _subagent_protocol_completed(trace, self.data)
            )
        elif family == "harness_state":
            success = _harness_state_changed(trace) and _completed_exactly(trace)
        else:
            success = (
                _child_cancelled(trace)
                and no_outstanding_subagents(trace)
                and _completed_exactly(trace)
            )
        return float(success)


class PrimeAgentCapabilitiesEnv(vf.SingleAgentEnv):
    async def run(self, task, agents):
        if task.data.family == "persistence":
            async with (
                agents.agent.provision(task) as runtime,
                agents.agent.interaction(task, runtime=runtime) as interaction,
            ):
                segments = []
                first = await interaction.turn(task.data.turns[0])
                segments.append(first)
                if not first.terminated:
                    segments.append(await interaction.turn(task.data.turns[1]))
                interaction.trace.info["prime_agent_segments"] = [
                    _segment_info(segment) for segment in segments
                ]
            return

        async with agents.agent.interaction(task) as interaction:
            segment = await interaction.turn(task.data.turns[0])
            if task.data.family == "ipython_cell":
                interaction.trace.info["prime_agent_segments"] = [
                    _segment_info(segment)
                ]


class PrimeAgentCapabilitiesConfig(vf.TasksetConfig):
    families: tuple[Family, ...] = Field(FAMILIES, min_length=1)
    instances_per_family: int = Field(2, ge=1)


def _task_data(family: Family, instance: int, idx: int) -> PrimeAgentCapabilityData:
    system_prompt = "Follow the request exactly and use Prime Agent's native capabilities."
    sentinel = None
    marker_name = None
    child_result = None
    if family == "ipython_cell":
        sentinel = f"prime-agent-capability-{instance}"
        cell = f"print({sentinel!r})"
        turns = (
            (
                f"Use IPython to execute exactly this cell:\n\n{cell}\n\n"
                "Then reply with exactly DONE."
            ),
        )
    elif family == "persistence":
        marker_name = f"_prime_capability_marker_{instance}"
        first_cell = (
            f"import secrets\n{marker_name} = secrets.token_hex(32)\n"
            f"print({marker_name})"
        )
        second_cell = f"print({marker_name})"
        turns = (
            (
                f"Use IPython to execute exactly this cell:\n\n{first_cell}\n\n"
                "Then reply with exactly READY without including the printed value."
            ),
            (
                "Use IPython to execute exactly this cell without defining, assigning, "
                f"or reconstructing anything:\n\n{second_cell}\n\n"
                "Then reply with exactly the printed value."
            ),
        )
    elif family == "subagent_lifecycle":
        value = 37 + instance
        child_result = value * (value + 1)
        turns = (
            (
                "Use IPython to spawn exactly one subagent with rlm(). Ask it to compute "
                f"{value} * {value + 1} and send the result to you. Wait for its explicit "
                "message, then reply with exactly DONE."
            ),
        )
    elif family == "harness_state":
        turns = (
            (
                "Run /refine to record that this environment prefers concise answers "
                f"for capability suite {instance}, then reply with exactly DONE."
            ),
        )
    else:
        turns = (
            (
                "Use IPython to spawn one subagent with rlm() that sleeps for a long "
                "time. Delete it with rlm.delete_subagent() before it finishes, wait "
                "until no child remains active, and reply with exactly DONE."
            ),
        )
    return PrimeAgentCapabilityData(
        idx=idx,
        name=f"{family}-i{instance}",
        prompt=None,
        system_prompt=system_prompt,
        family=family,
        instance=instance,
        turns=turns,
        sentinel=sentinel,
        marker_name=marker_name,
        child_result=child_result,
    )


class PrimeAgentCapabilitiesTaskset(
    vf.Taskset[PrimeAgentCapabilitiesTask, PrimeAgentCapabilitiesConfig]
):
    def load(self) -> list[PrimeAgentCapabilitiesTask]:
        tasks = []
        for instance in range(self.config.instances_per_family):
            for family in self.config.families:
                tasks.append(
                    PrimeAgentCapabilitiesTask(
                        _task_data(family, instance, len(tasks))
                    )
                )
        return tasks

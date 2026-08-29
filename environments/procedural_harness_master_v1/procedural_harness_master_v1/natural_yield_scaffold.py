"""Training-only constrained exploration for the natural N1 passive-yield boundary.

This module is deliberately opt-in.  When installed it changes exactly one model request in
eligible natural N1 or N1a rollouts: after a verified retained child spawn, when the hidden task
graph contains no remaining coordinator-local work and no child message has arrived, the next
coordinator request is sent upstream with no tools.  The response is still sampled by the model
and the ordinary Prime Agent runtime and frozen hard verifier decide whether the complete
trajectory succeeds.

The standalone scaffold never changes prompts, model responses, task answers, or scoring.  In the
earliest leaky interaction-curriculum phases, it also replaces the empty successful-spawn tool
observation with an explicitly labeled exact-yield observation and constrains that one generation
to the disclosed waiting sentence through the inference server's structured-choice decoder.
Natural N1 and N1a variants with required coordinator-local work are intentionally left untouched
and therefore serve as the anti-overgeneralization control ("do not learn: after every spawn,
always yield").
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
from typing import Any

from procedural_harness_master_v1.taskset import (
    PRIVATE_EVIDENCE_HEADER,
    _assigned_call_names,
    _branch_root,
    _call_name,
    _failed,
    _incoming_messages,
    _ipython_events,
    _spawn_name,
    _spawn_prompt,
)
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.session import RolloutSession
from verifiers.v1.trace import InterceptRecord
from verifiers.v1.types import (
    AssistantMessage,
    Request,
    Tool,
    ToolMessage,
    UserMessage,
    content_text,
)

SCAFFOLD_SCHEMA_VERSION = "prime-agent/natural-yield-scaffold/v1"
SCAFFOLD_INFO_KEY = "natural_yield_scaffold"
_HANDLER_NAME = "natural_yield_training_scaffold"
_INVALID_REPLAY_HANDLER_NAME = "natural_yield_invalid_tool_replay_compatibility"
_INVALID_ARGUMENTS_KEY = "__invalid_tool_arguments__"
_SESSION_PATCH_MARKER = "_natural_yield_scaffold_v1"
_CHAT_PATCH_MARKER = "_natural_yield_scaffold_chat_v1"
_ELIGIBLE_GRAPHS = {
    "natural_n1": "child_plus_private_state",
    "natural_n1a": "pure_async_child",
}
_INTERACTION_SCHEMA_VERSION = "prime-agent/interaction-curriculum/v1"
_INTERACTION_ROOT_PHASES = {
    "e0_full_actions",
    "e0b_select_child_value",
    "e0c_natural_child",
    "e0c2_natural_child_no_template",
    "e0c25_inline_evidence",
    "e0c275_inline_location",
    "e0c28_inline_only",
    "e0c29_evidence_available",
    "e0c3_natural_child_minimal",
    "e0c4_recursive_coordinator_return",
    "e0d_guided_yield",
    "e0d2_capped_yield",
    "e0d2_capped_yield_exact_child",
    "e0d3_uncapped_yield_exact_child",
    "e0d3_uncapped_yield",
    "e1_root_and_yield",
}
_EXACT_YIELD_PHASES = {
    "e0_full_actions",
    "e0b_select_child_value",
    "e0c_natural_child",
    "e0c2_natural_child_no_template",
    "e0c25_inline_evidence",
    "e0c275_inline_location",
    "e0c28_inline_only",
    "e0c29_evidence_available",
    "e0c3_natural_child_minimal",
    "e0c4_recursive_coordinator_return",
}
_GUIDED_YIELD_PHASES = {"e0d_guided_yield"}
_CAPPED_YIELD_PHASES = {
    "e0d2_capped_yield",
    "e0d2_capped_yield_exact_child",
}
_INTERACTION_PHASE_ENV_VAR = "PROCEDURAL_INTERACTION_CURRICULUM"
E0_YIELD_GUIDANCE_MARKER = "[interaction-curriculum exact yield observation]"
GUIDED_YIELD_MARKER = "[interaction-curriculum guided yield observation]"
E0_YIELD_MAX_TOKENS = 128
E0_YIELD_RESPONSE = "Waiting for the child report."


def _data(trace: Any) -> Any | None:
    task = getattr(trace, "task", None)
    return getattr(task, "data", None)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _field(value: Any, name: str) -> Any:
    """Read task data from both live trace dictionaries and typed fixtures."""

    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _is_child_context(request: Request) -> bool:
    """The benchmark's private-evidence injection is a hidden, environment-owned role tag."""

    return any(
        isinstance(message, UserMessage)
        and PRIVATE_EVIDENCE_HEADER in content_text(message.content)
        for message in request.messages
    )


def _matching_child_branch_after_spawn(
    trace: Any,
    *,
    spawn_node_index: int,
    prompt: str,
) -> bool:
    nodes = getattr(trace, "nodes", [])
    if not 0 <= spawn_node_index < len(nodes):
        return False
    spawned_at = getattr(nodes[spawn_node_index], "timestamp", 0.0)
    return any(
        index != 0
        and node.parent is None
        and isinstance(node.message, UserMessage)
        and prompt in content_text(node.message.content)
        and getattr(node, "timestamp", 0.0) >= spawned_at
        for index, node in enumerate(nodes)
    )


def _matches_constrained_root_action(trace: Any, code: str) -> bool:
    info = getattr(trace, "info", {})
    record = info.get("interaction_curriculum") if isinstance(info, dict) else None
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != _INTERACTION_SCHEMA_VERSION
        or record.get("phase") not in _INTERACTION_ROOT_PHASES
    ):
        return False
    events = record.get("events")
    root_events = [
        event
        for event in events or []
        if isinstance(event, dict) and event.get("kind") == "root_retained_spawn"
    ]
    return bool(
        len(root_events) == 1
        and root_events[0].get("mode") == "single_exact_ipython_action"
        and root_events[0].get("code_sha256")
        == hashlib.sha256(code.encode()).hexdigest()
    )


def _valid_retained_spawn_node(trace: Any, data: Any) -> int | None:
    """Return the unique clean coordinator spawn that establishes the N1 child handle."""

    nodes = getattr(trace, "nodes", [])
    if not nodes:
        return None
    oracle = _mapping(_field(data, "oracle"))
    children = oracle.get("children", [])
    if not isinstance(children, list) or len(children) != 1:
        return None
    child = children[0]
    if not isinstance(child, dict):
        return None
    child_name = child.get("name")
    child_path = child.get("resource_path")
    if not isinstance(child_name, str) or not isinstance(child_path, str):
        return None

    coordinator_root = _branch_root(trace, 0)
    eligible: list[int] = []
    for event in _ipython_events(trace):
        if _branch_root(trace, event.node_index) != coordinator_root or _failed(
            event.output
        ):
            continue
        try:
            tree = ast.parse(event.code)
        except SyntaxError:
            continue
        assigned_calls = _assigned_call_names(tree)
        valid_spawn = False
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            prompt = _spawn_prompt(call) or ""
            if not (
                _call_name(call) == "rlm"
                and id(call) in assigned_calls
                and _spawn_name(call, event.output) == child_name
                and child_path in prompt
            ):
                continue
            explicit_handle = "RLMSpawnHandle(" in event.output
            causal_child_branch = _matching_child_branch_after_spawn(
                trace,
                spawn_node_index=event.node_index,
                prompt=prompt,
            )
            constrained_root_action = _matches_constrained_root_action(
                trace, event.code
            )
            valid_spawn = (
                explicit_handle or causal_child_branch or constrained_root_action
            )
            if valid_spawn:
                break
        if valid_spawn:
            eligible.append(event.node_index)
    return eligible[0] if len(eligible) == 1 else None


def _has_forbidden_post_spawn_detour(trace: Any, spawn_node_index: int) -> bool:
    """Fail closed if the coordinator already acted again after the clean spawn."""

    coordinator_root = _branch_root(trace, 0)
    for index, node in enumerate(getattr(trace, "nodes", [])):
        if index <= spawn_node_index:
            continue
        if (
            isinstance(node.message, AssistantMessage)
            and node.sampled
            and _branch_root(trace, index) == coordinator_root
        ):
            return True
    return False


def _eligible(trace: Any, request: Request) -> tuple[bool, int | None, str]:
    data = _data(trace)
    family = _field(data, "family") if data is not None else None
    if family not in _ELIGIBLE_GRAPHS:
        return False, None, "not_natural_n1"
    metadata = _mapping(_field(data, "generation_metadata"))
    if metadata.get("graph_variant") != _ELIGIBLE_GRAPHS[family]:
        return False, None, "local_work_control"
    if _is_child_context(request):
        return False, None, "child_context"
    info = getattr(trace, "info", {})
    if isinstance(info, dict) and SCAFFOLD_INFO_KEY in info:
        return False, None, "already_fired"
    spawn_node_index = _valid_retained_spawn_node(trace, data)
    if spawn_node_index is None:
        return False, None, "no_clean_retained_spawn"
    if any(index > spawn_node_index for index, _, _ in _incoming_messages(trace)):
        return False, spawn_node_index, "child_message_already_visible"
    if _has_forbidden_post_spawn_detour(trace, spawn_node_index):
        return False, spawn_node_index, "post_spawn_detour"
    if not request.tools:
        return False, spawn_node_index, "tools_already_absent"
    return True, spawn_node_index, "eligible"


def _record_fire(
    trace: Any,
    request: Request,
    spawn_node_index: int,
    *,
    guidance_level: str,
) -> None:
    data = _data(trace)
    metadata = _mapping(_field(data, "generation_metadata")) if data is not None else {}
    oracle = _mapping(_field(data, "oracle")) if data is not None else {}
    children = oracle.get("children", [])
    child_name = (
        children[0].get("name") if children and isinstance(children[0], dict) else None
    )
    trace.info[SCAFFOLD_INFO_KEY] = {
        "schema_version": SCAFFOLD_SCHEMA_VERSION,
        "mode": "one_turn_no_tools",
        "fired": True,
        "spawn_node_index": spawn_node_index,
        "original_tool_count": len(request.tools or []),
        "graph_variant": metadata.get("graph_variant"),
        "semantic_family": metadata.get("semantic_family"),
        "child_name": child_name,
        "exact_yield_guidance": guidance_level == "exact",
        "guided_yield_instruction": guidance_level == "guided",
        "capped_yield_decode": guidance_level == "capped",
        "max_tokens": E0_YIELD_MAX_TOKENS if guidance_level != "none" else None,
        "decode_constraint": (
            "vllm_structured_outputs_choice" if guidance_level == "exact" else None
        ),
        "response_sha256": (
            hashlib.sha256(E0_YIELD_RESPONSE.encode()).hexdigest()
            if guidance_level == "exact"
            else None
        ),
    }


def _guide_yield(trace: Any, request: Request) -> tuple[Request, str]:
    """Expose the exact passive transition in the earliest leaky phases."""

    info = getattr(trace, "info", {})
    interaction = info.get("interaction_curriculum") if isinstance(info, dict) else None
    if (
        not isinstance(interaction, dict)
        or interaction.get("schema_version") != _INTERACTION_SCHEMA_VERSION
        or interaction.get("phase")
        not in _EXACT_YIELD_PHASES | _GUIDED_YIELD_PHASES | _CAPPED_YIELD_PHASES
        or not request.messages
        or not isinstance(request.messages[-1], ToolMessage)
    ):
        return request, "none"
    phase = interaction.get("phase")
    if phase in _CAPPED_YIELD_PHASES:
        return request, "capped"
    messages = list(request.messages)
    previous = content_text(messages[-1].content).rstrip()
    exact = phase in _EXACT_YIELD_PHASES
    if exact:
        guidance = (
            f"{E0_YIELD_GUIDANCE_MARKER}\n"
            "The delegated child is running asynchronously. Do not emit a tool call, tool "
            "syntax, JSON, or analysis. Reply now with exactly this sentence:\n"
            f"{E0_YIELD_RESPONSE}"
        )
    else:
        guidance = (
            f"{GUIDED_YIELD_MARKER}\n"
            "The delegated child is running asynchronously. Do not emit a tool call, tool "
            "syntax, JSON, or analysis. Reply now with one short waiting sentence in your "
            "own words."
        )
    content = f"{previous}\n\n{guidance}" if previous else guidance
    messages[-1] = messages[-1].model_copy(update={"content": content})
    return request.model_copy(
        update={"messages": messages}
    ), "exact" if exact else "guided"


def _chat_tool(tool: Tool) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }
    if tool.strict is not None:
        function["strict"] = tool.strict
    return {"type": "function", "function": function}


def _invalid_tool_arguments(request: Request) -> bool:
    for message in request.messages:
        if not isinstance(message, AssistantMessage):
            continue
        for call in message.tool_calls or []:
            try:
                value = json.loads(call.arguments)
            except (TypeError, json.JSONDecodeError):
                return True
            if not isinstance(value, dict):
                return True
    return False


def _repair_native_tool_argument_replay(body: dict, request: Request) -> None:
    """Keep a malformed no-tool generation visible while making its replay renderable."""

    for native, message in zip(body.get("messages", []), request.messages, strict=True):
        if not isinstance(message, AssistantMessage):
            continue
        native_calls = native.get("tool_calls") if isinstance(native, dict) else None
        if not isinstance(native_calls, list):
            continue
        for native_call, call in zip(
            native_calls, message.tool_calls or [], strict=False
        ):
            try:
                value = json.loads(call.arguments)
            except (TypeError, json.JSONDecodeError):
                value = None
            if isinstance(value, dict) or not isinstance(native_call, dict):
                continue
            function = native_call.get("function")
            if isinstance(function, dict):
                function["arguments"] = json.dumps(
                    {_INVALID_ARGUMENTS_KEY: call.arguments},
                    sort_keys=True,
                    separators=(",", ":"),
                )


def _install_chat_tool_rewrite() -> bool:
    """Carry the scaffold's typed tool removal into Prime Agent's native request."""

    if getattr(ChatDialect, _CHAT_PATCH_MARKER, False):
        return False
    current = ChatDialect.rewrite_request

    def rewrite_request_with_natural_yield_tools(
        self: ChatDialect,
        body: dict,
        before: Request,
        after: Request,
    ) -> None:
        current(self, body, before, after)
        _repair_native_tool_argument_replay(body, after)
        if after.tools == before.tools:
            return
        if after.tools:
            body["tools"] = [_chat_tool(tool) for tool in after.tools]
            return
        body.pop("tools", None)
        body.pop("tool_choice", None)
        body.pop("parallel_tool_calls", None)
        exact_guidance = any(
            isinstance(message, ToolMessage)
            and E0_YIELD_GUIDANCE_MARKER in content_text(message.content)
            for message in after.messages
        )
        guided = exact_guidance or any(
            isinstance(message, ToolMessage)
            and GUIDED_YIELD_MARKER in content_text(message.content)
            for message in after.messages
        )
        capped = (
            not _is_child_context(after)
            and os.environ.get(_INTERACTION_PHASE_ENV_VAR) in _CAPPED_YIELD_PHASES
        )
        if guided or capped:
            body.pop("max_completion_tokens", None)
            body["max_tokens"] = E0_YIELD_MAX_TOKENS
            body["temperature"] = 0.0
            body["reasoning_effort"] = "low"
            body["chat_template_kwargs"] = {"enable_thinking": False}
            if exact_guidance:
                body["structured_outputs"] = {"choice": [E0_YIELD_RESPONSE]}

    setattr(rewrite_request_with_natural_yield_tools, _CHAT_PATCH_MARKER, True)
    ChatDialect.rewrite_request = rewrite_request_with_natural_yield_tools
    setattr(ChatDialect, _CHAT_PATCH_MARKER, True)
    return True


def install_natural_yield_scaffold() -> bool:
    """Install the request-boundary scaffold once in this evaluator process.

    Returns True when this call installs the wrapper, False when it was already installed.
    The wrapper delegates all ordinary request interception/stops to Verifiers first and then,
    for one eligible natural N1 root request only, removes the tool list before the dialect
    rewrites the native wire body.  E0 additionally receives its disclosed exact-yield
    observation and per-transition structured-choice decoding.
    """

    installed = _install_chat_tool_rewrite()
    if getattr(RolloutSession, _SESSION_PATCH_MARKER, False):
        return installed
    current = RolloutSession.rewrite_request
    original = current

    async def rewrite_request_with_natural_yield_scaffold(
        self: RolloutSession,
        request: Request,
        *,
        run_stops: bool = True,
    ) -> tuple[Request, list[InterceptRecord], str | None]:
        lock = getattr(self, "_natural_yield_scaffold_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._natural_yield_scaffold_lock = lock
        async with lock:
            rewritten, records, stopped = await original(
                self, request, run_stops=run_stops
            )
            if stopped is not None:
                return rewritten, records, stopped
            if SCAFFOLD_INFO_KEY in getattr(
                self.trace, "info", {}
            ) and _invalid_tool_arguments(rewritten):
                return (
                    rewritten,
                    [
                        *records,
                        InterceptRecord(handler=_INVALID_REPLAY_HANDLER_NAME),
                    ],
                    None,
                )
            eligible, spawn_node_index, _ = _eligible(self.trace, rewritten)
            if not eligible or spawn_node_index is None:
                return rewritten, records, None
            guided, guidance_level = _guide_yield(self.trace, rewritten)
            _record_fire(
                self.trace,
                guided,
                spawn_node_index,
                guidance_level=guidance_level,
            )
            constrained = guided.model_copy(update={"tools": None})
            return constrained, [*records, InterceptRecord(handler=_HANDLER_NAME)], None

    setattr(
        rewrite_request_with_natural_yield_scaffold,
        _SESSION_PATCH_MARKER,
        True,
    )
    RolloutSession.rewrite_request = rewrite_request_with_natural_yield_scaffold
    setattr(RolloutSession, _SESSION_PATCH_MARKER, True)
    return True


def scaffolded_yield_node_index(trace: Any) -> int | None:
    """Locate the model-generated no-tool coordinator response caused by the scaffold."""

    info = getattr(trace, "info", {})
    record = info.get(SCAFFOLD_INFO_KEY) if isinstance(info, dict) else None
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != SCAFFOLD_SCHEMA_VERSION
    ):
        return None
    spawn_node_index = record.get("spawn_node_index")
    if not isinstance(spawn_node_index, int):
        return None
    coordinator_root = _branch_root(trace, 0)
    first_incoming = min(
        (
            index
            for index, _, _ in _incoming_messages(trace)
            if index > spawn_node_index
        ),
        default=len(getattr(trace, "nodes", [])),
    )
    candidates = [
        index
        for index, node in enumerate(getattr(trace, "nodes", []))
        if spawn_node_index < index < first_incoming
        and isinstance(node.message, AssistantMessage)
        and node.sampled
        and _branch_root(trace, index) == coordinator_root
        and not node.message.tool_calls
    ]
    return candidates[0] if len(candidates) == 1 else None


def keep_scaffolded_natural_yield_response(trace: Any) -> list[list[bool]]:
    """Select only the native sampled yield response from a scaffolded hard-success trace.

    Pair this with Prime-RL's ordinary ``minimum_reward(threshold=1.0)`` filter.  No synthetic
    target text is created here: the selected tokens are exactly the model tokens sampled while
    the one-turn tool constraint was active.
    """

    target_index = scaffolded_yield_node_index(trace)
    target = trace.nodes[target_index] if target_index is not None else None
    masks: list[list[bool]] = []
    trained_nodes: set[int] = set()
    for branch in trace.branches:
        branch_mask: list[bool] = []
        has_trainable_tokens = False
        for node in branch.nodes:
            is_new_trainable = (
                node.sampled and any(node.mask) and id(node) not in trained_nodes
            )
            if is_new_trainable:
                trained_nodes.add(id(node))
                has_trainable_tokens = True
            keep = is_new_trainable and node is target
            branch_mask.extend(bool(sampled and keep) for sampled in node.mask)
        if has_trainable_tokens:
            masks.append(branch_mask)
    return masks


def scaffold_audit(trace: Any) -> dict[str, Any]:
    """Compact, value-free audit for Y0/Y1 collection summaries."""

    info = getattr(trace, "info", {})
    record = info.get(SCAFFOLD_INFO_KEY) if isinstance(info, dict) else None
    index = scaffolded_yield_node_index(trace)
    return {
        "schema_version": SCAFFOLD_SCHEMA_VERSION,
        "fired": isinstance(record, dict) and record.get("fired") is True,
        "yield_node_index": index,
        "native_no_tool_response": index is not None,
        "graph_variant": record.get("graph_variant")
        if isinstance(record, dict)
        else None,
        "semantic_family": record.get("semantic_family")
        if isinstance(record, dict)
        else None,
    }

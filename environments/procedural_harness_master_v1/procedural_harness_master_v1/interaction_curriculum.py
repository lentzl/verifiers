"""Train-gen-only action-space curriculum for early natural N1a acquisition."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import re
from enum import StrEnum
from typing import Any

from procedural_harness_master_v1.taskset import (
    PRIVATE_EVIDENCE_HEADER,
    PRIVILEGED_BOOTSTRAP_HEADER,
)
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.session import RolloutSession
from verifiers.v1.trace import InterceptRecord
from verifiers.v1.types import (
    AssistantMessage,
    Request,
    Tool,
    UserMessage,
    content_text,
)

CURRICULUM_ENV_VAR = "PROCEDURAL_INTERACTION_CURRICULUM"
CURRICULUM_SCHEMA_VERSION = "prime-agent/interaction-curriculum/v1"
CURRICULUM_INFO_KEY = "interaction_curriculum"
TOOL_DESCRIPTION_MARKER = "[interaction-curriculum exact action]"
COEVOLUTION_CONTEXT_HEADER = "[spade-coevolution-environment/v1]"
CHILD_VALUE_SEND_PATTERN = (
    r"^await agent_message\.send\('[0-9]+', receiver_role='parent'\)$"
)
_HANDLER_NAME = "interaction_curriculum_exact_action"
_CHILD_STOP_HANDLER_NAME = "interaction_curriculum_child_stop"
_SESSION_PATCH_MARKER = "_interaction_curriculum_session_v1"
_CHAT_PATCH_MARKER = "_interaction_curriculum_chat_v1"


class InteractionCurriculumPhase(StrEnum):
    E0_FULL_ACTIONS = "e0_full_actions"
    E0B_SELECT_CHILD_VALUE = "e0b_select_child_value"
    E0C_NATURAL_CHILD = "e0c_natural_child"
    E0C2_NATURAL_CHILD_NO_TEMPLATE = "e0c2_natural_child_no_template"
    E0C25_INLINE_EVIDENCE = "e0c25_inline_evidence"
    E0C275_INLINE_LOCATION = "e0c275_inline_location"
    E0C28_INLINE_ONLY = "e0c28_inline_only"
    E0C29_EVIDENCE_AVAILABLE = "e0c29_evidence_available"
    E0C3_NATURAL_CHILD_MINIMAL = "e0c3_natural_child_minimal"
    E0D_GUIDED_YIELD = "e0d_guided_yield"
    E0D2_CAPPED_YIELD = "e0d2_capped_yield"
    E0D2_CAPPED_YIELD_EXACT_CHILD = "e0d2_capped_yield_exact_child"
    E0D3_UNCAPPED_YIELD_EXACT_CHILD = "e0d3_uncapped_yield_exact_child"
    E0D3_UNCAPPED_YIELD = "e0d3_uncapped_yield"
    E1_ROOT_AND_YIELD = "e1_root_and_yield"
    E2_YIELD_ONLY = "e2_yield_only"


def configured_phase() -> InteractionCurriculumPhase | None:
    value = os.environ.get(CURRICULUM_ENV_VAR)
    if value is None:
        return None
    return InteractionCurriculumPhase(value)


def _data(trace: Any) -> Any | None:
    task = getattr(trace, "task", None)
    return getattr(task, "data", None)


def _is_child_context(request: Request) -> bool:
    return any(
        isinstance(message, UserMessage)
        and PRIVATE_EVIDENCE_HEADER in content_text(message.content)
        for message in request.messages
    )


def _eligible_data(data: Any) -> bool:
    metadata = getattr(data, "generation_metadata", {})
    prompt = getattr(data, "prompt", "")
    return bool(
        getattr(data, "split", None) == "train_gen"
        and getattr(data, "family", None) == "natural_n1a"
        and isinstance(metadata, dict)
        and metadata.get("graph_variant") == "pure_async_child"
        and isinstance(prompt, str)
        and PRIVILEGED_BOOTSTRAP_HEADER in prompt
    )


def _child(data: Any) -> dict[str, Any] | None:
    oracle = getattr(data, "oracle", {})
    children = oracle.get("children", []) if isinstance(oracle, dict) else []
    if len(children) != 1 or not isinstance(children[0], dict):
        return None
    child = children[0]
    required = ("name", "resource_path", "expected_result")
    return child if all(key in child for key in required) else None


def _coevolution_child_context(data: Any) -> str | None:
    prompt = getattr(data, "prompt", "")
    marker = f"{COEVOLUTION_CONTEXT_HEADER}\n"
    if not isinstance(prompt, str) or marker not in prompt:
        return None
    encoded = prompt.split(marker, 1)[1].splitlines()[0]
    payload = json.loads(encoded)
    if not isinstance(payload, dict):
        raise TypeError("SPADE coevolution context must be a JSON object")
    context = payload.get("child_context")
    if not isinstance(context, str) or not context.strip() or len(context) > 2000:
        raise ValueError("SPADE coevolution child context must be 1-2000 characters")
    if PRIVATE_EVIDENCE_HEADER in context or PRIVILEGED_BOOTSTRAP_HEADER in context:
        raise ValueError("SPADE coevolution child context contains a reserved header")
    return context.strip()


def _child_prompt(
    child: dict[str, Any],
    *,
    reveal_result: bool,
    natural_child_template: bool = False,
    inline_evidence_hint: bool = False,
    concise_inline_evidence_hint: bool = False,
    positive_inline_location_hint: bool = False,
    inline_only_hint: bool = False,
    evidence_available_hint: bool = False,
) -> str:
    if reveal_result:
        action = (
            "In your first IPython call execute exactly: "
            "await agent_message.send("
            f"{str(child['expected_result'])!r}, receiver_role='parent')."
        )
    elif natural_child_template:
        action = (
            f"Process the supplied private evidence and {child['operation']}. The inline "
            "Evidence contents in your system message are authoritative; the "
            "evidence label is not a runtime path, so do not use Path, open, or filesystem "
            "search. Read the requested integer directly from those inline contents. In your "
            "first IPython call replace VALUE with that integer and execute "
            "await agent_message.send(str(VALUE), receiver_role='parent'). Do not send the "
            "literal word VALUE."
        )
    elif inline_evidence_hint:
        action = (
            f"Process the supplied private evidence and {child['operation']}. The inline "
            "Evidence contents in your system message are authoritative; the evidence label "
            "is not a runtime path, so do not use Path, open, or filesystem search. Read the "
            "requested integer directly from those inline contents. In your first IPython "
            "call send only the resulting integer as a string to receiver_role='parent'."
        )
    elif concise_inline_evidence_hint:
        action = (
            f"Process the supplied private evidence and {child['operation']}. The private "
            "evidence is the inline card in your system message, not a filesystem path. In "
            "your first IPython call send only the resulting integer as a string to "
            "receiver_role='parent'."
        )
    elif positive_inline_location_hint:
        action = (
            f"Process the supplied private evidence and {child['operation']}. The private "
            "evidence is supplied inline in your system message. In your first IPython call "
            "send only the resulting integer as a string to receiver_role='parent'."
        )
    elif inline_only_hint:
        action = (
            f"Process the supplied private evidence and {child['operation']}. The private "
            "evidence is supplied inline. In your first IPython call send only the resulting "
            "integer as a string to receiver_role='parent'."
        )
    elif evidence_available_hint:
        action = (
            f"Process the supplied private evidence and {child['operation']}. The private "
            "evidence has already been supplied to you. In your first IPython call send "
            "only the resulting integer as a string to receiver_role='parent'."
        )
    else:
        action = (
            "Process the supplied private evidence and "
            f"{child['operation']}. In your first IPython call send only the resulting "
            "integer as a string to receiver_role='parent'."
        )
    return (
        f"You are {child['name']}. Only you own {child['resource_path']}. {action} "
        "Send exactly once, then stop; do not return the result only as prose."
    )


def _root_code(
    child: dict[str, Any],
    phase: InteractionCurriculumPhase,
    *,
    child_context: str | None = None,
) -> str:
    prompt = _child_prompt(
        child,
        reveal_result=phase is InteractionCurriculumPhase.E0_FULL_ACTIONS,
        # Prime Agent child sessions are outside the parent verifier trace, so
        # the request interceptor cannot always constrain their first action.
        # Exact-child phases therefore also carry the value-free natural send
        # template across the spawn boundary as a fail-soft stabilization.
        natural_child_template=phase
        in {
            InteractionCurriculumPhase.E0C_NATURAL_CHILD,
            InteractionCurriculumPhase.E0D2_CAPPED_YIELD_EXACT_CHILD,
            InteractionCurriculumPhase.E0D3_UNCAPPED_YIELD_EXACT_CHILD,
        },
        inline_evidence_hint=(
            phase is InteractionCurriculumPhase.E0C2_NATURAL_CHILD_NO_TEMPLATE
        ),
        concise_inline_evidence_hint=(
            phase is InteractionCurriculumPhase.E0C25_INLINE_EVIDENCE
        ),
        positive_inline_location_hint=(
            phase is InteractionCurriculumPhase.E0C275_INLINE_LOCATION
        ),
        inline_only_hint=(phase is InteractionCurriculumPhase.E0C28_INLINE_ONLY),
        evidence_available_hint=(
            phase is InteractionCurriculumPhase.E0C29_EVIDENCE_AVAILABLE
        ),
    )
    if child_context is not None:
        prompt = f"{prompt}\n\n{COEVOLUTION_CONTEXT_HEADER}\n{child_context}"
    return f"reviewer = await rlm({prompt!r}, name={child['name']!r})"


def _child_code(child: dict[str, Any]) -> str:
    return (
        "await agent_message.send("
        f"{str(child['expected_result'])!r}, receiver_role='parent')"
    )


def _event_kind(
    request: Request,
    phase: InteractionCurriculumPhase,
) -> str | None:
    if _is_child_context(request):
        if phase is InteractionCurriculumPhase.E0_FULL_ACTIONS:
            return "child_typed_send"
        if phase is InteractionCurriculumPhase.E0B_SELECT_CHILD_VALUE:
            return "child_value_send"
        if phase is InteractionCurriculumPhase.E0D_GUIDED_YIELD:
            return "child_value_send"
        if phase is InteractionCurriculumPhase.E0D2_CAPPED_YIELD:
            return "child_value_send"
        if phase is InteractionCurriculumPhase.E0D2_CAPPED_YIELD_EXACT_CHILD:
            return "child_typed_send"
        if phase is InteractionCurriculumPhase.E0D3_UNCAPPED_YIELD_EXACT_CHILD:
            return "child_typed_send"
        if phase is InteractionCurriculumPhase.E0D3_UNCAPPED_YIELD:
            return "child_value_send"
        return None
    return (
        "root_retained_spawn"
        if phase
        in {
            InteractionCurriculumPhase.E0_FULL_ACTIONS,
            InteractionCurriculumPhase.E0B_SELECT_CHILD_VALUE,
            InteractionCurriculumPhase.E0C_NATURAL_CHILD,
            InteractionCurriculumPhase.E0C2_NATURAL_CHILD_NO_TEMPLATE,
            InteractionCurriculumPhase.E0C25_INLINE_EVIDENCE,
            InteractionCurriculumPhase.E0C275_INLINE_LOCATION,
            InteractionCurriculumPhase.E0C28_INLINE_ONLY,
            InteractionCurriculumPhase.E0C29_EVIDENCE_AVAILABLE,
            InteractionCurriculumPhase.E0C3_NATURAL_CHILD_MINIMAL,
            InteractionCurriculumPhase.E0D_GUIDED_YIELD,
            InteractionCurriculumPhase.E0D2_CAPPED_YIELD,
            InteractionCurriculumPhase.E0D2_CAPPED_YIELD_EXACT_CHILD,
            InteractionCurriculumPhase.E0D3_UNCAPPED_YIELD_EXACT_CHILD,
            InteractionCurriculumPhase.E0D3_UNCAPPED_YIELD,
            InteractionCurriculumPhase.E1_ROOT_AND_YIELD,
        }
        else None
    )


def _event_seen(trace: Any, kind: str) -> bool:
    info = getattr(trace, "info", {})
    record = info.get(CURRICULUM_INFO_KEY) if isinstance(info, dict) else None
    events = record.get("events", []) if isinstance(record, dict) else []
    return any(
        isinstance(event, dict) and event.get("kind") == kind for event in events
    )


def _sampled_event_action_seen(trace: Any, kind: str) -> bool:
    """Verify that the constrained action was sampled before mediating its follow-up."""

    info = getattr(trace, "info", {})
    record = info.get(CURRICULUM_INFO_KEY) if isinstance(info, dict) else None
    events = record.get("events", []) if isinstance(record, dict) else []
    matching_events = [
        event
        for event in events
        if isinstance(event, dict) and event.get("kind") == kind
    ]
    if len(matching_events) != 1:
        return False
    event = matching_events[0]
    digest = event.get("code_sha256")
    pattern_digest = event.get("code_pattern_sha256")
    expected_pattern_digest = hashlib.sha256(
        CHILD_VALUE_SEND_PATTERN.encode()
    ).hexdigest()
    pattern_mode = (
        event.get("mode") == "pattern_constrained_ipython_action"
        and pattern_digest == expected_pattern_digest
    )
    matching_codes: list[str] = []
    for node in getattr(trace, "nodes", []):
        message = getattr(node, "message", None)
        calls = getattr(message, "tool_calls", None) or []
        if (
            not isinstance(message, AssistantMessage)
            or not node.sampled
            or len(calls) != 1
        ):
            continue
        call = calls[0]
        if call.name != "ipython":
            continue
        try:
            arguments = json.loads(call.arguments)
        except (TypeError, json.JSONDecodeError):
            continue
        code = arguments.get("code") if isinstance(arguments, dict) else None
        if not isinstance(code, str):
            continue
        if (
            isinstance(digest, str)
            and hashlib.sha256(code.encode()).hexdigest() == digest
            or pattern_mode
            and re.fullmatch(CHILD_VALUE_SEND_PATTERN, code)
        ):
            matching_codes.append(code)
    if len(matching_codes) != 1:
        return False
    if pattern_mode:
        event["sampled_code_sha256"] = hashlib.sha256(
            matching_codes[0].encode()
        ).hexdigest()
    return True


def _child_stop_seen(trace: Any) -> bool:
    info = getattr(trace, "info", {})
    record = info.get(CURRICULUM_INFO_KEY) if isinstance(info, dict) else None
    child_stop = record.get("child_stop") if isinstance(record, dict) else None
    return isinstance(child_stop, dict) and child_stop.get("fired") is True


def _sampled_child_ipython_codes(trace: Any) -> list[str]:
    nodes = getattr(trace, "nodes", [])
    if not nodes:
        return []
    coordinator_root = 0
    while getattr(nodes[coordinator_root], "parent", None) is not None:
        coordinator_root = nodes[coordinator_root].parent
    result: list[str] = []
    for index, node in enumerate(nodes):
        message = getattr(node, "message", None)
        if not isinstance(message, AssistantMessage) or not node.sampled:
            continue
        root = index
        while getattr(nodes[root], "parent", None) is not None:
            root = nodes[root].parent
        if root == coordinator_root:
            continue
        calls = getattr(message, "tool_calls", None) or []
        if len(calls) != 1 or calls[0].name != "ipython":
            continue
        try:
            arguments = json.loads(calls[0].arguments)
        except (TypeError, json.JSONDecodeError):
            continue
        code = arguments.get("code") if isinstance(arguments, dict) else None
        if isinstance(code, str):
            result.append(code)
    return result


def _natural_child_send_value(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return None
    expression = tree.body[0].value
    if not isinstance(expression, ast.Await) or not isinstance(
        expression.value, ast.Call
    ):
        return None
    call = expression.value
    if not (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "send"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "agent_message"
        and len(call.args) == 1
    ):
        return None
    receivers = [
        keyword.value
        for keyword in call.keywords
        if keyword.arg == "receiver_role"
    ]
    if len(receivers) > 1 or (
        receivers
        and (
            not isinstance(receivers[0], ast.Constant)
            or receivers[0].value != "parent"
        )
    ):
        return None
    if any(keyword.arg == "receiver_name" for keyword in call.keywords):
        return None
    value = call.args[0]
    if isinstance(value, ast.Constant) and isinstance(value.value, (str, int)):
        return str(value.value)
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "str"
        and len(value.args) == 1
        and isinstance(value.args[0], ast.Constant)
        and isinstance(value.args[0].value, int)
    ):
        return str(value.args[0].value)
    return None


def _record_natural_child_send(
    trace: Any,
    phase: InteractionCurriculumPhase,
    code: str,
) -> None:
    _record_event(
        trace,
        phase=phase,
        kind="child_natural_send",
        mode="unconstrained_ipython_action",
        code=code,
    )


def _record_child_stop(trace: Any, request: Request) -> None:
    record = trace.info.get(CURRICULUM_INFO_KEY)
    if not isinstance(record, dict):
        raise TypeError("child stop requires an interaction curriculum audit")
    record["child_stop"] = {
        "mode": "one_turn_no_tools",
        "fired": True,
        "original_tool_count": len(request.tools or []),
    }


def _rewrite_child_stop(
    trace: Any,
    request: Request,
    phase: InteractionCurriculumPhase,
) -> Request | None:
    """Make the response after an early-rung child send a terminal prose turn."""

    data = _data(trace)
    child_event = {
        InteractionCurriculumPhase.E0_FULL_ACTIONS: "child_typed_send",
        InteractionCurriculumPhase.E0B_SELECT_CHILD_VALUE: "child_value_send",
        InteractionCurriculumPhase.E0C_NATURAL_CHILD: "child_natural_send",
        InteractionCurriculumPhase.E0C2_NATURAL_CHILD_NO_TEMPLATE: "child_natural_send",
        InteractionCurriculumPhase.E0C25_INLINE_EVIDENCE: "child_natural_send",
        InteractionCurriculumPhase.E0C275_INLINE_LOCATION: "child_natural_send",
        InteractionCurriculumPhase.E0C28_INLINE_ONLY: "child_natural_send",
        InteractionCurriculumPhase.E0C29_EVIDENCE_AVAILABLE: "child_natural_send",
        InteractionCurriculumPhase.E0C3_NATURAL_CHILD_MINIMAL: "child_natural_send",
        InteractionCurriculumPhase.E0D_GUIDED_YIELD: "child_value_send",
        InteractionCurriculumPhase.E0D2_CAPPED_YIELD: "child_value_send",
        InteractionCurriculumPhase.E0D2_CAPPED_YIELD_EXACT_CHILD: "child_typed_send",
        InteractionCurriculumPhase.E0D3_UNCAPPED_YIELD_EXACT_CHILD: "child_typed_send",
        InteractionCurriculumPhase.E0D3_UNCAPPED_YIELD: "child_value_send",
    }.get(phase)
    if phase in {
        InteractionCurriculumPhase.E0C_NATURAL_CHILD,
        InteractionCurriculumPhase.E0C2_NATURAL_CHILD_NO_TEMPLATE,
        InteractionCurriculumPhase.E0C25_INLINE_EVIDENCE,
        InteractionCurriculumPhase.E0C275_INLINE_LOCATION,
        InteractionCurriculumPhase.E0C28_INLINE_ONLY,
        InteractionCurriculumPhase.E0C29_EVIDENCE_AVAILABLE,
        InteractionCurriculumPhase.E0C3_NATURAL_CHILD_MINIMAL,
    }:
        child = _child(data) if data is not None else None
        codes = _sampled_child_ipython_codes(trace)
        if (
            child is not None
            and len(codes) == 1
            and _natural_child_send_value(codes[0]) == str(child["expected_result"])
            and not _event_seen(trace, child_event)
        ):
            _record_natural_child_send(trace, phase, codes[0])
    if (
        child_event is None
        or data is None
        or not _eligible_data(data)
        or not _is_child_context(request)
        or not request.tools
        or _child_stop_seen(trace)
        or not _sampled_event_action_seen(trace, child_event)
    ):
        return None
    _record_child_stop(trace, request)
    return request.model_copy(update={"tools": None})


def _record_event(
    trace: Any,
    *,
    phase: InteractionCurriculumPhase,
    kind: str,
    mode: str,
    code: str | None = None,
    code_pattern: str | None = None,
) -> None:
    info = trace.info
    record = info.get(CURRICULUM_INFO_KEY)
    if not isinstance(record, dict):
        record = {
            "schema_version": CURRICULUM_SCHEMA_VERSION,
            "phase": phase.value,
            "events": [],
        }
        info[CURRICULUM_INFO_KEY] = record
    if record.get("phase") != phase.value:
        raise RuntimeError("interaction curriculum phase changed within one trace")
    event = {"kind": kind, "mode": mode}
    if code is not None:
        event["code_sha256"] = hashlib.sha256(code.encode()).hexdigest()
    if code_pattern is not None:
        event["code_pattern_sha256"] = hashlib.sha256(code_pattern.encode()).hexdigest()
    record["events"].append(event)


def _exact_ipython_request(request: Request, code: str) -> Request | None:
    ipython = next(
        (tool for tool in request.tools or [] if tool.name == "ipython"), None
    )
    if ipython is None:
        return None
    constrained = Tool(
        name="ipython",
        description=f"{TOOL_DESCRIPTION_MARKER} {ipython.description}",
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string", "enum": [code]}},
            "required": ["code"],
            "additionalProperties": False,
        },
        strict=True,
    )
    return request.model_copy(update={"tools": [constrained]})


def _pattern_ipython_request(request: Request, pattern: str) -> Request | None:
    ipython = next(
        (tool for tool in request.tools or [] if tool.name == "ipython"), None
    )
    if ipython is None:
        return None
    constrained = Tool(
        name="ipython",
        description=f"{TOOL_DESCRIPTION_MARKER} {ipython.description}",
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string", "pattern": pattern}},
            "required": ["code"],
            "additionalProperties": False,
        },
        strict=True,
    )
    return request.model_copy(update={"tools": [constrained]})


def _forced_ipython_request(request: Request) -> Request | None:
    ipython = next(
        (tool for tool in request.tools or [] if tool.name == "ipython"), None
    )
    if ipython is None:
        return None
    forced = ipython.model_copy(
        update={"description": f"{TOOL_DESCRIPTION_MARKER} {ipython.description}"}
    )
    return request.model_copy(update={"tools": [forced]})


def _rewrite_for_curriculum(trace: Any, request: Request) -> Request | None:
    phase = configured_phase()
    data = _data(trace)
    if phase is None or data is None or not _eligible_data(data):
        return None
    if phase in {
        InteractionCurriculumPhase.E0C_NATURAL_CHILD,
        InteractionCurriculumPhase.E0C2_NATURAL_CHILD_NO_TEMPLATE,
        InteractionCurriculumPhase.E0C25_INLINE_EVIDENCE,
        InteractionCurriculumPhase.E0C275_INLINE_LOCATION,
        InteractionCurriculumPhase.E0C28_INLINE_ONLY,
        InteractionCurriculumPhase.E0C29_EVIDENCE_AVAILABLE,
        InteractionCurriculumPhase.E0C3_NATURAL_CHILD_MINIMAL,
    } and _is_child_context(request):
        if not _sampled_child_ipython_codes(trace):
            return _forced_ipython_request(request)
        return None
    kind = _event_kind(request, phase)
    if kind is None or _event_seen(trace, kind):
        return None
    child = _child(data)
    if child is None:
        return None
    if kind == "child_value_send":
        code = None
        rewritten = _pattern_ipython_request(request, CHILD_VALUE_SEND_PATTERN)
    else:
        code = (
            _child_code(child)
            if kind == "child_typed_send"
            else _root_code(
                child,
                phase,
                child_context=_coevolution_child_context(data),
            )
        )
        rewritten = _exact_ipython_request(request, code)
    if rewritten is None:
        return None
    _record_event(
        trace,
        phase=phase,
        kind=kind,
        mode=(
            "pattern_constrained_ipython_action"
            if kind == "child_value_send"
            else "single_exact_ipython_action"
        ),
        code=code,
        code_pattern=(CHILD_VALUE_SEND_PATTERN if kind == "child_value_send" else None),
    )
    return rewritten


def _chat_tool(tool: Tool) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }
    if tool.strict is not None:
        function["strict"] = tool.strict
    return {"type": "function", "function": function}


def _install_chat_rewrite() -> bool:
    if getattr(ChatDialect, _CHAT_PATCH_MARKER, False):
        return False
    current = ChatDialect.rewrite_request

    def rewrite_request_with_interaction_curriculum(
        self: ChatDialect,
        body: dict,
        before: Request,
        after: Request,
    ) -> None:
        current(self, body, before, after)
        forced = [
            tool
            for tool in after.tools or []
            if TOOL_DESCRIPTION_MARKER in tool.description
        ]
        if len(forced) != 1:
            return
        tool = forced[0]
        body["tools"] = [_chat_tool(tool)]
        body["tool_choice"] = {
            "type": "function",
            "function": {"name": tool.name},
        }
        body["parallel_tool_calls"] = False

    setattr(rewrite_request_with_interaction_curriculum, _CHAT_PATCH_MARKER, True)
    ChatDialect.rewrite_request = rewrite_request_with_interaction_curriculum
    setattr(ChatDialect, _CHAT_PATCH_MARKER, True)
    return True


def install_interaction_curriculum() -> bool:
    """Install exact-action request mediation once in the evaluator process."""

    installed = _install_chat_rewrite()
    if getattr(RolloutSession, _SESSION_PATCH_MARKER, False):
        return installed
    current = RolloutSession.rewrite_request
    original = current

    async def rewrite_request_with_interaction_curriculum(
        self: RolloutSession,
        request: Request,
        *,
        run_stops: bool = True,
    ) -> tuple[Request, list[InterceptRecord], str | None]:
        lock = getattr(self, "_interaction_curriculum_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._interaction_curriculum_lock = lock
        async with lock:
            rewritten, records, stopped = await original(
                self, request, run_stops=run_stops
            )
            if stopped is not None or rewritten.tools is None:
                return rewritten, records, stopped
            phase = configured_phase()
            if phase is not None:
                child_stop = _rewrite_child_stop(self.trace, rewritten, phase)
                if child_stop is not None:
                    return (
                        child_stop,
                        [
                            *records,
                            InterceptRecord(handler=_CHILD_STOP_HANDLER_NAME),
                        ],
                        None,
                    )
            constrained = _rewrite_for_curriculum(self.trace, rewritten)
            if constrained is None:
                return rewritten, records, None
            return constrained, [*records, InterceptRecord(handler=_HANDLER_NAME)], None

    setattr(
        rewrite_request_with_interaction_curriculum,
        _SESSION_PATCH_MARKER,
        True,
    )
    RolloutSession.rewrite_request = rewrite_request_with_interaction_curriculum
    setattr(RolloutSession, _SESSION_PATCH_MARKER, True)
    return True


def curriculum_audit(trace: Any) -> dict[str, Any]:
    """Return a value-free audit of constrained actions sampled in one trace."""

    info = getattr(trace, "info", {})
    record = info.get(CURRICULUM_INFO_KEY) if isinstance(info, dict) else None
    events = record.get("events", []) if isinstance(record, dict) else []
    return {
        "schema_version": CURRICULUM_SCHEMA_VERSION,
        "phase": record.get("phase") if isinstance(record, dict) else None,
        "event_kinds": [
            event.get("kind") for event in events if isinstance(event, dict)
        ],
        "event_count": len(events),
    }

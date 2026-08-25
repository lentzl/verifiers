"""Causal context loss for the natural N1b persistence qualification.

N1b is meaningful only if the ephemeral intake value cannot survive in the model's visible
conversation. This boundary retains the original task instructions and the child report, but
removes every assistant/tool turn that preceded the report from the actual Chat Completions
request. Prime Agent's persistent IPython kernel is deliberately left intact, so a value that
the coordinator captured before delegation remains recoverable while a merely observed value
does not.
"""

from __future__ import annotations

import asyncio
from typing import Any

from procedural_harness_master_v1.taskset import PRIVATE_EVIDENCE_HEADER
from verifiers.v1.dialects.chat import ChatDialect, message_to_wire
from verifiers.v1.session import RolloutSession
from verifiers.v1.trace import InterceptRecord
from verifiers.v1.types import AssistantMessage, Request, UserMessage, content_text

BOUNDARY_SCHEMA_VERSION = "procedural-harness-master-v1/causal-context-boundary/v1"
BOUNDARY_INFO_KEY = "causal_context_boundary"
BOUNDARY_MARKER = "[causal persistence context boundary]"
_HANDLER_NAME = "causal_persistence_context_boundary"
_SESSION_PATCH_MARKER = "_causal_context_boundary_session_v1"
_CHAT_PATCH_MARKER = "_causal_context_boundary_chat_v1"


def _data(trace: Any) -> Any | None:
    task = getattr(trace, "task", None)
    return getattr(task, "data", None)


def _child_name(data: Any) -> str | None:
    oracle = getattr(data, "oracle", {})
    children = oracle.get("children", []) if isinstance(oracle, dict) else []
    if len(children) != 1 or not isinstance(children[0], dict):
        return None
    name = children[0].get("name")
    return name if isinstance(name, str) else None


def _child_message_index(request: Request, child_name: str) -> int | None:
    prefix = f"[from child:{child_name}]"
    return next(
        (
            index
            for index in range(len(request.messages) - 1, -1, -1)
            if isinstance(request.messages[index], UserMessage)
            and content_text(request.messages[index].content).lstrip().startswith(prefix)
        ),
        None,
    )


def _initial_prefix_end(request: Request, child_index: int) -> int:
    """Cut before the first model tool action, preserving system/task instructions."""

    return next(
        (
            index
            for index, message in enumerate(request.messages[:child_index])
            if isinstance(message, AssistantMessage) and message.tool_calls
        ),
        child_index,
    )


def _rewrite_for_boundary(trace: Any, request: Request) -> Request | None:
    data = _data(trace)
    if data is None or getattr(data, "family", None) != "natural_n1b":
        return None
    if any(
        isinstance(message, UserMessage)
        and PRIVATE_EVIDENCE_HEADER in content_text(message.content)
        for message in request.messages
    ):
        return None
    info = getattr(trace, "info", {})
    lease = info.get("persistence_lease")
    if not isinstance(lease, dict) or lease.get("closed") is not True:
        return None
    child_name = _child_name(data)
    if child_name is None:
        return None
    child_index = _child_message_index(request, child_name)
    if child_index is None:
        return None

    prefix_end = _initial_prefix_end(request, child_index)
    prefix = list(request.messages[:prefix_end])
    suffix = list(request.messages[child_index:])
    child_message = suffix[0]
    assert isinstance(child_message, UserMessage)
    child_text = content_text(child_message.content)
    if BOUNDARY_MARKER not in child_text:
        child_message = child_message.model_copy(
            update={
                "content": (
                    f"{child_text.rstrip()}\n\n{BOUNDARY_MARKER}\n"
                    "All assistant and tool turns before this child report are unavailable. "
                    "The persistent IPython kernel is the coordinator's only surviving memory. "
                    "Continue from the delivered evidence and any value captured in that kernel; "
                    "do not repeat delegation or the expired intake read."
                )
            }
        )
        suffix[0] = child_message
    rewritten = request.model_copy(update={"messages": [*prefix, *suffix]})
    if rewritten == request:
        return None

    record = info.get(BOUNDARY_INFO_KEY)
    applications = (
        int(record.get("applications", 0)) + 1 if isinstance(record, dict) else 1
    )
    info[BOUNDARY_INFO_KEY] = {
        "schema_version": BOUNDARY_SCHEMA_VERSION,
        "applied": True,
        "applications": applications,
        "child_name": child_name,
        "removed_messages": child_index - prefix_end,
    }
    return rewritten


def _install_chat_rewrite() -> bool:
    current = ChatDialect.rewrite_request
    if getattr(current, _CHAT_PATCH_MARKER, False):
        return False

    def rewrite_request_with_context_boundary(
        self: ChatDialect,
        body: dict,
        before: Request,
        after: Request,
    ) -> None:
        if any(
            isinstance(message, UserMessage)
            and BOUNDARY_MARKER in content_text(message.content)
            for message in after.messages
        ):
            body["messages"] = [message_to_wire(message) for message in after.messages]
            return
        current(self, body, before, after)

    setattr(rewrite_request_with_context_boundary, _CHAT_PATCH_MARKER, True)
    ChatDialect.rewrite_request = rewrite_request_with_context_boundary
    return True


def install_causal_context_boundary() -> bool:
    """Install the tightly scoped N1b wire boundary once per evaluator process."""

    installed = _install_chat_rewrite()
    current = RolloutSession.rewrite_request
    if getattr(current, _SESSION_PATCH_MARKER, False):
        return installed
    original = current

    async def rewrite_request_with_context_boundary(
        self: RolloutSession,
        request: Request,
        *,
        run_stops: bool = True,
    ) -> tuple[Request, list[InterceptRecord], str | None]:
        lock = getattr(self, "_causal_context_boundary_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._causal_context_boundary_lock = lock
        async with lock:
            rewritten, records, stopped = await original(
                self, request, run_stops=run_stops
            )
            if stopped is not None:
                return rewritten, records, stopped
            bounded = _rewrite_for_boundary(self.trace, rewritten)
            if bounded is None:
                return rewritten, records, None
            return bounded, [*records, InterceptRecord(handler=_HANDLER_NAME)], None

    setattr(rewrite_request_with_context_boundary, _SESSION_PATCH_MARKER, True)
    RolloutSession.rewrite_request = rewrite_request_with_context_boundary
    return True

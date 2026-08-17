"""Typed, answer-free feedback for one bidirectional follow-up transition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

FEEDBACK_SCHEMA_VERSION = "prime-agent/procedural-followup-feedback/v1"


class FollowupFailureCategory(StrEnum):
    """Stable category used to route the diagnostic into SDPO."""

    BIDIRECTIONAL_CONTROL = "bidirectional_control"


class FollowupFailureCode(StrEnum):
    """Failure-local transition currently admitted for bootstrap."""

    REPLY_TO_CHILD_REQUEST = "reply_to_child_request"


@dataclass(frozen=True, slots=True)
class FollowupFailureDiagnostic:
    """A causal coordinator failure containing no task answer."""

    child_name: str
    request_node_index: int
    target_node_index: int
    turn_index: int
    code: FollowupFailureCode = FollowupFailureCode.REPLY_TO_CHILD_REQUEST

    @property
    def category(self) -> FollowupFailureCategory:
        return FollowupFailureCategory.BIDIRECTIONAL_CONTROL

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FEEDBACK_SCHEMA_VERSION,
            "code": self.code.value,
            "category": self.category.value,
            "answer_free": True,
            "retryable": True,
            "turn_index": self.turn_index,
            "target_node_index": self.target_node_index,
            "evidence": {
                "child_name": self.child_name,
                "request_node_index": self.request_node_index,
            },
        }


def render_followup_feedback(diagnostic: FollowupFailureDiagnostic) -> str:
    """Describe only the failed transition, without supplying a result."""

    return (
        f"The explicit request from child {diagnostic.child_name!r} requires one causal "
        "coordinator reply. Reuse the retained coordinator value, convert the message "
        "payload to a string, execute one directly awaited agent_message.send to that "
        "child, and end the turn immediately. Do not wrap the await in asyncio.run, "
        "print, poll, observe, discover agents, compute the final answer, or wait inside "
        "IPython. Resume only from the child's next explicit message."
    )


def feedback_contract_payload(
    diagnostic: FollowupFailureDiagnostic,
) -> dict[str, Any]:
    """Return the typed contract including its exact model-facing message."""

    payload = diagnostic.to_dict()
    payload["message"] = render_followup_feedback(diagnostic)
    return payload

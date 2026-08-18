"""Typed, answer-free feedback for one natural passive-yield failure."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

FEEDBACK_SCHEMA_VERSION = "prime-agent/natural-yield-feedback/v1"


class NaturalYieldFailureCategory(StrEnum):
    """Stable category used to route the diagnostic into SDPO."""

    EVENT_CONTROL = "event_control"


class NaturalYieldFailureCode(StrEnum):
    """Failure-local transition admitted by the natural N1 bootstrap."""

    TOOL_CALL_AFTER_DELEGATION = "tool_call_after_delegation"


@dataclass(frozen=True, slots=True)
class NaturalYieldFailureDiagnostic:
    """One coordinator tool call made while awaiting child-owned evidence."""

    child_name: str
    spawn_node_index: int
    target_node_index: int
    turn_index: int
    code: NaturalYieldFailureCode = NaturalYieldFailureCode.TOOL_CALL_AFTER_DELEGATION

    @property
    def category(self) -> NaturalYieldFailureCategory:
        return NaturalYieldFailureCategory.EVENT_CONTROL

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
                "spawn_node_index": self.spawn_node_index,
            },
        }


def render_natural_yield_feedback(
    diagnostic: NaturalYieldFailureDiagnostic,
) -> str:
    """Describe the failed event-control decision without supplying an answer."""

    return (
        f"Child {diagnostic.child_name!r} was already admitted and no visible child message "
        "had arrived, but the coordinator queried or inspected state. Prime Agent delivers "
        "child replies as visible messages. Preserve the existing child handle and end this "
        "coordinator turn with a brief no-tool waiting response. Do not call IPython, "
        "agent_message, agent_observe, rlm, sleep, or any status API. Resume only after a "
        "visible child message."
    )


def feedback_contract_payload(
    diagnostic: NaturalYieldFailureDiagnostic,
) -> dict[str, Any]:
    """Return the typed contract including its exact model-facing message."""

    payload = diagnostic.to_dict()
    payload["message"] = render_natural_yield_feedback(diagnostic)
    return payload

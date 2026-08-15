"""Typed, answer-free feedback for Prime Agent ownership decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

FEEDBACK_SCHEMA_VERSION = "prime-agent/ownership-decision-feedback/v1"

Ownership = Literal["child", "coordinator"]


class OwnershipFailureCategory(StrEnum):
    """Stable categories used for routing and aggregate diagnostics."""

    ROUTING = "routing"
    STATE = "state"
    OWNERSHIP = "ownership"
    CONTROL = "control"
    OUTPUT_CONTRACT = "output_contract"


class OwnershipFailureCode(StrEnum):
    """Stable leaf-level failure taxonomy for one ownership decision."""

    REQUIRED_DECISION_MISSING = "required_decision_missing"
    MULTIPLE_COORDINATOR_DECISIONS = "multiple_coordinator_decisions"
    COORDINATOR_STATE_NOT_RETAINED = "coordinator_state_not_retained"
    STATE_RETAINED_AFTER_DELEGATION = "state_retained_after_delegation"
    REQUIRED_DELEGATION_MISSING = "required_delegation_missing"
    MULTIPLE_DELEGATIONS = "multiple_delegations"
    CHILD_HANDLE_NOT_RETAINED = "child_handle_not_retained"
    WRONG_CHILD_SELECTED = "wrong_child_selected"
    DELEGATED_RESOURCE_NOT_ASSIGNED = "delegated_resource_not_assigned"
    CHILD_RESOURCE_ACCESSED_BY_COORDINATOR = "child_resource_accessed_by_coordinator"
    COORDINATOR_STATE_LEAKED_TO_CHILD = "coordinator_state_leaked_to_child"
    PROHIBITED_CONTROL_ACTION = "prohibited_control_action"
    CONTINUED_AFTER_DELEGATION = "continued_after_delegation"
    UNNECESSARY_DELEGATION = "unnecessary_delegation"
    COORDINATOR_RESOURCE_NOT_ACCESSED = "coordinator_resource_not_accessed"
    OUTPUT_CONTRACT_VIOLATION = "output_contract_violation"


_CATEGORY_BY_CODE: dict[OwnershipFailureCode, OwnershipFailureCategory] = {
    OwnershipFailureCode.REQUIRED_DECISION_MISSING: OwnershipFailureCategory.ROUTING,
    OwnershipFailureCode.MULTIPLE_COORDINATOR_DECISIONS: OwnershipFailureCategory.CONTROL,
    OwnershipFailureCode.COORDINATOR_STATE_NOT_RETAINED: OwnershipFailureCategory.STATE,
    OwnershipFailureCode.STATE_RETAINED_AFTER_DELEGATION: OwnershipFailureCategory.STATE,
    OwnershipFailureCode.REQUIRED_DELEGATION_MISSING: OwnershipFailureCategory.ROUTING,
    OwnershipFailureCode.MULTIPLE_DELEGATIONS: OwnershipFailureCategory.ROUTING,
    OwnershipFailureCode.CHILD_HANDLE_NOT_RETAINED: OwnershipFailureCategory.STATE,
    OwnershipFailureCode.WRONG_CHILD_SELECTED: OwnershipFailureCategory.ROUTING,
    OwnershipFailureCode.DELEGATED_RESOURCE_NOT_ASSIGNED: OwnershipFailureCategory.OWNERSHIP,
    OwnershipFailureCode.CHILD_RESOURCE_ACCESSED_BY_COORDINATOR: OwnershipFailureCategory.OWNERSHIP,
    OwnershipFailureCode.COORDINATOR_STATE_LEAKED_TO_CHILD: OwnershipFailureCategory.OWNERSHIP,
    OwnershipFailureCode.PROHIBITED_CONTROL_ACTION: OwnershipFailureCategory.CONTROL,
    OwnershipFailureCode.CONTINUED_AFTER_DELEGATION: OwnershipFailureCategory.CONTROL,
    OwnershipFailureCode.UNNECESSARY_DELEGATION: OwnershipFailureCategory.ROUTING,
    OwnershipFailureCode.COORDINATOR_RESOURCE_NOT_ACCESSED: OwnershipFailureCategory.OWNERSHIP,
    OwnershipFailureCode.OUTPUT_CONTRACT_VIOLATION: OwnershipFailureCategory.OUTPUT_CONTRACT,
}


@dataclass(frozen=True, slots=True)
class OwnershipFailureSignals:
    """Bounded evidence extracted from one failed first decision."""

    ownership: Ownership
    resource_family: str
    expected_child: str
    resource_path: str
    coordinator_ipython_calls: int
    spawn_calls: int
    strict_success: bool
    state_retained: bool
    state_precedes_spawn: bool
    retained_handle: bool
    expected_child_selected: bool
    delegated_path: bool
    parent_path_access: bool
    local_state_leaked: bool
    prohibited_control: bool
    post_spawn_action: bool
    direct_answer_accurate: bool


@dataclass(frozen=True, slots=True)
class OwnershipFailureDiagnostic:
    """A deterministic diagnosis containing no task answer."""

    code: OwnershipFailureCode
    signals: OwnershipFailureSignals

    @property
    def category(self) -> OwnershipFailureCategory:
        return _CATEGORY_BY_CODE[self.code]

    def to_dict(self) -> dict[str, Any]:
        signals = self.signals
        return {
            "schema_version": FEEDBACK_SCHEMA_VERSION,
            "code": self.code.value,
            "category": self.category.value,
            "family": signals.resource_family,
            "ownership": signals.ownership,
            "turn_index": 0,
            "answer_free": True,
            "retryable": True,
            "resource": {
                "path": signals.resource_path,
                "expected_child": signals.expected_child,
            },
            "evidence": {
                "coordinator_ipython_calls": signals.coordinator_ipython_calls,
                "spawn_calls": signals.spawn_calls,
                "state_retained": signals.state_retained,
                "state_precedes_spawn": signals.state_precedes_spawn,
                "retained_handle": signals.retained_handle,
                "expected_child_selected": signals.expected_child_selected,
                "delegated_path": signals.delegated_path,
                "parent_path_access": signals.parent_path_access,
                "local_state_leaked": signals.local_state_leaked,
                "prohibited_control": signals.prohibited_control,
                "post_spawn_action": signals.post_spawn_action,
                "direct_answer_accurate": signals.direct_answer_accurate,
            },
        }


def diagnose_ownership_failure(
    signals: OwnershipFailureSignals,
) -> OwnershipFailureDiagnostic | None:
    """Select one causally actionable failure using a fixed priority."""

    if signals.strict_success:
        return None

    code: OwnershipFailureCode
    if signals.coordinator_ipython_calls == 0:
        code = OwnershipFailureCode.REQUIRED_DECISION_MISSING
    elif signals.coordinator_ipython_calls > 1:
        code = OwnershipFailureCode.MULTIPLE_COORDINATOR_DECISIONS
    elif signals.ownership == "child":
        if signals.parent_path_access:
            code = OwnershipFailureCode.CHILD_RESOURCE_ACCESSED_BY_COORDINATOR
        elif signals.prohibited_control:
            code = OwnershipFailureCode.PROHIBITED_CONTROL_ACTION
        elif signals.local_state_leaked:
            code = OwnershipFailureCode.COORDINATOR_STATE_LEAKED_TO_CHILD
        elif signals.spawn_calls == 0:
            code = OwnershipFailureCode.REQUIRED_DELEGATION_MISSING
        elif signals.spawn_calls > 1:
            code = OwnershipFailureCode.MULTIPLE_DELEGATIONS
        elif not signals.expected_child_selected:
            code = OwnershipFailureCode.WRONG_CHILD_SELECTED
        elif not signals.delegated_path:
            code = OwnershipFailureCode.DELEGATED_RESOURCE_NOT_ASSIGNED
        elif not signals.state_retained:
            code = OwnershipFailureCode.COORDINATOR_STATE_NOT_RETAINED
        elif not signals.state_precedes_spawn:
            code = OwnershipFailureCode.STATE_RETAINED_AFTER_DELEGATION
        elif not signals.retained_handle:
            code = OwnershipFailureCode.CHILD_HANDLE_NOT_RETAINED
        elif signals.post_spawn_action:
            code = OwnershipFailureCode.CONTINUED_AFTER_DELEGATION
        else:
            return None
    else:
        if signals.prohibited_control:
            code = OwnershipFailureCode.PROHIBITED_CONTROL_ACTION
        elif signals.spawn_calls > 0:
            code = OwnershipFailureCode.UNNECESSARY_DELEGATION
        elif not signals.state_retained:
            code = OwnershipFailureCode.COORDINATOR_STATE_NOT_RETAINED
        elif not signals.parent_path_access:
            code = OwnershipFailureCode.COORDINATOR_RESOURCE_NOT_ACCESSED
        elif not signals.direct_answer_accurate:
            code = OwnershipFailureCode.OUTPUT_CONTRACT_VIOLATION
        else:
            return None

    return OwnershipFailureDiagnostic(code=code, signals=signals)


def render_ownership_feedback(diagnostic: OwnershipFailureDiagnostic) -> str:
    """Render natural corrective feedback without supplying a target action."""

    code = diagnostic.code
    signals = diagnostic.signals
    if code is OwnershipFailureCode.REQUIRED_DECISION_MISSING:
        return "No coordinator IPython decision was observed. Retry the same ownership decision using the harness."
    if code is OwnershipFailureCode.MULTIPLE_COORDINATOR_DECISIONS:
        return (
            "The ownership boundary requires one atomic coordinator decision, but this attempt used multiple "
            "IPython calls. Preserve useful state and retry the same first decision once."
        )
    if code is OwnershipFailureCode.COORDINATOR_STATE_NOT_RETAINED:
        return (
            "The coordinator-local state named by the task was not retained in persistent IPython state. "
            "Retain it while retrying the same ownership decision."
        )
    if code is OwnershipFailureCode.STATE_RETAINED_AFTER_DELEGATION:
        return (
            "The coordinator-local state was not established before delegation. Retry atomically with the "
            "local state retained before admitting the child."
        )
    if code is OwnershipFailureCode.REQUIRED_DELEGATION_MISSING:
        return (
            f"The task assigned the resource to child {signals.expected_child!r}, but no child was admitted. "
            "Retry the same decision without doing the child's resource work in the coordinator."
        )
    if code is OwnershipFailureCode.MULTIPLE_DELEGATIONS:
        return "More than one child admission was attempted. Retry with exactly the one child assigned by the task."
    if code is OwnershipFailureCode.CHILD_HANDLE_NOT_RETAINED:
        return (
            "The child admission result was not retained. Retry the same decision and preserve the returned "
            "handle in persistent coordinator state."
        )
    if code is OwnershipFailureCode.WRONG_CHILD_SELECTED:
        return (
            f"The admitted child did not match the assigned child {signals.expected_child!r}. "
            "Retry using the task's ownership assignment."
        )
    if code is OwnershipFailureCode.DELEGATED_RESOURCE_NOT_ASSIGNED:
        return (
            f"The child prompt did not contain its assigned resource {signals.resource_path}. "
            "Retry while keeping that resource inside the child assignment."
        )
    if code is OwnershipFailureCode.CHILD_RESOURCE_ACCESSED_BY_COORDINATOR:
        return (
            f"The coordinator accessed child-owned resource {signals.resource_path} outside the child "
            "assignment. Retry without opening, parsing, or inspecting that resource in coordinator code."
        )
    if code is OwnershipFailureCode.COORDINATOR_STATE_LEAKED_TO_CHILD:
        return (
            "Coordinator-local state was included in the child assignment. Keep that state in the coordinator "
            "and delegate only the resource work named by the task."
        )
    if code is OwnershipFailureCode.PROHIBITED_CONTROL_ACTION:
        return (
            "The attempt used polling, discovery, or messaging during the ownership decision. Retry using the "
            "event-driven harness contract without those control actions."
        )
    if code is OwnershipFailureCode.CONTINUED_AFTER_DELEGATION:
        return (
            "The coordinator continued with substantive work after admitting the child. Preserve the handle "
            "and yield so the explicit child message can resume the run."
        )
    if code is OwnershipFailureCode.UNNECESSARY_DELEGATION:
        return "This resource is coordinator-owned, but the attempt delegated it. Retry by handling it locally."
    if code is OwnershipFailureCode.COORDINATOR_RESOURCE_NOT_ACCESSED:
        return (
            f"The coordinator-owned resource {signals.resource_path} was not used. Retry the same task locally "
            "without delegating it."
        )
    return (
        "The computed response did not satisfy the requested output contract. Reuse the observed local "
        "evidence, recompute if needed, and return exactly the requested structure."
    )


def feedback_contract_payload(diagnostic: OwnershipFailureDiagnostic) -> dict[str, Any]:
    """Return the typed contract including its exact model-facing message."""

    payload = diagnostic.to_dict()
    payload["message"] = render_ownership_feedback(diagnostic)
    return payload

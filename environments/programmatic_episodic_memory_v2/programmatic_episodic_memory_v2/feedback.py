"""Typed, answer-free causal feedback for programmatic episodic-memory tasks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

FEEDBACK_SCHEMA_VERSION = "programmatic-episodic-memory-v2/causal-feedback/v1"


class MemoryFailureCategory(StrEnum):
    """Stable coarse categories for routing and aggregate diagnostics."""

    ROUTING = "routing"
    RETRIEVAL = "retrieval"
    EXECUTION = "execution"
    STATE_REUSE = "state_reuse"
    OUTPUT_CONTRACT = "output_contract"
    EVENT_SEMANTICS = "event_semantics"


class MemoryFailureCode(StrEnum):
    """Stable leaf-level failure taxonomy.

    Values are part of the machine-readable feedback contract. Add new codes
    rather than changing the meaning of an existing code.
    """

    REQUIRED_TOOL_MISSING = "required_tool_missing"
    UNNECESSARY_TOOL_USE = "unnecessary_tool_use"
    REQUIRED_HISTORY_NOT_RETRIEVED = "required_history_not_retrieved"
    UNNECESSARY_HISTORY_RETRIEVAL = "unnecessary_history_retrieval"
    TOOL_EXECUTION_ERROR = "tool_execution_error"
    RETRIEVAL_TOO_BROAD = "retrieval_too_broad"
    PERSISTENT_STATE_NOT_REUSED = "persistent_state_not_reused"
    OUTPUT_CONTRACT_VIOLATION = "output_contract_violation"
    EVENT_SEMANTICS_MISMATCH = "event_semantics_mismatch"


_CATEGORY_BY_CODE: dict[MemoryFailureCode, MemoryFailureCategory] = {
    MemoryFailureCode.REQUIRED_TOOL_MISSING: MemoryFailureCategory.ROUTING,
    MemoryFailureCode.UNNECESSARY_TOOL_USE: MemoryFailureCategory.ROUTING,
    MemoryFailureCode.REQUIRED_HISTORY_NOT_RETRIEVED: MemoryFailureCategory.RETRIEVAL,
    MemoryFailureCode.UNNECESSARY_HISTORY_RETRIEVAL: MemoryFailureCategory.RETRIEVAL,
    MemoryFailureCode.TOOL_EXECUTION_ERROR: MemoryFailureCategory.EXECUTION,
    MemoryFailureCode.RETRIEVAL_TOO_BROAD: MemoryFailureCategory.RETRIEVAL,
    MemoryFailureCode.PERSISTENT_STATE_NOT_REUSED: MemoryFailureCategory.STATE_REUSE,
    MemoryFailureCode.OUTPUT_CONTRACT_VIOLATION: MemoryFailureCategory.OUTPUT_CONTRACT,
    MemoryFailureCode.EVENT_SEMANTICS_MISMATCH: MemoryFailureCategory.EVENT_SEMANTICS,
}


_FAMILY_SEMANTIC_HINTS = {
    "accepted_requirement": "Distinguish accepted requirements from merely later proposals or rejections.",
    "approval_revocation": "Reconstruct the grant/revoke sequence instead of selecting an isolated approval event.",
    "checkpoint_resume": "Select the latest stable checkpoint, not a later corrupt or merely started checkpoint.",
    "child_result_verification": "Prefer the result that was explicitly verified after the child disagreement.",
    "constraint_update": "Apply the latest valid constraint update and account for superseded values.",
    "context_reset_resume": "Recover the durable state from history rather than guessing from the compacted context.",
    "correction_aggregate": "Apply every relevant correction as well as the base value and deltas.",
    "dataset_provenance": "Resolve provenance status from the authoritative event sequence and preserve the requested source detail.",
    "dependency_next_action": "Follow dependency order and choose the first unfinished or blocked action.",
    "experiment_best_valid_checkpoint": "Compare scores only among checkpoints whose recorded state is valid.",
    "latest_state": "Select the latest active state for the requested key.",
    "multi_key_join": "Join the requested keys from their current records before forming the answer.",
    "ownership_reclaim": "Track the ownership transition through failure and explicit reclaim.",
    "provenance_conflict": "Use the latest evidential verdict together with its source, following the requested output contract.",
    "repeated_lookup_index": "Build the persistent binding index once, then reuse that retained variable for the follow-up.",
    "research_retraction": "Exclude retracted findings and report the surviving supported result.",
    "software_debug_resolution": "Select the fix whose later validation resolved the failure.",
    "stale_note_override": "Treat derived notes as a cache and resolve conflicts against the append-only history.",
    "successful_attempt": "Filter attempts by recorded outcome and preserve the one that succeeded.",
}

_GENERIC_SEMANTIC_HINT = (
    "Apply the task's recorded status, correction, and source-of-truth rules to the "
    "retrieved evidence."
)


@dataclass(frozen=True, slots=True)
class MemoryFailureSignals:
    """Answer-free signals extracted from one attempt.

    The runtime adapter may use the private expected answer to derive the two
    booleans below, but the expected answer itself must never enter this object.
    """

    family: str
    turn_index: int
    history_path: str
    uses_ipython: bool
    requires_history: bool
    repeated_followup: bool
    tool_calls: int
    history_reads: int
    observation_chars: int
    tool_error: bool
    persistent_state_reused: bool | None
    answer_exact: bool
    expected_value_present: bool


@dataclass(frozen=True, slots=True)
class MemoryFailureDiagnostic:
    """Stable diagnostic selected from :class:`MemoryFailureSignals`."""

    code: MemoryFailureCode
    family: str
    turn_index: int
    history_path: str
    tool_calls: int
    history_reads: int
    observation_chars: int
    tool_error: bool
    persistent_state_reused: bool | None
    expected_value_present: bool

    @property
    def category(self) -> MemoryFailureCategory:
        return _CATEGORY_BY_CODE[self.code]

    def to_dict(self) -> dict[str, Any]:
        """Return the audit/routing contract without any expected-answer text."""

        return {
            "schema_version": FEEDBACK_SCHEMA_VERSION,
            "code": self.code.value,
            "category": self.category.value,
            "family": self.family,
            "turn_index": self.turn_index,
            "answer_free": True,
            "retryable": True,
            "resource": {"history_path": self.history_path},
            "evidence": {
                "tool_calls": self.tool_calls,
                "history_reads": self.history_reads,
                "observation_chars": self.observation_chars,
                "tool_error": self.tool_error,
                "persistent_state_reused": self.persistent_state_reused,
                "expected_value_present": self.expected_value_present,
            },
        }


def diagnose_memory_failure(
    signals: MemoryFailureSignals,
) -> MemoryFailureDiagnostic | None:
    """Classify one attempt using a deterministic, prospectively ordered policy."""

    code: MemoryFailureCode | None = None
    if signals.uses_ipython and signals.tool_calls == 0:
        code = MemoryFailureCode.REQUIRED_TOOL_MISSING
    elif not signals.uses_ipython and signals.tool_calls > 0:
        code = MemoryFailureCode.UNNECESSARY_TOOL_USE
    elif (
        signals.requires_history
        and not signals.repeated_followup
        and signals.history_reads == 0
    ):
        code = MemoryFailureCode.REQUIRED_HISTORY_NOT_RETRIEVED
    elif not signals.requires_history and signals.history_reads > 0:
        code = MemoryFailureCode.UNNECESSARY_HISTORY_RETRIEVAL
    elif signals.tool_error:
        code = MemoryFailureCode.TOOL_EXECUTION_ERROR
    elif signals.observation_chars > 4096:
        code = MemoryFailureCode.RETRIEVAL_TOO_BROAD
    elif signals.repeated_followup and signals.persistent_state_reused is not True:
        code = MemoryFailureCode.PERSISTENT_STATE_NOT_REUSED
    elif signals.answer_exact:
        return None
    elif signals.expected_value_present:
        code = MemoryFailureCode.OUTPUT_CONTRACT_VIOLATION
    else:
        code = MemoryFailureCode.EVENT_SEMANTICS_MISMATCH

    return MemoryFailureDiagnostic(
        code=code,
        family=signals.family,
        turn_index=signals.turn_index,
        history_path=signals.history_path,
        tool_calls=signals.tool_calls,
        history_reads=signals.history_reads,
        observation_chars=signals.observation_chars,
        tool_error=signals.tool_error,
        persistent_state_reused=signals.persistent_state_reused,
        expected_value_present=signals.expected_value_present,
    )


def render_memory_feedback(diagnostic: MemoryFailureDiagnostic) -> str:
    """Render deterministic model-facing feedback from a typed diagnostic."""

    code = diagnostic.code
    if code is MemoryFailureCode.REQUIRED_TOOL_MISSING:
        return (
            "That attempt did not use the persistent IPython environment. "
            "Perform the required retrieval or computation, preserve useful state, "
            "and then answer the same request again."
        )
    if code is MemoryFailureCode.UNNECESSARY_TOOL_USE:
        return (
            "The current request is authoritative and self-contained. Do not consult "
            "older workspace history; answer the same request from the current turn."
        )
    if code is MemoryFailureCode.REQUIRED_HISTORY_NOT_RETRIEVED:
        return (
            "This decision depends on earlier events, but the attempt did not inspect "
            f"the append-only history at {diagnostic.history_path}. Retrieve only the "
            "relevant events and retry."
        )
    if code is MemoryFailureCode.UNNECESSARY_HISTORY_RETRIEVAL:
        return (
            "The current user turn is the source of truth, so consulting historical "
            "state can only introduce stale information. Retry without reading history."
        )
    if code is MemoryFailureCode.TOOL_EXECUTION_ERROR:
        return (
            "The attempted operation produced an error. Inspect the actual traceback, "
            "change the failing operation rather than repeating it, and retry using "
            "state that was already established successfully."
        )
    if code is MemoryFailureCode.RETRIEVAL_TOO_BROAD:
        return (
            "The retrieval exposed too much history to the active context. Filter or "
            "parse it programmatically and retry using only the small relevant slice."
        )
    if code is MemoryFailureCode.PERSISTENT_STATE_NOT_REUSED:
        return (
            "The follow-up should reuse the persistent index created during the first "
            "lookup. Use the retained variable instead of rereading history or "
            "guessing from kernel names."
        )
    if code is MemoryFailureCode.OUTPUT_CONTRACT_VIOLATION:
        return (
            "The requested value is present, but the response violates the output "
            "contract. Retry with only the requested value and no explanation."
        )
    if code is MemoryFailureCode.EVENT_SEMANTICS_MISMATCH:
        guidance = _FAMILY_SEMANTIC_HINTS.get(
            diagnostic.family, _GENERIC_SEMANTIC_HINT
        )
        return (
            "The answer is not yet supported by the required event semantics. "
            f"{guidance} Retry the same request."
        )
    raise AssertionError(f"unhandled memory failure code: {code}")


def feedback_contract_payload(diagnostic: MemoryFailureDiagnostic) -> dict[str, Any]:
    """Return the typed audit payload plus its deterministic model-facing message."""

    payload = diagnostic.to_dict()
    payload["message"] = render_memory_feedback(diagnostic)
    return payload

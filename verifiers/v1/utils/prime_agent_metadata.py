"""Guards over Prime Agent metadata preserved through ACP."""

NAMESPACE = "ai.primeintellect.prime-agent"


class MissingAcpMeta(RuntimeError):
    """Required Prime Agent metadata never arrived, so the run cannot be scored."""


def events(trace) -> list[dict]:
    recorded = (trace.info or {}).get("acp_meta")
    if not recorded or NAMESPACE not in recorded:
        raise MissingAcpMeta(
            f"no {NAMESPACE!r} metadata in trace.info['acp_meta']; the harness did "
            "not preserve inbound ACP _meta, or this agent build does not emit it"
        )
    return [event for event in recorded[NAMESPACE] if isinstance(event, dict)]


def field_history(trace, key: str) -> list[dict]:
    """Return every value a metadata key took, in arrival order."""
    history = [
        event[key] for event in events(trace) if isinstance(event.get(key), dict)
    ]
    if not history:
        raise MissingAcpMeta(f"{NAMESPACE}.{key} never appeared in the ACP metadata")
    return history


def subagent_history(trace) -> list[list[dict]]:
    """Return every subagent roster snapshot, in arrival order."""
    seen = events(trace)
    if not seen:
        raise MissingAcpMeta(f"{NAMESPACE} metadata history is empty; nothing to score")
    snapshots = [
        event["subagents"]
        for event in seen
        if isinstance(event.get("subagents"), list)
    ]
    if not snapshots:
        raise MissingAcpMeta(
            f"{NAMESPACE}.subagents never appeared in the ACP metadata"
        )
    return snapshots


def observed_child_statuses(trace) -> dict[str, list[str]]:
    """Return the observed status sequence for each child identifier."""
    statuses: dict[str, list[str]] = {}
    for snapshot in subagent_history(trace):
        for child in snapshot:
            if not isinstance(child, dict):
                continue
            child_id = child.get("id")
            status = child.get("status")
            if not isinstance(child_id, str) or not isinstance(status, str):
                continue
            seen = statuses.setdefault(child_id, [])
            if not seen or seen[-1] != status:
                seen.append(status)
    if not statuses:
        raise MissingAcpMeta("no identifiable subagents appeared in the ACP metadata")
    return statuses


def spawned_and_finished(trace) -> bool:
    """Whether a child ran and reached a terminal state before scoring."""
    terminal = {"completed", "done", "error", "cancelled"}
    return any(
        any(status not in terminal for status in seen) and seen[-1] in terminal
        for seen in observed_child_statuses(trace).values()
    )


def child_tokens_attributed(trace) -> bool:
    """Whether any child reported nonzero token usage."""
    return any(
        isinstance(child, dict)
        and isinstance(child.get("tokenCount"), int)
        and child["tokenCount"] > 0
        for snapshot in subagent_history(trace)
        for child in snapshot
    )


def quiescent_at_end(trace) -> bool:
    """Whether the final snapshot reports no outstanding work."""
    final = field_history(trace, "quiescence")[-1]
    return final.get("outstandingSubagents") == 0 and (
        final.get("remainingAutonomousContinuations") in (0, None)
        or isinstance(final.get("remainingAutonomousContinuations"), int)
    )


def no_outstanding_subagents(trace) -> bool:
    return field_history(trace, "quiescence")[-1].get("outstandingSubagents") == 0


def autonomous_continued(trace) -> bool:
    """Whether autonomous mode engaged rather than remaining inert."""
    return any(
        event.get("enabled")
        and isinstance(event.get("continuationsUsed"), int)
        and event["continuationsUsed"] > 0
        for event in field_history(trace, "autonomous")
    )


def gate_attempted(trace) -> bool:
    return any(
        isinstance(event.get("gateAttempt"), int) and event["gateAttempt"] >= 1
        for event in field_history(trace, "autonomous")
    )


def gate_failure_reported(trace) -> bool:
    """Whether a failing gate surfaced its failure."""
    return any(event.get("gateFailure") for event in field_history(trace, "autonomous"))


def goal_progressed(trace) -> bool:
    return any(
        isinstance(status, str) and status
        for status in (
            event.get("status") for event in field_history(trace, "goal")
        )
    )


def refinement_applied(trace) -> bool:
    """Whether continual-harness state completed with enumerated changes."""
    return any(
        event.get("status") == "complete"
        and isinstance(event.get("changes"), list)
        and event["changes"]
        for event in field_history(trace, "refinement")
    )

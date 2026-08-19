"""Training-only constrained exploration for the natural N1 passive-yield boundary.

This module is deliberately opt-in.  When installed it changes exactly one model request in
eligible natural N1 rollouts: after a verified retained child spawn, when the hidden task graph
contains no remaining coordinator-local work and no child message has arrived, the next
coordinator request is sent upstream with no tools.  The response is still sampled by the model
and the ordinary Prime Agent runtime and frozen hard verifier decide whether the complete
trajectory succeeds.

The scaffold never changes prompts, model responses, task answers, or scoring.  Natural N1
variants with required coordinator-local work are intentionally left untouched and therefore
serve as the anti-overgeneralization control ("do not learn: after every spawn, always yield").
"""

from __future__ import annotations

import ast
import asyncio
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
from verifiers.v1.session import RolloutSession
from verifiers.v1.trace import InterceptRecord
from verifiers.v1.types import AssistantMessage, Request, UserMessage, content_text

SCAFFOLD_SCHEMA_VERSION = "prime-agent/natural-yield-scaffold/v1"
SCAFFOLD_INFO_KEY = "natural_yield_scaffold"
_HANDLER_NAME = "natural_yield_training_scaffold"
_PATCH_MARKER = "_natural_yield_scaffold_v1"


def _data(trace: Any) -> Any | None:
    task = getattr(trace, "task", None)
    return getattr(task, "data", None)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _is_child_context(request: Request) -> bool:
    """The benchmark's private-evidence injection is a hidden, environment-owned role tag."""

    return any(
        isinstance(message, UserMessage)
        and PRIVATE_EVIDENCE_HEADER in content_text(message.content)
        for message in request.messages
    )


def _valid_retained_spawn_node(trace: Any, data: Any) -> int | None:
    """Return the unique clean coordinator spawn that establishes the N1 child handle."""

    nodes = getattr(trace, "nodes", [])
    if not nodes:
        return None
    oracle = _mapping(getattr(data, "oracle", None))
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
        if _branch_root(trace, event.node_index) != coordinator_root or _failed(event.output):
            continue
        try:
            tree = ast.parse(event.code)
        except SyntaxError:
            continue
        assigned_calls = _assigned_call_names(tree)
        valid_spawn = any(
            _call_name(call) == "rlm"
            and id(call) in assigned_calls
            and _spawn_name(call, event.output) == child_name
            and child_path in (_spawn_prompt(call) or "")
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
        )
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
    if data is None or getattr(data, "family", None) != "natural_n1":
        return False, None, "not_natural_n1"
    metadata = _mapping(getattr(data, "generation_metadata", None))
    if metadata.get("graph_variant") != "child_plus_private_state":
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


def _record_fire(trace: Any, request: Request, spawn_node_index: int) -> None:
    data = _data(trace)
    metadata = _mapping(getattr(data, "generation_metadata", None)) if data else {}
    oracle = _mapping(getattr(data, "oracle", None)) if data else {}
    children = oracle.get("children", [])
    child_name = children[0].get("name") if children and isinstance(children[0], dict) else None
    trace.info[SCAFFOLD_INFO_KEY] = {
        "schema_version": SCAFFOLD_SCHEMA_VERSION,
        "mode": "one_turn_no_tools",
        "fired": True,
        "spawn_node_index": spawn_node_index,
        "original_tool_count": len(request.tools or []),
        "graph_variant": metadata.get("graph_variant"),
        "semantic_family": metadata.get("semantic_family"),
        "child_name": child_name,
    }


def install_natural_yield_scaffold() -> bool:
    """Install the request-boundary scaffold once in this evaluator process.

    Returns True when this call installs the wrapper, False when it was already installed.
    The wrapper delegates all ordinary request interception/stops to Verifiers first and then,
    for one eligible natural N1 root request only, removes the tool list before the dialect
    rewrites the native wire body.
    """

    current = RolloutSession.rewrite_request
    if getattr(current, _PATCH_MARKER, False):
        return False
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
            eligible, spawn_node_index, _ = _eligible(self.trace, rewritten)
            if not eligible or spawn_node_index is None:
                return rewritten, records, None
            _record_fire(self.trace, rewritten, spawn_node_index)
            constrained = rewritten.model_copy(update={"tools": None})
            return constrained, [*records, InterceptRecord(handler=_HANDLER_NAME)], None

    setattr(rewrite_request_with_natural_yield_scaffold, _PATCH_MARKER, True)
    RolloutSession.rewrite_request = rewrite_request_with_natural_yield_scaffold
    return True


def scaffolded_yield_node_index(trace: Any) -> int | None:
    """Locate the model-generated no-tool coordinator response caused by the scaffold."""

    info = getattr(trace, "info", {})
    record = info.get(SCAFFOLD_INFO_KEY) if isinstance(info, dict) else None
    if not isinstance(record, dict) or record.get("schema_version") != SCAFFOLD_SCHEMA_VERSION:
        return None
    spawn_node_index = record.get("spawn_node_index")
    if not isinstance(spawn_node_index, int):
        return None
    coordinator_root = _branch_root(trace, 0)
    first_incoming = min(
        (index for index, _, _ in _incoming_messages(trace) if index > spawn_node_index),
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
        "graph_variant": record.get("graph_variant") if isinstance(record, dict) else None,
        "semantic_family": record.get("semantic_family") if isinstance(record, dict) else None,
    }

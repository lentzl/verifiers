"""Executable procedural benchmark for end-to-end Prime Agent coordination."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from pydantic import Field
from subagent_communication_v1.taskset import (
    _assigned_call_names,
    _branch_root,
    _call_name,
    _completion_gate_source,
    _delegated_path_used_outside_spawn,
    _failed,
    _ipython_events,
    _message_sent,
    _spawn_name,
    _spawn_prompt,
)

import verifiers.v1 as vf
from procedural_harness_master_v1.followup_feedback import (
    FEEDBACK_SCHEMA_VERSION,
    FollowupFailureDiagnostic,
    feedback_contract_payload,
    render_followup_feedback,
)
from verifiers.v1.types import AssistantMessage, UserMessage, content_text

Split = Literal["train_gen", "valid_gen", "ood_gen"]
CurriculumRung = Literal[
    "atomic_state",
    "atomic_send",
    "atomic_child_request",
    "atomic_followup",
    "atomic_parallel",
]
COMPLETION_GATE_PATH = "/workspace/.procedural-harness-master/completion_gate.py"
COMPLETION_GATE_FEEDBACK = (
    "completion gate: the end-to-end coordinator task is not complete. Preserve successful "
    "state and existing child handles. React only to visible child messages; do not poll, "
    "invent a result, or spawn replacement children. If a visible failure explicitly transfers "
    "resource ownership, continue from that transition. Otherwise end the next waiting turn "
    "briefly so this active autonomous run can accept the child reply. Return bare strict JSON "
    "only after all required child evidence has arrived."
)


@lru_cache(maxsize=1)
def _generator() -> ModuleType:
    script = (
        Path(__file__).resolve().parents[3]
        / "datasets"
        / "procedural_harness_master_v1"
        / "generate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_procedural_harness_master_v1_generator", script
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load procedural generator from {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ProceduralHarnessMasterData(vf.TaskData):
    episode_id: str
    split: Split
    family: str
    workspace_files: dict[str, str] = Field(default_factory=dict)
    oracle: dict[str, Any]
    generation_metadata: dict[str, Any] = Field(default_factory=dict)


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def _keyword(call: ast.Call, name: str) -> Any:
    value = next((item.value for item in call.keywords if item.arg == name), None)
    return _literal(value) if value is not None else None


def _assigned_state(statement: ast.stmt, name: str, value: Any) -> bool:
    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        assigned_value = _literal(statement.value)
        assigned = assigned_value == value or str(assigned_value) == str(value)
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name and assigned:
                return True
            if (
                isinstance(target, ast.Subscript)
                and _literal(target.slice) == name
                and assigned
            ):
                return True
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Attribute)
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id == "os"
                and target.value.attr == "environ"
                and str(_literal(target.slice)).lower() == name.lower()
                and assigned
            ):
                return True
        if isinstance(statement.value, ast.Dict):
            pairs = zip(statement.value.keys, statement.value.values, strict=True)
            return any(
                _literal(key) == name and _literal(item) == value for key, item in pairs
            )
    return False


def _incoming_messages(trace: vf.Trace) -> list[tuple[int, str, str]]:
    messages = []
    for index, node in enumerate(trace.nodes):
        if not isinstance(node.message, UserMessage):
            continue
        text = content_text(node.message.content)
        match = re.match(r"\[from child:([^\]]+)\]", text)
        if not match:
            continue
        body = text.rsplit("\n\n", 1)[-1].strip()
        explicit_delivery = "Source: agent_message\n" in text
        injected_failure = "Source: benchmark fault injector\n" in text
        visible_child_failure = (
            body.startswith("RLM child failure") or "RESOURCE_UNAVAILABLE" in body
        )
        if explicit_delivery or injected_failure or visible_child_failure:
            messages.append((index, match.group(1), body))
    return messages


def _is_request_message(body: str) -> bool:
    lowered = body.lower()
    return lowered.startswith("need ") or bool(
        re.search(
            r"\b(?:please\s+provide|could\s+you\s+provide|send\s+me|share|supply|requesting)\b"
            r".{0,80}\bmultiplier\b",
            lowered,
        )
    )


def _is_awaited(statement: ast.stmt, call: ast.Call) -> bool:
    return any(
        isinstance(node, ast.Await) and node.value is call
        for node in ast.walk(statement)
    )


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else None
    return None


def _resolve_alias(name: str, aliases: dict[str, str]) -> str:
    seen = set()
    while name not in seen:
        seen.add(name)
        head, separator, tail = name.partition(".")
        target = aliases.get(head)
        if target is None or target == head:
            break
        name = target + (separator + tail if separator else "")
    return name


def _update_aliases(statement: ast.stmt, aliases: dict[str, str]) -> None:
    if isinstance(statement, ast.Import):
        for item in statement.names:
            bound = item.asname or item.name.split(".", 1)[0]
            aliases[bound] = item.name if item.asname else bound
        return
    if isinstance(statement, ast.ImportFrom) and statement.module:
        for item in statement.names:
            if item.name != "*":
                aliases[item.asname or item.name] = f"{statement.module}.{item.name}"
        return
    if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
        return
    targets = (
        statement.targets if isinstance(statement, ast.Assign) else [statement.target]
    )
    source = _dotted_name(statement.value)
    resolved = _resolve_alias(source, aliases) if source else None
    for target in targets:
        if isinstance(target, ast.Name):
            if resolved:
                aliases[target.id] = resolved
            else:
                aliases.pop(target.id, None)


def _final_node(trace: vf.Trace) -> int:
    return next(
        (
            index
            for index in range(len(trace.nodes) - 1, -1, -1)
            if isinstance(trace.nodes[index].message, AssistantMessage)
            and trace.nodes[index].sampled
            and content_text(trace.nodes[index].message.content).strip()
        ),
        len(trace.nodes),
    )


def _answer_exact(reply: str, expected: Any) -> bool:
    try:
        return json.loads(reply.strip()) == expected
    except (AttributeError, json.JSONDecodeError):
        return False


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, float):
        return "float"
    if isinstance(value, int):
        return "int"
    if isinstance(value, list):
        return "list"
    if value is None:
        return "null"
    if isinstance(value, str):
        return "str"
    raise TypeError(f"unsupported JSON answer type: {type(value).__name__}")


def _contract_behavior(
    trace: vf.Trace, data: ProceduralHarnessMasterData
) -> dict[str, float]:
    oracle = data.oracle
    contract = oracle["trajectory_contract"]
    ownership = oracle["resource_ownership"]
    children = oracle["children"]
    expected_names = [child["name"] for child in children]
    child_paths = {
        path
        for path, item in ownership.items()
        if str(item["owner"]).startswith("child:")
    }
    local_paths = {
        path for path, item in ownership.items() if item["owner"] == "coordinator"
    }
    state = oracle.get("coordinator_state", {})

    events = _ipython_events(trace)
    coordinator_root = _branch_root(trace, 0) if trace.nodes else -1
    branch_aware = any(
        _branch_root(trace, event.node_index) != coordinator_root for event in events
    )
    coordinator_events = [
        event
        for event in events
        if not branch_aware or _branch_root(trace, event.node_index) == coordinator_root
    ]
    observed: dict[str, list[float]] = {}
    counts: Counter[str] = Counter()

    def mark(atom: str, position: float) -> None:
        observed.setdefault(atom, []).append(position)

    spawn_records: list[tuple[str | None, str, float, bool]] = []
    poll_positions: list[float] = []
    child_access_positions: list[float] = []
    local_access_positions: list[tuple[str, float]] = []
    parent_sends: list[tuple[float, str | None]] = []
    aliases: dict[str, str] = {}
    retained_state_nodes: dict[str, list[int]] = {name: [] for name in state}

    for event in coordinator_events:
        try:
            tree = ast.parse(event.code)
        except SyntaxError:
            continue
        assigned_calls = _assigned_call_names(tree)
        for statement_index, statement in enumerate(tree.body):
            position = event.node_index + statement_index / 1000
            statement_source = ast.unparse(statement)
            _update_aliases(statement, aliases)
            for name, value in state.items():
                if _assigned_state(statement, name, value):
                    mark("retain_state", position)
                    retained_state_nodes[name].append(event.node_index)
                reused_later = any(
                    isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                    and node.id == name
                    for node in ast.walk(statement)
                ) and any(
                    node_index < event.node_index
                    for node_index in retained_state_nodes[name]
                )
                if reused_later:
                    mark(f"reuse_state:{name}", position)
            for call in (
                node for node in ast.walk(statement) if isinstance(node, ast.Call)
            ):
                raw_call_name = _dotted_name(call.func) or _call_name(call) or ""
                call_name = _resolve_alias(raw_call_name, aliases)
                if call_name == "rlm":
                    child_name = _spawn_name(call, event.output)
                    prompt = _spawn_prompt(call) or ""
                    successful = not _failed(event.output)
                    spawn_records.append((child_name, prompt, position, successful))
                    if successful:
                        counts["spawn_child"] += 1
                        mark("spawn_child", position)
                        if child_name:
                            mark(f"spawn:{child_name}", position)
                        if id(call) in assigned_calls:
                            mark("retain_handle", position)
                            if child_name:
                                mark(f"retain_handle:{child_name}", position)
                    if any(path in prompt for path in local_paths):
                        mark("delegate_coordinator_owned", position)
                    if any(name in prompt for name in state):
                        mark("delegate_coordinator_state", position)
                    continue
                if (
                    call_name.startswith("agent_observe.")
                    or call_name == "agent_observe"
                ):
                    poll_positions.append(position)
                    mark("poll", position)
                if call_name in {
                    "agent_message.list",
                    "rlm.list_subagents",
                    "agent_message.list_agents",
                    "agent_message.listen_for_messages",
                    "agent_message.recv",
                    "agent_message.list_messages",
                }:
                    poll_positions.append(position)
                    mark("poll", position)
                    mark("discover_child", position)
                if call_name == "sleep" or call_name.endswith(".sleep"):
                    poll_positions.append(position)
                    mark("poll", position)
                send_succeeded = _message_sent(event.output) or (
                    _is_awaited(statement, call) and not _failed(event.output)
                )
                if call_name == "agent_message.send" and send_succeeded:
                    receiver = _keyword(call, "receiver_name")
                    parent_sends.append(
                        (position, receiver if isinstance(receiver, str) else None)
                    )
                    counts["parent_to_child_message"] += 1
            for path in child_paths:
                if _delegated_path_used_outside_spawn(statement_source, path):
                    child_access_positions.append(position)
            for path in local_paths:
                if _delegated_path_used_outside_spawn(statement_source, path):
                    local_access_positions.append((path, position))

    for _, position in local_access_positions:
        mark("coordinator_read_local", position)
    manifest_path = next((path for path in local_paths if "verification" in path), None)
    if manifest_path is not None:
        for path, position in local_access_positions:
            if path == manifest_path:
                mark("coordinator_read_verification_manifest", position)

    incoming = _incoming_messages(trace)
    failure_messages = [
        item
        for item in incoming
        if "RESOURCE_UNAVAILABLE" in item[2] or "RLM child failure" in item[2]
    ]
    request_messages = [
        item
        for item in incoming
        if item not in failure_messages and _is_request_message(item[2])
    ]
    result_messages = [
        item
        for item in incoming
        if item not in failure_messages and item not in request_messages
    ]
    counts["child_result_message"] = len(result_messages)
    counts["child_to_parent_message"] = len(request_messages) + len(result_messages)

    for position, name, _ in result_messages:
        mark(f"receive:{name}", float(position))
        mark(f"receive_result:{name}", float(position))
    for position, name, _ in request_messages:
        mark(f"receive_request:{name}", float(position))
    for position, name, _ in failure_messages:
        mark(f"receive_failure:{name}", float(position))

    successful_spawns = [item for item in spawn_records if item[3]]
    first_incoming = min((item[0] for item in incoming), default=None)
    all_expected_spawned = set(expected_names) <= {
        name for name, _, _, successful in successful_spawns if successful and name
    }
    if first_incoming is not None and successful_spawns and all_expected_spawned:
        last_spawn = max(item[2] for item in successful_spawns)
        if last_spawn < first_incoming and not any(
            last_spawn < pos < first_incoming for pos in poll_positions
        ):
            mark("yield", first_incoming - 0.1)
    if first_incoming is not None and not all(
        position < first_incoming for _, _, position, ok in successful_spawns if ok
    ):
        mark("serialized_fanout_wait", float(first_incoming))

    for send_position, receiver in parent_sends:
        prior_requests = [item for item in request_messages if item[0] < send_position]
        target = receiver or (
            prior_requests[-1][1]
            if prior_requests
            else (expected_names[0] if len(expected_names) == 1 else None)
        )
        if target:
            mark(f"send_followup:{target}", send_position)
        if not prior_requests:
            mark("guess_followup_value", send_position)
        later_results = [item for item in result_messages if item[0] > send_position]
        if later_results and not any(
            send_position < pos < later_results[0][0] for pos in poll_positions
        ):
            mark("yield_after_followup", later_results[0][0] - 0.1)

    first_failure = min((item[0] for item in failure_messages), default=None)
    if child_access_positions:
        for position in child_access_positions:
            if first_failure is None or position < first_failure:
                mark("coordinator_read_child_owned", position)
                mark("coordinator_read_child_owned_before_reclaim", position)
                mark("reclaim_without_failure", position)
            else:
                mark("explicit_reclaim", first_failure + 0.1)
                mark("coordinator_read_after_reclaim", position)
                counts["reclaim"] += 1
    child_spawn_counts = Counter(
        name for name, _, _, ok in successful_spawns if ok and name
    )
    if first_failure is not None and any(
        count > 1 for count in child_spawn_counts.values()
    ):
        mark("respawn_same_failed_child", float(first_failure) + 0.2)

    if manifest_path is not None and result_messages:
        try:
            digest = json.loads(data.workspace_files[manifest_path])["expected_digest"]
        except (KeyError, TypeError, json.JSONDecodeError):
            digest = None
        matching = next(
            (item for item in result_messages if digest and digest in item[2]), None
        )
        if matching is not None:
            mark("verify_child_digest", matching[0] + 0.1)

    final_position = float(_final_node(trace))
    if (trace.last_reply or "").strip():
        mark("final_answer", final_position)
    final_exact = _answer_exact(trace.last_reply, oracle["final_answer"])
    if (
        final_exact
        and "verify_child_digest" not in observed
        and oracle["expected_route"] == "single_verify"
    ):
        mark("accept_unverified_child_result", final_position)

    required = contract["required_atoms"]
    forbidden = contract["forbidden_atoms"]
    missing = [atom for atom in required if atom not in observed]
    violations = [atom for atom in forbidden if atom in observed]
    ordering_failures = []
    for edge in contract["ordering"]:
        before, after = edge["before"], edge["after"]
        if (
            before not in observed
            or after not in observed
            or min(observed[before]) >= max(observed[after])
        ):
            ordering_failures.append(f"{before}->{after}")
    cardinality_failures = [
        key
        for key, expected in contract["cardinality"].items()
        if counts[key] != expected
    ]
    required_fraction = (
        (len(required) - len(missing)) / len(required) if required else 1.0
    )
    ordering_fraction = (
        (len(contract["ordering"]) - len(ordering_failures)) / len(contract["ordering"])
        if contract["ordering"]
        else 1.0
    )
    cardinality_fraction = (
        (len(contract["cardinality"]) - len(cardinality_failures))
        / len(contract["cardinality"])
        if contract["cardinality"]
        else 1.0
    )
    bootstrap_progress = (
        float(final_exact)
        * float(not violations)
        * required_fraction
        * ordering_fraction
        * cardinality_fraction
    )
    event_control_progress = (
        float(not violations)
        * required_fraction
        * ordering_fraction
        * cardinality_fraction
    )
    hard_gate = bool(
        final_exact
        and not missing
        and not violations
        and not ordering_failures
        and not cardinality_failures
    )
    return {
        "harness_score": float(hard_gate),
        "final_answer_exact": float(final_exact),
        "all_required_atoms": float(not missing),
        "no_forbidden_atoms": float(not violations),
        "ordering_satisfied": float(not ordering_failures),
        "cardinality_exact": float(not cardinality_failures),
        "required_atoms_fraction": required_fraction,
        "ordering_fraction": ordering_fraction,
        "cardinality_fraction": cardinality_fraction,
        "bootstrap_progress": bootstrap_progress,
        "event_control_progress": event_control_progress,
        "missing_required_atoms": float(len(missing)),
        "forbidden_atom_violations": float(len(violations)),
        "ordering_failures": float(len(ordering_failures)),
        "cardinality_failures": float(len(cardinality_failures)),
        "observed_atoms": float(len(observed)),
    }


def _followup_feedback_diagnostic(
    trace: vf.Trace,
    data: ProceduralHarnessMasterData,
) -> FollowupFailureDiagnostic | None:
    """Locate the failed coordinator response immediately after a child request."""

    if data.family != "atomic_followup":
        return None
    if _contract_behavior(trace, data)["harness_score"] == 1.0:
        return None

    request = next(
        (item for item in _incoming_messages(trace) if _is_request_message(item[2])),
        None,
    )
    if request is None or not trace.nodes:
        return None
    request_node_index, child_name, _ = request
    coordinator_root = _branch_root(trace, 0)
    target_node_index = next(
        (
            index
            for index in range(request_node_index + 1, len(trace.nodes))
            if isinstance(trace.nodes[index].message, AssistantMessage)
            and trace.nodes[index].sampled
            and _branch_root(trace, index) == coordinator_root
        ),
        None,
    )
    if target_node_index is None:
        return None
    coordinator_turns = [
        index
        for index, node in enumerate(trace.nodes[: target_node_index + 1])
        if isinstance(node.message, AssistantMessage)
        and node.sampled
        and _branch_root(trace, index) == coordinator_root
    ]
    return FollowupFailureDiagnostic(
        child_name=child_name,
        request_node_index=request_node_index,
        target_node_index=target_node_index,
        turn_index=len(coordinator_turns) - 1,
    )


def _record_followup_feedback(
    trace: vf.Trace,
    data: ProceduralHarnessMasterData,
) -> bool:
    diagnostic = _followup_feedback_diagnostic(trace, data)
    if diagnostic is None:
        return False
    feedback = render_followup_feedback(diagnostic)
    trace.info["feedback"] = feedback
    trace.info["feedback_contract"] = feedback_contract_payload(diagnostic)
    return True


def keep_atomic_child_request_coordinator_actions(
    trace: vf.Trace,
    *,
    tool_start_token_id: int = 248058,
    tool_end_token_id: int = 248059,
    thinking_end_token_id: int = 248069,
    message_end_token_id: int = 248046,
) -> list[list[bool]]:
    """Select executable coordinator actions from a successful request trace.

    Before the explicit child request, serialized tool calls retain state and
    spawn the child; the last visible no-tool response yields control. After
    the request, only the final visible no-tool response is retained. Qwen's
    free-form reasoning and every child-branch token remain context-only.
    """
    if not trace.nodes:
        return []
    data = trace.task.data
    if not isinstance(data, ProceduralHarnessMasterData):
        data = ProceduralHarnessMasterData.model_validate(
            data.model_dump()
        )
    if data.family != "atomic_child_request":
        return []
    primary_root = next(
        (index for index, node in enumerate(trace.nodes) if node.parent is None),
        None,
    )
    if primary_root is None:
        return []
    lineage_by_node: dict[int, set[str | None]] = {}
    for call in trace.calls:
        if call.node is not None:
            lineage_by_node.setdefault(call.node, set()).add(call.client_session_id)
    primary_sessions = {
        session_id
        for node_index, sessions in lineage_by_node.items()
        if _branch_root(trace, node_index) == primary_root
        for session_id in sessions
        if session_id is not None
    }
    if len(primary_sessions) != 1:
        raise ValueError(
            "atomic child-request action filtering requires exactly one primary client session"
        )
    primary_session = next(iter(primary_sessions))
    coordinator_nodes = {
        node_index
        for node_index, sessions in lineage_by_node.items()
        if sessions == {primary_session}
    }
    child_request_index = next(
        (
            index
            for index, node in enumerate(trace.nodes)
            if not node.sampled
            and isinstance(node.message, UserMessage)
            and content_text(node.message.content).lstrip().startswith("[from child:")
        ),
        None,
    )

    expected_children = data.oracle["children"]
    state = data.oracle.get("coordinator_state", {})

    def successful_spawn_node() -> int | None:
        if child_request_index is None or len(expected_children) != 1 or not state:
            return None
        expected_name = expected_children[0]["name"]
        eligible: list[int] = []
        for event in _ipython_events(trace):
            if event.node_index not in coordinator_nodes or event.node_index >= child_request_index:
                continue
            try:
                tree = ast.parse(event.code)
            except SyntaxError:
                continue
            rlm_calls = [
                call
                for call in ast.walk(tree)
                if isinstance(call, ast.Call) and (_dotted_name(call.func) or _call_name(call)) == "rlm"
            ]
            if len(rlm_calls) != 1:
                continue
            call = rlm_calls[0]
            prompt = _spawn_prompt(call) or ""
            request_protocol = (
                "agent_message.send" in prompt
                and "need multiplier" in prompt.lower()
                and bool(
                    re.search(
                        r"receiver_role\s*=\s*['\"]parent['\"]",
                        prompt,
                    )
                )
            )
            retains_state = all(
                any(_assigned_state(statement, name, value) for statement in tree.body)
                for name, value in state.items()
            )
            assigned_calls = _assigned_call_names(tree)
            successful_spawn = (
                not _failed(event.output)
                and "RLMSpawnHandle" in event.output
                and _spawn_name(call, event.output) == expected_name
                and id(call) in assigned_calls
            )
            if not (request_protocol and retains_state and successful_spawn):
                continue

            allowed_statements: set[int] = {
                id(statement)
                for statement in tree.body
                if any(_assigned_state(statement, name, value) for name, value in state.items())
                or call in ast.walk(statement)
            }
            handle_names = {
                target.id
                for statement in tree.body
                if call in ast.walk(statement)
                and isinstance(statement, (ast.Assign, ast.AnnAssign))
                for target in (
                    statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                )
                if isinstance(target, ast.Name)
            }
            for statement in tree.body:
                if (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Name)
                    and statement.value.id in handle_names
                ):
                    allowed_statements.add(id(statement))
            if len(allowed_statements) != len(tree.body):
                continue
            eligible.append(event.node_index)
        return eligible[0] if len(eligible) == 1 else None

    def tool_mask(node: Any) -> list[bool]:
        selected = [False] * len(node.token_ids)
        cursor = 0
        while True:
            try:
                start = node.token_ids.index(tool_start_token_id, cursor)
                end = node.token_ids.index(tool_end_token_id, start + 1)
            except ValueError:
                break
            for index in range(start, end + 1):
                selected[index] = bool(node.mask[index])
            if end + 1 < len(node.token_ids) and node.token_ids[end + 1] == message_end_token_id:
                selected[end + 1] = bool(node.mask[end + 1])
            cursor = end + 1
        return selected

    def visible_mask(node: Any) -> list[bool]:
        if isinstance(node.message, AssistantMessage) and node.message.tool_calls:
            return [False] * len(node.token_ids)
        try:
            start = len(node.token_ids) - 1 - node.token_ids[::-1].index(thinking_end_token_id)
        except ValueError:
            start = -1
        return [
            bool(sampled and index > start)
            for index, sampled in enumerate(node.mask)
        ]

    selected_by_node: dict[int, list[bool]] = {}
    spawn_node_index = successful_spawn_node()
    if child_request_index is not None and spawn_node_index is not None:
        visible_pre_request = next(
            (
                index
                for index in range(child_request_index - 1, -1, -1)
                if index in coordinator_nodes
                and isinstance(trace.nodes[index].message, AssistantMessage)
                and not trace.nodes[index].message.tool_calls
            ),
            None,
        )
        visible_post_request = next(
            (
                index
                for index in range(len(trace.nodes) - 1, child_request_index, -1)
                if index in coordinator_nodes
                and isinstance(trace.nodes[index].message, AssistantMessage)
                and not trace.nodes[index].message.tool_calls
            ),
            None,
        )
        intervening_tool_call = any(
            index in coordinator_nodes
            and isinstance(trace.nodes[index].message, AssistantMessage)
            and trace.nodes[index].message.tool_calls
            for index in range(spawn_node_index + 1, child_request_index)
        )
        if (
            visible_pre_request is None
            or visible_pre_request <= spawn_node_index
            or visible_post_request is None
            or intervening_tool_call
        ):
            spawn_node_index = None
        for node_index in coordinator_nodes:
            node = trace.nodes[node_index]
            selected = [False] * len(node.token_ids)
            if node_index == spawn_node_index:
                selected = tool_mask(node)
            if spawn_node_index is not None and node_index in {
                visible_pre_request,
                visible_post_request,
            }:
                visible = visible_mask(node)
                selected = [
                    executable or text
                    for executable, text in zip(selected, visible, strict=True)
                ]
            selected_by_node[id(node)] = selected

    masks: list[list[bool]] = []
    trained_nodes: set[int] = set()
    for branch in trace.branches:
        has_trainable_tokens = any(
            node.sampled and any(node.mask) and id(node) not in trained_nodes
            for node in branch.nodes
        )
        if not has_trainable_tokens:
            continue
        branch_mask: list[bool] = []
        for node in branch.nodes:
            span = len(node.token_ids)
            is_new_trainable = (
                node.sampled and any(node.mask) and id(node) not in trained_nodes
            )
            if is_new_trainable:
                trained_nodes.add(id(node))
            node_mask = (
                selected_by_node.get(id(node), [False] * span)
                if is_new_trainable
                else [False] * span
            )
            branch_mask.extend(node_mask)
        masks.append(branch_mask)
    return masks


def keep_followup_feedback_response(trace: vf.Trace) -> list[list[bool]]:
    """Select only the sampled response named by trusted follow-up feedback."""

    contract = trace.info.get("feedback_contract")
    target_index = (
        contract.get("target_node_index") if isinstance(contract, dict) else None
    )
    trusted = (
        isinstance(contract, dict)
        and contract.get("schema_version") == FEEDBACK_SCHEMA_VERSION
        and contract.get("answer_free") is True
        and contract.get("retryable") is True
        and contract.get("code") == "reply_to_child_request"
        and isinstance(target_index, int)
        and 0 <= target_index < len(trace.nodes)
    )
    target = trace.nodes[target_index] if trusted else None
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
            branch_mask.extend(sampled and keep for sampled in node.mask)
        if has_trainable_tokens:
            masks.append(branch_mask)
    return masks


class ProceduralHarnessMasterTaskConfig(vf.TaskConfig):
    reward_mode: Literal["hard", "bootstrap", "event_control"] = "hard"


class ProceduralHarnessMasterTask(
    vf.Task[ProceduralHarnessMasterData, vf.State, ProceduralHarnessMasterTaskConfig]
):
    NEEDS_CONTAINER = True

    @property
    def key(self) -> str:
        return self.data.episode_id

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        deferred = set()
        if self.data.family == "reclaim":
            deferred = {
                child["resource_path"] for child in self.data.oracle["children"]
            }
        directories = sorted(
            {
                str(Path(COMPLETION_GATE_PATH).parent),
                *(str(Path(path).parent) for path in self.data.workspace_files),
            }
        )
        result = await runtime.run(["mkdir", "-p", *directories], {})
        if result.exit_code != 0:
            raise RuntimeError(
                f"procedural workspace setup failed: {result.stderr[-500:]}"
            )
        for path, contents in self.data.workspace_files.items():
            if path not in deferred:
                await runtime.write(path, contents.encode())
        required_child_messages = {}
        for child in self.data.oracle["children"]:
            contract = child.get("message_contract")
            count = len(contract) if isinstance(contract, list) else 1
            required_child_messages[child["name"]] = count
        family = (
            "followup"
            if self.data.family in {"followup", "atomic_followup"}
            else "direct"
        )
        final_answer = self.data.oracle["final_answer"]
        await runtime.write(
            COMPLETION_GATE_PATH,
            _completion_gate_source(
                tuple(final_answer),
                family,
                expected_types={
                    key: _json_type(value) for key, value in final_answer.items()
                },
                required_child_messages=required_child_messages,
                feedback=COMPLETION_GATE_FEEDBACK,
            ).encode(),
        )

    @vf.reward(weight=1.0)
    async def harness_score(self, trace: vf.Trace) -> float:
        behavior = _contract_behavior(trace, self.data)
        if self.config.reward_mode == "bootstrap":
            return behavior["harness_score"] + 0.1 * behavior["bootstrap_progress"]
        if self.config.reward_mode == "event_control":
            return behavior["harness_score"] + behavior["event_control_progress"]
        return behavior["harness_score"]

    @vf.metric
    async def harness_contract(self, trace: vf.Trace) -> dict[str, float]:
        return _contract_behavior(trace, self.data)


class ProceduralHarnessMasterConfig(vf.TasksetConfig):
    task: ProceduralHarnessMasterTaskConfig = ProceduralHarnessMasterTaskConfig()
    split: Split = "train_gen"
    count: int = Field(64, ge=1)
    start_index: int = Field(0, ge=0)
    master_seed: int = 20260816
    families: tuple[str, ...] | None = None
    curriculum_rung: CurriculumRung | None = None
    record_causal_feedback: bool = False


class ProceduralHarnessMasterTaskset(
    vf.Taskset[ProceduralHarnessMasterTask, ProceduralHarnessMasterConfig]
):
    def load(self) -> list[ProceduralHarnessMasterTask]:
        generator = _generator()
        if self.config.curriculum_rung is not None and self.config.families is not None:
            raise ValueError("curriculum_rung and families are mutually exclusive")
        tasks = []
        index = self.config.start_index
        while len(tasks) < self.config.count:
            if self.config.curriculum_rung is None:
                row = generator.generate_episode(
                    self.config.split, index, self.config.master_seed
                )
            else:
                row = generator.generate_curriculum_episode(
                    self.config.curriculum_rung,
                    self.config.split,
                    index,
                    self.config.master_seed,
                )
            index += 1
            family = row["metadata"]["episode_family"]
            if self.config.families is not None and family not in self.config.families:
                continue
            public = row["public"]
            tasks.append(
                ProceduralHarnessMasterTask(
                    ProceduralHarnessMasterData(
                        idx=len(tasks),
                        name=row["episode_id"],
                        prompt=public["user_prompt"],
                        system_prompt=public["system_prompt"],
                        episode_id=row["episode_id"],
                        split=row["split"],
                        family=family,
                        workspace_files=public["workspace_files"],
                        oracle=row["oracle"],
                        generation_metadata=row["metadata"],
                    ),
                    self.config.task,
                )
            )
        return tasks


class ProceduralHarnessMasterEnv(vf.SingleAgentEnv):
    """Inject the OOD reclaim transition without exposing hidden oracle values."""

    async def run(self, task, agents) -> None:
        if task.data.family != "reclaim":
            async with agents.agent.interaction(task) as interaction:
                await interaction.turn()
                if self.taskset.config.record_causal_feedback:
                    _record_followup_feedback(interaction.trace, task.data)
            return
        child = task.data.oracle["children"][0]
        path = child["resource_path"]
        fault = task.data.oracle["fault_plan"]
        async with (
            agents.agent.provision(task) as runtime,
            agents.agent.interaction(task, runtime=runtime) as interaction,
        ):
            first = await interaction.turn()
            if first.terminated:
                return
            await runtime.write(path, task.data.workspace_files[path].encode())
            existing_failure = any(
                "RESOURCE_UNAVAILABLE" in body
                for _, _, body in _incoming_messages(interaction.trace)
            )
            if existing_failure:
                message = (
                    f"[benchmark runtime]\nThe explicit failure transferred ownership of {path} "
                    "to the coordinator. Continue from retained state."
                )
            else:
                message = (
                    f"[from child:{child['name']}]\n"
                    "Agent-to-agent message received.\n"
                    "Source: benchmark fault injector\n"
                    f"Message id: fault-{task.data.episode_id}\n\n"
                    f"{fault['message']}"
                )
            await interaction.turn(message)

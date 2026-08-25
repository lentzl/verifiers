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

import verifiers.v1 as vf
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
from verifiers.v1.types import AssistantMessage, UserMessage, content_text

from procedural_harness_master_v1.followup_feedback import (
    FEEDBACK_SCHEMA_VERSION,
    FollowupFailureDiagnostic,
    feedback_contract_payload,
    render_followup_feedback,
)
from procedural_harness_master_v1.natural_yield_feedback import (
    FEEDBACK_SCHEMA_VERSION as NATURAL_YIELD_FEEDBACK_SCHEMA_VERSION,
)
from procedural_harness_master_v1.natural_yield_feedback import (
    NaturalYieldFailureDiagnostic,
    render_natural_yield_feedback,
)
from procedural_harness_master_v1.natural_yield_feedback import (
    feedback_contract_payload as natural_yield_feedback_contract_payload,
)

Split = Literal["train_gen", "valid_gen", "ood_gen"]
CurriculumRung = Literal[
    "atomic_state",
    "atomic_send",
    "atomic_child_request",
    "atomic_followup",
    "atomic_parallel",
    "natural_n1",
    "natural_n1a",
    "natural_n1a_local",
    "natural_n1b",
    "natural_direct_control",
    "natural_n2",
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
PRIVATE_EVIDENCE_HEADER = "[private evidence supplied to this reviewer]"
CHILD_ACTION_SCAFFOLD_HEADER = "[training-only child action scaffold]"
PRIVILEGED_HINT_HEADER = "[privileged strategy hint]"
PRIVILEGED_BOOTSTRAP_HEADER = "[training-only environment scaffold]"


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


def _assignment_names(statement: ast.stmt) -> set[str]:
    if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
        return set()
    targets = (
        statement.targets if isinstance(statement, ast.Assign) else [statement.target]
    )
    return {
        node.id
        for target in targets
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }


def _assigned_from_path(statement: ast.stmt, path: str) -> set[str]:
    if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
        return set()
    if not any(
        isinstance(node, ast.Constant) and node.value == path
        for node in ast.walk(statement.value)
    ):
        return set()
    if not any(isinstance(node, ast.Call) for node in ast.walk(statement.value)):
        return set()
    return _assignment_names(statement)


def _response_proposes_delegation(response: vf.Response) -> bool:
    for tool_call in response.message.tool_calls or []:
        if tool_call.name != "ipython" or not isinstance(tool_call.arguments, str):
            continue
        try:
            arguments = json.loads(tool_call.arguments)
        except json.JSONDecodeError:
            continue
        code = arguments.get("code") if isinstance(arguments, dict) else None
        if not isinstance(code, str):
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        if any(
            _call_name(call) == "rlm"
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
        ):
            return True
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
        # Prime Agent emits a terminal lifecycle notification on the same visible
        # agent-message channel when a child exits without calling send().  It is
        # not child evidence and must not satisfy receive/cardinality contracts.
        # In particular, child-role GRPO must rank an explicit child reply above
        # an empty completion instead of assigning both the same protocol reward.
        completed_without_reply = bool(
            re.match(r"^RLM child .* completed without sending a reply\b", body)
        )
        if (
            explicit_delivery or injected_failure or visible_child_failure
        ) and not completed_without_reply:
            messages.append((index, match.group(1), body))
    return messages


def _sampled_child_ipython_codes(trace: vf.Trace) -> list[str]:
    """Return executable IPython payloads sampled on non-root agent branches."""

    if not trace.nodes:
        return []
    coordinator_root = _branch_root(trace, 0)
    codes = []
    for index, node in enumerate(trace.nodes):
        if (
            _branch_root(trace, index) == coordinator_root
            or not isinstance(node.message, AssistantMessage)
            or not node.sampled
        ):
            continue
        for call in node.message.tool_calls or []:
            if call.name != "ipython":
                continue
            try:
                arguments = json.loads(call.arguments)
            except (TypeError, json.JSONDecodeError):
                continue
            code = arguments.get("code") if isinstance(arguments, dict) else None
            if isinstance(code, str) and code.strip():
                codes.append(code)
    return codes


def _child_action_progress(trace: vf.Trace, expected_values: set[str]) -> float:
    """Grade the causal bridge from a child tool action to an explicit parent send."""

    progress = 0.0
    for code in _sampled_child_ipython_codes(trace):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        progress = max(progress, 0.25)
        for statement in tree.body:
            for call in (
                node for node in ast.walk(statement) if isinstance(node, ast.Call)
            ):
                if _dotted_name(call.func) != "agent_message.send" or not _is_awaited(
                    statement, call
                ):
                    continue
                progress = max(progress, 0.5)
                receiver = _keyword(call, "receiver_role")
                if receiver != "parent":
                    continue
                progress = max(progress, 0.75)
                message = _message_argument(call)
                value = _literal(message) if message is not None else None
                if (
                    isinstance(message, ast.Call)
                    and isinstance(message.func, ast.Name)
                    and message.func.id == "str"
                    and len(message.args) == 1
                ):
                    value = _literal(message.args[0])
                if value is not None and str(value) in expected_values:
                    progress = 1.0
    return progress


def _is_request_message(
    body: str, request_terms: tuple[str, ...] = ("multiplier",)
) -> bool:
    lowered = body.lower()
    normalized = lowered.replace("_", " ")
    terms = tuple(term.lower().replace("_", " ") for term in request_terms)
    request_cue = re.search(
        r"\b(?:need|please\s+provide|could\s+you\s+(?:provide|send)|send\s+me|"
        r"share|supply|request(?:ing)?|may\s+i\s+have|what\s+is)\b",
        normalized,
    )
    return lowered.startswith("need ") or bool(
        request_cue and any(term in normalized for term in terms)
    )


def _contains_state_value(prompt: str, value: Any) -> bool:
    if isinstance(value, int) and not isinstance(value, bool):
        return bool(re.search(rf"(?<!\d){re.escape(str(value))}(?!\d)", prompt))
    if isinstance(value, str):
        return value in prompt
    return False


def _message_argument(call: ast.Call) -> ast.AST | None:
    keyword = next(
        (item.value for item in call.keywords if item.arg == "message"), None
    )
    return keyword if keyword is not None else (call.args[0] if call.args else None)


def _references_coordinator_state(node: ast.AST | None, state: dict[str, Any]) -> bool:
    if node is None:
        return False
    for item in ast.walk(node):
        if isinstance(item, ast.Name) and item.id in state:
            return True
        if isinstance(item, ast.Subscript) and _literal(item.slice) in state:
            return True
        value = _literal(item)
        if value is not None and any(
            value == expected
            or str(value) == str(expected)
            or (isinstance(value, str) and _contains_state_value(value, expected))
            for expected in state.values()
        ):
            return True
    return False


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
    seen_heads: set[str] = set()
    while True:
        head, separator, tail = name.partition(".")
        if head in seen_heads:
            break
        seen_heads.add(head)
        target = aliases.get(head)
        if target is None or target == head:
            break
        name = target + (separator + tail if separator else "")
    return name


def _record_path_aliases(
    statement: ast.stmt,
    paths: set[str],
    path_aliases: dict[str, set[str]],
) -> None:
    if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
        return
    targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
    source = ast.unparse(statement.value)
    matched = {path for path in paths if path in source}
    if not matched:
        matched = {
            path
            for name in {
                node.id for node in ast.walk(statement.value) if isinstance(node, ast.Name)
            }
            for path in path_aliases.get(name, set())
        }
    for target in targets:
        if isinstance(target, ast.Name):
            if matched:
                path_aliases[target.id] = matched
            else:
                path_aliases.pop(target.id, None)


def _statement_accesses_owned_path(
    statement: ast.stmt,
    path: str,
    path_aliases: dict[str, set[str]],
    aliases: dict[str, str],
) -> bool:
    """Detect resource access without treating instruction construction as a read."""

    harmless_calls = {
        "rlm",
        "agent_message.send",
        "print",
        "repr",
        "str",
        "json.dumps",
    }
    for call in (node for node in ast.walk(statement) if isinstance(node, ast.Call)):
        raw_name = _dotted_name(call.func) or _call_name(call) or ""
        call_name = _resolve_alias(raw_name, aliases)
        if call_name in harmless_calls:
            continue
        source = ast.unparse(call)
        literal_match = path in source
        alias_match = any(
            path in path_aliases.get(node.id, set())
            for node in ast.walk(call)
            if isinstance(node, ast.Name)
        )
        if literal_match or alias_match:
            return True
    return False


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
    persistence_lease = oracle.get("persistence_lease", {})
    capture_path = (
        persistence_lease.get("path")
        if isinstance(persistence_lease, dict)
        else None
    )

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
    parent_sends: list[tuple[float, str | None, bool]] = []
    aliases: dict[str, str] = {}
    path_aliases: dict[str, set[str]] = {}
    retained_state_nodes: dict[str, list[int]] = {name: [] for name in state}
    captured_state_nodes: dict[str, list[int]] = {}

    for event in coordinator_events:
        mark("coordinator_tool", float(event.node_index))
        try:
            tree = ast.parse(event.code)
        except SyntaxError:
            continue
        assigned_calls = _assigned_call_names(tree)
        for statement_index, statement in enumerate(tree.body):
            position = event.node_index + statement_index / 1000
            _update_aliases(statement, aliases)
            _record_path_aliases(statement, child_paths | local_paths, path_aliases)
            if isinstance(capture_path, str):
                for name in _assigned_from_path(statement, capture_path):
                    mark("capture_state", position)
                    captured_state_nodes.setdefault(name, []).append(event.node_index)
            for name, node_indices in captured_state_nodes.items():
                reused_later = any(
                    isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                    and node.id == name
                    for node in ast.walk(statement)
                ) and any(node_index < event.node_index for node_index in node_indices)
                if reused_later:
                    mark("reuse_captured_state", position)
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
                    if any(
                        _contains_state_value(prompt, value) for value in state.values()
                    ):
                        mark("delegate_private_value", position)
                    continue
                if (
                    call_name.startswith("agent_observe.")
                    or call_name == "agent_observe"
                ):
                    poll_positions.append(position)
                    mark("poll", position)
                if call_name == "rlm.list_subagents" or (
                    call_name.startswith("agent_message.")
                    and call_name != "agent_message.send"
                ):
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
                        (
                            position,
                            receiver if isinstance(receiver, str) else None,
                            _references_coordinator_state(
                                _message_argument(call), state
                            ),
                        )
                    )
                    counts["parent_to_child_message"] += 1
            for path in child_paths:
                if _statement_accesses_owned_path(
                    statement, path, path_aliases, aliases
                ):
                    child_access_positions.append(position)
            for path in local_paths:
                if _statement_accesses_owned_path(
                    statement, path, path_aliases, aliases
                ):
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
    request_terms = tuple(oracle.get("request_terms", ("multiplier",)))
    request_messages = [
        item
        for item in incoming
        if item not in failure_messages and _is_request_message(item[2], request_terms)
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
    lease_record = trace.info.get("persistence_lease")
    if (
        successful_spawns
        and isinstance(lease_record, dict)
        and lease_record.get("closed") is True
    ):
        mark(
            "persistence_lease_closed",
            min(position for _, _, position, _ in successful_spawns) + 0.0001,
        )
    boundary_record = trace.info.get("causal_context_boundary")
    if (
        successful_spawns
        and isinstance(boundary_record, dict)
        and boundary_record.get("applied") is True
    ):
        mark(
            "context_boundary",
            min(position for _, _, position, _ in successful_spawns) + 0.0001,
        )
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
    last_spawn_node = (
        int(max(position for _, _, position, ok in successful_spawns if ok))
        if successful_spawns
        else None
    )
    post_spawn_tool_before_child = bool(
        last_spawn_node is not None
        and any(
            event.node_index > last_spawn_node
            and (first_incoming is None or event.node_index < first_incoming)
            for event in coordinator_events
        )
    )
    if post_spawn_tool_before_child:
        mark("post_spawn_tool_before_child", float(last_spawn_node) + 0.01)
    if first_incoming is not None and not all(
        position < first_incoming for _, _, position, ok in successful_spawns if ok
    ):
        mark("serialized_fanout_wait", float(first_incoming))

    for send_position, receiver, correct_value in parent_sends:
        prior_requests = [item for item in request_messages if item[0] < send_position]
        target = receiver or (
            prior_requests[-1][1]
            if prior_requests
            else (expected_names[0] if len(expected_names) == 1 else None)
        )
        if target:
            mark(f"send_followup:{target}", send_position)
            if correct_value:
                mark(f"send_followup_value:{target}", send_position)
            else:
                mark("wrong_followup_value", send_position)
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
    # Reward only the longest causal prefix of the declared protocol.  A later atom
    # by itself is not progress: for example, returning a premature final answer
    # before spawning the required child must not outrank a clean no-op.  Contracts
    # list required atoms in acquisition order; atoms that occur at the same event
    # (notably spawn + retained handle) are both admitted before the next boundary.
    causal_prefix_length = 0
    causal_prefix_position = float("-inf")
    for atom in required:
        positions = observed.get(atom)
        if not positions:
            break
        position = min(positions)
        if position < causal_prefix_position:
            break
        causal_prefix_length += 1
        causal_prefix_position = position
    causal_prefix_fraction = (
        causal_prefix_length / len(required) if required else 1.0
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
    # Early role learning needs credit for the first clean protocol step. Ordering
    # and cardinality refine that progress rather than erasing it, while the causal
    # prefix prevents out-of-order terminal behavior from receiving baby-step credit.
    # Keep forbidden behavior as a hard zero and preserve a unit score only for a
    # complete clean trajectory.
    event_control_progress = (
        float(not violations)
        * causal_prefix_fraction
        * (1.0 + ordering_fraction + cardinality_fraction)
        / 3.0
    )
    expected_child_values = {
        str(child["expected_result"])
        for child in children
        if isinstance(child, dict) and "expected_result" in child
    }
    child_action_progress = _child_action_progress(trace, expected_child_values)
    # Before a child delivery exists, reward a syntactically valid, correctly routed
    # send as a small causal bridge.  The bonus decays as the declared protocol prefix
    # fills and is zero on a complete trajectory, so terminal reward remains unchanged.
    child_action_bridge = (
        float(not violations)
        * 0.25
        * child_action_progress
        * (1.0 - causal_prefix_fraction)
    )
    local_work_required = "coordinator_read_local" in required
    local_work_before_yield = float(
        not local_work_required
        or (
            "coordinator_read_local" in observed
            and "yield" in observed
            and min(observed["coordinator_read_local"]) < max(observed["yield"])
        )
    )
    premature_yield_before_local_work = float(
        local_work_required
        and "yield" in observed
        and not local_work_before_yield
    )
    forbidden_post_spawn_tool_before_child = float(
        data.family in {"natural_n1", "natural_n1a", "natural_n1b"}
        and not local_work_required
        and post_spawn_tool_before_child
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
        "causal_prefix_fraction": causal_prefix_fraction,
        "ordering_fraction": ordering_fraction,
        "cardinality_fraction": cardinality_fraction,
        "bootstrap_progress": bootstrap_progress,
        "event_control_progress": event_control_progress,
        "child_action_progress": child_action_progress,
        "child_action_bridge": child_action_bridge,
        "local_work_before_yield": local_work_before_yield,
        "premature_yield_before_local_work": premature_yield_before_local_work,
        "forbidden_post_spawn_tool_before_child": (
            forbidden_post_spawn_tool_before_child
        ),
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


def _natural_yield_feedback_diagnostic(
    trace: vf.Trace,
    data: ProceduralHarnessMasterData,
) -> NaturalYieldFailureDiagnostic | None:
    """Locate the first coordinator tool call after a valid natural N1 spawn."""

    if data.family != "natural_n1":
        return None
    if _contract_behavior(trace, data)["harness_score"] == 1.0 or not trace.nodes:
        return None
    children = data.oracle.get("children", [])
    if len(children) != 1:
        return None
    child = children[0]
    child_name = child.get("name")
    child_path = child.get("resource_path")
    if not isinstance(child_name, str) or not isinstance(child_path, str):
        return None

    coordinator_root = _branch_root(trace, 0)
    spawn_node_index: int | None = None
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
            spawn_node_index = event.node_index
            break
    if spawn_node_index is None:
        return None
    if _has_pre_spawn_control_detour(
        trace,
        coordinator_root=coordinator_root,
        spawn_node_index=spawn_node_index,
        child_path=child_path,
    ):
        return None

    first_incoming = min(
        (
            node_index
            for node_index, _, _ in _incoming_messages(trace)
            if node_index > spawn_node_index
        ),
        default=None,
    )
    target_node_index = next(
        (
            index
            for index in range(spawn_node_index + 1, len(trace.nodes))
            if isinstance(trace.nodes[index].message, AssistantMessage)
            and trace.nodes[index].sampled
            and _branch_root(trace, index) == coordinator_root
        ),
        None,
    )
    if target_node_index is None or (
        first_incoming is not None and target_node_index >= first_incoming
    ):
        return None
    target = trace.nodes[target_node_index].message
    if not target.tool_calls:
        return None

    coordinator_turns = [
        index
        for index, node in enumerate(trace.nodes[: target_node_index + 1])
        if isinstance(node.message, AssistantMessage)
        and node.sampled
        and _branch_root(trace, index) == coordinator_root
    ]
    return NaturalYieldFailureDiagnostic(
        child_name=child_name,
        spawn_node_index=spawn_node_index,
        target_node_index=target_node_index,
        turn_index=len(coordinator_turns) - 1,
    )


def _has_pre_spawn_control_detour(
    trace: vf.Trace,
    *,
    coordinator_root: int,
    spawn_node_index: int,
    child_path: str,
) -> bool:
    """Reject harness detours before the first admitted delegation event."""

    aliases: dict[str, str] = {}
    spawn_calls = 0
    for event in _ipython_events(trace):
        if (
            event.node_index > spawn_node_index
            or _branch_root(trace, event.node_index) != coordinator_root
        ):
            continue
        try:
            tree = ast.parse(event.code)
        except SyntaxError:
            continue
        for statement in tree.body:
            source = ast.unparse(statement)
            _update_aliases(statement, aliases)
            if _delegated_path_used_outside_spawn(source, child_path):
                return True
            for call in (
                node for node in ast.walk(statement) if isinstance(node, ast.Call)
            ):
                raw_name = _dotted_name(call.func) or _call_name(call) or ""
                call_name = _resolve_alias(raw_name, aliases)
                if call_name == "rlm":
                    if event.node_index != spawn_node_index:
                        return True
                    spawn_calls += 1
                    continue
                if (
                    call_name.startswith(("rlm.", "agent_message.", "agent_observe."))
                    or call_name == "agent_message"
                    or call_name == "agent_observe"
                    or call_name == "sleep"
                    or call_name.endswith(".sleep")
                ):
                    return True
    return spawn_calls != 1


def _record_natural_yield_feedback(
    trace: vf.Trace,
    data: ProceduralHarnessMasterData,
) -> bool:
    diagnostic = _natural_yield_feedback_diagnostic(trace, data)
    if diagnostic is None:
        return False
    feedback = render_natural_yield_feedback(diagnostic)
    trace.info["feedback"] = feedback
    trace.info["feedback_contract"] = natural_yield_feedback_contract_payload(
        diagnostic
    )
    return True


def keep_natural_yield_feedback_response(trace: vf.Trace) -> list[list[bool]]:
    """Select only the sampled response named by trusted natural-yield feedback."""

    contract = trace.info.get("feedback_contract")
    target_index = (
        contract.get("target_node_index") if isinstance(contract, dict) else None
    )
    trusted = (
        isinstance(contract, dict)
        and contract.get("schema_version")
        == NATURAL_YIELD_FEEDBACK_SCHEMA_VERSION
        and contract.get("answer_free") is True
        and contract.get("retryable") is True
        and contract.get("code") == "tool_call_after_delegation"
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
        data = ProceduralHarnessMasterData.model_validate(data.model_dump())
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
            if (
                event.node_index not in coordinator_nodes
                or event.node_index >= child_request_index
            ):
                continue
            try:
                tree = ast.parse(event.code)
            except SyntaxError:
                continue
            rlm_calls = [
                call
                for call in ast.walk(tree)
                if isinstance(call, ast.Call)
                and (_dotted_name(call.func) or _call_name(call)) == "rlm"
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
                if any(
                    _assigned_state(statement, name, value)
                    for name, value in state.items()
                )
                or call in ast.walk(statement)
            }
            handle_names = {
                target.id
                for statement in tree.body
                if call in ast.walk(statement)
                and isinstance(statement, (ast.Assign, ast.AnnAssign))
                for target in (
                    statement.targets
                    if isinstance(statement, ast.Assign)
                    else [statement.target]
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
            if (
                end + 1 < len(node.token_ids)
                and node.token_ids[end + 1] == message_end_token_id
            ):
                selected[end + 1] = bool(node.mask[end + 1])
            cursor = end + 1
        return selected

    def visible_mask(node: Any) -> list[bool]:
        if isinstance(node.message, AssistantMessage) and node.message.tool_calls:
            return [False] * len(node.token_ids)
        try:
            start = (
                len(node.token_ids)
                - 1
                - node.token_ids[::-1].index(thinking_end_token_id)
            )
        except ValueError:
            start = -1
        return [
            bool(sampled and index > start) for index, sampled in enumerate(node.mask)
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
    leak_child_exact_action: bool = False


class ProceduralHarnessMasterTask(
    vf.Task[ProceduralHarnessMasterData, vf.State, ProceduralHarnessMasterTaskConfig]
):
    NEEDS_CONTAINER = True

    def __init__(
        self,
        data: ProceduralHarnessMasterData,
        config: ProceduralHarnessMasterTaskConfig | None = None,
    ) -> None:
        super().__init__(data, config)
        self._runtime_by_trace: dict[str, vf.Runtime] = {}

    @property
    def key(self) -> str:
        return self.data.episode_id

    @vf.intercept(priority=30)
    def arm_causal_persistence_lease(
        self, response: vf.Response, trace: vf.Trace
    ) -> None:
        if self.data.family != "natural_n1b":
            return
        existing = trace.info.get("persistence_lease")
        if isinstance(existing, dict) and (
            existing.get("pending") is True or existing.get("closed") is True
        ):
            return
        if not _response_proposes_delegation(response):
            return
        lease = self.data.oracle.get("persistence_lease", {})
        path = lease.get("path") if isinstance(lease, dict) else None
        if not isinstance(path, str):
            raise TypeError("natural N1b task lacks a persistence lease path")
        trace.info["persistence_lease"] = {
            "schema_version": "procedural-harness-master-v1/persistence-lease/v2",
            "pending": True,
            "closed": False,
            "path": path,
        }

    @vf.intercept(priority=20)
    async def close_causal_persistence_lease(
        self, request: vf.Request, trace: vf.Trace
    ) -> None:
        if self.data.family != "natural_n1b":
            return
        lease_record = trace.info.get("persistence_lease")
        if not isinstance(lease_record, dict) or lease_record.get("pending") is not True:
            return
        path = lease_record.get("path")
        if not isinstance(path, str):
            raise TypeError("natural N1b pending lease lacks a path")
        runtime = self._runtime_by_trace.get(trace.id)
        if runtime is None:
            raise RuntimeError("natural N1b persistence runtime is unavailable")
        result = await runtime.run(["rm", "-f", path], {})
        if result.exit_code != 0:
            raise RuntimeError(
                "natural N1b persistence lease closure failed: "
                f"{result.stderr[-500:]}"
            )
        trace.info["persistence_lease"] = {
            "schema_version": "procedural-harness-master-v1/persistence-lease/v2",
            "pending": False,
            "closed": True,
            "path": path,
        }

    @vf.intercept
    def inject_natural_private_evidence(self, request: vf.Request) -> vf.Request | None:
        """Attach hidden evidence only to the natural rung's child context."""
        private = self.data.oracle.get("private_resources", {})
        if not private:
            return None
        user_messages = [
            message for message in request.messages if isinstance(message, UserMessage)
        ]
        if any(
            content_text(message.content).strip() == self.data.prompt.strip()
            for message in user_messages
        ):
            return None
        if any(
            PRIVATE_EVIDENCE_HEADER in content_text(message.content)
            for message in user_messages
        ):
            return None
        target_index = next(
            (
                index
                for index in range(len(request.messages) - 1, -1, -1)
                if isinstance(request.messages[index], UserMessage)
                and isinstance(request.messages[index].content, str)
            ),
            None,
        )
        if target_index is None:
            return None

        ownership = self.data.oracle["resource_ownership"]
        sections = []
        for label, contents in private.items():
            sections.append(
                "\n".join(
                    (
                        f"Evidence label: {label}",
                        f"Required review: {ownership[label]['operation']}",
                        "Evidence contents:",
                        contents,
                    )
                )
            )
        child_action_scaffold = ""
        if self.config.leak_child_exact_action:
            children = self.data.oracle.get("children", [])
            if len(children) != 1 or "expected_result" not in children[0]:
                raise ValueError(
                    "exact child action scaffolding requires one child with an expected result"
                )
            expected = str(children[0]["expected_result"])
            child_action_scaffold = (
                f"\n\n{CHILD_ACTION_SCAFFOLD_HEADER}\n"
                "This scaffold is visible only while collecting child-training rollouts. "
                "In your first IPython call execute exactly:\n\n"
                "```python\n"
                f"await agent_message.send({expected!r}, receiver_role='parent')\n"
                "```\n\n"
                "After the call succeeds, stop; do not inspect the evidence label as a path, "
                "send a second message, or continue with prose."
            )
        target = request.messages[target_index]
        assert isinstance(target, UserMessage) and isinstance(target.content, str)
        replacement = target.model_copy(
            update={
                "content": (
                    f"{target.content.rstrip()}\n\n{PRIVATE_EVIDENCE_HEADER}\n"
                    "This evidence is available in this child context only; the coordinator "
                    "does not have a workspace copy. The evidence label is an identifier, "
                    "not a runtime file path; process the inline contents below directly.\n"
                    + "\n\n".join(sections)
                    + child_action_scaffold
                )
            }
        )
        messages = list(request.messages)
        messages[target_index] = replacement
        return request.model_copy(update={"messages": messages})

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        if self.data.family == "natural_n1b":
            self._runtime_by_trace[trace.id] = runtime
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
            if self.data.family in {"followup", "atomic_followup", "natural_n2"}
            else "single"
            if self.data.family
            in {"natural_n1", "natural_n1a", "natural_n1a_local", "natural_n1b"}
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

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        self._runtime_by_trace.pop(trace.id, None)

    @vf.reward(weight=1.0)
    async def harness_score(self, trace: vf.Trace) -> float:
        behavior = _contract_behavior(trace, self.data)
        if self.config.reward_mode == "bootstrap":
            return behavior["harness_score"] + 0.1 * behavior["bootstrap_progress"]
        if self.config.reward_mode == "event_control":
            return (
                behavior["harness_score"]
                + behavior["event_control_progress"]
                + behavior["child_action_bridge"]
            )
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
    private_payload_mode: Literal["raw_resource", "finding_card"] = "raw_resource"
    record_causal_feedback: bool = False
    privileged_hint_path: str | None = None
    privileged_bootstrap_path: str | None = None


def _load_privileged_hints(path: str | None) -> dict[str, str]:
    if path is None:
        return {}
    hint_path = Path(path)
    payload = json.loads(hint_path.read_text())
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "qwen35-2b-spade-rung0-hints/v1"
        or payload.get("status") != "complete"
    ):
        raise ValueError(f"invalid privileged hint artifact: {hint_path}")
    hints = payload.get("hints")
    if not isinstance(hints, dict) or not hints:
        raise ValueError(f"privileged hint artifact has no hints: {hint_path}")
    validated = {}
    for episode_id, hint in hints.items():
        if not isinstance(episode_id, str) or not episode_id:
            raise TypeError("privileged hint episode ids must be non-empty strings")
        if not isinstance(hint, str) or not hint.strip():
            raise TypeError(f"privileged hint for {episode_id!r} must be non-empty text")
        validated[episode_id] = hint.strip()
    return validated


def _load_privileged_bootstrap(path: str | None) -> dict[str, str]:
    if path is None:
        return {}
    bootstrap_path = Path(path)
    payload = json.loads(bootstrap_path.read_text())
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version")
        != "qwen35-2b-environment-bootstrap-context/v1"
        or payload.get("status") != "complete"
        or payload.get("split") != "train_gen"
    ):
        raise ValueError(f"invalid privileged bootstrap artifact: {bootstrap_path}")
    contexts = payload.get("contexts")
    if not isinstance(contexts, dict) or not contexts:
        raise ValueError(
            f"privileged bootstrap artifact has no contexts: {bootstrap_path}"
        )
    validated = {}
    for episode_id, context in contexts.items():
        if not isinstance(episode_id, str) or not episode_id:
            raise TypeError("privileged bootstrap episode ids must be non-empty strings")
        if not isinstance(context, str) or not context.strip():
            raise TypeError(
                f"privileged bootstrap context for {episode_id!r} must be non-empty text"
            )
        validated[episode_id] = context.strip()
    return validated


class ProceduralHarnessMasterTaskset(
    vf.Taskset[ProceduralHarnessMasterTask, ProceduralHarnessMasterConfig]
):
    def load(self) -> list[ProceduralHarnessMasterTask]:
        if self.config.curriculum_rung == "natural_n1b":
            from procedural_harness_master_v1.causal_context_boundary import (
                install_causal_context_boundary,
            )

            install_causal_context_boundary()
        generator = _generator()
        if (
            self.config.privileged_hint_path is not None
            and self.config.privileged_bootstrap_path is not None
        ):
            raise ValueError(
                "privileged_hint_path and privileged_bootstrap_path are mutually exclusive"
            )
        if (
            self.config.privileged_bootstrap_path is not None
            and self.config.split != "train_gen"
        ):
            raise ValueError("privileged bootstrap context is restricted to train_gen")
        privileged_hints = _load_privileged_hints(self.config.privileged_hint_path)
        privileged_bootstrap = _load_privileged_bootstrap(
            self.config.privileged_bootstrap_path
        )
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
                    self.config.private_payload_mode,
                )
            index += 1
            family = row["metadata"]["episode_family"]
            if self.config.families is not None and family not in self.config.families:
                continue
            public = row["public"]
            prompt = public["user_prompt"]
            if self.config.privileged_hint_path is not None:
                episode_id = row["episode_id"]
                if episode_id not in privileged_hints:
                    raise ValueError(
                        f"privileged hint artifact lacks selected task {episode_id}"
                    )
                prompt = (
                    f"{prompt.rstrip()}\n\n{PRIVILEGED_HINT_HEADER}\n"
                    f"{privileged_hints[episode_id]}"
                )
            if self.config.privileged_bootstrap_path is not None:
                episode_id = row["episode_id"]
                if episode_id not in privileged_bootstrap:
                    raise ValueError(
                        "privileged bootstrap artifact lacks selected task "
                        f"{episode_id}"
                    )
                prompt = (
                    f"{prompt.rstrip()}\n\n{PRIVILEGED_BOOTSTRAP_HEADER}\n"
                    f"{privileged_bootstrap[episode_id]}"
                )
            tasks.append(
                ProceduralHarnessMasterTask(
                    ProceduralHarnessMasterData(
                        idx=len(tasks),
                        name=row["episode_id"],
                        prompt=prompt,
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
                    _record_natural_yield_feedback(
                        interaction.trace, task.data
                    ) or _record_followup_feedback(interaction.trace, task.data)
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

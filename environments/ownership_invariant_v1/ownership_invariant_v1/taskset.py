"""Executable paired ownership decisions for native-sibling SDPO."""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import random
from dataclasses import dataclass
from typing import Any, Literal, Self

import verifiers.v1 as vf
from pydantic import Field, model_validator
from subagent_communication_v1.taskset import (
    SYSTEM_PROMPT,
    _assigned_call_names,
    _branch_root,
    _call_name,
    _delegated_path_used_outside_spawn,
    _failed,
    _ipython_events,
    _model_tool_events,
    _spawn_name,
    _spawn_prompt,
)

Ownership = Literal["child", "coordinator"]
Split = Literal["admission", "heldout_phrasing", "heldout_resource"]
YieldPolicy = Literal["literal", "semantic"]

TRAIN_RESOURCE_FAMILIES = (
    "json_sum",
    "csv_amount_total",
    "text_keyword_count",
    "markdown_heading_count",
    "log_error_count",
    "python_def_count",
    "json_max_value",
    "sha256_prefix",
)
HELDOUT_RESOURCE_FAMILIES = ("tsv_score_total", "xml_item_count")

CHILD_NAMES = {
    "json_sum": "ledger-worker",
    "csv_amount_total": "table-worker",
    "text_keyword_count": "text-worker",
    "markdown_heading_count": "outline-worker",
    "log_error_count": "log-worker",
    "python_def_count": "code-worker",
    "json_max_value": "metrics-worker",
    "sha256_prefix": "hash-worker",
    "tsv_score_total": "score-worker",
    "xml_item_count": "xml-worker",
}

CHILD_OWNED_PHRASINGS = (
    (
        "Keep {state_name}={state_value!r} in coordinator state. Delegate {operation} "
        "for {path} to one child named {child}. The child owns that resource; retain the "
        "admission handle and yield after spawning it."
    ),
    (
        "Preserve {state_name}={state_value!r} locally for later. Ask {child} to {operation} "
        "using {path}. Keep the returned child handle and end this first decision once the "
        "child is admitted."
    ),
    (
        "Divide responsibilities: {state_name}={state_value!r} stays with the coordinator, "
        "while {child} takes responsibility for {operation} at {path}. Admit that child, "
        "retain its handle, and then yield."
    ),
    (
        "Retain {state_name}={state_value!r}. For the resource portion, have {child} handle "
        "{path} and {operation}. Preserve the admission handle and stop after the first "
        "delegation decision."
    ),
)

COORDINATOR_OWNED_PHRASINGS = (
    (
        "Keep {state_name}={state_value!r} in coordinator state. This resource stays local: "
        "use {path} yourself to {operation}. Do not delegate. Return {answer_schema}."
    ),
    (
        "Preserve {state_name}={state_value!r} locally and directly {operation} using {path}. "
        "No child is assigned. Return {answer_schema}."
    ),
    (
        "The coordinator owns both {state_name}={state_value!r} and the work on {path}. "
        "Process it locally to {operation}; do not spawn a child. Return {answer_schema}."
    ),
    (
        "Handle this without delegation. Retain {state_name}={state_value!r}, access {path} "
        "in the coordinator, and {operation}. Return {answer_schema}."
    ),
)


class OwnershipInvariantData(vf.TaskData):
    ownership: Ownership
    split: Split
    resource_family: str
    phrasing_variant: int
    state_name: str
    state_value: str
    expected_child: str
    resource_path: str
    resource_content: str
    expected_result: int | str
    operation: str
    files: dict[str, str] = Field(default_factory=dict)


@dataclass(frozen=True)
class ResourceSpec:
    family: str
    path: str
    content: str
    result: int | str
    operation: str
    child_prompt: str


def _resource_spec(family: str, instance: int, seed: int) -> ResourceSpec:
    rng = random.Random(seed * 1_000_003 + instance * 10_007 + sum(map(ord, family)))
    root = f"/workspace/ownership-invariant/{family}/i{instance}"

    if family == "json_sum":
        path = f"{root}.json"
        values = [rng.randint(-9, 30) for _ in range(7)]
        content = json.dumps(values)
        result: int | str = sum(values)
        operation = "load the JSON integer list and return its sum"
    elif family == "csv_amount_total":
        path = f"{root}.csv"
        rows = [{"id": f"r{i}", "amount": str(rng.randint(2, 90))} for i in range(5)]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=["id", "amount"])
        writer.writeheader()
        writer.writerows(rows)
        content = buffer.getvalue()
        result = sum(int(row["amount"]) for row in rows)
        operation = "sum the integer values in the CSV amount column"
    elif family == "text_keyword_count":
        path = f"{root}.txt"
        keyword = "retry"
        content = "ready retry stable retry done retry"
        result = content.split().count(keyword)
        operation = f"count exact whitespace-delimited occurrences of {keyword!r}"
    elif family == "markdown_heading_count":
        path = f"{root}.md"
        content = "# Report\n\n## Alpha\ntext\n## Beta\ntext\n## Gamma\n"
        result = sum(line.startswith("## ") for line in content.splitlines())
        operation = "count level-2 Markdown headings"
    elif family == "log_error_count":
        path = f"{root}.log"
        content = "INFO boot\nERROR failed\nWARN slow\nERROR retry\nINFO done\n"
        result = sum(line.startswith("ERROR ") for line in content.splitlines())
        operation = "count log lines whose level is ERROR"
    elif family == "python_def_count":
        path = f"{root}.py"
        content = "def alpha():\n    pass\n\nasync def beta():\n    pass\n\ndef gamma():\n    pass\n"
        result = 3
        operation = "count top-level synchronous and asynchronous function definitions"
    elif family == "json_max_value":
        path = f"{root}.json"
        values = {f"m{i}": rng.randint(-20, 80) for i in range(6)}
        content = json.dumps(values)
        result = max(values.values())
        operation = "return the largest integer value in the JSON object"
    elif family == "sha256_prefix":
        path = f"{root}.bin"
        content = f"ownership payload {instance} xxxxxxxxx"
        result = hashlib.sha256(content.encode()).hexdigest()[:8]
        operation = "compute the first eight hexadecimal characters of its SHA-256 digest"
    elif family == "tsv_score_total":
        path = f"{root}.tsv"
        scores = [rng.randint(4, 70) for _ in range(5)]
        content = "name\tscore\n" + "\n".join(f"n{i}\t{score}" for i, score in enumerate(scores)) + "\n"
        result = sum(scores)
        operation = "sum the integer score column in the tab-separated file"
    elif family == "xml_item_count":
        path = f"{root}.xml"
        count = 4 + instance % 3
        content = "<root>" + "".join(f"<item id='{i}'/>" for i in range(count)) + "</root>"
        result = count
        operation = "count the XML item elements"
    else:
        raise ValueError(f"unknown resource family: {family}")

    child_prompt = (
        f"Read {path} and {operation}. Send only the result to your parent with "
        "await agent_message.send(str(result), receiver_role='parent'). After the "
        "delivery receipt succeeds, stop."
    )
    return ResourceSpec(family, path, content, result, operation, child_prompt)


def _statement_index(tree: ast.Module, target: ast.AST) -> int | None:
    for index, statement in enumerate(tree.body):
        if any(node is target for node in ast.walk(statement)):
            return index
    return None


def _retains_state(statement: ast.stmt, name: str, value: str) -> bool:
    for node in ast.walk(statement):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if (
            any(isinstance(target, ast.Name) and target.id == name for target in targets)
            and isinstance(node.value, ast.Constant)
            and node.value.value == value
        ):
            return True
        if isinstance(node.value, ast.Dict):
            for key, item in zip(node.value.keys, node.value.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == name
                    and isinstance(item, ast.Constant)
                    and item.value == value
                ):
                    return True
        for target in targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == name
                and isinstance(node.value, ast.Constant)
                and node.value.value == value
            ):
                return True
    return False


def _assigned_names_for_call(tree: ast.Module, target: ast.Call) -> set[str]:
    names: set[str] = set()
    for statement in tree.body:
        if not any(node is target for node in ast.walk(statement)):
            continue
        if isinstance(statement, ast.Assign):
            targets = statement.targets
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        else:
            continue
        names.update(item.id for item in targets if isinstance(item, ast.Name))
    return names


def _handle_only_value(node: ast.AST, handle_names: set[str], *, require_handle: bool) -> bool:
    names = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
    calls = [item for item in ast.walk(node) if isinstance(item, ast.Call)]
    return (not require_handle or bool(names & handle_names)) and names <= handle_names and not calls


def _passive_handle_statement(statement: ast.stmt, handle_names: set[str]) -> bool:
    if not handle_names:
        return False
    if isinstance(statement, ast.Expr):
        value = statement.value
        if isinstance(value, ast.Call) and _call_name(value) == "print":
            return all(
                _handle_only_value(argument, handle_names, require_handle=False)
                for argument in [*value.args, *(keyword.value for keyword in value.keywords)]
            ) and any(
                any(isinstance(item, ast.Name) and item.id in handle_names for item in ast.walk(argument))
                for argument in value.args
            )
        return _handle_only_value(value, handle_names, require_handle=True)
    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
        return _handle_only_value(statement.value, handle_names, require_handle=True)
    return False


def _passive_handle_statements(statements: list[ast.stmt], handle_names: set[str]) -> bool:
    passive_names = set(handle_names)
    for statement in statements:
        if not _passive_handle_statement(statement, passive_names):
            return False
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        for target in targets:
            if isinstance(target, ast.Name):
                passive_names.add(target.id)
            elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                passive_names.add(target.value.id)
    return True


def _first_decision_behavior(
    trace: vf.Trace,
    data: OwnershipInvariantData,
    yield_policy: YieldPolicy = "literal",
) -> dict[str, float]:
    keys = (
        "strict_success",
        "first_decision_only",
        "state_retained",
        "state_precedes_spawn",
        "one_spawn",
        "retained_handle",
        "expected_child",
        "delegated_path",
        "parent_path_access",
        "local_state_leaked",
        "prohibited_control",
        "post_spawn_statement",
        "passive_handle_tail",
        "post_spawn_action",
        "direct_answer_accuracy",
    )
    empty = {key: 0.0 for key in keys}
    events = _ipython_events(trace)
    if not events:
        return empty

    coordinator_root = _branch_root(trace, 0)
    branch_aware = any(_branch_root(trace, event.node_index) != coordinator_root for event in events)
    coordinator_events = [
        event for event in events if not branch_aware or _branch_root(trace, event.node_index) == coordinator_root
    ]
    if not coordinator_events:
        return empty

    try:
        trees = [ast.parse(event.code) for event in coordinator_events]
    except SyntaxError:
        return empty

    model_tools = [
        event
        for event in _model_tool_events(trace)
        if not branch_aware or _branch_root(trace, event.node_index) == coordinator_root
    ]
    first_decision_only = len(coordinator_events) == 1 and all(event.name == "ipython" for event in model_tools)
    all_calls = [node for tree in trees for node in ast.walk(tree) if isinstance(node, ast.Call)]
    spawns = [call for call in all_calls if _call_name(call) == "rlm"]
    prohibited = any(
        (_call_name(call) or "").startswith("agent_observe.")
        or _call_name(call)
        in {
            "rlm.list_subagents",
            "agent_message.list_agents",
            "agent_message.recv",
            "agent_message.send",
        }
        for call in all_calls
    ) or any(event.name != "ipython" for event in model_tools)
    parent_path_access = any(
        _delegated_path_used_outside_spawn(event.code, data.resource_path) for event in coordinator_events
    )

    first_tree = trees[0]
    state_indices = [
        index
        for index, statement in enumerate(first_tree.body)
        if _retains_state(statement, data.state_name, data.state_value)
    ]
    state_retained = bool(state_indices)
    one_spawn = len(spawns) == 1 and not _failed(coordinator_events[0].output)
    spawn = spawns[0] if one_spawn else None
    spawn_index = _statement_index(first_tree, spawn) if spawn is not None else None
    state_precedes_spawn = bool(state_indices and spawn_index is not None and min(state_indices) < spawn_index)
    retained_handle = bool(spawn is not None and id(spawn) in _assigned_call_names(first_tree))
    prompt = (_spawn_prompt(spawn) or "") if spawn is not None else ""
    expected_child = bool(spawn is not None and _spawn_name(spawn, coordinator_events[0].output) == data.expected_child)
    delegated_path = bool(prompt and data.resource_path in prompt)
    local_state_leaked = bool(prompt and (data.state_name in prompt or data.state_value in prompt))
    trailing_statements = first_tree.body[spawn_index + 1 :] if spawn_index is not None else []
    post_spawn_statement = bool(trailing_statements or len(coordinator_events) > 1)
    handle_names = _assigned_names_for_call(first_tree, spawn) if spawn is not None else set()
    passive_handle_tail = bool(
        trailing_statements
        and len(coordinator_events) == 1
        and _passive_handle_statements(trailing_statements, handle_names)
    )
    substantive_post_spawn_action = bool(
        len(coordinator_events) > 1 or (trailing_statements and not passive_handle_tail)
    )
    post_spawn_action = post_spawn_statement if yield_policy == "literal" else substantive_post_spawn_action

    if data.ownership == "child":
        components = (
            first_decision_only,
            state_retained,
            state_precedes_spawn,
            one_spawn,
            retained_handle,
            expected_child,
            delegated_path,
            not parent_path_access,
            not local_state_leaked,
            not prohibited,
            not post_spawn_action,
        )
        answer_accuracy = 0.0
    else:
        try:
            answer = json.loads(trace.last_reply.strip())
        except (AttributeError, json.JSONDecodeError):
            answer = None
        answer_accuracy = float(answer == {"local_state": data.state_value, "result": data.expected_result})
        components = (
            first_decision_only,
            state_retained,
            not spawns,
            parent_path_access,
            not prohibited,
            answer_accuracy == 1.0,
        )

    return {
        "strict_success": float(all(components)),
        "first_decision_only": float(first_decision_only),
        "state_retained": float(state_retained),
        "state_precedes_spawn": float(state_precedes_spawn),
        "one_spawn": float(one_spawn),
        "retained_handle": float(retained_handle),
        "expected_child": float(expected_child),
        "delegated_path": float(delegated_path),
        "parent_path_access": float(parent_path_access),
        "local_state_leaked": float(local_state_leaked),
        "prohibited_control": float(prohibited),
        "post_spawn_statement": float(post_spawn_statement),
        "passive_handle_tail": float(passive_handle_tail),
        "post_spawn_action": float(post_spawn_action),
        "direct_answer_accuracy": answer_accuracy,
    }


class OwnershipInvariantTaskConfig(vf.TaskConfig):
    yield_policy: YieldPolicy = "literal"


class OwnershipInvariantTask(vf.Task[OwnershipInvariantData, vf.State, OwnershipInvariantTaskConfig]):
    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        directories = sorted({path.rsplit("/", 1)[0] for path in self.data.files})
        created = await runtime.run(["mkdir", "-p", *directories], {})
        if created.exit_code != 0:
            raise RuntimeError(f"ownership resource setup failed: {created.stderr[-500:]}")
        for path, contents in self.data.files.items():
            await runtime.write(path, contents.encode())

    @vf.reward(weight=1.0)
    async def ownership_invariant_reward(self, trace: vf.Trace) -> float:
        return _first_decision_behavior(trace, self.data, self.config.yield_policy)["strict_success"]

    @vf.metric
    async def ownership_behavior(self, trace: vf.Trace) -> dict[str, float]:
        return _first_decision_behavior(trace, self.data, self.config.yield_policy)


class OwnershipInvariantConfig(vf.TasksetConfig):
    task: OwnershipInvariantTaskConfig = OwnershipInvariantTaskConfig()
    split: Split = "admission"
    ownership: Ownership = "child"
    yield_policy: YieldPolicy = "literal"
    instances_per_family: int = Field(1, ge=1)
    instance_offset: int = Field(0, ge=0)
    seed: int = 20260812

    @model_validator(mode="after")
    def propagate_yield_policy(self) -> Self:
        # Served tasks are rebuilt from task data plus this task subtree, so
        # mirror the taskset-level curriculum knob across that wire boundary.
        self.task = self.task.model_copy(update={"yield_policy": self.yield_policy})
        return self


def _split_entries(split: Split) -> list[tuple[str, int]]:
    if split == "admission":
        return [(family, index % 2) for index, family in enumerate(TRAIN_RESOURCE_FAMILIES)]
    if split == "heldout_phrasing":
        return [(family, 2 + index % 2) for index, family in enumerate(TRAIN_RESOURCE_FAMILIES)]
    return [(family, phrasing) for family in HELDOUT_RESOURCE_FAMILIES for phrasing in range(4)]


class OwnershipInvariantTaskset(vf.Taskset[OwnershipInvariantTask, OwnershipInvariantConfig]):
    def load(self) -> list[OwnershipInvariantTask]:
        tasks: list[OwnershipInvariantTask] = []
        for instance in range(
            self.config.instance_offset,
            self.config.instance_offset + self.config.instances_per_family,
        ):
            for family, phrasing in _split_entries(self.config.split):
                spec = _resource_spec(family, instance, self.config.seed)
                state_name = ("request_tag", "batch_marker", "trace_key", "route_token")[phrasing]
                state_value = f"coord-{family}-{instance}-{phrasing}"
                child = CHILD_NAMES[family]
                answer_schema = '{"local_state": value, "result": value}'
                prompt_values: dict[str, Any] = {
                    "state_name": state_name,
                    "state_value": state_value,
                    "operation": spec.operation,
                    "path": spec.path,
                    "child": child,
                    "answer_schema": answer_schema,
                }
                templates = CHILD_OWNED_PHRASINGS if self.config.ownership == "child" else COORDINATOR_OWNED_PHRASINGS
                prompt = templates[phrasing].format(**prompt_values)
                tasks.append(
                    OwnershipInvariantTask(
                        OwnershipInvariantData(
                            idx=len(tasks),
                            name=(f"{self.config.ownership}-{self.config.split}-{family}-p{phrasing}-i{instance}"),
                            prompt=prompt,
                            system_prompt=SYSTEM_PROMPT,
                            ownership=self.config.ownership,
                            split=self.config.split,
                            resource_family=family,
                            phrasing_variant=phrasing,
                            state_name=state_name,
                            state_value=state_value,
                            expected_child=child,
                            resource_path=spec.path,
                            resource_content=spec.content,
                            expected_result=spec.result,
                            operation=spec.operation,
                            files={spec.path: spec.content},
                        ),
                        self.config.task.model_copy(update={"yield_policy": self.config.yield_policy}),
                    )
                )
        return tasks

"""Pure execution-evidence scoring for one atomic source-worker call."""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from typing import Literal

Family = Literal["specialist_source_ast", "specialist_source_config"]


@dataclass(frozen=True)
class CellEvidence:
    code: str
    output: str


@dataclass(frozen=True)
class FirstCallScore:
    score: float
    exception_free_first_call: float
    correct_file_api: float
    exact_oracle_value: float
    atomic_compact_parent_send: float
    retries: int
    duplicate_cells: int
    extra_sends: int

    def metrics(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def _call_name(call: ast.Call) -> str | None:
    parts: list[str] = []
    value: ast.expr = call.func
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
        return ".".join(reversed(parts))
    return None


def _keyword_string(call: ast.Call, name: str) -> str | None:
    value = next((item.value for item in call.keywords if item.arg == name), None)
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def _failed(output: str) -> bool:
    markers = ("Traceback", "Error:", "SyntaxError", "Exception:")
    return any(marker in output for marker in markers)


def _delivered(output: str) -> bool:
    markers = ("agentmsg_", "Agent message sent", "Agent message queued")
    return not _failed(output) and any(marker in output for marker in markers)


def _parsed(code: str) -> ast.AST | None:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _calls(tree: ast.AST | None) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)] if tree else []


def _string_literals(tree: ast.AST | None) -> set[str]:
    if tree is None:
        return set()
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _bound_iterable_paths(tree: ast.AST, variable: str) -> set[str]:
    paths: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            if node.target.id == variable:
                paths.update(_string_literals(node.iter))
                if isinstance(node.iter, ast.Name):
                    paths.update(_assigned_paths(tree, node.iter.id))
        elif isinstance(node, ast.comprehension) and isinstance(
            node.target, ast.Name
        ):
            if node.target.id == variable:
                paths.update(_string_literals(node.iter))
                if isinstance(node.iter, ast.Name):
                    paths.update(_assigned_paths(tree, node.iter.id))
    return paths


def _assigned_paths(tree: ast.AST, variable: str) -> set[str]:
    paths: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == variable for target in targets):
            paths.update(_string_literals(node.value))
    return paths


def _read_path_coverage(tree: ast.AST | None) -> set[str]:
    """Return path literals that actually participate in a read/open call."""

    if tree is None:
        return set()
    covered: set[str] = set()
    for call in _calls(tree):
        name = _call_name(call)
        path_expression: ast.AST | None = None
        if isinstance(call.func, ast.Attribute) and call.func.attr in {
            "read_text",
            "read",
        }:
            path_expression = call.func.value
        elif name in {"open", "Path.open"} and call.args:
            path_expression = call.args[0]
        if path_expression is None:
            continue
        covered.update(_string_literals(path_expression))
        for value in ast.walk(path_expression):
            if isinstance(value, ast.Name):
                covered.update(_bound_iterable_paths(tree, value.id))
                covered.update(_assigned_paths(tree, value.id))
    return covered


def _has_split_key_value(call: ast.Call) -> bool:
    if _call_name(call) not in {"split", "str.split"} and not (
        isinstance(call.func, ast.Attribute) and call.func.attr == "split"
    ):
        return False
    if not call.args:
        return False
    separator = call.args[0]
    if not isinstance(separator, ast.Constant) or separator.value != "=":
        return False
    positional_once = (
        len(call.args) >= 2
        and isinstance(call.args[1], ast.Constant)
        and call.args[1].value == 1
    )
    keyword_once = any(
        item.arg == "maxsplit"
        and isinstance(item.value, ast.Constant)
        and item.value.value == 1
        for item in call.keywords
    )
    return positional_once or keyword_once


def _correct_api(family: Family, required_paths: tuple[str, ...], tree: ast.AST | None) -> bool:
    calls = _calls(tree)
    names = {_call_name(call) for call in calls}
    all_paths_read = set(required_paths).issubset(_read_path_coverage(tree))
    if family == "specialist_source_ast":
        return all_paths_read and "ast.parse" in names
    if family == "specialist_source_config":
        return (
            all_paths_read
            and ("tomllib.loads" in names or "tomllib.load" in names)
            and any(_has_split_key_value(call) for call in calls)
        )
    raise ValueError(f"unsupported source-worker family: {family}")


def _parent_sends(tree: ast.AST | None) -> list[ast.Call]:
    return [
        call
        for call in _calls(tree)
        if _call_name(call) == "agent_message.send"
        and _keyword_string(call, "receiver_role") == "parent"
    ]


def score_first_call(
    *,
    family: Family,
    required_paths: tuple[str, ...],
    expected_value: int,
    cells: tuple[CellEvidence, ...],
    delivered_bodies: tuple[str, ...],
) -> FirstCallScore:
    """Score only observed execution and delivered-message evidence.

    Static syntax is used to identify the APIs and path literals, while success
    credit additionally requires an exception-free tool result. Oracle and
    transport credit come from the message observed by the parent, not a token
    pattern in the candidate's code.
    """

    first = cells[0] if cells else None
    first_tree = _parsed(first.code) if first else None
    all_trees = tuple(_parsed(cell.code) for cell in cells)
    all_sends = [call for tree in all_trees for call in _parent_sends(tree)]
    first_sends = _parent_sends(first_tree)

    exception_free = bool(first and first_tree is not None and not _failed(first.output))
    correct_api = exception_free and _correct_api(family, required_paths, first_tree)
    compact = json.dumps({"value": expected_value}, separators=(",", ":"))
    # A later exact report earns bounded recovery credit, but never atomic credit.
    # The retry penalty ensures it cannot outrank an exception-free, correct-API
    # first call that has not yet delivered the answer.
    exact = len(delivered_bodies) == 1 and delivered_bodies[0] == compact
    atomic = (
        correct_api
        and exact
        and len(first_sends) == 1
        and len(all_sends) == 1
        and first is not None
        and _delivered(first.output)
    )

    retries = max(0, len(cells) - 1)
    normalized = [cell.code.strip() for cell in cells]
    duplicate_cells = len(normalized) - len(set(normalized))
    extra_sends = max(0, len(all_sends) - 1)
    penalty = min(
        0.79,
        0.31 * retries + 0.02 * duplicate_cells + 0.03 * extra_sends,
    )
    score = (
        0.10 * exception_free
        + 0.20 * correct_api
        + 0.30 * exact
        + 0.40 * atomic
        - penalty
    )
    return FirstCallScore(
        score=max(-0.79, min(1.0, score)),
        exception_free_first_call=float(exception_free),
        correct_file_api=float(correct_api),
        exact_oracle_value=float(exact),
        atomic_compact_parent_send=float(atomic),
        retries=retries,
        duplicate_cells=duplicate_cells,
        extra_sends=extra_sends,
    )

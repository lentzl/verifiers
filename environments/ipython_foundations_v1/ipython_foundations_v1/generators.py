"""Deterministic streams for persistent IPython fundamentals."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Literal

Family = Literal["assignment", "state", "recovery"]
FAMILIES: tuple[Family, ...] = ("assignment", "state", "recovery")
TRAIN_VARIANTS = range(4)
EVAL_VARIANTS = range(4, 6)


@dataclass(frozen=True)
class GeneratedRound:
    instruction: str
    explicit_operation: str
    answer: object
    files: dict[str, str]
    remove_after: tuple[str, ...] = ()


@dataclass(frozen=True)
class GeneratedStream:
    state_variable: str
    rounds: tuple[GeneratedRound, ...]


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _assignment_stream(rng: random.Random, variant: int) -> GeneratedStream:
    rounds = []
    for round_idx in range(3):
        values = [rng.randint(-20, 40) for _ in range(8)]
        path = "/workspace/inbox/values.json"
        rounds.append(
            GeneratedRound(
                instruction=(
                    "Load the JSON array into the persistent notebook variable `values`. "
                    "Use a later IPython call to compute its position-weighted checksum "
                    "sum((index + 1) * value), then return that integer as JSON."
                ),
                explicit_operation=(
                    "First execute only `import json; from pathlib import Path; "
                    "values = json.loads(Path("
                    + repr(path)
                    + ").read_text())`; an empty result is expected because assignment "
                    "is silent. In the next IPython call, evaluate `sum((i + 1) * value "
                    "for i, value in enumerate(values))`. Do not repeat the assignment cell."
                ),
                answer=sum((index + 1) * value for index, value in enumerate(values)),
                files={path: _json(values)},
            )
        )
    return GeneratedStream(state_variable="values", rounds=tuple(rounds))


def _state_stream(rng: random.Random, variant: int) -> GeneratedStream:
    labels = [f"group-{variant}-{index}" for index in range(4)]
    records = [
        {"group": labels[rng.randrange(len(labels))], "amount": rng.randint(1, 20)}
        for _ in range(24)
    ]
    path = "/workspace/inbox/records.json"
    totals = {
        label: sum(row["amount"] for row in records if row["group"] == label)
        for label in labels
    }
    rounds = (
        GeneratedRound(
            instruction=(
                "Load the records into the persistent notebook variable `records` and "
                "return the number of records. Retain the variable because the source "
                "file will be removed before later requests."
            ),
            explicit_operation=(
                "Assign the parsed JSON to `records`, then display `len(records)`. "
                "Do not discard or recreate `records` after answering."
            ),
            answer=len(records),
            files={path: _json(records)},
            remove_after=(path,),
        ),
        GeneratedRound(
            instruction=(
                "The source file is no longer available. Reuse `records` from the "
                "persistent IPython state and return the JSON object mapping every group "
                "to its summed amount."
            ),
            explicit_operation=(
                "Do not read or reconstruct the removed file. Compute the totals directly "
                "from the existing `records` variable and display the result."
            ),
            answer=totals,
            files={},
        ),
        GeneratedRound(
            instruction=(
                "Still using the retained `records`, return the sorted JSON list of group "
                "names tied for the largest summed amount."
            ),
            explicit_operation=(
                "Reuse `records` again. Derive grouped totals, find their maximum, and "
                "display the sorted names whose total equals it."
            ),
            answer=sorted(
                label
                for label, total in totals.items()
                if total == max(totals.values())
            ),
            files={},
        ),
    )
    return GeneratedStream(state_variable="records", rounds=rounds)


def _recovery_stream(rng: random.Random, variant: int) -> GeneratedStream:
    field_names = ("amount", "units", "score", "weight", "points", "count")
    rounds = []
    for round_idx in range(3):
        field = field_names[(variant + round_idx) % len(field_names)]
        rows = [
            {"label": f"item-{index}", field: rng.randint(2, 30)} for index in range(7)
        ]
        path = "/workspace/inbox/rows.json"
        rounds.append(
            GeneratedRound(
                instruction=(
                    "Load the JSON rows into persistent variable `rows`. A stale notebook "
                    "procedure assumes the numeric field is named `value`; run it, use the "
                    "resulting feedback to inspect the real schema, then correct the "
                    "calculation and return the numeric-field total as JSON."
                ),
                explicit_operation=(
                    "First run `import json; from pathlib import Path; rows = "
                    "json.loads(Path("
                    + repr(path)
                    + ").read_text()); sum(row['value'] for row in rows)` to obtain "
                    "executable feedback. After the KeyError, `rows` still exists: inspect "
                    "`rows[0]` in the next call, then reuse `rows` with the observed numeric key."
                ),
                answer=sum(row[field] for row in rows),
                files={path: _json(rows)},
            )
        )
    return GeneratedStream(state_variable="rows", rounds=tuple(rounds))


def generate(family: Family, variant: int, instance: int, seed: int) -> GeneratedStream:
    rng = random.Random((seed * 1_000_003) + (variant * 10_007) + instance)
    if family == "assignment":
        return _assignment_stream(rng, variant)
    if family == "state":
        return _state_stream(rng, variant)
    return _recovery_stream(rng, variant)

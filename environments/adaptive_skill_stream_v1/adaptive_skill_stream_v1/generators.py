"""Deterministic linked task streams for adaptation and skill lifecycle decisions."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Literal

Family = Literal["installed", "stable", "ephemeral"]
FAMILIES: tuple[Family, ...] = ("installed", "stable", "ephemeral")
TRAIN_VARIANTS = range(4)
EVAL_VARIANTS = range(4, 6)


@dataclass(frozen=True)
class GeneratedRound:
    instruction: str
    answer: object
    files: dict[str, str]


@dataclass(frozen=True)
class GeneratedStream:
    project_context: str
    rounds: tuple[GeneratedRound, ...]


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _canonical_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def _installed_stream(rng: random.Random, variant: int) -> GeneratedStream:
    bases = ["North Red", "South Blue", "West Gold", "East Green"]
    styles = [
        lambda value: f" {value.upper()} ",
        lambda value: value.replace(" ", " / "),
        lambda value: value.replace(" ", "___").lower(),
        lambda value: value.replace(" ", "-").swapcase(),
    ]
    rounds = []
    for round_idx in range(4):
        records = []
        expected: dict[str, int] = {}
        for item_idx in range(6):
            base = bases[(variant + round_idx + item_idx) % len(bases)]
            label = styles[(variant + round_idx + item_idx) % len(styles)](base)
            amount = rng.randint(1, 9)
            records.append({"label": label, "amount": amount})
            key = _canonical_label(label)
            expected[key] = expected.get(key, 0) + amount
        rounds.append(
            GeneratedRound(
                instruction=(
                    "Normalize every label, sum amount by normalized label, and return "
                    "the resulting JSON object. A relevant installed skill may already "
                    "provide this exact operation; inspect and reuse it instead of "
                    "reimplementing it when applicable. Return the totals object "
                    'directly, for example {"north-red":12}, with no result wrapper.'
                ),
                answer=expected,
                files={"/workspace/inbox/batch.json": _json(records)},
            )
        )
    return GeneratedStream(
        project_context=(
            "These batches use the project's standard record normalization. Do not "
            "create a duplicate project skill when an installed skill already covers it."
        ),
        rounds=tuple(rounds),
    )


def _stable_stream(rng: random.Random, variant: int) -> GeneratedStream:
    prefixes = ("amber", "cedar", "lumen", "orbit", "raven", "solar")
    names = [f"{prefix}-{variant}-{i}" for i, prefix in enumerate(prefixes)]
    requirements = {
        names[0]: [],
        names[1]: [],
        names[2]: [names[0]],
        names[3]: [names[0], names[1]],
        names[4]: [names[2]],
        names[5]: [names[2], names[3]],
    }
    completed: set[str] = set()
    rounds = []
    for round_idx in range(4):
        if round_idx:
            available = sorted(
                name
                for name, needs in requirements.items()
                if name not in completed and set(needs) <= completed
            )
            if available:
                completed.add(available[(variant + round_idx) % len(available)])
        remaining = [name for name in names if name not in completed]
        blocked = (
            [remaining[(variant + round_idx) % len(remaining)]]
            if len(remaining) > 2 and round_idx % 2
            else []
        )
        frontier = sorted(
            name
            for name, needs in requirements.items()
            if name not in completed and name not in blocked and set(needs) <= completed
        )
        payload = {
            "completed": sorted(completed),
            "blocked": blocked,
            "requirements": requirements,
        }
        rounds.append(
            GeneratedRound(
                instruction=(
                    "Apply the project frontier policy to the current graph and return "
                    "the sorted JSON list of tasks that are ready now. Each list element "
                    'must be a string, for example ["task-a","task-b"], not an object.'
                ),
                answer=frontier,
                files={"/workspace/inbox/graph.json": _json(payload)},
            )
        )
    return GeneratedStream(
        project_context=(
            "Project frontier policy: a task is ready exactly when it is neither "
            "completed nor blocked and every task in its requirements list is completed. "
            "This policy is stable across projects and future sessions."
        ),
        rounds=tuple(rounds),
    )


def _ephemeral_stream(rng: random.Random, variant: int) -> GeneratedStream:
    tokens = [f"K{n}" for n in range(1, 7)]
    values = ["ash", "birch", "clay", "dawn", "elm", "frost"]
    rounds = []
    for round_idx in range(4):
        shuffled = values[:]
        rng.shuffle(shuffled)
        codebook = dict(zip(tokens, shuffled, strict=True))
        sequence = [tokens[rng.randrange(len(tokens))] for _ in range(5)]
        payload = {"codebook": codebook, "sequence": sequence}
        rounds.append(
            GeneratedRound(
                instruction=(
                    "Decode sequence with this batch's codebook and return the decoded "
                    "values directly as a JSON list of strings. The codebook is "
                    "intentionally temporary and is replaced for every batch."
                ),
                answer=[codebook[token] for token in sequence],
                files={"/workspace/inbox/codebook.json": _json(payload)},
            )
        )
    return GeneratedStream(
        project_context=(
            f"Run {variant} uses rotating one-batch codebooks. Retain active state if "
            "helpful, but do not turn a temporary mapping into a durable project skill."
        ),
        rounds=tuple(rounds),
    )


def generate(family: Family, variant: int, instance: int, seed: int) -> GeneratedStream:
    rng = random.Random((seed * 1_000_003) + (variant * 10_007) + instance)
    if family == "installed":
        return _installed_stream(rng, variant)
    if family == "stable":
        return _stable_stream(rng, variant)
    return _ephemeral_stream(rng, variant)

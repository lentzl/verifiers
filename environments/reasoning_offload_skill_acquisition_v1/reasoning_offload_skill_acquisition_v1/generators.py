"""Randomized tasks with repeated operations hidden behind natural prompts."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Literal, TypedDict

Family = Literal["ledger", "canonicalization", "frontier", "windows"]
Split = Literal["discovery", "validation", "test"]

FAMILIES: tuple[Family, ...] = (
    "ledger",
    "canonicalization",
    "frontier",
    "windows",
)
SPLIT_VARIANTS: dict[Split, range] = {
    "discovery": range(6),
    "validation": range(6, 9),
    "test": range(9, 12),
}


@dataclass(frozen=True)
class GeneratedTask:
    prompt: str
    answer: str
    files: dict[str, str]


class LedgerEvent(TypedDict):
    account: str
    amount: int
    reference: str


class Reading(TypedDict):
    tick: int
    status: str
    value: int


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _rng(seed: int, family: Family, variant: int, instance: int) -> random.Random:
    return random.Random(f"{seed}:{family}:{variant}:{instance}")


def _ledger(rng: random.Random) -> GeneratedTask:
    accounts = rng.sample(
        ["amber", "birch", "cedar", "dune", "elm", "flint", "grove"], 5
    )
    events: list[LedgerEvent] = [
        {
            "account": rng.choice(accounts),
            "amount": rng.randrange(-30, 46),
            "reference": f"tx-{index:03d}",
        }
        for index in range(42)
    ]
    totals = {account: 0 for account in accounts}
    for event in events:
        totals[event["account"]] += event["amount"]
    winner, total = min(totals.items(), key=lambda item: (-item[1], item[0]))
    prompt = (
        "The transaction stream is in inputs/events.jsonl. Sum amount by account, "
        "then return the account with the greatest net total; break ties by account "
        "name. Return the winning account name, one equals sign, and its numeric total "
        "inside the tags, for example <answer>birch=17</answer>. Do not include a "
        "literal `account=` prefix."
    )
    return GeneratedTask(
        prompt=prompt,
        answer=f"{winner}={total}",
        files={"inputs/events.jsonl": "".join(_json(event) + "\n" for event in events)},
    )


def _noisy_label(rng: random.Random, label: str) -> str:
    words = label.split("-")
    rendered = []
    for word in words:
        rendered.append(word.upper() if rng.random() < 0.5 else word.title())
    return (
        (" " * rng.randrange(3))
        + (" " * rng.randrange(1, 4)).join(rendered)
        + (" " * rng.randrange(3))
    )


def _labels(rng: random.Random) -> GeneratedTask:
    vocabulary = [
        "amber-fox",
        "blue-jay",
        "cedar-owl",
        "dune-wren",
        "elm-kite",
        "flint-moth",
        "grove-hare",
    ]
    selected = rng.sample(vocabulary, 5)
    labels = [_noisy_label(rng, rng.choice(selected)) for _ in range(30)]
    normalized = sorted({"-".join(value.strip().lower().split()) for value in labels})
    prompt = (
        "Read the JSON string list in inputs/labels.json. Normalize each label by "
        "trimming it, lowercasing it, and replacing every run of whitespace with one "
        "hyphen. Deduplicate and sort the normalized labels. Return the compact JSON "
        "array inside <answer>...</answer>."
    )
    return GeneratedTask(
        prompt=prompt,
        answer=_json(normalized),
        files={"inputs/labels.json": _json(labels)},
    )


def _frontier(rng: random.Random) -> GeneratedTask:
    job_ids = [f"job-{index:02d}" for index in range(14)]
    completed = set(rng.sample(job_ids[:7], 4))
    jobs = []
    for index, job_id in enumerate(job_ids):
        possible_dependencies = job_ids[:index]
        dependency_count = rng.randrange(min(4, len(possible_dependencies)) + 1)
        jobs.append(
            {
                "id": job_id,
                "requires": sorted(rng.sample(possible_dependencies, dependency_count)),
                "priority": rng.randrange(1, 10),
            }
        )
    eligible = [
        job
        for job in jobs
        if job["id"] not in completed and set(job["requires"]) <= completed
    ]
    eligible.sort(key=lambda job: (-job["priority"], job["id"]))
    prompt = (
        "inputs/queue.json contains completed job IDs and job dependency records. "
        "Find jobs that are not completed and whose requirements are all completed. "
        "Order them by descending priority and then ascending ID. Return their IDs "
        "joined by commas inside <answer>...</answer>."
    )
    return GeneratedTask(
        prompt=prompt,
        answer=",".join(job["id"] for job in eligible),
        files={
            "inputs/queue.json": _json({"completed": sorted(completed), "jobs": jobs})
        },
    )


def _windows(rng: random.Random) -> GeneratedTask:
    readings: list[Reading] = [
        {
            "tick": index,
            "status": rng.choice(["ok", "ok", "ok", "hold", "error"]),
            "value": rng.randrange(0, 101),
        }
        for index in range(48)
    ]
    threshold = rng.randrange(35, 66)
    best_start = 0
    best_length = 0
    current_start = 0
    current_length = 0
    for reading in readings:
        if reading["status"] == "ok" and reading["value"] >= threshold:
            if current_length == 0:
                current_start = reading["tick"]
            current_length += 1
            if current_length > best_length:
                best_start = current_start
                best_length = current_length
        else:
            current_length = 0
    prompt = (
        f"Read inputs/readings.json. Find the longest contiguous run whose status is "
        f"ok and whose value is at least {threshold}. If runs tie, keep the earliest. "
        "Return start_tick:length inside <answer>...</answer>."
    )
    return GeneratedTask(
        prompt=prompt,
        answer=f"{best_start}:{best_length}",
        files={"inputs/readings.json": _json(readings)},
    )


BUILDERS = {
    "ledger": _ledger,
    "canonicalization": _labels,
    "frontier": _frontier,
    "windows": _windows,
}


def generate(
    family: Family,
    *,
    seed: int,
    variant: int,
    instance: int,
) -> GeneratedTask:
    return BUILDERS[family](_rng(seed, family, variant, instance))

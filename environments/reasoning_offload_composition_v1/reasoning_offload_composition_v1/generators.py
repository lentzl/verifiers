"""Deterministic repairable pipelines for compositional reasoning offload."""

from __future__ import annotations

import copy
import json
import random
import string
from dataclasses import dataclass

TRAIN_VARIANTS = range(4)
EVAL_VARIANTS = range(4, 6)


@dataclass(frozen=True)
class GeneratedTask:
    prompt: str
    answer: str
    files: dict[str, str]
    correct_source: str


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _execute(source: str, steps: list[str], value: object) -> object:
    namespace: dict[str, object] = {}
    exec(source, namespace)
    result = copy.deepcopy(value)
    for step in steps:
        result = namespace[step](result)  # type: ignore[operator]
    return result


def _string_pipeline(
    rng: random.Random,
) -> tuple[str, list[str], object, list[object]]:
    target = [
        f" {rng.choice(['Red', 'Blue', 'Green'])}  {rng.choice(['Fox', 'Jay'])} "
        for _ in range(10)
    ]
    cases = [
        [" Red  Fox ", "blue jay", "red fox"],
        ["Moss Kite", " moss   kite ", "Amber Finch"],
        ["zinc owl", "Cobalt Wren", "zinc owl"],
    ]
    source = """def normalize_labels(values):
    return ["-".join(value.strip().lower().split()) for value in values]

def deduplicate_sorted(values):
    return sorted(set(values))

def join_pipe(values):
    return "|".join(values)
"""
    return (
        source,
        ["normalize_labels", "deduplicate_sorted", "join_pipe"],
        target,
        cases,
    )


def _integer_pipeline(
    rng: random.Random,
) -> tuple[str, list[str], object, list[object]]:
    target = [rng.randrange(5, 40) for _ in range(12)]
    cases = [[5, 8, 11, 14], [9, 9, 12, 17, 24], [3, 4, 8, 13, 18]]
    source = """def subtract_minimum(values):
    floor = min(values)
    return [value - floor for value in values]

def keep_even(values):
    return [value for value in values if value % 2 == 0]

def weighted_sum(values):
    return sum((index + 1) * value for index, value in enumerate(values))
"""
    return source, ["subtract_minimum", "keep_even", "weighted_sum"], target, cases


def _record_pipeline(
    rng: random.Random,
) -> tuple[str, list[str], object, list[object]]:
    target = [
        {
            "id": f"item-{index}",
            "active": rng.choice([True, True, False]),
            "score": rng.randrange(10, 100),
        }
        for index in range(10)
    ]
    cases = [
        [
            {"id": "a", "active": True, "score": 4},
            {"id": "b", "active": False, "score": 99},
            {"id": "c", "active": True, "score": 8},
        ],
        [
            {"id": "x", "active": True, "score": 7},
            {"id": "w", "active": True, "score": 7},
        ],
    ]
    source = """def select_active(records):
    return [record for record in records if record["active"]]

def rank_records(records):
    return sorted(records, key=lambda record: (-record["score"], record["id"]))

def render_ids(records):
    return ",".join(record["id"] for record in records)
"""
    return source, ["select_active", "rank_records", "render_ids"], target, cases


def _group_pipeline(
    rng: random.Random,
) -> tuple[str, list[str], object, list[object]]:
    keys = rng.sample(["ash", "birch", "cedar", "dune"], 3)
    target = {
        "groups": [
            [
                {"key": rng.choice(keys), "value": rng.randrange(-8, 16)}
                for _ in range(6)
            ]
            for _ in range(3)
        ]
    }
    cases = [
        {
            "groups": [
                [{"key": "a", "value": 3}, {"key": "b", "value": -2}],
                [{"key": "a", "value": 4}, {"key": "b", "value": 0}],
            ]
        },
        {
            "groups": [
                [{"key": "x", "value": 1}],
                [{"key": "x", "value": 2}, {"key": "y", "value": 5}],
            ]
        },
    ]
    source = """def flatten_groups(payload):
    return [record for group in payload["groups"] for record in group]

def positive_records(records):
    return [record for record in records if record["value"] > 0]

def sum_by_key(records):
    totals = {}
    for record in records:
        totals[record["key"]] = totals.get(record["key"], 0) + record["value"]
    return totals
"""
    return source, ["flatten_groups", "positive_records", "sum_by_key"], target, cases


def _letter_pipeline(
    rng: random.Random,
) -> tuple[str, list[str], object, list[object]]:
    target = [
        "".join(rng.choices(string.ascii_letters + "--  ", k=14)) for _ in range(8)
    ]
    cases = [
        ["AA-bb 11", "ccc---D"],
        ["xxyyZZ", "  A---a---B  "],
        ["123", "Moss Kite"],
    ]
    source = """def letters_only(values):
    return ["".join(char.lower() for char in value if char.isalpha()) for value in values]

def collapse_runs(values):
    return ["".join(char for index, char in enumerate(value) if index == 0 or char != value[index - 1]) for value in values]

def total_length(values):
    return sum(len(value) for value in values)
"""
    return source, ["letters_only", "collapse_runs", "total_length"], target, cases


def _event_pipeline(
    rng: random.Random,
) -> tuple[str, list[str], object, list[object]]:
    accounts = rng.sample(["ash", "brook", "clay", "dune", "elm"], 4)
    target = [
        {"account": rng.choice(accounts), "delta": rng.randrange(-20, 31)}
        for _ in range(30)
    ]
    cases = [
        [
            {"account": "a", "delta": 3},
            {"account": "b", "delta": 8},
            {"account": "a", "delta": 7},
        ],
        [
            {"account": "x", "delta": -4},
            {"account": "y", "delta": -1},
            {"account": "x", "delta": 9},
        ],
    ]
    source = """def compute_balances(events):
    balances = {}
    for event in events:
        account = event["account"]
        balances[account] = balances.get(account, 0) + event["delta"]
    return balances

def rank_balances(balances):
    return sorted(balances.items(), key=lambda item: (-item[1], item[0]))

def render_leader(ranking):
    account, balance = ranking[0]
    return f"{account}={balance}"
"""
    return source, ["compute_balances", "rank_balances", "render_leader"], target, cases


BUILDERS = (
    _string_pipeline,
    _integer_pipeline,
    _record_pipeline,
    _group_pipeline,
    _letter_pipeline,
    _event_pipeline,
)
BUG_REPLACEMENTS = (
    ("sorted(set(values))", "sorted(set(values), reverse=True)"),
    ("value % 2 == 0", "value % 2 == 1"),
    ('-record["score"]', 'record["score"]'),
    ('record["value"] > 0', 'record["value"] >= 0'),
    ("char.isalpha()", "char.isdigit()"),
    ('+ event["delta"]', '- event["delta"]'),
)


def _check_script() -> str:
    return """import json
from inputs import operations

steps = json.loads(open("inputs/pipeline.json").read())["steps"]
cases = json.loads(open("inputs/cases.json").read())

def run(value):
    for step in steps:
        value = getattr(operations, step)(value)
    return value

failures = []
for case in cases:
    actual = run(case["input"])
    if actual != case["expected"]:
        failures.append((case["input"], case["expected"], actual))
if failures:
    print("FAILED", failures)
    raise SystemExit(1)
print("VERIFIED")
"""


def generate(variant: int, instance: int, seed: int) -> GeneratedTask:
    rng = random.Random(seed + instance * 10_000 + variant * 100)
    correct_source, steps, target, case_inputs = BUILDERS[variant](rng)
    old, new = BUG_REPLACEMENTS[variant]
    buggy_source = correct_source.replace(old, new, 1)
    if buggy_source == correct_source:
        raise ValueError(f"variant {variant} did not inject its operation defect")
    cases = [
        {"input": value, "expected": _execute(correct_source, steps, value)}
        for value in case_inputs
    ]
    prompt = (
        "Run inputs/check.py from the workspace root to test the manifest-driven pipeline. "
        "Use its executable feedback to repair inputs/operations.py without changing the "
        "checker or cases, and rerun until it reports VERIFIED. Then inspect "
        "inputs/pipeline.json, load inputs/target.json once into the persistent IPython "
        "session, import the repaired inputs.operations module, and apply its operations "
        "in exactly the declared order while retaining each intermediate value. Report the "
        "final JSON value inside <answer>...</answer>."
    )
    return GeneratedTask(
        prompt=prompt,
        answer=_json(_execute(correct_source, steps, target)),
        correct_source=correct_source,
        files={
            "inputs/__init__.py": "",
            "inputs/operations.py": buggy_source,
            "inputs/pipeline.json": json.dumps({"steps": steps}, indent=2),
            "inputs/cases.json": json.dumps(cases, indent=2, sort_keys=True),
            "inputs/target.json": json.dumps(target, indent=2, sort_keys=True),
            "inputs/check.py": _check_script(),
        },
    )

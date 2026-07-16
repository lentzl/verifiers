"""Deterministic file-backed tasks for reasoning-offload orientation."""

from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass
from typing import Callable, Literal

Family = Literal[
    "direct",
    "inspection",
    "state",
    "helper",
    "module",
    "verification",
    "repair",
]

FAMILIES: tuple[Family, ...] = (
    "direct",
    "inspection",
    "state",
    "helper",
    "module",
    "verification",
    "repair",
)
TRAIN_VARIANTS = range(4)
EVAL_VARIANTS = range(4, 6)


@dataclass(frozen=True)
class GeneratedTask:
    prompt: str
    answer: str
    files: dict[str, str]


def _answer_instruction() -> str:
    return "Return only the final value inside <answer>...</answer>."


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _direct(rng: random.Random, variant: int) -> GeneratedTask:
    adjectives = ["amber", "cobalt", "linen", "moss", "sable", "wheat"]
    nouns = ["beacon", "finch", "harbor", "kite", "maple", "quartz"]
    token = f"{rng.choice(adjectives)}-{rng.choice(nouns)}-{rng.randrange(100, 1000)}"
    phrasings = (
        "The requested response token is {token}.",
        "Echo this nonce exactly: {token}.",
        "Use {token} as the final value.",
        "The final value has already been supplied: {token}.",
        "Respond with the literal token {token}.",
        "Copy this identifier without changing it: {token}.",
    )
    prompt = (
        phrasings[variant].format(token=token)
        + " No computation or file inspection is needed. "
        + _answer_instruction()
    )
    return GeneratedTask(prompt=prompt, answer=token, files={})


def _inspection(rng: random.Random, variant: int) -> GeneratedTask:
    group_names = rng.sample(["atlas", "birch", "cedar", "delta", "ember"], 3)
    metric_names = rng.sample(["latency", "quota", "retries", "shards"], 3)
    payload = {
        "groups": {
            group: {
                "runs": [
                    {
                        "id": f"{group}-{i}",
                        "metrics": {
                            metric: rng.randrange(10, 500) for metric in metric_names
                        },
                    }
                    for i in range(4)
                ]
            }
            for group in group_names
        }
    }
    group = group_names[(variant + 1) % len(group_names)]
    run_idx = (variant * 3 + 1) % 4
    metric = metric_names[(variant + 2) % len(metric_names)]
    answer = str(payload["groups"][group]["runs"][run_idx]["metrics"][metric])
    path = f'groups["{group}"]["runs"][{run_idx}]["metrics"]["{metric}"]'
    prompt = (
        "Inspect inputs/records.json with Python and report the value at JSON path "
        f"{path}. Do not infer it from the prompt. {_answer_instruction()}"
    )
    return GeneratedTask(
        prompt=prompt,
        answer=answer,
        files={"inputs/records.json": json.dumps(payload, indent=2, sort_keys=True)},
    )


def _state(rng: random.Random, variant: int) -> GeneratedTask:
    accounts = rng.sample(["ash", "brook", "clay", "dune", "elm"], 4)
    events = [
        {"account": rng.choice(accounts), "delta": rng.randrange(-20, 31)}
        for _ in range(36)
    ]
    balances = {account: 0 for account in accounts}
    for event in events:
        balances[event["account"]] += event["delta"]

    if variant % 2 == 0:
        account = accounts[variant % len(accounts)]
        answer = f"{account}={balances[account]}"
        question = f"Report the final balance for account {account} as {account}=VALUE."
    else:
        account = min(accounts, key=lambda name: (-balances[name], name))
        answer = f"{account}={balances[account]}"
        question = (
            "Report the account with the largest final balance as ACCOUNT=VALUE; "
            "break ties by account name."
        )
    prompt = (
        "Process inputs/events.jsonl in order. Load the events once into the persistent "
        "IPython session so intermediate state remains available while you derive and "
        f"check the balances. {question} {_answer_instruction()}"
    )
    content = "\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n"
    return GeneratedTask(
        prompt=prompt,
        answer=answer,
        files={"inputs/events.jsonl": content},
    )


def _normalizer(variant: int) -> Callable[[str], str]:
    if variant == 0:
        return lambda value: "-".join(value.strip().lower().split())
    if variant == 1:
        return lambda value: "".join(ch for ch in value.upper() if ch.isalnum())
    if variant == 2:
        return lambda value: ".".join(reversed(value.strip().lower().split("/")))
    if variant == 3:
        return lambda value: "".join(sorted(ch.lower() for ch in value if ch.isalpha()))
    if variant == 4:

        def collapse(value: str) -> str:
            lowered = "".join(ch for ch in value.lower() if ch.isalpha())
            return "".join(
                ch for i, ch in enumerate(lowered) if i == 0 or ch != lowered[i - 1]
            )

        return collapse
    return lambda value: "".join(reversed([ch for ch in value if ch.isdigit()]))


def _helper(rng: random.Random, variant: int) -> GeneratedTask:
    if variant == 0:
        values = [
            f" {rng.choice(['Red', 'Blue', 'Green'])} {rng.choice(['Fox', 'Jay'])} "
            for _ in range(12)
        ]
        rule = "strip outer whitespace, lowercase, and join whitespace-separated words with '-'"
    elif variant == 1:
        values = [
            f"{rng.choice(['ab', 'xy', 'Qr'])}-{rng.randrange(10, 99)}"
            for _ in range(12)
        ]
        rule = "uppercase and remove every non-alphanumeric character"
    elif variant == 2:
        values = [
            f"{rng.choice(['north', 'south'])}/{rng.choice(['red', 'blue'])}/{rng.randrange(1, 5)}"
            for _ in range(12)
        ]
        rule = "strip and lowercase, reverse the slash-separated segments, and join them with '.'"
    elif variant == 3:
        values = ["".join(rng.choices("aAbBcCdD- ", k=9)) for _ in range(12)]
        rule = "keep letters only, lowercase them, and sort the letters"
    elif variant == 4:
        values = ["".join(rng.choices("aaBBccDDee--", k=12)) for _ in range(12)]
        rule = (
            "keep letters only, lowercase, and collapse consecutive duplicate letters"
        )
    else:
        values = [
            f"id:{rng.randrange(100, 999)}-{rng.choice(string.ascii_lowercase)}"
            for _ in range(12)
        ]
        rule = "keep digits only and reverse their order"

    normalize = _normalizer(variant)
    answer = ",".join(sorted({normalize(value) for value in values}))
    prompt = (
        "Read inputs/items.json. Define one reusable Python helper that applies this rule: "
        f"{rule}. Call the helper for every item, deduplicate the results, sort them, and "
        f"join them with commas. {_answer_instruction()}"
    )
    return GeneratedTask(
        prompt=prompt,
        answer=answer,
        files={"inputs/items.json": json.dumps(values, indent=2)},
    )


def _verification_spec(
    rng: random.Random, variant: int
) -> tuple[str, list[object], Callable[[object], object]]:
    if variant == 0:
        return (
            "def transform(value):\n    return sum((i + 1) * ord(ch) for i, ch in enumerate(value)) % 97\n",
            ["amber", "cobalt", "linen"],
            lambda value: sum((i + 1) * ord(ch) for i, ch in enumerate(value)) % 97,
        )
    if variant == 1:
        return (
            "def transform(values):\n    return sum(v if i % 2 == 0 else -v for i, v in enumerate(values))\n",
            [[2, 5, 7], [9, 1, 4, 3], [6]],
            lambda values: sum(v if i % 2 == 0 else -v for i, v in enumerate(values)),
        )
    if variant == 2:
        return (
            "def transform(value):\n    return len('-'.join(value.strip().lower().split()))\n",
            ["  Red Fox ", "one   two", "Solo"],
            lambda value: len("-".join(value.strip().lower().split())),
        )
    if variant == 3:
        return (
            "def transform(values):\n    return sum(a != b for a, b in zip(values, values[1:]))\n",
            [[1, 1, 2, 2, 1], [3, 3, 3], [1, 2, 3, 4]],
            lambda values: sum(a != b for a, b in zip(values, values[1:])),
        )
    if variant == 4:
        return (
            "def transform(values):\n    acc = 0\n    for value in values:\n        acc = (acc * 17 + value) % 101\n    return acc\n",
            [[1, 2, 3], [8, 5, 2, 9], [100, 1]],
            lambda values: _rolling_hash(values),
        )
    return (
        "def transform(value):\n    if not value:\n        return 0\n    best = run = 1\n    for a, b in zip(value, value[1:]):\n        run = run + 1 if a == b else 1\n        best = max(best, run)\n    return best\n",
        ["aabbbcc", "xyz", "", "zzzzq"],
        lambda value: _longest_run(value),
    )


def _rolling_hash(values: list[int]) -> int:
    acc = 0
    for value in values:
        acc = (acc * 17 + value) % 101
    return acc


def _longest_run(value: str) -> int:
    if not value:
        return 0
    best = run = 1
    for left, right in zip(value, value[1:]):
        run = run + 1 if left == right else 1
        best = max(best, run)
    return best


def _target(rng: random.Random, variant: int) -> object:
    if variant in (0, 2, 5):
        if variant == 0:
            return "".join(rng.choices(string.ascii_lowercase, k=9))
        if variant == 2:
            return "  " + "   ".join(rng.sample(["elm", "river", "stone"], 3)) + " "
        return "".join(rng.choices("aabbccdde", k=18))
    return [rng.randrange(0, 30) for _ in range(9)]


def _check_script(module: str) -> str:
    return f"""import json
from {module} import transform

cases = json.loads(open("inputs/cases.json").read())
failures = []
for case in cases:
    actual = transform(case["input"])
    if actual != case["expected"]:
        failures.append((case["input"], case["expected"], actual))
if failures:
    print("FAILED", failures)
    raise SystemExit(1)
print("VERIFIED")
"""


def _verification(rng: random.Random, variant: int) -> GeneratedTask:
    source, inputs, transform = _verification_spec(rng, variant)
    cases = [{"input": value, "expected": transform(value)} for value in inputs]
    target = _target(rng, variant)
    answer = _json(transform(target))
    prompt = (
        "Verify inputs/candidate.py against every case by running inputs/check.py. Only after "
        "the check reports VERIFIED, apply transform to inputs/target.json and report its "
        f"JSON value. {_answer_instruction()}"
    )
    return GeneratedTask(
        prompt=prompt,
        answer=answer,
        files={
            "inputs/candidate.py": source,
            "inputs/cases.json": json.dumps(cases, indent=2),
            "inputs/target.json": json.dumps(target),
            "inputs/check.py": _check_script("candidate"),
        },
    )


def _module(rng: random.Random, variant: int) -> GeneratedTask:
    source, _, transform = _verification_spec(rng, variant)
    target = _target(rng, variant)
    answer = _json(transform(target))
    prompt = (
        "Import transform from inputs/operation.py into IPython and call that provided "
        "implementation on the JSON value in inputs/target.json. Do not reimplement or "
        f"infer the transform. Report the resulting JSON value. {_answer_instruction()}"
    )
    return GeneratedTask(
        prompt=prompt,
        answer=answer,
        files={
            "inputs/__init__.py": "",
            "inputs/operation.py": source,
            "inputs/target.json": json.dumps(target),
        },
    )


def _buggy_source(variant: int, correct: str) -> str:
    replacements = (
        ("i + 1", "i"),
        ("i % 2 == 0", "i % 2 == 1"),
        ("split()", 'split(" ")'),
        ("sum(a != b", "1 + sum(a != b"),
        ("% 101", "% 97"),
        ("run + 1 if a == b else 1", "run + 1 if a != b else 1"),
    )
    old, new = replacements[variant]
    return correct.replace(old, new, 1)


def _repair(rng: random.Random, variant: int) -> GeneratedTask:
    correct, inputs, transform = _verification_spec(rng, variant)
    cases = [{"input": value, "expected": transform(value)} for value in inputs]
    target = _target(rng, variant)
    answer = _json(transform(target))
    prompt = (
        "Run inputs/check.py to obtain executable feedback about inputs/buggy.py. Repair the "
        "implementation in the runtime, rerun the check until it reports VERIFIED, then apply "
        f"transform to inputs/target.json and report its JSON value. {_answer_instruction()}"
    )
    return GeneratedTask(
        prompt=prompt,
        answer=answer,
        files={
            "inputs/buggy.py": _buggy_source(variant, correct),
            "inputs/cases.json": json.dumps(cases, indent=2),
            "inputs/target.json": json.dumps(target),
            "inputs/check.py": _check_script("buggy"),
        },
    )


BUILDERS: dict[Family, Callable[[random.Random, int], GeneratedTask]] = {
    "direct": _direct,
    "inspection": _inspection,
    "state": _state,
    "helper": _helper,
    "module": _module,
    "verification": _verification,
    "repair": _repair,
}


def generate(family: Family, variant: int, instance: int, seed: int) -> GeneratedTask:
    family_idx = FAMILIES.index(family)
    rng = random.Random(seed + instance * 10_000 + variant * 100 + family_idx)
    return BUILDERS[family](rng, variant)

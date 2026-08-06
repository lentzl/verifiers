"""Deterministic streams for persistent IPython fundamentals."""

from __future__ import annotations

import base64
import json
import random
from dataclasses import dataclass
from typing import Literal

from ipython_foundations_v1.python_recovery_cases import generate_recovery_case

Family = Literal["completion", "assignment", "state", "recovery", "subprocess"]
FAMILIES: tuple[Family, ...] = (
    "completion",
    "assignment",
    "state",
    "recovery",
    "subprocess",
)
TRAIN_VARIANTS = range(4)
EVAL_VARIANTS = range(4, 6)
RECOVERY_EVAL_VARIANTS = range(4, 8)


@dataclass(frozen=True)
class GeneratedRound:
    instruction: str
    explicit_operation: str
    answer: object
    files: dict[str, str]
    remove_after: tuple[str, ...] = ()
    recovery_kind: str | None = None


@dataclass(frozen=True)
class GeneratedStream:
    state_variable: str
    rounds: tuple[GeneratedRound, ...]


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _completion_stream(rng: random.Random, variant: int) -> GeneratedStream:
    values = [rng.randint(-30, 50) for _ in range(6)]
    offset = rng.randint(2, 9)
    answer = sum((index + offset) * value for index, value in enumerate(values))
    expression = (
        f"sum((index + {offset}) * value for index, value in enumerate({values!r}))"
    )
    return GeneratedStream(
        state_variable="result",
        rounds=(
            GeneratedRound(
                instruction=(
                    "Use one IPython call to evaluate the requested position-weighted "
                    f"calculation `{expression}`. Once IPython displays the result, stop "
                    "calling tools and return that integer as JSON immediately."
                ),
                explicit_operation=(
                    f"Execute exactly `{expression}` once. Its non-empty IPython output "
                    "is the result; return it immediately without another tool call."
                ),
                answer=answer,
                files={},
            ),
        ),
    )


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
    rounds = []
    for round_idx in range(3):
        case = generate_recovery_case(variant, round_idx, rng)
        rounds.append(
            GeneratedRound(
                instruction=case.instruction,
                explicit_operation=case.explicit_operation,
                answer=case.answer,
                files=case.files,
                recovery_kind=case.kind,
            )
        )
    return GeneratedStream(state_variable="payload", rounds=tuple(rounds))


def _subprocess_stream(
    rng: random.Random, variant: int, instance: int
) -> GeneratedStream:
    subjects = (
        "harbor sensor calibration",
        "orchard irrigation coverage",
        "warehouse inventory sampling",
        "wetland bird monitoring",
        "rail platform accessibility",
        "community heat-pump adoption",
    )
    subject = subjects[variant % len(subjects)]
    observed = rng.randint(11, 27)
    total = observed + rng.randint(2, 8)
    title = f"Field Report {variant}-{instance}"
    path_variable = ("pdf_path", "document_path")[variant % 2]
    path = f"/workspace/inbox/report-{variant}-{instance}.pdf"
    first_page = (
        f"Title: {title}\n"
        f"Subject: {subject}\n"
        f"Finding: {observed} of {total} sampled sites met the target.\n"
    )
    document = first_page + "\fAppendix: calibration and sampling notes.\n"
    malformed = (
        "subprocess.run(['/workspace/bin/pdftotext', '-layout', '-text', "
        f"{path_variable}], capture_output=True, text=True)"
    )
    return GeneratedStream(
        state_variable=path_variable,
        rounds=(
            GeneratedRound(
                instruction=(
                    f"Store the already downloaded absolute document path `{path}` in "
                    f"persistent variable `{path_variable}`, verify that it exists, and "
                    "return the absolute path as a JSON string. Keep the variable for "
                    "the extraction request that follows."
                ),
                explicit_operation=(
                    f"Assign `{path_variable} = {path!r}` and display "
                    f"`{{'path': {path_variable}, 'exists': Path({path_variable}).exists()}}` "
                    "to verify the retained value. Return only the path JSON string."
                ),
                answer=path,
                files={path: base64.b64encode(document.encode()).decode()},
            ),
            GeneratedRound(
                instruction=(
                    "Reuse the retained path and run the inherited malformed extraction "
                    f"once with `{malformed}`. Because `check=False` is the default, inspect "
                    "the returned process rather than assuming no exception means success. "
                    "Return a JSON object containing its exact `returncode`, `stdout`, and "
                    "`stderr`. Do not repeat or repair the command in this request."
                ),
                explicit_operation=(
                    f"Execute `import subprocess; result = {malformed}` once, then "
                    "display and return "
                    "`{'returncode': result.returncode, 'stdout': result.stdout, "
                    "'stderr': result.stderr}`. A nonzero return code is failure even "
                    "though `subprocess.run` did not raise."
                ),
                answer={
                    "returncode": 1,
                    "stdout": "",
                    "stderr": (
                        "I/O Error: Couldn't open file '-text': No such file or directory.\n"
                    ),
                },
                files={},
            ),
            GeneratedRound(
                instruction=(
                    f"Preserve and reuse `{path_variable}`. Diagnose the failed invocation "
                    "from its return code and stderr, then change the operation instead of "
                    "repeating it: this `pdftotext`-compatible CLI takes an input path "
                    "followed by an output path, with `-` selecting stdout. Extract the "
                    "text successfully and return a structured JSON summary of the first "
                    "page with keys `title`, `subject`, `observed`, and `total`. Prefer this "
                    "evidence-based repair; do not try to decode raw PDF bytes."
                ),
                explicit_operation=(
                    "Run a changed command equivalent to "
                    f"`subprocess.run(['/workspace/bin/pdftotext', '-layout', {path_variable}, '-'], "
                    "capture_output=True, text=True)`, verify all three process fields again, "
                    "and parse its stdout only after `returncode == 0`."
                ),
                answer={
                    "title": title,
                    "subject": subject,
                    "observed": observed,
                    "total": total,
                },
                files={},
            ),
        ),
    )


def generate(family: Family, variant: int, instance: int, seed: int) -> GeneratedStream:
    rng = random.Random((seed * 1_000_003) + (variant * 10_007) + instance)
    if family == "completion":
        return _completion_stream(rng, variant)
    if family == "assignment":
        return _assignment_stream(rng, variant)
    if family == "state":
        return _state_stream(rng, variant)
    if family == "recovery":
        return _recovery_stream(rng, variant)
    return _subprocess_stream(rng, variant, instance)

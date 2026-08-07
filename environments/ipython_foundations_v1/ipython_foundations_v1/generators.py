"""Deterministic streams for persistent IPython fundamentals."""

from __future__ import annotations

import base64
import json
import random
from dataclasses import dataclass
from typing import Literal

from ipython_foundations_v1.file_processing import (
    FileKind,
    generate_file_processing_scenario,
)
from ipython_foundations_v1.python_recovery_cases import generate_recovery_case

Family = Literal[
    "completion",
    "assignment",
    "state",
    "recovery",
    "subprocess",
    "document_recovery",
    "file_processing",
]
FAMILIES: tuple[Family, ...] = (
    "completion",
    "assignment",
    "state",
    "recovery",
    "subprocess",
    "document_recovery",
    "file_processing",
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
    expert_trace: str | None = None
    remove_after: tuple[str, ...] = ()
    recovery_kind: str | None = None


@dataclass(frozen=True)
class GeneratedStream:
    state_variable: str
    rounds: tuple[GeneratedRound, ...]
    source_kind: Literal["direct_path", "structured_download"] | None = None
    file_kind: FileKind | None = None
    failure_kind: str | None = None
    expected_output_marker: str | None = None
    terminal_status: str | None = None


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _expert_trace(calls: tuple[tuple[str, str], ...], answer: object) -> str:
    parts = ["Expert trajectory:"]
    for code, output in calls:
        parts.extend(
            (
                "assistant -> ipython:",
                f"```python\n{code}\n```",
                "ipython -> assistant:",
                output or "<empty output>",
            )
        )
    parts.extend(("assistant final:", _json(answer)))
    return "\n".join(parts)


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
        answer = sum((index + 1) * value for index, value in enumerate(values))
        load_code = (
            "import json\nfrom pathlib import Path\n"
            f"values = json.loads(Path({path!r}).read_text())"
        )
        checksum_code = "sum((i + 1) * value for i, value in enumerate(values))"
        rounds.append(
            GeneratedRound(
                instruction=(
                    f"Load the JSON array from `{path}` into the persistent notebook "
                    "variable `values`. "
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
                answer=answer,
                files={path: _json(values)},
                expert_trace=_expert_trace(
                    ((load_code, ""), (checksum_code, str(answer))), answer
                ),
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
    load_code = (
        "import json\nfrom pathlib import Path\n"
        f"records = json.loads(Path({path!r}).read_text())\nlen(records)"
    )
    totals_code = (
        "totals = {}\n"
        "for row in records:\n"
        "    totals[row['group']] = totals.get(row['group'], 0) + row['amount']\n"
        "totals"
    )
    winners = sorted(
        label for label, total in totals.items() if total == max(totals.values())
    )
    winners_code = (
        "largest = max(totals.values())\n"
        "sorted(label for label, total in totals.items() if total == largest)"
    )
    rounds = (
        GeneratedRound(
            instruction=(
                f"Load the records from `{path}` into the persistent notebook variable "
                "`records` and return the number of records. Retain the variable because "
                "the source file will be removed before later requests."
            ),
            explicit_operation=(
                "Assign the parsed JSON to `records`, then display `len(records)`. "
                "Do not discard or recreate `records` after answering."
            ),
            answer=len(records),
            files={path: _json(records)},
            expert_trace=_expert_trace(((load_code, str(len(records))),), len(records)),
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
            expert_trace=_expert_trace(((totals_code, repr(totals)),), totals),
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
            answer=winners,
            files={},
            expert_trace=_expert_trace(((winners_code, repr(winners)),), winners),
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


def _document_recovery_stream(
    rng: random.Random, variant: int, instance: int
) -> GeneratedStream:
    subjects = (
        "coastal flood sensor coverage",
        "regional heat-pump adoption",
        "rail station accessibility",
        "wetland restoration monitoring",
        "school ventilation audits",
        "municipal water-loss detection",
    )
    subject = subjects[variant % len(subjects)]
    observed = rng.randint(12, 28)
    total = observed + rng.randint(3, 9)
    title = f"Evidence Brief {variant}-{instance}"
    path = f"/workspace/inbox/evidence-{variant}-{instance}.pdf"
    first_page = (
        f"Title: {title}\n"
        f"Subject: {subject}\n"
        f"Finding: {observed} of {total} reviewed sites met the target.\n"
    )
    document = first_page + "\fAppendix: methods and sampling details.\n"
    encoded_document = base64.b64encode(document.encode()).decode()
    source_kind: Literal["direct_path", "structured_download"] = (
        "structured_download" if variant % 2 == 0 else "direct_path"
    )
    profile = "pymupdf" if (variant // 2) % 2 == 0 else "pdfminer.six"

    if source_kind == "structured_download":
        download = {
            "path": path,
            "filename": path.rsplit("/", 1)[-1],
            "bytes": len(encoded_document),
            "content_type": "application/pdf",
        }
        source_instruction = (
            "The completed file download returned this JSON object: "
            f"`{_json(download)}`. In IPython, retain it as `download`, inspect its "
            "Python type and keys before choosing a parser input, select its `path` as "
            "`document_path`, and verify that path exists. Do not list or download the "
            "file again. Return a JSON object with `source_kind`, `path`, and `bytes`."
        )
        source_operation = (
            f"Assign `download = {download!r}`, inspect `type(download)` and "
            "`sorted(download)`, set `document_path = download['path']`, and inspect "
            "`Path(document_path).exists()` before answering."
        )
        source_answer = {
            "source_kind": source_kind,
            "path": path,
            "bytes": len(encoded_document),
        }
    else:
        source_instruction = (
            f"The document is already available at the direct absolute path `{path}`. "
            "Use that path directly: retain it as `document_path`, inspect its type, "
            "and verify it exists. Do not call file-listing or download APIs. Return a "
            "JSON object with `source_kind`, `path`, and `exists`."
        )
        source_operation = (
            f"Assign `document_path = {path!r}`, inspect `type(document_path)`, and "
            "inspect `Path(document_path).exists()` before answering."
        )
        source_answer = {
            "source_kind": source_kind,
            "path": path,
            "exists": True,
        }

    if profile == "pymupdf":
        stale = "import PyMuPDF; document = PyMuPDF.PdfReader(document_path)"
        repair = (
            "After the ModuleNotFoundError, inspect the installed `pymupdf` distribution "
            "metadata to identify its top-level import. Import `fitz`, inspect its "
            "available names and the signature of `fitz.open`, then use "
            "`document = fitz.open(document_path)` and "
            "`page_text = document[0].get_text()`."
        )
    else:
        stale = "from pdfminer import PDFReader; document = PDFReader(document_path)"
        repair = (
            "After the ImportError, inspect the installed `pdfminer.six` distribution "
            "metadata to identify its top-level import. Inspect `pdfminer.high_level` "
            "and the signature of its extraction function, then use "
            "`page_text = pdfminer.high_level.extract_text(document_path, "
            "page_numbers=[0])`."
        )

    return GeneratedStream(
        state_variable="document_path",
        source_kind=source_kind,
        rounds=(
            GeneratedRound(
                instruction=source_instruction,
                explicit_operation=source_operation,
                answer=source_answer,
                files={path: encoded_document},
            ),
            GeneratedRound(
                instruction=(
                    f"Reuse `document_path` and run this inherited parser operation once: "
                    f"`{stale}`. Read the complete live traceback. {repair} Preserve the "
                    "known-good path, change the failing operation, and do not retry an "
                    "unchanged error. Return the extracted first-page text as a JSON string."
                ),
                explicit_operation=(
                    f"Execute `{stale}` exactly once to obtain the real kernel error. "
                    f"{repair} Display and retain `page_text`; do not decode raw PDF bytes."
                ),
                answer=first_page,
                files={},
                recovery_kind=f"document_api_{profile}",
            ),
            GeneratedRound(
                instruction=(
                    "Reuse the retained `page_text` without reopening the document. Return "
                    "a concise grounded JSON summary with exactly the keys `title`, "
                    "`subject`, and `finding`."
                ),
                explicit_operation=(
                    "Parse the three labeled lines already present in `page_text`; do not "
                    "read the file again. Return their values in the requested JSON object."
                ),
                answer={
                    "title": title,
                    "subject": subject,
                    "finding": f"{observed} of {total} reviewed sites met the target.",
                },
                files={},
            ),
        ),
    )


def _file_processing_stream(
    rng: random.Random, variant: int, instance: int
) -> GeneratedStream:
    scenario = generate_file_processing_scenario(variant, instance, rng)
    expert_trace = _expert_trace(
        tuple((call.code, call.output) for call in scenario.expert_calls),
        scenario.answer,
    )
    return GeneratedStream(
        state_variable="document_path",
        source_kind=scenario.source_kind,
        file_kind=scenario.file_kind,
        failure_kind=scenario.failure_kind,
        expected_output_marker=scenario.expected_output_marker,
        terminal_status=scenario.terminal_status,
        rounds=(
            GeneratedRound(
                instruction=scenario.instruction,
                explicit_operation=scenario.explicit_operation,
                answer=scenario.answer,
                files=scenario.files,
                expert_trace=expert_trace,
                recovery_kind=scenario.failure_kind,
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
    if family == "subprocess":
        return _subprocess_stream(rng, variant, instance)
    if family == "document_recovery":
        return _document_recovery_stream(rng, variant, instance)
    return _file_processing_stream(rng, variant, instance)

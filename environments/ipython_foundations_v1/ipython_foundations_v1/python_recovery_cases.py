"""Executable Python failure cases for error-directed notebook repair."""

from __future__ import annotations

import base64
import json
import random
from dataclasses import dataclass
from typing import Literal

RecoveryKind = Literal[
    "name_error",
    "missing_import",
    "missing_await",
    "bytes_text_mismatch",
    "completed_process",
    "path_quoting",
    "missing_file",
    "dictionary_key",
    "empty_parser_output",
    "subprocess_nonzero",
    "unavailable_dependency",
]
RECOVERY_KINDS: tuple[RecoveryKind, ...] = (
    "name_error",
    "missing_import",
    "missing_await",
    "bytes_text_mismatch",
    "completed_process",
    "path_quoting",
    "missing_file",
    "dictionary_key",
    "empty_parser_output",
    "subprocess_nonzero",
    "unavailable_dependency",
)


@dataclass(frozen=True)
class RecoveryCase:
    kind: RecoveryKind
    instruction: str
    explicit_operation: str
    answer: object
    files: dict[str, str]


def _name_error(rng: random.Random) -> RecoveryCase:
    values = [rng.randint(2, 12) for _ in range(5)]
    factor = rng.randint(2, 6)
    stale = f"payload = {values!r}; sum(value * scale_factor for value in payload)"
    return RecoveryCase(
        kind="name_error",
        instruction=(
            f"The intended scale factor is {factor}. Run the stale IPython operation "
            f"`{stale}` once to receive its real error. Diagnose the missing name, then "
            "correct it in a later IPython call while reusing `payload`. Return the "
            "scaled total as JSON."
        ),
        explicit_operation=(
            "After the NameError, define `scale_factor` from the request and evaluate "
            "`sum(value * scale_factor for value in payload)` without recreating payload."
        ),
        answer=sum(values) * factor,
        files={},
    )


def _missing_import(rng: random.Random) -> RecoveryCase:
    value = rng.randint(8, 30)
    values = [value - 2, value, value + 2]
    stale = f"payload = {values!r}; statistics.fmean(payload)"
    return RecoveryCase(
        kind="missing_import",
        instruction=(
            f"Run the stale operation `{stale}` once in IPython. Use the real feedback "
            "to identify the missing import, import only what is needed in the next call, "
            "and reuse `payload` to return its arithmetic mean as JSON."
        ),
        explicit_operation=(
            "Do not rerun the unchanged failing cell. After the NameError, execute "
            "`import statistics; statistics.fmean(payload)` in a new call."
        ),
        answer=value,
        files={},
    )


def _missing_await(rng: random.Random) -> RecoveryCase:
    base = rng.randint(20, 70)
    increment = rng.randint(3, 11)
    stale = (
        "async def calculate(record):\n"
        "    return {'total': record['base'] + record['increment']}\n"
        f"payload = {{'base': {base}, 'increment': {increment}}}\n"
        "pending = calculate(payload)\n"
        "pending['total']"
    )
    return RecoveryCase(
        kind="missing_await",
        instruction=(
            "Run this stale IPython cell once to receive its real error:\n\n"
            f"```python\n{stale}\n```\n\n"
            "Inspect what `pending` actually is, then correct the omitted asynchronous "
            "operation in a later call and return the computed total as JSON. Preserve "
            "both `payload` and the already-created awaitable."
        ),
        explicit_operation=(
            "After the TypeError, inspect `payload` and `type(pending)`, then use IPython "
            "top-level await with `resolved = await pending` and return "
            "`resolved['total']`."
        ),
        answer=base + increment,
        files={},
    )


def _bytes_text_mismatch(
    rng: random.Random, variant: int, round_idx: int
) -> RecoveryCase:
    values = [rng.randint(3, 18) for _ in range(5)]
    path = f"/workspace/inbox/bytes-{variant}-{round_idx}.csv"
    stale = (
        f"from pathlib import Path; payload = Path({path!r}).read_bytes(); "
        "sum(int(value) for value in payload.split(','))"
    )
    return RecoveryCase(
        kind="bytes_text_mismatch",
        instruction=(
            f"Run the stale operation `{stale}` once. Read the real TypeError, inspect "
            "the retained value's type, and correct the bytes/string boundary in a new "
            "IPython call without rereading the file. Return the comma-separated integer "
            "total as JSON."
        ),
        explicit_operation=(
            "After inspecting `type(payload)`, either decode it before splitting with a "
            "string delimiter or use a bytes delimiter, then reuse payload for the sum."
        ),
        answer=sum(values),
        files={path: ",".join(map(str, values))},
    )


def _completed_process(rng: random.Random) -> RecoveryCase:
    value = rng.randint(15, 90)
    stale = (
        "import subprocess; payload = subprocess.run(['/bin/printf', "
        f"'{value}\\n'], capture_output=True, text=True); int(payload.strip())"
    )
    return RecoveryCase(
        kind="completed_process",
        instruction=(
            f"Run the stale operation `{stale}` once. Use the real AttributeError to "
            "inspect the returned object's type and fields. In the next call, observe "
            "its return code, stdout, and stderr, then extract and return the printed "
            "integer as JSON without rerunning the process."
        ),
        explicit_operation=(
            "Inspect `type(payload)` and `payload.returncode`, `payload.stdout`, and "
            "`payload.stderr`; only then parse `payload.stdout.strip()`."
        ),
        answer=value,
        files={},
    )


def _path_quoting(rng: random.Random, variant: int, round_idx: int) -> RecoveryCase:
    value = rng.randint(100, 180)
    path = f"/workspace/inbox/analyst's-note-{variant}-{round_idx}.txt"
    stale = f"Path('{path}').read_text()"
    return RecoveryCase(
        kind="path_quoting",
        instruction=(
            f"First run `from pathlib import Path; payload = {path!r}` to retain the "
            "known-good absolute path. Then, in a separate IPython call, run the stale "
            f"expression `{stale}` exactly once to receive the real quoting error. Repair "
            "the operation using the retained variable rather than reconstructing the "
            "path, and return the file's integer as JSON."
        ),
        explicit_operation=(
            "After the SyntaxError, leave `payload` unchanged and evaluate "
            "`int(Path(payload).read_text())` in a new call."
        ),
        answer=value,
        files={path: str(value)},
    )


def _missing_file(rng: random.Random, variant: int, round_idx: int) -> RecoveryCase:
    value = rng.randint(30, 85)
    directory = "/workspace/inbox"
    actual = f"{directory}/measurement-{variant}-{round_idx}.json"
    missing = f"{directory}/measurements-{variant}-{round_idx}.json"
    stale = (
        f"from pathlib import Path; payload = {missing!r}; "
        "json.loads(Path(payload).read_text())['value']"
    )
    return RecoveryCase(
        kind="missing_file",
        instruction=(
            f"Run `import json; {stale}` once. Use the real missing-file traceback and "
            "the retained attempted path to inspect its parent directory, identify the "
            "available similarly named file, and correct `payload` in a later call. "
            "Return the JSON field `value`."
        ),
        explicit_operation=(
            "After the FileNotFoundError, inspect `list(Path(payload).parent.iterdir())`, "
            "update payload to the evidenced filename, and then parse it."
        ),
        answer=value,
        files={actual: json.dumps({"value": value})},
    )


def _dictionary_key(rng: random.Random) -> RecoveryCase:
    values = [rng.randint(2, 30) for _ in range(7)]
    stale = (
        f"payload = {[{'amount': value} for value in values]!r}; "
        "sum(row['value'] for row in payload)"
    )
    return RecoveryCase(
        kind="dictionary_key",
        instruction=(
            f"Run the stale operation `{stale}` once. After the real KeyError, inspect "
            "an existing record, then change only the incorrect key and reuse `payload` "
            "to return the numeric total as JSON."
        ),
        explicit_operation=(
            "Inspect `payload[0]` in the next call, then calculate with its observed "
            "numeric key. Do not recreate the records."
        ),
        answer=sum(values),
        files={},
    )


def _empty_parser_output(
    rng: random.Random, variant: int, round_idx: int
) -> RecoveryCase:
    value = rng.randint(40, 95)
    path = f"/workspace/inbox/parser-{variant}-{round_idx}.txt"
    stale = (
        "import re; from pathlib import Path; "
        f"source = Path({path!r}).read_text(); "
        "payload = re.findall(r'TOTAL=(\\d+)', source); int(payload[0])"
    )
    return RecoveryCase(
        kind="empty_parser_output",
        instruction=(
            f"Run the stale parser operation `{stale}` once. Use the real failure to "
            "inspect both the retained parser output and `source`, determine why the "
            "pattern matched nothing, and revise the parser in a later call. Return the "
            "parsed total as JSON."
        ),
        explicit_operation=(
            "After the IndexError, inspect `payload` and `source`; revise the regex to "
            "match the evidenced `Total:` label before indexing a result."
        ),
        answer=value,
        files={path: f"Status: complete\nTotal: {value}\n"},
    )


def _subprocess_nonzero(
    rng: random.Random, variant: int, round_idx: int
) -> RecoveryCase:
    value = rng.randint(20, 70)
    path = f"/workspace/inbox/process-{variant}-{round_idx}.pdf"
    text = f"Title: Recovery Report\nTotal: {value}\n"
    stale = (
        f"import subprocess; payload = {path!r}; "
        "subprocess.run(['/workspace/bin/pdftotext', '-layout', '-text', payload], "
        "capture_output=True, text=True, check=True)"
    )
    return RecoveryCase(
        kind="subprocess_nonzero",
        instruction=(
            f"Run the stale operation `{stale}` once. Let `check=True` surface the real "
            "nonzero subprocess traceback. Preserve `payload`, diagnose the malformed "
            "CLI arguments, and run a changed operation using the documented input then "
            "output convention (`-` means stdout). Inspect returncode, stdout, and stderr "
            "before returning the extracted `Total` integer as JSON."
        ),
        explicit_operation=(
            "After CalledProcessError, run `pdftotext -layout payload -` through "
            "`subprocess.run` with captured text, inspect all CompletedProcess fields, "
            "then parse the total only when returncode is zero."
        ),
        answer=value,
        files={path: base64.b64encode(text.encode()).decode()},
    )


def _unavailable_dependency(rng: random.Random) -> RecoveryCase:
    values = [rng.randint(2, 15) for _ in range(4)]
    module = "site_specific_document_parser"
    stale = f"payload = {values!r}; import {module}; {module}.extract(payload)"
    return RecoveryCase(
        kind="unavailable_dependency",
        instruction=(
            f"Run the stale operation `{stale}` once to receive its real import error. "
            "Preserve `payload`. In the next call, use import-system or package metadata "
            "introspection to verify whether the named dependency is available. If it is "
            "not available, do not retry, invent an API, or install arbitrary packages; "
            "return the JSON object "
            f'`{{"status":"unavailable","module":"{module}"}}`.'
        ),
        explicit_operation=(
            "After the ModuleNotFoundError, inspect `payload`, "
            f"`importlib.util.find_spec({module!r})`, and package mappings once. When the "
            "result confirms absence, stop tool use and report the requested structured "
            "limitation."
        ),
        answer={"status": "unavailable", "module": module},
        files={},
    )


def generate_recovery_case(
    variant: int, round_idx: int, rng: random.Random
) -> RecoveryCase:
    kind = RECOVERY_KINDS[(variant * 3 + round_idx) % len(RECOVERY_KINDS)]
    if kind == "name_error":
        return _name_error(rng)
    if kind == "missing_import":
        return _missing_import(rng)
    if kind == "missing_await":
        return _missing_await(rng)
    if kind == "bytes_text_mismatch":
        return _bytes_text_mismatch(rng, variant, round_idx)
    if kind == "completed_process":
        return _completed_process(rng)
    if kind == "path_quoting":
        return _path_quoting(rng, variant, round_idx)
    if kind == "missing_file":
        return _missing_file(rng, variant, round_idx)
    if kind == "dictionary_key":
        return _dictionary_key(rng)
    if kind == "empty_parser_output":
        return _empty_parser_output(rng, variant, round_idx)
    if kind == "subprocess_nonzero":
        return _subprocess_nonzero(rng, variant, round_idx)
    return _unavailable_dependency(rng)


__all__ = [
    "RECOVERY_KINDS",
    "RecoveryCase",
    "RecoveryKind",
    "generate_recovery_case",
]

"""Prime Agent streams for persistent IPython fundamentals."""

from __future__ import annotations

import ast
import json
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal

from pydantic import BaseModel, Field

import verifiers.v1 as vf
from ipython_foundations_v1.document_recovery import PARSER_FIXTURES
from ipython_foundations_v1.file_processing import FILE_PROCESSING_FIXTURES, FileKind
from ipython_foundations_v1.generators import (
    EVAL_VARIANTS,
    FAMILIES,
    RECOVERY_EVAL_VARIANTS,
    TRAIN_VARIANTS,
    Family,
    generate,
)
from verifiers.v1.types import content_text

WORKSPACE = "/workspace"
SYSTEM_PROMPT = (
    "Use IPython as one persistent notebook. Variables, imports, functions, and loaded "
    "objects survive across IPython calls and later user messages in this session. An "
    "empty tool result after an assignment normally means the assignment succeeded; it "
    "does not mean the cell should be repeated. Inspect or use the retained variable in "
    "a later call. After a traceback, preserve any state created before the failing "
    "statement and correct only the failed operation. End each request with the requested "
    "JSON value only, without Markdown or explanation. For subprocess results, always "
    "inspect returncode, stdout, and stderr; check=False does not make a nonzero return "
    "code successful. Use the reported error to change a failed operation instead of "
    "repeating it, and choose fallbacks from evidence rather than bypassing structured "
    "formats with raw-byte decoding."
    " When a tool returns a structured object, inspect its type and fields before "
    "passing one of its values to another API. Distinguish a distribution name from "
    "its Python import and public API; use metadata and introspection rather than "
    "inventing classes. Do not repeat an unchanged failure."
)
GUIDED_OPERATIONS = {
    "completion": (
        "Use one IPython call, observe its non-empty result, then return that result "
        "immediately without another tool call."
    ),
    "assignment": (
        "Use two separate IPython calls. First assign the requested variable and accept "
        "the empty result as success. In the next call, read that variable without "
        "reassigning it, then return the computed JSON value."
    ),
    "state": (
        "Answer the current request, retain the requested variable, and read that same "
        "variable in later requests without reloading or reconstructing the source."
    ),
    "recovery": (
        "Run the failing operation once, inspect the real traceback, then change only "
        "the failed operation while reusing state created before the error."
    ),
    "subprocess": (
        "Inspect returncode, stdout, and stderr, then revise the failed command from "
        "that evidence without repeating it or decoding raw document bytes."
    ),
    "document_recovery": (
        "Inspect whether the supplied source is a path or structured result, retain the "
        "evidenced path, run the inherited parser once, and use its live error plus "
        "package/API introspection to make one evidence-directed repair."
    ),
    "file_processing": (
        "Inspect the supplied Python value and select its evidenced path before parsing. "
        "Choose a normal parser from file type and MIME evidence, preserve successful "
        "state, and change only the failed operation after live feedback."
    ),
    "document_control": (
        "Open the document once, execute the inherited failing expression once, and "
        "repair that expression from its live traceback. Iterate every page, retain "
        "the joined extracted text, preserve negation, and return only the requested JSON."
    ),
}

PDFTOTEXT_COMPAT = r"""#!/usr/bin/env python3
import base64
import sys
from pathlib import Path

args = sys.argv[1:]
if "-text" in args:
    sys.stderr.write("I/O Error: Couldn't open file '-text': No such file or directory.\n")
    raise SystemExit(1)

positional = [arg for arg in args if arg != "-layout"]
if not positional:
    sys.stderr.write("Syntax Error: No input file specified.\n")
    raise SystemExit(1)

source = Path(positional[0])
if not source.is_file():
    sys.stderr.write(f"I/O Error: Couldn't open file '{source}': No such file or directory.\n")
    raise SystemExit(1)

text = base64.b64decode(source.read_bytes()).decode()
output = positional[1] if len(positional) > 1 else str(source.with_suffix(".txt"))
if output == "-":
    sys.stdout.write(text)
else:
    Path(output).write_text(text)
"""


class FoundationRound(BaseModel):
    instruction: str
    explicit_operation: str
    answer: Any
    files: dict[str, str]
    remove_after: tuple[str, ...] = ()
    recovery_kind: str | None = None


class IpythonFoundationsData(vf.TaskData):
    family: Family
    template_variant: int
    instruction_level: Literal["standard", "guided", "explicit"] = "standard"
    state_variable: str
    demonstration: str
    rounds: tuple[FoundationRound, ...]
    source_kind: Literal["direct_path", "structured_download"] | None = None
    file_kind: FileKind | None = None
    failure_kind: str | None = None
    expected_output_marker: str | None = None
    terminal_status: str | None = None


@dataclass
class IpythonEvent:
    code: str
    call_id: str
    segment: int
    output: str = ""


def _extract_json(reply: str) -> object | None:
    try:
        return json.loads(reply.strip())
    except json.JSONDecodeError:
        return None


def _partial_score(actual: object, expected: object) -> float:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return 0.0
        return (
            sum(actual.get(key) == value for key, value in expected.items())
            / len(expected)
            if expected
            else float(not actual)
        )
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return 0.0
        return (
            sum(
                actual[index] == value
                for index, value in enumerate(expected)
                if index < len(actual)
            )
            / len(expected)
            if expected
            else float(not actual)
        )
    return float(actual == expected)


def _ipython_events(trace: vf.Trace) -> list[IpythonEvent]:
    events: list[IpythonEvent] = []
    by_call_id: dict[str, IpythonEvent] = {}
    segment = -1
    for node in trace.nodes:
        message = node.message
        if isinstance(message, vf.UserMessage):
            segment += 1
        elif isinstance(message, vf.AssistantMessage):
            for call in message.tool_calls or []:
                if call.name != "ipython":
                    continue
                try:
                    arguments = json.loads(call.arguments)
                except json.JSONDecodeError:
                    continue
                code = arguments.get("code")
                if not isinstance(code, str):
                    continue
                event = IpythonEvent(code=code, call_id=call.id, segment=segment)
                events.append(event)
                by_call_id[call.id] = event
        elif isinstance(message, vf.ToolMessage) and (
            event := by_call_id.get(message.tool_call_id)
        ):
            event.output = content_text(message.content)
    return events


def _name_contexts(code: str, name: str) -> tuple[bool, bool]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, False
    assigned = any(
        isinstance(node, ast.Name)
        and node.id == name
        and isinstance(node.ctx, ast.Store)
        for node in ast.walk(tree)
    )
    loaded = any(
        isinstance(node, ast.Name)
        and node.id == name
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(tree)
    )
    return assigned, loaded


def _code_attributes(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    return {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}


def _uses_stdout_convention(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    strings = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    return "pdftotext" in code and "-" in strings


def _uses_raw_pdf_fallback(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    if attributes & {"read_bytes", "decode"}:
        return True
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "open":
            continue
        mode = node.args[1] if len(node.args) > 1 else None
        mode = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "mode"),
            mode,
        )
        if (
            isinstance(mode, ast.Constant)
            and isinstance(mode.value, str)
            and "b" in mode.value
        ):
            return True
    return False


def _error_signature(output: str) -> str | None:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines or not any("Error" in line for line in lines):
        return None
    return lines[-1]


def _uses_distribution_introspection(code: str) -> bool:
    return any(
        marker in code
        for marker in (
            "importlib.metadata",
            "metadata.distribution",
            "packages_distributions",
        )
    )


def _uses_api_introspection(code: str) -> bool:
    return any(marker in code for marker in ("dir(", "help(", "inspect.signature"))


def _uses_document_parser(code: str) -> bool:
    return "fitz.open" in code or "extract_text" in code


def _uses_file_acquisition_api(code: str) -> bool:
    return any(
        marker in code
        for marker in (
            "list_files",
            "download_file",
            "omnigent_list_files",
            "omnigent_download_file",
        )
    )


def _uses_parser_for_kind(code: str, file_kind: FileKind | None) -> bool:
    markers = {
        "text": ("read_text",),
        "markdown": ("read_text",),
        "csv": ("csv.", "csv import", "pandas", "read_csv"),
        "json": ("json.load",),
        "pdf": ("PdfReader", "pdftotext", "fitz.open", "extract_text"),
        "docx": ("docx", "Document("),
        "unknown": ("mimetypes", "read_bytes", "read_text"),
    }
    return bool(file_kind and any(marker in code for marker in markers[file_kind]))


def _inspects_structured_result(code: str) -> bool:
    return (
        "download" in code
        and "type(" in code
        and any(marker in code for marker in ("keys", "sorted(download)"))
    )


def _selects_download_path(code: str) -> bool:
    return (
        "download" in code
        and any(marker in code for marker in ("['path']", '["path"]'))
    )


def _is_import_only(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return bool(tree.body) and all(
        isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body
    )


def _behavior(
    trace: vf.Trace,
    family: Family,
    state_variable: str,
    expected_segments: int | None = None,
    source_kind: Literal["direct_path", "structured_download"] | None = None,
    file_kind: FileKind | None = None,
    failure_kind: str | None = None,
    expected_output_marker: str | None = None,
    terminal_status: str | None = None,
) -> dict[str, float]:
    events = _ipython_events(trace)
    contexts = [_name_contexts(event.code, state_variable) for event in events]
    attributes = [_code_attributes(event.code) for event in events]
    assignment_indices = [
        index for index, (assigned, _) in enumerate(contexts) if assigned
    ]
    first_assignment = assignment_indices[0] if assignment_indices else None
    later_reuse = next(
        (
            index
            for index, (assigned, loaded) in enumerate(contexts)
            if (
                loaded
                and not assigned
                and first_assignment is not None
                and index > first_assignment
            )
        ),
        None,
    )
    silent_assignment_recovered = bool(
        first_assignment is not None
        and not events[first_assignment].output.strip()
        and later_reuse is not None
    )
    cross_turn_reuse = bool(
        first_assignment is not None
        and later_reuse is not None
        and events[later_reuse].segment > events[first_assignment].segment
    )
    error_index = next(
        (
            index
            for index, event in enumerate(events)
            if "Traceback" in event.output or "Error" in event.output
        ),
        None,
    )
    recovered_after_error = bool(
        error_index is not None
        and any(
            loaded and index > error_index for index, (_, loaded) in enumerate(contexts)
        )
    )
    active_segments = {event.segment for event in events}
    recovery_error_indices = {
        segment: next(
            (
                index
                for index, event in enumerate(events)
                if event.segment == segment
                and ("Traceback" in event.output or "Error" in event.output)
            ),
            None,
        )
        for segment in active_segments
    }
    recovery_repaired_segments = {
        segment
        for segment, segment_error_index in recovery_error_indices.items()
        if segment_error_index is not None
        and any(
            index > segment_error_index
            and event.segment == segment
            and contexts[index][1]
            and event.code.strip() != events[segment_error_index].code.strip()
            for index, event in enumerate(events)
        )
    }
    required_recovery_segments = (
        set(range(expected_segments))
        if expected_segments is not None
        else active_segments
    )
    recovery_rounds_aligned = bool(
        required_recovery_segments
        and required_recovery_segments <= recovery_repaired_segments
    )
    repeated = sum(
        left.code.strip() == right.code.strip() for left, right in pairwise(events)
    )
    error_signatures = [
        signature
        for event in events
        if (signature := _error_signature(event.output)) is not None
    ]
    repeated_error_signatures = sum(
        count - 1 for count in Counter(error_signatures).values() if count > 1
    )
    subprocess_observed = any(
        {"returncode", "stdout", "stderr"} <= event_attributes
        for event_attributes in attributes
    )
    subprocess_failure_index = next(
        (
            index
            for index, event in enumerate(events)
            if "returncode" in event.code
            and "stderr" in event.code
            and "I/O Error" in event.output
        ),
        None,
    )
    subprocess_failures = sum(
        "pdftotext" in event.code and "I/O Error" in event.output for event in events
    )
    subprocess_failure_retries = max(subprocess_failures - 1, 0)
    subprocess_revised = bool(
        subprocess_failure_index is not None
        and any(
            index > subprocess_failure_index
            and "pdftotext" in event.code
            and event.code.strip() != events[subprocess_failure_index].code.strip()
            for index, event in enumerate(events)
        )
    )
    cli_stdout_used = any(
        _uses_stdout_convention(event.code)
        for index, event in enumerate(events)
        if subprocess_failure_index is not None and index > subprocess_failure_index
    )
    raw_pdf_fallback = any(_uses_raw_pdf_fallback(event.code) for event in events)
    successful_result_observed = any(
        event.output.strip()
        and "Traceback" not in event.output
        and "Error" not in event.output
        for event in events
    )
    source_events = [event for event in events if event.segment == 0]
    document_source_inspected = bool(
        source_kind == "structured_download"
        and any(
            "download" in event.code and "type(" in event.code
            for event in source_events
        )
        and any(
            "download" in event.code
            and ("keys" in event.code or "sorted(download)" in event.code)
            for event in source_events
        )
        or source_kind == "direct_path"
        and any(
            "document_path" in event.code and "type(" in event.code
            for event in source_events
        )
        and any(
            "document_path" in event.code
            and ("exists" in event.code or "is_file" in event.code)
            for event in source_events
        )
    )
    document_error_indices = [
        index
        for index, event in enumerate(events)
        if event.segment == 1 and _error_signature(event.output) is not None
    ]
    document_error_index = document_error_indices[0] if document_error_indices else None
    document_extra_errors = max(len(document_error_indices) - 1, 0)
    document_operation_revised = bool(
        document_error_index is not None
        and any(
            index > document_error_index
            and event.segment == 1
            and event.code.strip() != events[document_error_index].code.strip()
            for index, event in enumerate(events)
        )
    )
    distribution_inspected = any(
        index > document_error_index and _uses_distribution_introspection(event.code)
        for index, event in enumerate(events)
        if document_error_index is not None
    )
    api_surface_inspected = any(
        index > document_error_index and _uses_api_introspection(event.code)
        for index, event in enumerate(events)
        if document_error_index is not None
    )
    package_api_inspected = distribution_inspected and api_surface_inspected
    page_contexts = [_name_contexts(event.code, "page_text") for event in events]
    document_text_extracted = any(
        index > document_error_index
        and event.segment == 1
        and page_contexts[index][0]
        and _uses_document_parser(event.code)
        and "Title:" in event.output
        and _error_signature(event.output) is None
        for index, event in enumerate(events)
        if document_error_index is not None
    )
    page_assignment = next(
        (index for index, (assigned, _) in enumerate(page_contexts) if assigned), None
    )
    summary_reused_extraction = bool(
        page_assignment is not None
        and any(
            index > page_assignment
            and loaded
            and not assigned
            and events[index].segment > events[page_assignment].segment
            for index, (assigned, loaded) in enumerate(page_contexts)
        )
    )
    file_acquisition_calls = sum(
        _uses_file_acquisition_api(event.code) for event in events
    )
    structured_result_inspected = bool(
        source_kind == "structured_download"
        and any(_inspects_structured_result(event.code) for event in events)
        or source_kind == "direct_path"
        and any(
            "document_path" in event.code and "type(" in event.code for event in events
        )
    )
    selected_path_index = next(
        (
            index
            for index, event in enumerate(events)
            if (
                source_kind == "structured_download"
                and _selects_download_path(event.code)
                or source_kind == "direct_path"
                and "document_path" in event.code
                and _name_contexts(event.code, "document_path")[0]
            )
            and _error_signature(event.output) is None
        ),
        None,
    )
    parser_selected = any(
        _uses_parser_for_kind(event.code, file_kind) for event in events
    )
    path_reused_for_parser = bool(
        selected_path_index is not None
        and any(
            index >= selected_path_index
            and (
                "document_path" in event.code
                or source_kind == "structured_download"
                and _selects_download_path(event.code)
            )
            and _uses_parser_for_kind(event.code, file_kind)
            for index, event in enumerate(events)
        )
    )
    page_object_index = next(
        (
            index
            for index, event in enumerate(events)
            if "Page object" in event.output and "extract_text" not in event.code
        ),
        None,
    )
    traceback_informed_change = bool(
        error_index is not None
        and any(
            index > error_index
            and event.code.strip() != events[error_index].code.strip()
            and _error_signature(event.output)
            != _error_signature(events[error_index].output)
            for index, event in enumerate(events)
        )
        or page_object_index is not None
        and any(
            index > page_object_index
            and "extract_text" in event.code
            and expected_output_marker in event.output
            for index, event in enumerate(events)
            if expected_output_marker
        )
    )
    import_indices = [
        index
        for index, event in enumerate(events)
        if _is_import_only(event.code) and not event.output.strip()
    ]
    silent_import_progressed = all(
        index + 1 < len(events)
        and events[index + 1].code.strip() != events[index].code.strip()
        for index in import_indices
    )
    extracted_output_observed = bool(
        expected_output_marker
        and any(
            expected_output_marker in event.output
            and _uses_parser_for_kind(event.code, file_kind)
            and _error_signature(event.output) is None
            for event in events
        )
    )
    terminal_failure_observed = bool(
        terminal_status == "malformed_csv"
        and any("unexpected end of data" in event.output for event in events)
        or terminal_status == "invalid_json"
        and any("JSONDecodeError" in event.output for event in events)
        or terminal_status == "password_protected"
        and any("FileNotDecryptedError" in event.output for event in events)
        or terminal_status == "no_extractable_text"
        and any(
            "extract_text" in event.code and event.output.strip() in {"''", '""'}
            for event in events
        )
    )
    processing_outcome_observed = extracted_output_observed or terminal_failure_observed
    feedback_handled = traceback_informed_change or terminal_failure_observed
    repair_outcome_observed = bool(
        error_index is not None
        and expected_output_marker
        and any(
            index > error_index
            and event.code.strip() != events[error_index].code.strip()
            and _error_signature(event.output) is None
            and expected_output_marker in event.output
            for index, event in enumerate(events)
        )
    )
    full_document_text_extracted = bool(
        expected_output_marker
        and any(
            "reader.pages" in event.code
            and "extract_text" in event.code
            and "full_text" in event.code
            and expected_output_marker in event.output
            and _error_signature(event.output) is None
            for event in events
        )
    )
    expected_file_errors = int(
        failure_kind not in {None, "page_object_not_text", "scanned_pdf"}
    )
    file_processing_extra_errors = max(len(error_signatures) - expected_file_errors, 0)
    observed_segments = max(len(active_segments), 1)
    expected_rounds = expected_segments or observed_segments
    efficient_call_budget = {
        "completion": expected_rounds,
        "assignment": 2 * expected_rounds,
        "state": expected_rounds,
        "recovery": 3 * expected_rounds,
        "subprocess": expected_rounds,
        "document_recovery": 7,
        "file_processing": 8,
        "document_control": 7,
    }[family]
    call_efficiency = min(efficient_call_budget / max(len(events), 1), 1.0)
    recovery_round_coverage = (
        len(recovery_repaired_segments) / len(required_recovery_segments)
        if required_recovery_segments
        else 0.0
    )
    subprocess_progress = (
        sum(
            (
                later_reuse is not None,
                subprocess_observed,
                subprocess_revised,
                cli_stdout_used,
            )
        )
        / 4
    )
    if raw_pdf_fallback:
        subprocess_progress *= 0.5
    if subprocess_failure_retries:
        subprocess_progress /= subprocess_failure_retries + 1
    document_progress = (
        sum(
            (
                document_source_inspected,
                later_reuse is not None,
                document_error_index is not None,
                document_operation_revised,
                distribution_inspected,
                api_surface_inspected,
                document_text_extracted,
                summary_reused_extraction,
            )
        )
        / 8
    )
    if raw_pdf_fallback or file_acquisition_calls:
        document_progress *= 0.5
    if document_extra_errors or repeated_error_signatures:
        document_progress /= 1 + document_extra_errors + repeated_error_signatures
    file_processing_progress = (
        sum(
            (
                structured_result_inspected,
                selected_path_index is not None,
                path_reused_for_parser,
                parser_selected,
                feedback_handled,
                silent_import_progressed,
                processing_outcome_observed,
            )
        )
        / 7
    )
    if file_acquisition_calls or file_processing_extra_errors:
        file_processing_progress /= (
            1 + file_acquisition_calls + file_processing_extra_errors
        )
    document_control_progress = (
        sum(
            (
                structured_result_inspected,
                selected_path_index is not None,
                path_reused_for_parser,
                parser_selected,
                traceback_informed_change,
                repair_outcome_observed,
                full_document_text_extracted,
                silent_import_progressed,
            )
        )
        / 8
    )
    if file_acquisition_calls or file_processing_extra_errors:
        document_control_progress /= (
            1 + file_acquisition_calls + file_processing_extra_errors
        )
    family_progress = {
        "completion": float(successful_result_observed),
        "assignment": float(silent_assignment_recovered),
        "state": float(cross_turn_reuse),
        "recovery": recovery_round_coverage,
        "subprocess": subprocess_progress,
        "document_recovery": document_progress,
        "file_processing": file_processing_progress,
        "document_control": document_control_progress,
    }[family]
    process_score = family_progress * call_efficiency / (repeated + 1)
    family_aligned = {
        "completion": successful_result_observed and len(events) == expected_rounds,
        "assignment": silent_assignment_recovered,
        "state": cross_turn_reuse,
        "recovery": recovery_rounds_aligned,
        "subprocess": (
            later_reuse is not None
            and subprocess_observed
            and subprocess_revised
            and cli_stdout_used
            and not raw_pdf_fallback
            and subprocess_failure_retries == 0
        ),
        "document_recovery": (
            document_source_inspected
            and later_reuse is not None
            and document_error_index is not None
            and document_operation_revised
            and distribution_inspected
            and api_surface_inspected
            and document_text_extracted
            and summary_reused_extraction
            and document_extra_errors == 0
            and repeated_error_signatures == 0
            and file_acquisition_calls == 0
            and not raw_pdf_fallback
        ),
        "file_processing": (
            structured_result_inspected
            and selected_path_index is not None
            and path_reused_for_parser
            and parser_selected
            and feedback_handled
            and silent_import_progressed
            and processing_outcome_observed
            and file_processing_extra_errors == 0
            and repeated_error_signatures == 0
            and file_acquisition_calls == 0
        ),
        "document_control": (
            structured_result_inspected
            and selected_path_index is not None
            and path_reused_for_parser
            and parser_selected
            and traceback_informed_change
            and repair_outcome_observed
            and full_document_text_extracted
            and silent_import_progressed
            and file_processing_extra_errors == 0
            and repeated_error_signatures == 0
            and file_acquisition_calls == 0
        ),
    }[family]
    return {
        "ipython_calls": float(len(events)),
        "state_assigned": float(bool(assignment_indices)),
        "state_reused": float(later_reuse is not None),
        "silent_assignment_recovered": float(silent_assignment_recovered),
        "cross_turn_state_reused": float(cross_turn_reuse),
        "error_observed": float(error_index is not None),
        "recovered_after_error": float(recovered_after_error),
        "recovery_error_segments": float(
            sum(index is not None for index in recovery_error_indices.values())
        ),
        "recovery_repaired_segments": float(len(recovery_repaired_segments)),
        "recovery_round_coverage": recovery_round_coverage,
        "subprocess_result_observed": float(subprocess_observed),
        "subprocess_operation_revised": float(subprocess_revised),
        "subprocess_failure_retries": float(subprocess_failure_retries),
        "cli_stdout_convention_used": float(cli_stdout_used),
        "raw_pdf_fallback_used": float(raw_pdf_fallback),
        "document_source_inspected": float(document_source_inspected),
        "document_operation_revised": float(document_operation_revised),
        "distribution_inspected": float(distribution_inspected),
        "api_surface_inspected": float(api_surface_inspected),
        "package_api_inspected": float(package_api_inspected),
        "document_text_extracted": float(document_text_extracted),
        "summary_reused_extraction": float(summary_reused_extraction),
        "document_extra_errors": float(document_extra_errors),
        "file_acquisition_calls": float(file_acquisition_calls),
        "structured_result_inspected": float(structured_result_inspected),
        "download_path_selected": float(selected_path_index is not None),
        "path_reused_for_parser": float(path_reused_for_parser),
        "file_parser_selected": float(parser_selected),
        "traceback_informed_change": float(traceback_informed_change),
        "file_feedback_handled": float(feedback_handled),
        "silent_import_progressed": float(silent_import_progressed),
        "extracted_output_observed": float(extracted_output_observed),
        "terminal_failure_observed": float(terminal_failure_observed),
        "processing_outcome_observed": float(processing_outcome_observed),
        "repair_outcome_observed": float(repair_outcome_observed),
        "full_document_text_extracted": float(full_document_text_extracted),
        "file_processing_extra_errors": float(file_processing_extra_errors),
        "repeated_error_signatures": float(repeated_error_signatures),
        "identical_consecutive_calls": float(repeated),
        "successful_result_observed": float(successful_result_observed),
        "ipython_call_efficiency": call_efficiency,
        "process_score": process_score,
        "process_aligned": float(
            family_aligned and repeated == 0 and len(events) <= efficient_call_budget
        ),
    }


class IpythonFoundationsTask(vf.Task[IpythonFoundationsData]):
    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        result = await runtime.run(
            ["mkdir", "-p", f"{WORKSPACE}/inbox", f"{WORKSPACE}/bin"], {}
        )
        if result.exit_code != 0:
            raise RuntimeError(f"workspace setup failed: {result.stderr[-500:]}")
        await runtime.write(f"{WORKSPACE}/bin/pdftotext", PDFTOTEXT_COMPAT.encode())
        fixtures = PARSER_FIXTURES | FILE_PROCESSING_FIXTURES
        fixture_directories = sorted(
            {
                str((WORKSPACE + "/" + path).rsplit("/", 1)[0])
                for path in fixtures
                if "/" in path
            }
        )
        fixture_setup = await runtime.run(["mkdir", "-p", *fixture_directories], {})
        if fixture_setup.exit_code != 0:
            raise RuntimeError(
                f"parser fixture setup failed: {fixture_setup.stderr[-500:]}"
            )
        for relative_path, content in fixtures.items():
            await runtime.write(f"{WORKSPACE}/{relative_path}", content.encode())
        executable = await runtime.run(
            ["chmod", "755", f"{WORKSPACE}/bin/pdftotext"], {}
        )
        if executable.exit_code != 0:
            raise RuntimeError(f"extractor setup failed: {executable.stderr[-500:]}")

    @vf.reward(weight=1.0)
    async def notebook_semantics(self, trace: vf.Trace) -> float:
        behavior = _behavior(
            trace,
            self.data.family,
            self.data.state_variable,
            expected_segments=len(self.data.rounds),
            source_kind=self.data.source_kind,
            file_kind=self.data.file_kind,
            failure_kind=self.data.failure_kind,
            expected_output_marker=self.data.expected_output_marker,
            terminal_status=self.data.terminal_status,
        )
        return behavior["process_score"]

    @vf.metric
    async def notebook_behavior(self, trace: vf.Trace) -> dict[str, float]:
        return _behavior(
            trace,
            self.data.family,
            self.data.state_variable,
            expected_segments=len(self.data.rounds),
            source_kind=self.data.source_kind,
            file_kind=self.data.file_kind,
            failure_kind=self.data.failure_kind,
            expected_output_marker=self.data.expected_output_marker,
            terminal_status=self.data.terminal_status,
        )


def _round_prompt(
    task: IpythonFoundationsTask,
    round_idx: int,
    previous_correct: bool | None,
) -> str:
    current = task.data.rounds[round_idx]
    parts = []
    if previous_correct is None:
        parts.append(
            "This is one continuing notebook session. Complete the current request, "
            "then retain useful IPython state for later requests."
        )
    else:
        verdict = "passed" if previous_correct else "failed"
        parts.append(
            f"The previous answer {verdict} validation. The expected value is not "
            "revealed; continue from the existing notebook state."
        )
    parts.append(current.instruction)
    if task.data.instruction_level == "guided":
        parts.append(f"Foundation hint: {GUIDED_OPERATIONS[task.data.family]}")
    elif task.data.instruction_level == "explicit":
        parts.append(f"Foundation exercise: {current.explicit_operation}")
    return "\n\n".join(parts)


class IpythonFoundationsEnv(vf.SingleAgentEnv):
    """Drive dependent requests through one Prime Agent session and IPython kernel."""

    async def run(self, task, agents):
        scores: list[float] = []
        replies: list[object | None] = []
        async with agents.agent.provision(task) as runtime:
            async with agents.agent.interaction(task, runtime=runtime) as interaction:
                previous: bool | None = None
                for round_idx, current in enumerate(task.data.rounds):
                    for path, content in current.files.items():
                        await runtime.write(path, content.encode())
                    segment = await interaction.turn(
                        _round_prompt(task, round_idx, previous)
                    )
                    if segment.terminated:
                        break
                    actual = _extract_json(segment.last_reply)
                    score = _partial_score(actual, current.answer)
                    scores.append(score)
                    replies.append(actual)
                    previous = score == 1.0
                    if current.remove_after:
                        removed = await runtime.run(
                            ["rm", "-f", *current.remove_after], {}
                        )
                        if removed.exit_code != 0:
                            raise RuntimeError(
                                f"source cleanup failed: {removed.stderr[-500:]}"
                            )
                interaction.trace.info["ipython_foundations"] = {
                    "scores": scores,
                    "replies": replies,
                    "rounds_completed": len(scores),
                    "recovery_kinds": [
                        round_.recovery_kind
                        for round_ in task.data.rounds
                        if round_.recovery_kind is not None
                    ],
                }
            trace = interaction.trace

        total_rounds = len(task.data.rounds)
        padded = [*scores, *([0.0] * (total_rounds - len(scores)))]
        trace.record_reward("stream_accuracy", sum(padded) / total_rounds, weight=0.5)
        trace.record_metric("first_request_correct", float(padded[0] == 1.0))
        trace.record_metric("final_request_correct", float(padded[-1] == 1.0))
        trace.record_metric("completed_stream", float(len(scores) == total_rounds))
        trace.record_metric(
            "json_contract_followed",
            float(
                len(replies) == total_rounds
                and all(reply is not None for reply in replies)
            ),
        )
        if task.data.family in {"file_processing", "document_control"}:
            behavior = _behavior(
                trace,
                task.data.family,
                task.data.state_variable,
                expected_segments=len(task.data.rounds),
                source_kind=task.data.source_kind,
                file_kind=task.data.file_kind,
                failure_kind=task.data.failure_kind,
                expected_output_marker=task.data.expected_output_marker,
                terminal_status=task.data.terminal_status,
            )
            trace.record_metric(
                "grounded_file_answer",
                float(padded[-1] == 1.0 and behavior["processing_outcome_observed"]),
            )
        if task.data.family == "document_control":
            trace.record_metric(
                "source_grounded_claim",
                float(
                    padded[-1] == 1.0
                    and behavior["repair_outcome_observed"]
                    and behavior["full_document_text_extracted"]
                ),
            )


class IpythonFoundationsConfig(vf.TasksetConfig):
    split: Literal["train", "eval"] = "train"
    families: tuple[Family, ...] = Field(FAMILIES, min_length=1)
    instruction_level: Literal["standard", "guided", "explicit"] = "standard"
    instances_per_template: int = Field(4, ge=1)
    rounds_per_task: int | None = Field(None, ge=1)
    seed: int = 20260806


class IpythonFoundationsTaskset(
    vf.Taskset[IpythonFoundationsTask, IpythonFoundationsConfig]
):
    def load(self) -> list[IpythonFoundationsTask]:
        variants = TRAIN_VARIANTS if self.config.split == "train" else EVAL_VARIANTS
        templates = [
            (family, variant) for variant in variants for family in self.config.families
        ]
        if self.config.split == "eval" and "recovery" in self.config.families:
            templates.extend(
                ("recovery", variant)
                for variant in RECOVERY_EVAL_VARIANTS
                if variant not in variants
            )
        tasks = []
        idx = 0
        for instance in range(self.config.instances_per_template):
            for family, variant in templates:
                generated = generate(family, variant, instance, self.config.seed)
                rounds = generated.rounds[: self.config.rounds_per_task]
                demonstration = "\n\n".join(
                    f"Request {round_idx + 1}:\n"
                    f"{round_.expert_trace or round_.explicit_operation}"
                    for round_idx, round_ in enumerate(rounds)
                )
                tasks.append(
                    IpythonFoundationsTask(
                        IpythonFoundationsData(
                            idx=idx,
                            name=f"{family}-v{variant}-i{instance}",
                            prompt=None,
                            system_prompt=SYSTEM_PROMPT,
                            workdir=WORKSPACE,
                            family=family,
                            template_variant=variant,
                            instruction_level=self.config.instruction_level,
                            state_variable=generated.state_variable,
                            demonstration=demonstration,
                            rounds=tuple(
                                FoundationRound(
                                    instruction=round_.instruction,
                                    explicit_operation=round_.explicit_operation,
                                    answer=round_.answer,
                                    files=round_.files,
                                    remove_after=round_.remove_after,
                                    recovery_kind=round_.recovery_kind,
                                )
                                for round_ in rounds
                            ),
                            source_kind=generated.source_kind,
                            file_kind=generated.file_kind,
                            failure_kind=generated.failure_kind,
                            expected_output_marker=generated.expected_output_marker,
                            terminal_status=generated.terminal_status,
                        ),
                        self.config.task,
                    )
                )
                idx += 1
        return tasks


__all__ = [
    "FoundationRound",
    "IpythonFoundationsConfig",
    "IpythonFoundationsData",
    "IpythonFoundationsEnv",
    "IpythonFoundationsTask",
    "IpythonFoundationsTaskset",
]

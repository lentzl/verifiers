"""Typed file-processing scenarios for evidence-directed IPython control."""

from __future__ import annotations

import base64
import json
import random
from dataclasses import dataclass
from typing import Literal

FileKind = Literal["text", "markdown", "csv", "json", "pdf", "docx", "unknown"]


@dataclass(frozen=True)
class ExpertCall:
    code: str
    output: str


@dataclass(frozen=True)
class FileProcessingScenario:
    instruction: str
    explicit_operation: str
    answer: object
    files: dict[str, str]
    expert_calls: tuple[ExpertCall, ...]
    source_kind: Literal["direct_path", "structured_download"]
    file_kind: FileKind
    failure_kind: str
    expected_output_marker: str | None
    terminal_status: str | None = None


FILE_PROCESSING_FIXTURES = {
    "PyPDF2/__init__.py": """import base64
from pathlib import Path

from .errors import FileNotDecryptedError


class Page:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class PdfReader:
    def __init__(self, stream):
        if not isinstance(stream, (str, Path)):
            raise TypeError("stream must be a path-like object")
        text = base64.b64decode(Path(stream).read_bytes()).decode()
        if text.startswith("PASSWORD_PROTECTED"):
            raise FileNotDecryptedError("File has not been decrypted")
        if text.startswith("SCANNED_IMAGE_ONLY"):
            text = ""
        self.pages = [Page(page) for page in text.split("\\f")]
""",
    "PyPDF2/errors.py": """class FileNotDecryptedError(Exception):
    pass
""",
    "PyPDF2-3.0.1.dist-info/METADATA": (
        "Metadata-Version: 2.1\nName: PyPDF2\nVersion: 3.0.1\n"
    ),
    "PyPDF2-3.0.1.dist-info/top_level.txt": "PyPDF2\n",
    "docx/__init__.py": """import base64
from pathlib import Path


class Paragraph:
    def __init__(self, text):
        self.text = text


class _Document:
    def __init__(self, text):
        self.paragraphs = [Paragraph(line) for line in text.splitlines()]


def Document(source):
    if not isinstance(source, (str, Path)):
        raise TypeError("source must be a path-like object")
    text = base64.b64decode(Path(source).read_bytes()).decode()
    return _Document(text)
""",
    "python_docx-1.2.0.dist-info/METADATA": (
        "Metadata-Version: 2.1\nName: python-docx\nVersion: 1.2.0\n"
    ),
    "python_docx-1.2.0.dist-info/top_level.txt": "docx\n",
}


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _traceback(error: str, message: str) -> str:
    return (
        "Traceback (most recent call last):\n"
        '  File "<ipython-input>", line 1, in <module>\n'
        f"{error}: {message}"
    )


def _download(path: str, payload: str, content_type: str) -> dict[str, object]:
    return {
        "path": path,
        "filename": path.rsplit("/", 1)[-1],
        "bytes": len(payload.encode()),
        "content_type": content_type,
    }


def _source_prompt(download: dict[str, object]) -> str:
    return (
        "The completed file download returned this object: "
        f"`{_json(download)}`. Retain it as `download`; do not list or download the "
        "file again. Work from the live Python value and preserve successful state "
        "while repairing any requested stale operation."
    )


def _source_calls(download: dict[str, object], path: str) -> tuple[ExpertCall, ...]:
    return (
        ExpertCall(
            f"download = {download!r}\ntype(download), sorted(download)",
            "(<class 'dict'>, ['bytes', 'content_type', 'filename', 'path'])",
        ),
        ExpertCall(
            "document_path = download['path']\ndocument_path",
            repr(path),
        ),
    )


def _pdf_payload(title: str, finding: str) -> tuple[str, str]:
    first_page = f"Title: {title}\nFinding: {finding}\n"
    document = first_page + "\fAppendix: supporting details.\n"
    return base64.b64encode(document.encode()).decode(), first_page


def _pdf_answer(title: str, finding: str) -> dict[str, str]:
    return {"title": title, "finding": finding}


def _pdf_finish_calls(first_page: str) -> tuple[ExpertCall, ...]:
    return (
        ExpertCall("from PyPDF2 import PdfReader", ""),
        ExpertCall(
            "reader = PdfReader(document_path)\nlen(reader.pages)",
            "2",
        ),
        ExpertCall(
            "first_page = reader.pages[0]\nfirst_page",
            "<PyPDF2.Page object>",
        ),
        ExpertCall(
            "first_page_text = first_page.extract_text()\nfirst_page_text",
            repr(first_page),
        ),
    )


def _pdf_scenario(
    variant: int,
    instance: int,
    failure_kind: str,
    stale_code: str,
    stale_output: str,
    repair_calls: tuple[ExpertCall, ...] = (),
) -> FileProcessingScenario:
    title = f"Operations Note {variant}-{instance}"
    finding = f"Batch {variant + instance + 3} passed the controlled review."
    payload, first_page = _pdf_payload(title, finding)
    path = f"/workspace/inbox/operations-{variant}-{instance}.pdf"
    download = _download(path, payload, "application/pdf")
    calls = (*_source_calls(download, path), ExpertCall(stale_code, stale_output))
    calls += repair_calls or _pdf_finish_calls(first_page)
    instruction = (
        f"{_source_prompt(download)} Run this inherited operation once so its real "
        f"feedback is visible: `{stale_code}`. Inspect the supplied object before "
        "selecting its path, translate the feedback into one constrained correction, "
        "and use an installed PDF parser. Obtaining a page object is not extraction: "
        "retain `first_page_text = first_page.extract_text()` and answer only from that "
        "nonempty text with JSON keys `title` and `finding`. Do not repeat successful "
        "imports or unchanged cells."
    )
    return FileProcessingScenario(
        instruction=instruction,
        explicit_operation=(
            "Inspect `type(download)` and its keys, select `download['path']`, run the "
            "stale operation once, then alter only the failed step. Retain the reader, "
            "page, and extracted text in separate variables before answering."
        ),
        answer=_pdf_answer(title, finding),
        files={path: payload},
        expert_calls=calls,
        source_kind="structured_download",
        file_kind="pdf",
        failure_kind=failure_kind,
        expected_output_marker=title,
    )


def _build_pdf_scenario(
    index: int, variant: int, instance: int
) -> FileProcessingScenario:
    if index == 0:
        stale = "document_path = download['content']"
        output = _traceback("KeyError", "'content'")
        repair = (
            ExpertCall(
                "type(download), sorted(download)",
                "(<class 'dict'>, ['bytes', 'content_type', 'filename', 'path'])",
            ),
            ExpertCall(
                "document_path = download['path']\ndocument_path", "path retained"
            ),
        )
        scenario = _pdf_scenario(
            variant, instance, "missing_download_key", stale, output, repair
        )
        payload = next(iter(scenario.files.values()))
        first_page = base64.b64decode(payload).decode().split("\f", 1)[0]
        return FileProcessingScenario(
            **{
                **scenario.__dict__,
                "expert_calls": (
                    ExpertCall(
                        f"download = {_download(next(iter(scenario.files)), payload, 'application/pdf')!r}\n{stale}",
                        output,
                    ),
                    *repair,
                    *_pdf_finish_calls(first_page),
                ),
            }
        )
    if index == 1:
        stale = "import json\nparsed = json.loads(download)"
        return _pdf_scenario(
            variant,
            instance,
            "structured_result_is_dict",
            stale,
            _traceback(
                "TypeError",
                "the JSON object must be str, bytes or bytearray, not dict",
            ),
        )
    if index == 2:
        stale = "from PyPDF2 import PdfReader\nreader = PdfReader()"
        _, first_page = _pdf_payload(
            f"Operations Note {variant}-{instance}",
            f"Batch {variant + instance + 3} passed the controlled review.",
        )
        return _pdf_scenario(
            variant,
            instance,
            "missing_pdf_stream",
            stale,
            _traceback(
                "TypeError",
                "PdfReader.__init__() missing 1 required positional argument: 'stream'",
            ),
            _pdf_finish_calls(first_page)[1:],
        )
    if index == 3:
        stale = "reader = PyPDF2.PdfReader(document_path)"
        _, first_page = _pdf_payload(
            f"Operations Note {variant}-{instance}",
            f"Batch {variant + instance + 3} passed the controlled review.",
        )
        repair = (
            ExpertCall("import PyPDF2", ""),
            ExpertCall(
                "reader = PyPDF2.PdfReader(document_path)\nlen(reader.pages)",
                "2",
            ),
            ExpertCall(
                "first_page = reader.pages[0]\nfirst_page",
                "<PyPDF2.Page object>",
            ),
            ExpertCall(
                "first_page_text = first_page.extract_text()\nfirst_page_text",
                repr(first_page),
            ),
        )
        return _pdf_scenario(
            variant,
            instance,
            "missing_pdf_import",
            stale,
            _traceback("NameError", "name 'PyPDF2' is not defined"),
            repair,
        )
    if index == 4:
        stale = (
            "from PyPDF2 import PdfReader\nreader = PdfReader(document_path)\n"
            "first_page_text = reader.pages[0]\nfirst_page_text"
        )
        _, first_page = _pdf_payload(
            f"Operations Note {variant}-{instance}",
            f"Batch {variant + instance + 3} passed the controlled review.",
        )
        return _pdf_scenario(
            variant,
            instance,
            "page_object_not_text",
            stale,
            "<PyPDF2.Page object>",
            (
                ExpertCall(
                    "first_page = reader.pages[0]\nfirst_page_text = first_page.extract_text()\nfirst_page_text",
                    repr(first_page),
                ),
            ),
        )
    if index == 5:
        stale = "from pdfminer import PDFPage"
        _, first_page = _pdf_payload(
            f"Operations Note {variant}-{instance}",
            f"Batch {variant + instance + 3} passed the controlled review.",
        )
        return _pdf_scenario(
            variant,
            instance,
            "pdfminer_public_api",
            stale,
            _traceback(
                "ImportError",
                "cannot import name 'PDFPage' from 'pdfminer'",
            ),
            (
                ExpertCall(
                    "import pdfminer.high_level, inspect\ninspect.signature(pdfminer.high_level.extract_text)",
                    "(source, page_numbers=None)",
                ),
                ExpertCall(
                    "first_page_text = pdfminer.high_level.extract_text(document_path, page_numbers=[0])\nfirst_page_text",
                    repr(first_page),
                ),
            ),
        )
    stale = (
        "import subprocess\nsubprocess.run(['pdftotext', document_path, '-'], "
        "capture_output=True, text=True, check=True)"
    )
    return _pdf_scenario(
        variant,
        instance,
        "pdftotext_missing",
        stale,
        _traceback(
            "FileNotFoundError", "[Errno 2] No such file or directory: 'pdftotext'"
        ),
    )


def _text_scenario(
    variant: int, instance: int, *, markdown: bool
) -> FileProcessingScenario:
    suffix = "md" if markdown else "txt"
    file_kind: FileKind = "markdown" if markdown else "text"
    title = f"Field Memo {variant}-{instance}"
    finding = f"Sensor group {variant + instance + 2} remained stable."
    text = (
        f"# {title}\n\n{finding}\n"
        if markdown
        else f"Title: {title}\nFinding: {finding}\n"
    )
    path = f"/workspace/inbox/memo-{variant}-{instance}.{suffix}"
    if markdown:
        stale = "import json\nparsed = json.loads(Path(document_path).read_text())"
        output = _traceback(
            "JSONDecodeError", "Expecting value: line 1 column 1 (char 0)"
        )
    else:
        stale = "extracted_text = Path(document_path).read_text(encoding='ascii')"
        output = _traceback(
            "UnicodeDecodeError",
            "'ascii' codec can't decode byte 0xc3 in position 0: ordinal not in range(128)",
        )
        text = "Preamble: café\n" + text
    download = _download(path, text, "text/markdown" if markdown else "text/plain")
    calls = (
        *_source_calls(download, path),
        ExpertCall("from pathlib import Path", ""),
        ExpertCall(stale, output),
        ExpertCall(
            "extracted_text = Path(document_path).read_text(encoding='utf-8')\nextracted_text",
            repr(text),
        ),
    )
    return FileProcessingScenario(
        instruction=(
            f"{_source_prompt(download)} Run `{stale}` once, use its live error to "
            f"choose the normal `{suffix}` processing method, retain nonempty "
            "`extracted_text`, and return JSON with `title` and `finding` grounded only "
            "in that text. A silent successful import is complete; proceed instead of "
            "importing it again."
        ),
        explicit_operation=(
            "Inspect the download mapping, retain its path, run the stale operation once, "
            "then use `Path.read_text()` with evidence-supported encoding and answer from "
            "the retained text."
        ),
        answer={"title": title, "finding": finding},
        files={path: text},
        expert_calls=calls,
        source_kind="structured_download",
        file_kind=file_kind,
        failure_kind="wrong_parser" if markdown else "wrong_encoding",
        expected_output_marker=title,
    )


def _csv_scenario(
    variant: int, instance: int, *, malformed: bool
) -> FileProcessingScenario:
    path = f"/workspace/inbox/counts-{variant}-{instance}.csv"
    content = (
        'site,count\nalpha,4\n"beta,7\n'
        if malformed
        else "site,count\nalpha,4\nbeta,7\n"
    )
    download = _download(path, content, "text/csv")
    stale = (
        "import csv\nwith open(document_path, newline='') as handle:\n"
        "    rows = list(csv.DictReader(handle, strict=True))\nrows"
    )
    if malformed:
        output = _traceback("Error", "unexpected end of data")
        answer: object = {"status": "malformed_csv"}
        calls = (*_source_calls(download, path), ExpertCall(stale, output))
        marker = None
        terminal = "malformed_csv"
    else:
        wrong = "import json\nrows = json.load(open(document_path))"
        output = _traceback(
            "JSONDecodeError", "Expecting value: line 1 column 1 (char 0)"
        )
        answer = {"rows": 2, "total": 11}
        calls = (
            *_source_calls(download, path),
            ExpertCall(wrong, output),
            ExpertCall(
                stale,
                "[{'site': 'alpha', 'count': '4'}, {'site': 'beta', 'count': '7'}]",
            ),
            ExpertCall("sum(int(row['count']) for row in rows)", "11"),
        )
        stale = wrong
        marker = "alpha"
        terminal = None
    return FileProcessingScenario(
        instruction=(
            f"{_source_prompt(download)} Run `{stale}` once. Use the resulting feedback "
            "to process the CSV with `csv` or pandas. If parsing succeeds, retain the "
            "rows and return their count and total; if strict parsing proves the file is "
            "malformed, return `{'status':'malformed_csv'}` without repeating or inventing "
            "data."
        ),
        explicit_operation=(
            "Inspect the mapping and path, use a CSV parser, and let its live result decide "
            "between grounded aggregation and the malformed terminal status."
        ),
        answer=answer,
        files={path: content},
        expert_calls=calls,
        source_kind="structured_download",
        file_kind="csv",
        failure_kind="malformed_csv" if malformed else "wrong_parser",
        expected_output_marker=marker,
        terminal_status=terminal,
    )


def _json_scenario(
    variant: int, instance: int, *, invalid: bool
) -> FileProcessingScenario:
    path = f"/workspace/inbox/records-{variant}-{instance}.json"
    records = [{"amount": 4}, {"amount": 7}]
    content = '[{"amount":4},{"amount":7},]' if invalid else json.dumps(records)
    download = _download(path, content, "application/json")
    if invalid:
        stale = "import json\nrecords = json.load(open(document_path))"
        output = _traceback(
            "JSONDecodeError", "Expecting value: line 1 column 29 (char 28)"
        )
        calls = (*_source_calls(download, path), ExpertCall(stale, output))
        answer: object = {"status": "invalid_json"}
        marker = None
        terminal = "invalid_json"
    else:
        stale = "import json\nrecords = json.loads(download)"
        output = _traceback(
            "TypeError", "the JSON object must be str, bytes or bytearray, not dict"
        )
        calls = (
            ExpertCall(f"download = {download!r}\n{stale}", output),
            ExpertCall(
                "type(download), sorted(download)",
                "(<class 'dict'>, ['bytes', 'content_type', 'filename', 'path'])",
            ),
            ExpertCall("document_path = download['path']", ""),
            ExpertCall(
                "records = json.load(open(document_path))\nrecords", repr(records)
            ),
        )
        answer = {"records": 2, "total": 11}
        marker = "amount"
        terminal = None
    return FileProcessingScenario(
        instruction=(
            f"{_source_prompt(download)} Run `{stale}` once. Distinguish the already "
            "structured download mapping from JSON file contents. Use `json.load` on the "
            "retained path when valid; if the parser proves the file is invalid, return "
            "`{'status':'invalid_json'}` without silently rewriting the source."
        ),
        explicit_operation=(
            "Inspect the mapping after the stale failure, select its path, and use the JSON "
            "parser exactly once on the file. Preserve parsed records when successful."
        ),
        answer=answer,
        files={path: content},
        expert_calls=calls,
        source_kind="structured_download",
        file_kind="json",
        failure_kind="invalid_json" if invalid else "structured_result_is_dict",
        expected_output_marker=marker,
        terminal_status=terminal,
    )


def _docx_scenario(variant: int, instance: int) -> FileProcessingScenario:
    title = f"Review Brief {variant}-{instance}"
    finding = f"Team {variant + instance + 1} completed the handoff."
    text = f"Title: {title}\nFinding: {finding}\n"
    payload = base64.b64encode(text.encode()).decode()
    path = f"/workspace/inbox/review-{variant}-{instance}.docx"
    download = _download(
        path,
        payload,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    stale = "from docx import Document\ndocument = Document(download)"
    calls = (
        *_source_calls(download, path),
        ExpertCall(stale, _traceback("TypeError", "source must be a path-like object")),
        ExpertCall(
            "document = Document(document_path)\nextracted_text = '\\n'.join(p.text for p in document.paragraphs)\nextracted_text",
            repr(text.rstrip()),
        ),
    )
    return FileProcessingScenario(
        instruction=(
            f"{_source_prompt(download)} Run `{stale}` once. Repair the path/type boundary "
            "without reacquiring the file, use `python-docx`, retain paragraph text, and "
            "answer with JSON keys `title` and `finding`."
        ),
        explicit_operation=(
            "Inspect the download object, pass its retained path to `docx.Document`, join "
            "paragraph text, and answer only from the nonempty result."
        ),
        answer={"title": title, "finding": finding},
        files={path: payload},
        expert_calls=calls,
        source_kind="structured_download",
        file_kind="docx",
        failure_kind="structured_result_is_dict",
        expected_output_marker=title,
    )


def _unknown_scenario(variant: int, instance: int) -> FileProcessingScenario:
    text = f"Unknown note {variant}-{instance}"
    payload = f"RLM1\n{text}\n"
    path = f"/workspace/inbox/payload-{variant}-{instance}.dat"
    stale = "import json\nparsed = json.load(open(document_path))"
    calls = (
        ExpertCall(
            f"document_path = {path!r}\ntype(document_path)",
            "<class 'str'>",
        ),
        ExpertCall(
            stale,
            _traceback("JSONDecodeError", "Expecting value: line 1 column 1 (char 0)"),
        ),
        ExpertCall(
            "import mimetypes\nfrom pathlib import Path\nmime = mimetypes.guess_type(document_path)[0]\nmagic = Path(document_path).read_bytes()[:8]\n(mime, magic)",
            "(None, b'RLM1\\nUnk')",
        ),
        ExpertCall(
            "extracted_text = Path(document_path).read_text().splitlines()[1]\nextracted_text",
            repr(text),
        ),
    )
    return FileProcessingScenario(
        instruction=(
            f"The file is already available at `{path}` with an unknown extension. Run "
            f"`{stale}` once, then inspect MIME inference and leading magic bytes before "
            "choosing a fallback. Retain the extracted text and return JSON with keys "
            "`format` and `text`. Raw-byte inspection is appropriate here because this is "
            "an unknown format, not a PDF bypass."
        ),
        explicit_operation=(
            "Retain the direct path, inspect `mimetypes.guess_type` and the first eight "
            "bytes after the stale parser fails, then decode only the evidenced RLM1 text."
        ),
        answer={"format": "rlm-note", "text": text},
        files={path: payload},
        expert_calls=calls,
        source_kind="direct_path",
        file_kind="unknown",
        failure_kind="unknown_format",
        expected_output_marker=text,
    )


def _terminal_pdf_scenario(
    variant: int, instance: int, *, password_protected: bool
) -> FileProcessingScenario:
    marker = "PASSWORD_PROTECTED" if password_protected else "SCANNED_IMAGE_ONLY"
    payload = base64.b64encode(marker.encode()).decode()
    path = f"/workspace/inbox/scan-{variant}-{instance}.pdf"
    download = _download(path, payload, "application/pdf")
    base_calls = (
        *_source_calls(download, path),
        ExpertCall("from PyPDF2 import PdfReader", ""),
    )
    if password_protected:
        calls = (
            *base_calls,
            ExpertCall(
                "reader = PdfReader(document_path)",
                _traceback("FileNotDecryptedError", "File has not been decrypted"),
            ),
        )
        status = "password_protected"
        failure = "password_protected_pdf"
    else:
        calls = (
            *base_calls,
            ExpertCall(
                "reader = PdfReader(document_path)\nfirst_page = reader.pages[0]", ""
            ),
            ExpertCall(
                "first_page_text = first_page.extract_text()\nrepr(first_page_text)",
                "''",
            ),
        )
        status = "no_extractable_text"
        failure = "scanned_pdf"
    return FileProcessingScenario(
        instruction=(
            f"{_source_prompt(download)} Use `PyPDF2.PdfReader` on the retained path and "
            "inspect the real result. Return the evidenced terminal JSON status if the PDF "
            "is protected or its first page has no extractable text. Do not fabricate text, "
            "retry unchanged cells, or install random fallback packages."
        ),
        explicit_operation=(
            "Inspect the mapping, select its path, import the parser once, and distinguish "
            "a protected-file exception from a successful empty text extraction."
        ),
        answer={"status": status},
        files={path: payload},
        expert_calls=calls,
        source_kind="structured_download",
        file_kind="pdf",
        failure_kind=failure,
        expected_output_marker=None,
        terminal_status=status,
    )


def generate_file_processing_scenario(
    variant: int, instance: int, rng: random.Random
) -> FileProcessingScenario:
    del rng
    index = (variant * 5 + instance) % 17
    if index <= 6:
        return _build_pdf_scenario(index, variant, instance)
    if index == 7:
        return _text_scenario(variant, instance, markdown=False)
    if index == 8:
        return _text_scenario(variant, instance, markdown=True)
    if index == 9:
        return _csv_scenario(variant, instance, malformed=False)
    if index == 10:
        return _csv_scenario(variant, instance, malformed=True)
    if index == 11:
        return _json_scenario(variant, instance, invalid=False)
    if index == 12:
        return _json_scenario(variant, instance, invalid=True)
    if index == 13:
        return _docx_scenario(variant, instance)
    if index == 14:
        return _unknown_scenario(variant, instance)
    return _terminal_pdf_scenario(variant, instance, password_protected=index == 16)


__all__ = [
    "FILE_PROCESSING_FIXTURES",
    "ExpertCall",
    "FileKind",
    "FileProcessingScenario",
    "generate_file_processing_scenario",
]

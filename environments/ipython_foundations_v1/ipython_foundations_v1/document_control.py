"""Full-document repair scenarios with contrastive factual claims."""

from __future__ import annotations

import base64
import json
import random
from dataclasses import dataclass

from ipython_foundations_v1.file_processing import ExpertCall


@dataclass(frozen=True)
class DocumentControlScenario:
    instruction: str
    explicit_operation: str
    answer: dict[str, object]
    files: dict[str, str]
    expert_calls: tuple[ExpertCall, ...]
    failure_kind: str
    expected_output_marker: str


def _traceback(error: str, message: str) -> str:
    return f'Traceback (most recent call last):\n  File "<ipython-input>", line 1, in <module>\n{error}: {message}'


def _download(path: str, payload: str) -> dict[str, object]:
    return {
        "path": path,
        "filename": path.rsplit("/", 1)[-1],
        "bytes": len(payload.encode()),
        "content_type": "application/pdf",
    }


def _claim(index: int) -> tuple[str, str]:
    claims = (
        ("The safety review did not approve deployment.", "not_approved"),
        ("The safety review approved deployment.", "approved"),
        ("No material variance was detected.", "no_variance"),
        ("A material variance was detected.", "variance_detected"),
    )
    return claims[index % len(claims)]


def _stale_operation(index: int) -> tuple[str, str, str]:
    cases = (
        (
            "page_count = len(reader)\npage_count",
            "TypeError",
            "object of type 'PdfReader' has no len()",
        ),
        (
            "page_texts = reader.pages.extract_text()\npage_texts",
            "AttributeError",
            "'list' object has no attribute 'extract_text'",
        ),
        (
            "full_text = '\\n'.join(reader.pages)\nfull_text",
            "TypeError",
            "sequence item 0: expected str instance, Page found",
        ),
        (
            "page_texts = [page.text for page in reader.pages]\npage_texts",
            "AttributeError",
            "'Page' object has no attribute 'text'",
        ),
    )
    return cases[index % len(cases)]


def generate_document_control_scenario(variant: int, instance: int, rng: random.Random) -> DocumentControlScenario:
    del rng
    index = (variant + instance) % 4
    title = f"Deployment Review {variant}-{instance}"
    finding, status = _claim(index)
    pages = (
        f"Title: {title}\nScope: release readiness.\n",
        f"Finding: {finding}\n",
        "Context: This conclusion applies only to the reviewed deployment.\n",
    )
    payload = base64.b64encode("\f".join(pages).encode()).decode()
    path = f"/workspace/inbox/deployment-review-{variant}-{instance}.pdf"
    download = _download(path, payload)
    stale, error, message = _stale_operation(index)
    combined_text = "\n".join(pages)
    answer: dict[str, object] = {
        "title": title,
        "finding": finding,
        "status": status,
    }
    calls = (
        ExpertCall(
            f"download = {download!r}\ntype(download), sorted(download)",
            "(<class 'dict'>, ['bytes', 'content_type', 'filename', 'path'])",
        ),
        ExpertCall("document_path = download['path']\ndocument_path", repr(path)),
        ExpertCall("from PyPDF2 import PdfReader", ""),
        ExpertCall("reader = PdfReader(document_path)\ntype(reader)", "<class 'PyPDF2.PdfReader'>"),
        ExpertCall(stale, _traceback(error, message)),
        ExpertCall(
            "page_count = len(reader.pages)\n"
            "page_texts = [page.extract_text() for page in reader.pages]\n"
            "full_text = '\\n'.join(page_texts)\n"
            "page_count, full_text",
            repr((3, combined_text)),
        ),
    )
    return DocumentControlScenario(
        instruction=(
            "The completed download is already retained as the structured object "
            f"`{json.dumps(download, sort_keys=True)}`. Assign it to `download`, inspect "
            "its type and keys, and retain `download['path']` as `document_path`; do not "
            "list or download the file again. Open the PDF once with `PdfReader`, then run "
            f"the inherited operation `{stale}` exactly once so its live traceback is "
            "visible. Repair the failed expression from that traceback, iterate "
            "`reader.pages`, call `extract_text()` on every page, and retain their joined "
            "text as `full_text`. Return exactly one JSON object with keys `title`, "
            "`finding`, and `status`. Preserve the source's negation literally; do not "
            "weaken, reverse, or infer beyond the extracted text."
        ),
        explicit_operation=(
            "Inspect the download mapping, retain its path, create one reader, execute the "
            "stale expression once, and replace only that failed expression. Demonstrate "
            "repair by displaying the page count and joined text before returning strict "
            "JSON grounded in `full_text`."
        ),
        answer=answer,
        files={path: payload},
        expert_calls=calls,
        failure_kind=("document_control_" + ("len_reader", "pages_api", "page_join", "page_attribute")[index]),
        expected_output_marker=finding,
    )


__all__ = ["DocumentControlScenario", "generate_document_control_scenario"]

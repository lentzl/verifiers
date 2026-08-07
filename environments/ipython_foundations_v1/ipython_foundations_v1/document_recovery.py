"""Controlled document parsers for traceback-driven API recovery."""

from __future__ import annotations

PARSER_FIXTURES = {
    "fitz.py": """import base64
from pathlib import Path


class Page:
    def __init__(self, text):
        self._text = text

    def get_text(self):
        return self._text


class Document:
    def __init__(self, text):
        self._pages = [Page(page) for page in text.split("\\f")]

    def __getitem__(self, index):
        return self._pages[index]

    def __len__(self):
        return len(self._pages)


def open(source):
    encoded = Path(source).read_bytes()
    return Document(base64.b64decode(encoded).decode())
""",
    "pymupdf-1.26.3.dist-info/METADATA": (
        "Metadata-Version: 2.1\nName: pymupdf\nVersion: 1.26.3\n"
    ),
    "pymupdf-1.26.3.dist-info/top_level.txt": "fitz\n",
    "pymupdf-1.26.3.dist-info/RECORD": (
        "fitz.py,,\n"
        "pymupdf-1.26.3.dist-info/METADATA,,\n"
        "pymupdf-1.26.3.dist-info/top_level.txt,,\n"
    ),
    "pdfminer/__init__.py": "",
    "pdfminer/high_level.py": """import base64
from pathlib import Path


def extract_text(source, page_numbers=None):
    encoded = Path(source).read_bytes()
    pages = base64.b64decode(encoded).decode().split("\\f")
    if page_numbers is None:
        return "\\f".join(pages)
    return "\\f".join(pages[index] for index in page_numbers)
""",
    "pdfminer.six-20250506.dist-info/METADATA": (
        "Metadata-Version: 2.1\nName: pdfminer.six\nVersion: 20250506\n"
    ),
    "pdfminer.six-20250506.dist-info/top_level.txt": "pdfminer\n",
    "pdfminer.six-20250506.dist-info/RECORD": (
        "pdfminer/__init__.py,,\n"
        "pdfminer/high_level.py,,\n"
        "pdfminer.six-20250506.dist-info/METADATA,,\n"
        "pdfminer.six-20250506.dist-info/top_level.txt,,\n"
    ),
}

__all__ = ["PARSER_FIXTURES"]

from __future__ import annotations

import logging
from pathlib import Path
import warnings

try:
    from pypdf import PdfReader
except ImportError as exc:
    PdfReader = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


logging.getLogger("pypdf").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="pypdf")


def parse_pdf(file_path: str | Path) -> str:
    """Extract text from a PDF file."""
    if PdfReader is None:
        raise ImportError(
            "Missing dependency 'pypdf'. Install it with: pip install pypdf"
        ) from _IMPORT_ERROR

    pdf_path = Path(file_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    page_text = []

    for page in reader.pages:
        text = page.extract_text() or ""
        page_text.append(text)

    return "\n".join(page_text).strip()


def parse_text_file(file_path: str | Path) -> str:
    """Read plain text files using common encodings for this corpus."""
    text_path = Path(file_path)
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")

    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return text_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    return text_path.read_text(encoding="latin-1")

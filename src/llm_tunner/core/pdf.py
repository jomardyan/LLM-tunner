"""PDF text extraction.

Primary backend: **Docling** (MIT, layout/table-aware, AI-ready Markdown). Fallback:
**pypdf** (BSD, pure-Python, always available). We deliberately avoid PyMuPDF because
its AGPL-3.0 license is a copyleft trap for redistributable software.

Both backends return :class:`ExtractedDoc`, so downstream chunking/RAG/QA code is
backend-agnostic.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PageText:
    page_number: int  # 1-based
    text: str


@dataclass
class ExtractedDoc:
    """Normalised result of parsing one PDF."""

    source: str  # absolute path
    pages: list[PageText] = field(default_factory=list)
    markdown: str = ""  # full-document Markdown when the backend provides it
    backend: str = "pypdf"

    @property
    def full_text(self) -> str:
        if self.markdown:
            return self.markdown
        return "\n\n".join(p.text for p in self.pages)

    @property
    def num_pages(self) -> int:
        return len(self.pages)


def docling_available() -> bool:
    return importlib.util.find_spec("docling") is not None


def extract_pdf(path: str | Path, prefer: str = "auto") -> ExtractedDoc:
    """Extract text from a PDF.

    Args:
        path: Path to the PDF.
        prefer: ``"auto"`` (Docling if installed, else pypdf), ``"docling"`` or ``"pypdf"``.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    if prefer in ("auto", "docling") and docling_available():
        try:
            return _extract_with_docling(path)
        except Exception:
            if prefer == "docling":
                raise
            # fall through to pypdf on any Docling failure
    return _extract_with_pypdf(path)


def _extract_with_docling(path: Path) -> ExtractedDoc:
    from docling.document_converter import DocumentConverter  # type: ignore

    converter = DocumentConverter()
    result = converter.convert(str(path))
    doc = result.document
    markdown = doc.export_to_markdown()

    # Docling exposes per-page info; we keep a best-effort page split for citations.
    pages: list[PageText] = []
    try:
        # Group text items by their page number when available.
        by_page: dict[int, list[str]] = {}
        for item, _level in doc.iterate_items():
            text = getattr(item, "text", None)
            if not text:
                continue
            prov = getattr(item, "prov", None) or []
            page_no = getattr(prov[0], "page_no", 1) if prov else 1
            by_page.setdefault(int(page_no), []).append(text)
        for page_no in sorted(by_page):
            pages.append(PageText(page_number=page_no, text="\n".join(by_page[page_no])))
    except Exception:
        pass

    if not pages:
        pages = [PageText(page_number=1, text=markdown)]

    return ExtractedDoc(source=str(path), pages=pages, markdown=markdown, backend="docling")


def _extract_with_pypdf(path: Path) -> ExtractedDoc:
    from pypdf import PdfReader  # type: ignore

    reader = PdfReader(str(path))
    pages: list[PageText] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append(PageText(page_number=i, text=text))
    return ExtractedDoc(source=str(path), pages=pages, backend="pypdf")

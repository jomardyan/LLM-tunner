"""PDF extraction via the pypdf backend (always available)."""

from __future__ import annotations

from llm_tunner.core.chunking import chunk_document
from llm_tunner.core.pdf import ExtractedDoc, PageText, _classify_native_text, extract_pdf


def test_extract_pypdf(sample_pdf):
    doc = extract_pdf(sample_pdf, prefer="pypdf")
    assert doc.num_pages == 1
    assert doc.backend == "pypdf"
    assert doc.document_type == "text"
    assert not doc.used_ocr
    assert "Paris" in doc.full_text


def test_auto_uses_native_text_without_docling(sample_pdf, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Docling should not run for a native-text PDF")

    monkeypatch.setattr("llm_tunner.core.pdf._extract_with_docling", fail_if_called)
    doc = extract_pdf(sample_pdf)
    assert doc.backend == "pypdf"
    assert doc.document_type == "text"


def test_auto_uses_docling_ocr_for_scanned_pdf(sample_pdf, monkeypatch):
    scanned = ExtractedDoc(
        source=str(sample_pdf),
        pages=[PageText(1, "")],
        document_type="scanned",
    )
    ocr_result = ExtractedDoc(
        source=str(sample_pdf),
        pages=[PageText(1, "OCR output")],
        backend="docling",
        document_type="scanned",
        used_ocr=True,
    )
    calls = []

    monkeypatch.setattr("llm_tunner.core.pdf._extract_with_pypdf", lambda _path: scanned)
    monkeypatch.setattr("llm_tunner.core.pdf.docling_available", lambda: True)

    def fake_docling(_path, *, do_ocr):
        calls.append(do_ocr)
        return ocr_result

    monkeypatch.setattr("llm_tunner.core.pdf._extract_with_docling", fake_docling)

    assert extract_pdf(sample_pdf) is ocr_result
    assert calls == [True]


def test_native_text_classification():
    text_page = PageText(1, "This page contains enough normal embedded text to extract.")
    blank_page = PageText(2, "")

    assert _classify_native_text(ExtractedDoc("text.pdf", [text_page])) == "text"
    assert _classify_native_text(ExtractedDoc("scan.pdf", [blank_page])) == "scanned"
    assert (
        _classify_native_text(ExtractedDoc("mixed.pdf", [text_page, blank_page]))
        == "mixed"
    )


def test_chunk_document_preserves_provenance(sample_pdf):
    doc = extract_pdf(sample_pdf, prefer="pypdf")
    chunks = chunk_document(doc)
    assert chunks
    assert all(c.source == str(sample_pdf) for c in chunks)
    assert all(c.page_number == 1 for c in chunks)
    assert any("Paris" in c.text for c in chunks)


def test_missing_file():
    import pytest

    with pytest.raises(FileNotFoundError):
        extract_pdf("/no/such/file.pdf")

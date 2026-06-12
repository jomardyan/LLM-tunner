"""PDF extraction via the pypdf backend (always available)."""

from __future__ import annotations

from llm_tunner.core.chunking import chunk_document
from llm_tunner.core.pdf import extract_pdf


def test_extract_pypdf(sample_pdf):
    doc = extract_pdf(sample_pdf, prefer="pypdf")
    assert doc.num_pages == 1
    assert doc.backend == "pypdf"
    assert "Paris" in doc.full_text


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

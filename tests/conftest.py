"""Shared test fixtures, including a generator for a minimal valid text PDF.

We build the PDF bytes by hand (with correct xref offsets) so the test suite needs no
checked-in binary fixtures and no PDF-creation dependency.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def make_pdf_bytes(paragraphs: list[str]) -> bytes:
    """Build a single-page PDF whose content stream prints the given paragraphs.

    The text is laid out one paragraph per line; pypdf/Docling can extract it.
    """
    # Build the content stream: each paragraph on its own line via Td offsets.
    lines = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
    for i, para in enumerate(paragraphs):
        safe = para.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        if i == 0:
            lines.append(f"({safe}) Tj")
        else:
            lines.append("T*")
            lines.append(f"({safe}) Tj")
    lines.append("ET")
    content = "\n".join(lines).encode("latin-1")

    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
    )
    objects.append(
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream"
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"

    xref_pos = len(out)
    n = len(objects) + 1
    out += f"xref\n0 {n}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode()
    return bytes(out)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "sample.pdf"
    path.write_bytes(
        make_pdf_bytes(
            [
                "LLM-tunner is a desktop application for customizing language models.",
                "It supports retrieval augmented generation over PDF documents.",
                "The capital of France is Paris.",
            ]
        )
    )
    return path


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Point LLM_TUNNER_HOME at a temp dir so tests never touch the real user data."""
    monkeypatch.setenv("LLM_TUNNER_HOME", str(tmp_path / "home"))
    os.makedirs(tmp_path / "home", exist_ok=True)
    yield

"""Text chunking for RAG indexing and QA generation.

Implements a dependency-free recursive character splitter (the research-recommended
generic default), so chunking works even before the heavy ML stack is installed.
Splits hierarchically on paragraph -> line -> sentence -> word -> char boundaries,
targeting a configurable size with overlap. "Size" is measured in characters by
default; pass a ``length_fn`` (e.g. a tokenizer) to count tokens instead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..config import ChunkConfig
from .pdf import ExtractedDoc

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    text: str
    source: str
    page_number: int
    index: int  # ordinal within the document


def _split_recursive(text: str, size: int, length_fn: Callable[[str], int], seps: list[str]) -> list[str]:
    if length_fn(text) <= size or not seps:
        return [text] if text.strip() else []

    sep, *rest = seps
    parts = text.split(sep) if sep else list(text)

    chunks: list[str] = []
    buf = ""
    for part in parts:
        candidate = part if not buf else buf + sep + part
        if length_fn(candidate) <= size:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
        # `part` alone may still be too big -> recurse with finer separators.
        if length_fn(part) > size:
            chunks.extend(_split_recursive(part, size, length_fn, rest))
            buf = ""
        else:
            buf = part
    if buf.strip():
        chunks.append(buf)
    return chunks


def _apply_overlap(chunks: list[str], overlap: int, length_fn: Callable[[str], int]) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        tail = prev[-overlap * 6 :]  # ~chars; coarse but effective for char/token sizes
        out.append((tail + " " + cur).strip())
    return out


def chunk_text(
    text: str,
    config: ChunkConfig | None = None,
    length_fn: Callable[[str], int] | None = None,
) -> list[str]:
    """Split a raw string into overlapping chunks."""
    config = config or ChunkConfig()
    length_fn = length_fn or len  # default: character count
    raw = _split_recursive(text, config.chunk_size, length_fn, list(_SEPARATORS))
    cleaned = [c.strip() for c in raw if c.strip()]
    return _apply_overlap(cleaned, config.chunk_overlap, length_fn)


def chunk_document(
    doc: ExtractedDoc,
    config: ChunkConfig | None = None,
    length_fn: Callable[[str], int] | None = None,
) -> list[Chunk]:
    """Chunk a parsed PDF, preserving page provenance for citations."""
    config = config or ChunkConfig()
    chunks: list[Chunk] = []
    idx = 0
    for page in doc.pages:
        for piece in chunk_text(page.text, config, length_fn):
            chunks.append(
                Chunk(text=piece, source=doc.source, page_number=page.page_number, index=idx)
            )
            idx += 1
    return chunks

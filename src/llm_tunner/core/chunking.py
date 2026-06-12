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


def _suffix_within(text: str, limit: int, length_fn: Callable[[str], int]) -> str:
    """Return the longest character suffix whose measured length fits ``limit``."""
    if limit <= 0:
        return ""
    if length_fn(text) <= limit:
        return text

    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if length_fn(text[-mid:]) <= limit:
            low = mid
        else:
            high = mid - 1
    if not low:
        return ""

    start = len(text) - low
    suffix = text[start:]
    if start > 0 and not text[start - 1].isspace():
        boundary = suffix.find(" ")
        suffix = suffix[boundary + 1 :] if boundary >= 0 else ""
    return suffix


def _apply_overlap(
    chunks: list[str],
    size: int,
    overlap: int,
    length_fn: Callable[[str], int],
) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        separator = " "
        room = size - length_fn(cur) - length_fn(separator)
        tail = _suffix_within(prev, min(overlap, room), length_fn).strip()
        out.append(f"{tail}{separator}{cur}".strip() if tail else cur)
    return out


def chunk_text(
    text: str,
    config: ChunkConfig | None = None,
    length_fn: Callable[[str], int] | None = None,
) -> list[str]:
    """Split a raw string into overlapping chunks."""
    config = config or ChunkConfig()
    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if config.chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    length_fn = length_fn or len  # default: character count
    raw = _split_recursive(text, config.chunk_size, length_fn, list(_SEPARATORS))
    cleaned = [c.strip() for c in raw if c.strip()]
    return _apply_overlap(cleaned, config.chunk_size, config.chunk_overlap, length_fn)


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

"""Chunking is dependency-free and deterministic — test it directly."""

from __future__ import annotations

from llm_tunner.config import ChunkConfig
from llm_tunner.core.chunking import chunk_text


def test_short_text_single_chunk():
    chunks = chunk_text("Hello world.", ChunkConfig(chunk_size=100, chunk_overlap=0))
    assert chunks == ["Hello world."]


def test_long_text_splits():
    text = ("Sentence one. " * 50).strip()
    chunks = chunk_text(text, ChunkConfig(chunk_size=80, chunk_overlap=0))
    assert len(chunks) > 1
    assert all(len(c) <= 80 + 20 for c in chunks)  # small slack for boundary words


def test_deterministic():
    text = "Para one.\n\nPara two is a bit longer than para one.\n\nPara three."
    cfg = ChunkConfig(chunk_size=40, chunk_overlap=0)
    assert chunk_text(text, cfg) == chunk_text(text, cfg)


def test_empty_text():
    assert chunk_text("   ", ChunkConfig()) == []

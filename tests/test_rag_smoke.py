"""End-to-end RAG smoke test. Skipped unless chromadb + sentence-transformers are present.

Downloads a tiny embedding model on first run; marked slow/optional for CI without net.
"""

from __future__ import annotations

import importlib.util

import pytest

_HAVE_RAG = all(
    importlib.util.find_spec(m) for m in ("chromadb", "sentence_transformers")
)

pytestmark = pytest.mark.skipif(not _HAVE_RAG, reason="RAG extras not installed")


def test_index_and_retrieve(sample_pdf):
    from llm_tunner.config import RagConfig
    from llm_tunner.core.rag import RagIndex

    index = RagIndex("smoke_kb", config=RagConfig(top_k=2))
    added = index.add_pdf(sample_pdf)
    assert added > 0
    assert index.count() == added

    results = index.retrieve("What is the capital of France?", top_k=2)
    assert results
    # The most relevant chunk should mention Paris.
    assert any("Paris" in r.text for r in results)
    assert results[0].citation.endswith(("p.1", "p.2", "p.3"))

    messages = index.build_prompt("What is the capital of France?", results)
    assert messages[0]["role"] == "system"
    assert "Context:" in messages[0]["content"]

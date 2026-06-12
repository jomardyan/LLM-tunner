from __future__ import annotations

from llm_tunner.core.chunking import Chunk
from llm_tunner.core.rag import (
    RagIndex,
    _chunk_id,
    _collection_name,
    _storage_name,
)


class FakeCollection:
    def __init__(self, existing_ids: dict[str, list[str]] | None = None) -> None:
        self.existing_ids = existing_ids or {}
        self.upserts: list[dict] = []
        self.deleted: list[str] = []

    def get(self, *, where, include):
        return {"ids": self.existing_ids.get(where["source"], [])}

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def delete(self, *, ids):
        self.deleted.extend(ids)


class FakeEmbedder:
    def encode(self, texts):
        return [[float(len(text))] for text in texts]


def _chunk(source: str, index: int, text: str = "text") -> Chunk:
    return Chunk(text=text, source=source, page_number=1, index=index)


def test_collection_name_handles_short_names():
    assert len(_collection_name("a")) >= 3
    assert _collection_name("---")[0].isalnum()
    assert _collection_name("---")[-1].isalnum()


def test_storage_name_is_windows_safe_and_bounded():
    assert _storage_name("CON") == "kb-con"
    first = _storage_name("a" * 100)
    second = _storage_name("a" * 99 + "b")
    assert len(first) <= 64
    assert first != second


def test_chunk_ids_do_not_collide_for_same_basename():
    first = _chunk("/one/report.pdf", 0)
    second = _chunk("/two/report.pdf", 0)
    assert _chunk_id(first) != _chunk_id(second)


def test_reindex_upserts_and_removes_stale_chunks(monkeypatch):
    source = "/docs/report.pdf"
    chunks = [_chunk(source, 0, "new zero"), _chunk(source, 1, "new one")]
    stale = _chunk(source, 2, "old")
    collection = FakeCollection(
        existing_ids={source: [_chunk_id(chunks[0]), _chunk_id(stale)]}
    )
    index = RagIndex.__new__(RagIndex)
    index._embedder = FakeEmbedder()
    monkeypatch.setattr(index, "_coll", lambda: collection)

    assert index._add_chunks(chunks) == 2
    assert collection.upserts[0]["ids"] == [_chunk_id(c) for c in chunks]
    assert collection.deleted == [_chunk_id(stale)]

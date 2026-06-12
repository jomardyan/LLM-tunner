"""Retrieval-Augmented Generation: the default "feed PDF as knowledge" path.

Pipeline: PDF -> :mod:`core.pdf` -> :mod:`core.chunking` -> embed (sentence-transformers)
-> persist in a per-knowledge-base **ChromaDB** collection -> retrieve top-k at query
time -> build a grounded prompt with **page citations**.

Heavy deps (chromadb, sentence-transformers) are imported lazily inside methods so the
GUI launches without them; a clear error is raised only when RAG is actually used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..config import RagConfig, kb_dir
from .chunking import Chunk, chunk_document
from .pdf import ExtractedDoc, extract_pdf


@dataclass
class Retrieved:
    text: str
    source: str
    page_number: int
    score: float

    @property
    def citation(self) -> str:
        return f"{Path(self.source).name} p.{self.page_number}"


@dataclass
class RagAnswer:
    answer: str
    contexts: list[Retrieved]

    def formatted_sources(self) -> str:
        seen: list[str] = []
        for c in self.contexts:
            if c.citation not in seen:
                seen.append(c.citation)
        return "; ".join(seen)


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower() or "kb"


class Embedder:
    """Lazily-loaded sentence-transformers wrapper, device-aware."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            from .device import detect_device

            device = detect_device().kind
            st_device = device if device in ("cuda", "mps") else "cpu"
            self._model = SentenceTransformer(self.model_name, device=st_device)
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


class RagIndex:
    """A persistent ChromaDB-backed knowledge base.

    One :class:`RagIndex` == one named knowledge base on disk under ``kb_dir()``.
    """

    def __init__(self, name: str, config: RagConfig | None = None) -> None:
        self.name = name
        self.config = config or RagConfig()
        self.path = kb_dir() / _slug(name)
        self._embedder = Embedder(self.config.embedding_model)
        self._client = None
        self._collection = None

    # -- storage ------------------------------------------------------------------
    def _coll(self):
        if self._collection is None:
            import chromadb  # type: ignore

            self._client = chromadb.PersistentClient(path=str(self.path))
            self._collection = self._client.get_or_create_collection(
                name=_slug(self.name),
                metadata={"hnsw:space": "cosine", "embedding_model": self.config.embedding_model},
            )
        return self._collection

    def count(self) -> int:
        try:
            return self._coll().count()
        except Exception:
            return 0

    # -- ingestion ----------------------------------------------------------------
    def add_pdf(self, pdf_path: str | Path, progress=None) -> int:
        """Extract, chunk, embed and store one PDF. Returns the number of chunks added.

        ``progress`` is an optional callable ``(done, total, message)`` for GUI updates.
        """
        doc: ExtractedDoc = extract_pdf(pdf_path)
        chunks: list[Chunk] = chunk_document(doc, self.config.chunk)
        if not chunks:
            return 0
        return self._add_chunks(chunks, progress=progress)

    def _add_chunks(self, chunks: list[Chunk], progress=None) -> int:
        coll = self._coll()
        batch = 64
        total = len(chunks)
        for start in range(0, total, batch):
            window = chunks[start : start + batch]
            embeddings = self._embedder.encode([c.text for c in window])
            coll.add(
                ids=[f"{_slug(Path(c.source).name)}-{c.page_number}-{c.index}" for c in window],
                documents=[c.text for c in window],
                embeddings=embeddings,
                metadatas=[
                    {"source": c.source, "page_number": c.page_number, "index": c.index}
                    for c in window
                ],
            )
            if progress:
                progress(min(start + batch, total), total, f"Embedded {min(start + batch, total)}/{total} chunks")
        return total

    # -- retrieval ----------------------------------------------------------------
    def retrieve(self, query: str, top_k: int | None = None) -> list[Retrieved]:
        top_k = top_k or self.config.top_k
        coll = self._coll()
        if coll.count() == 0:
            return []
        q_emb = self._embedder.encode([query])[0]
        res = coll.query(query_embeddings=[q_emb], n_results=top_k)
        out: list[Retrieved] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists, strict=False):
            out.append(
                Retrieved(
                    text=doc,
                    source=str(meta.get("source", "")),
                    page_number=int(meta.get("page_number", 1)),
                    score=1.0 - float(dist),  # cosine distance -> similarity
                )
            )
        return out

    def build_prompt(self, query: str, contexts: list[Retrieved]) -> list[dict]:
        """Construct chat `messages` grounding the model in retrieved context."""
        context_block = "\n\n".join(
            f"[{i + 1}] ({c.citation})\n{c.text}" for i, c in enumerate(contexts)
        )
        system = (
            "You are a helpful assistant. Answer the user's question using ONLY the "
            "context below. Cite sources inline using the bracketed numbers, e.g. [1]. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context_block}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]

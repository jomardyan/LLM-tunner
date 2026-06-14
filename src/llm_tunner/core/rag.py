"""Retrieval-Augmented Generation: the default "feed PDF as knowledge" path.

Pipeline: PDF -> :mod:`core.pdf` -> :mod:`core.chunking` -> embed (ONNX Runtime)
-> persist in a per-knowledge-base **ChromaDB** collection -> retrieve top-k at query
time -> build a grounded prompt with **page citations**.

Heavy deps (chromadb, ONNX Runtime) are imported lazily inside methods so the
GUI launches without them; a clear error is raised only when RAG is actually used.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from ..config import RagConfig, kb_dir
from ..diagnostics import configure_ml_output
from .chunking import Chunk, chunk_document
from .pdf import ExtractedDoc, extract_pdf

_RAG_MODULES = {
    "chromadb": "chromadb",
    "onnxruntime": "onnxruntime",
    "tokenizers": "tokenizers",
}


def missing_rag_dependencies() -> list[str]:
    """Return missing distribution names required by the RAG pipeline."""
    return [
        distribution
        for module, distribution in _RAG_MODULES.items()
        if importlib.util.find_spec(module) is None
    ]


def rag_install_command() -> str:
    """Return an install command targeting the interpreter running the app."""
    return f'"{sys.executable}" -m pip install -e ".[rag]"'


def require_rag_dependencies() -> None:
    """Fail with actionable installation guidance when RAG extras are absent."""
    missing = missing_rag_dependencies()
    if not missing:
        return

    names = ", ".join(missing)
    raise RuntimeError(
        f"RAG dependencies are not installed: {names}. "
        f"Install them in the Python environment running this app with: "
        f"{rag_install_command()} "
        "(or run: make install-rag), then restart the app."
    )


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


_WINDOWS_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def _bounded_slug(name: str, max_length: int) -> str:
    slug = _slug(name).strip("-_") or "kb"
    if slug in _WINDOWS_RESERVED_NAMES:
        slug = f"kb-{slug}"
    if len(slug) > max_length:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:10]
        slug = f"{slug[: max_length - len(digest) - 1].rstrip('-_')}-{digest}"
    return slug


def _storage_name(name: str) -> str:
    """Return a short cross-platform directory name, safe on Windows filesystems."""
    return _bounded_slug(name, 64)


def _collection_name(name: str) -> str:
    """Return a Chroma-compatible collection name, including for short user input."""
    slug = _bounded_slug(name, 63)
    if len(slug) < 3:
        slug = f"{slug or 'kb'}-index"
    return slug


def _chunk_id(chunk: Chunk) -> str:
    source_hash = hashlib.sha256(chunk.source.encode("utf-8")).hexdigest()[:16]
    return f"{source_hash}-{chunk.page_number}-{chunk.index}"


def _recent_turns(history: list[dict] | None, max_turns: int) -> list[dict]:
    """Return the last ``max_turns`` user/assistant turns, starting on a user turn.

    Bounds prompt growth for multi-turn chat and avoids slicing into the middle of a
    turn (which would start the history on an assistant message).
    """
    if not history or max_turns <= 0:
        return []
    recent = list(history[-max_turns * 2 :])
    if recent and recent[0].get("role") != "user":
        recent = recent[1:]
    return recent


# ONNX Runtime execution providers, fastest first; CPU is always the floor. An
# accelerated provider only appears here if the matching onnxruntime build is installed
# (onnxruntime-directml / onnxruntime-openvino / onnxruntime-gpu), so this degrades
# gracefully to CPU otherwise.
_ONNX_PROVIDER_PRIORITY = (
    "DmlExecutionProvider",  # Windows: any DirectX-12 GPU (Intel/AMD/NVIDIA iGPU+dGPU)
    "OpenVINOExecutionProvider",  # Intel CPU/iGPU/NPU (supports simultaneous CPU+iGPU)
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",  # Apple Silicon
    "CPUExecutionProvider",
)


def onnx_providers(preference: str = "auto") -> list:
    """Resolve the ONNX Runtime providers list for the embedder.

    With ``preference="auto"`` (default) the fastest installed accelerator is used and CPU
    is appended as a fallback. An explicit preference (``cpu``/``directml``/``openvino``/
    ``openvino-multi``/``cuda``/``coreml``) is honored only when that provider is actually
    installed, otherwise it falls through to auto. OpenVINO entries carry a ``device_type``
    so the integrated GPU is used (``MULTI:GPU,CPU`` runs CPU and iGPU simultaneously).
    The result is always filtered to installed providers and ends with CPU, so a missing
    accelerator never errors — it just runs on CPU.
    """
    try:
        import onnxruntime  # type: ignore

        available = set(onnxruntime.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]
    available.add("CPUExecutionProvider")
    cpu = "CPUExecutionProvider"
    pref = (preference or "auto").strip().lower()

    def ov(device_type: str):
        return ("OpenVINOExecutionProvider", {"device_type": device_type})

    explicit = {
        "cpu": [cpu],
        "directml": ["DmlExecutionProvider", cpu] if "DmlExecutionProvider" in available else None,
        "dml": ["DmlExecutionProvider", cpu] if "DmlExecutionProvider" in available else None,
        "cuda": ["CUDAExecutionProvider", cpu] if "CUDAExecutionProvider" in available else None,
        "coreml": ["CoreMLExecutionProvider", cpu]
        if "CoreMLExecutionProvider" in available
        else None,
        "openvino": [ov("AUTO"), cpu] if "OpenVINOExecutionProvider" in available else None,
        "openvino-multi": [ov("MULTI:GPU,CPU"), cpu]
        if "OpenVINOExecutionProvider" in available
        else None,
    }
    if pref in explicit and explicit[pref] is not None:
        return explicit[pref]

    ordered: list = []
    for name in _ONNX_PROVIDER_PRIORITY:
        if name not in available:
            continue
        ordered.append(ov("AUTO") if name == "OpenVINOExecutionProvider" else name)
    if not any((p[0] if isinstance(p, tuple) else p) == cpu for p in ordered):
        ordered.append(cpu)
    return ordered


class Embedder:
    """ONNX encoder with attention-mask mean pooling and L2 normalization.

    Runs on the fastest available ONNX Runtime provider (integrated/discrete GPU when an
    accelerated runtime is installed), falling back to CPU. ``provider`` selects/forces a
    backend; see :func:`onnx_providers`.
    """

    def __init__(self, model_name: str, provider: str = "auto") -> None:
        self.model_name = model_name
        self.provider = provider
        self._session = None
        self._tokenizer = None

    def _load(self):
        if self._session is None:
            require_rag_dependencies()
            import onnxruntime  # type: ignore
            from huggingface_hub import hf_hub_download  # type: ignore
            from huggingface_hub.errors import EntryNotFoundError  # type: ignore
            from tokenizers import Tokenizer  # type: ignore

            configure_ml_output()
            model_path = hf_hub_download(self.model_name, "onnx/model.onnx")
            try:
                hf_hub_download(self.model_name, "onnx/model.onnx_data")
            except EntryNotFoundError:
                pass
            try:
                tokenizer_path = hf_hub_download(self.model_name, "onnx/tokenizer.json")
            except EntryNotFoundError:
                tokenizer_path = hf_hub_download(self.model_name, "tokenizer.json")

            self._tokenizer = Tokenizer.from_file(tokenizer_path)
            self._tokenizer.enable_truncation(max_length=512)
            pad_token = next(
                (
                    token
                    for token in ("[PAD]", "<pad>", "<|pad|>")
                    if self._tokenizer.token_to_id(token) is not None
                ),
                "[PAD]",
            )
            pad_id = self._tokenizer.token_to_id(pad_token) or 0
            self._tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)

            providers = onnx_providers(self.provider)
            options = onnxruntime.SessionOptions()
            # Use all CPU cores for ops that fall back to (or stay on) the CPU provider.
            options.intra_op_num_threads = os.cpu_count() or 0
            # DirectML does not support ORT's memory pattern planner.
            if any(
                (p[0] if isinstance(p, tuple) else p) == "DmlExecutionProvider" for p in providers
            ):
                options.enable_mem_pattern = False
            self._session = onnxruntime.InferenceSession(
                model_path,
                sess_options=options,
                providers=providers,
            )
        return self._tokenizer, self._session

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import numpy  # type: ignore

        tokenizer, session = self._load()
        encodings = tokenizer.encode_batch(texts)
        available_inputs = {item.name for item in session.get_inputs()}
        values = {
            "input_ids": numpy.asarray([item.ids for item in encodings], dtype=numpy.int64),
            "attention_mask": numpy.asarray(
                [item.attention_mask for item in encodings],
                dtype=numpy.int64,
            ),
            "token_type_ids": numpy.asarray(
                [item.type_ids for item in encodings],
                dtype=numpy.int64,
            ),
        }
        outputs = session.run(None, {key: value for key, value in values.items()
                                     if key in available_inputs})
        embeddings = next((output for output in outputs if output.ndim == 2), None)
        if embeddings is None:
            hidden = next((output for output in outputs if output.ndim == 3), None)
            if hidden is None:
                raise RuntimeError(
                    f"ONNX model '{self.model_name}' produced no 2D or 3D output tensor; "
                    f"output shapes: {[o.shape for o in outputs]}"
                )
            mask = values["attention_mask"][..., None].astype(numpy.float32)
            embeddings = (hidden * mask).sum(axis=1) / numpy.clip(mask.sum(axis=1), 1e-9, None)
        norms = numpy.linalg.norm(embeddings, axis=1, keepdims=True)
        return (embeddings / numpy.clip(norms, 1e-12, None)).tolist()


class RagIndex:
    """A persistent ChromaDB-backed knowledge base.

    One :class:`RagIndex` == one named knowledge base on disk under ``kb_dir()``.
    """

    def __init__(self, name: str, config: RagConfig | None = None) -> None:
        self.name = name
        self.config = config or RagConfig()
        self.path = kb_dir() / _storage_name(name)
        self._embedder = Embedder(self.config.embedding_model, provider=self.config.onnx_provider)
        self._client = None
        self._collection = None

    # -- storage ------------------------------------------------------------------
    def _coll(self):
        if self._collection is None:
            require_rag_dependencies()
            import chromadb  # type: ignore

            self._client = chromadb.PersistentClient(path=str(self.path))
            collection = self._client.get_or_create_collection(
                name=_collection_name(self.name),
                metadata={"hnsw:space": "cosine", "embedding_model": self.config.embedding_model},
            )
            metadata = collection.metadata or {}
            stored_model = metadata.get("embedding_model")
            if (
                collection.count() > 0
                and stored_model
                and stored_model != self.config.embedding_model
            ):
                raise ValueError(
                    f"Knowledge base '{self.name}' uses embedding model '{stored_model}', "
                    f"not '{self.config.embedding_model}'. Select the original model or "
                    "build a new knowledge base."
                )
            self._collection = collection
        return self._collection

    def count(self) -> int:
        return self._coll().count()

    # -- ingestion ----------------------------------------------------------------
    def add_pdf(self, pdf_path: str | Path, progress=None) -> int:
        """Extract, chunk, embed and store one PDF. Returns the number of chunks added.

        ``progress`` is an optional callable ``(done, total, message)`` for GUI updates.
        """
        doc: ExtractedDoc = extract_pdf(pdf_path)
        return self.add_document(doc, progress=progress)

    def add_document(self, doc: ExtractedDoc, progress=None) -> int:
        """Chunk, embed, and store a previously extracted document."""
        chunks: list[Chunk] = chunk_document(doc, self.config.chunk)
        return self.add_chunks(chunks, progress=progress)

    def add_chunks(self, chunks: list[Chunk], progress=None) -> int:
        """Embed and store precomputed chunks from one or more documents."""
        if not chunks:
            return 0
        coll = self._coll()
        batch = 64
        total = len(chunks)
        existing_ids: dict[str, set[str]] = {}
        new_ids: dict[str, set[str]] = {}
        for chunk in chunks:
            new_ids.setdefault(chunk.source, set()).add(_chunk_id(chunk))
        for source in new_ids:
            existing = coll.get(where={"source": source}, include=[])
            existing_ids[source] = set(existing.get("ids", []))

        for start in range(0, total, batch):
            window = chunks[start : start + batch]
            embeddings = self._embedder.encode([c.text for c in window])
            coll.upsert(
                ids=[_chunk_id(c) for c in window],
                documents=[c.text for c in window],
                embeddings=embeddings,
                metadatas=[
                    {"source": c.source, "page_number": c.page_number, "index": c.index}
                    for c in window
                ],
            )
            if progress:
                progress(min(start + batch, total), total, f"Embedded {min(start + batch, total)}/{total} chunks")

        for source, old_ids in existing_ids.items():
            stale_ids = sorted(old_ids - new_ids[source])
            if stale_ids:
                coll.delete(ids=stale_ids)
        return total

    # Kept for compatibility with integrations that used the original internal helper.
    def _add_chunks(self, chunks: list[Chunk], progress=None) -> int:
        return self.add_chunks(chunks, progress=progress)

    # -- retrieval ----------------------------------------------------------------
    def retrieve(self, query: str, top_k: int | None = None) -> list[Retrieved]:
        top_k = max(1, int(top_k or self.config.top_k))
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
        threshold = self.config.score_threshold
        if threshold > 0:
            out = [r for r in out if r.score >= threshold]
        return out

    def build_prompt(
        self,
        query: str,
        contexts: list[Retrieved],
        history: list[dict] | None = None,
        system_prefix: str = "",
        max_turns: int = 3,
    ) -> list[dict]:
        """Construct chat `messages` grounding the model in retrieved context.

        Prior ``history`` turns are inserted between the grounding system message and
        the current question so follow-ups stay conversational (capped to the last
        ``max_turns`` turns). An optional ``system_prefix`` (user style/behaviour
        guidance) is prepended to — never replaces — the grounding/citation rules.
        """
        context_block = "\n\n".join(
            f"[{i + 1}] ({c.citation})\n{c.text}" for i, c in enumerate(contexts)
        )
        grounding = (
            "You are a helpful assistant. Answer the user's question using ONLY the "
            "context below. Cite sources inline using the bracketed numbers, e.g. [1]. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context_block}"
        )
        prefix = system_prefix.strip()
        system = f"{prefix}\n\n{grounding}" if prefix else grounding
        messages: list[dict] = [{"role": "system", "content": system}]
        messages.extend(_recent_turns(history, max_turns))
        messages.append({"role": "user", "content": query})
        return messages

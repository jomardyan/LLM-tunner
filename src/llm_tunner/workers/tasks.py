"""Concrete task functions run by :class:`~llm_tunner.workers.base.Worker`.

Each function takes a ``signals`` kwarg (a :class:`WorkerSignals`) to stream progress
back to the GUI. They wrap the ``core`` modules; keeping them thin means the heavy
imports happen on the worker thread, not at GUI startup.
"""

from __future__ import annotations

from pathlib import Path


def _progress_cb(signals):
    def cb(done: int, total: int, message: str = "") -> None:
        signals.progress.emit(int(done), int(total), message)

    return cb


def ingest_pdf_task(pdf_path: str, *, signals) -> dict:
    """Extract text from a PDF and return a preview + page count."""
    from ..core.pdf import extract_pdf

    signals.log.emit(f"Extracting {Path(pdf_path).name}…")
    doc = extract_pdf(pdf_path)
    mode = "OCR" if doc.used_ocr else "native text"
    signals.progress.emit(
        1,
        1,
        f"Extracted {doc.num_pages} page(s) via {doc.backend} ({mode})",
    )
    preview = doc.full_text[:2000]
    return {
        "source": pdf_path,
        "pages": doc.num_pages,
        "backend": doc.backend,
        "document_type": doc.document_type,
        "used_ocr": doc.used_ocr,
        "preview": preview,
    }


def huggingface_login_task(token: str, *, signals) -> dict:
    """Validate and save a Hugging Face token without exposing it to app settings."""
    from ..core.models import login_huggingface

    login_huggingface(token)
    return {"authenticated": True}


def build_kb_task(kb_name: str, pdf_paths: list[str], embedding_model: str, *, signals) -> dict:
    """Embed and index a list of PDFs into a named knowledge base."""
    from ..config import RagConfig
    from ..core.rag import RagIndex, require_rag_dependencies

    require_rag_dependencies()
    config = RagConfig(embedding_model=embedding_model)
    index = RagIndex(kb_name, config=config)
    total_chunks = 0
    for i, pdf in enumerate(pdf_paths, start=1):
        signals.log.emit(f"Indexing {Path(pdf).name} ({i}/{len(pdf_paths)})…")
        total_chunks += index.add_pdf(pdf, progress=_progress_cb(signals))
    return {"kb": kb_name, "chunks": total_chunks, "count": index.count()}


def rag_query_task(model_id: str, adapter_path: str | None, kb_name: str, query: str,
                   embedding_model: str, top_k: int, *, signals) -> dict:
    """Answer a query against a knowledge base with citations."""
    from ..config import RagConfig
    from ..core.inference import ChatModel
    from ..core.rag import RagIndex, require_rag_dependencies

    require_rag_dependencies()
    signals.log.emit("Loading model…")
    model = ChatModel(model_id, adapter_path=adapter_path)
    index = RagIndex(kb_name, config=RagConfig(embedding_model=embedding_model, top_k=top_k))
    signals.log.emit("Retrieving and generating…")
    result = model.rag_answer(query, index, top_k=top_k)
    return {"answer": result.answer, "sources": result.formatted_sources()}


def plain_chat_task(model_id: str, adapter_path: str | None, query: str,
                    history: list[dict], *, signals) -> dict:
    """Chat with the model without retrieval."""
    from ..core.inference import ChatModel

    signals.log.emit("Loading model…")
    model = ChatModel(model_id, adapter_path=adapter_path)
    signals.log.emit("Generating…")
    answer = model.chat(query, history=history)
    return {"answer": answer, "sources": ""}


def generate_dataset_task(kb_pdfs: list[str], use_llm_model: str | None, *, signals) -> dict:
    """Build a fine-tuning dataset (Q&A pairs) from PDFs and save it as JSONL."""
    from ..core.chunking import chunk_document
    from ..core.dataset import save_jsonl, to_messages
    from ..core.pdf import extract_pdf
    from ..core.qa_gen import generate_qa, heuristic_generator, llm_generator

    generator = heuristic_generator
    if use_llm_model:
        from ..core.inference import ChatModel

        signals.log.emit(f"Loading {use_llm_model} for Q&A generation…")
        cm = ChatModel(use_llm_model)
        generator = llm_generator(cm)

    all_chunks = []
    for pdf in kb_pdfs:
        signals.log.emit(f"Chunking {Path(pdf).name}…")
        all_chunks.extend(chunk_document(extract_pdf(pdf)))

    pairs = generate_qa(all_chunks, generator=generator, progress=_progress_cb(signals))
    rows = to_messages(pairs)
    path = save_jsonl(rows, "finetune_dataset")
    return {"pairs": len(pairs), "path": str(path), "rows": rows}


def finetune_task(model_id: str, dataset_rows: list[dict], output_name: str, handle, *, signals):
    """Run QLoRA fine-tuning, draining metrics from ``handle`` to the UI.

    Metric draining runs in a small helper thread so loss points stream live while
    ``run_finetune`` blocks on training.
    """
    import threading

    from ..core.training import run_finetune

    def _drain():
        while not done.is_set() or not handle.metrics.empty():
            try:
                m = handle.metrics.get(timeout=0.2)
            except Exception:
                continue
            signals.metric.emit(m)

    done = threading.Event()
    drainer = threading.Thread(target=_drain, daemon=True)
    drainer.start()
    try:
        signals.log.emit(f"Starting fine-tune of {model_id}…")
        result = run_finetune(model_id, dataset_rows, output_name=output_name, handle=handle)
    finally:
        done.set()
        drainer.join(timeout=2)
    return {
        "adapter_path": result.adapter_path,
        "final_loss": result.final_loss,
        "steps": result.steps,
        "used_4bit": result.used_4bit,
    }

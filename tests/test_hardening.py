"""Backend input-validation guards. Torch-free: the guards run before any heavy import."""

from __future__ import annotations

import pytest


class _NullSignals:
    class _Channel:
        def emit(self, *args, **kwargs) -> None:
            pass

    def __init__(self) -> None:
        self.progress = self._Channel()
        self.log = self._Channel()
        self.metric = self._Channel()


def test_build_kb_task_rejects_empty_inputs():
    from llm_tunner.workers.tasks import build_kb_task

    sig = _NullSignals()
    with pytest.raises(ValueError, match="name is required"):
        build_kb_task("  ", ["a.pdf"], "embed", signals=sig)
    with pytest.raises(ValueError, match="No PDF files"):
        build_kb_task("kb", [], "embed", signals=sig)


def test_rag_query_task_rejects_empty_query_or_kb():
    from llm_tunner.workers.tasks import rag_query_task

    sig = _NullSignals()
    with pytest.raises(ValueError, match="question"):
        rag_query_task("m", None, "kb", "   ", "embed", 4, signals=sig)
    with pytest.raises(ValueError, match="knowledge base"):
        rag_query_task("m", None, "", "question", "embed", 4, signals=sig)


def test_plain_chat_task_rejects_empty_query():
    from llm_tunner.workers.tasks import plain_chat_task

    with pytest.raises(ValueError, match="question"):
        plain_chat_task("m", None, "", [], signals=_NullSignals())


def test_generate_dataset_task_rejects_empty_pdfs():
    from llm_tunner.workers.tasks import generate_dataset_task

    with pytest.raises(ValueError, match="No PDF"):
        generate_dataset_task([], None, signals=_NullSignals())


def test_finetune_task_rejects_empty_dataset():
    from llm_tunner.workers.tasks import finetune_task

    with pytest.raises(ValueError, match="No training data"):
        finetune_task("m", [], "out", object(), signals=_NullSignals())


def test_run_finetune_rejects_empty_dataset():
    from llm_tunner.core.training import run_finetune

    with pytest.raises(ValueError, match="empty dataset"):
        run_finetune("m", [])


def test_embedder_encode_empty_returns_empty():
    from llm_tunner.core.rag import Embedder

    # Returns before loading any model / importing numpy.
    assert Embedder("any-model").encode([]) == []


def test_extract_pdf_rejects_missing_directory_and_empty(tmp_path):
    from llm_tunner.core.pdf import extract_pdf

    with pytest.raises(FileNotFoundError):
        extract_pdf(tmp_path / "does-not-exist.pdf")
    with pytest.raises(FileNotFoundError):
        extract_pdf(tmp_path)  # a directory is not a file
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        extract_pdf(empty)

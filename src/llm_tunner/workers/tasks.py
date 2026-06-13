"""Concrete task functions run by :class:`~llm_tunner.workers.base.Worker`.

Each function takes a ``signals`` kwarg (a :class:`WorkerSignals`) to stream progress
back to the GUI. They wrap the ``core`` modules; keeping them thin means the heavy
imports happen on the worker thread, not at GUI startup.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path


def _progress_cb(signals):
    def cb(done: int, total: int, message: str = "") -> None:
        signals.progress.emit(int(done), int(total), message)

    return cb


def _generation_config(temperature: float, top_p: float, max_new_tokens: int):
    """Build a GenerationConfig from primitive params (greedy when temperature is 0)."""
    from ..core.inference import GenerationConfig

    temperature = float(temperature)
    return GenerationConfig(
        max_new_tokens=int(max_new_tokens),
        temperature=temperature,
        top_p=float(top_p),
        do_sample=temperature > 0,
    )


def ingest_pdf_task(pdf_path: str, *, signals) -> dict:
    """Extract text from a PDF and return a preview + page count."""
    from ..core.pdf import extract_pdf

    started = time.perf_counter()
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
        "characters": len(doc.full_text),
        "words": len(doc.full_text.split()),
        "file_size_mb": Path(pdf_path).stat().st_size / (1024**2),
        "elapsed_seconds": time.perf_counter() - started,
    }


def runtime_metrics_task(*, signals) -> dict:
    """Collect lightweight GPU and storage metrics."""
    from ..core.metrics import collect_runtime_metrics

    return collect_runtime_metrics().as_dict()


def huggingface_login_task(token: str, *, signals) -> dict:
    """Validate and save a Hugging Face token without exposing it to app settings."""
    from ..core.models import login_huggingface

    login_huggingface(token)
    return {"authenticated": True}


def download_model_task(model_id: str, *, signals) -> dict:
    """Pre-download a model snapshot to the local Hugging Face cache."""
    from ..core.models import download

    started = time.perf_counter()
    signals.log.emit(f"Downloading {model_id}…")
    path = download(model_id, progress=_progress_cb(signals))
    return {
        "model_id": model_id,
        "path": path,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _build_kb(
    kb_name: str,
    pdf_paths: list[str],
    embedding_model: str,
    *,
    signals,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> dict:
    """Embed and index PDFs in the current process."""
    from ..config import ChunkConfig, RagConfig
    from ..core.chunking import chunk_document
    from ..core.pdf import extract_pdf
    from ..core.rag import RagIndex, require_rag_dependencies

    started = time.perf_counter()
    require_rag_dependencies()
    config = RagConfig(
        embedding_model=embedding_model,
        chunk=ChunkConfig(chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap)),
    )
    index = RagIndex(kb_name, config=config)
    total_chunks = 0
    total_pages = 0
    total_characters = 0
    native_documents = 0
    ocr_documents = 0
    all_chunks = []
    signals.progress.emit(0, 0, "Extracting and chunking documents")
    for i, pdf in enumerate(pdf_paths, start=1):
        signals.log.emit(f"Extracting {Path(pdf).name} ({i}/{len(pdf_paths)})…")
        doc = extract_pdf(pdf)
        total_pages += doc.num_pages
        total_characters += len(doc.full_text)
        if doc.used_ocr:
            ocr_documents += 1
        else:
            native_documents += 1
        signals.log.emit(
            f"Prepared {doc.num_pages} page(s) via "
            f"{'OCR' if doc.used_ocr else 'native text'}…"
        )
        all_chunks.extend(chunk_document(doc, config.chunk))

    total_chunks = len(all_chunks)
    signals.progress.emit(0, total_chunks, f"Embedding 0/{total_chunks} chunks")
    index.add_chunks(all_chunks, progress=_progress_cb(signals))
    elapsed = time.perf_counter() - started
    return {
        "kb": kb_name,
        "chunks": total_chunks,
        "count": index.count(),
        "documents": len(pdf_paths),
        "pages": total_pages,
        "characters": total_characters,
        "native_documents": native_documents,
        "ocr_documents": ocr_documents,
        "elapsed_seconds": elapsed,
        "chunks_per_second": total_chunks / elapsed if elapsed else 0.0,
    }


class _JsonLineSignal:
    """Persist signal arguments for a parent process to replay."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def emit(self, *values) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(values) + "\n")
            stream.flush()


class _ProcessSignals:
    def __init__(self, progress_path: Path, log_path: Path) -> None:
        self.progress = _JsonLineSignal(progress_path)
        self.log = _JsonLineSignal(log_path)


def _build_kb_child(
    kb_name: str,
    pdf_paths: list[str],
    embedding_model: str,
    progress_path: str,
    log_path: str,
    result_path: str,
    error_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> None:
    """Build a knowledge base on a process main thread for native Arrow safety."""
    from ..diagnostics import configure_crash_diagnostics, configure_logging

    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    configure_crash_diagnostics()
    configure_logging()
    signals = _ProcessSignals(Path(progress_path), Path(log_path))
    try:
        result = _build_kb(
            kb_name,
            pdf_paths,
            embedding_model,
            signals=signals,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        _write_json(Path(result_path), result)
    except BaseException as exc:  # noqa: BLE001 - serialize child failures to the GUI
        _write_json(
            Path(error_path),
            {
                "kind": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def _replay_json_lines(path: Path, offset: int, emit) -> int:
    if not path.exists():
        return offset
    with path.open("r", encoding="utf-8") as stream:
        stream.seek(offset)
        for line in stream:
            emit(*json.loads(line))
        return stream.tell()


def _build_kb_isolated(
    kb_name: str,
    pdf_paths: list[str],
    embedding_model: str,
    *,
    signals,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> dict:
    import multiprocessing

    from ..config import data_dir

    runtime_dir = data_dir() / "runtime" / f"knowledge-{uuid.uuid4().hex}"
    runtime_dir.mkdir(parents=True)
    progress_path = runtime_dir / "progress.jsonl"
    log_path = runtime_dir / "log.jsonl"
    result_path = runtime_dir / "result.json"
    error_path = runtime_dir / "error.json"
    process = multiprocessing.get_context("spawn").Process(
        target=_build_kb_child,
        args=(
            kb_name,
            pdf_paths,
            embedding_model,
            str(progress_path),
            str(log_path),
            str(result_path),
            str(error_path),
            int(chunk_size),
            int(chunk_overlap),
        ),
        name="llm-tunner-knowledge-builder",
    )
    process.start()
    progress_offset = 0
    log_offset = 0
    try:
        while process.is_alive():
            progress_offset = _replay_json_lines(
                progress_path, progress_offset, signals.progress.emit
            )
            log_offset = _replay_json_lines(log_path, log_offset, signals.log.emit)
            process.join(timeout=0.2)
        progress_offset = _replay_json_lines(
            progress_path, progress_offset, signals.progress.emit
        )
        log_offset = _replay_json_lines(log_path, log_offset, signals.log.emit)
        process.join()

        if result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))
        if error_path.exists():
            error = json.loads(error_path.read_text(encoding="utf-8"))
            raise RuntimeError(
                f"{error['kind']}: {error['message']}\n\nChild traceback:\n{error['traceback']}"
            )
        raise RuntimeError(
            f"Knowledge-base process exited with code {process.exitcode} before producing "
            "a result. See ~/.llm-tunner/logs/native-crash.log."
        )
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        shutil.rmtree(runtime_dir, ignore_errors=True)


def build_kb_task(kb_name: str, pdf_paths: list[str], embedding_model: str, *, signals,
                  chunk_size: int = 512, chunk_overlap: int = 64) -> dict:
    """Embed and index PDFs, isolating Windows native ML imports from Qt threads."""
    if sys.platform == "win32":
        return _build_kb_isolated(
            kb_name,
            pdf_paths,
            embedding_model,
            signals=signals,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    return _build_kb(
        kb_name, pdf_paths, embedding_model, signals=signals,
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )


def rag_query_task(model_id: str, adapter_path: str | None, kb_name: str, query: str,
                   embedding_model: str, top_k: int, *, signals,
                   history: list[dict] | None = None, system_prompt: str = "",
                   temperature: float = 0.7, top_p: float = 0.9,
                   max_new_tokens: int = 512, score_threshold: float = 0.0) -> dict:
    """Answer a query against a knowledge base with citations.

    ``history`` enables multi-turn (conversational) retrieval and generation; the
    generation/retrieval knobs are forwarded from the user's settings.
    """
    from ..config import RagConfig
    from ..core.inference import ChatModel
    from ..core.rag import RagIndex, require_rag_dependencies

    started = time.perf_counter()
    require_rag_dependencies()
    signals.log.emit("Loading model…")
    model = ChatModel(model_id, adapter_path=adapter_path)
    index = RagIndex(kb_name, config=RagConfig(
        embedding_model=embedding_model, top_k=top_k, score_threshold=score_threshold
    ))
    signals.log.emit("Retrieving and generating…")
    result = model.rag_answer(
        query, index, top_k=top_k,
        config=_generation_config(temperature, top_p, max_new_tokens),
        history=history, system_prompt=system_prompt,
    )
    return {
        "answer": result.answer,
        "sources": result.formatted_sources(),
        "elapsed_seconds": time.perf_counter() - started,
        "contexts": len(result.contexts),
        "response_words": len(result.answer.split()),
    }


def plain_chat_task(model_id: str, adapter_path: str | None, query: str,
                    history: list[dict], *, signals, system_prompt: str = "",
                    temperature: float = 0.7, top_p: float = 0.9,
                    max_new_tokens: int = 512) -> dict:
    """Chat with the model without retrieval."""
    from ..core.inference import ChatModel

    started = time.perf_counter()
    signals.log.emit("Loading model…")
    model = ChatModel(model_id, adapter_path=adapter_path)
    signals.log.emit("Generating…")
    answer = model.chat(
        query, history=history,
        config=_generation_config(temperature, top_p, max_new_tokens),
        system_prompt=system_prompt,
    )
    return {
        "answer": answer,
        "sources": "",
        "elapsed_seconds": time.perf_counter() - started,
        "contexts": 0,
        "response_words": len(answer.split()),
    }


def generate_dataset_task(
    kb_pdfs: list[str],
    use_llm_model: str | None,
    handle=None,
    *,
    signals,
) -> dict:
    """Build a fine-tuning dataset (Q&A pairs) from PDFs and save it as JSONL."""
    from ..core.cancellation import CancelHandle
    from ..core.chunking import chunk_document
    from ..core.dataset import save_jsonl, to_messages
    from ..core.pdf import extract_pdf
    from ..core.qa_gen import generate_qa, heuristic_generator, llm_generator

    started = time.perf_counter()
    handle = handle or CancelHandle()
    generator = heuristic_generator
    if use_llm_model:
        from ..core.inference import ChatModel

        signals.log.emit(f"Loading {use_llm_model} for Q&A generation…")
        cm = ChatModel(use_llm_model)
        generator = llm_generator(cm)

    all_chunks = []
    for index, pdf in enumerate(kb_pdfs, start=1):
        if handle.is_stopped():
            return {"cancelled": True}
        signals.log.emit(f"Chunking {Path(pdf).name} ({index}/{len(kb_pdfs)})…")
        all_chunks.extend(chunk_document(extract_pdf(pdf)))

    if handle.is_stopped():
        return {"cancelled": True}
    pairs = generate_qa(
        all_chunks,
        generator=generator,
        progress=_progress_cb(signals),
        should_stop=handle.is_stopped,
    )
    if handle.is_stopped():
        return {"cancelled": True}
    rows = to_messages(pairs)
    path = save_jsonl(rows, "finetune_dataset")
    return {
        "pairs": len(pairs),
        "path": str(path),
        "rows": rows,
        "chunks": len(all_chunks),
        "documents": len(kb_pdfs),
        "elapsed_seconds": time.perf_counter() - started,
    }


class _FileMetricSink:
    """Append child-process training metrics as JSON lines."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def put(self, metric) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(metric)) + "\n")
            stream.flush()


class _FileStopEvent:
    """Expose the ``threading.Event`` interface expected by the trainer callback."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def is_set(self) -> bool:
        return self.path.exists()


class _ProcessTrainHandle:
    def __init__(self, metrics_path: Path, stop_path: Path) -> None:
        self.metrics = _FileMetricSink(metrics_path)
        self.stop_event = _FileStopEvent(stop_path)


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)


def _train_config(overrides: dict | None):
    """Build a TrainConfig from primitive UI overrides (unset keys keep defaults)."""
    from ..config import LoraConfig, TrainConfig

    overrides = overrides or {}
    lora = LoraConfig(
        r=int(overrides.get("lora_r", LoraConfig.r)),
        lora_alpha=int(overrides.get("lora_alpha", LoraConfig.lora_alpha)),
        lora_dropout=float(overrides.get("lora_dropout", LoraConfig.lora_dropout)),
    )
    return TrainConfig(
        learning_rate=float(overrides.get("learning_rate", TrainConfig.learning_rate)),
        num_train_epochs=float(overrides.get("num_train_epochs", TrainConfig.num_train_epochs)),
        per_device_train_batch_size=int(
            overrides.get("per_device_train_batch_size", TrainConfig.per_device_train_batch_size)
        ),
        gradient_accumulation_steps=int(
            overrides.get("gradient_accumulation_steps", TrainConfig.gradient_accumulation_steps)
        ),
        max_seq_length=int(overrides.get("max_seq_length", TrainConfig.max_seq_length)),
        lora=lora,
    )


def _finetune_child(
    model_id: str,
    dataset_rows: list[dict],
    output_name: str,
    metrics_path: str,
    stop_path: str,
    result_path: str,
    error_path: str,
    train_overrides: dict | None = None,
) -> None:
    """Run the native ML stack outside the Qt process on Windows."""
    from ..core.training import run_finetune
    from ..diagnostics import configure_crash_diagnostics, configure_logging

    configure_crash_diagnostics()
    configure_logging()
    started = time.perf_counter()
    metrics_file = Path(metrics_path)
    stop_file = Path(stop_path)
    result_file = Path(result_path)
    error_file = Path(error_path)
    handle = _ProcessTrainHandle(metrics_file, stop_file)

    def complete(result) -> None:
        import torch  # type: ignore

        elapsed = time.perf_counter() - started
        peak_vram_mb = (
            torch.cuda.max_memory_allocated() / (1024**2)
            if torch.cuda.is_available()
            else 0.0
        )
        _write_json(
            result_file,
            {
                "adapter_path": result.adapter_path,
                "final_loss": result.final_loss,
                "steps": result.steps,
                "used_4bit": result.used_4bit,
                "elapsed_seconds": elapsed,
                "steps_per_second": result.steps / elapsed if elapsed else 0.0,
                "peak_vram_mb": peak_vram_mb,
                "dataset_rows": len(dataset_rows),
            },
        )
        # bitsandbytes/PyTorch can access-violate while destroying a completed
        # QLoRA graph on Windows. This process is dedicated to one training run,
        # so bypass native teardown after all durable output has been written.
        os._exit(0)

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        run_finetune(
            model_id,
            dataset_rows,
            config=_train_config(train_overrides),
            output_name=output_name,
            handle=handle,
            completion_callback=complete,
        )
    except BaseException as exc:  # noqa: BLE001 - serialize child failures to the GUI
        _write_json(
            error_file,
            {
                "kind": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        os._exit(1)


def _emit_process_metrics(metrics_path: Path, offset: int, signals) -> int:
    if not metrics_path.exists():
        return offset
    from ..core.training import TrainMetric

    with metrics_path.open("r", encoding="utf-8") as stream:
        stream.seek(offset)
        for line in stream:
            payload = json.loads(line)
            signals.metric.emit(TrainMetric(**payload))
        return stream.tell()


def _finetune_task_isolated(
    model_id: str,
    dataset_rows: list[dict],
    output_name: str,
    handle,
    signals,
    train_overrides: dict | None = None,
) -> dict:
    import multiprocessing

    from ..config import data_dir

    runtime_dir = data_dir() / "runtime" / f"training-{uuid.uuid4().hex}"
    runtime_dir.mkdir(parents=True)
    metrics_path = runtime_dir / "metrics.jsonl"
    stop_path = runtime_dir / "stop"
    result_path = runtime_dir / "result.json"
    error_path = runtime_dir / "error.json"
    process = multiprocessing.get_context("spawn").Process(
        target=_finetune_child,
        args=(
            model_id,
            dataset_rows,
            output_name,
            str(metrics_path),
            str(stop_path),
            str(result_path),
            str(error_path),
            train_overrides,
        ),
        name="llm-tunner-training",
    )
    process.start()
    offset = 0
    try:
        while process.is_alive():
            offset = _emit_process_metrics(metrics_path, offset, signals)
            if handle.stop_event.is_set() and not stop_path.exists():
                stop_path.touch()
            process.join(timeout=0.2)
        offset = _emit_process_metrics(metrics_path, offset, signals)
        process.join()

        if result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))
        if error_path.exists():
            error = json.loads(error_path.read_text(encoding="utf-8"))
            raise RuntimeError(
                f"{error['kind']}: {error['message']}\n\nChild traceback:\n{error['traceback']}"
            )
        raise RuntimeError(
            f"Fine-tuning process exited with code {process.exitcode} before producing "
            "a result. See ~/.llm-tunner/logs/native-crash.log."
        )
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        shutil.rmtree(runtime_dir, ignore_errors=True)


def _finetune_task_in_process(
    model_id: str,
    dataset_rows: list[dict],
    output_name: str,
    handle,
    signals,
    train_overrides: dict | None = None,
) -> dict:
    """Run training directly on platforms without the Windows teardown issue."""
    import threading

    import torch  # type: ignore

    from ..core.training import run_finetune

    def drain() -> None:
        while not done.is_set() or not handle.metrics.empty():
            try:
                metric = handle.metrics.get(timeout=0.2)
            except Exception:
                continue
            signals.metric.emit(metric)

    done = threading.Event()
    drainer = threading.Thread(target=drain, daemon=True)
    drainer.start()
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        result = run_finetune(
            model_id, dataset_rows,
            config=_train_config(train_overrides),
            output_name=output_name, handle=handle,
        )
    finally:
        done.set()
        drainer.join(timeout=2)
    elapsed = time.perf_counter() - started
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    )
    return {
        "adapter_path": result.adapter_path,
        "final_loss": result.final_loss,
        "steps": result.steps,
        "used_4bit": result.used_4bit,
        "elapsed_seconds": elapsed,
        "steps_per_second": result.steps / elapsed if elapsed else 0.0,
        "peak_vram_mb": peak_vram_mb,
        "dataset_rows": len(dataset_rows),
    }


def finetune_task(model_id: str, dataset_rows: list[dict], output_name: str, handle, *, signals,
                  train_overrides: dict | None = None):
    """Run fine-tuning with live metrics and cooperative cancellation.

    Windows uses a dedicated process because CUDA/bitsandbytes teardown can terminate
    the host process after a successful QLoRA save. ``train_overrides`` carries the
    user's hyperparameters (LR, epochs, batch, LoRA rank, …); unset keys keep defaults.
    """
    signals.log.emit(f"Starting fine-tune of {model_id}…")
    if sys.platform == "win32":
        return _finetune_task_isolated(
            model_id, dataset_rows, output_name, handle, signals, train_overrides
        )
    return _finetune_task_in_process(
        model_id, dataset_rows, output_name, handle, signals, train_overrides
    )

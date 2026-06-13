from __future__ import annotations

import json
import queue
import threading

from llm_tunner.core.training import TrainMetric
from llm_tunner.workers import tasks
from llm_tunner.workers.tasks import _emit_process_metrics, _replay_json_lines, _write_json


class _MetricSignal:
    def __init__(self) -> None:
        self.values = []

    def emit(self, value) -> None:
        self.values.append(value)


class _Signals:
    def __init__(self) -> None:
        self.metric = _MetricSignal()


def test_write_json_replaces_result_atomically(tmp_path):
    path = tmp_path / "result.json"
    _write_json(path, {"steps": 4})

    assert json.loads(path.read_text(encoding="utf-8")) == {"steps": 4}
    assert not path.with_suffix(".tmp").exists()


def test_process_metrics_are_rehydrated(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps(
            {
                "step": 2,
                "loss": 1.5,
                "learning_rate": 0.0001,
                "epoch": 0.5,
                "grad_norm": 3.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    signals = _Signals()

    offset = _emit_process_metrics(path, 0, signals)

    assert offset == path.stat().st_size
    assert signals.metric.values == [
        TrainMetric(
            step=2,
            loss=1.5,
            learning_rate=0.0001,
            epoch=0.5,
            grad_norm=3.0,
        )
    ]


def test_process_metrics_skip_malformed_lines(tmp_path):
    path = tmp_path / "metrics.jsonl"
    good = {
        "step": 1,
        "loss": 2.0,
        "learning_rate": 0.0002,
        "epoch": 0.25,
        "grad_norm": 1.0,
    }
    # One valid metric line, one corrupted/partial line with unexpected keys.
    path.write_text(
        json.dumps(good) + "\n" + json.dumps({"unexpected": "field"}) + "\n",
        encoding="utf-8",
    )
    signals = _Signals()

    # Must not raise even though the second line cannot build a TrainMetric.
    offset = _emit_process_metrics(path, 0, signals)

    assert offset == path.stat().st_size
    assert signals.metric.values == [
        TrainMetric(step=1, loss=2.0, learning_rate=0.0002, epoch=0.25, grad_norm=1.0)
    ]


def test_emit_process_metrics_returns_offset_when_no_new_lines(tmp_path):
    path = tmp_path / "metrics.jsonl"
    line = (
        json.dumps(
            {
                "step": 3,
                "loss": 0.9,
                "learning_rate": 0.0001,
                "epoch": 1.0,
                "grad_norm": None,
            }
        )
        + "\n"
    )
    path.write_text(line, encoding="utf-8")
    signals = _Signals()

    offset = _emit_process_metrics(path, 0, signals)
    assert offset == path.stat().st_size
    assert len(signals.metric.values) == 1

    # A second poll starting at the end of file emits nothing and keeps the offset.
    again = _emit_process_metrics(path, offset, signals)
    assert again == offset
    assert len(signals.metric.values) == 1


def test_drain_handles_empty_queue_and_emits_metrics():
    """Mirror the in-process drain loop: skip queue.Empty, emit real metrics."""
    metrics: queue.Queue = queue.Queue()
    emitted: list = []
    done = threading.Event()

    metric = TrainMetric(step=5, loss=0.5, learning_rate=0.0001, epoch=2.0, grad_norm=2.0)
    metrics.put(metric)

    # Replicate the drain() closure body in _finetune_task_in_process.
    def drain() -> None:
        while not done.is_set() or not metrics.empty():
            try:
                value = metrics.get(timeout=0.05)
            except queue.Empty:
                continue
            emitted.append(value)

    drainer = threading.Thread(target=drain, daemon=True)
    drainer.start()
    done.set()
    drainer.join(timeout=2)

    assert not drainer.is_alive()
    assert emitted == [metric]


def test_json_line_signals_are_replayed(tmp_path):
    path = tmp_path / "progress.jsonl"
    path.write_text("[2, 10, \"Embedded 2/10 chunks\"]\n", encoding="utf-8")
    emitted = []

    offset = _replay_json_lines(path, 0, lambda *values: emitted.append(values))

    assert offset == path.stat().st_size
    assert emitted == [(2, 10, "Embedded 2/10 chunks")]


def test_knowledge_build_is_isolated_on_windows(monkeypatch):
    expected = {"kb": "docs", "chunks": 3}
    called = {}

    def fake_isolated(kb_name, pdf_paths, embedding_model, *, signals,
                      chunk_size=512, chunk_overlap=64, onnx_provider="auto"):
        called.update(
            kb_name=kb_name,
            pdf_paths=pdf_paths,
            embedding_model=embedding_model,
            signals=signals,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            onnx_provider=onnx_provider,
        )
        return expected

    signals = object()
    monkeypatch.setattr(tasks.sys, "platform", "win32")
    monkeypatch.setattr(tasks, "_build_kb_isolated", fake_isolated)

    result = tasks.build_kb_task(
        "docs", ["one.pdf"], "embedder", signals=signals,
        chunk_size=256, chunk_overlap=32, onnx_provider="directml",
    )

    assert result == expected
    assert called == {
        "kb_name": "docs",
        "pdf_paths": ["one.pdf"],
        "embedding_model": "embedder",
        "signals": signals,
        "chunk_size": 256,
        "chunk_overlap": 32,
        "onnx_provider": "directml",
    }

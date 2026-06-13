from __future__ import annotations

import json

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

    def fake_isolated(kb_name, pdf_paths, embedding_model, *, signals):
        called.update(
            kb_name=kb_name,
            pdf_paths=pdf_paths,
            embedding_model=embedding_model,
            signals=signals,
        )
        return expected

    signals = object()
    monkeypatch.setattr(tasks.sys, "platform", "win32")
    monkeypatch.setattr(tasks, "_build_kb_isolated", fake_isolated)

    result = tasks.build_kb_task("docs", ["one.pdf"], "embedder", signals=signals)

    assert result == expected
    assert called == {
        "kb_name": "docs",
        "pdf_paths": ["one.pdf"],
        "embedding_model": "embedder",
        "signals": signals,
    }

from __future__ import annotations

import importlib.util
import os

import pytest

_HAVE_GUI = all(importlib.util.find_spec(module) for module in ("PySide6", "pyqtgraph"))

pytestmark = pytest.mark.skipif(not _HAVE_GUI, reason="GUI dependencies not installed")


def test_main_window_constructs_offscreen():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from llm_tunner.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert window.windowTitle().startswith("LLM-tunner")
    assert window.navigation.count() == 5
    assert not window.metrics_panel.isHidden()
    window.close()
    app.processEvents()


def test_finetune_progress_callback_updates_progress_bar():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from llm_tunner.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.finetune_tab._on_progress(2, 5, "")

    assert window.finetune_tab.progress.maximum() == 5
    assert window.finetune_tab.progress.value() == 2

    window.close()
    app.processEvents()


def test_dataset_generation_enables_stop_immediately(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from llm_tunner.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.state.documents = ["one.pdf"]
    submitted = {}

    def fake_submit(fn, *args, **kwargs):
        submitted["fn"] = fn
        submitted["args"] = args
        submitted["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(window, "submit", fake_submit)
    window.finetune_tab._generate()

    assert window.finetune_tab.stop_btn.isEnabled()
    assert window.finetune_tab.stop_btn.text() == "Stop dataset generation"
    assert window.finetune_tab._active_phase == "dataset generation"
    assert submitted["args"][2] is window.finetune_tab._handle

    window.finetune_tab._stop()
    assert not window.finetune_tab.stop_btn.isEnabled()
    assert window.finetune_tab.stop_btn.text() == "Stopping…"

    window.close()
    app.processEvents()


def test_finetuning_enables_stop_immediately(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from llm_tunner.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.finetune_tab._dataset_rows = [{"messages": []}]
    monkeypatch.setattr(window, "submit", lambda *args, **kwargs: object())

    window.finetune_tab._train()

    assert window.finetune_tab.stop_btn.isEnabled()
    assert window.finetune_tab.stop_btn.text() == "Stop fine-tuning"
    assert window.finetune_tab.stop_btn.objectName() == "danger"
    assert window.finetune_tab._active_phase == "fine-tuning"

    window.close()
    app.processEvents()


def test_rag_progress_uses_aggregate_chunk_total():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from llm_tunner.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    window.rag_tab._on_progress(75, 100, "Embedding 75/100 chunks")

    assert window.rag_tab.progress.maximum() == 100
    assert window.rag_tab.progress.value() == 75
    assert window.rag_tab.progress.toolTip() == "Embedding 75/100 chunks"
    assert window.rag_tab.progress.objectName() == "workflowProgress"
    assert window.rag_tab.progress.minimumHeight() == 28

    window.rag_tab.build_btn.setEnabled(False)
    window.rag_tab.build_btn.setText("Building knowledge base…")
    window.rag_tab.progress.setVisible(True)
    window.rag_tab._on_build_finished()

    assert window.rag_tab.build_btn.isEnabled()
    assert window.rag_tab.build_btn.text() == "Build / update knowledge base"
    assert window.rag_tab.progress.isHidden()

    window.close()
    app.processEvents()


def test_documents_preview_uses_cached_metrics(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from llm_tunner.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.state.document_metrics["/some/file.pdf"] = {
        "backend": "pypdf",
        "used_ocr": False,
        "pages": 2,
        "document_type": "text",
        "preview": "hello",
        "source": "/some/file.pdf",
    }
    submitted = []
    monkeypatch.setattr(window, "submit", lambda *args, **kwargs: submitted.append((args, kwargs)))

    window.documents_tab._preview("/some/file.pdf")

    assert submitted == []
    assert "hello" in window.documents_tab.preview.toPlainText()

    window.close()
    app.processEvents()


def test_rag_chat_passes_history_and_generation_params(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from llm_tunner.main_window import MainWindow
    from llm_tunner.workers.tasks import rag_query_task

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.state.current_kb = "kb"
    window.state.base_model = "demo-model"
    window.chat_tab.use_rag.setChecked(True)
    # Seed a prior turn so the follow-up is conversational.
    window.chat_tab.history = [
        {"role": "user", "content": "Tell me about the Q3 plan"},
        {"role": "assistant", "content": "It covers hiring and budget."},
    ]
    captured = {}

    def fake_submit(fn, *args, **kwargs):
        captured["fn"] = fn
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(window, "submit", fake_submit)
    window.chat_tab.input.setText("What about its cost?")
    window.chat_tab._send()

    assert captured["fn"] is rag_query_task
    kwargs = captured["kwargs"]
    assert kwargs["history"] == window.chat_tab.history
    assert kwargs["temperature"] == window.settings.temperature
    assert kwargs["top_p"] == window.settings.top_p
    assert kwargs["max_new_tokens"] == window.settings.max_new_tokens
    assert "system_prompt" in kwargs
    assert "score_threshold" in kwargs
    assert kwargs["onnx_provider"] == window.settings.onnx_provider

    window.close()
    app.processEvents()


def test_finetune_passes_hyperparameter_overrides(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from llm_tunner.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.finetune_tab._dataset_rows = [{"messages": []}]
    window.finetune_tab.lora_r.setValue(8)
    window.finetune_tab.epochs.setValue(2.0)
    captured = {}

    def fake_submit(fn, *args, **kwargs):
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(window, "submit", fake_submit)
    window.finetune_tab._train()

    overrides = captured["kwargs"]["train_overrides"]
    assert overrides["lora_r"] == 8
    assert overrides["num_train_epochs"] == 2.0
    assert "learning_rate" in overrides

    window.close()
    app.processEvents()


def test_document_metrics_are_aggregated():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from llm_tunner.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.state.documents = ["one.pdf", "two.pdf"]
    window.state.document_metrics = {
        "one.pdf": {"pages": 3, "used_ocr": False},
        "two.pdf": {"pages": 2, "used_ocr": True},
    }
    window.update_document_metrics()

    assert window.metrics_panel._values["documents"].text() == "2"
    assert window.metrics_panel._values["pages"].text() == "5"
    assert window.metrics_panel._values["extraction"].text() == "1 native / 1 OCR"

    window.close()
    app.processEvents()

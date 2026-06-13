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

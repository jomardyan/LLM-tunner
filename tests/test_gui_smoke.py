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
    window.close()
    app.processEvents()

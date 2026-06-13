from __future__ import annotations

import importlib.util
import multiprocessing
import os

import pytest

from llm_tunner.app import _configure_windows_process, _validate_runtime

_HAVE_GUI = all(importlib.util.find_spec(module) for module in ("PySide6", "pyqtgraph"))


def test_process_configuration_calls_freeze_support(monkeypatch):
    called = False

    def fake_freeze_support():
        nonlocal called
        called = True

    monkeypatch.setattr(multiprocessing, "freeze_support", fake_freeze_support)
    _configure_windows_process()
    assert called


def test_runtime_validation_accepts_python_314(monkeypatch):
    monkeypatch.setattr("llm_tunner.app.sys.version_info", (3, 14))
    _validate_runtime()


def test_runtime_validation_rejects_stale_python(monkeypatch):
    monkeypatch.setattr("llm_tunner.app.sys.version_info", (3, 11))
    with pytest.raises(RuntimeError, match="requires Python 3.14"):
        _validate_runtime()


@pytest.mark.skipif(not _HAVE_GUI, reason="GUI dependencies not installed")
def test_close_event_stops_recurring_timers():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from llm_tunner.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert window._clock.isActive()
    assert window._runtime_timer.isActive()

    window.close()
    app.processEvents()

    assert not window._clock.isActive()
    assert not window._runtime_timer.isActive()

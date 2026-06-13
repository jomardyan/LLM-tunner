from __future__ import annotations

import multiprocessing

import pytest

from llm_tunner.app import _configure_windows_process, _validate_runtime


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

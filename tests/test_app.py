from __future__ import annotations

import multiprocessing

from llm_tunner.app import _configure_windows_process


def test_process_configuration_calls_freeze_support(monkeypatch):
    called = False

    def fake_freeze_support():
        nonlocal called
        called = True

    monkeypatch.setattr(multiprocessing, "freeze_support", fake_freeze_support)
    _configure_windows_process()
    assert called

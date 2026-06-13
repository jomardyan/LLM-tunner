"""Device detection must always return a sane, gated result."""

from __future__ import annotations

import pytest

from llm_tunner.core.device import (
    _bitsandbytes_available,
    detect_device,
    detect_device_isolated,
    recommended_models,
)


def test_detect_returns_known_kind():
    info = detect_device()
    assert info.kind in ("cuda", "mps", "cpu")
    assert isinstance(info.capability_banner(), str)
    # QLoRA is only ever allowed on CUDA + bitsandbytes.
    if info.supports_qlora:
        assert info.kind == "cuda" and info.bnb_available


def test_recommended_models_nonempty():
    assert recommended_models() == recommended_models()  # cached/stable
    assert len(recommended_models()) >= 1


def test_bitsandbytes_checks_installed_distribution(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    monkeypatch.setattr(
        "llm_tunner.core.device.importlib.metadata.version",
        lambda _name: (_ for _ in ()).throw(PackageNotFoundError),
    )
    assert not _bitsandbytes_available()

    monkeypatch.setattr(
        "llm_tunner.core.device.importlib.metadata.version",
        lambda _name: "0.49.2",
    )
    assert _bitsandbytes_available()


def test_detect_isolated_empty_output_raises_runtimeerror(monkeypatch):
    """Empty probe stdout must raise a clear RuntimeError, not an opaque IndexError."""

    class _FakeResult:
        stdout = "   \n  \n"
        stderr = ""
        returncode = 0

    monkeypatch.setattr(
        "llm_tunner.core.device.subprocess.run",
        lambda *args, **kwargs: _FakeResult(),
    )
    with pytest.raises(RuntimeError):
        detect_device_isolated()

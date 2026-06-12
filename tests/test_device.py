"""Device detection must always return a sane, gated result."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from llm_tunner.core.device import (
    _bitsandbytes_available,
    detect_device,
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


def test_bitsandbytes_requires_cuda_native_library(monkeypatch):
    package = ModuleType("bitsandbytes")
    extension = ModuleType("bitsandbytes.cextension")
    extension.lib = SimpleNamespace(compiled_with_cuda=False)
    monkeypatch.setitem(sys.modules, "bitsandbytes", package)
    monkeypatch.setitem(sys.modules, "bitsandbytes.cextension", extension)
    assert not _bitsandbytes_available()

    extension.lib = SimpleNamespace(compiled_with_cuda=True)
    assert _bitsandbytes_available()

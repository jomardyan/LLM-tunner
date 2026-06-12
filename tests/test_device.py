"""Device detection must always return a sane, gated result."""

from __future__ import annotations

from llm_tunner.core.device import detect_device, recommended_models


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

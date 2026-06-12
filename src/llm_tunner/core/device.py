"""Hardware detection and capability gating.

Every feature that depends on accelerators (QLoRA, embedding throughput, model size
recommendations) routes its decision through :func:`detect_device` /
:class:`DeviceInfo` so the rest of the app never imports torch eagerly or guesses at
capabilities. ``torch`` is imported lazily: the GUI must launch even when the heavy
ML stack is not installed.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class DeviceInfo:
    """A snapshot of the local accelerator situation.

    Attributes:
        kind: ``"cuda"``, ``"mps"`` or ``"cpu"``.
        name: Human-readable device name (e.g. GPU model).
        total_vram_gb: Total device memory in GiB, or ``None`` if unknown.
        torch_available: Whether ``torch`` could be imported at all.
        bnb_available: Whether 4-bit QLoRA (bitsandbytes) is usable here.
    """

    kind: str
    name: str
    total_vram_gb: float | None
    torch_available: bool
    bnb_available: bool

    @property
    def is_gpu(self) -> bool:
        return self.kind in ("cuda", "mps")

    @property
    def supports_qlora(self) -> bool:
        """True only when 4-bit QLoRA can actually run (CUDA + bitsandbytes)."""
        return self.kind == "cuda" and self.bnb_available

    def capability_banner(self) -> str:
        """A one-line, user-facing summary shown in the GUI status bar/banner."""
        if not self.torch_available:
            return (
                "PyTorch not installed — install with `pip install -e \".[all]\"`. "
                "GUI runs, but RAG and fine-tuning are unavailable."
            )
        if self.kind == "cuda":
            vram = f"{self.total_vram_gb:.0f} GB" if self.total_vram_gb else "unknown VRAM"
            ft = "QLoRA fine-tuning enabled" if self.bnb_available else (
                "fine-tuning enabled (bitsandbytes missing — 4-bit disabled)"
            )
            return f"CUDA GPU: {self.name} ({vram}). RAG + {ft}."
        if self.kind == "mps":
            return (
                f"Apple Silicon (MPS): {self.name}. RAG fully available; "
                "fine-tuning limited to small models (no 4-bit/bitsandbytes)."
            )
        return (
            "CPU only — no GPU detected. RAG fully available; fine-tuning limited to "
            "tiny models (slow). Consider a cloud GPU for real fine-tuning."
        )


@lru_cache(maxsize=1)
def _import_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception:  # pragma: no cover - environment dependent
        return None


def _bitsandbytes_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("bitsandbytes") is not None
    except Exception:  # pragma: no cover
        return False


@lru_cache(maxsize=1)
def detect_device() -> DeviceInfo:
    """Detect the best available device, preferring CUDA > MPS > CPU.

    Cached: hardware does not change during a session. Safe to call from any thread.
    """
    torch = _import_torch()
    if torch is None:
        return DeviceInfo(
            kind="cpu",
            name=platform.processor() or platform.machine() or "CPU",
            total_vram_gb=None,
            torch_available=False,
            bnb_available=False,
        )

    bnb = _bitsandbytes_available()

    # CUDA
    try:
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            return DeviceInfo(
                kind="cuda",
                name=props.name,
                total_vram_gb=props.total_memory / (1024**3),
                torch_available=True,
                bnb_available=bnb,
            )
    except Exception:  # pragma: no cover - driver quirks
        pass

    # Apple Silicon MPS
    try:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return DeviceInfo(
                kind="mps",
                name=f"Apple {platform.machine()}",
                total_vram_gb=None,
                torch_available=True,
                bnb_available=False,  # bitsandbytes is CUDA-only
            )
    except Exception:  # pragma: no cover
        pass

    return DeviceInfo(
        kind="cpu",
        name=platform.processor() or platform.machine() or "CPU",
        total_vram_gb=None,
        torch_available=True,
        bnb_available=False,
    )


def recommended_models(info: DeviceInfo | None = None) -> list[str]:
    """Suggest base models that comfortably fit the detected hardware."""
    info = info or detect_device()
    if info.kind == "cuda" and (info.total_vram_gb or 0) >= 12:
        return ["Qwen/Qwen2.5-7B-Instruct", "microsoft/Phi-4-mini-instruct", "Qwen/Qwen2.5-3B-Instruct"]
    if info.kind == "cuda":
        return ["Qwen/Qwen2.5-3B-Instruct", "microsoft/Phi-4-mini-instruct", "Qwen/Qwen2.5-1.5B-Instruct"]
    if info.kind == "mps":
        return ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]
    # CPU
    return ["Qwen/Qwen2.5-0.5B-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct"]

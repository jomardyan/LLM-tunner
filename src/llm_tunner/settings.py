"""Thin wrapper over QSettings for persisting user preferences.

Falls back to an in-memory dict when Qt is unavailable (e.g. headless unit tests),
so core logic can read settings without importing PySide6.
"""

from __future__ import annotations

from typing import Any

from .config import (
    APP_NAME,
    DEFAULT_BASE_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    ORG_NAME,
)

_DEFAULTS: dict[str, Any] = {
    "base_model": DEFAULT_BASE_MODEL,
    "embedding_model": DEFAULT_EMBEDDING_MODEL,
    "top_k": 4,
    "hf_cache_dir": "",
    "last_kb": "",
}


class _MemoryBackend:
    """Used when Qt is not importable (headless tests)."""

    def __init__(self) -> None:
        self._d: dict[str, Any] = {}

    def value(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default)

    def setValue(self, key: str, value: Any) -> None:  # noqa: N802 (Qt API name)
        self._d[key] = value


def _make_backend():
    try:
        from PySide6.QtCore import QSettings  # type: ignore

        return QSettings(ORG_NAME, APP_NAME)
    except Exception:
        return _MemoryBackend()


class Settings:
    """Typed accessor for persisted preferences."""

    def __init__(self) -> None:
        self._backend = _make_backend()

    def get(self, key: str, default: Any = None) -> Any:
        if default is None:
            default = _DEFAULTS.get(key)
        return self._backend.value(key, default)

    def set(self, key: str, value: Any) -> None:
        self._backend.setValue(key, value)

    # Convenience typed properties -------------------------------------------------
    @property
    def base_model(self) -> str:
        return str(self.get("base_model"))

    @base_model.setter
    def base_model(self, v: str) -> None:
        self.set("base_model", v)

    @property
    def embedding_model(self) -> str:
        return str(self.get("embedding_model"))

    @embedding_model.setter
    def embedding_model(self, v: str) -> None:
        self.set("embedding_model", v)

    @property
    def top_k(self) -> int:
        return int(self.get("top_k", 4))

    @top_k.setter
    def top_k(self, v: int) -> None:
        self.set("top_k", int(v))

    @property
    def last_kb(self) -> str:
        return str(self.get("last_kb"))

    @last_kb.setter
    def last_kb(self, v: str) -> None:
        self.set("last_kb", v)

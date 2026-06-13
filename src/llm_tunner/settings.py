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
    # Generation controls (Chat).
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 512,
    "system_prompt": "",
    # Retrieval / chunking (Knowledge base).
    "score_threshold": 0.0,
    "chunk_size": 512,
    "chunk_overlap": 64,
    # Fine-tuning hyperparameters.
    "learning_rate": 2e-4,
    "num_train_epochs": 1.0,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_seq_length": 1024,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
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

    # -- robust typed access (QSettings round-trips values as strings) -------------
    def _get_float(self, key: str) -> float:
        try:
            return float(self.get(key))
        except (TypeError, ValueError):
            return float(_DEFAULTS.get(key, 0.0))

    def _get_int(self, key: str) -> int:
        try:
            return int(self.get(key))
        except (TypeError, ValueError):
            return int(_DEFAULTS.get(key, 0))

    # Generation controls -----------------------------------------------------------
    @property
    def temperature(self) -> float:
        return self._get_float("temperature")

    @temperature.setter
    def temperature(self, v: float) -> None:
        self.set("temperature", float(v))

    @property
    def top_p(self) -> float:
        return self._get_float("top_p")

    @top_p.setter
    def top_p(self, v: float) -> None:
        self.set("top_p", float(v))

    @property
    def max_new_tokens(self) -> int:
        return self._get_int("max_new_tokens")

    @max_new_tokens.setter
    def max_new_tokens(self, v: int) -> None:
        self.set("max_new_tokens", int(v))

    @property
    def system_prompt(self) -> str:
        return str(self.get("system_prompt"))

    @system_prompt.setter
    def system_prompt(self, v: str) -> None:
        self.set("system_prompt", str(v))

    # Retrieval / chunking ----------------------------------------------------------
    @property
    def score_threshold(self) -> float:
        return self._get_float("score_threshold")

    @score_threshold.setter
    def score_threshold(self, v: float) -> None:
        self.set("score_threshold", float(v))

    @property
    def chunk_size(self) -> int:
        return self._get_int("chunk_size")

    @chunk_size.setter
    def chunk_size(self, v: int) -> None:
        self.set("chunk_size", int(v))

    @property
    def chunk_overlap(self) -> int:
        return self._get_int("chunk_overlap")

    @chunk_overlap.setter
    def chunk_overlap(self, v: int) -> None:
        self.set("chunk_overlap", int(v))

    # Fine-tuning hyperparameters ---------------------------------------------------
    @property
    def learning_rate(self) -> float:
        return self._get_float("learning_rate")

    @learning_rate.setter
    def learning_rate(self, v: float) -> None:
        self.set("learning_rate", float(v))

    @property
    def num_train_epochs(self) -> float:
        return self._get_float("num_train_epochs")

    @num_train_epochs.setter
    def num_train_epochs(self, v: float) -> None:
        self.set("num_train_epochs", float(v))

    @property
    def per_device_train_batch_size(self) -> int:
        return self._get_int("per_device_train_batch_size")

    @per_device_train_batch_size.setter
    def per_device_train_batch_size(self, v: int) -> None:
        self.set("per_device_train_batch_size", int(v))

    @property
    def gradient_accumulation_steps(self) -> int:
        return self._get_int("gradient_accumulation_steps")

    @gradient_accumulation_steps.setter
    def gradient_accumulation_steps(self, v: int) -> None:
        self.set("gradient_accumulation_steps", int(v))

    @property
    def max_seq_length(self) -> int:
        return self._get_int("max_seq_length")

    @max_seq_length.setter
    def max_seq_length(self, v: int) -> None:
        self.set("max_seq_length", int(v))

    @property
    def lora_r(self) -> int:
        return self._get_int("lora_r")

    @lora_r.setter
    def lora_r(self, v: int) -> None:
        self.set("lora_r", int(v))

    @property
    def lora_alpha(self) -> int:
        return self._get_int("lora_alpha")

    @lora_alpha.setter
    def lora_alpha(self, v: int) -> None:
        self.set("lora_alpha", int(v))

    @property
    def lora_dropout(self) -> float:
        return self._get_float("lora_dropout")

    @lora_dropout.setter
    def lora_dropout(self, v: float) -> None:
        self.set("lora_dropout", float(v))

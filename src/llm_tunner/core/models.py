"""Hugging Face model registry helpers and (optional) download with progress."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import MODEL_REGISTRY


@dataclass
class ModelEntry:
    model_id: str
    license: str
    size: str
    notes: str


def registry() -> list[ModelEntry]:
    return [ModelEntry(*row) for row in MODEL_REGISTRY]


def is_cached(model_id: str) -> bool:
    """Best-effort check whether a model is already in the local HF cache."""
    try:
        from huggingface_hub import scan_cache_dir  # type: ignore

        cache = scan_cache_dir()
        return any(repo.repo_id == model_id for repo in cache.repos)
    except Exception:
        return False


def download(model_id: str, progress=None) -> str:
    """Download a model snapshot to the HF cache. Returns the local path."""
    from huggingface_hub import snapshot_download  # type: ignore

    if progress:
        progress(0, 1, f"Downloading {model_id}…")
    path = snapshot_download(repo_id=model_id)
    if progress:
        progress(1, 1, "Download complete")
    return path

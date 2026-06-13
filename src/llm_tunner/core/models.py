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


def huggingface_authenticated() -> bool:
    """Return whether a stored login or HF_TOKEN is available."""
    try:
        from huggingface_hub import get_token  # type: ignore

        return bool(get_token())
    except Exception:
        return False


def login_huggingface(token: str) -> None:
    """Validate and save a Hub token using Hugging Face's standard token store."""
    if not token.strip():
        raise ValueError("Enter a Hugging Face access token.")

    from huggingface_hub import login  # type: ignore

    clean_token = token.strip()
    try:
        login(token=clean_token, add_to_git_credential=False)
    except Exception as exc:
        message = str(exc).replace(clean_token, "[redacted]")
        raise RuntimeError(message) from None


def download(model_id: str, progress=None) -> str:
    """Download a model snapshot to the HF cache. Returns the local path."""
    from huggingface_hub import snapshot_download  # type: ignore

    if progress:
        progress(0, 1, f"Downloading {model_id}…")
    path = snapshot_download(repo_id=model_id)
    if progress:
        progress(1, 1, "Download complete")
    return path

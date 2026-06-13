"""Application logging and user-facing error normalization."""

from __future__ import annotations

import faulthandler
import logging
import re
import uuid
import warnings
from dataclasses import dataclass
from io import TextIOWrapper
from logging.handlers import RotatingFileHandler

from .config import data_dir

_LOGGER_NAME = "llm_tunner"
_CRASH_LOG: TextIOWrapper | None = None


@dataclass(frozen=True)
class UserError:
    incident_id: str
    title: str
    summary: str
    action: str
    technical_detail: str


def configure_logging() -> None:
    """Write diagnostics to a bounded local log instead of noisy console output."""
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return

    log_dir = data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / "llm-tunner.log",
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # These warnings are represented in the GUI with actionable status instead.
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)


def logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)


def configure_crash_diagnostics() -> None:
    """Persist Python/native-extension fault tracebacks when the process aborts."""
    global _CRASH_LOG
    if _CRASH_LOG is not None:
        return

    log_dir = data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    _CRASH_LOG = (log_dir / "native-crash.log").open("a", encoding="utf-8")
    faulthandler.enable(file=_CRASH_LOG, all_threads=True)


def configure_ml_output() -> None:
    """Keep third-party progress/advisories out of the terminal-based GUI launch."""
    warnings.filterwarnings(
        "ignore",
        message=r"_check_is_size will be removed in a future PyTorch release.*",
        category=FutureWarning,
        module=r"bitsandbytes\.backends\.cuda\.ops",
    )
    try:
        from huggingface_hub.utils import logging as hf_logging  # type: ignore

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bars()
    except (ImportError, AttributeError):
        pass
    try:
        from transformers.utils import logging as transformers_logging  # type: ignore

        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
    except (ImportError, AttributeError):
        pass


def normalize_error(kind: str, message: str, context: str = "") -> UserError:
    """Translate common ML, Hub, CUDA, PDF, and filesystem failures."""
    incident_id = uuid.uuid4().hex[:8].upper()
    detail = f"{kind}: {message}".strip()
    lowered = detail.lower()

    title = "Operation failed"
    summary = message or kind
    action = "Review the details, then retry the operation."

    if "cuda out of memory" in lowered or "outofmemoryerror" in lowered:
        title = "GPU memory exhausted"
        summary = "The selected model or batch does not fit in available GPU memory."
        action = (
            "Close other GPU applications, select a smaller model, reduce sequence length, "
            "or use 4-bit QLoRA."
        )
    elif "401" in lowered or "unauthorized" in lowered or "invalid token" in lowered:
        title = "Hugging Face authentication failed"
        summary = "The Hub rejected the saved access token."
        action = "Open Settings and log in with a valid Hugging Face read token."
    elif "403" in lowered or "gated repo" in lowered:
        title = "Model access denied"
        summary = "Your Hugging Face account does not have access to this model."
        action = "Accept the model license on Hugging Face or choose a public model."
    elif "429" in lowered or "rate limit" in lowered:
        title = "Hugging Face rate limit reached"
        summary = "The Hub temporarily limited download requests."
        action = "Log in under Settings, wait briefly, and retry."
    elif any(token in lowered for token in ("timed out", "timeout", "connectionerror")):
        title = "Network request timed out"
        summary = "The remote model service did not respond in time."
        action = "Check the network connection and retry. Cached files will be reused."
    elif "no space left" in lowered or "disk full" in lowered:
        title = "Storage is full"
        summary = "There is not enough free disk space to finish this operation."
        action = "Free disk space or move HF_HOME and LLM_TUNNER_HOME to a larger drive."
    elif "filenotfounderror" in lowered or "no such file" in lowered:
        title = "File not found"
        summary = "A required document or model file is no longer available."
        action = "Re-add the document or verify the configured path."
    elif "pdf" in lowered and any(
        token in lowered for token in ("encrypted", "password", "invalid", "malformed")
    ):
        title = "PDF could not be read"
        summary = "The PDF is encrypted, damaged, or uses an unsupported structure."
        action = "Open and re-save the PDF without a password, then add it again."
    elif "dependencies are not installed" in lowered or "modulenotfounderror" in lowered:
        title = "Required component is missing"
        summary = "This feature needs an optional package that is not installed."
        action = "Run `make install`, restart the app, and retry."
    elif "cuda" in lowered and "not available" in lowered:
        title = "CUDA is unavailable"
        summary = "PyTorch cannot use the NVIDIA GPU in the current environment."
        action = "Run `make install-gpu`, restart the app, and verify the GPU in Settings."

    if context:
        summary = f"{context}: {summary}"

    # Redact token-shaped values before storing/displaying technical detail.
    safe_detail = re.sub(r"hf_[A-Za-z0-9]{8,}", "hf_[redacted]", detail)
    return UserError(
        incident_id=incident_id,
        title=title,
        summary=summary,
        action=action,
        technical_detail=safe_detail,
    )

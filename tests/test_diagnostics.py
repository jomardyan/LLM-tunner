from __future__ import annotations

from llm_tunner.diagnostics import normalize_error


def test_cuda_oom_error_is_actionable():
    error = normalize_error(
        "OutOfMemoryError",
        "CUDA out of memory. Tried to allocate 2 GiB.",
        "Fine-tuning",
    )
    assert error.title == "GPU memory exhausted"
    assert "smaller model" in error.action
    assert error.incident_id


def test_huggingface_token_is_redacted():
    error = normalize_error("RuntimeError", "Invalid token hf_1234567890ABCDEF")
    assert "hf_1234567890ABCDEF" not in error.technical_detail
    assert "[redacted]" in error.technical_detail


def test_rate_limit_error_has_login_recovery():
    error = normalize_error("HTTPError", "429 rate limit exceeded")
    assert error.title == "Hugging Face rate limit reached"
    assert "Log in" in error.action

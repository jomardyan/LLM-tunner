from __future__ import annotations

from types import SimpleNamespace

from llm_tunner.core.metrics import collect_runtime_metrics


def test_runtime_metrics_parse_nvidia_smi(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "llm_tunner.core.metrics.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="17, 1024, 4096, 63\n"),
    )
    monkeypatch.setattr(
        "llm_tunner.core.metrics.data_dir",
        lambda: tmp_path,
    )

    metrics = collect_runtime_metrics()
    assert metrics.gpu_utilization_pct == 17
    assert metrics.gpu_memory_used_mb == 1024
    assert metrics.gpu_memory_total_mb == 4096
    assert metrics.gpu_temperature_c == 63
    assert metrics.disk_free_gb is not None


def test_runtime_metrics_suppresses_console_window(monkeypatch, tmp_path):
    """nvidia-smi must run with creationflags so no console flashes on Windows."""
    captured: dict = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(stdout="17, 1024, 4096, 63\n")

    monkeypatch.setattr("llm_tunner.core.metrics.subprocess.run", fake_run)
    monkeypatch.setattr("llm_tunner.core.metrics.data_dir", lambda: tmp_path)

    collect_runtime_metrics()
    assert "creationflags" in captured

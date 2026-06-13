"""Lightweight runtime metrics that do not import the ML stack."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass

from ..config import data_dir


@dataclass(frozen=True)
class RuntimeMetrics:
    gpu_utilization_pct: int | None = None
    gpu_memory_used_mb: int | None = None
    gpu_memory_total_mb: int | None = None
    gpu_temperature_c: int | None = None
    disk_free_gb: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def collect_runtime_metrics() -> RuntimeMetrics:
    """Collect GPU and storage metrics without importing torch."""
    gpu_values: list[int | None] = [None, None, None, None]
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            text=True,
            timeout=5,
        )
        fields = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
        # Parse each field independently so one unsupported "N/A" value
        # (e.g. temperature on some laptop/virtual GPUs) doesn't discard the rest.
        for position, value in enumerate(fields[:4]):
            try:
                gpu_values[position] = int(value)
            except ValueError:
                pass
    except (OSError, subprocess.SubprocessError, IndexError):
        pass

    try:
        disk_free_gb = shutil.disk_usage(data_dir()).free / (1024**3)
    except OSError:
        disk_free_gb = None

    return RuntimeMetrics(
        gpu_utilization_pct=gpu_values[0],
        gpu_memory_used_mb=gpu_values[1],
        gpu_memory_total_mb=gpu_values[2],
        gpu_temperature_c=gpu_values[3],
        disk_free_gb=disk_free_gb,
    )

"""Compact live metrics rail shared by all workflows."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)


class MetricsPanel(QFrame):
    def __init__(self, open_settings) -> None:
        super().__init__()
        self.setObjectName("metricsPanel")
        self.setMinimumWidth(260)
        self.setMaximumWidth(300)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        title = QLabel("Live metrics")
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        self._grid = QGridLayout()
        self._grid.setColumnStretch(1, 1)
        root.addLayout(self._grid)
        self._values: dict[str, QLabel] = {}
        for row, (key, label, value) in enumerate(
            [
                ("documents", "Documents", "0"),
                ("pages", "Pages", "0"),
                ("chunks", "Chunks", "0"),
                ("extraction", "Extraction", "-"),
                ("operation", "Operation", "Ready"),
                ("elapsed", "Elapsed", "-"),
            ]
        ):
            self._add_metric(row, key, label, value)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        root.addWidget(self.progress)

        system = QLabel("System")
        system.setObjectName("sectionTitle")
        root.addWidget(system)
        self._system_grid = QGridLayout()
        root.addLayout(self._system_grid)
        for row, (key, label, value) in enumerate(
            [
                ("device", "Device", "Detecting..."),
                ("gpu", "GPU usage", "-"),
                ("vram", "VRAM", "-"),
                ("temperature", "Temperature", "-"),
                ("disk", "Disk free", "-"),
                ("model", "Model", "-"),
            ]
        ):
            self._add_metric(row, key, label, value, grid=self._system_grid)

        self.hf_warning = QLabel(
            "Hugging Face is not authenticated. Downloads may be rate-limited."
        )
        self.hf_warning.setObjectName("warning")
        self.hf_warning.setWordWrap(True)
        root.addWidget(self.hf_warning)
        self.settings_btn = QPushButton("Open Settings")
        self.settings_btn.clicked.connect(open_settings)
        root.addWidget(self.settings_btn)
        root.addStretch()

    def _add_metric(self, row, key, label, value, grid=None) -> None:
        grid = grid or self._grid
        label_widget = QLabel(label)
        label_widget.setObjectName("metricLabel")
        value_widget = QLabel(value)
        value_widget.setObjectName("metricValue")
        value_widget.setWordWrap(True)
        value_widget.setAlignment(value_widget.alignment())
        grid.addWidget(label_widget, row, 0)
        grid.addWidget(value_widget, row, 1)
        self._values[key] = value_widget

    def set_value(self, key: str, value: str | int) -> None:
        if key in self._values:
            self._values[key].setText(str(value))

    def set_progress(self, done: int, total: int) -> None:
        if total <= 0:
            self.progress.setRange(0, 0)
            return
        self.progress.setRange(0, 100)
        self.progress.setValue(round(done / total * 100))

    def set_authenticated(self, authenticated: bool) -> None:
        self.hf_warning.setVisible(not authenticated)
        self.settings_btn.setVisible(not authenticated)

    def update_runtime(self, metrics: dict) -> None:
        utilization = metrics.get("gpu_utilization_pct")
        used = metrics.get("gpu_memory_used_mb")
        total = metrics.get("gpu_memory_total_mb")
        temperature = metrics.get("gpu_temperature_c")
        disk = metrics.get("disk_free_gb")

        self.set_value("gpu", f"{utilization}%" if utilization is not None else "-")
        self.set_value(
            "vram",
            f"{used / 1024:.1f} / {total / 1024:.1f} GB"
            if used is not None and total
            else "-",
        )
        self.set_value(
            "temperature",
            f"{temperature} C" if temperature is not None else "-",
        )
        self.set_value("disk", f"{disk:.1f} GB" if disk is not None else "-")

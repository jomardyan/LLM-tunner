"""The main application window: tabs, shared state, and the worker-submission helper.

All long-running work goes through :meth:`MainWindow.submit`, which wraps a task
function in a :class:`Worker`, wires its signals to caller-supplied callbacks, and
hands it to a shared ``QThreadPool``. Tabs never create threads directly.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from PySide6.QtCore import QThreadPool, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .core.device import DeviceInfo, detect_device_isolated
from .core.models import huggingface_authenticated
from .diagnostics import logger, normalize_error
from .settings import Settings
from .ui.chat_tab import ChatTab
from .ui.documents_tab import DocumentsTab
from .ui.error_dialog import show_error_dialog
from .ui.finetune_tab import FineTuneTab
from .ui.rag_tab import RagTab
from .ui.settings_tab import SettingsTab
from .ui.widgets.metrics_panel import MetricsPanel
from .workers.base import Worker
from .workers.tasks import runtime_metrics_task


@dataclass
class AppState:
    """Selections shared across tabs."""

    base_model: str = ""
    embedding_model: str = ""
    adapter_path: str | None = None
    current_kb: str = ""
    # Absolute paths of PDFs the user has added in this session.
    documents: list[str] = field(default_factory=list)
    document_metrics: dict[str, dict] = field(default_factory=dict)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = Settings()
        self.state = AppState(
            base_model=self.settings.base_model,
            embedding_model=self.settings.embedding_model,
            current_kb=self.settings.last_kb,
        )
        self.pool = QThreadPool.globalInstance()
        self.device = DeviceInfo(
            kind="cpu",
            name="Detecting hardware...",
            total_vram_gb=None,
            torch_available=False,
            bnb_available=False,
        )

        self.setWindowTitle("LLM-tunner — customize open LLMs with your PDFs")
        self.resize(1380, 820)

        self.documents_tab = DocumentsTab(self)
        self.rag_tab = RagTab(self)
        self.finetune_tab = FineTuneTab(self)
        self.chat_tab = ChatTab(self)
        self.settings_tab = SettingsTab(self)

        central = QWidget()
        shell = QVBoxLayout(central)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        title = QLabel("LLM-tunner")
        title.setObjectName("appTitle")
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(QLabel("Model:"))
        self._model_label = QLabel(self.state.base_model)
        self._model_label.setObjectName("metricValue")
        header_layout.addWidget(self._model_label)
        self._device_label = QLabel("Detecting hardware...")
        self._device_label.setObjectName("muted")
        header_layout.addWidget(self._device_label)
        shell.addWidget(header)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(185)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 12, 8, 12)
        self.navigation = QListWidget()
        self.navigation.setObjectName("navigation")
        self.navigation.addItems(
            ["Documents", "Knowledge", "Fine-tune", "Chat", "Settings"]
        )
        sidebar_layout.addWidget(self.navigation)
        sidebar_layout.addStretch()
        self._kb_label = QLabel("Active KB\n(none)")
        self._kb_label.setObjectName("muted")
        self._kb_label.setWordWrap(True)
        sidebar_layout.addWidget(self._kb_label)
        body.addWidget(sidebar)

        self.pages = QStackedWidget()
        for page in (
            self.documents_tab,
            self.rag_tab,
            self.finetune_tab,
            self.chat_tab,
            self.settings_tab,
        ):
            self.pages.addWidget(page)
        self.navigation.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.navigation.setCurrentRow(0)
        body.addWidget(self.pages, 1)

        self.metrics_panel = MetricsPanel(self.open_settings)
        self.metrics_panel.set_value("model", self.state.base_model)
        self.metrics_panel.set_authenticated(huggingface_authenticated())
        body.addWidget(self.metrics_panel)
        shell.addLayout(body, 1)
        self.setCentralWidget(central)

        status = QStatusBar()
        self.setStatusBar(status)
        self._banner = QLabel("Detecting hardware and PyTorch capabilities...")
        status.addWidget(self._banner)
        self._device_worker = self.submit(
            lambda *, signals: detect_device_isolated(),
            on_result=self._on_device_detected,
            on_error=self._on_device_detection_error,
        )
        self._task_started: float | None = None
        self._active_task_name = ""
        self._runtime_worker = None
        self._clock = QTimer(self)
        self._clock.timeout.connect(self._update_elapsed)
        self._clock.start(1000)
        self._runtime_timer = QTimer(self)
        self._runtime_timer.timeout.connect(self.refresh_runtime_metrics)
        self._runtime_timer.start(5000)
        QTimer.singleShot(0, self.refresh_runtime_metrics)

    def closeEvent(self, event) -> None:
        """Stop the recurring timers before the window is destroyed.

        Leaving ``self._clock`` and ``self._runtime_timer`` running would let
        them fire on widgets whose C++ objects are already being torn down.
        """
        self._clock.stop()
        self._runtime_timer.stop()
        super().closeEvent(event)

    def _on_device_detected(self, info: DeviceInfo) -> None:
        self.device = info
        banner = info.capability_banner()
        self._banner.setText(banner)
        self._device_label.setText(
            f"{info.name} · {info.total_vram_gb:.0f} GB"
            if info.total_vram_gb
            else info.name
        )
        self.metrics_panel.set_value("device", info.name)
        self.settings_tab.update_device_banner(banner)
        self.settings_tab.update_model_guidance()

    def _on_device_detection_error(self, kind: str, message: str) -> None:
        banner = f"Hardware detection failed ({kind}: {message})."
        self._banner.setText(banner)
        self.settings_tab.update_device_banner(banner)

    # -- worker submission ------------------------------------------------------
    def submit(
        self,
        fn: Callable,
        *args,
        on_result: Callable | None = None,
        on_progress: Callable | None = None,
        on_metric: Callable | None = None,
        on_log: Callable | None = None,
        on_error: Callable | None = None,
        on_finished: Callable | None = None,
        task_name: str | None = None,
        **kwargs,
    ) -> Worker:
        """Run ``fn(*args, **kwargs)`` on the thread pool, wiring signals to callbacks.

        ``fn`` receives a ``signals`` kwarg (see :class:`WorkerSignals`).
        """
        worker = Worker(fn, *args, **kwargs)
        if task_name:
            self._start_task(task_name)
        if on_result:
            worker.signals.result.connect(on_result)
        worker.signals.result.connect(self.publish_metrics)
        if on_progress:
            worker.signals.progress.connect(on_progress)
        if task_name:
            worker.signals.progress.connect(self._on_task_progress)
        if on_metric:
            worker.signals.metric.connect(on_metric)
        if on_log:
            worker.signals.log.connect(on_log)
        if on_error:
            worker.signals.error.connect(on_error)
        else:
            worker.signals.error.connect(self._default_error)
        if on_finished:
            worker.signals.finished.connect(on_finished)
        if task_name:
            worker.signals.finished.connect(self._finish_task)
        self.pool.start(worker)
        return worker

    def _start_task(self, name: str) -> None:
        self._active_task_name = name
        self._task_started = time.perf_counter()
        self.metrics_panel.set_value("operation", name)
        self.metrics_panel.set_value("elapsed", "0s")
        self.metrics_panel.set_progress(0, 0)
        self.notify(f"{name} started")

    def _on_task_progress(self, done: int, total: int, message: str) -> None:
        self.metrics_panel.set_progress(done, total)
        if message:
            self.metrics_panel.set_value("operation", message)

    def _finish_task(self) -> None:
        self.metrics_panel.set_value("operation", "Ready")
        self.metrics_panel.set_progress(100, 100)
        QTimer.singleShot(2500, self._reset_progress_later)
        self._active_task_name = ""
        self._task_started = None

    def _reset_progress_later(self) -> None:
        """Clear the progress bar after the post-task delay.

        Tolerates the window closing before the delayed timer fires: the
        underlying C++ panel may already be destroyed, in which case the call
        is a harmless no-op rather than a ``RuntimeError``.
        """
        try:
            self.metrics_panel.set_progress(0, 100)
        except RuntimeError:
            pass

    def _update_elapsed(self) -> None:
        if self._task_started is None:
            return
        elapsed = int(time.perf_counter() - self._task_started)
        minutes, seconds = divmod(elapsed, 60)
        self.metrics_panel.set_value("elapsed", f"{minutes:02d}:{seconds:02d}")

    def publish_metrics(self, result) -> None:
        if not isinstance(result, dict):
            return
        for key in ("documents", "pages", "chunks"):
            if key in result:
                self.metrics_panel.set_value(key, result[key])
        if "elapsed_seconds" in result:
            self.metrics_panel.set_value("elapsed", f"{result['elapsed_seconds']:.1f}s")
        if "native_documents" in result or "ocr_documents" in result:
            self.metrics_panel.set_value(
                "extraction",
                f"{result.get('native_documents', 0)} native / "
                f"{result.get('ocr_documents', 0)} OCR",
            )

    def refresh_runtime_metrics(self) -> None:
        if self._runtime_worker is not None:
            return
        self._runtime_worker = self.submit(
            runtime_metrics_task,
            on_result=self.metrics_panel.update_runtime,
            on_finished=lambda: setattr(self, "_runtime_worker", None),
        )

    def open_settings(self) -> None:
        self.navigation.setCurrentRow(4)

    def update_model_label(self, model_id: str) -> None:
        self._model_label.setText(model_id)
        self.metrics_panel.set_value("model", model_id)

    def update_kb_label(self, name: str) -> None:
        self._kb_label.setText(f"Active KB\n{name or '(none)'}")

    def update_document_metrics(self) -> None:
        metrics = list(self.state.document_metrics.values())
        self.metrics_panel.set_value("documents", len(self.state.documents))
        self.metrics_panel.set_value("pages", sum(item.get("pages", 0) for item in metrics))
        native = sum(not item.get("used_ocr", False) for item in metrics)
        ocr = sum(bool(item.get("used_ocr", False)) for item in metrics)
        self.metrics_panel.set_value("extraction", f"{native} native / {ocr} OCR")

    def _default_error(self, kind: str, message: str) -> None:
        self.show_error(kind, message)

    def show_error(self, kind: str, message: str, context: str = "") -> str:
        error = normalize_error(kind, message, context)
        logger().error(
            "Incident %s: %s | %s",
            error.incident_id,
            error.summary,
            error.technical_detail,
        )
        self.statusBar().showMessage(
            f"{error.title}: {error.summary} [{error.incident_id}]",
            12000,
        )
        show_error_dialog(self, error)
        return f"{error.title}: {error.summary} [{error.incident_id}]"

    def notify(self, message: str, timeout: int = 4000) -> None:
        self.statusBar().showMessage(message, timeout)

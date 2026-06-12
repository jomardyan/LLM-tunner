"""The main application window: tabs, shared state, and the worker-submission helper.

All long-running work goes through :meth:`MainWindow.submit`, which wraps a task
function in a :class:`Worker`, wires its signals to caller-supplied callbacks, and
hands it to a shared ``QThreadPool``. Tabs never create threads directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QLabel, QMainWindow, QStatusBar, QTabWidget

from .core.device import detect_device
from .settings import Settings
from .ui.chat_tab import ChatTab
from .ui.documents_tab import DocumentsTab
from .ui.finetune_tab import FineTuneTab
from .ui.rag_tab import RagTab
from .ui.settings_tab import SettingsTab
from .workers.base import Worker


@dataclass
class AppState:
    """Selections shared across tabs."""

    base_model: str = ""
    embedding_model: str = ""
    adapter_path: str | None = None
    current_kb: str = ""
    # Absolute paths of PDFs the user has added in this session.
    documents: list[str] = field(default_factory=list)


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
        self.device = detect_device()

        self.setWindowTitle("LLM-tunner — customize open LLMs with your PDFs")
        self.resize(1100, 760)

        tabs = QTabWidget()
        self.documents_tab = DocumentsTab(self)
        self.rag_tab = RagTab(self)
        self.finetune_tab = FineTuneTab(self)
        self.chat_tab = ChatTab(self)
        self.settings_tab = SettingsTab(self)
        tabs.addTab(self.documents_tab, "1. Documents")
        tabs.addTab(self.rag_tab, "2. Knowledge (RAG)")
        tabs.addTab(self.finetune_tab, "3. Fine-tune")
        tabs.addTab(self.chat_tab, "4. Chat")
        tabs.addTab(self.settings_tab, "Settings")
        self.setCentralWidget(tabs)

        status = QStatusBar()
        self.setStatusBar(status)
        self._banner = QLabel(self.device.capability_banner())
        status.addWidget(self._banner)

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
        **kwargs,
    ) -> Worker:
        """Run ``fn(*args, **kwargs)`` on the thread pool, wiring signals to callbacks.

        ``fn`` receives a ``signals`` kwarg (see :class:`WorkerSignals`).
        """
        worker = Worker(fn, *args, **kwargs)
        if on_result:
            worker.signals.result.connect(on_result)
        if on_progress:
            worker.signals.progress.connect(on_progress)
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
        self.pool.start(worker)
        return worker

    def _default_error(self, kind: str, message: str) -> None:
        self.statusBar().showMessage(f"{kind}: {message}", 8000)

    def notify(self, message: str, timeout: int = 4000) -> None:
        self.statusBar().showMessage(message, timeout)

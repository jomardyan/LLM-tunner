"""Shared worker plumbing.

The golden rule of Qt threading: **never touch widgets from a worker thread.** Workers
are :class:`QRunnable`s submitted to a ``QThreadPool``; they communicate results back to
the GUI thread exclusively through :class:`WorkerSignals`, which Qt delivers via thread-safe
queued connections. ML libraries release the GIL during heavy compute, so a worker thread
keeps the UI responsive in practice.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    """The single signal contract every worker uses.

    Signals:
        progress: (done, total, message) for progress bars / status text.
        metric:   arbitrary metric payload (e.g. a training step's loss dict).
        log:      a human-readable log line.
        result:   the worker's return value on success.
        error:    (exc_type_name, message) on failure.
        finished: always emitted last, success or failure.
    """

    progress = Signal(int, int, str)
    metric = Signal(object)
    log = Signal(str)
    result = Signal(object)
    error = Signal(str, str)
    finished = Signal()


class Worker(QRunnable):
    """Run ``fn(*args, **kwargs)`` on a thread-pool thread.

    ``fn`` receives a ``signals`` keyword giving it access to ``progress``/``log``/``metric``
    so long tasks can stream updates. Its return value is emitted on ``result``.
    """

    def __init__(self, fn: Callable, *args, **kwargs) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.kwargs.setdefault("signals", self.signals)

    @Slot()
    def run(self) -> None:  # noqa: D102 - Qt entry point
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as exc:  # noqa: BLE001 - surface everything to the UI
            try:
                self.signals.error.emit(type(exc).__name__, str(exc))
            except RuntimeError:
                pass  # The application closed while the worker was running.
        else:
            try:
                self.signals.result.emit(result)
            except RuntimeError:
                pass
        finally:
            try:
                self.signals.finished.emit()
            except RuntimeError:
                pass

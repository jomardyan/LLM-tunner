"""Application entry point.

``llm-tunner`` (console script) and ``llm-tunner-gui`` both call :func:`main`.
"""

from __future__ import annotations

import multiprocessing
import os
import sys

_SUPPORTED_PYTHON = (3, 14)


def _validate_runtime() -> None:
    """Reject stale environments before loading Qt, PyTorch, or other native modules."""
    if sys.version_info[:2] == _SUPPORTED_PYTHON:
        return
    expected = ".".join(map(str, _SUPPORTED_PYTHON))
    raise RuntimeError(
        f"LLM-tunner requires Python {expected}; this process is Python "
        f"{sys.version.split()[0]} at {sys.executable}. Run `make install`, then "
        "`make run-gui`."
    )


def _configure_windows_process() -> None:
    """Apply Windows process setup needed by GUI and frozen application builds."""
    multiprocessing.freeze_support()
    if sys.platform != "win32":
        return

    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(  # type: ignore[attr-defined]
            "LLM-tunner.Desktop"
        )
    except (AttributeError, OSError):
        pass


def main() -> int:
    _validate_runtime()
    _configure_windows_process()
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    from PySide6.QtWidgets import QApplication

    from .config import APP_NAME, ORG_NAME
    from .diagnostics import configure_crash_diagnostics, configure_logging
    from .main_window import MainWindow
    from .ui.theme import APP_STYLESHEET

    configure_crash_diagnostics()
    configure_logging()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

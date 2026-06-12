"""Application entry point.

``llm-tunner`` (console script) and ``llm-tunner-gui`` both call :func:`main`.
"""

from __future__ import annotations

import multiprocessing
import sys


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
    _configure_windows_process()

    from PySide6.QtWidgets import QApplication

    from .config import APP_NAME, ORG_NAME
    from .main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

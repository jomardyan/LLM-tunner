"""Detailed, actionable error dialog shared by every workflow."""

from __future__ import annotations

from PySide6.QtWidgets import QMessageBox, QWidget

from ..diagnostics import UserError


def show_error_dialog(parent: QWidget, error: UserError) -> None:
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle(error.title)
    box.setText(error.summary)
    box.setInformativeText(
        f"What to do:\n{error.action}\n\nIncident ID: {error.incident_id}"
    )
    box.setDetailedText(error.technical_detail)
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.exec()

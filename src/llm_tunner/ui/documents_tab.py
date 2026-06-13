"""Documents tab: add PDFs, extract text in the background, preview the result."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..workers.tasks import ingest_pdf_task

if TYPE_CHECKING:
    from ..main_window import MainWindow


class DocumentsTab(QWidget):
    def __init__(self, window: MainWindow) -> None:
        super().__init__()
        self.window = window

        root = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.add_btn = QPushButton("Add PDF(s)…")
        self.remove_btn = QPushButton("Remove selected")
        self.add_btn.clicked.connect(self._add_pdfs)
        self.remove_btn.clicked.connect(self._remove_selected)
        controls.addWidget(self.add_btn)
        controls.addWidget(self.remove_btn)
        controls.addStretch()
        root.addLayout(controls)

        body = QHBoxLayout()
        self.file_list = QListWidget()
        self.file_list.currentTextChanged.connect(self._preview)
        body.addWidget(self.file_list, 1)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setPlaceholderText("Select a PDF to preview its extracted text.")
        body.addWidget(self.preview, 2)
        root.addLayout(body, 1)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)

    def _add_pdfs(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select PDF files", "", "PDF files (*.pdf)")
        for path in paths:
            if path not in self.window.state.documents:
                self.window.state.documents.append(path)
                self.file_list.addItem(path)
        self.window.rag_tab.refresh_documents()
        self.window.finetune_tab.refresh_documents()

    def _remove_selected(self) -> None:
        for item in self.file_list.selectedItems():
            path = item.text()
            if path in self.window.state.documents:
                self.window.state.documents.remove(path)
            self.file_list.takeItem(self.file_list.row(item))
        self.window.rag_tab.refresh_documents()
        self.window.finetune_tab.refresh_documents()

    def _preview(self, path: str) -> None:
        if not path:
            return
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # busy indicator
        self.preview.setPlainText("Extracting…")
        self.window.submit(
            ingest_pdf_task,
            path,
            on_result=self._show_preview,
            on_finished=lambda: self.progress.setVisible(False),
            on_error=self._on_error,
        )

    def _show_preview(self, info: dict) -> None:
        mode = "OCR" if info["used_ocr"] else "native text"
        header = (
            f"[{info['backend']}, {mode}] {info['pages']} page(s), "
            f"type: {info['document_type']}\n{'-' * 40}\n"
        )
        self.preview.setPlainText(header + info["preview"])

    def _on_error(self, kind: str, message: str) -> None:
        self.preview.setPlainText(f"Extraction failed ({kind}): {message}")

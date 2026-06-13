"""Documents tab: add PDFs, extract text in the background, preview the result."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
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
        root.setContentsMargins(18, 16, 18, 16)

        title = QLabel("Documents")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        subtitle = QLabel(
            "Add PDFs and inspect the automatically selected native-text or OCR extraction."
        )
        subtitle.setObjectName("muted")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        controls = QHBoxLayout()
        self.add_btn = QPushButton("Add PDF(s)…")
        self.remove_btn = QPushButton("Remove selected")
        self.clear_btn = QPushButton("Clear all")
        self.add_btn.clicked.connect(self._add_pdfs)
        self.remove_btn.clicked.connect(self._remove_selected)
        self.clear_btn.clicked.connect(self._clear_all)
        self.add_btn.setObjectName("primary")
        controls.addWidget(self.add_btn)
        controls.addWidget(self.remove_btn)
        controls.addWidget(self.clear_btn)
        controls.addStretch()
        self.summary = QLabel("0 documents")
        self.summary.setObjectName("muted")
        controls.addWidget(self.summary)
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
        self._refresh_summary()

    def _remove_selected(self) -> None:
        for item in self.file_list.selectedItems():
            path = item.text()
            if path in self.window.state.documents:
                self.window.state.documents.remove(path)
            self.window.state.document_metrics.pop(path, None)
            self.file_list.takeItem(self.file_list.row(item))
        self.window.rag_tab.refresh_documents()
        self.window.finetune_tab.refresh_documents()
        self.window.update_document_metrics()
        self._refresh_summary()

    def _clear_all(self) -> None:
        self.window.state.documents.clear()
        self.window.state.document_metrics.clear()
        self.file_list.clear()
        self.preview.clear()
        self.window.rag_tab.refresh_documents()
        self.window.finetune_tab.refresh_documents()
        self.window.update_document_metrics()
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        count = len(self.window.state.documents)
        pages = sum(
            item.get("pages", 0) for item in self.window.state.document_metrics.values()
        )
        self.summary.setText(f"{count} document(s) · {pages} page(s)")

    def _preview(self, path: str) -> None:
        if not path:
            return
        cached = self.window.state.document_metrics.get(path)
        if cached is not None:
            self.preview.setPlainText(self._render_preview(cached))
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
            task_name="Extracting PDF",
        )

    def _render_preview(self, info: dict) -> str:
        """Build the header + preview body string shown for an extracted document."""
        mode = "OCR" if info["used_ocr"] else "native text"
        header = (
            f"[{info['backend']}, {mode}] {info['pages']} page(s), "
            f"type: {info['document_type']}\n{'-' * 40}\n"
        )
        return header + info["preview"]

    def _show_preview(self, info: dict) -> None:
        self.preview.setPlainText(self._render_preview(info))
        self.window.state.document_metrics[info["source"]] = info
        self.window.update_document_metrics()
        self._refresh_summary()

    def _on_error(self, kind: str, message: str) -> None:
        summary = self.window.show_error(kind, message, "PDF extraction")
        self.preview.setPlainText(summary)

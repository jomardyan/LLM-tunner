"""Knowledge (RAG) tab: build a persistent vector knowledge base from the added PDFs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..workers.tasks import build_kb_task

if TYPE_CHECKING:
    from ..main_window import MainWindow


class RagTab(QWidget):
    def __init__(self, window: MainWindow) -> None:
        super().__init__()
        self.window = window

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        title = QLabel("Knowledge base")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        description = QLabel(
            "Extract, chunk, and embed your PDFs into a searchable local knowledge base."
        )
        description.setObjectName("muted")
        description.setWordWrap(True)
        root.addWidget(description)

        form = QFormLayout()
        self.kb_name = QLineEdit("my_knowledge_base")
        form.addRow("Knowledge base name:", self.kb_name)
        self.embed_label = QLabel(self.window.state.embedding_model)
        form.addRow("Embedding model:", self.embed_label)
        root.addLayout(form)

        self.doc_list = QListWidget()
        root.addWidget(QLabel("PDFs to index (add them in the Documents tab):"))
        root.addWidget(self.doc_list, 1)

        controls = QHBoxLayout()
        self.build_btn = QPushButton("Build / update knowledge base")
        self.build_btn.setObjectName("primary")
        self.build_btn.clicked.connect(self._build)
        controls.addWidget(self.build_btn)
        controls.addStretch()
        root.addLayout(controls)

        self.progress = QProgressBar()
        self.progress.setObjectName("workflowProgress")
        self.progress.setFixedHeight(28)
        self.progress.setFormat("%v / %m chunks (%p%)")
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        root.addWidget(self.log, 1)

        self.refresh_documents()

    def refresh_documents(self) -> None:
        self.doc_list.clear()
        self.doc_list.addItems(self.window.state.documents)
        self.embed_label.setText(self.window.state.embedding_model)

    def _build(self) -> None:
        from ..core.rag import missing_rag_dependencies, rag_install_command

        pdfs = list(self.window.state.documents)
        if not pdfs:
            self.window.notify("Add PDFs in the Documents tab first.")
            return
        missing = missing_rag_dependencies()
        if missing:
            message = (
                f"RAG dependencies are not installed: {', '.join(missing)}. "
                f"Run: {rag_install_command()}"
            )
            self.log.appendPlainText(f"ERROR: {message}")
            self.window.notify(message, 10000)
            return
        name = self.kb_name.text().strip() or "my_knowledge_base"
        self.build_btn.setEnabled(False)
        self.build_btn.setText("Building knowledge base…")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.progress.setValue(0)
        self.log.appendPlainText(f"Building '{name}' from {len(pdfs)} PDF(s)…")
        self.window.submit(
            build_kb_task,
            name,
            pdfs,
            self.window.state.embedding_model,
            on_progress=self._on_progress,
            on_log=self.log.appendPlainText,
            on_result=self._on_done,
            on_error=self._on_error,
            on_finished=self._on_build_finished,
            task_name="Building knowledge base",
        )

    def _on_progress(self, done: int, total: int, message: str) -> None:
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(done)
        else:
            self.progress.setRange(0, 0)
        if message:
            self.progress.setToolTip(message)
            self.window.notify(message)

    def _on_done(self, result: dict) -> None:
        self.window.state.current_kb = result["kb"]
        self.window.settings.last_kb = result["kb"]
        self.log.appendPlainText(
            f"Done. {result['chunks']} chunks indexed; collection now holds {result['count']}."
        )
        self.window.chat_tab.refresh_kb()
        self.window.update_kb_label(result["kb"])
        self.window.metrics_panel.set_value(
            "operation",
            f"{result['chunks']} chunks · {result['chunks_per_second']:.1f}/s",
        )
        self.window.notify(f"Knowledge base '{result['kb']}' ready.")

    def _on_error(self, kind: str, message: str) -> None:
        self.log.appendPlainText(self.window.show_error(kind, message, "Knowledge base"))

    def _on_build_finished(self) -> None:
        self.build_btn.setEnabled(True)
        self.build_btn.setText("Build / update knowledge base")
        self.progress.setVisible(False)

"""Chat tab: talk to the model, optionally grounded in a RAG knowledge base."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..workers.tasks import plain_chat_task, rag_query_task

if TYPE_CHECKING:
    from ..main_window import MainWindow


class ChatTab(QWidget):
    def __init__(self, window: MainWindow) -> None:
        super().__init__()
        self.window = window
        self.history: list[dict] = []

        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.use_rag = QCheckBox("Use knowledge base (RAG)")
        self.use_rag.setChecked(True)
        top.addWidget(self.use_rag)
        top.addWidget(QLabel("KB:"))
        self.kb_label = QLabel("(none)")
        top.addWidget(self.kb_label)
        top.addStretch()
        self.adapter_label = QLabel("adapter: none")
        top.addWidget(self.adapter_label)
        root.addLayout(top)

        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        root.addWidget(self.transcript, 1)

        entry = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask a question about your documents…")
        self.input.returnPressed.connect(self._send)
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._send)
        entry.addWidget(self.input, 1)
        entry.addWidget(self.send_btn)
        root.addLayout(entry)

        self.refresh_kb()

    def refresh_kb(self) -> None:
        self.kb_label.setText(self.window.state.current_kb or "(none)")
        adapter = self.window.state.adapter_path
        self.adapter_label.setText(f"adapter: {adapter.split('/')[-1] if adapter else 'none'}")

    def _append(self, role: str, text: str) -> None:
        self.transcript.append(f"<b>{role}:</b> {text}<br>")

    def _send(self) -> None:
        query = self.input.text().strip()
        if not query:
            return
        self.input.clear()
        self._append("You", query)
        self.send_btn.setEnabled(False)

        model_id = self.window.state.base_model
        adapter = self.window.state.adapter_path
        if self.use_rag.isChecked() and self.window.state.current_kb:
            self.window.submit(
                rag_query_task,
                model_id,
                adapter,
                self.window.state.current_kb,
                query,
                self.window.state.embedding_model,
                self.window.settings.top_k,
                on_result=self._on_answer,
                on_log=self.window.notify,
                on_finished=lambda: self.send_btn.setEnabled(True),
                on_error=self._on_error,
            )
        else:
            self.window.submit(
                plain_chat_task,
                model_id,
                adapter,
                query,
                list(self.history),
                on_result=lambda r: self._on_answer(r, query),
                on_log=self.window.notify,
                on_finished=lambda: self.send_btn.setEnabled(True),
                on_error=self._on_error,
            )

    def _on_answer(self, result: dict, query: str | None = None) -> None:
        answer = result.get("answer", "")
        sources = result.get("sources", "")
        self._append("Assistant", answer + (f"<br><i>Sources: {sources}</i>" if sources else ""))
        if query is not None:
            self.history.append({"role": "user", "content": query})
            self.history.append({"role": "assistant", "content": answer})

    def _on_error(self, kind: str, message: str) -> None:
        self._append("Error", f"{kind}: {message}")

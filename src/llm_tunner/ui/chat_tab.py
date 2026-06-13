"""Chat tab: talk to the model, optionally grounded in a RAG knowledge base."""

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
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
        self._last_answer = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)

        title = QLabel("Chat")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        subtitle = QLabel(
            "Ask the base model directly or ground answers in the active knowledge base."
        )
        subtitle.setObjectName("muted")
        root.addWidget(subtitle)

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
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear)
        top.addWidget(self.clear_btn)
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self._export)
        top.addWidget(self.export_btn)
        root.addLayout(top)

        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        root.addWidget(self.transcript, 1)

        entry = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask a question about your documents…")
        self.input.returnPressed.connect(self._send)
        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("primary")
        self.send_btn.clicked.connect(self._send)
        entry.addWidget(self.input, 1)
        entry.addWidget(self.send_btn)
        root.addLayout(entry)

        self.refresh_kb()

    def refresh_kb(self) -> None:
        self.kb_label.setText(self.window.state.current_kb or "(none)")
        adapter = self.window.state.adapter_path
        self.adapter_label.setText(f"adapter: {Path(adapter).name if adapter else 'none'}")

    def _append(self, role: str, text: str, sources: str = "") -> None:
        safe_role = html.escape(role)
        safe_text = html.escape(text).replace("\n", "<br>")
        source_html = f"<br><i>Sources: {html.escape(sources)}</i>" if sources else ""
        self.transcript.append(f"<b>{safe_role}:</b> {safe_text}{source_html}<br>")

    def _send(self) -> None:
        query = self.input.text().strip()
        if not query:
            return
        self.input.clear()
        self._append("You", query)
        self.send_btn.setEnabled(False)

        settings = self.window.settings
        gen = {
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "max_new_tokens": settings.max_new_tokens,
            "system_prompt": settings.system_prompt,
        }
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
                settings.top_k,
                history=list(self.history),
                score_threshold=settings.score_threshold,
                **gen,
                on_result=lambda result: self._on_answer(result, query),
                on_log=self.window.notify,
                on_finished=lambda: self.send_btn.setEnabled(True),
                on_error=self._on_error,
                task_name="Generating grounded answer",
            )
        else:
            self.window.submit(
                plain_chat_task,
                model_id,
                adapter,
                query,
                list(self.history),
                **gen,
                on_result=lambda r: self._on_answer(r, query),
                on_log=self.window.notify,
                on_finished=lambda: self.send_btn.setEnabled(True),
                on_error=self._on_error,
                task_name="Generating answer",
            )

    def _on_answer(self, result: dict, query: str | None = None) -> None:
        answer = result.get("answer", "")
        sources = result.get("sources", "")
        self._append("Assistant", answer, sources)
        self._last_answer = answer
        elapsed = result.get("elapsed_seconds")
        if elapsed is not None:
            self.window.metrics_panel.set_value(
                "operation",
                f"Answer: {elapsed:.1f}s · {result.get('response_words', 0)} words · "
                f"{result.get('contexts', 0)} context(s)",
            )
        if query is not None:
            self.history.append({"role": "user", "content": query})
            self.history.append({"role": "assistant", "content": answer})

    def _on_error(self, kind: str, message: str) -> None:
        self._append("Error", self.window.show_error(kind, message, "Chat"))

    def _clear(self) -> None:
        self.history.clear()
        self._last_answer = ""
        self.transcript.clear()
        self.window.notify("Chat cleared.")

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export chat",
            "llm-tunner-chat.txt",
            "Text files (*.txt)",
        )
        if not path:
            return
        try:
            Path(path).write_text(self.transcript.toPlainText(), encoding="utf-8")
        except OSError as exc:
            self.window.show_error(type(exc).__name__, str(exc), "Export chat")
            return
        self.window.notify(f"Chat exported to {path}", 8000)

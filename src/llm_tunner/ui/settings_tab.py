"""Settings tab: base model picker, embedding model, retrieval depth, device info."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    DEFAULT_EMBEDDING_MODEL,
    HIGH_QUALITY_EMBEDDING_MODEL,
    data_dir,
)
from ..core.models import registry
from ..workers.tasks import download_model_task, huggingface_login_task

if TYPE_CHECKING:
    from ..main_window import MainWindow


class SettingsTab(QWidget):
    def __init__(self, window: MainWindow) -> None:
        super().__init__()
        self.window = window

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        title = QLabel("Settings")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        subtitle = QLabel(
            "Choose models, retrieval behavior, authentication, and diagnostics."
        )
        subtitle.setObjectName("muted")
        root.addWidget(subtitle)
        form = QFormLayout()

        self.model_combo = QComboBox()
        self._model_entries = {entry.model_id: entry for entry in registry()}
        for entry in self._model_entries.values():
            self.model_combo.addItem(
                f"{entry.model_id}  —  {entry.size}, {entry.license}", entry.model_id
            )
        current_model = self.window.state.base_model
        if current_model and current_model not in self._model_entries:
            self.model_combo.addItem(f"{current_model}  —  custom", current_model)
        self._select_current(self.model_combo, current_model)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        form.addRow("Base model:", self.model_combo)

        custom_row = QHBoxLayout()
        self.custom_model = QLineEdit()
        self.custom_model.setPlaceholderText("Custom model ID or local path — then press Use")
        self.use_custom_btn = QPushButton("Use")
        self.use_custom_btn.clicked.connect(self._use_custom_model)
        self.download_btn = QPushButton("Download")
        self.download_btn.clicked.connect(self._download_model)
        custom_row.addWidget(self.custom_model, 1)
        custom_row.addWidget(self.use_custom_btn)
        custom_row.addWidget(self.download_btn)
        form.addRow("Custom / download:", custom_row)

        self.embed_combo = QComboBox()
        self.embed_combo.addItem(f"{DEFAULT_EMBEDDING_MODEL} (fast)", DEFAULT_EMBEDDING_MODEL)
        self.embed_combo.addItem(f"{HIGH_QUALITY_EMBEDDING_MODEL} (quality)", HIGH_QUALITY_EMBEDDING_MODEL)
        self._select_current(self.embed_combo, self.window.state.embedding_model)
        self.embed_combo.currentIndexChanged.connect(self._on_embed_changed)
        form.addRow("Embedding model:", self.embed_combo)

        self.topk = QSpinBox()
        self.topk.setRange(1, 20)
        self.topk.setValue(self.window.settings.top_k)
        self.topk.valueChanged.connect(self._on_topk_changed)
        form.addRow("Retrieval top-k:", self.topk)

        settings = self.window.settings
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setDecimals(2)
        self.temperature.setValue(settings.temperature)
        self.temperature.valueChanged.connect(
            lambda v: setattr(self.window.settings, "temperature", v)
        )
        form.addRow("Temperature:", self.temperature)

        self.top_p = QDoubleSpinBox()
        self.top_p.setRange(0.0, 1.0)
        self.top_p.setSingleStep(0.05)
        self.top_p.setDecimals(2)
        self.top_p.setValue(settings.top_p)
        self.top_p.valueChanged.connect(lambda v: setattr(self.window.settings, "top_p", v))
        form.addRow("Top-p:", self.top_p)

        self.max_new_tokens = QSpinBox()
        self.max_new_tokens.setRange(16, 8192)
        self.max_new_tokens.setSingleStep(16)
        self.max_new_tokens.setValue(settings.max_new_tokens)
        self.max_new_tokens.valueChanged.connect(
            lambda v: setattr(self.window.settings, "max_new_tokens", v)
        )
        form.addRow("Max new tokens:", self.max_new_tokens)

        self.score_threshold = QDoubleSpinBox()
        self.score_threshold.setRange(0.0, 1.0)
        self.score_threshold.setSingleStep(0.05)
        self.score_threshold.setDecimals(2)
        self.score_threshold.setToolTip("Drop retrieved chunks below this cosine similarity (0 = keep all).")
        self.score_threshold.setValue(settings.score_threshold)
        self.score_threshold.valueChanged.connect(
            lambda v: setattr(self.window.settings, "score_threshold", v)
        )
        form.addRow("Retrieval score threshold:", self.score_threshold)

        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(128, 4096)
        self.chunk_size.setSingleStep(64)
        self.chunk_size.setToolTip("Applies to the next knowledge-base build.")
        self.chunk_size.setValue(settings.chunk_size)
        self.chunk_size.valueChanged.connect(
            lambda v: setattr(self.window.settings, "chunk_size", v)
        )
        form.addRow("Chunk size (chars):", self.chunk_size)

        self.chunk_overlap = QSpinBox()
        self.chunk_overlap.setRange(0, 1024)
        self.chunk_overlap.setSingleStep(16)
        self.chunk_overlap.setToolTip("Applies to the next knowledge-base build.")
        self.chunk_overlap.setValue(settings.chunk_overlap)
        self.chunk_overlap.valueChanged.connect(
            lambda v: setattr(self.window.settings, "chunk_overlap", v)
        )
        form.addRow("Chunk overlap (chars):", self.chunk_overlap)

        root.addLayout(form)

        root.addWidget(QLabel("System prompt (optional — applies to chat):"))
        self.system_prompt = QPlainTextEdit()
        self.system_prompt.setPlaceholderText(
            "Optional style/behaviour guidance, e.g. 'Answer concisely in bullet points.'"
        )
        self.system_prompt.setFixedHeight(70)
        self.system_prompt.setPlainText(settings.system_prompt)
        self.system_prompt.textChanged.connect(self._on_system_prompt_changed)
        root.addWidget(self.system_prompt)
        self.model_guidance = QLabel()
        self.model_guidance.setWordWrap(True)
        self.model_guidance.setObjectName("warning")
        root.addWidget(self.model_guidance)
        self.update_model_guidance()

        root.addWidget(QLabel("<b>Detected hardware</b>"))
        self.device_banner = QLabel("Detecting hardware and PyTorch capabilities...")
        self.device_banner.setWordWrap(True)
        root.addWidget(self.device_banner)

        root.addWidget(QLabel("<b>Hugging Face Hub</b>"))
        from ..core.models import huggingface_authenticated

        self.hf_status = QLabel()
        self.hf_status.setWordWrap(True)
        root.addWidget(self.hf_status)

        hf_row = QHBoxLayout()
        self.hf_token = QLineEdit()
        self.hf_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token.setPlaceholderText("Hugging Face read token (hf_...)")
        self.hf_login_btn = QPushButton("Log in")
        self.hf_login_btn.clicked.connect(self._login_huggingface)
        hf_row.addWidget(self.hf_token, 1)
        hf_row.addWidget(self.hf_login_btn)
        root.addLayout(hf_row)
        self._set_hf_status(huggingface_authenticated())

        diagnostics = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh system metrics")
        self.refresh_btn.clicked.connect(self.window.refresh_runtime_metrics)
        diagnostics.addWidget(self.refresh_btn)
        self.logs_btn = QPushButton("Open diagnostics folder")
        self.logs_btn.clicked.connect(self._open_logs)
        diagnostics.addWidget(self.logs_btn)
        diagnostics.addStretch()
        root.addLayout(diagnostics)
        root.addStretch()

    def update_device_banner(self, text: str) -> None:
        self.device_banner.setText(text)

    def update_model_guidance(self) -> None:
        model_id = self.model_combo.currentData()
        entry = self._model_entries.get(model_id)
        vram = self.window.device.total_vram_gb or 0
        large_model = entry and entry.size in {"3B", "3.8B", "7B", "8B"}
        if vram and vram <= 4.5 and large_model:
            self.model_guidance.setText(
                f"{entry.size} is likely too large for {vram:.0f} GB VRAM. "
                "Use Qwen2.5-0.5B/1.5B for local work, or expect CPU offload and slow output."
            )
            self.model_guidance.show()
        else:
            self.model_guidance.hide()

    def _set_hf_status(self, authenticated: bool, detail: str = "") -> None:
        if authenticated:
            self.hf_status.setText(
                "Authenticated. Model downloads use your Hugging Face account limits."
            )
            return
        suffix = f" {detail}" if detail else ""
        self.hf_status.setText(
            "Not authenticated. Public downloads still work, but may be slower or "
            f"rate-limited.{suffix}"
        )

    def _login_huggingface(self) -> None:
        token = self.hf_token.text().strip()
        self.hf_token.clear()
        if not token:
            self._set_hf_status(False, "Enter a read token first.")
            return

        self.hf_login_btn.setEnabled(False)
        self.hf_status.setText("Validating Hugging Face token...")
        self.window.submit(
            huggingface_login_task,
            token,
            on_result=self._on_hf_login_success,
            on_error=self._on_hf_login_error,
            on_finished=lambda: self.hf_login_btn.setEnabled(True),
            task_name="Signing in to Hugging Face",
        )

    def _on_hf_login_success(self, _result: dict) -> None:
        self._set_hf_status(True)
        self.window.metrics_panel.set_authenticated(True)
        self.window.notify("Hugging Face authentication saved.")

    def _on_hf_login_error(self, kind: str, message: str) -> None:
        summary = self.window.show_error(kind, message, "Hugging Face login")
        self._set_hf_status(False, summary)

    def _open_logs(self) -> None:
        log_dir = data_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_dir)))

    def _select_current(self, combo: QComboBox, value: str) -> None:
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return

    def _on_model_changed(self) -> None:
        model_id = self.model_combo.currentData()
        self.window.state.base_model = model_id
        self.window.settings.base_model = model_id
        self.window.update_model_label(model_id)
        self.update_model_guidance()
        self.window.notify(f"Base model set to {model_id}")

    def _on_embed_changed(self) -> None:
        model = self.embed_combo.currentData()
        self.window.state.embedding_model = model
        self.window.settings.embedding_model = model
        self.window.rag_tab.refresh_documents()

    def _on_topk_changed(self, value: int) -> None:
        self.window.settings.top_k = value

    def _on_system_prompt_changed(self) -> None:
        self.window.settings.system_prompt = self.system_prompt.toPlainText()

    def _use_custom_model(self) -> None:
        model_id = self.custom_model.text().strip()
        if not model_id:
            return
        index = self.model_combo.findData(model_id)
        if index < 0:
            self.model_combo.addItem(f"{model_id}  —  custom", model_id)
            index = self.model_combo.count() - 1
        self.custom_model.clear()
        self.model_combo.setCurrentIndex(index)  # triggers _on_model_changed

    def _download_model(self) -> None:
        model_id = self.model_combo.currentData()
        if not model_id:
            return
        self.download_btn.setEnabled(False)
        self.window.submit(
            download_model_task,
            model_id,
            on_finished=lambda: self.download_btn.setEnabled(True),
            on_error=self._on_download_error,
            task_name=f"Downloading {model_id}",
        )

    def _on_download_error(self, kind: str, message: str) -> None:
        self.window.show_error(kind, message, "Model download")

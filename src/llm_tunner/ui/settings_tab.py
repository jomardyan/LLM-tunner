"""Settings tab: base model picker, embedding model, retrieval depth, device info."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    DEFAULT_EMBEDDING_MODEL,
    HIGH_QUALITY_EMBEDDING_MODEL,
)
from ..core.models import registry
from ..workers.tasks import huggingface_login_task

if TYPE_CHECKING:
    from ..main_window import MainWindow


class SettingsTab(QWidget):
    def __init__(self, window: MainWindow) -> None:
        super().__init__()
        self.window = window

        root = QVBoxLayout(self)
        form = QFormLayout()

        self.model_combo = QComboBox()
        for entry in registry():
            self.model_combo.addItem(
                f"{entry.model_id}  —  {entry.size}, {entry.license}", entry.model_id
            )
        self._select_current(self.model_combo, self.window.state.base_model)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        form.addRow("Base model:", self.model_combo)

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

        root.addLayout(form)

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
        root.addStretch()

    def update_device_banner(self, text: str) -> None:
        self.device_banner.setText(text)

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
            on_result=lambda _result: self._set_hf_status(True),
            on_error=self._on_hf_login_error,
            on_finished=lambda: self.hf_login_btn.setEnabled(True),
        )

    def _on_hf_login_error(self, kind: str, message: str) -> None:
        self._set_hf_status(False, f"Login failed ({kind}: {message})")

    def _select_current(self, combo: QComboBox, value: str) -> None:
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return

    def _on_model_changed(self) -> None:
        model_id = self.model_combo.currentData()
        self.window.state.base_model = model_id
        self.window.settings.base_model = model_id
        self.window.notify(f"Base model set to {model_id}")

    def _on_embed_changed(self) -> None:
        model = self.embed_combo.currentData()
        self.window.state.embedding_model = model
        self.window.settings.embedding_model = model
        self.window.rag_tab.refresh_documents()

    def _on_topk_changed(self, value: int) -> None:
        self.window.settings.top_k = value

"""Settings tab: base model picker, embedding model, retrieval depth, device info."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    DEFAULT_EMBEDDING_MODEL,
    HIGH_QUALITY_EMBEDDING_MODEL,
)
from ..core.models import registry

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
        banner = QLabel(self.window.device.capability_banner())
        banner.setWordWrap(True)
        root.addWidget(banner)
        root.addStretch()

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

"""Fine-tune tab: generate a Q&A dataset from PDFs and run QLoRA, with a live loss curve.

Note the prominent reminder that fine-tuning adapts *style/behaviour*, not facts — use
the RAG tab for injecting document knowledge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..workers.tasks import finetune_task, generate_dataset_task
from .widgets.loss_plot import LossPlot

if TYPE_CHECKING:
    from ..main_window import MainWindow


_NOTE = (
    "Fine-tuning teaches the model a <b>style/format/behaviour</b>, not new facts. "
    "To make the model <i>know</i> your documents, use the Knowledge (RAG) tab. "
    "Real QLoRA (4-bit) needs an NVIDIA GPU; otherwise training falls back to small models."
)


class FineTuneTab(QWidget):
    def __init__(self, window: MainWindow) -> None:
        super().__init__()
        self.window = window
        self._dataset_rows: list[dict] = []
        self._handle = None
        self._active_phase = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        title = QLabel("Fine-tune")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        note = QLabel(_NOTE)
        note.setWordWrap(True)
        root.addWidget(note)

        gen_row = QHBoxLayout()
        self.use_llm = QCheckBox("Use base model to generate Q&A (slower, higher quality)")
        self.gen_btn = QPushButton("1. Generate dataset from PDFs")
        self.gen_btn.setObjectName("primary")
        self.gen_btn.clicked.connect(self._generate)
        gen_row.addWidget(self.gen_btn)
        gen_row.addWidget(self.use_llm)
        gen_row.addStretch()
        root.addLayout(gen_row)

        train_row = QHBoxLayout()
        self.train_btn = QPushButton("2. Start QLoRA fine-tune")
        self.train_btn.setEnabled(False)
        self.train_btn.clicked.connect(self._train)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        train_row.addWidget(self.train_btn)
        train_row.addWidget(self.stop_btn)
        train_row.addStretch()
        root.addLayout(train_row)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        self.metrics = QLabel("No training metrics yet.")
        self.metrics.setObjectName("muted")
        root.addWidget(self.metrics)

        self.loss_plot = LossPlot()
        root.addWidget(self.loss_plot, 1)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        root.addWidget(self.log, 1)

    def refresh_documents(self) -> None:
        n = len(self.window.state.documents)
        self.gen_btn.setText(f"1. Generate dataset from PDFs ({n})")

    # -- dataset generation ------------------------------------------------------
    def _generate(self) -> None:
        pdfs = list(self.window.state.documents)
        if not pdfs:
            self.window.notify("Add PDFs in the Documents tab first.")
            return
        model = self.window.state.base_model if self.use_llm.isChecked() else None
        from ..core.cancellation import CancelHandle

        self._handle = CancelHandle()
        self._active_phase = "dataset generation"
        self.gen_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.stop_btn.setText("Stop dataset generation")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.log.appendPlainText("Generating Q&A dataset…")
        self.window.submit(
            generate_dataset_task,
            pdfs,
            model,
            self._handle,
            on_progress=self._on_progress,
            on_log=self.log.appendPlainText,
            on_result=self._on_dataset,
            on_error=self._on_error,
            on_finished=self._on_generation_finished,
            task_name="Generating training dataset",
        )

    def _on_progress(self, done: int, total: int, message: str) -> None:
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(done)
        if message:
            self.window.notify(message)

    def _on_dataset(self, result: dict) -> None:
        if result.get("cancelled"):
            self.log.appendPlainText("Dataset generation stopped. No partial dataset was saved.")
            self.metrics.setText("Dataset generation stopped.")
            return
        self._dataset_rows = result["rows"]
        self.log.appendPlainText(f"Dataset: {result['pairs']} pairs saved to {result['path']}")
        self.metrics.setText(
            f"Dataset: {result['pairs']} pairs from {result['chunks']} chunks in "
            f"{result['elapsed_seconds']:.1f}s"
        )
        self.train_btn.setEnabled(bool(self._dataset_rows))

    # -- training ----------------------------------------------------------------
    def _train(self) -> None:
        if not self._dataset_rows:
            return
        from ..core.training import TrainHandle

        self._handle = TrainHandle()
        self._active_phase = "fine-tuning"
        self.loss_plot.clear()
        self.gen_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.stop_btn.setText("Stop fine-tuning")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.log.appendPlainText("Training started…")
        self.window.submit(
            finetune_task,
            self.window.state.base_model,
            self._dataset_rows,
            "pdf_adapter",
            self._handle,
            on_metric=self._on_metric,
            on_log=self.log.appendPlainText,
            on_result=self._on_trained,
            on_finished=self._on_train_finished,
            on_error=self._on_error,
            task_name="Fine-tuning model",
        )

    def _on_metric(self, metric) -> None:
        self.loss_plot.add_metric(metric)
        if getattr(metric, "loss", None) is not None:
            self.log.appendPlainText(f"step {metric.step}: loss={metric.loss:.4f}")
        values = [f"Step {metric.step}"]
        if metric.loss is not None:
            values.append(f"loss {metric.loss:.4f}")
        if metric.learning_rate is not None:
            values.append(f"LR {metric.learning_rate:.2e}")
        if metric.epoch is not None:
            values.append(f"epoch {metric.epoch:.2f}")
        if metric.grad_norm is not None:
            values.append(f"grad {metric.grad_norm:.3f}")
        self.metrics.setText(" · ".join(values))

    def _stop(self) -> None:
        if self._handle:
            self._handle.request_stop()
            self.stop_btn.setEnabled(False)
            self.stop_btn.setText("Stopping…")
            unit = "step" if self._active_phase == "fine-tuning" else "item"
            self.log.appendPlainText(f"Stop requested — finishing current {unit}…")

    def _on_trained(self, result: dict) -> None:
        self.window.state.adapter_path = result["adapter_path"]
        mode = "QLoRA 4-bit" if result["used_4bit"] else "LoRA (no 4-bit)"
        loss_str = f"{result['final_loss']:.4f}" if result["final_loss"] is not None else "N/A"
        self.log.appendPlainText(
            f"Done [{mode}]: {result['steps']} steps, final loss "
            f"{loss_str}. Adapter: {result['adapter_path']}"
        )
        self.metrics.setText(
            f"{mode} · {result['steps']} steps · {result['steps_per_second']:.2f} step/s · "
            f"peak VRAM {result['peak_vram_mb'] / 1024:.2f} GB · "
            f"{result['elapsed_seconds']:.1f}s"
        )
        self.window.chat_tab.refresh_kb()
        self.window.notify("Fine-tuning complete — adapter loaded for chat.")

    def _on_train_finished(self) -> None:
        self.train_btn.setEnabled(True)
        self.gen_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("Stop")
        self.progress.setVisible(False)
        self._handle = None
        self._active_phase = ""

    def _on_generation_finished(self) -> None:
        self.gen_btn.setEnabled(True)
        self.train_btn.setEnabled(bool(self._dataset_rows))
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("Stop")
        self.progress.setVisible(False)
        self._handle = None
        self._active_phase = ""

    def _on_error(self, kind: str, message: str) -> None:
        self.log.appendPlainText(self.window.show_error(kind, message, "Fine-tuning"))

"""Fine-tune tab: generate a Q&A dataset from PDFs and run QLoRA, with a live loss curve.

Note the prominent reminder that fine-tuning adapts *style/behaviour*, not facts — use
the RAG tab for injecting document knowledge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
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

        root.addWidget(QLabel("Hyperparameters"))
        root.addLayout(self._build_hyperparameter_form())

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

    # -- hyperparameters ---------------------------------------------------------
    def _build_hyperparameter_form(self) -> QFormLayout:
        """Editable QLoRA/LoRA hyperparameters, seeded from (and persisted to) settings."""
        form = QFormLayout()
        s = self.window.settings

        self.lr = QDoubleSpinBox()
        self.lr.setRange(0.000001, 1.0)
        self.lr.setDecimals(6)
        self.lr.setSingleStep(0.0001)
        self.lr.setValue(s.learning_rate)
        self.lr.valueChanged.connect(lambda v: setattr(self.window.settings, "learning_rate", v))
        form.addRow("Learning rate:", self.lr)

        self.epochs = QDoubleSpinBox()
        self.epochs.setRange(0.1, 100.0)
        self.epochs.setDecimals(1)
        self.epochs.setSingleStep(0.5)
        self.epochs.setValue(s.num_train_epochs)
        self.epochs.valueChanged.connect(
            lambda v: setattr(self.window.settings, "num_train_epochs", v)
        )
        form.addRow("Epochs:", self.epochs)

        self.batch = QSpinBox()
        self.batch.setRange(1, 64)
        self.batch.setValue(s.per_device_train_batch_size)
        self.batch.valueChanged.connect(
            lambda v: setattr(self.window.settings, "per_device_train_batch_size", v)
        )
        form.addRow("Batch size:", self.batch)

        self.grad_accum = QSpinBox()
        self.grad_accum.setRange(1, 256)
        self.grad_accum.setValue(s.gradient_accumulation_steps)
        self.grad_accum.valueChanged.connect(
            lambda v: setattr(self.window.settings, "gradient_accumulation_steps", v)
        )
        form.addRow("Gradient accumulation:", self.grad_accum)

        self.max_seq = QSpinBox()
        self.max_seq.setRange(64, 8192)
        self.max_seq.setSingleStep(64)
        self.max_seq.setValue(s.max_seq_length)
        self.max_seq.valueChanged.connect(
            lambda v: setattr(self.window.settings, "max_seq_length", v)
        )
        form.addRow("Max sequence length:", self.max_seq)

        self.lora_r = QSpinBox()
        self.lora_r.setRange(1, 256)
        self.lora_r.setValue(s.lora_r)
        self.lora_r.valueChanged.connect(lambda v: setattr(self.window.settings, "lora_r", v))
        form.addRow("LoRA rank (r):", self.lora_r)

        self.lora_alpha = QSpinBox()
        self.lora_alpha.setRange(1, 512)
        self.lora_alpha.setValue(s.lora_alpha)
        self.lora_alpha.valueChanged.connect(
            lambda v: setattr(self.window.settings, "lora_alpha", v)
        )
        form.addRow("LoRA alpha:", self.lora_alpha)

        self.lora_dropout = QDoubleSpinBox()
        self.lora_dropout.setRange(0.0, 0.9)
        self.lora_dropout.setDecimals(2)
        self.lora_dropout.setSingleStep(0.05)
        self.lora_dropout.setValue(s.lora_dropout)
        self.lora_dropout.valueChanged.connect(
            lambda v: setattr(self.window.settings, "lora_dropout", v)
        )
        form.addRow("LoRA dropout:", self.lora_dropout)

        return form

    def _train_overrides(self) -> dict:
        return {
            "learning_rate": self.lr.value(),
            "num_train_epochs": self.epochs.value(),
            "per_device_train_batch_size": self.batch.value(),
            "gradient_accumulation_steps": self.grad_accum.value(),
            "max_seq_length": self.max_seq.value(),
            "lora_r": self.lora_r.value(),
            "lora_alpha": self.lora_alpha.value(),
            "lora_dropout": self.lora_dropout.value(),
        }

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
            train_overrides=self._train_overrides(),
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

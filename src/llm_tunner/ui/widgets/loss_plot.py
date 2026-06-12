"""Live training-loss plot backed by pyqtgraph.

Append points as :class:`~llm_tunner.core.training.TrainMetric` objects arrive on the
worker's ``metric`` signal. pyqtgraph redraws cheaply, so per-step updates stay smooth.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget


class LossPlot(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plot = pg.PlotWidget()
        self._plot.setBackground("w")
        self._plot.setLabel("left", "loss")
        self._plot.setLabel("bottom", "step")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._curve = self._plot.plot(pen=pg.mkPen(width=2))
        layout.addWidget(self._plot)

        self._steps: list[int] = []
        self._losses: list[float] = []

    def clear(self) -> None:
        self._steps.clear()
        self._losses.clear()
        self._curve.setData([], [])

    def add_metric(self, metric) -> None:
        """Slot for the worker's ``metric`` signal. Ignores entries without a loss."""
        loss = getattr(metric, "loss", None)
        if loss is None:
            return
        self._steps.append(int(getattr(metric, "step", len(self._steps))))
        self._losses.append(float(loss))
        self._curve.setData(self._steps, self._losses)

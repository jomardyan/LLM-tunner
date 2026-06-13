"""Cooperative cancellation primitives shared by long-running core workflows."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class CancelHandle:
    """Thread-safe stop flag for tasks that can cancel between work units."""

    stop_event: threading.Event = field(default_factory=threading.Event)

    def request_stop(self) -> None:
        self.stop_event.set()

    def is_stopped(self) -> bool:
        return self.stop_event.is_set()

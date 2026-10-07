"""One settings save owns its immutable input until physical settlement."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from pixelup.app_config import AppConfig, ConfigSaveResult

SAVE_WAIT_SECONDS = 5


@dataclass(frozen=True, slots=True)
class ConfigSaveOutcome:
    result: ConfigSaveResult | None = None
    error: Exception | None = None


class ConfigSave(QObject):
    completed = Signal(object)
    timed_out = Signal()

    def __init__(
        self,
        candidate: AppConfig,
        previous: AppConfig,
        save: Callable[[AppConfig, AppConfig], ConfigSaveResult],
    ) -> None:
        super().__init__()
        self.candidate = candidate
        self.previous = previous
        self._save = save
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(SAVE_WAIT_SECONDS * 1000)
        self._timer.timeout.connect(self.timed_out)
        self.completed.connect(self._stop_timer)

    def start(self) -> None:
        self._timer.start()
        try:
            threading.Thread(target=self._run, name="pixelup-config-save", daemon=True).start()
        except Exception as exc:
            self.completed.emit(ConfigSaveOutcome(error=exc))

    def _run(self) -> None:
        try:
            outcome = ConfigSaveOutcome(result=self._save(self.candidate, self.previous))
        except Exception as exc:
            outcome = ConfigSaveOutcome(error=exc)
        self.completed.emit(outcome)

    @Slot(object)
    def _stop_timer(self, _outcome: ConfigSaveOutcome) -> None:
        self._timer.stop()

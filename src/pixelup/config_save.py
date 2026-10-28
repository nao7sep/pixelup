"""One settings save owns its immutable input until physical settlement."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal

from pixelup.app_config import AppConfig, ConfigSaveResult
from pixelup.read_threads import run_in_background


@dataclass(frozen=True, slots=True)
class ConfigSaveOutcome:
    result: ConfigSaveResult | None = None
    error: Exception | None = None


class ConfigSave(QObject):
    completed = Signal(object)

    def __init__(
        self,
        candidate: AppConfig,
        previous: AppConfig,
        save: Callable[[AppConfig, AppConfig], ConfigSaveResult],
        parent: QObject | None = None,
    ) -> None:
        # With a parent, Qt alone deletes this object: the save thread's reference
        # can then never be the one that deletes it off the window thread.
        super().__init__(parent)
        self.candidate = candidate
        self.previous = previous
        self._save = save

    def start(self) -> None:
        # Local storage gets no timeout: the save is awaited to its end.
        run_in_background(self._run)

    def _run(self) -> None:
        try:
            outcome = ConfigSaveOutcome(result=self._save(self.candidate, self.previous))
        except Exception as exc:
            outcome = ConfigSaveOutcome(error=exc)
        self.completed.emit(outcome)

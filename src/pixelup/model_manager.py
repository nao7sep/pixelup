from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from PySide6.QtCore import QObject, QThread, Signal, Slot

from pixelup.errors import ErrorCode, PixelupError
from pixelup.model_management import MANAGED_ARTIFACT_NAMES, artifact_size_bytes
from pixelup.models import download_model, model_is_ready
from pixelup.session_log import log

_DOWNLOAD_TIMEOUT_SECONDS = 600
_LOCK_TIMEOUT_SECONDS = 600


@dataclass(frozen=True, slots=True)
class ModelOperation:
    kind: Literal["idle", "running", "failed", "cancelled"] = "idle"
    artifact_names: tuple[str, ...] = ()
    current_artifact: str | None = None
    completed_bytes: int = 0
    total_bytes: int = 0
    error: str = ""
    cancelling: bool = False


class ModelInstallWorker(QObject):
    progress = Signal(str, int, int)
    waiting = Signal(str)
    finished = Signal(bool, bool, str)

    def __init__(self, models_dir: Path, artifact_names: tuple[str, ...], *, force: bool) -> None:
        super().__init__()
        self._models_dir = models_dir
        self._artifact_names = artifact_names
        self._force = force
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        return self._cancel_requested

    @Slot()
    def run(self) -> None:
        try:
            total_bytes = artifact_size_bytes(self._artifact_names)
            completed_bytes = 0
            for name in self._artifact_names:
                if self._is_cancelled():
                    raise PixelupError(ErrorCode.JOB_CANCELLED, "Model installation cancelled.")

                def report_progress(
                    _model: str,
                    done: int,
                    _total: int | None,
                    *,
                    artifact_name: str = name,
                    completed: int = completed_bytes,
                ) -> None:
                    self.progress.emit(
                        artifact_name,
                        min(total_bytes, completed + done),
                        total_bytes,
                    )

                download_model(
                    self._models_dir,
                    name,
                    download_timeout=_DOWNLOAD_TIMEOUT_SECONDS,
                    lock_timeout=_LOCK_TIMEOUT_SECONDS,
                    on_download=report_progress,
                    on_waiting=lambda model, _elapsed: self.waiting.emit(model),
                    should_cancel=self._is_cancelled,
                    force=self._force,
                )
                completed_bytes += artifact_size_bytes((name,))
                self.progress.emit(name, completed_bytes, total_bytes)
        except PixelupError as exc:
            cancelled = exc.code == ErrorCode.JOB_CANCELLED
            message = f"{exc.message} {exc.hint}" if exc.hint else exc.message
            self.finished.emit(False, cancelled, message)
        except Exception as exc:  # noqa: BLE001 - every worker outcome must settle.
            log.exception("models.install_failed_unexpectedly")
            self.finished.emit(False, False, f"Unexpected error: {exc}")
        else:
            self.finished.emit(True, False, "")


class ModelManager(QObject):
    """Application-owned readiness and acquisition state for PixelUp's model cache."""

    changed = Signal()
    idle = Signal()

    def __init__(self, models_dir: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.models_dir = models_dir
        self._ready_names = self._read_ready_names()
        self._operation = ModelOperation()
        self._thread: QThread | None = None
        self._worker: ModelInstallWorker | None = None
        self._install_result: tuple[bool, bool, str] | None = None
        self._shutting_down = False

    @property
    def operation(self) -> ModelOperation:
        return self._operation

    @property
    def ready_names(self) -> frozenset[str]:
        return self._ready_names

    def missing(self, artifact_names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(name for name in artifact_names if name not in self._ready_names)

    def ready_count(self) -> tuple[int, int]:
        return len(self._ready_names), len(MANAGED_ARTIFACT_NAMES)

    def refresh_readiness(self) -> None:
        ready_names = self._read_ready_names()
        if ready_names == self._ready_names:
            return
        self._ready_names = ready_names
        self.changed.emit()

    def install(self, artifact_names: tuple[str, ...], *, force: bool) -> bool:
        if self._worker is not None or self._shutting_down:
            return False
        names = tuple(dict.fromkeys(artifact_names))
        if not names:
            return False
        total_bytes = artifact_size_bytes(names)
        self._operation = ModelOperation(
            kind="running",
            artifact_names=names,
            total_bytes=total_bytes,
        )
        self._install_result = None
        log.info("models.install_requested", artifacts=list(names), force=force)

        thread = QThread(self)
        worker = ModelInstallWorker(self.models_dir, names, force=force)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._show_progress)
        worker.waiting.connect(self._show_waiting)
        worker.finished.connect(self._worker_finished)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        thread.finished.connect(self._thread_finished)
        self._thread = thread
        self._worker = worker
        self.changed.emit()
        thread.start()
        return True

    def cancel(self) -> bool:
        if self._worker is None:
            return False
        self._operation = replace(self._operation, cancelling=True)
        self._worker.request_cancel()
        self.changed.emit()
        return True

    def begin_shutdown(self) -> None:
        self._shutting_down = True
        self.cancel()

    def cleanup_for_quit(self) -> bool:
        return self._thread is None

    @Slot(str, int, int)
    def _show_progress(self, name: str, done: int, total: int) -> None:
        self._operation = replace(
            self._operation,
            current_artifact=name,
            completed_bytes=done,
            total_bytes=total,
        )
        self.changed.emit()

    @Slot(str)
    def _show_waiting(self, name: str) -> None:
        self._operation = replace(self._operation, current_artifact=name)
        self.changed.emit()

    @Slot(bool, bool, str)
    def _worker_finished(self, succeeded: bool, cancelled: bool, message: str) -> None:
        self._install_result = (succeeded, cancelled, message)
        log.info(
            "models.install_finished",
            succeeded=succeeded,
            cancelled=cancelled,
            reason=message,
        )

    @Slot()
    def _thread_finished(self) -> None:
        result = self._install_result
        self._install_result = None
        self._worker = None
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.deleteLater()

        # A multi-artifact operation may have completed some atomic downloads
        # before a later artifact failed or cancellation arrived. Read the cache
        # once at this operation boundary so every view receives the real partial
        # result without doing filesystem work while rendering.
        self._ready_names = self._read_ready_names()
        if result is None:
            self._operation = replace(
                self._operation,
                kind="failed",
                error="Installation stopped.",
                cancelling=False,
            )
        else:
            succeeded, cancelled, message = result
            if succeeded:
                self._operation = ModelOperation()
            elif cancelled:
                self._operation = replace(
                    self._operation,
                    kind="cancelled",
                    error="",
                    cancelling=False,
                )
            else:
                self._operation = replace(
                    self._operation,
                    kind="failed",
                    error=message,
                    cancelling=False,
                )
        self.changed.emit()
        self.idle.emit()

    def _read_ready_names(self) -> frozenset[str]:
        return frozenset(
            name for name in MANAGED_ARTIFACT_NAMES if model_is_ready(self.models_dir, name)
        )

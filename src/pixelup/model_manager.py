from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import count
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
    id: int
    kind: Literal["running", "completed", "failed", "cancelled"]
    artifact_names: tuple[str, ...]
    current_artifact: str | None = None
    completed_bytes: int = 0
    total_bytes: int = 0
    error: str = ""
    cancelling: bool = False


class ModelInstallWorker(QObject):
    progress = Signal(int, str, int, int)
    waiting = Signal(int, str)
    finished = Signal(int, bool, bool, str)

    def __init__(
        self,
        operation_id: int,
        models_dir: Path,
        artifact_names: tuple[str, ...],
        *,
        force: bool,
    ) -> None:
        super().__init__()
        self._operation_id = operation_id
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
                        self._operation_id,
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
                    on_waiting=lambda model, _elapsed: self.waiting.emit(
                        self._operation_id, model
                    ),
                    should_cancel=self._is_cancelled,
                    force=self._force,
                )
                completed_bytes += artifact_size_bytes((name,))
                self.progress.emit(
                    self._operation_id,
                    name,
                    completed_bytes,
                    total_bytes,
                )
        except PixelupError as exc:
            cancelled = exc.code == ErrorCode.JOB_CANCELLED
            message = f"{exc.message} {exc.hint}" if exc.hint else exc.message
            self.finished.emit(self._operation_id, False, cancelled, message)
        except Exception as exc:  # noqa: BLE001 - every worker outcome must settle.
            log.exception("models.install_failed_unexpectedly")
            self.finished.emit(
                self._operation_id,
                False,
                False,
                f"Unexpected error: {exc}",
            )
        else:
            self.finished.emit(self._operation_id, True, False, "")


class ModelManager(QObject):
    """Application-owned readiness and concurrent acquisition state."""

    changed = Signal()
    idle = Signal()
    cancelled = Signal(object)

    def __init__(self, models_dir: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.models_dir = models_dir
        self._ready_names = self._read_ready_names()
        self._operation_ids = count(1)
        self._operations: dict[int, ModelOperation] = {}
        self._threads: dict[int, QThread] = {}
        self._workers: dict[int, ModelInstallWorker] = {}
        self._install_results: dict[int, tuple[bool, bool, str]] = {}
        self._shutting_down = False

    @property
    def operations(self) -> tuple[ModelOperation, ...]:
        return tuple(self._operations.values())

    @property
    def active_operations(self) -> tuple[ModelOperation, ...]:
        return tuple(operation for operation in self.operations if operation.kind == "running")

    @property
    def failed_operations(self) -> tuple[ModelOperation, ...]:
        return tuple(operation for operation in self.operations if operation.kind == "failed")

    @property
    def ready_names(self) -> frozenset[str]:
        return self._ready_names

    def operations_for(self, artifact_names: tuple[str, ...]) -> tuple[ModelOperation, ...]:
        names = set(artifact_names)
        return tuple(
            operation
            for operation in self.operations
            if names.intersection(operation.artifact_names)
        )

    def active_for(self, artifact_names: tuple[str, ...]) -> tuple[ModelOperation, ...]:
        return tuple(
            operation
            for operation in self.operations_for(artifact_names)
            if operation.kind == "running"
        )

    def missing(self, artifact_names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(name for name in artifact_names if name not in self._ready_names)

    def available_to_install(self, artifact_names: tuple[str, ...]) -> tuple[str, ...]:
        active = {
            name
            for operation in self.active_operations
            for name in operation.artifact_names
        }
        return tuple(name for name in artifact_names if name not in active)

    def ready_count(self) -> tuple[int, int]:
        return len(self._ready_names), len(MANAGED_ARTIFACT_NAMES)

    def aggregate_progress(self) -> tuple[int, int]:
        operations = tuple(
            operation
            for operation in self.operations
            if operation.kind in {"running", "completed"}
        )
        return (
            sum(operation.completed_bytes for operation in operations),
            sum(operation.total_bytes for operation in operations),
        )

    def refresh_readiness(self) -> None:
        ready_names = self._read_ready_names()
        if ready_names == self._ready_names:
            return
        self._ready_names = ready_names
        self.changed.emit()

    def install(self, artifact_names: tuple[str, ...], *, force: bool) -> int | None:
        if self._shutting_down:
            return None
        names = tuple(dict.fromkeys(artifact_names))
        if not names or self.active_for(names):
            return None

        # A retry supersedes terminal state for the same artifacts. Unrelated
        # failures remain visible while their own row awaits a retry.
        names_set = set(names)
        self._operations = {
            operation_id: operation
            for operation_id, operation in self._operations.items()
            if operation.kind == "running"
            or not names_set.intersection(operation.artifact_names)
        }

        operation_id = next(self._operation_ids)
        operation = ModelOperation(
            id=operation_id,
            kind="running",
            artifact_names=names,
            total_bytes=artifact_size_bytes(names),
        )
        self._operations[operation_id] = operation
        log.info(
            "models.install_requested",
            operation_id=operation_id,
            artifacts=list(names),
            force=force,
        )

        thread = QThread(self)
        worker = ModelInstallWorker(operation_id, self.models_dir, names, force=force)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._show_progress)
        worker.waiting.connect(self._show_waiting)
        worker.finished.connect(self._worker_finished)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        thread.setProperty("operation_id", operation_id)
        thread.finished.connect(self._thread_finished)
        self._threads[operation_id] = thread
        self._workers[operation_id] = worker
        self.changed.emit()
        thread.start()
        return operation_id

    def cancel(self, operation_id: int) -> bool:
        worker = self._workers.get(operation_id)
        operation = self._operations.get(operation_id)
        if worker is None or operation is None or operation.kind != "running":
            return False
        self._operations[operation_id] = replace(operation, cancelling=True)
        worker.request_cancel()
        self.changed.emit()
        return True

    def cancel_for(self, artifact_names: tuple[str, ...]) -> bool:
        cancelled = False
        for operation in self.active_for(artifact_names):
            cancelled = self.cancel(operation.id) or cancelled
        return cancelled

    def cancel_all(self) -> bool:
        cancelled = False
        for operation in self.active_operations:
            cancelled = self.cancel(operation.id) or cancelled
        return cancelled

    def begin_shutdown(self) -> None:
        self._shutting_down = True
        self.cancel_all()

    def cleanup_for_quit(self) -> bool:
        return not self._threads

    @Slot(int, str, int, int)
    def _show_progress(self, operation_id: int, name: str, done: int, total: int) -> None:
        operation = self._operations.get(operation_id)
        if operation is None or operation.kind != "running":
            return
        self._operations[operation_id] = replace(
            operation,
            current_artifact=name,
            completed_bytes=done,
            total_bytes=total,
        )
        self.changed.emit()

    @Slot(int, str)
    def _show_waiting(self, operation_id: int, name: str) -> None:
        operation = self._operations.get(operation_id)
        if operation is None or operation.kind != "running":
            return
        self._operations[operation_id] = replace(operation, current_artifact=name)
        self.changed.emit()

    @Slot(int, bool, bool, str)
    def _worker_finished(
        self,
        operation_id: int,
        succeeded: bool,
        cancelled: bool,
        message: str,
    ) -> None:
        self._install_results[operation_id] = (succeeded, cancelled, message)
        log.info(
            "models.install_finished",
            operation_id=operation_id,
            succeeded=succeeded,
            cancelled=cancelled,
            reason=message,
        )

    @Slot()
    def _thread_finished(self) -> None:
        sender = self.sender()
        if not isinstance(sender, QThread):
            return
        operation_id = int(sender.property("operation_id"))
        result = self._install_results.pop(operation_id, None)
        self._workers.pop(operation_id, None)
        thread = self._threads.pop(operation_id, None)
        if thread is not None:
            thread.deleteLater()

        # Concurrent operations publish disjoint artifacts atomically. Re-read the
        # shared cache at each operation boundary so every view sees partial success.
        self._ready_names = self._read_ready_names()
        operation = self._operations.get(operation_id)
        if operation is not None:
            if result is None:
                self._operations[operation_id] = replace(
                    operation,
                    kind="failed",
                    error="Installation stopped.",
                    cancelling=False,
                )
            else:
                succeeded, cancelled, message = result
                if succeeded:
                    self._operations[operation_id] = replace(
                        operation,
                        kind="completed",
                        completed_bytes=operation.total_bytes,
                        cancelling=False,
                    )
                elif cancelled:
                    self._operations[operation_id] = replace(
                        operation,
                        kind="cancelled",
                        error="",
                        cancelling=False,
                    )
                else:
                    self._operations[operation_id] = replace(
                        operation,
                        kind="failed",
                        error=message,
                        cancelling=False,
                    )
        if not self._threads:
            self._operations = {
                operation_id: operation
                for operation_id, operation in self._operations.items()
                if operation.kind != "completed"
            }
        cancelled_operation = self._operations.get(operation_id)
        if cancelled_operation is not None and cancelled_operation.kind == "cancelled":
            self.cancelled.emit(cancelled_operation.artifact_names)
        # Cancellation is an event, not durable row state. Retaining it would make
        # a later, unrelated preflight for the same artifact look newly cancelled.
        self._operations = {
            operation_id: operation
            for operation_id, operation in self._operations.items()
            if operation.kind != "cancelled"
        }
        self.changed.emit()
        if not self._threads:
            self.idle.emit()

    def _read_ready_names(self) -> frozenset[str]:
        return frozenset(
            name for name in MANAGED_ARTIFACT_NAMES if model_is_ready(self.models_dir, name)
        )

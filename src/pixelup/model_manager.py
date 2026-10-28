from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from itertools import count
from pathlib import Path
from threading import Event
from typing import Literal

from PySide6.QtCore import QObject, QThread, Signal, Slot

from pixelup.errors import ErrorCode, PixelupError, user_text
from pixelup.i18n.localizer import english
from pixelup.i18n.message import Message
from pixelup.model_management import MANAGED_ARTIFACT_NAMES, artifact_size_bytes
from pixelup.models import download_model, model_is_ready
from pixelup.read_threads import run_in_background
from pixelup.session_log import log

_DOWNLOAD_TIMEOUT_SECONDS = 600

INSTALL_FAILED = Message("modelInstall.failedFallback")



@dataclass(frozen=True, slots=True)
class ModelOperation:
    id: int
    kind: Literal["queued", "running", "completed", "failed", "cancelled"]
    artifact_names: tuple[str, ...]
    current_artifact: str | None = None
    completed_bytes: int = 0
    total_bytes: int = 0
    # Why the operation failed, rendered by the dialog in the reader's language.
    error: Message | None = None
    cancelling: bool = False


class ModelInstallWorker(QObject):
    """Runs one install at a time on the manager's thread and reports each outcome."""

    progress = Signal(int, str, int, int)
    finished = Signal(int, bool, bool, object)

    @Slot(object, object)
    def run_install(self, command: _Install, models_dir: Path) -> None:
        outcome: tuple[bool, bool, Message | None] = (False, False, INSTALL_FAILED)
        try:
            outcome = self._run(command, models_dir)
        finally:
            # Every install reports, so the manager always moves on to the next.
            self.finished.emit(command.operation_id, *outcome)

    def _run(self, command: _Install, models_dir: Path) -> tuple[bool, bool, Message | None]:
        name: str | None = None
        operation_id, names = command.operation_id, command.artifact_names
        try:
            total_bytes = artifact_size_bytes(names)
            completed_bytes = 0
            for name in names:
                if command.cancel.is_set():
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
                        operation_id,
                        artifact_name,
                        min(total_bytes, completed + done),
                        total_bytes,
                    )

                download_model(
                    models_dir,
                    name,
                    download_timeout=_DOWNLOAD_TIMEOUT_SECONDS,
                    on_download=report_progress,
                    should_cancel=command.cancel.is_set,
                    force=command.force,
                )
                completed_bytes += artifact_size_bytes((name,))
                self.progress.emit(operation_id, name, completed_bytes, total_bytes)
        except PixelupError as exc:
            cancelled = exc.code == ErrorCode.JOB_CANCELLED
            if not cancelled:
                log.exception(
                    "models.install_failed",
                    operation_id=operation_id,
                    artifact=name,
                    artifacts=list(names),
                    code=exc.code.value,
                    error_details=exc.details,
                )
            return False, cancelled, user_text(exc, internal_fallback=INSTALL_FAILED)
        except Exception:  # noqa: BLE001 - every worker outcome must settle.
            log.exception(
                "models.install_failed",
                operation_id=operation_id,
                artifact=name,
                artifacts=list(names),
            )
            return False, False, INSTALL_FAILED
        return True, False, None


@dataclass(frozen=True, slots=True)
class _Install:
    operation_id: int
    artifact_names: tuple[str, ...]
    force: bool
    cancel: Event


@dataclass(frozen=True, slots=True)
class _FolderChange:
    """Installs requested while the models folder changes wait behind it."""


def read_ready_names(models_dir: Path) -> frozenset[str]:
    """The managed artifacts the models folder holds a usable file for."""
    return frozenset(name for name in MANAGED_ARTIFACT_NAMES if model_is_ready(models_dir, name))


@dataclass(frozen=True, slots=True)
class ModelReadinessResult:
    ready_names: frozenset[str] | None = None
    error: Exception | None = None


class ModelManager(QObject):
    """Application-owned readiness and model installation, one install at a time.

    Installs and models-folder changes take their turn in one command list; the
    installs run on one worker thread. Each download already uses the bandwidth.
    """

    changed = Signal()
    idle = Signal()
    cancelled = Signal(object)
    _scanned = Signal(int, object)
    _run_install = Signal(object, object)
    folder_available = Signal(object, bool)

    def __init__(self, models_dir: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.models_dir = models_dir
        self._ready_names: frozenset[str] = frozenset()
        self.readiness_known = False
        self.readiness_error: Exception | None = None
        # The latest scan request; an older answer never overwrites a newer one.
        self._readiness_version = 0
        self._scanned.connect(self._scan_finished)
        self._operation_ids = count(1)
        self._operations: dict[int, ModelOperation] = {}
        self._commands: deque[_Install | _FolderChange] = deque()
        self._current: _Install | _FolderChange | None = None
        self._shutting_down = False
        self._thread = QThread(self)
        self._worker = ModelInstallWorker()
        self._worker.moveToThread(self._thread)
        self._run_install.connect(self._worker.run_install)
        self._worker.progress.connect(self._show_progress)
        self._worker.finished.connect(self._install_finished)
        thread = self._thread
        # A QThread destroyed while running ends the process; whatever deletes
        # the manager, its thread stops first.
        self.destroyed.connect(lambda: _stop_thread(thread))
        self.rescan()

    @property
    def operations(self) -> tuple[ModelOperation, ...]:
        return tuple(self._operations.values())

    @property
    def active_operations(self) -> tuple[ModelOperation, ...]:
        """The operation actually downloading right now."""
        return tuple(operation for operation in self.operations if operation.kind == "running")

    @property
    def in_progress_operations(self) -> tuple[ModelOperation, ...]:
        """Operations that are running or waiting their turn."""
        return tuple(
            operation for operation in self.operations if operation.kind in {"running", "queued"}
        )

    @property
    def failed_operations(self) -> tuple[ModelOperation, ...]:
        return tuple(operation for operation in self.operations if operation.kind == "failed")

    @property
    def folder_changing(self) -> bool:
        """Whether a models-folder change is waiting or under way."""
        return any(isinstance(command, _FolderChange)
                   for command in (self._current, *self._commands))

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

    def in_progress_for(self, artifact_names: tuple[str, ...]) -> tuple[ModelOperation, ...]:
        return tuple(
            operation
            for operation in self.operations_for(artifact_names)
            if operation.kind in {"running", "queued"}
        )

    def missing(self, artifact_names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(name for name in artifact_names if name not in self._ready_names)

    def available_to_install(self, artifact_names: tuple[str, ...]) -> tuple[str, ...]:
        active = {
            name
            for operation in self.in_progress_operations
            for name in operation.artifact_names
        }
        return tuple(name for name in artifact_names if name not in active)

    def ready_count(self) -> tuple[int, int]:
        return len(self._ready_names), len(MANAGED_ARTIFACT_NAMES)

    def aggregate_progress(self) -> tuple[int, int]:
        operations = tuple(
            operation
            for operation in self.operations
            if operation.kind in {"running", "queued", "completed"}
        )
        return (
            sum(operation.completed_bytes for operation in operations),
            sum(operation.total_bytes for operation in operations),
        )

    def refresh_readiness(self) -> None:
        self.readiness_known = False
        self.rescan(force=True)

    def begin_folder_change(self) -> None:
        """Hold later installs until :meth:`end_folder_change`; installations
        settle before the folder can change, so none is running now."""
        self._commands.append(_FolderChange())
        self._advance()
        self.changed.emit()

    def end_folder_change(self, models_dir: Path) -> None:
        """Switch to ``models_dir`` (unchanged when the change did not land);
        existing files are never moved. Waiting installs then go there."""
        if not isinstance(self._current, _FolderChange):
            raise RuntimeError("no models-folder change is under way")
        self._current = None
        if models_dir != self.models_dir:
            self.models_dir = models_dir
            self.readiness_error = None
            self._ready_names = frozenset()
            self.readiness_known = False
            self._operations = {
                operation_id: operation
                for operation_id, operation in self._operations.items()
                if operation.kind == "queued"
            }
            self.rescan(force=True)
        self._advance()
        self.changed.emit()

    def rescan(self, *, force: bool = False) -> None:
        """Read readiness off the window thread; the latest request wins."""
        if self._shutting_down or (self.in_progress_operations and not force):
            return
        self._readiness_version += 1
        version = self._readiness_version
        if not self.readiness_known:
            self.changed.emit()
        folder = self.models_dir
        run_in_background(lambda: self._scan(folder, version))

    def _scan(self, models_dir: Path, version: int) -> None:
        try:
            result = ModelReadinessResult(ready_names=read_ready_names(models_dir))
        except Exception as exc:  # noqa: BLE001 - a failed scan must still settle readiness.
            log.exception("models.scan_failed", models_dir=str(models_dir))
            result = ModelReadinessResult(error=exc)
        try:
            self._scanned.emit(version, result)
        except RuntimeError:
            pass  # The manager was deleted while the scan ran.

    @Slot(int, object)
    def _scan_finished(self, version: int, result: ModelReadinessResult) -> None:
        if version != self._readiness_version or self._shutting_down:
            return
        changed = (self.readiness_error is not result.error
                   or self.readiness_known != (result.error is None))
        self.readiness_error = result.error
        self.readiness_known = result.error is None
        if result.ready_names is not None:
            changed = self._set_ready_names(result.ready_names) or changed
        if changed:
            self.changed.emit()

    def _set_ready_names(self, ready_names: frozenset[str]) -> bool:
        changed = ready_names != self._ready_names
        self._ready_names = ready_names
        return changed

    def prepare_folder_reveal(self) -> None:
        folder = self.models_dir

        def prepare() -> None:
            ok = True
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except Exception:  # noqa: BLE001 - the Reveal button waits on this answer.
                log.exception("models.reveal_failed", models_dir=str(folder))
                ok = False
            try:
                self.folder_available.emit(folder, ok)
            except RuntimeError:
                pass  # The app exited while storage was unavailable.

        run_in_background(prepare)

    def install(self, artifact_names: tuple[str, ...], *, force: bool) -> int | None:
        """Queue an install; it runs when the earlier commands have ended. An
        artifact already installed by then ends as installed unless forced."""
        if self._shutting_down:
            return None
        names = tuple(dict.fromkeys(artifact_names))
        if not names:
            return None
        # A retry supersedes terminal state for the same artifacts. Unrelated
        # failures remain visible while their own row awaits a retry.
        names_set = set(names)
        self._operations = {
            operation_id: operation
            for operation_id, operation in self._operations.items()
            if operation.kind in {"running", "queued"}
            or not names_set.intersection(operation.artifact_names)
        }
        operation_id = next(self._operation_ids)
        self._operations[operation_id] = ModelOperation(
            id=operation_id,
            kind="queued",
            artifact_names=names,
            total_bytes=artifact_size_bytes(names),
        )
        log.info(
            "models.install_requested",
            operation_id=operation_id,
            artifacts=list(names),
            force=force,
        )
        self._commands.append(_Install(operation_id, names, force, Event()))
        self._advance()
        self.changed.emit()
        return operation_id

    def _advance(self) -> None:
        """Start the next command unless one is under way."""
        if self._current is not None or not self._commands:
            return
        command = self._commands.popleft()
        self._current = command
        if isinstance(command, _FolderChange):
            return
        operation = self._operations[command.operation_id]
        self._operations[command.operation_id] = replace(operation, kind="running")
        if not self._thread.isRunning():
            self._thread.start()
        self._run_install.emit(command, self.models_dir)

    def cancel(self, operation_id: int) -> bool:
        operation = self._operations.get(operation_id)
        if operation is None:
            return False
        if operation.kind == "queued":
            self._commands = deque(
                command for command in self._commands
                if not isinstance(command, _Install) or command.operation_id != operation_id
            )
            del self._operations[operation_id]
            self.changed.emit()
            self.cancelled.emit(operation.artifact_names)
            return True
        current = self._current
        if (operation.kind != "running" or not isinstance(current, _Install)
                or current.operation_id != operation_id):
            return False
        self._operations[operation_id] = replace(operation, cancelling=True)
        current.cancel.set()
        self.changed.emit()
        return True

    def cancel_for(self, artifact_names: tuple[str, ...]) -> bool:
        cancelled = False
        for operation in self.in_progress_for(artifact_names):
            cancelled = self.cancel(operation.id) or cancelled
        return cancelled

    def cancel_all(self) -> bool:
        cancelled = False
        for operation in self.in_progress_operations:
            cancelled = self.cancel(operation.id) or cancelled
        return cancelled

    def begin_shutdown(self) -> None:
        self._shutting_down = True
        self.cancel_all()

    def cleanup_for_quit(self) -> bool:
        """Whether the running install has reported; then the worker thread stops."""
        if isinstance(self._current, _Install):
            return False
        _stop_thread(self._thread)
        return True

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

    @Slot(int, bool, bool, object)
    def _install_finished(
        self,
        operation_id: int,
        succeeded: bool,
        cancelled: bool,
        message: Message | None,
    ) -> None:
        log.info(
            "models.install_finished",
            operation_id=operation_id,
            succeeded=succeeded,
            cancelled=cancelled,
            reason=english().of(message) if message is not None else "",
        )
        self._current = None
        operation = self._operations[operation_id]
        if succeeded:
            operation = replace(operation, kind="completed",
                                completed_bytes=operation.total_bytes, cancelling=False)
        elif cancelled:
            operation = replace(operation, kind="cancelled", error=None, cancelling=False)
        else:
            operation = replace(operation, kind="failed", error=message, cancelling=False)
        self._operations[operation_id] = operation
        self._advance()
        if not self.in_progress_operations:
            self._operations = {
                key: value for key, value in self._operations.items()
                if value.kind != "completed"
            }
        if operation.kind == "cancelled":
            # Cancellation is an event, not durable row state. Retaining it would
            # make a later, unrelated preflight for the same artifact look cancelled.
            del self._operations[operation_id]
            self.cancelled.emit(operation.artifact_names)
        # Re-read the folder at each operation boundary so every view sees
        # partial success.
        self.changed.emit()
        self.refresh_readiness()
        if self._current is None:
            self.idle.emit()


def _stop_thread(thread: QThread) -> None:
    thread.quit()
    thread.wait()

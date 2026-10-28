from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from threading import Event

from PySide6.QtCore import QObject, QThread, Signal, Slot

from pixelup.config import RuntimeDirs, resolve_runtime_dirs
from pixelup.errors import ErrorCode, PixelupError, user_text
from pixelup.i18n.localizer import english
from pixelup.i18n.message import Message
from pixelup.inference import out_of_memory_error
from pixelup.jobs import Job, job_log_payload, options_for_job
from pixelup.output_reservation import assert_output_bundle_available, remove_published_file
from pixelup.session_log import log
from pixelup.sidecar import write_sidecar
from pixelup.upscale import run_upscale

FAILED_FALLBACK = Message("jobs.failedFallback")


def failure_message(exc: PixelupError) -> Message:
    """The queue-row text for a failed job: the diagnosis plus its remedy.

    The row is the only place a processing failure surfaces, so its corrective hint
    must ride along rather than existing only in the log.
    """
    return user_text(
        exc,
        internal_fallback=FAILED_FALLBACK,
    )


JobOutcome = tuple[bool, object, object, object]


class JobWorker(QObject):
    """Runs one job at a time on the runner's thread and reports each outcome."""

    progress = Signal(int, object)
    finished = Signal(int, bool, object, object, object)

    @Slot(object, object, object)
    def run_job(self, job: Job, runtime_dirs: RuntimeDirs | None, cancel: Event) -> None:
        outcome: JobOutcome = (False, FAILED_FALLBACK, {}, [])
        try:
            outcome = self._run(job, runtime_dirs, cancel)
        finally:
            # Every job reports, even one ended by an unexpected error, so the
            # runner always moves on to the next.
            self.finished.emit(job.id, *outcome)

    def _run(self, job: Job, runtime_dirs: RuntimeDirs | None, cancel: Event) -> JobOutcome:
        warnings: list[Message] = []
        published_image: Path | None = None
        sidecar: Path | None = None
        committed = False

        def capture_published_image(published: Path) -> None:
            nonlocal published_image
            published_image = published

        def discard_uncommitted() -> None:
            if not committed:
                for claim in (sidecar, published_image):
                    if claim is not None:
                        remove_published_file(claim)

        def check_cancelled() -> None:
            if cancel.is_set():
                raise PixelupError(ErrorCode.JOB_CANCELLED, "Job cancelled.")

        try:
            log.info("job.started", job_id=job.id, details=job_log_payload(job))
            options = options_for_job(job)
            # Fail fast, before a long inference, when the bundle is already taken.
            assert_output_bundle_available(job.output_path)
            result = run_upscale(
                options,
                runtime_dirs or resolve_runtime_dirs(),
                on_progress=lambda phase: self.progress.emit(job.id, _progress_text(phase)),
                on_tile=lambda done, total: self.progress.emit(
                    job.id, _tile_progress_text(done, total),
                ),
                on_warning=warnings.append,
                should_cancel=cancel.is_set,
                on_output_published=capture_published_image,
            )
            check_cancelled()
            sidecar = write_sidecar(
                input_path=job.input_path,
                output_path=job.output_path,
                options=options,
                result=result,
                warnings=warnings,
                should_cancel=cancel.is_set,
            )
            if published_image is None:
                raise PixelupError(
                    ErrorCode.INTERNAL_ERROR,
                    "Output image publication did not return its path.",
                )
            # Cancellation before this commit removes only this job's files;
            # cancellation after it cannot undo or deny the accepted success.
            check_cancelled()
            committed = True
            result["sidecar"] = str(sidecar)
            log.info(
                "job.finished",
                job_id=job.id,
                output=str(job.output_path),
                sidecar=str(sidecar),
                warnings=[english().of(warning) for warning in warnings],
            )
            return True, Message("queue.statusDone"), result, warnings
        except PixelupError as exc:
            discard_uncommitted()
            if exc.code == ErrorCode.JOB_CANCELLED:
                log.info("job.cancelled", job_id=job.id)
                return False, Message("queue.statusCancelled"), {"cancelled": True}, warnings
            log.exception(
                "job.failed",
                job_id=job.id,
                code=exc.code.value,
                error_details=exc.details,
                warnings=[english().of(warning) for warning in warnings],
                details=job_log_payload(job),
            )
            return False, failure_message(exc), {}, warnings
        except MemoryError:
            # Outside inference, such as while encoding a large output.
            discard_uncommitted()
            log.exception("job.out_of_memory", job_id=job.id, details=job_log_payload(job))
            return False, failure_message(out_of_memory_error()), {}, warnings
        except Exception:
            discard_uncommitted()
            log.exception("job.failed_unexpectedly", job_id=job.id, details=job_log_payload(job))
            return False, FAILED_FALLBACK, {}, warnings


class JobRunner(QObject):
    """The queue's one owner: runs its jobs one at a time on one worker thread.

    Jobs run serially because each already saturates the GPU or CPU it runs on.
    """

    progress = Signal(int, object)
    finished = Signal(int, bool, object, object, object)
    idle = Signal()
    _run_job = Signal(object, object, object)

    def __init__(
        self,
        jobs: list[Job],
        parent: QObject | None = None,
        *,
        runtime_dirs: RuntimeDirs | None = None,
    ) -> None:
        super().__init__(parent)
        self._jobs = jobs
        self._running: tuple[int, Event] | None = None
        self._shutting_down = False
        self._runtime_dirs = runtime_dirs
        self._queued_runtime_dirs: dict[int, RuntimeDirs | None] = {}
        self._thread = QThread(self)
        self._worker = JobWorker()
        self._worker.moveToThread(self._thread)
        self._run_job.connect(self._worker.run_job)
        self._worker.progress.connect(self.progress)
        self._worker.finished.connect(self._job_finished)
        thread = self._thread
        # A QThread destroyed while running ends the process. Whatever deletes the
        # runner, its thread stops first; an idle thread stops at once.
        self.destroyed.connect(lambda: _stop_thread(thread))

    @property
    def running(self) -> bool:
        return self._running is not None

    def retain_runtime_dirs(self, job_ids: Iterable[int], runtime_dirs: RuntimeDirs) -> None:
        """Keep the folder accepted for newly enqueued or retried jobs."""
        for job_id in job_ids:
            self._queued_runtime_dirs[job_id] = runtime_dirs

    def set_runtime_dirs(self, runtime_dirs: RuntimeDirs) -> None:
        # The running job holds its own snapshot. Preserve the same promise for
        # jobs accepted but not started before a folder change.
        for job in self._jobs:
            if job.status == "pending":
                self._queued_runtime_dirs.setdefault(job.id, self._runtime_dirs)
        self._runtime_dirs = runtime_dirs

    def schedule(self) -> None:
        """Start the next pending job unless one is running."""
        if self._shutting_down or self._running is not None:
            return
        job = next((job for job in self._jobs if job.status == "pending"), None)
        if job is None:
            return
        job.status = "running"
        job.message = Message("progress.starting")
        cancel = Event()
        self._running = (job.id, cancel)
        self.progress.emit(job.id, job.message)
        if not self._thread.isRunning():
            self._thread.start()
        self._run_job.emit(job, self._queued_runtime_dirs.pop(job.id, self._runtime_dirs), cancel)

    def request_cancel(self, job_id: int) -> None:
        if self._running is not None and self._running[0] == job_id:
            self._running[1].set()

    def begin_shutdown(self) -> None:
        """Irreversibly stop queue scheduling and cancel every unstarted job."""
        if self._shutting_down:
            return
        self._shutting_down = True
        cancelled: list[int] = []
        for job in self._jobs:
            if job.status == "pending":
                job.status = "cancelled"
                job.message = Message("queue.statusCancelled")
                cancelled.append(job.id)
        if self._running is not None:
            self._running[1].set()
        for job_id in cancelled:
            self.finished.emit(
                job_id, False, Message("queue.statusCancelled"), {"cancelled": True}, []
            )

    def cleanup_for_quit(self) -> bool:
        """Whether the running job has reported; then the worker thread is stopped."""
        self.begin_shutdown()
        if self._running is not None:
            return False
        _stop_thread(self._thread)
        return True

    @Slot(int, bool, object, object, object)
    def _job_finished(
        self,
        job_id: int,
        ok: bool,
        message: object,
        result: object,
        warnings: object,
    ) -> None:
        self._running = None
        self.finished.emit(job_id, ok, message, result, warnings)
        self.schedule()
        if self._running is None:
            self.idle.emit()


def _stop_thread(thread: QThread) -> None:
    thread.quit()
    thread.wait()


_PHASE_KEYS = {
    "load_model": "progress.loadingModel",
    "upscale": "progress.upscaling",
    "encode": "progress.saving",
}


def _progress_text(phase: str) -> Message:
    return Message(_PHASE_KEYS[phase])


def _tile_progress_text(done: int, total: int) -> Message:
    return Message.of("progress.tiles", done=done, count=total)

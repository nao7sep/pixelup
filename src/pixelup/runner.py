from __future__ import annotations

import time

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot

from pixelup.config import RuntimeDirs, resolve_runtime_dirs
from pixelup.errors import ErrorCode, PixelupError, user_text
from pixelup.i18n.localizer import english
from pixelup.i18n.message import Message
from pixelup.jobs import Job, job_log_payload, options_for_job
from pixelup.output_reservation import (
    PublishedFile,
    assert_output_bundle_claims_current,
    remove_published_file,
    reserve_output_bundle,
)
from pixelup.session_log import log
from pixelup.sidecar import write_sidecar
from pixelup.upscale import run_upscale


class JobSignals(QObject):
    # A job's status text travels as a Message and is rendered by the window, in
    # whatever language it speaks when the row is drawn.
    progress = Signal(int, object)
    finished = Signal(int, bool, object, object, object)


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


class JobWorker(QObject):
    def __init__(self, job: Job, runtime_dirs: RuntimeDirs | None = None) -> None:
        super().__init__()
        self.job = job
        self.signals = JobSignals()
        self._cancel_requested = False
        self._runtime_dirs = runtime_dirs

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        return self._cancel_requested

    @Slot()
    def run(self) -> None:
        warnings: list[Message] = []
        published_image: PublishedFile | None = None

        def capture_published_image(published: PublishedFile) -> None:
            nonlocal published_image
            published_image = published

        log.info("job.started", job_id=self.job.id, details=job_log_payload(self.job))
        try:
            options = options_for_job(self.job)
            runtime_dirs = self._runtime_dirs or resolve_runtime_dirs()
            with reserve_output_bundle(
                self.job.output_path,
                runtime_dirs.temp_dir,
                timeout=options.lock_timeout,
                should_cancel=self._is_cancelled,
                on_waiting=lambda: self.signals.progress.emit(
                    self.job.id, Message("progress.waitingForOutput")
                ),
            ):
                result = run_upscale(
                    options,
                    runtime_dirs,
                    on_progress=lambda phase: self.signals.progress.emit(
                        self.job.id,
                        _progress_text(phase),
                    ),
                    on_tile=lambda done, total: self.signals.progress.emit(
                        self.job.id,
                        _tile_progress_text(done, total),
                    ),
                    on_warning=warnings.append,
                    should_cancel=self._is_cancelled,
                    on_output_published=capture_published_image,
                )
                try:
                    sidecar_claim = write_sidecar(
                        input_path=self.job.input_path,
                        output_path=self.job.output_path,
                        options=options,
                        result=result,
                        warnings=warnings,
                    )
                except Exception:
                    if published_image is not None:
                        remove_published_file(published_image)
                    raise
                if published_image is None:
                    remove_published_file(sidecar_claim)
                    raise PixelupError(
                        ErrorCode.INTERNAL_ERROR,
                        "Output image publication did not return an ownership claim.",
                    )
                try:
                    # Image + sidecar together are the commit point. Revalidate both
                    # physical claims and every normalized shared-stem companion only
                    # after the sidecar descriptor has flushed and closed.
                    assert_output_bundle_claims_current(
                        self.job.output_path,
                        (published_image, sidecar_claim),
                    )
                except Exception:
                    remove_published_file(sidecar_claim)
                    remove_published_file(published_image)
                    raise
            sidecar = sidecar_claim.path
            result["sidecar"] = str(sidecar)
            log.info(
                "job.finished",
                job_id=self.job.id,
                output=str(self.job.output_path),
                sidecar=str(sidecar),
                warnings=[english().of(warning) for warning in warnings],
            )
            self.signals.finished.emit(
                self.job.id, True, Message("queue.statusDone"), result, warnings
            )
        except PixelupError as exc:
            if exc.code == ErrorCode.JOB_CANCELLED:
                log.info("job.cancelled", job_id=self.job.id)
                self.signals.finished.emit(
                    self.job.id,
                    False,
                    Message("queue.statusCancelled"),
                    {"cancelled": True},
                    warnings,
                )
                return
            log.exception(
                "job.failed",
                job_id=self.job.id,
                code=exc.code.value,
                error_details=exc.details,
                warnings=[english().of(warning) for warning in warnings],
                details=job_log_payload(self.job),
            )
            self.signals.finished.emit(self.job.id, False, failure_message(exc), {}, warnings)
        except Exception:
            log.exception(
                "job.failed_unexpectedly",
                job_id=self.job.id,
                details=job_log_payload(self.job),
            )
            self.signals.finished.emit(self.job.id, False, FAILED_FALLBACK, {}, warnings)


class JobRunner(QObject):
    progress = Signal(int, object)
    finished = Signal(int, bool, object, object, object)
    idle = Signal()

    def __init__(
        self,
        jobs: list[Job],
        parent: QObject | None = None,
        *,
        runtime_dirs: RuntimeDirs | None = None,
    ) -> None:
        super().__init__(parent)
        self._jobs = jobs
        self._threads: dict[int, tuple[QThread, JobWorker]] = {}
        self._active_jobs = 0
        self._max_concurrent_jobs = 1
        self._shutting_down = False
        self._runtime_dirs = runtime_dirs

    def schedule(self, max_concurrent_jobs: int) -> None:
        if self._shutting_down:
            return
        self._max_concurrent_jobs = max(1, max_concurrent_jobs)
        while self._active_jobs < self._max_concurrent_jobs:
            job = self._next_pending_job()
            if job is None:
                return
            self._start_job(job)

    def request_cancel(self, job_id: int) -> None:
        entry = self._threads.get(job_id)
        if entry is not None:
            _, worker = entry
            worker.request_cancel()

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
        for _thread, worker in self._threads.values():
            try:
                worker.request_cancel()
            except Exception:
                log.debug("quit.cancel_request_failed", job_id=worker.job.id)
        for job_id in cancelled:
            self.finished.emit(
                job_id, False, Message("queue.statusCancelled"), {"cancelled": True}, []
            )

    def cleanup_for_quit(self) -> bool:
        self.begin_shutdown()
        if not self._threads:
            return True
        entries = list(self._threads.values())
        deadline = time.monotonic() + 2.0
        stopped = True
        for thread, worker in entries:
            try:
                thread.quit()
                remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
                stopped = thread.wait(remaining_ms) and stopped
            except Exception:
                log.debug("quit.thread_wait_failed", job_id=worker.job.id)
                stopped = False
        if not stopped:
            log.info("quit.workers_still_stopping", count=len(entries))
            return False

        # Only detach the window-facing routes after every worker has actually
        # stopped. If one misses the deadline, the runner retains full ownership
        # and its normal finished/thread-finished wiring until the final worker
        # exits; the window can then retry closing through ``idle``.
        for _thread, worker in entries:
            for signal, slot in (
                (worker.signals.progress, self.progress.emit),
                (worker.signals.finished, self._job_finished),
            ):
                try:
                    signal.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
        self._threads.clear()
        log.info("quit.workers_cleaned", count=len(entries))
        return True

    def _next_pending_job(self) -> Job | None:
        for job in self._jobs:
            if job.status == "pending":
                return job
        return None

    def _start_job(self, job: Job) -> None:
        job.status = "running"
        job.message = Message("progress.starting")
        self._active_jobs += 1
        self.progress.emit(job.id, job.message)

        thread = QThread(self)
        worker = JobWorker(job, self._runtime_dirs)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.signals.progress.connect(self.progress.emit)
        worker.signals.finished.connect(self._job_finished)
        worker.signals.finished.connect(thread.quit)
        worker.signals.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda job_id=job.id: self._thread_finished(job_id))
        self._threads[job.id] = (thread, worker)
        thread.start()

    def _thread_finished(self, job_id: int) -> None:
        self._threads.pop(job_id, None)
        if not self._threads:
            self.idle.emit()

    @Slot(int, bool, object, object, object)
    def _job_finished(
        self,
        job_id: int,
        ok: bool,
        message: object,
        result: object,
        warnings: object,
    ) -> None:
        self._active_jobs = max(0, self._active_jobs - 1)
        self.finished.emit(job_id, ok, message, result, warnings)
        if not self._shutting_down:
            QTimer.singleShot(0, lambda: self.schedule(self._max_concurrent_jobs))


_PHASE_KEYS = {
    "load_model": "progress.loadingModel",
    "upscale": "progress.upscaling",
    "encode": "progress.saving",
}


def _progress_text(phase: str) -> Message:
    return Message(_PHASE_KEYS[phase])


def _tile_progress_text(done: int, total: int) -> Message:
    return Message.of("progress.tiles", done=done, count=total)

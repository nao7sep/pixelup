from __future__ import annotations

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot

from pixelup.config import resolve_runtime_dirs
from pixelup.errors import ErrorCode, PixelupError
from pixelup.jobs import Job, job_log_payload, options_for_job
from pixelup.session_log import log
from pixelup.sidecar import write_sidecar
from pixelup.upscale import run_upscale


class JobSignals(QObject):
    progress = Signal(int, str)
    finished = Signal(int, bool, str, object, object)


class JobWorker(QObject):
    def __init__(self, job: Job) -> None:
        super().__init__()
        self.job = job
        self.signals = JobSignals()
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        return self._cancel_requested

    @Slot()
    def run(self) -> None:
        warnings: list[str] = []
        log.info("job.started", job_id=self.job.id, details=job_log_payload(self.job))
        try:
            options = options_for_job(self.job)
            result = run_upscale(
                options,
                resolve_runtime_dirs(),
                on_progress=lambda phase: self.signals.progress.emit(
                    self.job.id,
                    _progress_text(phase),
                ),
                on_tile=lambda done, total: self.signals.progress.emit(
                    self.job.id,
                    _tile_progress_text(done, total),
                ),
                on_download=lambda model, done, total: self.signals.progress.emit(
                    self.job.id,
                    _download_text(model, done, total),
                ),
                on_warning=warnings.append,
                should_cancel=self._is_cancelled,
            )
            sidecar = write_sidecar(
                input_path=self.job.input_path,
                output_path=self.job.output_path,
                options=options,
                result=result,
                warnings=warnings,
            )
            result["sidecar"] = str(sidecar)
            log.info(
                "job.finished",
                job_id=self.job.id,
                output=str(self.job.output_path),
                sidecar=str(sidecar),
                warnings=warnings,
            )
            self.signals.finished.emit(self.job.id, True, "Done", result, warnings)
        except PixelupError as exc:
            if exc.code == ErrorCode.JOB_CANCELLED:
                log.info("job.cancelled", job_id=self.job.id)
                self.signals.finished.emit(
                    self.job.id,
                    False,
                    "Cancelled",
                    {"cancelled": True},
                    warnings,
                )
                return
            log.warning(
                "job.failed",
                job_id=self.job.id,
                code=exc.code.value,
                reason=exc.message,
                warnings=warnings,
                details=job_log_payload(self.job),
            )
            self.signals.finished.emit(self.job.id, False, exc.message, {}, warnings)
        except Exception as exc:
            log.exception(
                "job.failed_unexpectedly",
                job_id=self.job.id,
                details=job_log_payload(self.job),
            )
            self.signals.finished.emit(self.job.id, False, f"Unexpected error: {exc}", {}, warnings)


class JobRunner(QObject):
    progress = Signal(int, str)
    finished = Signal(int, bool, str, object, object)

    def __init__(self, jobs: list[Job], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._jobs = jobs
        self._threads: dict[int, tuple[QThread, JobWorker]] = {}
        self._active_jobs = 0
        self._max_concurrent_jobs = 1

    def schedule(self, max_concurrent_jobs: int) -> None:
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

    def cleanup_for_quit(self) -> None:
        if not self._threads:
            return
        entries = list(self._threads.values())
        for _thread, worker in entries:
            try:
                worker.request_cancel()
            except Exception:
                log.debug("quit.cancel_request_failed", job_id=worker.job.id)
            # Stop UI-facing reschedules/updates during shutdown, but keep the
            # finished -> thread.quit -> deleteLater wiring intact so each thread
            # event loop actually exits and tears down cleanly.
            for signal, slot in (
                (worker.signals.progress, self.progress.emit),
                (worker.signals.finished, self._job_finished),
            ):
                try:
                    signal.disconnect(slot)
                except (RuntimeError, TypeError):
                    # Expected during teardown: the slot may already be
                    # disconnected or its sender gone. Noise, not an incident.
                    pass
        for thread, worker in entries:
            try:
                thread.quit()
                thread.wait(2000)
            except Exception:
                log.debug("quit.thread_wait_failed", job_id=worker.job.id)
        log.info("quit.workers_cleaned", count=len(entries))

    def _next_pending_job(self) -> Job | None:
        for job in self._jobs:
            if job.status == "pending":
                return job
        return None

    def _start_job(self, job: Job) -> None:
        job.status = "running"
        job.message = "Starting"
        self._active_jobs += 1
        self.progress.emit(job.id, job.message)

        thread = QThread(self)
        worker = JobWorker(job)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.signals.progress.connect(self.progress.emit)
        worker.signals.finished.connect(self._job_finished)
        worker.signals.finished.connect(thread.quit)
        worker.signals.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda job_id=job.id: self._threads.pop(job_id, None))
        self._threads[job.id] = (thread, worker)
        thread.start()

    @Slot(int, bool, str, object, object)
    def _job_finished(
        self,
        job_id: int,
        ok: bool,
        message: str,
        result: object,
        warnings: object,
    ) -> None:
        self._active_jobs = max(0, self._active_jobs - 1)
        self.finished.emit(job_id, ok, message, result, warnings)
        QTimer.singleShot(0, lambda: self.schedule(self._max_concurrent_jobs))


def _download_text(model: str, done: int, total: int | None) -> str:
    if total:
        return f"{done * 100 // total}% - downloading {model}"
    return f"Downloading {model}"


def _progress_text(phase: str) -> str:
    match phase:
        case "upscale":
            return "Upscaling"
        case "encode":
            return "Saving"
        case _:
            return phase.replace("-", " ").replace("_", " ").capitalize()


def _tile_progress_text(done: int, total: int) -> str:
    return f"{done}/{total} tiles processed"

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot

from pixelup.config import resolve_runtime_dirs
from pixelup.errors import ErrorCode, PixelupError
from pixelup.jobs import Job, job_log_payload, options_for_job
from pixelup.session_log import get_logger
from pixelup.sidecar import write_sidecar
from pixelup.upscale import run_upscale

LOGGER = get_logger()


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
        LOGGER.info("Starting job %s details=%s", self.job.id, job_log_payload(self.job))
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
            LOGGER.info(
                "Finished job %s output=%s sidecar=%s warnings=%s",
                self.job.id,
                self.job.output_path,
                sidecar,
                warnings,
            )
            self.signals.finished.emit(self.job.id, True, "Done", result, warnings)
        except PixelupError as exc:
            if exc.code == ErrorCode.JOB_CANCELLED:
                LOGGER.info("Job %s cancelled", self.job.id)
                self.signals.finished.emit(
                    self.job.id,
                    False,
                    "Cancelled",
                    {"cancelled": True},
                    warnings,
                )
                return
            LOGGER.warning(
                "Job %s failed message=%s warnings=%s details=%s",
                self.job.id,
                exc.message,
                warnings,
                job_log_payload(self.job),
            )
            self.signals.finished.emit(self.job.id, False, exc.message, {}, warnings)
        except Exception as exc:
            LOGGER.exception(
                "Job %s failed unexpectedly details=%s",
                self.job.id,
                job_log_payload(self.job),
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
                pass
            for signal in (worker.signals.progress, worker.signals.finished):
                try:
                    signal.disconnect()
                except (RuntimeError, TypeError):
                    pass
        for thread, _worker in entries:
            try:
                thread.wait(2000)
            except Exception:
                pass
        LOGGER.info("Cleaned up %s worker thread(s) for quit", len(entries))

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

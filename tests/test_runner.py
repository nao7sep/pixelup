from __future__ import annotations

import errno
import os
import subprocess
import sys
from itertools import count
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from PIL import Image, ImageChops, ImageStat
from PySide6.QtCore import QObject, QThread, Slot
from PySide6.QtWidgets import QApplication

from pixelup.config import RuntimeDirs
from pixelup.errors import ErrorCode, PixelupError
from pixelup.i18n.localizer import english
from pixelup.i18n.message import Message
from pixelup.jobs import Job, JobSettings, create_jobs, retry_failed_jobs
from pixelup.model_management import UPSCALE_MODELS
from pixelup.output_reservation import PublishedFile
from pixelup.runner import (
    FAILED_FALLBACK,
    JobRunner,
    JobSignals,
    JobWorker,
    _progress_text,
    _tile_progress_text,
    failure_message,
)
from pixelup.session_log import configure_session_logging


def test_progress_text_maps_every_phase_inference_reports() -> None:
    assert english().of(_progress_text("load_model")) == "Loading model…"
    assert english().of(_progress_text("upscale")) == "Upscaling…"
    assert english().of(_progress_text("encode")) == "Saving…"


def test_tile_progress_text() -> None:
    assert english().of(_tile_progress_text(3, 8)) == "3/8 tiles processed"
    assert english().of(_tile_progress_text(1, 1)) == "1/1 tile processed"


# --- JobRunner scheduling -------------------------------------------------
#
# Scheduling decisions use bookkeeping-only spawns and unstarted QThreads whose
# finished signals can be driven directly. The lifecycle regressions below run
# real QObject slots on native QThreads, with explicit gates holding result
# emission and physical return. Neither path needs inference or model downloads.


def _make_job(job_id: int, tmp_path: Path) -> Job:
    return Job(
        id=job_id,
        input_path=tmp_path / f"in-{job_id}.png",
        model="realesr-general-x4v3",
        output_path=tmp_path / f"out-{job_id}.png",
        settings=JobSettings(),
    )


class _HeldResultWorker(QObject):
    """A real Qt worker whose result can precede its native thread's return."""

    def __init__(self, job: Job, runtime_dirs: RuntimeDirs | None = None) -> None:
        super().__init__()
        self.job = job
        self.signals = JobSignals()
        self.started = Event()
        self.emit_result = Event()
        self.allow_return = Event()
        self.cancel_requested = False
        self._runtime_dirs = runtime_dirs
        self.native_thread: QThread | None = None

    def request_cancel(self) -> None:
        self.cancel_requested = True

    @Slot()
    def run(self) -> None:
        self.native_thread = QThread.currentThread()
        self.signals.progress.emit(self.job.id, Message("progress.upscaling"))
        self.started.set()
        self.emit_result.wait()
        message = Message(
            "queue.statusCancelled" if self.cancel_requested else "jobs.failedFallback",
        )
        result = {"cancelled": True} if self.cancel_requested else {"failed": True}
        self.signals.finished.emit(
            self.job.id, False, message, result, [],
        )
        self.allow_return.wait()


class _ResultReceipt(QObject):
    """Proves the worker result reached the main event loop before assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    @Slot(int, bool, object, object, object)
    def receive(self, *args: object) -> None:
        self.count += 1


def _stop_held_workers(runner: JobRunner, workers: list[_HeldResultWorker], process_until) -> None:
    # Keep native ownership even if an assertion exposed the old registry bug.
    for worker in workers:
        worker.emit_result.set()
        worker.allow_return.set()
    runner.cleanup_for_quit()
    for worker in workers:
        if worker.native_thread is not None:
            try:
                worker.native_thread.quit()
                assert worker.native_thread.wait(5000)
            except RuntimeError:
                pass  # Qt already deleted a thread whose finish was delivered.
    process_until(lambda: not runner._threads, timeout_s=5, what="Held workers teardown")


def test_worker_result_keeps_terminal_state_and_capacity_until_thread_returns(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process_until,
) -> None:
    jobs = [_make_job(i, tmp_path) for i in (1, 2)]
    workers: list[_HeldResultWorker] = []
    receipt = _ResultReceipt()

    def make_worker(job: Job, runtime_dirs: RuntimeDirs | None = None) -> _HeldResultWorker:
        worker = _HeldResultWorker(job, runtime_dirs)
        worker.emit_result.set()
        worker.signals.finished.connect(receipt.receive)
        workers.append(worker)
        return worker

    monkeypatch.setattr("pixelup.runner.JobWorker", make_worker)
    runner = JobRunner(jobs)
    outcomes: list[tuple[object, ...]] = []
    runner.finished.connect(lambda *args: outcomes.append(args))
    runner.schedule(1)
    try:
        process_until(lambda: receipt.count == 1, timeout_s=5, what="Worker result receipt")

        assert outcomes == []
        assert jobs[0].status == "running"
        assert jobs[1].status == "pending"
        assert runner._active_jobs == 1
        assert len(workers) == 1
        assert runner._threads[1][1] is workers[0]
        assert runner._threads[1][0].isRunning()

        workers[0].allow_return.set()
        process_until(lambda: len(workers) == 2, timeout_s=5, what="Freed job capacity")
        assert len(outcomes) == 1
        assert outcomes[0][0:2] == (1, False)
        assert runner._active_jobs == 1
        assert set(runner._threads) == {2}
    finally:
        _stop_held_workers(runner, workers, process_until)


def test_retry_with_same_job_id_keeps_cancel_and_quit_owner(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process_until,
) -> None:
    job = _make_job(1, tmp_path)
    workers: list[_HeldResultWorker] = []
    receipt = _ResultReceipt()

    def make_worker(job: Job, runtime_dirs: RuntimeDirs | None = None) -> _HeldResultWorker:
        worker = _HeldResultWorker(job, runtime_dirs)
        if not workers:
            worker.emit_result.set()
        worker.signals.finished.connect(receipt.receive)
        workers.append(worker)
        return worker

    monkeypatch.setattr("pixelup.runner.JobWorker", make_worker)
    runner = JobRunner([job])
    outcomes: list[tuple[object, ...]] = []
    progress: list[tuple[object, ...]] = []
    settled_before_retry: list[bool] = []
    retry_ids: list[list[int]] = []
    runner.progress.connect(lambda *args: progress.append(args))

    def retry_after_failure(*outcome: object) -> None:
        outcomes.append(outcome)
        if len(outcomes) == 1:
            settled_before_retry.append(not first_thread.isRunning())
            job.status = "failed"
            retry_ids.append(retry_failed_jobs([job]))
            runner.schedule(1)

    runner.finished.connect(retry_after_failure)
    runner.schedule(1)
    first_thread = runner._threads[job.id][0]
    try:
        process_until(lambda: receipt.count == 1, timeout_s=5, what="Failed worker result")
        assert outcomes == []
        workers[0].allow_return.set()
        process_until(
            lambda: len(workers) == 2 and workers[1].started.is_set()
            and progress.count((job.id, Message("progress.upscaling"))) == 2,
            timeout_s=5, what="Same-ID retry start",
        )
        assert len(outcomes) == 1
        assert settled_before_retry == [True]
        assert retry_ids == [[job.id]]
        assert runner._threads[job.id][1] is workers[1]
        assert runner._active_jobs == 1
        assert progress.count((job.id, Message("progress.upscaling"))) == 2

        runner.request_cancel(job.id)
        assert workers[1].cancel_requested is True
        workers[1].emit_result.set()
        process_until(lambda: receipt.count == 2, timeout_s=5, what="Cancelled retry result")

        assert runner.cleanup_for_quit() is False
        assert runner._threads[job.id][1] is workers[1]
        assert runner._threads[job.id][0].isRunning()
        assert runner._active_jobs == 1
        assert len(outcomes) == 1

        workers[1].allow_return.set()
        process_until(lambda: not runner._threads, timeout_s=5, what="Cancelled retry return")
        assert outcomes[1] == (job.id, False, Message("queue.statusCancelled"),
                               {"cancelled": True}, [])
        assert len(outcomes) == 2
        assert runner._job_results == {}
        assert runner._active_jobs == 0
        assert runner.cleanup_for_quit() is True
    finally:
        _stop_held_workers(runner, workers, process_until)


def test_successful_quit_wait_preserves_queued_completion_once(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process_until,
) -> None:
    workers: list[_HeldResultWorker] = []

    def make_worker(job: Job, runtime_dirs: RuntimeDirs | None = None) -> _HeldResultWorker:
        worker = _HeldResultWorker(job, runtime_dirs)
        worker.emit_result.set()
        worker.allow_return.set()
        workers.append(worker)
        return worker

    monkeypatch.setattr("pixelup.runner.JobWorker", make_worker)
    runner = JobRunner([_make_job(1, tmp_path)])
    outcomes: list[tuple[object, ...]] = []
    runner.finished.connect(lambda *args: outcomes.append(args))
    runner.schedule(1)
    try:
        # No main-loop pumping between start and wait: both Qt completions are
        # queued while cleanup proves the native worker has already stopped.
        assert runner.cleanup_for_quit() is True
        assert outcomes == []
        assert set(runner._threads) == {1}
        assert runner._active_jobs == 1
        process_until(lambda: not runner._threads, timeout_s=5, what="Queued quit completion")
        assert len(outcomes) == 1
        assert runner._job_results == {}
        assert runner._active_jobs == 0
        assert runner.cleanup_for_quit() is True
        qapp.processEvents()
        assert len(outcomes) == 1
    finally:
        _stop_held_workers(runner, workers, process_until)


def test_admitted_jobs_replace_old_models_folder_capture(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process_until,
) -> None:
    jobs = [_make_job(i, tmp_path) for i in (1, 2)]
    workers: list[_HeldResultWorker] = []
    original = RuntimeDirs(tmp_path / "original", tmp_path / "temp")
    accepted = RuntimeDirs(tmp_path / "accepted", tmp_path / "temp")
    replacement = RuntimeDirs(tmp_path / "replacement", tmp_path / "temp")

    def make_worker(job: Job, runtime_dirs: RuntimeDirs | None = None) -> _HeldResultWorker:
        worker = _HeldResultWorker(job, runtime_dirs)
        workers.append(worker)
        return worker

    monkeypatch.setattr("pixelup.runner.JobWorker", make_worker)
    runner = JobRunner(jobs, runtime_dirs=original)
    runner.set_runtime_dirs(replacement)  # Captures original for existing pending jobs.
    runner.retain_runtime_dirs((job.id for job in jobs), accepted)
    runner.schedule(2)
    try:
        assert [worker._runtime_dirs for worker in workers] == [accepted, accepted]
        assert runner._queued_runtime_dirs == {}
    finally:
        _stop_held_workers(runner, workers, process_until)


class _RecordingRunner(JobRunner):
    """JobRunner whose thread spawn is replaced by bookkeeping only.

    Records the order jobs were started so the scheduling logic can be asserted
    directly, while leaving scheduling and terminal delivery completely real.
    """

    def __init__(self, jobs: list[Job]) -> None:
        super().__init__(jobs)
        self.started_ids: list[int] = []

    def _start_job(self, job: Job) -> None:
        job.status = "running"
        job.message = "Starting"
        self._active_jobs += 1
        self.started_ids.append(job.id)
        thread = QThread(self)
        thread.setProperty("job_id", job.id)
        thread.finished.connect(self._thread_finished)
        self._threads[job.id] = (thread, _FakeWorker(job.id))  # type: ignore[assignment]

    def finish(self, job_id: int, *outcome: object) -> None:
        self._job_finished(job_id, *outcome)
        self._threads[job_id][0].finished.emit()


def test_schedule_caps_active_jobs(qapp: QApplication, tmp_path: Path) -> None:
    jobs = [_make_job(i, tmp_path) for i in range(1, 5)]
    runner = _RecordingRunner(jobs)

    runner.schedule(2)

    assert runner.started_ids == [1, 2]
    assert runner._active_jobs == 2
    assert [job.status for job in jobs] == ["running", "running", "pending", "pending"]


def test_finishing_a_job_starts_the_next(qapp: QApplication, tmp_path: Path) -> None:
    jobs = [_make_job(i, tmp_path) for i in range(1, 4)]
    runner = _RecordingRunner(jobs)
    finished: list[tuple[object, ...]] = []
    runner.finished.connect(lambda *args: finished.append(args))

    runner.schedule(1)
    assert runner.started_ids == [1]

    for job_id in (1, 2, 3):
        runner.finish(job_id, True, "Done", {"ok": True}, [])
        qapp.processEvents()  # let the QTimer-posted reschedule run

    # Each finish pulled exactly the next pending job, one at a time.
    assert runner.started_ids == [1, 2, 3]
    assert [record[0] for record in finished] == [1, 2, 3]
    assert runner._active_jobs == 0


def test_reschedule_after_finish_refills_one_freed_slot(
    qapp: QApplication, tmp_path: Path
) -> None:
    # The QTimer.singleShot(0, schedule) that _thread_finished posts reads the *current*
    # _active_jobs and cap when it fires. With four jobs and a cap of two, finishing one frees
    # exactly one slot, so the reschedule must start exactly one more — not drain the whole queue.
    jobs = [_make_job(i, tmp_path) for i in range(1, 5)]
    runner = _RecordingRunner(jobs)

    runner.schedule(2)
    assert runner.started_ids == [1, 2]

    runner.finish(1, True, "Done", {"ok": True}, [])
    qapp.processEvents()  # let the posted reschedule run

    assert runner.started_ids == [1, 2, 3]
    assert runner._active_jobs == 2


def test_finished_emits_outcome_to_listeners(qapp: QApplication, tmp_path: Path) -> None:
    runner = _RecordingRunner([_make_job(1, tmp_path)])
    finished: list[tuple[object, ...]] = []
    runner.finished.connect(lambda *args: finished.append(args))

    runner.schedule(1)
    runner._job_finished(1, False, "Cancelled", {"cancelled": True}, ["w"])
    assert finished == []
    assert runner._active_jobs == 1
    runner._threads[1][0].finished.emit()
    qapp.processEvents()

    assert finished == [(1, False, "Cancelled", {"cancelled": True}, ["w"])]


def test_shutdown_cancels_pending_jobs_and_never_reschedules_them(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    jobs = [_make_job(i, tmp_path) for i in range(1, 4)]
    runner = _RecordingRunner(jobs)
    finished: list[tuple[object, ...]] = []
    runner.finished.connect(lambda *args: finished.append(args))
    runner.schedule(1)
    assert runner.started_ids == [1]

    runner.begin_shutdown()
    runner.finish(1, False, Message("queue.statusCancelled"), {"cancelled": True}, [])
    qapp.processEvents()
    runner.schedule(3)

    assert runner.started_ids == [1]
    assert [job.status for job in jobs[1:]] == ["cancelled", "cancelled"]
    assert [english().of(job.message) for job in jobs[1:]] == ["Cancelled", "Cancelled"]
    assert [record[0] for record in finished] == [2, 3, 1]


def test_failure_message_carries_the_hint_when_there_is_one() -> None:
    # The queue row is the only surface a failed processing job gets, so the remedy
    # must ride along with the diagnosis rather than existing only in the log.
    with_hint = PixelupError(
        ErrorCode.MODEL_NOT_FOUND,
        Message.of("error.modelMissing", model="x4plus"),
        hint=Message("error.hintInstallModel"),
    )
    assert english().of(failure_message(with_hint)) == (
        "Model 'x4plus' is not present in the models directory. "
        "Open Managed models to install it, or place the .pth file in the models directory."
    )

    bare = PixelupError(ErrorCode.OUT_OF_MEMORY, Message("error.outOfMemory"))
    assert english().of(failure_message(bare)) == "Inference ran out of memory."

    internal = PixelupError(
        ErrorCode.INTERNAL_ERROR,
        "Inference returned an unsupported private tensor layout.",
    )
    assert english().of(failure_message(internal)) == (
        "The image could not be upscaled. Retry the job."
    )


def test_a_failure_the_reader_sees_must_be_a_catalogue_message() -> None:
    # Only a failure no reader ever sees may carry plain English: anything else
    # would reach the queue row in English whatever the interface speaks.
    with pytest.raises(TypeError):
        PixelupError(ErrorCode.OUT_OF_MEMORY, "Ran out of memory.")
    assert str(PixelupError(ErrorCode.JOB_CANCELLED, "Job cancelled.")) == "Job cancelled."


def test_request_cancel_for_unknown_job_is_a_noop(tmp_path: Path) -> None:
    runner = JobRunner([_make_job(1, tmp_path)])
    # Nothing is running, so cancelling any id must simply do nothing.
    runner.request_cancel(1)
    runner.request_cancel(999)


def test_request_cancel_signals_the_running_worker(tmp_path: Path) -> None:
    # The pure routing case is complemented by the real retry lifecycle below.
    runner = JobRunner([_make_job(1, tmp_path)])
    worker = _FakeWorker(1)
    runner._threads = {1: (SimpleNamespace(), worker)}  # type: ignore[dict-item]

    runner.request_cancel(1)

    assert worker.cancel_requested is True


# --- cleanup_for_quit teardown sequence -----------------------------------
#
# Fake native threads isolate the shared wait-budget ordering. Real gated workers
# also prove that ownership and queued completion delivery survive these waits.


class _FakeWorker:
    def __init__(self, job_id: int) -> None:
        self.job = SimpleNamespace(id=job_id)
        self.cancel_requested = False

    def request_cancel(self) -> None:
        self.cancel_requested = True


class _FakeThread:
    def __init__(
        self,
        name: str,
        calls: list[tuple[str, str]],
        *,
        stops: bool = True,
    ) -> None:
        self._name = name
        self._calls = calls
        self._stops = stops

    def quit(self) -> None:
        self._calls.append((self._name, "quit"))

    def wait(self, msecs: int) -> bool:
        self._calls.append((self._name, f"wait:{msecs}"))
        return self._stops


@pytest.fixture
def _session_log() -> None:
    # Route the singleton logger to the test's records database so the quit.* lines
    # don't spill to stderr. conftest restores the logger after the test.
    configure_session_logging()


def test_cleanup_for_quit_teardown_sequence(_session_log: None) -> None:
    runner = JobRunner([])
    calls: list[tuple[str, str]] = []
    workers = [_FakeWorker(1), _FakeWorker(2)]
    threads = [_FakeThread("a", calls), _FakeThread("b", calls)]
    runner._threads = {  # type: ignore[assignment]
        1: (threads[0], workers[0]),
        2: (threads[1], workers[1]),
    }

    assert runner.cleanup_for_quit() is True

    for worker in workers:
        assert worker.cancel_requested is True

    # Each thread is quit before it is waited on, within one shared 2s budget.
    for name in ("a", "b"):
        thread_calls = [call for call in calls if call[0] == name]
        assert [call[1].split(":")[0] for call in thread_calls] == ["quit", "wait"]
        assert 0 <= int(thread_calls[1][1].split(":")[1]) <= 2000
    # Native completion does not discard callbacks still queued on the main loop.
    assert runner._threads == {1: (threads[0], workers[0]), 2: (threads[1], workers[1])}


def test_cleanup_for_quit_retains_ownership_when_a_thread_is_still_running(
    _session_log: None,
) -> None:
    runner = JobRunner([])
    calls: list[tuple[str, str]] = []
    worker = _FakeWorker(1)
    thread = _FakeThread("slow", calls, stops=False)
    runner._threads = {1: (thread, worker)}  # type: ignore[assignment]

    assert runner.cleanup_for_quit() is False

    assert runner._threads == {1: (thread, worker)}
    assert worker.cancel_requested is True


def test_cleanup_for_quit_is_a_noop_without_threads(
    _session_log: None,
    tmp_path: Path,
) -> None:
    jobs = [_make_job(1, tmp_path)]
    runner = JobRunner(jobs)
    assert runner.cleanup_for_quit() is True
    assert runner._threads == {}
    assert jobs[0].status == "cancelled"


def test_thread_finished_releases_registry_and_emits_idle(qapp: QApplication) -> None:
    runner = JobRunner([])
    thread = QThread(runner)
    thread.setProperty("job_id", 1)
    thread.finished.connect(runner._thread_finished)
    runner._threads = {1: (thread, _FakeWorker(1))}  # type: ignore[dict-item]
    runner._active_jobs = 1
    idle: list[bool] = []
    runner.idle.connect(lambda: idle.append(True))

    thread.finished.emit()

    assert runner._threads == {}
    assert runner._active_jobs == 0
    assert idle == [True]


# --- JobWorker.run branches (no threads) ----------------------------------
#
# Run the worker body directly on the main thread (no moveToThread / QThread) with
# run_upscale + write_sidecar mocked, capturing the emitted signals. This covers
# the success / cancelled / failed / unexpected-error branches and the progress
# and cancel callback wiring independently of native thread dispatch.


def _capture_worker(worker: JobWorker) -> tuple[list, list]:
    finished: list[tuple[object, ...]] = []
    progress: list[tuple[object, ...]] = []
    # Status text travels as a Message; the capture reads it back in English, the
    # source language, so each test can still say what the row is told.
    def english_args(args: tuple[object, ...]) -> tuple[object, ...]:
        return tuple(english().of(arg) if isinstance(arg, Message) else arg for arg in args)

    worker.signals.finished.connect(lambda *args: finished.append(english_args(args)))
    worker.signals.progress.connect(lambda *args: progress.append(english_args(args)))
    return finished, progress


def _publish_fixture(path: Path, data: bytes) -> PublishedFile:
    path.write_bytes(data)
    written = os.lstat(path)
    return PublishedFile(path, written.st_dev, written.st_ino)


def _worker_with_real_image_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> JobWorker:
    from pixelup import inference
    from pixelup.imaging import image_from_bgr_array

    job = _make_job(81, tmp_path)
    job.model = "custom-model"
    job.settings = JobSettings(device="cpu")
    Image.new("RGB", (3, 2), (20, 40, 60)).save(job.input_path)
    models = tmp_path / "models"
    models.mkdir()
    (models / "custom-model.pth").write_bytes(b"weights")
    monkeypatch.setattr(inference, "_import_torch", lambda: None)
    monkeypatch.setattr(inference, "to_torch_device", lambda *args: "cpu")
    monkeypatch.setattr(inference, "_create_upsampler", lambda *args, **kwargs:
                        SimpleNamespace(
                            enhance=lambda pixels, **kwargs: (image_from_bgr_array(pixels), None)
                        ))
    return JobWorker(job, RuntimeDirs(models, tmp_path / "temp"))


@pytest.mark.parametrize("boundary", ["encode", "sidecar"])
def test_worker_cancel_during_real_writes_removes_owned_output(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    worker = _worker_with_real_image_pipeline(tmp_path, monkeypatch)
    finished, _ = _capture_worker(worker)
    sidecar = worker.job.output_path.with_suffix(".json")
    files_at_completion = []
    worker.signals.finished.connect(lambda *args: files_at_completion.append(
        (worker.job.output_path.exists(), sidecar.exists())
    ))
    if boundary == "encode":
        original = Image.Image.save

        def cancel_in_encoder(self, file, **kwargs):
            original(self, file, **kwargs)
            worker.request_cancel()

        monkeypatch.setattr(Image.Image, "save", cancel_in_encoder)
    else:
        original_fsync = os.fsync

        def cancel_in_sidecar(descriptor):
            original_fsync(descriptor)
            if sidecar.exists():
                worker.request_cancel()

        monkeypatch.setattr(os, "fsync", cancel_in_sidecar)

    worker.run()

    assert len(finished) == 1
    assert finished[0][1:4] == (False, "Cancelled", {"cancelled": True})
    assert files_at_completion == [(False, False)]
    assert not worker.job.output_path.exists()
    assert not sidecar.exists()
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("replacement", ["image", "sidecar"])
def test_worker_cancel_before_commit_preserves_replacement_winner(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement: str,
) -> None:
    worker = _worker_with_real_image_pipeline(tmp_path, monkeypatch)
    finished, _ = _capture_worker(worker)
    sidecar = worker.job.output_path.with_suffix(".json")
    target = worker.job.output_path if replacement == "image" else sidecar
    original_fsync = os.fsync

    def cancel_and_replace(descriptor):
        original_fsync(descriptor)
        if sidecar.exists():
            winner = tmp_path / "winner"
            winner.write_bytes(b"replacement-winner")
            winner.replace(target)
            worker.request_cancel()

    monkeypatch.setattr(os, "fsync", cancel_and_replace)
    worker.run()

    assert len(finished) == 1
    assert finished[0][1:4] == (False, "Cancelled", {"cancelled": True})
    assert target.read_bytes() == b"replacement-winner"
    other = sidecar if replacement == "image" else worker.job.output_path
    assert not other.exists()


def test_worker_cancel_after_bundle_commit_keeps_success(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _worker_with_real_image_pipeline(tmp_path, monkeypatch)
    finished, _ = _capture_worker(worker)
    from pixelup.runner import log

    original = log.info

    def cancel_after_commit(event, **kwargs):
        if event == "job.finished":
            worker.request_cancel()
        original(event, **kwargs)

    monkeypatch.setattr(log, "info", cancel_after_commit)
    worker.run()

    assert finished[0][1] is True
    assert worker.job.output_path.is_file()
    assert worker.job.output_path.with_suffix(".json").is_file()


NOTE = Message.of("warning.extensionMismatch", extension="png", format="jpg")


def test_worker_run_success_emits_done_with_sidecar(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _session_log: None,
) -> None:
    job = _make_job(1, tmp_path)

    def fake_upscale(options: object, runtime_dirs: object, **kwargs: object) -> dict[str, object]:
        kwargs["on_warning"](NOTE)  # type: ignore[operator]
        kwargs["on_output_published"](_publish_fixture(job.output_path, b"image"))  # type: ignore[operator]
        return {"ok": True, "output": str(getattr(options, "output_path", ""))}

    monkeypatch.setattr("pixelup.runner.run_upscale", fake_upscale)
    monkeypatch.setattr(
        "pixelup.runner.write_sidecar",
        lambda **kwargs: _publish_fixture(kwargs["output_path"].with_suffix(".json"), b"{}"),
    )
    worker = JobWorker(job)
    finished, _progress = _capture_worker(worker)

    worker.run()

    assert len(finished) == 1
    job_id, ok, message, result, warnings = finished[0]
    assert (job_id, ok, message) == (job.id, True, "Done")
    assert result["ok"] is True
    assert str(result["sidecar"]).endswith(".json")
    assert warnings == [NOTE]


def test_worker_rejects_occupied_output_before_inference(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _session_log: None,
) -> None:
    job = _make_job(8, tmp_path)
    job.output_path.write_bytes(b"existing")
    called = False

    def _upscale(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(
        "pixelup.runner.resolve_runtime_dirs",
        lambda: SimpleNamespace(temp_dir=tmp_path / "temp"),
    )
    monkeypatch.setattr("pixelup.runner.run_upscale", _upscale)
    worker = JobWorker(job)
    finished, _progress = _capture_worker(worker)

    worker.run()

    assert called is False
    assert finished[0][1] is False
    assert "already exists" in finished[0][2]


@pytest.mark.parametrize("hard_links", [True, False])
def test_worker_removes_its_image_when_sidecar_publication_loses(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _session_log: None,
    hard_links: bool,
) -> None:
    job = _make_job(9, tmp_path)
    if not hard_links:

        def unsupported_link(*args: object, **kwargs: object) -> None:
            raise OSError(errno.EOPNOTSUPP, "hard links unsupported")

        monkeypatch.setattr("pixelup.output_reservation.os.link", unsupported_link)

    def _upscale(*args: object, **kwargs: object) -> dict[str, object]:
        job.output_path.write_bytes(b"pixelup")
        written = os.lstat(job.output_path)
        kwargs["on_output_published"](
            PublishedFile(job.output_path, written.st_dev, written.st_ino)
        )
        return {"ok": True}

    def _lose_sidecar(**kwargs: object) -> Path:
        job.output_path.with_suffix(".json").write_bytes(b"external-sidecar")
        raise PixelupError(ErrorCode.OUTPUT_EXISTS, Message("error.sidecarExists"))

    monkeypatch.setattr("pixelup.runner.run_upscale", _upscale)
    monkeypatch.setattr("pixelup.runner.write_sidecar", _lose_sidecar)
    worker = JobWorker(job)
    finished, _progress = _capture_worker(worker)

    worker.run()

    assert not job.output_path.exists()
    assert job.output_path.with_suffix(".json").read_bytes() == b"external-sidecar"
    assert finished[0][1] is False


def test_sidecar_loss_cleanup_preserves_a_replacement_image_winner(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _session_log: None,
) -> None:
    job = _make_job(10, tmp_path)

    def _upscale(*args: object, **kwargs: object) -> dict[str, object]:
        job.output_path.write_bytes(b"pixelup")
        written = os.lstat(job.output_path)
        kwargs["on_output_published"](
            PublishedFile(job.output_path, written.st_dev, written.st_ino)
        )
        return {"ok": True}

    def _replace_then_lose(**kwargs: object) -> Path:
        job.output_path.unlink()
        job.output_path.write_bytes(b"external-winner")
        raise PixelupError(ErrorCode.OUTPUT_EXISTS, Message("error.sidecarExists"))

    monkeypatch.setattr("pixelup.runner.run_upscale", _upscale)
    monkeypatch.setattr("pixelup.runner.write_sidecar", _replace_then_lose)
    worker = JobWorker(job)
    finished, _progress = _capture_worker(worker)

    worker.run()

    assert job.output_path.read_bytes() == b"external-winner"
    assert finished[0][1] is False


def test_final_bundle_commit_preserves_a_late_case_variant_companion(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _session_log: None,
) -> None:
    job = _make_job(11, tmp_path)
    companion = job.output_path.with_suffix(".JPG")

    def _upscale(*args: object, **kwargs: object) -> dict[str, object]:
        kwargs["on_output_published"](_publish_fixture(job.output_path, b"pixelup"))  # type: ignore[operator]
        return {"ok": True}

    def _sidecar(**kwargs: object) -> PublishedFile:
        claim = _publish_fixture(job.output_path.with_suffix(".json"), b"{}")
        companion.write_bytes(b"external companion")
        return claim

    monkeypatch.setattr("pixelup.runner.run_upscale", _upscale)
    monkeypatch.setattr("pixelup.runner.write_sidecar", _sidecar)
    worker = JobWorker(job)
    finished, _progress = _capture_worker(worker)

    worker.run()

    assert not job.output_path.exists()
    assert not job.output_path.with_suffix(".json").exists()
    assert companion.read_bytes() == b"external companion"
    assert finished[0][1] is False


def test_final_bundle_commit_preserves_a_replacement_image_and_cleans_owned_sidecar(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _session_log: None,
) -> None:
    job = _make_job(12, tmp_path)

    def _upscale(*args: object, **kwargs: object) -> dict[str, object]:
        kwargs["on_output_published"](_publish_fixture(job.output_path, b"pixelup"))  # type: ignore[operator]
        return {"ok": True}

    def _sidecar(**kwargs: object) -> PublishedFile:
        claim = _publish_fixture(job.output_path.with_suffix(".json"), b"{}")
        winner = tmp_path / "winner.tmp"
        winner.write_bytes(b"external winner")
        os.replace(winner, job.output_path)
        return claim

    monkeypatch.setattr("pixelup.runner.run_upscale", _upscale)
    monkeypatch.setattr("pixelup.runner.write_sidecar", _sidecar)
    worker = JobWorker(job)
    finished, _progress = _capture_worker(worker)

    worker.run()

    assert job.output_path.read_bytes() == b"external winner"
    assert not job.output_path.with_suffix(".json").exists()
    assert finished[0][1] is False


def test_worker_run_cancelled_emits_cancelled_result(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _session_log: None,
) -> None:
    job = _make_job(2, tmp_path)

    def raise_cancel(options: object, runtime_dirs: object, **kwargs: object) -> dict[str, object]:
        raise PixelupError(ErrorCode.JOB_CANCELLED, "Job cancelled.")

    monkeypatch.setattr("pixelup.runner.run_upscale", raise_cancel)
    worker = JobWorker(job)
    finished, _progress = _capture_worker(worker)

    worker.run()

    assert finished == [(job.id, False, "Cancelled", {"cancelled": True}, [])]


def test_worker_run_pixelup_error_emits_its_message(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _session_log: None,
) -> None:
    job = _make_job(3, tmp_path)

    def raise_error(options: object, runtime_dirs: object, **kwargs: object) -> dict[str, object]:
        raise PixelupError(ErrorCode.MODEL_NOT_FOUND, Message("error.modelFileMissing"))

    monkeypatch.setattr("pixelup.runner.run_upscale", raise_error)
    worker = JobWorker(job)
    finished, _progress = _capture_worker(worker)

    worker.run()

    assert finished == [(job.id, False, "Model file is missing.", {}, [])]


def test_worker_run_unexpected_error_is_wrapped(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _session_log: None,
) -> None:
    job = _make_job(4, tmp_path)

    diagnostic_sentinel = "ValueError EACCES /private/var/tmp/pixelup-internal"

    def boom(options: object, runtime_dirs: object, **kwargs: object) -> dict[str, object]:
        raise ValueError(diagnostic_sentinel)

    monkeypatch.setattr("pixelup.runner.run_upscale", boom)
    worker = JobWorker(job)
    finished, _progress = _capture_worker(worker)

    worker.run()

    assert finished == [(job.id, False, "The image could not be upscaled. Retry the job.", {}, [])]
    assert diagnostic_sentinel not in finished[0][2]


def test_worker_run_forwards_processing_progress_as_text(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _session_log: None,
) -> None:
    job = _make_job(5, tmp_path)

    def fake_upscale(options: object, runtime_dirs: object, **kwargs: object) -> dict[str, object]:
        kwargs["on_progress"]("upscale")  # type: ignore[operator]
        kwargs["on_tile"](1, 4)  # type: ignore[operator]
        kwargs["on_output_published"](_publish_fixture(job.output_path, b"image"))  # type: ignore[operator]
        return {"ok": True, "output": str(getattr(options, "output_path", ""))}

    monkeypatch.setattr("pixelup.runner.run_upscale", fake_upscale)
    monkeypatch.setattr(
        "pixelup.runner.write_sidecar",
        lambda **kwargs: _publish_fixture(kwargs["output_path"].with_suffix(".json"), b"{}"),
    )
    worker = JobWorker(job)
    _finished, progress = _capture_worker(worker)

    worker.run()

    assert (job.id, "Upscaling…") in progress
    assert (job.id, "1/4 tiles processed") in progress


def test_worker_request_cancel_sets_the_should_cancel_flag(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    worker = JobWorker(_make_job(6, tmp_path))
    assert worker._is_cancelled() is False
    worker.request_cancel()
    assert worker._is_cancelled() is True


@pytest.mark.heavy
def test_every_upscale_model_enlarges_a_corpus_photo_faithfully(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    heavy_models_dir: Path,
    process_until,
    corpus_file,
) -> None:
    monkeypatch.setenv("PIXELUP_DATA_DIR", str(tmp_path / "home"))
    source = tmp_path / "input" / "apartment-cat.png"
    source.parent.mkdir()
    with Image.open(corpus_file("photos/similarity/apartment-cat/reference.jpg")) as photo:
        small = photo.convert("RGB").resize((96, 64), Image.Resampling.LANCZOS)
    small.save(source)

    settings = JobSettings()
    jobs = create_jobs(
        input_paths=[source],
        models=list(UPSCALE_MODELS),
        settings=settings,
        existing_jobs=[],
        job_ids=count(1),
    )
    runner = JobRunner(
        jobs,
        runtime_dirs=RuntimeDirs(models_dir=heavy_models_dir, temp_dir=tmp_path / "temp"),
    )
    outcomes: dict[int, tuple[bool, str, object]] = {}
    runner.finished.connect(
        lambda job_id, ok, message, result, _warnings: outcomes.__setitem__(
            job_id, (ok, message, result)
        )
    )
    runner.schedule(1)
    process_until(
        lambda: len(outcomes) == len(jobs) and runner.cleanup_for_quit(),
        timeout_s=30 * 60,
        what="Upscaling with every model",
    )

    for job in jobs:
        ok, message, result = outcomes[job.id]
        assert ok, f"{job.model}: {message}"
        assert isinstance(result, dict)
        assert result["output_size"] == [96 * settings.scale, 64 * settings.scale], job.model
        with Image.open(job.output_path) as output:
            reduced = output.convert("RGB").resize(small.size, Image.Resampling.LANCZOS)
        difference = ImageChops.difference(small, reduced)
        mean = sum(ImageStat.Stat(difference).mean) / 3
        # Measured on the M4: faithful models drift 5 to 12 levels, since the GANs
        # invent fine texture, while the photo against its own mirror image
        # differs by 92. The bound sits between them.
        assert mean < 25, f"{job.model} drifted from the photo by {mean:.1f} levels on average"


def test_folder_switch_keeps_accepted_jobs_on_original_folder(
    qapp, tmp_path, monkeypatch, process_until
):
    first, second, later = (_make_job(i, tmp_path) for i in range(1, 4))
    jobs = [first, second]
    original = RuntimeDirs(tmp_path / "old", tmp_path / "temp")
    replacement = RuntimeDirs(tmp_path / "new", tmp_path / "temp")
    runner = JobRunner(jobs, runtime_dirs=original)
    observed = {}

    def run(worker):
        observed[worker.job.id] = worker._runtime_dirs.models_dir
        worker.job.status = "completed"
        worker.signals.finished.emit(worker.job.id, True, Message("queue.statusDone"), {}, [])

    monkeypatch.setattr(JobWorker, "run", run)
    runner.set_runtime_dirs(replacement)
    jobs.append(later)
    runner.schedule(1)
    process_until(lambda: len(observed) == 3 and not runner._threads,
                  timeout_s=5, what="Jobs captured before and after the folder switch")
    assert observed == {1: original.models_dir, 2: original.models_dir, 3: replacement.models_dir}


# A disposable process runs real job workers whose upscale fails at once. Each
# worker's destruction is slowed by a hook, which widens the overlap in which a
# Qt-side deleteLater on the job thread and the runner's release on the window
# thread both deleted the worker (SIGSEGV). Every worker must instead be deleted
# once, by the runner, on the window thread.
_WORKER_RELEASE_PROCESS = r"""
import sys
import threading
import time
from pathlib import Path
from PySide6.QtCore import QCoreApplication, Qt, QTimer
from pixelup import runner
from pixelup.errors import ErrorCode, PixelupError
from pixelup.i18n.message import Message
from pixelup.jobs import Job, JobSettings

def failing_upscale(*_args, **_kwargs):
    time.sleep(0.02)
    raise PixelupError(ErrorCode.INVALID_ARGUMENT, Message("jobs.failedFallback"))

runner.run_upscale = failing_upscale
deleted_on_window_thread = []
start_job = runner.JobRunner._start_job

def start_watched_job(self, job):
    start_job(self, job)
    _thread, worker = self._threads[job.id]
    def destroyed(*_args):
        deleted_on_window_thread.append(threading.current_thread() is threading.main_thread())
        time.sleep(0.02)
    worker.destroyed.connect(destroyed, Qt.ConnectionType.DirectConnection)

runner.JobRunner._start_job = start_watched_job
folder = Path(sys.argv[1])
count = 10
app = QCoreApplication([])
jobs = [
    Job(id=i, input_path=folder / f"{i}.png", model="realesr-general-x4v3",
        output_path=folder / f"{i}-out.png", settings=JobSettings(), status="pending")
    for i in range(1, count + 1)
]
job_runner = runner.JobRunner(jobs)
finished = []
def record(job_id, *_outcome):
    finished.append(job_id)
    if len(finished) == count:
        app.quit()
job_runner.finished.connect(record)
QTimer.singleShot(0, lambda: job_runner.schedule(1))
app.exec()
app.processEvents()
print("finished", len(finished), deleted_on_window_thread.count(True), flush=True)
"""


def test_finished_workers_are_released_once_without_crashing(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _WORKER_RELEASE_PROCESS, str(tmp_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.splitlines()[-1] == "finished 10 10"


def test_a_worker_that_ends_without_a_result_still_fails_its_job_and_frees_the_slot(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process_until,
) -> None:
    # An unexpected error outside the job's own handling must not leave it running.
    jobs = [_make_job(i, tmp_path) for i in (1, 2)]

    def broken(self) -> None:
        raise RuntimeError("unexpected")

    monkeypatch.setattr("pixelup.runner.JobWorker._run", broken)
    monkeypatch.setattr("sys.excepthook", lambda *_args: None)
    runner = JobRunner(jobs)
    outcomes: list[tuple[object, ...]] = []
    runner.finished.connect(lambda *args: outcomes.append(args))
    runner.schedule(1)
    process_until(lambda: len(outcomes) == 2, timeout_s=10, what="Both jobs settled")

    assert [(job_id, ok) for job_id, ok, *_rest in outcomes] == [(1, False), (2, False)]
    assert outcomes[0][2] == FAILED_FALLBACK
    assert runner._active_jobs == 0
    assert runner._threads == {}


def test_running_out_of_memory_outside_inference_fails_the_job_with_its_advice(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process_until,
) -> None:
    # Encoding a large output can exhaust memory after inference succeeded.
    def exhausted(*_args, **_kwargs):
        raise MemoryError

    monkeypatch.setattr("pixelup.runner.run_upscale", exhausted)
    jobs = [_make_job(1, tmp_path)]
    runner = JobRunner(jobs)
    outcomes: list[tuple[object, ...]] = []
    runner.finished.connect(lambda *args: outcomes.append(args))
    runner.schedule(1)
    process_until(lambda: len(outcomes) == 1, timeout_s=10, what="Job settled")

    job_id, ok, message, _result, _warnings = outcomes[0]
    assert (job_id, ok) == (1, False)
    text = english().of(message)
    assert english().t("error.outOfMemory") in text
    assert english().t("error.hintOutOfMemory") in text
    assert list(tmp_path.glob("out-1*")) == []

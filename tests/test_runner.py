from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
from itertools import count
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from PIL import Image, ImageChops, ImageStat
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from pixelup.config import RuntimeDirs
from pixelup.errors import ErrorCode, PixelupError
from pixelup.i18n.localizer import english
from pixelup.i18n.message import Message
from pixelup.jobs import Job, JobSettings, create_jobs, retry_failed_jobs
from pixelup.model_management import UPSCALE_MODELS
from pixelup.runner import (
    FAILED_FALLBACK,
    JobRunner,
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
# The runner's real worker thread runs a gated stand-in for the job body, so the
# order, cancellation and quit can be held at any point without inference.


def _make_job(job_id: int, tmp_path: Path) -> Job:
    return Job(
        id=job_id,
        input_path=tmp_path / f"in-{job_id}.png",
        model="realesr-general-x4v3",
        output_path=tmp_path / f"out-{job_id}.png",
        settings=JobSettings(),
    )


class _HeldJobs:
    """Replaces the job body: each run records itself and waits for its release."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.runs: list[tuple[int, RuntimeDirs | None, Event, object]] = []
        self.release: dict[int, Event] = {}
        self.started = Event()

        def run(worker, job, runtime_dirs, cancel):
            release = self.release.setdefault(len(self.runs), Event())
            self.runs.append((job.id, runtime_dirs, cancel, QThread.currentThread()))
            self.started.set()
            release.wait()
            if cancel.is_set():
                return False, Message("queue.statusCancelled"), {"cancelled": True}, []
            return True, Message("queue.statusDone"), {}, []

        monkeypatch.setattr(JobWorker, "_run", run)

    def finish(self, index: int) -> None:
        self.release.setdefault(index, Event()).set()

    def finish_all(self) -> None:
        for index in range(len(self.runs) + 8):
            self.finish(index)


def _stop(runner: JobRunner, held: _HeldJobs, process_until) -> None:
    held.finish_all()
    process_until(runner.cleanup_for_quit, timeout_s=5, what="Runner quit")


def test_jobs_run_one_at_a_time_in_order_on_one_worker_thread(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process_until,
) -> None:
    held = _HeldJobs(monkeypatch)
    jobs = [_make_job(i, tmp_path) for i in (1, 2, 3)]
    runner = JobRunner(jobs)
    outcomes: list[tuple[object, ...]] = []
    runner.finished.connect(lambda *args: outcomes.append(args))
    runner.schedule()
    try:
        process_until(lambda: len(held.runs) == 1, timeout_s=5, what="First job start")
        runner.schedule()
        qapp.processEvents()
        assert [job.status for job in jobs] == ["running", "pending", "pending"]
        assert len(held.runs) == 1

        for index in range(3):
            held.finish(index)
            process_until(
                lambda expected=index + 1: len(outcomes) == expected,
                timeout_s=5, what="Job outcome",
            )
        assert [outcome[0:2] for outcome in outcomes] == [(1, True), (2, True), (3, True)]
        assert [run[0] for run in held.runs] == [1, 2, 3]
        threads = {run[3] for run in held.runs}
        assert len(threads) == 1 and QThread.currentThread() not in threads
    finally:
        _stop(runner, held, process_until)


def test_retry_with_same_job_id_runs_after_its_predecessor_and_keeps_cancel(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process_until,
) -> None:
    held = _HeldJobs(monkeypatch)
    job = _make_job(1, tmp_path)
    runner = JobRunner([job])
    outcomes: list[tuple[object, ...]] = []

    def retry_after_first(*outcome: object) -> None:
        outcomes.append(outcome)
        if len(outcomes) == 1:
            job.status = "failed"
            assert retry_failed_jobs([job]) == [job.id]
            runner.schedule()

    runner.finished.connect(retry_after_first)
    runner.schedule()
    try:
        process_until(lambda: len(held.runs) == 1, timeout_s=5, what="First run")
        held.finish(0)
        process_until(lambda: len(held.runs) == 2, timeout_s=5, what="Same-ID retry start")
        assert len(outcomes) == 1

        runner.request_cancel(job.id)
        assert held.runs[1][2].is_set()
        assert not held.runs[0][2].is_set()
        assert runner.cleanup_for_quit() is False
        held.finish(1)
        process_until(lambda: len(outcomes) == 2, timeout_s=5, what="Cancelled retry result")
        assert outcomes[1] == (job.id, False, Message("queue.statusCancelled"),
                               {"cancelled": True}, [])
        assert runner.cleanup_for_quit() is True
    finally:
        _stop(runner, held, process_until)


def test_quit_cancels_the_running_job_and_stops_the_thread_once_it_reports(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process_until,
) -> None:
    held = _HeldJobs(monkeypatch)
    jobs = [_make_job(i, tmp_path) for i in (1, 2, 3)]
    runner = JobRunner(jobs)
    outcomes: list[tuple[object, ...]] = []
    idle: list[bool] = []
    runner.finished.connect(lambda *args: outcomes.append(args))
    runner.idle.connect(lambda: idle.append(True))
    runner.schedule()
    try:
        process_until(lambda: len(held.runs) == 1, timeout_s=5, what="First job start")
        assert runner.cleanup_for_quit() is False
        assert held.runs[0][2].is_set()
        # Unstarted jobs end at once and are never started afterwards.
        assert [outcome[0] for outcome in outcomes] == [2, 3]
        assert [job.status for job in jobs[1:]] == ["cancelled", "cancelled"]
        assert [english().of(job.message) for job in jobs[1:]] == ["Cancelled", "Cancelled"]
        held.finish(0)
        process_until(lambda: idle == [True], timeout_s=5, what="Cancelled job report")
        assert [outcome[0] for outcome in outcomes] == [2, 3, 1]
        assert runner.cleanup_for_quit() is True
        assert not runner._thread.isRunning()
        runner.schedule()
        assert len(held.runs) == 1
    finally:
        _stop(runner, held, process_until)


def test_admitted_jobs_replace_old_models_folder_capture(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process_until,
) -> None:
    held = _HeldJobs(monkeypatch)
    first, second, later = (_make_job(i, tmp_path) for i in range(1, 4))
    jobs = [first, second]
    original = RuntimeDirs(tmp_path / "original")
    accepted = RuntimeDirs(tmp_path / "accepted")
    replacement = RuntimeDirs(tmp_path / "replacement")
    runner = JobRunner(jobs, runtime_dirs=original)
    runner.set_runtime_dirs(replacement)  # Captures original for existing pending jobs.
    runner.retain_runtime_dirs([second.id], accepted)
    jobs.append(later)
    held.finish_all()
    runner.schedule()
    try:
        process_until(lambda: len(held.runs) == 3, timeout_s=5, what="Three jobs")
        assert [run[1] for run in held.runs] == [original, accepted, replacement]
        assert runner._queued_runtime_dirs == {}
    finally:
        _stop(runner, held, process_until)


def test_request_cancel_for_a_job_not_running_is_a_noop(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process_until,
) -> None:
    held = _HeldJobs(monkeypatch)
    runner = JobRunner([_make_job(1, tmp_path), _make_job(2, tmp_path)])
    runner.request_cancel(1)
    runner.schedule()
    try:
        process_until(lambda: len(held.runs) == 1, timeout_s=5, what="First job start")
        runner.request_cancel(2)
        runner.request_cancel(999)
        assert not held.runs[0][2].is_set()
    finally:
        _stop(runner, held, process_until)


def test_cleanup_for_quit_without_jobs_cancels_pending_ones(tmp_path: Path) -> None:
    jobs = [_make_job(1, tmp_path)]
    runner = JobRunner(jobs)
    assert runner.cleanup_for_quit() is True
    assert jobs[0].status == "cancelled"


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


@pytest.fixture
def _session_log() -> None:
    # Route the singleton logger to the test's records database so job lines
    # don't spill to stderr. conftest restores the logger after the test.
    configure_session_logging()


# --- The job body (no threads) ----------------------------------------
#
# Run the worker body directly on the main thread (no moveToThread / QThread) with
# run_upscale + write_sidecar mocked, capturing the emitted signals. This covers
# the success / cancelled / failed / unexpected-error branches and the progress
# and cancel callback wiring independently of native thread dispatch.


class _Worker:
    """The real job body, run on the test thread."""

    def __init__(self, job: Job, runtime_dirs: RuntimeDirs | None = None) -> None:
        self.job = job
        self.runtime_dirs = runtime_dirs
        self.cancel = Event()
        self.body = JobWorker()

    def request_cancel(self) -> None:
        self.cancel.set()

    def run(self) -> None:
        self.body.run_job(self.job, self.runtime_dirs, self.cancel)


def _capture_worker(worker: _Worker) -> tuple[list, list]:
    finished: list[tuple[object, ...]] = []
    progress: list[tuple[object, ...]] = []
    # Status text travels as a Message; the capture reads it back in English, the
    # source language, so each test can still say what the row is told.
    def english_args(args: tuple[object, ...]) -> tuple[object, ...]:
        return tuple(english().of(arg) if isinstance(arg, Message) else arg for arg in args)

    worker.body.finished.connect(lambda *args: finished.append(english_args(args)))
    worker.body.progress.connect(lambda *args: progress.append(english_args(args)))
    return finished, progress


def _publish_fixture(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def _worker_with_real_image_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Worker:
    from imaging_support import image_from_bgr_array

    from pixelup import inference

    job = _make_job(81, tmp_path)
    job.model = "RealESRGAN_x4plus"
    job.settings = JobSettings(device="cpu")
    Image.new("RGB", (3, 2), (20, 40, 60)).save(job.input_path)
    models = tmp_path / "models"
    models.mkdir()
    (models / "RealESRGAN_x4plus.pth").write_bytes(b"weights")
    monkeypatch.setattr(inference, "_import_torch", lambda: None)
    monkeypatch.setattr(inference, "to_torch_device", lambda *args: "cpu")
    monkeypatch.setattr(inference, "_create_upsampler", lambda *args, **kwargs:
                        SimpleNamespace(
                            enhance=lambda pixels, **kwargs: (image_from_bgr_array(pixels), None)
                        ))
    return _Worker(job, RuntimeDirs(models))


@pytest.mark.parametrize("boundary", ["encode", "sidecar"])
@pytest.mark.parametrize("hard_links", [True, False])
def test_worker_cancel_during_real_writes_removes_owned_output(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str,
    hard_links: bool,
) -> None:
    if not hard_links:
        def unsupported(*_args: object, **_kwargs: object) -> None:
            raise OSError(errno.EOPNOTSUPP, "hard links unsupported")
        monkeypatch.setattr("pixelup.output_publication.os.link", unsupported)
    worker = _worker_with_real_image_pipeline(tmp_path, monkeypatch)
    finished, _ = _capture_worker(worker)
    sidecar = worker.job.output_path.with_suffix(".json")
    boundaries = []
    files_at_completion = []
    worker.body.finished.connect(lambda *args: files_at_completion.append(
        (worker.job.output_path.exists(), sidecar.exists())
    ))
    if boundary == "encode":
        original = Image.Image.save

        def cancel_in_encoder(self, file, **kwargs):
            original(self, file, **kwargs)
            boundaries.append("encode")
            worker.request_cancel()

        monkeypatch.setattr(Image.Image, "save", cancel_in_encoder)
    else:
        original_fsync = os.fsync

        def cancel_in_sidecar(descriptor):
            original_fsync(descriptor)
            if worker.job.output_path.exists():
                assert not sidecar.exists()
                stages = list(tmp_path.glob("pixelup-*.tmp"))
                assert len(stages) == 1
                assert json.loads(stages[0].read_bytes())["status"] == "success"
                boundaries.append("sidecar")
                worker.request_cancel()

        monkeypatch.setattr(os, "fsync", cancel_in_sidecar)

    worker.run()

    assert len(finished) == 1
    assert finished[0][1:4] == (False, "Cancelled", {"cancelled": True})
    assert boundaries == [boundary]
    assert files_at_completion == [(False, False)]
    assert not worker.job.output_path.exists()
    assert not sidecar.exists()
    assert list(tmp_path.glob("*.tmp")) == []


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
    worker = _Worker(job)
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
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr("pixelup.runner.run_upscale", _upscale)
    worker = _Worker(job)
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
        kwargs["on_output_published"](_publish_fixture(job.output_path, b"pixelup"))
        return {"ok": True}

    def _lose_sidecar(**kwargs: object) -> Path:
        job.output_path.with_suffix(".json").write_bytes(b"external-sidecar")
        raise PixelupError(ErrorCode.OUTPUT_EXISTS, Message("error.sidecarExists"))

    monkeypatch.setattr("pixelup.runner.run_upscale", _upscale)
    monkeypatch.setattr("pixelup.runner.write_sidecar", _lose_sidecar)
    worker = _Worker(job)
    finished, _progress = _capture_worker(worker)

    worker.run()

    assert not job.output_path.exists()
    assert job.output_path.with_suffix(".json").read_bytes() == b"external-sidecar"
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
    worker = _Worker(job)
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
    worker = _Worker(job)
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
    worker = _Worker(job)
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
    worker = _Worker(job)
    _finished, progress = _capture_worker(worker)

    worker.run()

    assert (job.id, "Upscaling…") in progress
    assert (job.id, "1/4 tiles processed") in progress


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
        runtime_dirs=RuntimeDirs(models_dir=heavy_models_dir),
    )
    outcomes: dict[int, tuple[bool, str, object]] = {}
    runner.finished.connect(
        lambda job_id, ok, message, result, _warnings: outcomes.__setitem__(
            job_id, (ok, message, result)
        )
    )
    runner.schedule()
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


# A QThread destroyed while it runs ends the process with a fatal Qt error. A
# disposable process runs jobs on the runner's worker thread, then deletes the
# runner without a quit, as discarding a window does, and exits normally.
_RUNNER_RELEASE_PROCESS = r"""
import sys
from pathlib import Path
from PySide6.QtCore import QCoreApplication, QTimer
from shiboken6 import delete
from pixelup import runner
from pixelup.errors import ErrorCode, PixelupError
from pixelup.i18n.message import Message
from pixelup.jobs import Job, JobSettings

def failing_upscale(*_args, **_kwargs):
    raise PixelupError(ErrorCode.INVALID_ARGUMENT, Message("jobs.failedFallback"))

runner.run_upscale = failing_upscale
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
QTimer.singleShot(0, job_runner.schedule)
app.exec()
thread_was_running = job_runner._thread.isRunning()
delete(job_runner)
print("finished", len(finished), thread_was_running, flush=True)
"""


def test_deleting_a_runner_stops_its_worker_thread_first(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _RUNNER_RELEASE_PROCESS, str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.splitlines()[-1] == "finished 10 True"


def test_a_worker_that_ends_without_a_result_still_fails_its_job_and_frees_the_slot(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process_until,
) -> None:
    # An unexpected error outside the job's own handling must not leave it running.
    jobs = [_make_job(i, tmp_path) for i in (1, 2)]

    def broken(self, *_args) -> None:
        raise RuntimeError("unexpected")

    monkeypatch.setattr("pixelup.runner.JobWorker._run", broken)
    monkeypatch.setattr("sys.excepthook", lambda *_args: None)
    runner = JobRunner(jobs)
    outcomes: list[tuple[object, ...]] = []
    runner.finished.connect(lambda *args: outcomes.append(args))
    runner.schedule()
    process_until(lambda: len(outcomes) == 2, timeout_s=10, what="Both jobs settled")

    assert [(job_id, ok) for job_id, ok, *_rest in outcomes] == [(1, False), (2, False)]
    assert outcomes[0][2] == FAILED_FALLBACK
    assert not runner.running
    assert runner.cleanup_for_quit() is True


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
    runner.schedule()
    process_until(lambda: len(outcomes) == 1, timeout_s=10, what="Job settled")

    job_id, ok, message, _result, _warnings = outcomes[0]
    assert (job_id, ok) == (1, False)
    text = english().of(message)
    assert english().t("error.outOfMemory") in text
    assert english().t("error.hintOutOfMemory") in text
    assert list(tmp_path.glob("out-1*")) == []

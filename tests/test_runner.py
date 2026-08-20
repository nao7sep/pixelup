from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

from pixelup.errors import ErrorCode, PixelupError
from pixelup.jobs import Job, JobSettings
from pixelup.runner import (
    JobRunner,
    JobWorker,
    _download_text,
    _progress_text,
    _tile_progress_text,
    failure_message,
)
from pixelup.session_log import configure_session_logging


def test_download_text_with_total_shows_percentage() -> None:
    assert _download_text("RealESRGAN_x4plus", 1, 4) == "25% - downloading RealESRGAN_x4plus"


def test_download_text_without_total_shows_plain_message() -> None:
    assert _download_text("modelx", 0, None) == "Downloading modelx"
    assert _download_text("modelx", 0, 0) == "Downloading modelx"


def test_progress_text_maps_known_phases() -> None:
    assert _progress_text("upscale") == "Upscaling"
    assert _progress_text("encode") == "Saving"


def test_progress_text_humanizes_unknown_phase() -> None:
    assert _progress_text("post-process_step") == "Post process step"


def test_tile_progress_text() -> None:
    assert _tile_progress_text(3, 8) == "3/8 tiles processed"


# --- JobRunner scheduling -------------------------------------------------
#
# The scheduling decisions (concurrency cap, pending selection, reschedule on
# finish) are tested without spinning real QThreads: _start_job is stubbed to do
# only the bookkeeping a started job would, and finishes are simulated by calling
# the real _job_finished. The reschedule it posts via QTimer.singleShot fires on
# the main thread under qapp.processEvents — no worker thread, no real inference.
# (Real-thread end-to-end runs are intentionally not driven here: rapid QThread
# create/destroy under a pumped event loop is crash-prone, so the suite stays on
# the deterministic main-thread path, the same choice the rest of the suite makes.)


def _make_job(job_id: int, tmp_path: Path) -> Job:
    return Job(
        id=job_id,
        input_path=tmp_path / f"in-{job_id}.png",
        model="realesr-general-x4v3",
        output_path=tmp_path / f"out-{job_id}.png",
        settings=JobSettings(),
        auto_download=False,
    )


class _RecordingRunner(JobRunner):
    """JobRunner whose thread spawn is replaced by bookkeeping only.

    Records the order jobs were started so the scheduling logic can be asserted
    directly, while leaving schedule()/_next_pending_job()/_job_finished() — the
    parts under test — completely real.
    """

    def __init__(self, jobs: list[Job]) -> None:
        super().__init__(jobs)
        self.started_ids: list[int] = []

    def _start_job(self, job: Job) -> None:
        job.status = "running"
        job.message = "Starting"
        self._active_jobs += 1
        self.started_ids.append(job.id)


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
        runner._job_finished(job_id, True, "Done", {"ok": True}, [])
        qapp.processEvents()  # let the QTimer-posted reschedule run

    # Each finish pulled exactly the next pending job, one at a time.
    assert runner.started_ids == [1, 2, 3]
    assert [record[0] for record in finished] == [1, 2, 3]
    assert runner._active_jobs == 0


def test_reschedule_after_finish_refills_one_freed_slot(
    qapp: QApplication, tmp_path: Path
) -> None:
    # The QTimer.singleShot(0, schedule) that _job_finished posts (runner.py) reads the *current*
    # _active_jobs and cap when it fires. With four jobs and a cap of two, finishing one frees
    # exactly one slot, so the reschedule must start exactly one more — not drain the whole queue.
    jobs = [_make_job(i, tmp_path) for i in range(1, 5)]
    runner = _RecordingRunner(jobs)

    runner.schedule(2)
    assert runner.started_ids == [1, 2]

    runner._job_finished(1, True, "Done", {"ok": True}, [])
    qapp.processEvents()  # let the posted reschedule run

    assert runner.started_ids == [1, 2, 3]
    assert runner._active_jobs == 2


def test_finished_emits_outcome_to_listeners(qapp: QApplication, tmp_path: Path) -> None:
    runner = _RecordingRunner([_make_job(1, tmp_path)])
    finished: list[tuple[object, ...]] = []
    runner.finished.connect(lambda *args: finished.append(args))

    runner.schedule(1)
    runner._job_finished(1, False, "Cancelled", {"cancelled": True}, ["w"])
    qapp.processEvents()

    assert finished == [(1, False, "Cancelled", {"cancelled": True}, ["w"])]


def test_failure_message_carries_the_hint_when_there_is_one() -> None:
    # The queue row is the only surface a failed job gets, so the remedy — e.g.
    # the Settings toggle that fixes a missing model — must ride along with the
    # diagnosis rather than existing only in the log.
    with_hint = PixelupError(
        ErrorCode.MODEL_NOT_FOUND,
        "Model 'x4plus' is not present in the models directory.",
        hint="Turn on \u201cDownload missing models automatically\u201d in Settings.",
    )
    assert failure_message(with_hint) == (
        "Model 'x4plus' is not present in the models directory. "
        "Turn on \u201cDownload missing models automatically\u201d in Settings."
    )

    bare = PixelupError(ErrorCode.OUT_OF_MEMORY, "Ran out of memory.")
    assert failure_message(bare) == "Ran out of memory."


def test_request_cancel_for_unknown_job_is_a_noop(tmp_path: Path) -> None:
    runner = JobRunner([_make_job(1, tmp_path)])
    # Nothing is running, so cancelling any id must simply do nothing.
    runner.request_cancel(1)
    runner.request_cancel(999)


def test_request_cancel_signals_the_running_worker(tmp_path: Path) -> None:
    # When a job is live, request_cancel reaches its worker. Driven through a fake registry entry
    # so the cancel routing is asserted without a real QThread (see the no-real-threads note above).
    runner = JobRunner([_make_job(1, tmp_path)])
    worker = _FakeWorker(1)
    runner._threads = {1: (SimpleNamespace(), worker)}  # type: ignore[dict-item]

    runner.request_cancel(1)

    assert worker.cancel_requested is True


# --- cleanup_for_quit teardown sequence -----------------------------------
#
# cleanup_for_quit coordinates the shutdown of live worker threads. Real threads
# are not used (see note above); instead fakes stand in for the (thread, worker)
# registry so the exact teardown contract can be asserted deterministically:
# every worker is cancelled, the two UI-facing slots are disconnected (while the
# finished -> quit -> deleteLater wiring is left intact), and each thread is quit
# *before* it is waited on.


class _FakeSignal:
    def __init__(self) -> None:
        self.disconnected: list[object] = []

    def disconnect(self, slot: object) -> None:
        self.disconnected.append(slot)


class _FakeWorker:
    def __init__(self, job_id: int) -> None:
        self.job = SimpleNamespace(id=job_id)
        self.signals = SimpleNamespace(progress=_FakeSignal(), finished=_FakeSignal())
        self.cancel_requested = False

    def request_cancel(self) -> None:
        self.cancel_requested = True


class _FakeThread:
    def __init__(self, name: str, calls: list[tuple[str, str]]) -> None:
        self._name = name
        self._calls = calls

    def quit(self) -> None:
        self._calls.append((self._name, "quit"))

    def wait(self, msecs: int) -> bool:
        self._calls.append((self._name, f"wait:{msecs}"))
        return True


@pytest.fixture
def _session_log(tmp_path: Path) -> None:
    # Route the singleton logger to a temp file so the quit.* lines don't spill
    # to stderr. conftest restores the logger after the test.
    configure_session_logging(tmp_path / "logs" / "session.log")


def test_cleanup_for_quit_teardown_sequence(_session_log: None) -> None:
    runner = JobRunner([])
    calls: list[tuple[str, str]] = []
    workers = [_FakeWorker(1), _FakeWorker(2)]
    threads = [_FakeThread("a", calls), _FakeThread("b", calls)]
    runner._threads = {  # type: ignore[assignment]
        1: (threads[0], workers[0]),
        2: (threads[1], workers[1]),
    }

    runner.cleanup_for_quit()

    for worker in workers:
        assert worker.cancel_requested is True
        # The finished slot disconnected is exactly _job_finished — the
        # finished -> quit/deleteLater wiring is deliberately left connected.
        assert worker.signals.finished.disconnected == [runner._job_finished]
        # Progress is disconnected once (from runner.progress.emit).
        assert len(worker.signals.progress.disconnected) == 1

    # Each thread is quit before it is waited on, with the bounded 2s timeout.
    for name in ("a", "b"):
        thread_calls = [call for call in calls if call[0] == name]
        assert thread_calls == [(name, "quit"), (name, "wait:2000")]


def test_cleanup_for_quit_is_a_noop_without_threads(_session_log: None) -> None:
    runner = JobRunner([])
    runner.cleanup_for_quit()
    assert runner._threads == {}


# --- JobWorker.run branches (no threads) ----------------------------------
#
# Run the worker body directly on the main thread (no moveToThread / QThread) with
# run_upscale + write_sidecar mocked, capturing the emitted signals. This covers
# the success / cancelled / failed / unexpected-error branches and the progress
# and cancel callback wiring — the part the real-thread harness can't reach
# without segfaulting.


def _capture_worker(worker: JobWorker) -> tuple[list, list]:
    finished: list[tuple[object, ...]] = []
    progress: list[tuple[object, ...]] = []
    worker.signals.finished.connect(lambda *args: finished.append(args))
    worker.signals.progress.connect(lambda *args: progress.append(args))
    return finished, progress


def test_worker_run_success_emits_done_with_sidecar(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _session_log: None,
) -> None:
    job = _make_job(1, tmp_path)

    def fake_upscale(options: object, runtime_dirs: object, **kwargs: object) -> dict[str, object]:
        kwargs["on_warning"]("note")  # type: ignore[operator]
        return {"ok": True, "output": getattr(options, "output_arg", "")}

    monkeypatch.setattr("pixelup.runner.run_upscale", fake_upscale)
    monkeypatch.setattr(
        "pixelup.runner.write_sidecar",
        lambda **kwargs: kwargs["output_path"].with_suffix(".json"),
    )
    worker = JobWorker(job)
    finished, _progress = _capture_worker(worker)

    worker.run()

    assert len(finished) == 1
    job_id, ok, message, result, warnings = finished[0]
    assert (job_id, ok, message) == (job.id, True, "Done")
    assert result["ok"] is True
    assert str(result["sidecar"]).endswith(".json")
    assert warnings == ["note"]


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
        raise PixelupError(ErrorCode.MODEL_NOT_FOUND, "Model missing.")

    monkeypatch.setattr("pixelup.runner.run_upscale", raise_error)
    worker = JobWorker(job)
    finished, _progress = _capture_worker(worker)

    worker.run()

    assert finished == [(job.id, False, "Model missing.", {}, [])]


def test_worker_run_unexpected_error_is_wrapped(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _session_log: None,
) -> None:
    job = _make_job(4, tmp_path)

    def boom(options: object, runtime_dirs: object, **kwargs: object) -> dict[str, object]:
        raise ValueError("boom")

    monkeypatch.setattr("pixelup.runner.run_upscale", boom)
    worker = JobWorker(job)
    finished, _progress = _capture_worker(worker)

    worker.run()

    assert finished == [(job.id, False, "Unexpected error: boom", {}, [])]


def test_worker_run_forwards_progress_phases_as_text(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _session_log: None,
) -> None:
    job = _make_job(5, tmp_path)

    def fake_upscale(options: object, runtime_dirs: object, **kwargs: object) -> dict[str, object]:
        kwargs["on_progress"]("upscale")  # type: ignore[operator]
        kwargs["on_tile"](1, 4)  # type: ignore[operator]
        kwargs["on_download"]("RealESRGAN_x4plus", 1, 4)  # type: ignore[operator]
        return {"ok": True, "output": getattr(options, "output_arg", "")}

    monkeypatch.setattr("pixelup.runner.run_upscale", fake_upscale)
    monkeypatch.setattr(
        "pixelup.runner.write_sidecar",
        lambda **kwargs: kwargs["output_path"].with_suffix(".json"),
    )
    worker = JobWorker(job)
    _finished, progress = _capture_worker(worker)

    worker.run()

    assert (job.id, "Upscaling") in progress
    assert (job.id, "1/4 tiles processed") in progress
    assert (job.id, "25% - downloading RealESRGAN_x4plus") in progress


def test_worker_request_cancel_sets_the_should_cancel_flag(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    worker = JobWorker(_make_job(6, tmp_path))
    assert worker._is_cancelled() is False
    worker.request_cancel()
    assert worker._is_cancelled() is True

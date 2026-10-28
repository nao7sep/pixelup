from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from pixelup import model_manager
from pixelup.model_management import MANAGED_ARTIFACT_NAMES
from pixelup.model_manager import ModelManager, ModelOperation
from pixelup.model_registry import model_info
from pixelup.models import model_file, verify_model_file


class _HeldScans:
    """Stands in for the folder read, holding each background scan until released."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.answer: frozenset[str] = frozenset()
        self.release = threading.Event()
        self.threads: list[threading.Thread] = []
        real = model_manager.read_ready_names

        def read(models_dir: Path) -> frozenset[str]:
            if threading.current_thread() is threading.main_thread():
                return real(models_dir)
            self.threads.append(threading.current_thread())
            assert self.release.wait(10), "a held scan was never released"
            return self.answer

        monkeypatch.setattr(model_manager, "read_ready_names", read)


@pytest.fixture
def settle(qapp: QApplication, background, process_until):
    def run(_manager: ModelManager | None = None) -> None:
        process_until(background.settled, timeout_s=10, what="The models scan")
        qapp.processEvents()

    return run


def test_async_readiness_uses_present_nonempty_files_and_refreshes_its_cache(
    qapp: QApplication, tmp_path: Path, settle
) -> None:
    present, empty, missing = MANAGED_ARTIFACT_NAMES[:3]
    model_file(tmp_path, present).write_bytes(b"present")
    model_file(tmp_path, empty).write_bytes(b"")
    manager = ModelManager(tmp_path)
    try:
        settle()
        assert manager.readiness_known
        assert manager.ready_names == frozenset({present})
        assert manager.missing((missing, empty, present)) == (missing, empty)
        assert manager.ready_count() == (1, len(MANAGED_ARTIFACT_NAMES))

        model_file(tmp_path, present).write_bytes(b"")
        model_file(tmp_path, empty).write_bytes(b"now present")
        manager.rescan()
        settle()

        assert manager.ready_names == frozenset({empty})
        assert manager.missing((missing, empty, present)) == (missing, present)
        assert manager.ready_count() == (1, len(MANAGED_ARTIFACT_NAMES))
    finally:
        manager.deleteLater()


def test_rescan_reads_the_folder_off_the_ui_thread_and_reports_a_change(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settle
) -> None:
    scans = _HeldScans(monkeypatch)
    manager = ModelManager(tmp_path)
    changes: list[None] = []
    manager.changed.connect(lambda: changes.append(None))
    try:
        assert manager.ready_names == frozenset()
        scans.answer = frozenset({MANAGED_ARTIFACT_NAMES[0]})
        scans.release.set()

        settle()

        assert len(scans.threads) == 1
        assert scans.threads[0] is not threading.main_thread()
        assert manager.ready_names == frozenset({MANAGED_ARTIFACT_NAMES[0]})
        assert changes == [None]
    finally:
        manager.deleteLater()


def test_each_rescan_reads_again_and_the_latest_answer_applies(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settle
) -> None:
    scans = _HeldScans(monkeypatch)
    manager = ModelManager(tmp_path)
    try:
        for _ in range(3):
            manager.rescan()
        scans.answer = frozenset({MANAGED_ARTIFACT_NAMES[0]})
        scans.release.set()
        settle()
        assert len(scans.threads) == 4
        # Older answers are dropped by their request token.
        manager._scan_finished(1, model_manager.ModelReadinessResult(frozenset()))
        assert manager.ready_names == frozenset({MANAGED_ARTIFACT_NAMES[0]})
    finally:
        manager.deleteLater()


def test_no_rescan_starts_while_an_install_runs_or_once_quitting_began(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settle
) -> None:
    scans = _HeldScans(monkeypatch)
    scans.release.set()
    manager = ModelManager(tmp_path)
    settle()
    scans.threads.clear()
    try:
        manager._operations[1] = ModelOperation(
            id=1, kind="running", artifact_names=(MANAGED_ARTIFACT_NAMES[0],)
        )
        manager.rescan()

        manager._operations.clear()
        manager.begin_shutdown()
        manager.rescan()
        settle()
        assert scans.threads == []
    finally:
        manager.deleteLater()


def test_switching_folder_ignores_a_stalled_old_scan(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process_until
) -> None:
    release = threading.Event()
    entered = threading.Event()
    old, new = tmp_path / "old", tmp_path / "new"
    name = MANAGED_ARTIFACT_NAMES[0]

    def read(folder):
        if folder == old:
            entered.set()
            assert release.wait(10)
            return frozenset({name})
        return frozenset()

    monkeypatch.setattr(model_manager, "read_ready_names", read)
    manager = ModelManager(old)
    try:
        assert entered.wait(5)
        manager.begin_folder_change()
        manager.end_folder_change(new)
        process_until(lambda: manager.readiness_known, timeout_s=10, what="The new folder scan")
        assert manager.readiness_known
        assert manager.ready_names == frozenset()
        release.set()
        # Deliver the stale result explicitly as well: it cannot mutate current facts.
        manager._scan_finished(1, model_manager.ModelReadinessResult(frozenset({name})))
        assert manager.ready_names == frozenset()
    finally:
        release.set()
        manager.deleteLater()


@pytest.mark.heavy
def test_every_managed_model_is_installed_and_matches_its_pin(
    qapp: QApplication,
    heavy_models_dir: Path,
    settle,
) -> None:
    manager = ModelManager(heavy_models_dir)
    settle()
    total = len(MANAGED_ARTIFACT_NAMES)
    assert manager.ready_count() == (total, total)
    for name in MANAGED_ARTIFACT_NAMES:
        # Raises unless the file's size and SHA-256 match the registry's pin.
        verify_model_file(model_file(heavy_models_dir, name), model_info(name))
    manager.deleteLater()


def test_installs_run_one_at_a_time_in_request_order(
    qapp, tmp_path, monkeypatch, process_until, settle
):
    first, second = MANAGED_ARTIFACT_NAMES[:2]
    entered, release = threading.Event(), threading.Event()
    downloads: list[tuple[str, int]] = []

    def install(folder, name, **kwargs):
        downloads.append((name, threading.get_ident()))
        if name == second:
            entered.set()
            release.wait()
        model_file(folder, name).write_bytes(b"model")

    monkeypatch.setattr(model_manager, "download_model", install)
    manager = ModelManager(tmp_path)
    settle()
    try:
        manager.install((second,), force=False)
        assert entered.wait(3)
        manager.install((first,), force=False)
        qapp.processEvents()
        assert manager.operations_for((second,))[0].kind == "running"
        assert manager.operations_for((first,))[0].kind == "queued"
        assert [name for name, _thread in downloads] == [second]
        release.set()
        process_until(lambda: manager.missing((first, second)) == () and not
                      manager.in_progress_operations, timeout_s=5, what="Both installs")
        assert [name for name, _thread in downloads] == [second, first]
        assert len({thread for _name, thread in downloads}) == 1
    finally:
        release.set()
        process_until(manager.cleanup_for_quit, timeout_s=5, what="Installs settle")
        manager.deleteLater()


def test_installs_requested_during_a_folder_change_go_to_the_new_folder(
    qapp, tmp_path, monkeypatch, process_until, settle
):
    old, new = tmp_path / "old", tmp_path / "new"
    name = MANAGED_ARTIFACT_NAMES[0]
    folders: list[Path] = []

    def install(folder, artifact, **kwargs):
        folders.append(folder)
        folder.mkdir(parents=True, exist_ok=True)
        model_file(folder, artifact).write_bytes(b"model")

    monkeypatch.setattr(model_manager, "download_model", install)
    manager = ModelManager(old)
    settle()
    manager.begin_folder_change()
    assert manager.folder_changing
    manager.install((name,), force=False)
    qapp.processEvents()
    assert manager.operations_for((name,))[0].kind == "queued"
    manager.end_folder_change(new)
    process_until(lambda: not manager.in_progress_operations, timeout_s=5, what="The install")
    assert folders == [new]
    assert not manager.folder_changing
    process_until(manager.cleanup_for_quit, timeout_s=5, what="Installs settle")
    manager.deleteLater()


def test_quit_never_waits_for_a_held_scan(qapp, tmp_path, monkeypatch, settle):
    scans = _HeldScans(monkeypatch)
    manager = ModelManager(tmp_path)
    try:
        manager.begin_shutdown()
        assert manager.cleanup_for_quit()
    finally:
        scans.release.set()
        settle()
        manager.deleteLater()


# A QThread destroyed while it runs ends the process with a fatal Qt error. A
# disposable process runs installs on the manager's worker thread, then deletes
# the manager without a quit and exits normally.
_INSTALL_RELEASE_PROCESS = r"""
import sys
from pathlib import Path
from PySide6.QtCore import QCoreApplication, QTimer
from shiboken6 import delete
from pixelup import model_manager
from pixelup.errors import ErrorCode, PixelupError
from pixelup.i18n.message import Message

def failing_download(*_args, **_kwargs):
    raise PixelupError(ErrorCode.INVALID_ARGUMENT, Message("jobs.failedFallback"))

model_manager.download_model = failing_download
app = QCoreApplication([])
manager = model_manager.ModelManager(Path(sys.argv[1]))
count = 10
done = []
def next_install():
    if len(done) == count:
        app.quit()
        return
    done.append(manager.install(("realesr-general-x4v3",), force=True))
manager.idle.connect(lambda: QTimer.singleShot(0, next_install))
QTimer.singleShot(0, next_install)
app.exec()
thread_was_running = manager._thread.isRunning()
delete(manager)
print(len(done), thread_was_running, flush=True)
"""


def test_deleting_a_manager_stops_its_worker_thread_first(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _INSTALL_RELEASE_PROCESS, str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.splitlines()[-1] == "10 True"


def test_failed_scan_preserves_known_inventory_and_error_until_successful_retry(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settle,
) -> None:
    name = MANAGED_ARTIFACT_NAMES[0]
    model_file(tmp_path, name).write_bytes(b"present")
    manager = ModelManager(tmp_path)
    settle()
    failure = PermissionError(13, "unreadable", str(tmp_path))

    def broken(_models_dir: Path) -> frozenset[str]:
        raise failure

    real_read = model_manager.read_ready_names
    monkeypatch.setattr(model_manager, "read_ready_names", broken)
    try:
        manager.refresh_readiness()
        settle()
        assert not manager.readiness_known
        assert manager.readiness_error is failure
        assert manager.ready_names == frozenset({name})
        monkeypatch.setattr(model_manager, "read_ready_names", real_read)
        manager.refresh_readiness()
        settle()
        assert manager.readiness_known
        assert manager.readiness_error is None
        assert manager.ready_names == frozenset({name})
    finally:
        manager.deleteLater()


def test_a_cancelled_install_keeps_its_turn_until_its_connection_returns(
    qapp, tmp_path, monkeypatch, process_until, settle,
) -> None:
    from pixelup import models

    entered, release, closed = (threading.Event() for _ in range(3))

    class LateResponse:
        def close(self):
            closed.set()

    class BlockingOpener:
        def open(self, _url, *, timeout):
            entered.set()
            assert release.wait(5)
            return LateResponse()

    monkeypatch.setattr(models, "build_opener", lambda *_handlers: BlockingOpener())
    monkeypatch.setattr(model_manager, "download_model", models.download_model)
    manager = ModelManager(tmp_path)
    settle()
    first, second = MANAGED_ARTIFACT_NAMES[:2]
    cancelled = []
    manager.cancelled.connect(cancelled.append)
    operation_id = manager.install((first,), force=False)
    try:
        assert entered.wait(5)
        assert manager.cancel(operation_id)
        manager.install((second,), force=False)
        qapp.processEvents()
        assert manager.operations_for((first,))[0].cancelling
        assert manager.operations_for((second,))[0].kind == "queued"
        assert not manager.cleanup_for_quit()
        assert not cancelled
        manager.cancel_for((second,))
        assert cancelled == [(second,)]
    finally:
        release.set()
        process_until(manager.cleanup_for_quit, timeout_s=5, what="Connection settlement")
        manager.deleteLater()
    assert closed.is_set()
    assert cancelled == [(second,), (first,)]
    assert not model_file(tmp_path, first).exists()


def test_install_failure_logs_once_with_operation_artifact_and_original_cause(
    tmp_path, monkeypatch, caplog,
) -> None:
    from pixelup import models

    original = TimeoutError("sentinel transport failure")

    class FailingOpener:
        def open(self, _url, *, timeout):
            raise original

    monkeypatch.setattr(models, "build_opener", lambda *_handlers: FailingOpener())
    monkeypatch.setattr(model_manager, "download_model", models.download_model)
    name = MANAGED_ARTIFACT_NAMES[0]
    worker = model_manager.ModelInstallWorker()
    command = model_manager._Install(19, (name,), False, threading.Event())
    with caplog.at_level("INFO", logger="pixelup"):
        worker.run_install(command, tmp_path)
    failures = [record for record in caplog.records if record.levelname == "ERROR"]
    assert len(failures) == 1
    record = failures[0]
    assert record.msg == "models.install_failed"
    assert record.fields["operation_id"] == 19
    assert record.fields["artifact"] == name
    assert record.exc_info[1].__cause__ is original
    assert not any(record.msg == "model.download_failed" for record in caplog.records)
    assert any(record.msg == "model.download_started" for record in caplog.records)

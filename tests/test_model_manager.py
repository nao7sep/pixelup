from __future__ import annotations

import threading
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from pixelup import model_manager
from pixelup.model_management import MANAGED_ARTIFACT_NAMES
from pixelup.model_manager import ModelManager, ModelOperation
from pixelup.model_registry import known_model
from pixelup.models import model_file, verify_model_file


class _QuittableThread:
    def __init__(self) -> None:
        self.quit_calls = 0

    def quit(self) -> None:
        self.quit_calls += 1


def test_terminal_result_is_recorded_before_manager_requests_thread_shutdown(
    qapp: QApplication,
    tmp_path: Path,
    process_until,
) -> None:
    manager = ModelManager(tmp_path)
    _settle(manager, process_until)
    thread = _QuittableThread()
    manager._threads[7] = thread  # type: ignore[assignment]

    manager._worker_finished(7, True, False, None)

    assert manager._install_results[7] == (True, False, None)
    assert thread.quit_calls == 1
    manager.deleteLater()


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


def _settle(manager: ModelManager, process_until) -> None:
    process_until(lambda: not manager._scan_in_flight, timeout_s=10, what="The models scan")


def test_rescan_reads_the_folder_off_the_ui_thread_and_reports_a_change(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process_until
) -> None:
    scans = _HeldScans(monkeypatch)
    manager = ModelManager(tmp_path)
    changes: list[None] = []
    manager.changed.connect(lambda: changes.append(None))
    try:
        assert manager.ready_names == frozenset()
        scans.answer = frozenset({MANAGED_ARTIFACT_NAMES[0]})
        scans.release.set()

        _settle(manager, process_until)

        assert len(scans.threads) == 1
        assert scans.threads[0] is not threading.main_thread()
        assert manager.ready_names == frozenset({MANAGED_ARTIFACT_NAMES[0]})
        assert changes == [None]
    finally:
        manager.deleteLater()


def test_rescans_requested_while_one_runs_coalesce_into_one_more(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process_until
) -> None:
    scans = _HeldScans(monkeypatch)
    manager = ModelManager(tmp_path)
    try:
        manager.rescan()
        for _ in range(5):
            manager.rescan()
        scans.release.set()
        process_until(
            lambda: len(scans.threads) == 2 and not manager._scan_in_flight,
            timeout_s=10,
            what="The follow-up models scan",
        )
        qapp.processEvents()

        assert len(scans.threads) == 2
        assert manager._scan_again is False
    finally:
        manager.deleteLater()


def test_no_rescan_starts_while_an_install_runs_or_once_quitting_began(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process_until
) -> None:
    scans = _HeldScans(monkeypatch)
    scans.release.set()
    manager = ModelManager(tmp_path)
    _settle(manager, process_until)
    scans.threads.clear()
    try:
        manager._operations[1] = ModelOperation(
            id=1, kind="running", artifact_names=(MANAGED_ARTIFACT_NAMES[0],)
        )
        manager.rescan()
        assert manager._scan_in_flight is False

        manager._operations.clear()
        manager.begin_shutdown()
        manager.rescan()
        assert manager._scan_in_flight is False
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
        manager.set_models_dir(new)
        _settle(manager, process_until)
        assert manager.readiness_known
        assert manager.ready_names == frozenset()
        release.set()
        # Deliver the stale result explicitly as well: it cannot mutate current facts.
        manager._scan_finished(1, frozenset({name}))
        assert manager.ready_names == frozenset()
    finally:
        release.set()
        manager.deleteLater()


@pytest.mark.heavy
def test_every_managed_model_is_installed_and_matches_its_pin(
    qapp: QApplication,
    heavy_models_dir: Path,
    process_until,
) -> None:
    manager = ModelManager(heavy_models_dir)
    _settle(manager, process_until)
    total = len(MANAGED_ARTIFACT_NAMES)
    assert manager.ready_count() == (total, total)
    for name in MANAGED_ARTIFACT_NAMES:
        # Raises unless the file's size and SHA-256 match the registry's pin.
        verify_model_file(model_file(heavy_models_dir, name), known_model(name))
    manager.deleteLater()


def test_completed_install_readiness_does_not_wait_for_unrelated_install(
    qapp, tmp_path, monkeypatch, process_until
):
    first, second = MANAGED_ARTIFACT_NAMES[:2]
    entered, release = threading.Event(), threading.Event()

    def install(folder, name, **kwargs):
        if name == second:
            entered.set()
            assert release.wait(5)
        model_file(folder, name).write_bytes(b"model")

    monkeypatch.setattr(model_manager, "download_model", install)
    manager = ModelManager(tmp_path)
    _settle(manager, process_until)
    try:
        manager.install((second,), force=False)
        assert entered.wait(3)
        manager.install((first,), force=False)
        process_until(lambda: manager.readiness_known and first in manager.ready_names,
                      timeout_s=3, what="The independent completed installation")
        assert manager.in_progress_for((second,))
        assert manager.missing((first,)) == ()
        with pytest.raises(RuntimeError):
            manager.set_models_dir(tmp_path / "other")
    finally:
        release.set()
        process_until(lambda: manager.cleanup_for_quit() and manager.readiness_known,
                      timeout_s=5, what="The remaining installation")
        manager.deleteLater()

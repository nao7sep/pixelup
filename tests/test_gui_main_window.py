from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

from pixelup import gui
from pixelup.app_config import AppConfig
from pixelup.gui import MainWindow
from pixelup.runner import JobRunner
from pixelup.session_log import configure_session_logging

# The MainWindow orchestration (open/dedup, enqueue -> rows + summaries, the
# job-finished status mapping, remove-guard, retry, cancel, and action-button
# enabling) is the app's load-bearing logic and was previously untested. These
# tests drive it with the offscreen QApplication, with JobRunner.schedule stubbed
# so enqueueing never starts a real worker thread or touches inference. Job
# completion is simulated by calling the real _job_finished handler directly.


def _png(tmp_path: Path, name: str, size: tuple[int, int] = (8, 6)) -> Path:
    path = tmp_path / name
    Image.new("RGB", size, "white").save(path)
    return path


@pytest.fixture
def make_window(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    # No real scheduling (no threads / inference), and a clean in-memory config
    # rather than the developer's real ~/.pixelup/config.json.
    monkeypatch.setattr(JobRunner, "schedule", lambda self, max_concurrent_jobs: None)
    monkeypatch.setattr("pixelup.gui.load_app_config", lambda: AppConfig())
    log_file = tmp_path / "logs" / "session.log"
    configure_session_logging(log_file)

    created: list[MainWindow] = []

    def _make() -> MainWindow:
        window = MainWindow(log_file=log_file)
        created.append(window)
        return window

    yield _make

    for window in created:
        # Close via the session-shutdown path so closeEvent does not raise the
        # quit-confirmation QMessageBox, whose exec() would block headlessly.
        window._session_shutdown = True
        window.close()
        window.deleteLater()
    qapp.processEvents()


def _summary(window: MainWindow, path: Path) -> str:
    row = window._image_rows[path.resolve()]
    return window.image_table.item(row, 2).text()


def test_build_app_wires_application_and_opens_argv_paths(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # build_app is everything main() does except the blocking app.exec(); driving it headlessly
    # is the whole point of extracting it (main() is then a 3-line untestable shell). log_file and
    # runtime_dirs are injected at a temp location so nothing touches the real ~/.pixelup.
    monkeypatch.setattr("pixelup.gui.load_app_config", lambda: AppConfig())
    image = _png(tmp_path, "a.png")
    log_file = tmp_path / "logs" / "session.log"
    runtime_dirs = SimpleNamespace(models_dir=tmp_path / "models", temp_dir=tmp_path / "temp")

    app, window = gui.build_app(
        ["pixelup", str(image)],
        log_file=log_file,
        runtime_dirs=runtime_dirs,
    )
    try:
        assert app is qapp  # reuses the running QApplication rather than constructing a second
        assert app.applicationName() == "PixelUp"
        assert window.windowTitle() == "PixelUp"
        # The image path on the command line was opened into the window.
        assert window.image_table.rowCount() == 1
        assert window._selected_path() == image.resolve()
    finally:
        window._session_shutdown = True
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_open_paths_adds_unique_rows_and_focuses_existing(
    make_window, tmp_path: Path
) -> None:
    window = make_window()
    first = _png(tmp_path, "a.png")
    second = _png(tmp_path, "b.png")

    window.open_paths([first, second])
    assert window.image_table.rowCount() == 2
    assert window._selected_path() == second.resolve()

    # Re-opening an existing path does not duplicate it; it focuses the row.
    window.open_paths([first])
    assert window.image_table.rowCount() == 2
    assert window._selected_path() == first.resolve()


def test_open_paths_ignores_non_files(make_window, tmp_path: Path) -> None:
    window = make_window()
    real = _png(tmp_path, "a.png")

    window.open_paths([tmp_path / "missing.png", real, tmp_path])

    assert window.image_table.rowCount() == 1
    assert window._selected_path() == real.resolve()


def test_queue_selected_image_creates_rows_and_summary(
    make_window, tmp_path: Path
) -> None:
    window = make_window()
    image = _png(tmp_path, "a.png")
    window.open_paths([image])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    window.model_checks["RealESRGAN_x4plus"].setChecked(True)

    window._queue_selected_image()

    assert len(window.jobs) == 2
    assert window.queue_table.rowCount() == 2
    assert _summary(window, image) == "2 queued"
    assert window.cancel_button.isEnabled() is True


def test_job_finished_maps_outcomes_and_updates_summary(
    make_window, tmp_path: Path
) -> None:
    window = make_window()
    image = _png(tmp_path, "a.png")
    window.open_paths([image])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    window._queue_selected_image()
    job = window.jobs[0]

    window._job_finished(job.id, True, "Done", {"ok": True}, [])
    assert job.status == "succeeded"
    assert _summary(window, image) == "1 done"

    window._job_finished(job.id, False, "boom", {}, ["disk almost full"])
    assert job.status == "failed"
    assert job.warnings == ["disk almost full"]
    assert window.retry_button.isEnabled() is True
    assert _summary(window, image) == "1 failed"

    window._job_finished(job.id, False, "Cancelled", {"cancelled": True}, [])
    assert job.status == "cancelled"
    assert _summary(window, image) == "1 cancelled"


def test_remove_is_blocked_while_jobs_active_then_allowed(
    make_window, tmp_path: Path
) -> None:
    window = make_window()
    image = _png(tmp_path, "a.png")
    window.open_paths([image])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    window._queue_selected_image()

    # A pending job still uses this image, so Remove is disabled.
    assert window.remove_image_button.isEnabled() is False

    window._job_finished(window.jobs[0].id, True, "Done", {"ok": True}, [])
    assert window.remove_image_button.isEnabled() is True

    window._remove_selected_image()
    assert window.image_table.rowCount() == 0
    assert window._selected_path() is None


def test_retry_failed_requeues_jobs(make_window, tmp_path: Path) -> None:
    window = make_window()
    image = _png(tmp_path, "a.png")
    window.open_paths([image])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    window._queue_selected_image()
    job = window.jobs[0]
    window._job_finished(job.id, False, "boom", {}, [])
    assert window.retry_button.isEnabled() is True

    window._retry_failed()

    assert job.status == "pending"
    assert job.message == ""
    assert window.retry_button.isEnabled() is False
    assert _summary(window, image) == "1 queued"


def test_cancel_queue_marks_pending_and_signals_running(
    make_window, tmp_path: Path
) -> None:
    window = make_window()
    image = _png(tmp_path, "a.png")
    window.open_paths([image])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    window.model_checks["RealESRGAN_x4plus"].setChecked(True)
    window._queue_selected_image()
    window.jobs[0].status = "running"
    window.jobs[1].status = "pending"

    window._cancel_queue()

    assert window.jobs[0].status == "cancelling"
    assert window.jobs[1].status == "cancelled"


def test_action_buttons_reflect_state(make_window, tmp_path: Path) -> None:
    window = make_window()
    # No images: every queue action is disabled.
    assert window.queue_selected_button.isEnabled() is False
    assert window.queue_selected_all_models_button.isEnabled() is False
    assert window.queue_all_selected_models_button.isEnabled() is False
    assert window.queue_all_all_models_button.isEnabled() is False

    window.open_paths([_png(tmp_path, "a.png")])
    # Image present, no model checked: only the all-models actions are usable.
    assert window.queue_selected_button.isEnabled() is False
    assert window.queue_all_selected_models_button.isEnabled() is False
    assert window.queue_selected_all_models_button.isEnabled() is True
    assert window.queue_all_all_models_button.isEnabled() is True

    window.model_checks["realesr-general-x4v3"].setChecked(True)
    assert window.queue_selected_button.isEnabled() is True
    assert window.queue_all_selected_models_button.isEnabled() is True


def test_reveal_log_file_survives_failure(
    make_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = make_window()

    # A False result (reveal could not be confirmed) must not raise.
    monkeypatch.setattr("pixelup.gui._reveal_in_file_browser", lambda path: False)
    window._reveal_log_file()

    # An OSError (e.g. the helper binary is missing) is caught and logged.
    def _raise(path: Path) -> bool:
        raise OSError("no file browser")

    monkeypatch.setattr("pixelup.gui._reveal_in_file_browser", _raise)
    window._reveal_log_file()


def test_reveal_in_file_browser_darwin_reports_returncode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = _png(tmp_path, "a.png")
    monkeypatch.setattr(sys, "platform", "darwin")

    monkeypatch.setattr(gui.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    assert gui._reveal_in_file_browser(target) is True

    monkeypatch.setattr(gui.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    assert gui._reveal_in_file_browser(target) is False


def test_reveal_in_file_browser_windows_assumes_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = _png(tmp_path, "a.png")
    monkeypatch.setattr(sys, "platform", "win32")
    ran: list[object] = []

    def _run(*args: object, **kwargs: object) -> SimpleNamespace:
        ran.append(args)
        return SimpleNamespace(returncode=1)  # explorer returns nonzero even on success

    monkeypatch.setattr(gui.subprocess, "run", _run)
    assert gui._reveal_in_file_browser(target) is True
    assert ran

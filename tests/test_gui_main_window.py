from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, PngImagePlugin
from PySide6.QtCore import QEvent, QMargins, QRect, QSettings, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QCloseEvent, QColor, QImageReader, QKeySequence, QPalette
from PySide6.QtWidgets import QApplication, QCheckBox, QDialog, QPushButton, QStyle

from pixelup import gui, theme
from pixelup.app_config import (
    AppConfig,
    ConfigLoadResult,
    ConfigSaveResult,
    SettingsFile,
    config_path,
    load_app_config,
)
from pixelup.errors import ErrorCode, PixelupError
from pixelup.gui import MainWindow
from pixelup.i18n.localizer import english
from pixelup.i18n.message import Message
from pixelup.jobs import Job, JobSettings
from pixelup.model_registry import ALL_MODELS
from pixelup.models import model_file
from pixelup.parameters import DEFAULT_SCALE, TILE_VALUES
from pixelup.paths import absolute_user_path
from pixelup.runner import JobRunner
from pixelup.session_log import configure_session_logging, flush_records
from pixelup.window_placement import restored_window_state

# The MainWindow orchestration (open/dedup, enqueue -> rows + summaries, the
# job-finished status mapping, remove-guard, retry, cancel, and action-button
# enabling) is the app's load-bearing logic and was previously untested. These
# tests drive it with the offscreen QApplication, with JobRunner.schedule stubbed
# so enqueueing never starts a real worker thread or touches inference. Job
# completion is simulated by calling the real _job_finished handler directly.


DONE = Message("queue.statusDone")
CANCELLED = Message("queue.statusCancelled")
FAILED = Message("jobs.failedFallback")
DISK_WARNING = Message.of("warning.extensionMismatch", extension="png", format="jpg")



def _settle_models(window) -> None:
    planning_reads = getattr(window, "_planning_reads", None)
    deadline = time.monotonic() + 10
    while (
        not window.model_manager.readiness_known or window.model_manager._scan_in_flight
        or getattr(window, "_active_job_plan", None) is not None
        or getattr(window, "_pending_job_plans", ())
        or (planning_reads is not None and not planning_reads.settled)
    ):
        assert time.monotonic() < deadline, "model readiness and planning did not settle"
        QApplication.processEvents()
        time.sleep(0.001)


def _settle_opens(window) -> None:
    import time as _time

    from PySide6.QtWidgets import QApplication as _QApplication

    deadline = _time.monotonic() + 10
    while window.opening or window.preview.loading:
        assert _time.monotonic() < deadline, "opening images did not finish"
        _QApplication.processEvents()
        _time.sleep(0.005)

def _open(window, paths) -> None:
    """Offer ``paths`` and wait until the window has checked and applied them, and
    any preview they selected has loaded."""
    import time as _time

    from PySide6.QtWidgets import QApplication as _QApplication

    window.open_paths(paths)
    deadline = _time.monotonic() + 10
    while window.opening or window.preview.loading:
        assert _time.monotonic() < deadline, "opening images did not finish"
        _QApplication.processEvents()
        _time.sleep(0.005)

def _png(tmp_path: Path, name: str, size: tuple[int, int] = (8, 6)) -> Path:
    path = tmp_path / name
    Image.new("RGB", size, "white").save(path)
    return path


def _settle_config(window: MainWindow) -> None:
    deadline = time.monotonic() + 5
    while window._config_save is not None:
        assert time.monotonic() < deadline, "settings save did not settle"
        QApplication.instance().processEvents()
        threading.Event().wait(0.001)
    QApplication.instance().processEvents()


def _flush(window: MainWindow) -> bool:
    result: list[bool] = []
    window._flush_parameters_save(finished=result.append)
    _settle_config(window)
    assert len(result) == 1
    return result[0]


class _UnavailableSettingsRoot:
    """A deterministic unavailable volume, reversible before fixture teardown."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.displaced = root.with_name(root.name + "-displaced")

    def replace(self) -> None:
        self.root.rename(self.displaced)
        self.root.write_text("no longer a directory", encoding="utf-8")

    def restore(self) -> None:
        if self.displaced.exists():
            self.root.unlink()
            self.displaced.rename(self.root)


def _isolate_settings_root(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> _UnavailableSettingsRoot:
    # Records and window geometry keep their already-open owners on the startup
    # root. Disconnect a separate accepted settings volume with no live handles,
    # so this failure is reproducible on Windows as well as POSIX.
    monkeypatch.setenv("PIXELUP_DATA_DIR", str(tmp_path / "settings-volume"))
    accepted_path = config_path()
    window._settings_file = replace(window._settings_file, path=accepted_path)
    return _UnavailableSettingsRoot(accepted_path.parent)


def _track_runtime_config_resolution(monkeypatch: pytest.MonkeyPatch) -> list[None]:
    calls: list[None] = []
    resolve = gui.config_path

    def resolving() -> Path:
        calls.append(None)
        return resolve()

    monkeypatch.setattr(gui, "config_path", resolving)
    return calls


@pytest.fixture
def make_window(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    process_until,
):
    # No real job scheduling or inference, and a clean in-memory config
    # rather than the developer's real ~/.pixelup/config.json. PIXELUP_DATA_DIR is
    # redirected as well as the load stubbed, because the Parameters panel's save
    # resolves config.json through the storage root and must not land in the real one.
    monkeypatch.setenv("PIXELUP_DATA_DIR", str(tmp_path / "home"))
    monkeypatch.setattr(JobRunner, "schedule", lambda self, max_concurrent_jobs: None)
    monkeypatch.setattr(
        "pixelup.gui.load_app_config_result",
        lambda: ConfigLoadResult(AppConfig(), SettingsFile(config_path())),
    )
    log_file = configure_session_logging()

    created: list[MainWindow] = []

    def _make(*, injected: bool = True) -> MainWindow:
        models_dir = tmp_path / "home" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        for info in ALL_MODELS:
            (models_dir / info.filename).write_bytes(b"ready")
        runtime_dirs = SimpleNamespace(
            models_dir=models_dir,
            temp_dir=tmp_path / "home" / "temp",
        )
        window = MainWindow(log_file=log_file, runtime_dirs=runtime_dirs if injected else None)
        created.append(window)
        _settle_models(window)
        return window

    yield _make

    for window in created:
        # Close via the session-shutdown path so closeEvent does not raise the
        # quit-confirmation QMessageBox, whose exec() would block headlessly.
        window._is_session_shutdown = lambda: True
        window.close()
        process_until(
            window._workers_clean_for_quit, timeout_s=5, what="Window background settlement",
        )
        window.deleteLater()
    qapp.processEvents()


def _summary(window: MainWindow, path: Path) -> str:
    row = window._image_rows[absolute_user_path(path)]
    return window.image_table.item(row, 2).text()


def test_open_picker_failure_is_authored_and_retained_by_images_group(
    make_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    hostile = "EACCES IPC /private/tmp/PIXELUP-PICKER-SENTINEL"
    window = make_window()
    monkeypatch.setattr(
        gui.QFileDialog,
        "getOpenFileNames",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(hostile)),
    )

    window._open_dialog()

    assert not window.open_result.isHidden()
    assert "image picker" in window.open_result.message_label.text()
    assert hostile not in window.open_result.message_label.text()


class _DropEvent:
    def __init__(self, urls: list[QUrl]) -> None:
        self._mime_data = SimpleNamespace(urls=lambda: urls)
        self.accepted = False
        self.ignored = False

    def mimeData(self):
        return self._mime_data

    def acceptProposedAction(self) -> None:
        self.accepted = True

    def ignore(self) -> None:
        self.ignored = True


class _LeaveEvent:
    def __init__(self) -> None:
        self.accepted = False

    def accept(self) -> None:
        self.accepted = True


def test_build_app_wires_application_and_opens_argv_paths(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # build_app is everything main() does except the blocking app.exec(); driving it headlessly
    # is the whole point of extracting it (main() is then a 3-line untestable shell). runtime_dirs
    # is injected at a temp location so nothing touches the real ~/.pixelup.
    monkeypatch.setenv("PIXELUP_DATA_DIR", str(tmp_path / "home"))
    monkeypatch.setattr(
        "pixelup.gui.load_app_config_result",
        lambda: ConfigLoadResult(AppConfig(), SettingsFile(config_path())),
    )
    image = _png(tmp_path, "a.png")
    runtime_dirs = SimpleNamespace(models_dir=tmp_path / "models", temp_dir=tmp_path / "temp")

    app, window = gui.build_app(["pixelup", str(image)], runtime_dirs=runtime_dirs)
    try:
        assert app is qapp  # reuses the running QApplication rather than constructing a second
        assert app.applicationName() == "PixelUp"
        assert app.style().styleHint(
            QStyle.StyleHint.SH_ComboBox_ListMouseTracking_Current
        ) == 0
        assert app.style().styleHint(
            QStyle.StyleHint.SH_ComboBox_ListMouseTracking_Active
        ) == 1
        assert window.windowTitle() == "PixelUp"
        assert window.runtime_dirs is runtime_dirs
        # The image path on the command line was opened into the window.
        _settle_opens(window)
        assert window.image_table.rowCount() == 1
        assert window._selected_path() == image.resolve()
    finally:
        window._is_session_shutdown = lambda: True
        window.close()
        _settle_config(window)
        window.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize(
    ("platform", "expected_icon"),
    [("darwin", None), ("win32", "icon-win.png")],
)
def test_build_app_assigns_runtime_icon_only_on_windows(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    expected_icon: str | None,
) -> None:
    monkeypatch.setenv("PIXELUP_DATA_DIR", str(tmp_path / "home"))
    monkeypatch.setattr(
        "pixelup.gui.load_app_config_result",
        lambda: ConfigLoadResult(AppConfig(), SettingsFile(config_path())),
    )
    monkeypatch.setattr(gui.sys, "platform", platform)
    assigned_icons: list[str] = []
    monkeypatch.setattr(gui, "QIcon", lambda path: path)
    monkeypatch.setattr(qapp, "setWindowIcon", assigned_icons.append)
    runtime_dirs = SimpleNamespace(models_dir=tmp_path / "models", temp_dir=tmp_path / "temp")

    _, window = gui.build_app(["pixelup"], runtime_dirs=runtime_dirs)
    try:
        if expected_icon is None:
            assert assigned_icons == []
        else:
            assert len(assigned_icons) == 1
            assert Path(assigned_icons[0]).name == expected_icon
    finally:
        window._is_session_shutdown = lambda: True
        window.close()
        _settle_config(window)
        window.deleteLater()
        qapp.processEvents()


def test_open_paths_adds_unique_rows_and_focuses_existing(make_window, tmp_path: Path) -> None:
    window = make_window()
    assert window.image_table.empty_state_visible is True
    assert window.image_table.empty_text == "No images yet. Open images or drop them here."
    first = _png(tmp_path, "a.png")
    second = _png(tmp_path, "b.png")

    _open(window, [first, second])
    assert window.image_table.rowCount() == 2
    assert window.image_table.empty_state_visible is False
    assert window._selected_path() == second.resolve()

    # Re-opening an existing path does not duplicate it; it focuses the row.
    _open(window, [first])
    assert window.image_table.rowCount() == 2
    assert window._selected_path() == first.resolve()


def test_open_paths_preserves_literal_symlink_for_display_and_default_output(
    make_window, tmp_path: Path, file_symlink_capability: None
) -> None:
    window = make_window()
    source_dir = tmp_path / "source"
    chosen_dir = tmp_path / "chosen"
    source_dir.mkdir()
    chosen_dir.mkdir()
    target = _png(source_dir, "target.png")
    chosen = chosen_dir / "dropped-link.png"
    chosen.symlink_to(target)

    _open(window, [chosen])

    assert window._selected_path() == chosen
    assert window.image_table.item(0, 0).text() == "dropped-link.png"
    assert window.image_table.item(0, 0).toolTip() == str(chosen)

    window.model_checks["realesr-general-x4v3"].setChecked(True)
    window._queue_selected_image()

    _settle_models(window)
    assert window.jobs[0].input_path == chosen
    assert window.jobs[0].output_path.parent == chosen_dir


def test_external_drop_accepts_local_files_and_rejects_remote_urls(
    make_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = make_window()
    assert window.image_table.acceptDrops() is True
    local = _png(tmp_path, "local.png")
    remote_url = QUrl("https://example.com/remote.png")
    directory_url = QUrl.fromLocalFile(str(tmp_path))
    missing = tmp_path / "missing.png"
    non_image = tmp_path / "not-an-image.txt"
    non_image.write_text("not image data", encoding="utf-8")

    assert "dragEnterEvent" not in MainWindow.__dict__
    assert "dropEvent" not in MainWindow.__dict__

    remote_drag = _DropEvent([remote_url])
    window.image_table.dragEnterEvent(remote_drag)  # type: ignore[arg-type]
    assert remote_drag.ignored is True
    assert remote_drag.accepted is False

    remote_drop = _DropEvent([remote_url])
    window.image_table.dropEvent(remote_drop)  # type: ignore[arg-type]
    assert remote_drop.ignored is True
    assert window.image_table.rowCount() == 0

    local_drop = _DropEvent(
        [
            remote_url,
            directory_url,
            QUrl.fromLocalFile(str(missing)),
            QUrl.fromLocalFile(str(non_image)),
            QUrl.fromLocalFile(str(local)),
        ]
    )
    original_is_file = Path.is_file

    def fail_is_file(_path: Path) -> bool:
        raise AssertionError("hover touched the filesystem")

    monkeypatch.setattr(Path, "is_file", fail_is_file)
    window.image_table.dragEnterEvent(local_drop)  # type: ignore[arg-type]
    monkeypatch.setattr(Path, "is_file", original_is_file)
    assert window.image_table.property("dropActive") is True
    window.image_table.dropEvent(local_drop)  # type: ignore[arg-type]
    assert local_drop.accepted is True
    _settle_opens(window)
    assert window._selected_path() == local
    assert window.image_table.rowCount() == 1
    assert window.image_table.property("dropActive") is False
    assert window.open_result.isVisibleTo(window)
    message = window.open_result.message_label.text()
    assert f"{tmp_path.name}: folders are not supported" in message
    assert "missing.png: the file is unavailable" in message
    assert "not-an-image.txt: not a readable supported image" in message
    assert window.open_result.accessibleName() == message


def test_external_drag_leave_clears_receiver_presentation(make_window, tmp_path: Path) -> None:
    window = make_window()
    drag = _DropEvent([QUrl.fromLocalFile(str(_png(tmp_path, "leave.png")))])

    window.image_table.dragEnterEvent(drag)  # type: ignore[arg-type]
    assert window.image_table.property("dropActive") is True

    leave = _LeaveEvent()
    window.image_table.dragLeaveEvent(leave)  # type: ignore[arg-type]

    assert leave.accepted is True
    assert window.image_table.property("dropActive") is False


def test_open_duplicate_is_information_and_unrelated_success_does_not_clear_it(
    make_window, tmp_path: Path
) -> None:
    window = make_window()
    first = _png(tmp_path, "first.png")
    second = _png(tmp_path, "second.png")

    _open(window, [first])
    assert window.open_result.isHidden()

    _open(window, [first])
    assert window.open_result.message_label.text() == "Already open: first.png."
    assert window.open_result.isVisibleTo(window)

    _open(window, [second])
    assert window.open_result.message_label.text() == "Already open: first.png."

    window.open_result.dismiss_button.click()
    assert window.open_result.isHidden()


def test_exact_corrected_open_clears_the_previous_rejection(make_window, tmp_path: Path) -> None:
    window = make_window()
    broken = tmp_path / "repair.png"
    broken.write_text("not an image", encoding="utf-8")

    _open(window, [broken])
    assert "not a readable supported image" in window.open_result.message_label.text()

    _png(tmp_path, "repair.png")
    another = _png(tmp_path, "another.png")
    _open(window, [broken, another])

    assert window.image_table.rowCount() == 2
    assert window.open_result.isHidden()


def test_unexpected_open_failure_uses_error_presentation(
    make_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = make_window()
    image = _png(tmp_path, "failure.png")

    diagnostic_sentinel = "EACCES /private/var/tmp/pixelup-internal"

    def fail_path(_path: Path) -> Path:
        raise OSError(diagnostic_sentinel)

    monkeypatch.setattr("pixelup.gui.absolute_user_path", fail_path)
    _open(window, [image])

    assert "could not be read" in window.open_result.message_label.text()
    assert diagnostic_sentinel not in window.open_result.message_label.text()
    assert window.open_result.accessibleName() == window.open_result.message_label.text()


def test_queue_preconditions_stay_with_the_queue_actions_and_clear_on_correction(
    make_window, tmp_path: Path
) -> None:
    window = make_window()

    window._enqueue_jobs([], ["realesr-general-x4v3"])

    _settle_models(window)
    assert window.queue_action_result.isVisibleTo(window)
    assert "image" in window.queue_action_result.message_label.text()
    assert (
        window.queue_action_result.layout().indexOf(window.queue_action_result.dismiss_button) == 1
    )

    image = _png(tmp_path, "queue.png")
    _open(window, [image])
    assert window.queue_action_result.isHidden()

    window._enqueue_jobs([image], [])
    _settle_models(window)
    assert window.queue_action_result.isVisibleTo(window)
    assert "model" in window.queue_action_result.message_label.text()

    window.model_checks["realesr-general-x4v3"].setChecked(True)
    assert window.queue_action_result.isHidden()


def test_image_in_use_failure_stays_with_remove_and_clears_when_work_finishes(
    make_window, tmp_path: Path
) -> None:
    window = make_window()
    image = _png(tmp_path, "busy.png")
    _open(window, [image])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    window._queue_selected_image()

    _settle_models(window)
    window._remove_selected_image()

    assert window.image_table.rowCount() == 1
    assert window.remove_result.isVisibleTo(window)
    assert "cannot be removed" in window.remove_result.message_label.text()

    job = window.jobs[0]
    window._job_finished(job.id, True, DONE, {"ok": True}, [])

    assert window.remove_result.isHidden()


def test_shortcuts_help_chords_open_the_catalogue(
    make_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[object] = []
    monkeypatch.setattr(
        "pixelup.gui.ShortcutsDialog.exec", lambda dialog: opened.append(dialog.parent())
    )
    window = make_window()

    assert window.shortcuts_shortcut.key() == QKeySequence("Ctrl+/")
    assert window.shortcuts_question_shortcut.key() == QKeySequence("Ctrl+?")

    window.shortcuts_shortcut.activated.emit()
    window.shortcuts_question_shortcut.activated.emit()

    assert opened == [window, window]


def test_window_utilities_sit_below_queue_actions(make_window, qapp: QApplication) -> None:
    window = make_window()
    window.show()
    qapp.processEvents()

    utilities = window.records_button.parentWidget()
    queue_group = window.queue_selected_button.parentWidget()

    assert utilities.parentWidget() is queue_group.parentWidget()
    assert utilities.geometry().top() >= queue_group.geometry().bottom()


def test_open_paths_ignores_non_files(make_window, tmp_path: Path) -> None:
    window = make_window()
    real = _png(tmp_path, "a.png")

    _open(window, [tmp_path / "missing.png", real, tmp_path])

    assert window.image_table.rowCount() == 1
    assert window._selected_path() == real.resolve()


def test_queue_selected_image_creates_rows_and_summary(make_window, tmp_path: Path) -> None:
    window = make_window()
    assert window.queue_table.empty_state_visible is True
    assert window.queue_table.empty_text == "No jobs queued yet."
    image = _png(tmp_path, "a.png")
    _open(window, [image])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    window.model_checks["RealESRGAN_x4plus"].setChecked(True)

    window._queue_selected_image()

    _settle_models(window)
    assert len(window.jobs) == 2
    assert window.queue_table.rowCount() == 2
    assert window.queue_table.empty_state_visible is False
    assert _summary(window, image) == "2 queued"
    assert window.cancel_button.isEnabled() is True


def test_job_finished_maps_outcomes_and_updates_summary(make_window, tmp_path: Path) -> None:
    window = make_window()
    image = _png(tmp_path, "a.png")
    _open(window, [image])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    window._queue_selected_image()
    _settle_models(window)
    job = window.jobs[0]

    window._job_finished(job.id, True, DONE, {"ok": True}, [])
    assert job.status == "succeeded"
    assert _summary(window, image) == "1 done"

    window._job_finished(job.id, False, FAILED, {}, [DISK_WARNING])
    assert job.status == "failed"
    assert job.warnings == [DISK_WARNING]
    assert window.retry_button.isEnabled() is True
    assert _summary(window, image) == "1 failed"

    window._job_finished(job.id, False, CANCELLED, {"cancelled": True}, [])
    assert job.status == "cancelled"
    assert _summary(window, image) == "1 cancelled"


def test_job_finished_recomputes_only_its_own_images_summary(
    make_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # PU-5: a single job's completion must touch only its own image's row, not
    # rescan every job and rewrite every image row in the batch.
    window = make_window()
    image_a = _png(tmp_path, "a.png")
    image_b = _png(tmp_path, "b.png")
    _open(window, [image_a, image_b])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    window._queue_all_images_selected_models()
    _settle_models(window)
    assert len(window.jobs) == 2
    job_a = next(job for job in window.jobs if job.input_path == image_a)

    calls: list[list[str]] = []
    original = gui.job_status_summary

    def _counting_summary(statuses: list[str]) -> str:
        calls.append(list(statuses))
        return original(statuses)

    monkeypatch.setattr("pixelup.gui.job_status_summary", _counting_summary)

    window._job_finished(job_a.id, True, DONE, {"ok": True}, [])

    # Exactly one image's summary was recomputed — image_a's — not both.
    assert calls == [["succeeded"]]
    assert _summary(window, image_a) == "1 done"
    assert _summary(window, image_b) == "1 queued"


def test_find_job_uses_the_id_index_and_rejects_an_unknown_id(make_window, tmp_path: Path) -> None:
    window = make_window()
    image = _png(tmp_path, "a.png")
    _open(window, [image])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    window._queue_selected_image()
    _settle_models(window)
    job = window.jobs[0]

    assert window._find_job(job.id) is job
    with pytest.raises(RuntimeError, match="Unknown job id"):
        window._find_job(job.id + 999)


def test_failed_jobs_keep_a_queue_local_accessible_summary_until_retry(
    make_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    announced: list[object] = []
    monkeypatch.setattr("pixelup.gui.announce_accessible_alert", announced.append)
    window = make_window()
    image = _png(tmp_path, "a.png")
    _open(window, [image])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    window.model_checks["RealESRGAN_x4plus"].setChecked(True)
    window._queue_selected_image()

    _settle_models(window)
    first, second = window.jobs
    window._job_finished(first.id, False, FAILED, {}, [])

    assert window.queue_failure_result.isVisibleTo(window) is True
    assert window.queue_failure_label.text() == (
        "1 queue job has failed. Review the failed row or retry it."
    )
    assert window.queue_failure_result.accessibleName() == window.queue_failure_label.text()
    assert announced == [window.queue_failure_result]

    # An unrelated success does not erase the unresolved failure.
    window._job_finished(second.id, True, DONE, {"ok": True}, [])
    assert window.queue_failure_result.isVisibleTo(window) is True
    assert "1 queue job has failed" in window.queue_failure_label.text()
    assert announced == [window.queue_failure_result]

    window._retry_failed()
    _settle_models(window)
    assert window.queue_failure_result.isHidden() is True
    assert window.queue_failure_result.accessibleName() == ""


def test_font_metric_refresh_remeasures_tables_and_window_floor(
    make_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = make_window()
    widths = {
        "Size": 111,
        "Jobs": 122,
        "Model": 133,
        "Scale": 144,
        "Status": 155,
    }
    monkeypatch.setattr(
        "pixelup.gui._fit_columns",
        lambda _widget, header, *_samples: widths[header],
    )

    hint = window._refresh_layout_metrics()

    assert window.image_table.horizontalHeader().sectionSize(1) == 111
    assert window.image_table.horizontalHeader().sectionSize(2) == 122
    assert window.image_table.minimumWidth() == 180 + 111 + 122
    assert window.queue_table.horizontalHeader().sectionSize(1) == 133
    assert window.queue_table.horizontalHeader().sectionSize(2) == 144
    assert window.queue_table.horizontalHeader().sectionSize(4) == 155
    assert window.queue_table.minimumWidth() == 180 + 133 + 144 + 180 + 155
    assert window._content_root.minimumSize() == hint
    assert window.minimumWidth() <= hint.width()
    assert window.minimumHeight() <= hint.height()


def test_small_screen_caps_native_floor_without_clipping_content(make_window, monkeypatch) -> None:
    window = make_window()
    monkeypatch.setattr(
        MainWindow,
        "screen",
        lambda _self: SimpleNamespace(
            availableGeometry=lambda: QRect(50, 70, 600, 400),
        ),
    )
    window._update_native_minimum()
    assert window.minimumWidth() <= 600
    assert window.minimumHeight() <= 400
    assert window._content_root.minimumWidth() > window.minimumWidth()
    window.resize(600, 400)
    window.show()
    QApplication.processEvents()
    viewport = window.centralWidget()
    assert viewport.widget() is window._content_root
    assert viewport.horizontalScrollBar().maximum() > 0


def test_first_normal_target_leaves_room_for_native_frame(make_window, monkeypatch) -> None:
    monkeypatch.setattr(
        MainWindow,
        "screen",
        lambda _self: SimpleNamespace(
            availableGeometry=lambda: QRect(50, 70, 600, 400),
        ),
    )
    monkeypatch.setattr(
        MainWindow,
        "windowHandle",
        lambda _self: SimpleNamespace(
            frameMargins=lambda: QMargins(8, 30, 8, 8),
        ),
    )
    window = make_window()
    assert window.size() == QSize(584, 362)
    assert window.minimumSize() == QSize(584, 362)
    assert window._content_root.minimumWidth() > window.width()


def test_saved_font_change_refreshes_live_layout_metrics(
    make_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = make_window()
    candidate = replace(window.config, font_family="Menlo")

    class AcceptedSettings:
        def __init__(self, _config, _parent, *, try_save, session_shutdown) -> None:
            self._try_save = try_save

        def exec(self):
            saved = []
            self._try_save(candidate, saved.append, lambda: None)
            _settle_config(window)
            assert saved == [None]
            return QDialog.DialogCode.Accepted

        def config(self):
            return candidate

    applied: list[str] = []
    refreshed: list[bool] = []
    monkeypatch.setattr("pixelup.gui.SettingsDialog", AcceptedSettings)
    monkeypatch.setattr("pixelup.gui.apply_ui_font", lambda _app, family: applied.append(family))
    monkeypatch.setattr(window, "_refresh_layout_metrics", lambda: refreshed.append(True))

    window._settings_dialog()

    assert window.config.font_family == "Menlo"
    assert applied == ["Menlo"]
    assert refreshed == [True]


def test_remove_is_blocked_while_jobs_active_then_allowed(make_window, tmp_path: Path) -> None:
    window = make_window()
    image = _png(tmp_path, "a.png")
    _open(window, [image])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    window._queue_selected_image()

    _settle_models(window)
    # A pending job still uses this image, so Remove is disabled.
    assert window.remove_image_button.isEnabled() is False

    window._job_finished(window.jobs[0].id, True, DONE, {"ok": True}, [])
    assert window.remove_image_button.isEnabled() is True

    window._remove_selected_image()
    assert window.image_table.rowCount() == 0
    assert window.image_table.empty_state_visible is True
    assert window._selected_path() is None


def test_retry_failed_requeues_jobs(make_window, tmp_path: Path) -> None:
    window = make_window()
    image = _png(tmp_path, "a.png")
    _open(window, [image])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    window._queue_selected_image()
    _settle_models(window)
    job = window.jobs[0]
    window._job_finished(job.id, False, FAILED, {}, [])
    assert window.retry_button.isEnabled() is True

    window._retry_failed()

    _settle_models(window)
    assert job.status == "pending"
    assert job.message is None
    assert window.retry_button.isEnabled() is False
    assert _summary(window, image) == "1 queued"


def test_cancel_queue_marks_pending_and_signals_running(make_window, tmp_path: Path) -> None:
    window = make_window()
    image = _png(tmp_path, "a.png")
    _open(window, [image])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    window.model_checks["RealESRGAN_x4plus"].setChecked(True)
    window._queue_selected_image()
    _settle_models(window)
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

    _open(window, [_png(tmp_path, "a.png")])
    # Image present, no model checked: only the all-models actions are usable.
    assert window.queue_selected_button.isEnabled() is False
    assert window.queue_all_selected_models_button.isEnabled() is False
    assert window.queue_selected_all_models_button.isEnabled() is True
    assert window.queue_all_all_models_button.isEnabled() is True

    window.model_checks["realesr-general-x4v3"].setChecked(True)
    assert window.queue_selected_button.isEnabled() is True
    assert window.queue_all_selected_models_button.isEnabled() is True


def test_settings_only_options_have_no_main_window_control(make_window) -> None:
    # One home per thing, checked from the other side. max_concurrent_jobs is the
    # settings dialog's; model acquisition belongs to Managed models, not to a toggle.
    window = make_window()

    assert set(window.model_checks) == set(gui.UPSCALE_MODELS)
    checkbox_labels = {box.text() for box in window.findChildren(QCheckBox)}
    assert checkbox_labels == set(gui.UPSCALE_MODELS) | {"Strip metadata"}
    # The window reads both from config and never offers a widget onto them.
    assert not hasattr(window, "auto_download")
    assert not hasattr(window, "concurrent")
    assert window.manage_models_button.text() == "Managed models"
    assert window.manage_models_button.accessibleName() == ("Managed models, all models installed")
    assert window.manage_models_button.icon().isNull()


def test_the_main_action_leads_by_weight_not_by_an_accent_fill(make_window) -> None:
    # Queue selected image leads its column by the weight of its label; the three
    # queue buttons under it do the same job at another scope, so no button in the
    # main window takes the accent fill, which is kept for a dialog's commit.
    window = make_window()
    assert window.queue_selected_button.property("role") == "main"
    assert all(button.property("role") != "primary" for button in window.findChildren(QPushButton))


def test_missing_models_show_as_a_warning_line_under_the_button(make_window) -> None:
    """The button always says what it opens; the models' state is a warning line of
    its own under it. Drawn as a filled amber button labelled "Models are missing",
    the state read as a second action beside the real one."""
    window = make_window()
    assert window.model_status.isHidden()

    model_file(window.runtime_dirs.models_dir, "realesr-general-x4v3").unlink()
    window.model_manager.refresh_readiness()
    _settle_models(window)

    assert window.manage_models_button.text() == "Managed models"
    assert window.manage_models_button.styleSheet() == ""
    assert window.manage_models_button.property("role") is None
    assert window.manage_models_button.icon().isNull()
    assert window.manage_models_button.accessibleName() == (
        "Managed models, some models are missing"
    )
    assert "not installed" in window.manage_models_button.toolTip()
    assert not window.model_status.isHidden()
    assert window.model_status.text() == "Some models are not installed."
    assert window.model_status.property("severity") == "warning"


def test_coming_to_the_front_finds_a_model_file_placed_by_hand(
    make_window, tmp_path: Path, process_until
) -> None:
    window = make_window()
    image = _png(tmp_path, "a.png")
    _open(window, [image])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    path = model_file(window.runtime_dirs.models_dir, "realesr-general-x4v3")
    content = path.read_bytes()
    path.unlink()
    window.model_manager.refresh_readiness()
    _settle_models(window)
    assert not window.model_status.isHidden()

    path.write_bytes(content)
    QApplication.sendEvent(window, QEvent(QEvent.Type.WindowActivate))
    process_until(lambda: window.model_status.isHidden(), timeout_s=10, what="The window's rescan")

    assert window.model_manager.missing(("realesr-general-x4v3",)) == ()
    assert window._selected_path() == image.resolve()
    assert window._selected_models() == ["realesr-general-x4v3"]


def test_the_warning_line_has_an_amber_for_each_theme() -> None:
    light = QPalette()
    light.setColor(QPalette.ColorRole.Window, QColor("#ffffff"))
    dark = QPalette()
    dark.setColor(QPalette.ColorRole.Window, QColor("#111111"))

    assert theme.warning_text(light) != theme.warning_text(dark)
    for palette in (light, dark):
        sheet = theme.build_stylesheet(palette)
        assert f'QLabel[severity="warning"] {{\n    color: {theme.warning_text(palette)};' in sheet


def test_missing_models_cancel_before_queue_materialization(
    make_window,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    window = make_window()
    image = _png(tmp_path, "a.png")
    _open(window, [image])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    for name in ("realesr-general-x4v3", "realesr-general-wdn-x4v3"):
        model_file(window.runtime_dirs.models_dir, name).unlink()
    window._queue_selected_image()
    _settle_models(window)
    dialog = window._active_models_dialog
    assert dialog is not None
    assert dialog._required_artifacts == (
        "realesr-general-x4v3",
        "realesr-general-wdn-x4v3",
    )
    assert dialog._pending_job_count == 1
    dialog.reject()
    qapp.processEvents()

    assert not window._pending_model_work
    assert window.jobs == []
    assert window.queue_table.rowCount() == 0


def test_install_and_queue_materializes_jobs_after_requirements_are_ready(
    make_window,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    window = make_window()
    image = _png(tmp_path, "a.png")
    _open(window, [image])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    required = ("realesr-general-x4v3", "realesr-general-wdn-x4v3")
    for name in required:
        model_file(window.runtime_dirs.models_dir, name).unlink()

    window._queue_selected_image()
    _settle_models(window)
    assert window._pending_model_work
    assert window.jobs == []
    for name in required:
        target = model_file(window.runtime_dirs.models_dir, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"ready")
    window.model_manager.refresh_readiness()
    _settle_models(window)
    qapp.processEvents()

    assert not window._pending_model_work
    assert len(window.jobs) == 1
    assert window.queue_table.rowCount() == 1
    assert window.manage_models_button.text() == "Managed models"


def test_install_and_queue_survives_closing_its_presentation(
    make_window,
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = make_window()
    image = _png(tmp_path, "a.png")
    _open(window, [image])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    required = ("realesr-general-x4v3", "realesr-general-wdn-x4v3")
    for name in required:
        model_file(window.runtime_dirs.models_dir, name).unlink()
    window.model_manager.refresh_readiness()
    _settle_models(window)
    started = threading.Event()
    release = threading.Event()

    def install(models_dir: Path, name: str, **_kwargs: object) -> dict[str, object]:
        started.set()
        assert release.wait(2)
        model_file(models_dir, name).write_bytes(b"ready")
        return {"status": "downloaded"}

    monkeypatch.setattr("pixelup.model_manager.download_model", install)
    window._queue_selected_image()
    _settle_models(window)
    dialog = window._active_models_dialog
    assert dialog is not None
    dialog._install_all_or_cancel()
    assert started.wait(1)

    dialog.reject()
    qapp.processEvents()
    assert window._active_models_dialog is None
    assert window._pending_model_work
    assert window.jobs == []

    release.set()
    deadline = time.monotonic() + 3
    while not window.model_manager.cleanup_for_quit() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.001)
    qapp.processEvents()

    assert window.model_manager.cleanup_for_quit()
    _settle_models(window)
    assert not window._pending_model_work
    assert len(window.jobs) == 1


def test_cancelling_preflight_downloads_abandons_the_captured_queue(
    make_window,
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = make_window()
    image = _png(tmp_path, "a.png")
    _open(window, [image])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    required = ("realesr-general-x4v3", "realesr-general-wdn-x4v3")
    for name in required:
        model_file(window.runtime_dirs.models_dir, name).unlink()
    window.model_manager.refresh_readiness()
    _settle_models(window)
    started: set[str] = set()
    started_lock = threading.Lock()
    both_started = threading.Event()

    def wait_for_cancel(_models_dir: Path, name: str, **kwargs: object) -> dict[str, object]:
        with started_lock:
            started.add(name)
            if started == set(required):
                both_started.set()
        should_cancel = kwargs["should_cancel"]
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if should_cancel():  # type: ignore[operator]
                raise PixelupError(ErrorCode.JOB_CANCELLED, "Installation cancelled.")
            time.sleep(0.001)
        raise AssertionError("cancel was not delivered")

    monkeypatch.setattr("pixelup.model_manager.download_model", wait_for_cancel)
    window._queue_selected_image()
    _settle_models(window)
    dialog = window._active_models_dialog
    assert dialog is not None
    dialog._install_all_or_cancel()
    assert both_started.wait(1)

    dialog._install_all_or_cancel()
    deadline = time.monotonic() + 3
    while not window.model_manager.cleanup_for_quit() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.001)
    qapp.processEvents()

    assert window.model_manager.cleanup_for_quit()
    assert not window._pending_model_work
    assert window._active_models_dialog is None
    assert window.jobs == []


def test_retry_stays_failed_when_model_install_is_cancelled(
    make_window,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    window = make_window()
    image = _png(tmp_path, "a.png")
    _open(window, [image])
    window.model_checks["RealESRGAN_x4plus"].setChecked(True)
    window._queue_selected_image()
    _settle_models(window)
    job = window.jobs[0]
    job.status = "failed"
    job.message = Message("error.modelFileMissing")
    window._update_job(job)
    model_file(window.runtime_dirs.models_dir, job.model).unlink()

    window._retry_failed()
    _settle_models(window)
    dialog = window._active_models_dialog
    assert dialog is not None
    dialog.reject()
    qapp.processEvents()

    assert not window._pending_model_work
    assert job.status == "failed"
    assert job.message == Message("error.modelFileMissing")


def _reset_button(window: MainWindow) -> QPushButton:
    return next(
        button for button in window.findChildren(QPushButton) if button.text() == "Reset parameters"
    )


def _wander_from_defaults(window: MainWindow) -> None:
    window.scale_buttons[2].setChecked(True)
    window.denoise_strength.setValue(0.75)
    window.quality.setValue(10)
    window.tile.setCurrentIndex(window.tile.findData(1024))
    window.strip_metadata.setChecked(True)
    window.device.setCurrentIndex(window.device.findData("cpu"))


def test_reset_parameters_button_restores_the_built_ins(make_window) -> None:
    # Drives the real button rather than the handler, so the label, the wiring, and
    # the reset behavior are pinned together. It restores JobSettings() — the
    # built-ins — and not the user's persisted config, which is what a reset exists
    # to escape. The font is an app-appearance setting owned by the settings dialog
    # and must be no business of this reset.
    window = make_window()
    reset = _reset_button(window)

    _wander_from_defaults(window)
    assert window.current_job_settings() != JobSettings()

    reset.click()

    assert window.current_job_settings() == JobSettings()
    assert window.tile.currentData() == JobSettings().tile == 256
    assert window.config.font_family == AppConfig().font_family


def test_reset_parameters_is_off_while_the_panel_holds_the_built_ins(make_window) -> None:
    # A reset that would change nothing is not offered: the button is off at the
    # built-ins, comes on with any edit, and goes off again once it has reset.
    window = make_window()
    reset = _reset_button(window)
    assert window.current_job_settings() == JobSettings()
    assert reset.isEnabled() is False

    _wander_from_defaults(window)
    assert reset.isEnabled() is True

    reset.click()
    assert reset.isEnabled() is False

    window.quality.setValue(window.quality.value() - 1)
    assert reset.isEnabled() is True
    window.quality.setValue(JobSettings().quality)
    assert reset.isEnabled() is False


def test_reset_parameters_button_restores_the_built_in_scale(make_window) -> None:
    # Scale is a panel parameter like any other now, so the real button restores it
    # too — the holdout that used to sit outside both persistence and reset. Driven by
    # the button's label text, and asserted on the radios themselves, so the widget
    # the user actually looks at is what got reset.
    window = make_window()
    window.scale_buttons[2].setChecked(True)
    _flush(window)
    # Persisted first, deliberately: with 2x saved, a reset that reached for the user's
    # config instead of the built-ins would land back on 2x and still look like it had
    # worked. Only a persisted-then-reset run can tell the two apart.
    assert window.config.parameters.scale == 2
    assert window.current_job_settings().scale == 2

    _reset_button(window).click()
    _settle_config(window)

    assert window.current_job_settings().scale == DEFAULT_SCALE
    assert window.scale_buttons[DEFAULT_SCALE].isChecked() is True
    assert window.scale_buttons[2].isChecked() is False
    # And the reset is persisted like any other panel edit, so it survives a relaunch.
    assert window.config.parameters.scale == DEFAULT_SCALE


def test_reset_parameters_ignores_the_persisted_config(make_window) -> None:
    # The pin that the deleted defaults layer cannot come back: even with a persisted
    # panel far from the built-ins, reset goes to the built-ins, not back to what the
    # user had saved.
    window = make_window()
    window.quality.setValue(10)
    window.tile.setCurrentIndex(window.tile.findData(1024))
    _flush(window)
    assert window.config.parameters.quality == 10

    _reset_button(window).click()
    _settle_config(window)

    assert window.current_job_settings() == JobSettings()
    # And the reset is itself persisted, so it survives the next launch.
    assert window.config.parameters == JobSettings()
    assert "parameters" not in json.loads(config_path().read_text(encoding="utf-8"))


def test_parameter_edits_persist_to_config_json(make_window) -> None:
    # The panel is durable: an edit reaches config.json, so a relaunch reads it back.
    # Loaded from disk (not from window.config) so the serializer is in the loop.
    window = make_window()

    _wander_from_defaults(window)
    _flush(window)

    persisted = load_app_config(config_path()).parameters
    assert persisted == window.current_job_settings()
    assert persisted.quality == 10
    assert persisted.tile == 1024
    assert persisted.device == "cpu"
    assert persisted.scale == 2
    assert persisted != JobSettings()


def test_failed_parameter_save_keeps_old_authority_and_retries_the_visible_draft(
    make_window,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = make_window()
    original = window.config
    window.quality.setValue(10)
    attempts: list[AppConfig] = []

    def _save(candidate: AppConfig, _previous: AppConfig, file) -> ConfigSaveResult:
        attempts.append(candidate)
        if len(attempts) == 1:
            raise OSError("disk full")
        return ConfigSaveResult(candidate, file)

    monkeypatch.setattr("pixelup.gui.save_app_config", _save)

    assert _flush(window) is False
    assert window.config is original
    assert window.current_job_settings().quality == 10
    assert window.parameters_result.isVisibleTo(window)
    assert "changes are still shown" in window.parameters_result.message_label.text()
    assert window.parameters_result.layout().indexOf(window.parameters_result.dismiss_button) == 1

    assert _flush(window) is True
    assert window.config.parameters.quality == 10
    assert len(attempts) == 2
    assert window.parameters_result.isHidden()


@pytest.mark.parametrize("structured", [False, True])
def test_delivered_config_failure_keeps_original_diagnostics_and_safe_presentation(
    make_window, monkeypatch: pytest.MonkeyPatch, structured: bool
) -> None:
    window = make_window()
    original = window.config
    sentinel = "EACCES /internal/config ORIGINAL-CONFIG-SENTINEL"

    def failing_save(*_args):
        try:
            raise OSError(sentinel)
        except OSError as cause:
            if structured:
                raise PixelupError(ErrorCode.INTERNAL_ERROR, "wrapped " + sentinel) from cause
            raise RuntimeError("wrapped " + sentinel) from cause

    monkeypatch.setattr(gui, "save_app_config", failing_save)
    window.quality.setValue(10)

    assert _flush(window) is False

    assert window.config is original
    assert window.current_job_settings().quality == 10
    presented = window.parameters_result.message_label.text()
    assert presented == english().of(Message("parameters.saveFailed"))
    assert sentinel not in presented
    assert "RuntimeError" not in presented
    assert "PixelupError" not in presented
    assert flush_records(5)
    with sqlite3.connect(config_path().parent / "records.sqlite3") as connection:
        entries = connection.execute(
            "SELECT level, fields FROM logs WHERE message = 'config.save_failed'"
        ).fetchall()
    assert len(entries) == 1
    level, fields = entries[0]
    assert level == "warn"
    error = json.loads(fields)["error"]
    assert error["type"] == ("PixelupError" if structured else "RuntimeError")
    assert error["message"] == "wrapped " + sentinel
    assert "OSError: " + sentinel in error["traceback"]
    assert "in failing_save" in error["traceback"]
    assert "in _run" in error["traceback"]


@pytest.mark.parametrize("setting", ["parameters", "settings"])
def test_landed_config_completion_uses_its_accepted_path_after_root_becomes_unavailable(
    make_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setting: str
) -> None:
    window = make_window()
    unavailable = _isolate_settings_root(window, tmp_path, monkeypatch)
    accepted_path = window._settings_file.path
    calls = _track_runtime_config_resolution(monkeypatch)
    real_save = gui.save_app_config
    outcomes = []
    # Backup behavior has its own real-boundary tests; keep this save's primary
    # settlement independent from the timing of optional store opening.
    monkeypatch.setattr("pixelup.backup_store.record", lambda *_: None)
    assert flush_records(5)

    def land_then_disconnect(*args):
        result = real_save(*args)
        unavailable.replace()
        return result

    monkeypatch.setattr(gui, "save_app_config", land_then_disconnect)
    try:
        if setting == "parameters":
            window.quality.setValue(73)
            window._flush_parameters_save(finished=outcomes.append)
        else:
            window._save_config_candidate(
                replace(window.config, max_concurrent_jobs=3),
                Message("settings.saveFailed"), outcomes.append,
            )
        _settle_config(window)

        assert unavailable.root.is_file()
        assert outcomes == ([True] if setting == "parameters" else [None])
        assert calls == []
        assert window._settings_file.path == accepted_path
        assert not window._unsaved_config_changes
        landed = load_app_config(unavailable.displaced / accepted_path.name)
        assert window.config == landed
        if setting == "parameters":
            assert window.config.parameters.quality == 73
            assert window.parameters_result.isHidden()
        else:
            assert window.config.max_concurrent_jobs == 3
    finally:
        unavailable.restore()


def test_failed_folder_completion_releases_and_retries_after_root_becomes_unavailable(
    make_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = make_window(injected=False)
    original = window.config
    unavailable = _isolate_settings_root(window, tmp_path, monkeypatch)
    accepted_path = window._settings_file.path
    calls = _track_runtime_config_resolution(monkeypatch)
    target = tmp_path / "chosen models"
    real_save = gui.save_app_config
    attempts = []
    outcomes = []
    monkeypatch.setattr("pixelup.backup_store.record", lambda *_: None)
    assert flush_records(5)

    def disconnect_once(*args):
        attempts.append(args[0])
        if len(attempts) == 1:
            unavailable.replace()
        return real_save(*args)

    monkeypatch.setattr(gui, "save_app_config", disconnect_once)
    try:
        window._save_models_folder(str(target), outcomes.append)
        _settle_config(window)

        assert outcomes == [Message("settings.saveFailed")]
        assert calls == []
        assert window.config is original
        assert window._unsaved_config_changes == {"models_folder": str(target)}
        assert not window.model_manager.folder_change_pending
        assert not window.model_manager.operations
        unavailable.restore()

        window._save_models_folder(str(target), outcomes.append)
        _settle_config(window)
        _settle_models(window)

        assert outcomes == [Message("settings.saveFailed"), None]
        assert calls == []
        assert len(attempts) >= 2
        assert all(candidate.models_folder == str(target) for candidate in attempts)
        assert window.config.models_folder == str(target)
        assert window.model_manager.models_dir == target
        assert not window.model_manager.folder_change_pending
        assert not window._unsaved_config_changes
        assert load_app_config(accepted_path).models_folder == str(target)
    finally:
        unavailable.restore()


@pytest.mark.parametrize("os_ending", [False, True])
def test_failed_quit_save_settles_without_resolving_the_unavailable_root(
    make_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quit_bound, os_ending: bool
) -> None:
    window = make_window()
    unavailable = _isolate_settings_root(window, tmp_path, monkeypatch)
    calls = _track_runtime_config_resolution(monkeypatch)
    original = window.config
    real_save = gui.save_app_config
    dialog = _ChoicesDialog(["cancel"])
    monkeypatch.setattr(gui, "QuitSaveFailedDialog", _never if os_ending else dialog)
    monkeypatch.setattr("pixelup.backup_store.record", lambda *_: None)
    window._is_session_shutdown = lambda: os_ending
    assert flush_records(5)

    def disconnect_before_save(*args):
        unavailable.replace()
        return real_save(*args)

    monkeypatch.setattr(gui, "save_app_config", disconnect_before_save)
    window.quality.setValue(73)
    try:
        event = QCloseEvent()
        window.closeEvent(event)
        _settle_config(window)

        assert calls == []
        assert window.config is original
        assert window.current_job_settings().quality == 73
        assert window._unsaved_config_changes["parameters"].quality == 73
        assert quit_bound.exits == 0
        if os_ending:
            assert event.isAccepted()
            assert window._os_close_accepted
            assert quit_bound.started == [gui.QUIT_BUDGET_S]
            assert dialog.shown == 0
        else:
            assert not event.isAccepted()
            assert dialog.shown == 1
            assert not window._closing
            assert not window.runner._shutting_down
            assert quit_bound.started == []
            assert window.parameters_result.message_label.text() == english().of(
                Message("parameters.saveFailed")
            )
    finally:
        unavailable.restore()
        monkeypatch.setattr(gui, "save_app_config", real_save)


def test_close_waits_for_worker_ownership_to_end(
    make_window,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = make_window()
    cleanup_results = iter((False, True))
    warnings: list[object] = []
    shutdown_calls: list[bool] = []
    monkeypatch.setattr(window.runner, "begin_shutdown", lambda: shutdown_calls.append(True))
    monkeypatch.setattr(window.runner, "cleanup_for_quit", lambda: next(cleanup_results, True))
    monkeypatch.setattr("pixelup.gui.warn_jobs_stopping", warnings.append)

    first = QCloseEvent()
    window.closeEvent(first)

    assert first.isAccepted() is False
    assert window._window_state.value(window._GEOMETRY_KEY) is None
    assert window._quit_when_workers_idle is True
    assert shutdown_calls == [True]
    assert warnings == [window]

    second = QCloseEvent()
    window.closeEvent(second)
    assert second.isAccepted() is True
    assert window._window_state.value(window._GEOMETRY_KEY) is not None


class _ChoicesDialog:
    """Stands in for QuitSaveFailedDialog, answering from a script."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.shown = 0

    def __call__(self, _parent: object) -> _ChoicesDialog:
        return self

    def choose(self) -> str:
        self.shown += 1
        return self.answers.pop(0)


def _never(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("asked during an OS session end")


def _fields(caplog: pytest.LogCaptureFixture, message: str) -> list[dict]:
    return [record.fields for record in caplog.records if record.getMessage() == message]


@pytest.mark.parametrize("os_ending", [True, False])
def test_each_commit_request_says_afresh_whether_the_os_is_ending_the_session(
    make_window, monkeypatch: pytest.MonkeyPatch, os_ending: bool
) -> None:
    # macOS asks for every quit, Cmd+Q and the Dock's included, so an earlier
    # logout that did not end PixelUp must not make a later Quit silent.
    window = make_window()
    window._session_shutdown = not os_ending
    monkeypatch.setattr(gui, "os_session_ending", lambda: os_ending)

    window._on_commit_data_request(None)

    assert window._session_shutdown is os_ending


def test_a_failed_save_holds_a_user_quit_until_cancel(
    make_window, monkeypatch: pytest.MonkeyPatch, quit_bound
) -> None:
    window = make_window()
    window.quality.setValue(10)
    monkeypatch.setattr(
        gui, "save_app_config", lambda *_: (_ for _ in ()).throw(OSError("read-only"))
    )
    dialog = _ChoicesDialog(["cancel"])
    monkeypatch.setattr(gui, "QuitSaveFailedDialog", dialog)

    event = QCloseEvent()
    window.closeEvent(event)
    _settle_config(window)

    assert event.isAccepted() is False
    assert dialog.shown == 1
    assert window.parameters_result.isVisibleTo(window)
    assert window.current_job_settings().quality == 10
    # Nothing of the quit began: no bound, no shutdown, the queue still runs.
    assert quit_bound.started == []
    assert window.runner._shutting_down is False


def test_retry_saves_and_the_quit_goes_on(make_window, monkeypatch: pytest.MonkeyPatch) -> None:
    window = make_window()
    window.quality.setValue(10)
    real_save = gui.save_app_config
    attempts: list[int] = []

    def fail_once(candidate, previous, file):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("read-only")
        return real_save(candidate, previous, file)

    monkeypatch.setattr(gui, "save_app_config", fail_once)
    dialog = _ChoicesDialog(["retry"])
    monkeypatch.setattr(gui, "QuitSaveFailedDialog", dialog)

    event = QCloseEvent()
    window.closeEvent(event)
    _settle_config(window)

    assert event.isAccepted() is False
    assert window._quit_ready is True
    assert len(attempts) == 2
    assert load_app_config(config_path()).parameters.quality == 10


def test_quit_anyway_quits_without_the_save_and_logs_what_was_lost(
    make_window, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    window = make_window()
    window.quality.setValue(10)
    monkeypatch.setattr(
        gui, "save_app_config", lambda *_: (_ for _ in ()).throw(OSError("read-only"))
    )
    dialog = _ChoicesDialog(["retry", "quit_anyway"])
    monkeypatch.setattr(gui, "QuitSaveFailedDialog", dialog)

    event = QCloseEvent()
    window.closeEvent(event)
    _settle_config(window)

    assert event.isAccepted() is False
    assert window._quit_ready is True
    assert dialog.shown == 2
    assert not config_path().exists()
    [lost] = _fields(caplog, "quit.parameters_unsaved")
    assert lost["values"]["quality"] == 10


def test_a_user_quit_waits_for_running_work_only_until_the_bound(
    make_window,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quit_bound,
    caplog: pytest.LogCaptureFixture,
) -> None:
    window = make_window()
    image = _png(tmp_path, "a.png")
    _open(window, [image])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    window._queue_selected_image()
    _settle_models(window)
    window.jobs[0].status = "running"
    monkeypatch.setattr(
        gui,
        "QuitConfirmDialog",
        lambda *_: SimpleNamespace(exec=lambda: QDialog.DialogCode.Accepted),
    )
    notices: list[object] = []
    monkeypatch.setattr(gui, "warn_jobs_stopping", notices.append)

    with monkeypatch.context() as running_work:
        running_work.setattr(window.runner, "cleanup_for_quit", lambda: False)
        event = QCloseEvent()
        window.closeEvent(event)
        _settle_config(window)

        # The quit is under way and bounded; the window waits for the worker.
        assert event.isAccepted() is False
        assert window._quit_when_workers_idle is True
        assert notices == [window]
        assert quit_bound.started == [gui.QUIT_BUDGET_S]
        assert quit_bound.exits == 0

        quit_bound.expire()

        assert quit_bound.exits == 1
        deadline = time.monotonic() + 5
        while not _fields(caplog, "quit.forced_exit"):
            assert time.monotonic() < deadline
            threading.Event().wait(0.001)
        [forced] = _fields(caplog, "quit.forced_exit")
        assert forced["running_jobs"] == [
            {"job_id": window.jobs[0].id, "output": str(window.jobs[0].output_path)}
        ]


def test_the_bound_starts_once_however_often_the_quit_is_asked(
    make_window, monkeypatch: pytest.MonkeyPatch, quit_bound
) -> None:
    window = make_window()
    monkeypatch.setattr(gui, "warn_jobs_stopping", lambda _parent: None)

    with monkeypatch.context() as running_work:
        running_work.setattr(window.runner, "cleanup_for_quit", lambda: False)
        window.closeEvent(QCloseEvent())
        window.closeEvent(QCloseEvent())

        assert quit_bound.started == [gui.QUIT_BUDGET_S]


def test_an_os_session_end_never_asks_and_logs_the_failed_save(
    make_window,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quit_bound,
    caplog: pytest.LogCaptureFixture,
) -> None:
    window = make_window()
    _open(window, [_png(tmp_path, "a.png")])
    window.quality.setValue(10)
    bounded_before_save: list[bool] = []

    def failing_save(*_args: object) -> AppConfig:
        bounded_before_save.append(quit_bound.started == [gui.QUIT_BUDGET_S])
        raise OSError("read-only")

    monkeypatch.setattr(gui, "save_app_config", failing_save)
    for name in ("QuitConfirmDialog", "QuitSaveFailedDialog", "warn_jobs_stopping"):
        monkeypatch.setattr(gui, name, _never)
    window._is_session_shutdown = lambda: True

    event = QCloseEvent()
    window.closeEvent(event)
    _settle_config(window)

    assert event.isAccepted() is True
    assert bounded_before_save == [True]
    assert _fields(caplog, "config.save_failed")
    assert quit_bound.exits == 0


def test_an_os_session_end_with_work_still_running_exits_rather_than_refusing(
    make_window, monkeypatch: pytest.MonkeyPatch, quit_bound, caplog: pytest.LogCaptureFixture
) -> None:
    # Refusing the close would cancel a macOS logout, and waiting on would outlast
    # what the OS allows, so the worker is left to the exit.
    window = make_window()
    monkeypatch.setattr(gui, "warn_jobs_stopping", _never)
    window._is_session_shutdown = lambda: True

    with monkeypatch.context() as running_work:
        running_work.setattr(window.runner, "cleanup_for_quit", lambda: False)
        event = QCloseEvent()
        window.closeEvent(event)
        _settle_config(window)

        assert event.isAccepted() is True
        assert window._os_close_accepted is True
        assert quit_bound.exits == 0
        quit_bound.expire()
        assert quit_bound.exits == 1
        assert window._quit_when_workers_idle is False


def test_a_quit_that_settles_in_time_never_forces_the_exit(make_window, quit_bound) -> None:
    window = make_window()

    event = QCloseEvent()
    window.closeEvent(event)
    _settle_config(window)

    assert event.isAccepted() is True
    assert quit_bound.started == [gui.QUIT_BUDGET_S]
    assert quit_bound.exits == 0


def test_the_forced_exit_happens_even_when_its_log_fails(
    make_window, monkeypatch: pytest.MonkeyPatch, quit_bound
) -> None:
    window = make_window()
    monkeypatch.setattr(window, "jobs", None)  # what the watchdog reads is mid-change

    window._force_exit()

    assert quit_bound.exits == 1


def test_close_claim_precedes_the_confirmation_and_os_can_take_it_over(
    make_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quit_bound
) -> None:
    window = make_window()
    _open(window, [_png(tmp_path, "a.png")])
    confirmations: list[bool] = []
    shutdown = False
    monkeypatch.setattr(window, "_is_session_shutdown", lambda: shutdown)
    monkeypatch.setattr(gui, "os_session_ending", lambda: shutdown)

    class Confirmation:
        def __init__(self, *_args):
            self.rejected = False

        def reject(self):
            self.rejected = True

        def exec(self):
            nonlocal shutdown
            confirmations.append(window._closing)
            nested = QCloseEvent()
            window.closeEvent(nested)
            assert not nested.isAccepted()
            assert len(confirmations) == 1
            shutdown = True
            window._on_commit_data_request(None)
            assert self.rejected
            assert quit_bound.started == [gui.QUIT_BUDGET_S]
            return QDialog.DialogCode.Rejected

    monkeypatch.setattr(gui, "QuitConfirmDialog", Confirmation)
    event = QCloseEvent()
    window.closeEvent(event)
    assert confirmations == [True]
    assert window._quit_ready
    assert window._closing
    assert quit_bound.started == [gui.QUIT_BUDGET_S]


@pytest.mark.parametrize(
    ("platform", "live_session", "native_session", "expected"),
    [
        ("win32", False, True, False),
        ("win32", True, False, True),
        ("darwin", True, False, False),
        ("darwin", False, True, True),
        ("darwin", True, True, True),
    ],
)
def test_session_shutdown_requires_the_current_platform_session(
    make_window,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    live_session: bool,
    native_session: bool,
    expected: bool,
) -> None:
    window = make_window()
    window._session_shutdown = native_session
    monkeypatch.setattr(gui.sys, "platform", platform)
    monkeypatch.setattr(
        gui,
        "QGuiApplication",
        SimpleNamespace(instance=lambda: SimpleNamespace(isSavingSession=lambda: live_session)),
    )
    assert window._is_session_shutdown() is expected


def test_hard_quit_bounds_a_blocked_logging_tail() -> None:
    # Use the real os._exit in a disposable process: neither Handler.close nor
    # the final diagnostic can delay the already committed quit deadline.
    script = """
import logging
import threading
from types import SimpleNamespace
from pixelup import gui
class HeldHandler(logging.Handler):
    def close(self):
        print('logging tail entered', flush=True)
        threading.Event().wait()
logging.getLogger().addHandler(HeldHandler())
owner = SimpleNamespace(_log_forced_exit=lambda: threading.Event().wait())
gui._start_quit_watchdog(0.05, lambda: gui.MainWindow._force_exit(owner))
logging.shutdown()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0
    assert "logging tail entered" in result.stdout


def test_main_window_restores_geometry_saved_on_normal_close(make_window) -> None:
    first = make_window()
    # The offscreen Qt test screen is only 800 px wide and the content-derived
    # minimum is nearly that wide, so keep the fixture geometry wholly on-screen.
    first.setGeometry(1, 40, first.minimumWidth(), 650)
    first._is_session_shutdown = lambda: True
    assert first.close()

    second = make_window()
    assert second.geometry() == first.geometry()


def test_main_window_saves_and_restores_normal_geometry(make_window) -> None:
    first = make_window()
    first.setGeometry(1, 40, first.minimumWidth(), 650)
    expected = first.saveGeometry()
    first._is_session_shutdown = lambda: True

    assert first.close()

    settings = QSettings(str(first._window_state.path), QSettings.Format.IniFormat)
    assert settings.value(first._GEOMETRY_KEY) == expected

    second = make_window()
    assert second.saveGeometry() == expected


@pytest.mark.parametrize(
    ("platform", "saved", "expected"),
    [
        ("darwin", Qt.WindowState.WindowMaximized, Qt.WindowState.WindowNoState),
        ("darwin", Qt.WindowState.WindowFullScreen, Qt.WindowState.WindowNoState),
        ("win32", Qt.WindowState.WindowMaximized, Qt.WindowState.WindowMaximized),
        ("win32", Qt.WindowState.WindowFullScreen, Qt.WindowState.WindowFullScreen),
        ("win32", Qt.WindowState.WindowMinimized, Qt.WindowState.WindowNoState),
        (
            "win32",
            Qt.WindowState.WindowMinimized | Qt.WindowState.WindowMaximized,
            Qt.WindowState.WindowMaximized,
        ),
    ],
)
def test_restored_window_state_is_platform_appropriate(
    platform: str, saved: Qt.WindowState, expected: Qt.WindowState
) -> None:
    assert restored_window_state(saved, platform) == expected


@pytest.mark.parametrize(
    "window_state",
    [
        Qt.WindowState.WindowMinimized,
        Qt.WindowState.WindowMaximized,
        Qt.WindowState.WindowFullScreen,
    ],
)
def test_main_window_saves_geometry_on_every_accepted_close(make_window, window_state) -> None:
    window = make_window()
    window.setGeometry(1, 40, window.minimumWidth(), 650)
    window.setWindowState(window_state)
    expected = window.saveGeometry()
    window._is_session_shutdown = lambda: True

    assert window.close()

    settings = QSettings(str(window._window_state.path), QSettings.Format.IniFormat)
    assert settings.value(window._GEOMETRY_KEY) == expected


@pytest.mark.parametrize(
    "window_state",
    [Qt.WindowState.WindowMaximized, Qt.WindowState.WindowFullScreen],
)
def test_main_window_restores_native_state_according_to_platform(make_window, window_state) -> None:
    first = make_window()
    first.setGeometry(1, 40, first.minimumWidth(), 650)
    normal_geometry = first.normalGeometry()
    first.setWindowState(window_state)
    first._is_session_shutdown = lambda: True
    assert first.close()

    second = make_window()
    assert second.windowState() == restored_window_state(window_state, sys.platform)
    assert second.normalGeometry() == normal_geometry


def test_main_surfaces_startup_storage_failure(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shown: list[tuple[Message, Message | None]] = []
    error = gui.PixelupError(
        ErrorCode.OUTPUT_UNWRITABLE,
        Message("error.storageCreateFailed"),
        hint=Message.of("error.hintHomeWritableLocation", variable="PIXELUP_DATA_DIR"),
    )
    monkeypatch.setattr("pixelup.gui.build_app", lambda argv: (_ for _ in ()).throw(error))
    monkeypatch.setattr(
        "pixelup.gui.show_startup_failure",
        lambda detail, hint: shown.append((detail, hint)),
    )

    assert gui.main() == 1
    assert shown == [(error.message, error.hint)]


def test_main_does_not_surface_an_unexpected_startup_exception(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shown: list[tuple[Message, Message | None]] = []
    diagnostic_sentinel = "EACCES Error invoking remote method /private/var/tmp/pixelup"
    monkeypatch.setattr(
        "pixelup.gui.build_app",
        lambda argv: (_ for _ in ()).throw(OSError(diagnostic_sentinel)),
    )
    monkeypatch.setattr(
        "pixelup.gui.show_startup_failure",
        lambda detail, hint: shown.append((detail, hint)),
    )

    assert gui.main() == 1
    assert shown == [(Message("startup.storageUnreadable"), None)]
    assert diagnostic_sentinel not in english().of(shown[0][0])


def test_scale_edit_persists_to_config_json(make_window) -> None:
    # Scale rides the same debounced save as the rest of the panel: clicking the radio
    # (not calling a setter) reaches config.json, so the next launch opens on it. Read
    # back from disk so the serializer and the loader are both in the loop.
    window = make_window()
    assert window.current_job_settings().scale == DEFAULT_SCALE
    assert window._parameters_save_timer.isActive() is False

    window.scale_buttons[2].click()

    # The click alone schedules the debounced save. Asserted before the flush, because
    # a flush would hide an unwired radio: closeEvent flushes too, so a scale that only
    # ever saved at close would still pass a flush-then-read check while quietly losing
    # the edit on a crash or a force-quit.
    assert window._parameters_save_timer.isActive() is True

    _flush(window)

    assert load_app_config(config_path()).parameters.scale == 2


def test_window_seeds_the_panel_from_the_persisted_parameters(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Startup reads the persisted panel, not a defaults factory: whatever the user
    # left behind is what the panel opens on.
    monkeypatch.setenv("PIXELUP_DATA_DIR", str(tmp_path / "home"))
    monkeypatch.setattr(JobRunner, "schedule", lambda self, max_concurrent_jobs: None)
    parameters = JobSettings(
        scale=2, quality=42, tile=512, alpha_mode="bicubic", target_profile="p3"
    )
    monkeypatch.setattr(
        "pixelup.gui.load_app_config_result",
        lambda: ConfigLoadResult(AppConfig(parameters=parameters), SettingsFile(config_path())),
    )
    log_file = configure_session_logging()

    window = MainWindow(log_file=log_file)
    try:
        assert window.current_job_settings() == parameters
        assert window.quality.value() == 42
        assert window.tile.currentData() == 512
        assert window.alpha_mode.currentData() == "bicubic"
        assert window.target_profile.currentData() == "p3"
        # The persisted scale reaches the radios, not just the settings snapshot.
        assert window.scale_buttons[2].isChecked() is True
        assert window.scale_buttons[4].isChecked() is False
    finally:
        window._is_session_shutdown = lambda: True
        window.close()
        _settle_config(window)
        window.deleteLater()
        qapp.processEvents()


def test_queued_jobs_keep_their_snapshot_when_the_panel_changes(
    make_window, tmp_path: Path
) -> None:
    # Persisting the panel must not reach backwards into work already queued: each job
    # captures the panel at creation and holds that snapshot, so editing the panel
    # afterwards changes the next job, never the queued one.
    window = make_window()
    _open(window, [_png(tmp_path, "a.png")])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    window.tile.setCurrentIndex(window.tile.findData(128))
    window._queue_selected_image()
    _settle_models(window)
    queued = window.jobs[0]
    assert queued.settings.tile == 128

    window.tile.setCurrentIndex(window.tile.findData(1024))
    _flush(window)

    assert queued.settings.tile == 128
    assert window.config.parameters.tile == 1024
    window._queue_selected_image()
    _settle_models(window)
    assert window.jobs[1].settings.tile == 1024


def test_queued_jobs_keep_the_scale_they_were_enqueued_with(make_window, tmp_path: Path) -> None:
    # The enqueue snapshot, through the real window, for the field that just joined it.
    # Scale is now persisted and resettable, so it is exactly the field that could
    # start leaking backwards into queued work — a job must keep the scale the panel
    # had when the user pressed Queue, in its settings, its planned output name, and
    # the row the queue table shows.
    window = make_window()
    _open(window, [_png(tmp_path, "a.png")])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    window.scale_buttons[2].click()
    window._queue_selected_image()
    _settle_models(window)
    queued = window.jobs[0]

    assert queued.settings.scale == 2
    assert queued.output_path.name == "a-realesr-general-x4v3-2x.png"
    assert window.queue_table.item(0, 2).text() == "2x"

    # The panel moves on, and the reset button is the most forceful way it can.
    _reset_button(window).click()
    _settle_config(window)

    assert queued.settings.scale == 2
    assert queued.output_path.name == "a-realesr-general-x4v3-2x.png"
    assert window.queue_table.item(0, 2).text() == "2x"
    assert window.config.parameters.scale == DEFAULT_SCALE

    # The next job takes the panel as it now stands.
    window._queue_selected_image()
    _settle_models(window)
    assert window.jobs[1].settings.scale == DEFAULT_SCALE
    assert window.queue_table.item(1, 2).text() == "4x"


def test_a_stray_persisted_tile_falls_back_before_the_panel_is_seeded(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The panel's combo seeds via findData, which answers -1 for a value it does not
    # hold — and setCurrentIndex(-1) blanks the box, making currentData() None. That
    # None would flow straight back into JobSettings.tile on the next enqueue or save.
    #
    # Nothing in the panel prevents it, so this walks the real loader (not an injected
    # AppConfig) with a tile that is not a choice and pins the whole recovery chain:
    # invalid parameters -> built-ins -> a panel that shows a real default value.
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "max_concurrent_jobs": 1,
                "auto_download": True,
                "font_family": "",
                "parameters": {"tile": 9999},
            }
        )
    )
    monkeypatch.setenv("PIXELUP_DATA_DIR", str(home))
    monkeypatch.setattr(JobRunner, "schedule", lambda self, max_concurrent_jobs: None)
    notices: list[object] = []
    monkeypatch.setattr(
        "pixelup.gui.warn_config_reset",
        lambda _parent, _path: notices.append("reset"),
    )
    monkeypatch.setattr(
        "pixelup.gui.warn_config_rejected",
        lambda _parent, settings, path: notices.append((settings, path)),
    )
    log_file = configure_session_logging()

    window = MainWindow(log_file=log_file)
    try:
        assert window._config_quarantined_to is None
        assert window.tile.currentIndex() != -1, "the combo blanked on an unknown tile"
        assert window.tile.currentData() is not None
        assert window.tile.currentData() == JobSettings().tile
        # And the value the panel would hand a job is a real one, not None.
        assert window.current_job_settings().tile in TILE_VALUES
        qapp.processEvents()
        # The user is told which settings run on their built-ins, and where the
        # file keeping their stored values is; nothing was reset.
        assert notices == [((Message("parameters.title"),), home / "config.json")]
    finally:
        window._is_session_shutdown = lambda: True
        window.close()
        _settle_config(window)
        window.deleteLater()
        qapp.processEvents()


def test_the_window_itself_takes_the_configured_font_at_launch(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The window exists before the configured font is applied, so without its own
    # setFont it kept the platform font, and anything inheriting from it did too.
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.json").write_text(json.dumps({"format_version": 1, "font_family": "Menlo"}))
    monkeypatch.setenv("PIXELUP_DATA_DIR", str(home))
    monkeypatch.setattr(JobRunner, "schedule", lambda self, max_concurrent_jobs: None)
    original = qapp.font()
    window = MainWindow(log_file=configure_session_logging())
    try:
        assert window.font().family() == "Menlo"
        assert window.settings_button.font().family() == "Menlo"
    finally:
        window._is_session_shutdown = lambda: True
        window.close()
        _settle_config(window)
        window.deleteLater()
        qapp.processEvents()
        qapp.setFont(original)


def test_fresh_launch_and_close_do_not_create_config(make_window) -> None:
    window = make_window()
    assert not config_path().exists()
    assert window.config == AppConfig()
    window._is_session_shutdown = lambda: True
    window.close()
    assert not config_path().exists()


def test_failed_reset_retries_and_the_retry_removes_the_stored_set(
    make_window,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = make_window()
    window.quality.setValue(10)
    _flush(window)
    original = window.config
    saved = config_path().read_bytes()
    real_save = gui.save_app_config
    attempts: list[JobSettings] = []

    def fail_once(candidate, previous, file):
        attempts.append(candidate.parameters)
        if len(attempts) == 1:
            raise OSError("disk full")
        return real_save(candidate, previous, file)

    monkeypatch.setattr(gui, "save_app_config", fail_once)
    _reset_button(window).click()
    _settle_config(window)
    assert window.config == original
    assert window.current_job_settings() == JobSettings()
    assert config_path().read_bytes() == saved
    assert _flush(window) is True
    assert attempts == [JobSettings(), JobSettings()]
    assert json.loads(config_path().read_text(encoding="utf-8")) == {"format_version": 1}
    assert window.config.parameters == JobSettings()


@pytest.mark.parametrize("continuation", ["quit", "settings"])
def test_saved_continuation_waits_for_the_newest_visible_panel(
    make_window,
    monkeypatch: pytest.MonkeyPatch,
    process_until,
    continuation: str,
) -> None:
    window = make_window()
    window.quality.setValue(10)
    entered = [threading.Event(), threading.Event()]
    released = [threading.Event(), threading.Event()]
    attempts = []
    opened = []

    def save(candidate, previous, file):
        index = len(attempts)
        attempts.append(candidate.parameters.quality)
        entered[index].set()
        assert released[index].wait(5)
        return ConfigSaveResult(candidate, file)

    monkeypatch.setattr(gui, "save_app_config", save)
    monkeypatch.setattr(
        window,
        "_open_settings_after_save",
        lambda saved: opened.append(
            (saved, window.config.parameters.quality, window.current_job_settings().quality)
        ),
    )
    if continuation == "quit":
        window.closeEvent(QCloseEvent())
    else:
        window._settings_dialog()
    try:
        process_until(entered[0].is_set, timeout_s=5, what="first panel save")
        window.quality.setValue(11)
        released[0].set()
        process_until(entered[1].is_set, timeout_s=5, what="newest panel save")
        assert attempts == [10, 11]
        assert window._quit_ready is False
        assert window._quit_bounded is False
        assert opened == []
        released[1].set()
        _settle_config(window)
        assert window.config.parameters.quality == 11
        if continuation == "quit":
            assert window._quit_ready is True
        else:
            assert opened == [(True, 11, 11)]
    finally:
        for release in released:
            release.set()
        _settle_config(window)


def test_held_config_save_keeps_gui_live_and_retains_newer_panel_edits(
    make_window, monkeypatch: pytest.MonkeyPatch, process_until
) -> None:
    window = make_window()
    window.quality.setValue(10)
    started = threading.Event()
    release = threading.Event()
    attempts = []
    saved = []

    def save(candidate, previous, file):
        attempts.append((candidate, previous))
        started.set()
        assert release.wait(5)
        return ConfigSaveResult(candidate, file)

    monkeypatch.setattr(gui, "save_app_config", save)
    window._flush_parameters_save(finished=saved.append)
    try:
        process_until(started.is_set, timeout_s=5, what="held save admission")
        window.quality.setValue(11)
        window._flush_parameters_save()
        window._config_save_timed_out()
        assert len(attempts) == 1
        assert window._config_save is not None
        assert window.config.parameters.quality != 10
        assert window.current_job_settings().quality == 11
    finally:
        release.set()
        _settle_config(window)
    assert saved == [True]
    assert window.config.parameters.quality == 11
    assert window.current_job_settings().quality == 11
    assert _flush(window) is True
    assert window.config.parameters.quality == 11
    assert len(attempts) == 2


def test_every_stored_settings_set_has_a_name_for_the_rejected_notice() -> None:
    from pixelup import app_config

    assert set(gui._SET_LABELS) == set(app_config._SET_DECODERS)


def test_opening_images_never_waits_on_the_volume(
    make_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    import time

    window = make_window()
    image = _png(tmp_path, "slow.png")
    entered = threading.Event()
    release = threading.Event()
    real_size = gui._safe_image_size

    def stalled_size(path: Path):
        entered.set()
        assert release.wait(5)
        return real_size(path)

    monkeypatch.setattr(gui, "_safe_image_size", stalled_size)
    started = time.monotonic()
    window.open_paths([image])
    assert time.monotonic() - started < 1
    assert entered.wait(5)
    assert window.opening is True
    assert window.image_table.rowCount() == 0

    release.set()
    _settle_opens(window)
    assert window.image_table.rowCount() == 1
    assert window._selected_path() == image.resolve()


@pytest.mark.parametrize(
    ("metadata", "orientation", "expected_size", "quadrants"),
    [
        ("jpeg", 2, (40, 20), ("green", "red", "yellow", "blue")),
        ("jpeg", 6, (20, 40), ("blue", "red", "yellow", "green")),
        ("png-exif", 2, (40, 20), ("green", "red", "yellow", "blue")),
        ("png-exif", 3, (40, 20), ("yellow", "blue", "green", "red")),
        ("png-exif", 4, (40, 20), ("blue", "yellow", "red", "green")),
        ("png-exif", 5, (20, 40), ("red", "blue", "green", "yellow")),
        ("png-exif", 6, (20, 40), ("blue", "red", "yellow", "green")),
        ("png-exif", 7, (20, 40), ("yellow", "green", "blue", "red")),
        ("png-exif", 8, (20, 40), ("green", "yellow", "red", "blue")),
        ("png-xmp-attribute", 6, (20, 40), ("blue", "red", "yellow", "green")),
        ("png-xmp-element", 6, (20, 40), ("blue", "red", "yellow", "green")),
    ],
)
def test_background_preview_honors_orientation_in_dimensions_and_pixels(
    make_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process_until,
    metadata: str, orientation: int, expected_size: tuple[int, int], quadrants: tuple[str, ...],
) -> None:
    colors = {
        "red": (240, 20, 20), "green": (20, 240, 20),
        "blue": (20, 20, 240), "yellow": (240, 240, 20),
    }
    pixels = Image.new("RGB", (40, 20))
    for color, box in zip(
        ("red", "green", "blue", "yellow"),
        ((0, 0, 20, 10), (20, 0, 40, 10), (0, 10, 20, 20), (20, 10, 40, 20)),
        strict=True,
    ):
        pixels.paste(colors[color], box)
    source = tmp_path / ("oriented.jpg" if metadata == "jpeg" else "oriented.png")
    if metadata.startswith("png-xmp"):
        xmp = (
            '<rdf:Description xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
            ' xmlns:tiff="http://ns.adobe.com/tiff/1.0/"'
        )
        xmp += (
            f' tiff:Orientation="{orientation}"/>' if metadata.endswith("attribute")
            else f'><tiff:Orientation>{orientation}</tiff:Orientation></rdf:Description>'
        )
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_itxt("XML:com.adobe.xmp", xmp)
        pixels.save(source, pnginfo=pnginfo)
    else:
        exif = Image.Exif()
        exif[0x0112] = orientation
        if metadata == "jpeg":
            pixels.save(source, "JPEG", quality=100, subsampling=0, exif=exif)
        else:
            pixels.save(source, exif=exif)
    reads: list[tuple[Path, threading.Thread]] = []
    real_read = QImageReader.read

    def read(reader):
        reads.append((Path(reader.fileName()), threading.current_thread()))
        return real_read(reader)

    monkeypatch.setattr(QImageReader, "read", read)
    window = make_window()

    _open(window, [source])

    preview = window.preview._pixmap
    assert preview is not None
    assert (preview.width(), preview.height()) == expected_size
    assert window._images_by_path[source].input_size == expected_size
    displayed = preview.toImage()
    width, height = expected_size
    points = ((width // 4, height // 4), (3 * width // 4, height // 4),
              (width // 4, 3 * height // 4), (3 * width // 4, 3 * height // 4))
    for (x, y), color in zip(points, quadrants, strict=True):
        pixel = displayed.pixelColor(x, y)
        assert (pixel.red(), pixel.green(), pixel.blue()) == pytest.approx(colors[color], abs=4)
    assert len(reads) == 1
    assert reads[0][0] == source
    assert reads[0][1] is not threading.main_thread()
    process_until(lambda: window.preview.reads.settled, timeout_s=5, what="Preview read return")


def test_optional_preview_metadata_failure_preserves_the_decoded_image(
    make_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process_until,
) -> None:
    window = make_window()
    source = _png(tmp_path, "readable.png")
    sentinel = "EACCES /internal/image ORIGINAL-PREVIEW-METADATA-SENTINEL"
    probes: list[threading.Thread] = []
    reads: list[Path] = []
    real_read = QImageReader.read

    def read(reader):
        reads.append(Path(reader.fileName()))
        return real_read(reader)

    def fail_probe(path):
        assert path == source
        probes.append(threading.current_thread())
        try:
            raise OSError(sentinel)
        except OSError as cause:
            raise RuntimeError("metadata probe failed") from cause

    monkeypatch.setattr(QImageReader, "read", read)
    monkeypatch.setattr(gui, "read_image_orientation", fail_probe)

    _open(window, [source])

    assert reads == [source]
    assert len(probes) == 1
    assert probes[0] is not threading.main_thread()
    preview = window.preview._pixmap
    assert preview is not None
    assert (preview.width(), preview.height()) == (8, 6)
    assert preview.toImage().pixelColor(0, 0) == QColor("white")
    assert window.preview.text() == ""
    process_until(lambda: window.preview.reads.settled, timeout_s=5, what="Preview read return")
    assert flush_records(5)
    with sqlite3.connect(config_path().parent / "records.sqlite3") as connection:
        entries = connection.execute(
            "SELECT level, fields FROM logs WHERE message = 'preview.orientation_failed'"
        ).fetchall()
    assert len(entries) == 1
    level, fields = entries[0]
    assert level == "warn"
    error = json.loads(fields)["error"]
    assert error["type"] == "RuntimeError"
    assert error["message"] == "metadata probe failed"
    assert "OSError: " + sentinel in error["traceback"]


def test_the_preview_reads_an_image_once_while_it_stays_selected(
    make_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = make_window()
    image = _png(tmp_path, "a.png")
    reads: list[str] = []
    real_read = gui.QImageReader.read

    def counted_read(reader):
        reads.append(reader.fileName())
        return real_read(reader)

    monkeypatch.setattr(gui.QImageReader, "read", counted_read)
    _open(window, [image])
    assert reads == [str(image)]
    for _ in range(3):
        window._update_selected_image()
    _settle_opens(window)

    assert reads == [str(image)]


def test_a_settings_save_that_lands_after_cancel_still_takes_effect(
    make_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The dialog may close while its save is slow; the save still lands, so its
    # language, font and job count must apply rather than wait for a restart.
    from pixelup.i18n import localizer

    window = make_window()
    scheduled: list[int] = []
    monkeypatch.setattr(window.runner, "schedule", scheduled.append)
    release = threading.Event()
    real_save = gui.save_app_config

    def slow_save(candidate, previous, file):
        assert release.wait(5)
        return real_save(candidate, previous, file)

    monkeypatch.setattr(gui, "save_app_config", slow_save)
    outcomes: list[object] = []
    candidate = replace(window.config, language="ja", max_concurrent_jobs=3)
    try:
        window._save_config_candidate(candidate, Message("settings.saveFailed"), outcomes.append)
        # The dialog that submitted it is gone; nothing waits for the answer.
        release.set()
        _settle_config(window)

        assert outcomes == [None]
        assert window.config == candidate
        assert localizer.preference() == "ja"
        assert scheduled == [3]
    finally:
        localizer.use("system")


def test_install_and_queue_skips_an_image_removed_while_models_downloaded(
    make_window,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    window = make_window()
    image = _png(tmp_path, "a.png")
    _open(window, [image])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    required = ("realesr-general-x4v3", "realesr-general-wdn-x4v3")
    for name in required:
        model_file(window.runtime_dirs.models_dir, name).unlink()

    window._queue_selected_image()
    _settle_models(window)
    assert window._pending_model_work
    window._remove_selected_image()
    assert window.image_table.rowCount() == 0
    assert not window._pending_model_work
    _open(window, [image])
    for name in required:
        target = model_file(window.runtime_dirs.models_dir, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"ready")
    window.model_manager.refresh_readiness()
    _settle_models(window)
    qapp.processEvents()

    assert not window._pending_model_work
    assert window.jobs == []


def test_pending_requests_preserve_each_settings_snapshot(make_window, tmp_path, monkeypatch):
    window = make_window()
    first, second = _png(tmp_path, "first.png"), _png(tmp_path, "second.png")
    _open(window, [first, second])
    model = "RealESRGAN_x4plus"
    model_file(window.runtime_dirs.models_dir, model).unlink()
    settings = JobSettings(scale=2)
    monkeypatch.setattr(window, "current_job_settings", lambda: settings)
    window._enqueue_jobs([first], [model])
    _settle_models(window)
    dialog = window._active_models_dialog
    settings = JobSettings(scale=3)
    window._enqueue_jobs([second], [model])
    _settle_models(window)
    settings = JobSettings(scale=4)
    assert window._active_models_dialog is dialog
    assert dialog._pending_job_count == 2
    assert len(window._pending_model_work) == 2

    model_file(window.runtime_dirs.models_dir, model).write_bytes(b"ready")
    window.model_manager.refresh_readiness()
    _settle_models(window)
    window._model_manager_changed()
    assert [(job.input_path, job.settings.scale) for job in window.jobs] == [
        (first, 2), (second, 3),
    ]
    assert not window._pending_model_work
    assert window._active_models_dialog is None


def test_ready_request_does_not_abandon_unrelated_pending_retry(make_window, tmp_path):
    window = make_window()
    image = _png(tmp_path, "image.png")
    _open(window, [image])
    first, second = "RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"
    window._enqueue_jobs([image], [second])
    _settle_models(window)
    retry = window.jobs[0]
    retry.status = "failed"
    for model in (first, second):
        model_file(window.runtime_dirs.models_dir, model).unlink()
    window._enqueue_jobs([image], [first])
    _settle_models(window)
    window._retry_failed()
    _settle_models(window)
    dialog = window._active_models_dialog
    assert dialog._pending_job_count == 2
    assert set(dialog._required_artifacts) == {first, second}

    model_file(window.runtime_dirs.models_dir, first).write_bytes(b"ready")
    window.model_manager.refresh_readiness()
    _settle_models(window)
    assert len(window.jobs) == 2
    assert retry.status == "failed"
    assert len(window._pending_model_work) == 1
    assert window._active_models_dialog is dialog
    assert dialog._pending_job_count == 1
    assert dialog._required_artifacts == (second,)

    model_file(window.runtime_dirs.models_dir, second).write_bytes(b"ready")
    window.model_manager.refresh_readiness()
    _settle_models(window)
    assert retry.status == "pending"
    assert len(window.jobs) == 2
    assert not window._pending_model_work


def test_cancelled_model_preserves_requests_for_other_models(make_window, tmp_path):
    window = make_window()
    image = _png(tmp_path, "image.png")
    _open(window, [image])
    first, second = "RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"
    for model in (first, second):
        model_file(window.runtime_dirs.models_dir, model).unlink()
    for model in (first, first, second):
        window._enqueue_jobs([image], [model])
        _settle_models(window)
    dialog = window._active_models_dialog
    window._model_install_cancelled((first,))
    assert len(window._pending_model_work) == 1
    assert window._active_models_dialog is dialog
    assert dialog._required_artifacts == (second,)
    assert dialog._pending_job_count == 1
    model_file(window.runtime_dirs.models_dir, second).write_bytes(b"ready")
    window.model_manager.refresh_readiness()
    _settle_models(window)
    assert [job.model for job in window.jobs] == [second]


def test_dismissal_keeps_only_requests_with_active_downloads(make_window, tmp_path, monkeypatch):
    window = make_window()
    image = _png(tmp_path, "image.png")
    _open(window, [image])
    first, second = "RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"
    for model in (first, second):
        model_file(window.runtime_dirs.models_dir, model).unlink()
        window._enqueue_jobs([image], [model])
        _settle_models(window)
    monkeypatch.setattr(
        window.model_manager, "in_progress_for", lambda names: (object(),) if first in names else ()
    )
    window._active_models_dialog.reject()
    assert len(window._pending_model_work) == 1
    assert window._pending_model_work[0].required_artifacts == (first,)


def test_hard_quit_bounds_a_native_qthread_wait() -> None:
    # A disposable process exercises the real PySide6 wait and production timer.
    script = """
import threading
from types import SimpleNamespace
from PySide6.QtCore import QThread
from pixelup import gui
assert 'torch' not in __import__('sys').modules
class HeldThread(QThread):
    def run(self):
        threading.Event().wait()
worker = HeldThread()
worker.start()
owner = SimpleNamespace(_log_forced_exit=lambda: None)
def expire():
    print('watchdog expired', flush=True)
    gui.MainWindow._force_exit(owner)
gui._start_quit_watchdog(0.05, expire)
print('native wait entered', flush=True)
worker.wait()
raise RuntimeError('wait unexpectedly returned')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["native wait entered", "watchdog expired"]


def test_models_folder_save_applies_and_preserves_old_files(make_window, tmp_path):
    window = make_window(injected=False)
    old = window.runtime_dirs.models_dir
    old_files = set(old.iterdir())
    target = tmp_path / "new models"
    outcomes = []
    window._save_models_folder(str(target), outcomes.append)
    _settle_config(window)
    _settle_models(window)
    assert outcomes == [None]
    assert window.runtime_dirs.models_dir == target
    assert window.model_manager.models_dir == target
    assert window.runner._runtime_dirs.models_dir == target
    assert load_app_config().models_folder == str(target)
    assert set(old.iterdir()) == old_files
    assert not target.exists()  # choosing a location is not a download
    window._save_models_folder("", outcomes.append)
    _settle_config(window)
    _settle_models(window)
    assert window.model_manager.models_dir == old
    assert "models_folder" not in json.loads(config_path().read_text())


def test_failed_folder_save_keeps_current_location_and_starts_no_install(
    make_window, tmp_path, monkeypatch
):
    window = make_window(injected=False)
    old = window.runtime_dirs.models_dir
    window._managed_models_dialog()
    _settle_models(window)
    dialog = window._active_models_dialog
    dialog.folder_edit.setText(str(tmp_path / "new"))
    def fail(*args):
        raise OSError("private failure details")
    monkeypatch.setattr(gui, "save_app_config", fail)
    dialog._install_bundle(0)
    _settle_config(window)
    assert window.runtime_dirs.models_dir == old
    assert not window.model_manager.operations
    assert not window.model_manager.folder_change_pending
    assert dialog.result_view.isVisibleTo(dialog)
    assert dialog._folder_error == Message("settings.saveFailed")
    dialog.folder_edit.setText("")


def test_late_folder_save_after_dialog_close_applies_without_starting_download(
    make_window, tmp_path, monkeypatch
):
    window = make_window(injected=False)
    entered, release = threading.Event(), threading.Event()
    real_save = gui.save_app_config
    def delayed(*args):
        entered.set()
        assert release.wait(5)
        return real_save(*args)
    monkeypatch.setattr(gui, "save_app_config", delayed)
    window._managed_models_dialog()
    _settle_models(window)
    dialog = window._active_models_dialog
    target = tmp_path / "new"
    dialog.folder_edit.setText(str(target))
    try:
        dialog._install_bundle(0)
        assert entered.wait(3)
        assert window.model_manager.folder_change_pending
        assert window.model_manager.install(("RealESRGAN_x4plus",), force=False) is None
        dialog.reject()
        QApplication.processEvents()
    finally:
        release.set()
    _settle_config(window)
    _settle_models(window)
    assert window.runtime_dirs.models_dir == target
    assert not window.model_manager.operations
    assert not window.model_manager.folder_change_pending


def test_saved_folder_load_and_environment_override(make_window, tmp_path, monkeypatch):
    target = tmp_path / "saved models"
    monkeypatch.setattr(gui, "load_app_config_result", lambda: ConfigLoadResult(
        AppConfig(models_folder=str(target)), SettingsFile(config_path())
    ))
    window = make_window(injected=False)
    assert window.runtime_dirs.models_dir == target
    override = tmp_path / "override"
    monkeypatch.setenv("PIXELUP_MODELS_DIR", str(override))
    overridden = make_window(injected=False)
    assert overridden.runtime_dirs.models_dir == override
    assert overridden._models_folder_locked
    results = []
    overridden._save_models_folder(str(tmp_path / "ignored"), results.append)
    assert results == [None]
    assert overridden.config.models_folder == str(target)


def test_first_download_uses_saved_choice_and_releases_captured_request(
    make_window, tmp_path, monkeypatch, process_until
):
    window = make_window(injected=False)
    image = _png(tmp_path, "input.png")
    _open(window, [image])
    model = "RealESRGAN_x4plus"
    model_file(window.runtime_dirs.models_dir, model).unlink()
    monkeypatch.setattr(window, "current_job_settings", lambda: JobSettings(scale=3))
    window._enqueue_jobs([image], [model])
    _settle_models(window)
    dialog = window._active_models_dialog
    assert dialog is not None
    assert dialog.folder_edit.text() == ""
    assert dialog.folder_edit.placeholderText() == str(window.runtime_dirs.models_dir)
    target = tmp_path / "chosen"
    downloads = []
    def install(folder, name, **kwargs):
        downloads.append((folder, name))
        folder.mkdir(parents=True, exist_ok=True)
        model_file(folder, name).write_bytes(b"model")
    monkeypatch.setattr("pixelup.model_manager.download_model", install)
    dialog.folder_edit.setText(str(target))
    dialog._install_all_or_cancel()
    process_until(lambda: len(window.jobs) == 1 and window.model_manager.cleanup_for_quit(),
                  timeout_s=5, what="Saved-folder download and captured job")
    assert downloads == [(target, model)]
    assert window.jobs[0].settings.scale == 3
    assert window.config.models_folder == str(target)
    assert not window._pending_model_work


@pytest.mark.parametrize("setting", ["models_folder", "max_concurrent_jobs"])
def test_landed_save_applies_while_quit_question_is_cancelled(
    make_window, tmp_path, monkeypatch, setting
):
    window = make_window(injected=False)
    entered, release = threading.Event(), threading.Event()
    real_save = gui.save_app_config
    value = str(tmp_path / "new") if setting == "models_folder" else 3

    def delayed(*args):
        entered.set()
        assert release.wait(5)
        return real_save(*args)

    monkeypatch.setattr(gui, "save_app_config", delayed)
    effects = []
    real_apply = window._apply_saved_settings

    def apply(previous, current):
        effects.append(current)
        real_apply(previous, current)

    monkeypatch.setattr(window, "_apply_saved_settings", apply)
    window._save_config_candidate(
        replace(window.config, **{setting: value}), Message("settings.saveFailed"), lambda _: None
    )
    assert entered.wait(3)

    class Question:
        def __init__(self, parent):
            pass

        def choose(self):
            assert window._closing and not window._quit_bounded
            release.set()
            _settle_config(window)
            return "cancel"

    monkeypatch.setattr(gui, "QuitSaveFailedDialog", Question)
    try:
        window.closeEvent(QCloseEvent())
    finally:
        release.set()
    assert not window._closing
    assert getattr(window.config, setting) == value
    assert len(effects) == 1
    if setting == "models_folder":
        assert window.runtime_dirs.models_dir == Path(value)
        assert window.model_manager.models_dir == Path(value)
    assert not window._unsaved_config_changes


@pytest.mark.parametrize("failure_during_quit", [False, True])
def test_dismissed_folder_failure_is_retried_with_latest_parameters(
    make_window, tmp_path, monkeypatch, failure_during_quit
):
    window = make_window(injected=False)
    target = tmp_path / "chosen"
    entered, release = threading.Event(), threading.Event()
    real_save = gui.save_app_config
    allow_save = False
    attempts = []

    def save(candidate, previous, file):
        attempts.append(candidate)
        if len(attempts) == 1:
            entered.set()
            assert release.wait(5)
        if not allow_save:
            raise OSError("save denied")
        return real_save(candidate, previous, file)

    monkeypatch.setattr(gui, "save_app_config", save)
    window._managed_models_dialog()
    _settle_models(window)
    dialog = window._active_models_dialog
    dialog.folder_edit.setText(str(target))
    dialog.apply_folder_button.click()
    assert entered.wait(3)
    dialog.reject()
    QApplication.processEvents()
    if not failure_during_quit:
        release.set()
        _settle_config(window)
    window.quality.setValue(73)
    questions = []

    class Question:
        def __init__(self, parent):
            pass

        def choose(self):
            nonlocal allow_save
            questions.append(True)
            release.set()
            _settle_config(window)
            allow_save = True
            return "retry"

    monkeypatch.setattr(gui, "QuitSaveFailedDialog", Question)
    try:
        window.closeEvent(QCloseEvent())
        _settle_config(window)
    finally:
        release.set()
    assert questions == [True]
    assert load_app_config().models_folder == str(target)
    assert load_app_config().parameters.quality == 73
    assert window.runtime_dirs.models_dir == target
    assert not window._unsaved_config_changes
    assert attempts[-1].models_folder == str(target)


def test_os_takeover_saves_newer_parameters_before_declaring_workers_clean(
    make_window, monkeypatch, quit_bound
):
    window = make_window()
    entered, release = threading.Event(), threading.Event()
    real_save = gui.save_app_config
    qualities = []

    def save(candidate, previous, file):
        qualities.append(candidate.parameters.quality)
        if len(qualities) == 1:
            entered.set()
            assert release.wait(5)
        return real_save(candidate, previous, file)

    monkeypatch.setattr(gui, "save_app_config", save)
    window.quality.setValue(61)
    window._flush_parameters_save()
    assert entered.wait(3)
    window.quality.setValue(72)
    window._is_session_shutdown = lambda: True
    try:
        window.closeEvent(QCloseEvent())
        assert not window._workers_clean_for_quit()
    finally:
        release.set()
    _settle_config(window)
    assert qualities == [61, 72]
    assert load_app_config().parameters.quality == 72
    assert quit_bound.started == [gui.QUIT_BUDGET_S]


def test_clean_os_close_skips_optional_backup_drain(make_window, monkeypatch, quit_bound):
    window = make_window()
    window._is_session_shutdown = lambda: True
    window.closeEvent(QCloseEvent())
    assert window._os_close_accepted
    monkeypatch.setattr(gui, "build_app", lambda _: (SimpleNamespace(exec=lambda: 0), window))
    monkeypatch.setattr(gui, "drain_backups", lambda _: pytest.fail("OS quit drained backups"))
    monkeypatch.setattr(gui.logging, "shutdown", lambda: None)
    assert gui.main() == 0


def test_removed_image_drops_only_its_pending_retry_even_after_reopening(
    make_window, tmp_path, monkeypatch
):
    window = make_window()
    first, second = _png(tmp_path, "first.png"), _png(tmp_path, "second.png")
    _open(window, [first, second])
    model = "RealESRGAN_x4plus"
    window._enqueue_jobs([first, second], [model])
    _settle_models(window)
    for job in window.jobs:
        job.status = "failed"
    model_file(window.runtime_dirs.models_dir, model).unlink()
    window._retry_failed()
    _settle_models(window)
    monkeypatch.setattr(window, "_selected_path", lambda: first)
    window._remove_selected_image()
    _open(window, [first])
    model_file(window.runtime_dirs.models_dir, model).write_bytes(b"ready")
    window.model_manager.refresh_readiness()
    _settle_models(window)
    assert [job.status for job in window.jobs] == ["failed", "pending"]
    assert not window._pending_model_work


@pytest.mark.parametrize("kind", ["image_offer", "preview", "superseded_preview"])
def test_quit_retains_background_reads_until_physical_return(
    make_window, tmp_path, monkeypatch, process_until, quit_bound, kind
):
    window = make_window()
    entered, release = threading.Event(), threading.Event()
    if kind == "image_offer":
        real_probe = gui._probe_open_paths

        def probe(paths):
            entered.set()
            assert release.wait(5)
            return real_probe(paths)

        monkeypatch.setattr(gui, "_probe_open_paths", probe)
        window.open_paths([tmp_path / "missing.png"])
    else:
        image = _png(tmp_path, "held.png")
        real_read = gui.QImageReader.read

        def decode(reader):
            if Path(reader.fileName()) == image:
                entered.set()
                assert release.wait(5)
            return real_read(reader)

        monkeypatch.setattr(gui.QImageReader, "read", decode)
        window.preview.set_image(image)
        if kind == "superseded_preview":
            window.preview.set_image(None)
            assert not window.preview.loading
    assert entered.wait(3)
    window._is_session_shutdown = lambda: True
    try:
        window.closeEvent(QCloseEvent())
        assert not window._workers_clean_for_quit()
        assert quit_bound.started == [gui.QUIT_BUDGET_S]
        assert window._os_close_accepted
    finally:
        release.set()
    process_until(window._workers_clean_for_quit, timeout_s=5, what="Background read settlement")
    assert quit_bound.exits == 0


@pytest.mark.parametrize("settles", [True, False])
def test_native_process_quit_owns_background_reads_until_settlement_or_bound(settles):
    # Real Qt event loop and real Python thread in a disposable process. This
    # exercises process teardown without sending a logout to the user's OS.
    script = r'''
import os
import threading
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from pixelup import gui
app = QApplication([])
window = gui.MainWindow(log_file=Path(os.environ['PIXELUP_DATA_DIR']) / 'test.log')
entered, release = threading.Event(), threading.Event()
def held_probe(paths):
    entered.set()
    release.wait()
    return []
gui._probe_open_paths = held_probe
gui.QUIT_BUDGET_S = 0.25
window.open_paths([Path('held.png')])
assert entered.wait(3)
window._is_session_shutdown = lambda: True
window.close()
assert not window._workers_clean_for_quit()
print('read owned during quit', flush=True)
if SETTLES:
    QTimer.singleShot(20, release.set)
app.exec()
assert window._workers_clean_for_quit()
print('reads physically settled', flush=True)
'''.replace('SETTLES', repr(settles))
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert "read owned during quit" in result.stdout
    assert ("reads physically settled" in result.stdout) is settles


@pytest.mark.parametrize("read_kind", ["listing", "historical_resolve", "retry_listing"])
def test_output_planning_keeps_the_window_live_while_storage_is_held(
    make_window, tmp_path, monkeypatch, process_until, read_kind
):
    window = make_window()
    directory = tmp_path / "inputs"
    directory.mkdir()
    image = _png(directory, "image.png")
    _open(window, [image])
    model = "RealESRGAN_x4plus"
    if read_kind == "retry_listing":
        window._enqueue_jobs([image], [model])
        _settle_models(window)
        window.jobs[0].status = "failed"
    historical = Job(
        id=9000, input_path=tmp_path / "old-volume" / "old.png", model=model,
        output_path=tmp_path / "old-volume" / "old-output.png", settings=JobSettings(),
        status="succeeded",
    )
    if read_kind == "historical_resolve":
        window.jobs.append(historical)
    entered, release = threading.Event(), threading.Event()
    readers = []
    main_thread = threading.current_thread()
    real_iterdir, real_resolve = Path.iterdir, Path.resolve

    def hold():
        readers.append(threading.current_thread())
        entered.set()
        # Preserve a deterministic baseline failure without deadlocking the test
        # when the actual filesystem read incorrectly runs on the window thread.
        if threading.current_thread() is not main_thread:
            assert release.wait(5)

    def iterdir(path):
        if path == directory:
            hold()
        return real_iterdir(path)

    def resolve(path, *args, **kwargs):
        if path == historical.output_path:
            hold()
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve" if read_kind == "historical_resolve" else "iterdir",
                        resolve if read_kind == "historical_resolve" else iterdir)
    try:
        if read_kind == "retry_listing":
            window._retry_failed()
        else:
            window._enqueue_jobs([image], [model])
        process_until(entered.is_set, timeout_s=5, what="Held planning filesystem read")
        assert readers[0] is not main_thread, "Output planning read storage on the window thread"
        tick = threading.Event()
        QTimer.singleShot(0, tick.set)
        process_until(tick.is_set, timeout_s=5, what="Window callback during held planning")
        if read_kind == "retry_listing":
            assert window.jobs[0].status == "failed"
        else:
            assert all(job.input_path != image for job in window.jobs)
    finally:
        release.set()
    process_until(
        lambda: any(job.input_path == image and job.status == "pending" for job in window.jobs),
        timeout_s=5, what="Adopted output plan",
    )


def test_failed_ready_request_does_not_drop_an_unrelated_ready_request(
    make_window, tmp_path, monkeypatch, process_until, caplog
):
    window = make_window()
    first, second = _png(tmp_path, "first.png"), _png(tmp_path, "second.png")
    _open(window, [first, second])
    monkeypatch.setattr(window.model_manager, "refresh_readiness", lambda: None)
    real_create = gui.create_jobs

    def create(**kwargs):
        if kwargs["input_paths"] == [first]:
            raise OSError("EACCES /private/PLANNING-SENTINEL")
        return real_create(**kwargs)

    monkeypatch.setattr(gui, "create_jobs", create)
    window._enqueue_jobs([first], ["RealESRGAN_x4plus"])
    window._enqueue_jobs([second], ["RealESRGAN_x4plus"])
    window._model_manager_changed()
    process_until(lambda: len(window.jobs) == 1, timeout_s=5, what="Unrelated ready request")
    assert window.jobs[0].input_path == second
    assert "PLANNING-SENTINEL" not in window.queue_action_result.message_label.text()
    assert "could not be queued" in window.queue_action_result.message_label.text()
    failures = [
        record for record in caplog.records if record.getMessage() == "enqueue.planning_failed"
    ]
    assert len(failures) == 1
    assert failures[0].exc_info is not None
    assert "PLANNING-SENTINEL" in str(failures[0].exc_info[1])


@pytest.fixture
def held_output_listing(monkeypatch):
    holds = []
    real_iterdir = Path.iterdir

    def hold(directory):
        entered, release = threading.Event(), threading.Event()
        holds.append((directory, entered, release))
        return entered, release

    def iterdir(path):
        for directory, entered, release in holds:
            if path == directory:
                entered.set()
                assert release.wait(5)
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    yield hold
    for _directory, _entered, release in holds:
        release.set()


def test_serial_planning_retains_settings_folder_and_bundle_reservations(
    make_window, tmp_path, monkeypatch, process_until, held_output_listing
):
    window = make_window(injected=False)
    image = _png(tmp_path, "image.png")
    _open(window, [image])
    accepted_dirs = window.runtime_dirs
    settings = JobSettings(scale=2, quality=70)
    monkeypatch.setattr(window, "current_job_settings", lambda: settings)
    monkeypatch.setattr(window.model_manager, "refresh_readiness", lambda: None)
    entered, release = held_output_listing(tmp_path)
    model = "RealESRGAN_x4plus"
    window._enqueue_jobs([image], [model])
    window._enqueue_jobs([image], [model])
    window._model_manager_changed()
    process_until(entered.is_set, timeout_s=5, what="First serialized plan")
    assert len(window._pending_job_plans) == 1
    settings = JobSettings(scale=4, quality=95)
    window._apply_saved_settings(
        window.config, replace(window.config, models_folder=str(tmp_path / "new-models")),
    )
    assert window.runtime_dirs.models_dir != accepted_dirs.models_dir
    release.set()
    process_until(lambda: len(window.jobs) == 2, timeout_s=5, what="Both serialized plans")
    assert [job.output_path.name for job in window.jobs] == [
        "image-realesrgan-x4plus-2x.png", "image-realesrgan-x4plus-2x-2.png",
    ]
    assert all(job.settings == JobSettings(scale=2, quality=70, denoise_strength=1.0)
               for job in window.jobs)
    assert all(window.runner._queued_runtime_dirs[job.id] == accepted_dirs for job in window.jobs)


@pytest.mark.parametrize("retry", [False, True])
def test_held_plan_cannot_revive_a_removed_and_reopened_image(
    make_window, tmp_path, monkeypatch, process_until, held_output_listing, retry
):
    window = make_window()
    first, second = _png(tmp_path, "first.png"), _png(tmp_path, "second.png")
    _open(window, [first, second])
    model = "RealESRGAN_x4plus"
    if retry:
        window._enqueue_jobs([first, second], [model])
        _settle_models(window)
        for job in window.jobs:
            job.status = "failed"
    old_entry = window._images_by_path[first]
    entered, release = held_output_listing(tmp_path)
    if retry:
        window._retry_failed()
    else:
        window._enqueue_jobs([first, second], [model])
    process_until(entered.is_set, timeout_s=5, what="Plan before image removal")
    monkeypatch.setattr(window, "_selected_path", lambda: first)
    window._remove_selected_image()
    _open(window, [first])
    assert window._images_by_path[first] is not old_entry
    release.set()
    _settle_models(window)
    if retry:
        assert [(job.input_path, job.status) for job in window.jobs] == [
            (first, "failed"), (second, "pending"),
        ]
    else:
        assert [job.input_path for job in window.jobs] == [second]


def test_queue_cancel_abandons_plans_and_retains_the_physical_read(
    make_window, tmp_path, monkeypatch, process_until, held_output_listing
):
    window = make_window()
    image = _png(tmp_path, "image.png")
    _open(window, [image])
    monkeypatch.setattr(window.model_manager, "refresh_readiness", lambda: None)
    entered, release = held_output_listing(tmp_path)
    window._enqueue_jobs([image], ["RealESRGAN_x4plus"])
    window._enqueue_jobs([image], ["RealESRGAN_x4plus"])
    window._model_manager_changed()
    process_until(entered.is_set, timeout_s=5, what="Plan before queue cancellation")
    assert window.cancel_button.isEnabled()
    window.cancel_button.click()
    assert not window.cancel_button.isEnabled()
    assert not window._planning_reads.settled
    assert not window._pending_job_plans
    release.set()
    process_until(
        lambda: window._planning_reads.settled, timeout_s=5, what="Cancelled physical read",
    )
    QApplication.processEvents()
    assert window.jobs == []


def test_quit_abandons_unstarted_plans_and_owns_the_held_read(
    make_window, tmp_path, monkeypatch, process_until, held_output_listing, quit_bound
):
    window = make_window()
    image = _png(tmp_path, "image.png")
    _open(window, [image])
    monkeypatch.setattr(window.model_manager, "refresh_readiness", lambda: None)
    entered, release = held_output_listing(tmp_path)
    window._enqueue_jobs([image], ["RealESRGAN_x4plus"])
    window._enqueue_jobs([image], ["RealESRGAN_x4plus"])
    window._model_manager_changed()
    process_until(entered.is_set, timeout_s=5, what="Plan before quit")
    window._is_session_shutdown = lambda: True
    window.closeEvent(QCloseEvent())
    assert not window._pending_job_plans
    assert not window._workers_clean_for_quit()
    assert quit_bound.started == [gui.QUIT_BUDGET_S]
    release.set()
    process_until(window._workers_clean_for_quit, timeout_s=5, what="Planning quit settlement")
    assert window.jobs == []
    assert quit_bound.exits == 0


def test_planning_thread_start_failure_settles_one_request_and_continues(
    make_window, tmp_path, monkeypatch, process_until, caplog
):
    window = make_window()
    first, second = _png(tmp_path, "first.png"), _png(tmp_path, "second.png")
    _open(window, [first, second])
    monkeypatch.setattr(window.model_manager, "refresh_readiness", lambda: None)
    real_start = threading.Thread.start
    rejected = []

    def start(thread):
        if thread.name == "pixelup-plan" and not rejected:
            rejected.append(thread)
            raise RuntimeError("THREAD-START-SENTINEL")
        return real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    window._enqueue_jobs([first], ["RealESRGAN_x4plus"])
    window._enqueue_jobs([second], ["RealESRGAN_x4plus"])
    window._model_manager_changed()
    process_until(
        lambda: len(window.jobs) == 1, timeout_s=5, what="Plan after thread start failure",
    )
    assert window.jobs[0].input_path == second
    assert "THREAD-START-SENTINEL" not in window.queue_action_result.message_label.text()
    assert "See Records" in window.queue_action_result.message_label.text()
    failures = [
        record for record in caplog.records if record.getMessage() == "enqueue.planning_failed"
    ]
    assert len(failures) == 1
    assert failures[0].exc_info[0] is RuntimeError
    assert "THREAD-START-SENTINEL" in str(failures[0].exc_info[1])


def test_historical_retry_is_not_revived_by_opening_and_removing_its_image(
    make_window, tmp_path, monkeypatch, process_until, held_output_listing
):
    window = make_window()
    image = _png(tmp_path, "historical.png")
    _open(window, [image])
    window._enqueue_jobs([image], ["RealESRGAN_x4plus"])
    _settle_models(window)
    [job] = window.jobs
    job.status = "failed"
    monkeypatch.setattr(window, "_selected_path", lambda: image)
    window._remove_selected_image()
    assert image not in window._images_by_path
    entered, release = held_output_listing(tmp_path)
    window._retry_failed()
    process_until(entered.is_set, timeout_s=5, what="Historical retry planning")
    _open(window, [image])
    window._remove_selected_image()
    release.set()
    _settle_models(window)
    assert job.status == "failed"


def test_historical_retry_survives_opening_its_image_during_planning(
    make_window, tmp_path, monkeypatch, process_until, held_output_listing
):
    window = make_window()
    image = _png(tmp_path, "historical.png")
    _open(window, [image])
    window._enqueue_jobs([image], ["RealESRGAN_x4plus"])
    _settle_models(window)
    [job] = window.jobs
    job.status = "failed"
    monkeypatch.setattr(window, "_selected_path", lambda: image)
    window._remove_selected_image()
    entered, release = held_output_listing(tmp_path)
    window._retry_failed()
    process_until(entered.is_set, timeout_s=5, what="Historical retry before image opening")
    _open(window, [image])
    release.set()
    _settle_models(window)
    assert job.status == "pending"


def test_queued_historical_retry_is_cancelled_by_opening_and_removing_its_image(
    make_window, tmp_path, monkeypatch, process_until, held_output_listing
):
    window = make_window()
    first_dir, historical_dir = tmp_path / "first", tmp_path / "historical"
    first_dir.mkdir()
    historical_dir.mkdir()
    first = _png(first_dir, "first.png")
    historical = _png(historical_dir, "historical.png")
    _open(window, [first, historical])
    model = "RealESRGAN_x4plus"
    window._enqueue_jobs([historical], [model])
    _settle_models(window)
    [job] = window.jobs
    job.status = "failed"
    monkeypatch.setattr(window, "_selected_path", lambda: historical)
    window._remove_selected_image()
    monkeypatch.setattr(window.model_manager, "refresh_readiness", lambda: None)
    entered, release = held_output_listing(first_dir)
    window._enqueue_jobs([first], [model])
    window._retry_failed()
    window._model_manager_changed()
    process_until(entered.is_set, timeout_s=5, what="Plan ahead of historical retry")
    assert window._pending_job_plans[0].images == ((historical, None),)
    _open(window, [historical])
    window._remove_selected_image()
    release.set()
    _settle_models(window)
    assert job.status == "failed"
    assert [(item.input_path, item.status) for item in window.jobs] == [
        (historical, "failed"), (first, "pending"),
    ]


def test_held_retry_limits_adoption_to_the_disclosed_failed_jobs(
    make_window, tmp_path, process_until, held_output_listing
):
    window = make_window()
    first, second = _png(tmp_path, "first.png"), _png(tmp_path, "second.png")
    _open(window, [first, second])
    window._enqueue_jobs([first, second], ["RealESRGAN_x4plus"])
    _settle_models(window)
    window.jobs[0].status = "failed"
    entered, release = held_output_listing(tmp_path)
    window._retry_failed()
    process_until(entered.is_set, timeout_s=5, what="Disclosed retry planning")
    window.jobs[1].status = "failed"
    release.set()
    _settle_models(window)
    assert [job.status for job in window.jobs] == ["pending", "failed"]


@pytest.mark.parametrize("retain_other_image", [False, True])
def test_planning_failure_is_presented_only_for_retained_input_intent(
    make_window, tmp_path, monkeypatch, process_until, held_output_listing, retain_other_image
):
    window = make_window()
    first, second = _png(tmp_path, "first.png"), _png(tmp_path, "second.png")
    _open(window, [first, second])
    entered, release = held_output_listing(tmp_path)
    real_create = gui.create_jobs

    def create(**kwargs):
        real_create(**kwargs)
        raise OSError("STALE-PLAN-SENTINEL")

    monkeypatch.setattr(gui, "create_jobs", create)
    window._enqueue_jobs([first, second] if retain_other_image else [first],
                         ["RealESRGAN_x4plus"])
    process_until(entered.is_set, timeout_s=5, what="Plan before failure and removal")
    monkeypatch.setattr(window, "_selected_path", lambda: first)
    window._remove_selected_image()
    release.set()
    _settle_models(window)
    assert window.jobs == []
    assert window.queue_action_result.isHidden() is not retain_other_image
    assert not window._removed_plan_inputs


def test_selected_rows_show_their_complete_values_by_keyboard(make_window, tmp_path) -> None:
    # Cells elide long names; the current row's detail, for whichever table was
    # last entered, is the keyboard and touch path to the whole value
    # (text-cleanup-conventions, rule 4), not a hover tooltip.
    from PySide6.QtTest import QTest

    window = make_window()
    folder = tmp_path / ("a-folder-name-long-enough-to-elide-in-any-column " * 3).strip()
    folder.mkdir()
    first = _png(folder, "first.png")
    second = _png(folder, "second-" + "x" * 120 + ".png")
    _open(window, [first, second])
    window.image_table.setFocus()
    window.image_table.setCurrentCell(0, 0)
    QTest.keyClick(window.image_table, Qt.Key.Key_Down)
    _settle_opens(window)
    detail = window.row_detail
    assert detail.rows() == (("Image", str(second)), ("Jobs", "No jobs"))
    assert detail.accessibleName() == "Selected image details"

    window.model_checks["realesr-general-x4v3"].setChecked(True)
    window._queue_selected_image()
    _settle_models(window)
    job = window.jobs[0]
    assert detail.rows() == (("Image", str(second)), ("Jobs", "1 queued"))
    window.queue_table.setFocus()
    QTest.keyClick(window.queue_table, Qt.Key.Key_Down)
    assert window.queue_table.currentRow() == 0
    assert detail.rows() == (
        ("Image", str(second)),
        ("Output", str(job.output_path)),
        ("Status", window.queue_table.item(0, 4).text()),
    )
    assert detail.accessibleName() == "Selected job details"

    window._job_finished(job.id, False, FAILED, {}, [DISK_WARNING])
    assert detail.rows()[2] == (
        "Status",
        f"{english().of(FAILED)}\n{english().of(DISK_WARNING)}",
    )

    # Entering the image table again shows its current row without moving it.
    window._focus_moved(window.queue_table, window.image_table)
    assert detail.rows() == (("Image", str(second)), ("Jobs", "1 failed"))


def test_row_detail_wraps_an_unbroken_value_instead_of_cutting_it(qapp) -> None:
    from pixelup.widgets import RowDetail

    detail = RowDetail(2, accessible_name="images.selectedDetail")
    detail.resize(240, detail.height())
    detail.show()
    value = "/" + "x" * 400 + ".png"
    detail.set_rows((("Image", value),))
    QApplication.processEvents()
    assert detail.horizontalScrollBar().maximum() == 0
    assert value in detail.toPlainText().replace("\n", "").replace(" ", "")
    assert detail.verticalScrollBar().maximum() > 0
    detail.close()

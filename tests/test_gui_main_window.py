from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from PySide6.QtCore import QUrl
from PySide6.QtGui import QCloseEvent, QKeySequence
from PySide6.QtWidgets import QApplication, QCheckBox, QPushButton

from pixelup import gui
from pixelup.app_config import AppConfig, ConfigLoadResult, config_path, load_app_config
from pixelup.errors import ErrorCode
from pixelup.gui import MainWindow
from pixelup.jobs import JobSettings
from pixelup.parameters import DEFAULT_SCALE, TILE_VALUES
from pixelup.paths import absolute_user_path
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
    # rather than the developer's real ~/.pixelup/config.json. PIXELUP_HOME is
    # redirected as well as the load stubbed, because the window now *writes*:
    # ensure_app_config() and the Parameters panel's save both resolve config.json
    # through the storage root, and neither may land in the real one.
    monkeypatch.setenv("PIXELUP_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(JobRunner, "schedule", lambda self, max_concurrent_jobs: None)
    monkeypatch.setattr("pixelup.gui.load_app_config_result", lambda: ConfigLoadResult(AppConfig()))
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
    row = window._image_rows[absolute_user_path(path)]
    return window.image_table.item(row, 2).text()


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


def test_build_app_wires_application_and_opens_argv_paths(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # build_app is everything main() does except the blocking app.exec(); driving it headlessly
    # is the whole point of extracting it (main() is then a 3-line untestable shell). log_file and
    # runtime_dirs are injected at a temp location so nothing touches the real ~/.pixelup.
    monkeypatch.setenv("PIXELUP_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("pixelup.gui.load_app_config_result", lambda: ConfigLoadResult(AppConfig()))
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


def test_open_paths_preserves_literal_symlink_for_display_and_default_output(
    make_window, tmp_path: Path
) -> None:
    window = make_window()
    source_dir = tmp_path / "source"
    chosen_dir = tmp_path / "chosen"
    source_dir.mkdir()
    chosen_dir.mkdir()
    target = _png(source_dir, "target.png")
    chosen = chosen_dir / "dropped-link.png"
    chosen.symlink_to(target)

    window.open_paths([chosen])

    assert window._selected_path() == chosen
    assert window.image_table.item(0, 0).text() == "dropped-link.png"
    assert window.image_table.item(0, 0).toolTip() == str(chosen)

    window.model_checks["realesr-general-x4v3"].setChecked(True)
    window._queue_selected_image()

    assert window.jobs[0].input_path == chosen
    assert window.jobs[0].output_path.parent == chosen_dir


def test_external_drop_accepts_local_files_and_rejects_remote_urls(
    make_window, tmp_path: Path
) -> None:
    window = make_window()
    local = _png(tmp_path, "local.png")
    remote_url = QUrl("https://example.com/remote.png")
    directory_url = QUrl.fromLocalFile(str(tmp_path))

    remote_drag = _DropEvent([remote_url, directory_url])
    window.dragEnterEvent(remote_drag)  # type: ignore[arg-type]
    assert remote_drag.ignored is True
    assert remote_drag.accepted is False

    remote_drop = _DropEvent([remote_url, directory_url])
    window.dropEvent(remote_drop)  # type: ignore[arg-type]
    assert remote_drop.ignored is True
    assert window.image_table.rowCount() == 0

    local_drop = _DropEvent([remote_url, QUrl.fromLocalFile(str(local))])
    window.dragEnterEvent(local_drop)  # type: ignore[arg-type]
    window.dropEvent(local_drop)  # type: ignore[arg-type]
    assert local_drop.accepted is True
    assert window._selected_path() == local


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


def test_settings_only_options_have_no_main_window_control(make_window) -> None:
    # One home per thing, checked from the other side. auto_download and
    # max_concurrent_jobs are the settings dialog's; a control for either here would
    # rebuild the exact half-persisted/half-transient split this design removed. The
    # Models group is model *selection* only — no download toggle rides along in it.
    window = make_window()

    assert set(window.model_checks) == set(gui.UPSCALE_MODELS)
    checkbox_labels = {box.text() for box in window.findChildren(QCheckBox)}
    assert checkbox_labels == set(gui.UPSCALE_MODELS) | {"Face enhancement", "Strip metadata"}
    # The window reads both from config and never offers a widget onto them.
    assert not hasattr(window, "auto_download")
    assert not hasattr(window, "concurrent")


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

    def _timeout(path: Path) -> bool:
        raise gui.subprocess.TimeoutExpired(["open", "-R", str(path)], 5)

    monkeypatch.setattr("pixelup.gui._reveal_in_file_browser", _timeout)
    window._reveal_log_file()


def test_reveal_in_file_browser_darwin_reports_returncode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = _png(tmp_path, "a.png")
    monkeypatch.setattr(sys, "platform", "darwin")

    calls: list[dict[str, object]] = []

    def _run(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(gui.subprocess, "run", _run)
    assert gui._reveal_in_file_browser(target) is True
    assert calls == [{"check": False, "timeout": gui._REVEAL_TIMEOUT_SECONDS}]

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


def _reset_button(window: MainWindow) -> QPushButton:
    return next(
        button
        for button in window.findChildren(QPushButton)
        if button.text() == "Reset parameters"
    )


def _wander_from_defaults(window: MainWindow) -> None:
    window.scale_buttons[2].setChecked(True)
    window.face_enhance.setChecked(True)
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


def test_reset_parameters_button_restores_the_built_in_scale(make_window) -> None:
    # Scale is a panel parameter like any other now, so the real button restores it
    # too — the holdout that used to sit outside both persistence and reset. Driven by
    # the button's label text, and asserted on the radios themselves, so the widget
    # the user actually looks at is what got reset.
    window = make_window()
    window.scale_buttons[2].setChecked(True)
    window._flush_parameters_save()
    # Persisted first, deliberately: with 2x saved, a reset that reached for the user's
    # config instead of the built-ins would land back on 2x and still look like it had
    # worked. Only a persisted-then-reset run can tell the two apart.
    assert window.config.parameters.scale == 2
    assert window.current_job_settings().scale == 2

    _reset_button(window).click()

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
    window._flush_parameters_save()
    assert window.config.parameters.quality == 10

    _reset_button(window).click()

    assert window.current_job_settings() == JobSettings()
    # And the reset is itself persisted, so it survives the next launch.
    assert window.config.parameters == JobSettings()


def test_parameter_edits_persist_to_config_json(make_window) -> None:
    # The panel is durable: an edit reaches config.json, so a relaunch reads it back.
    # Loaded from disk (not from window.config) so the serializer is in the loop.
    window = make_window()

    _wander_from_defaults(window)
    window._flush_parameters_save()

    persisted = load_app_config(config_path()).parameters
    assert persisted == window.current_job_settings()
    assert persisted.quality == 10
    assert persisted.tile == 1024
    assert persisted.face_enhance is True
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
    warnings: list[object] = []

    def _save(candidate: AppConfig) -> None:
        attempts.append(candidate)
        if len(attempts) == 1:
            raise OSError("disk full")

    monkeypatch.setattr("pixelup.gui.save_app_config", _save)
    monkeypatch.setattr("pixelup.gui.warn_config_save_failed", warnings.append)

    assert window._flush_parameters_save() is False
    assert window.config is original
    assert window.current_job_settings().quality == 10
    assert warnings == [window]

    assert window._flush_parameters_save() is True
    assert window.config.parameters.quality == 10
    assert len(attempts) == 2


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
    assert window._quit_when_workers_idle is True
    assert shutdown_calls == [True]
    assert warnings == [window]

    second = QCloseEvent()
    window.closeEvent(second)
    assert second.isAccepted() is True


def test_main_surfaces_startup_storage_failure(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shown: list[tuple[str, str | None]] = []
    error = gui.PixelupError(
        ErrorCode.OUTPUT_UNWRITABLE,
        "Could not open PixelUp home.",
        hint="Choose a writable PIXELUP_HOME.",
    )
    monkeypatch.setattr("pixelup.gui.build_app", lambda argv: (_ for _ in ()).throw(error))
    monkeypatch.setattr(
        "pixelup.gui.show_startup_failure",
        lambda detail, hint: shown.append((detail, hint)),
    )

    assert gui.main() == 1
    assert shown == [("Could not open PixelUp home.", "Choose a writable PIXELUP_HOME.")]


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

    window._flush_parameters_save()

    assert load_app_config(config_path()).parameters.scale == 2


def test_window_seeds_the_panel_from_the_persisted_parameters(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Startup reads the persisted panel, not a defaults factory: whatever the user
    # left behind is what the panel opens on.
    monkeypatch.setenv("PIXELUP_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(JobRunner, "schedule", lambda self, max_concurrent_jobs: None)
    parameters = JobSettings(
        scale=2, quality=42, tile=512, alpha_mode="bicubic", target_profile="p3"
    )
    monkeypatch.setattr(
        "pixelup.gui.load_app_config_result",
        lambda: ConfigLoadResult(AppConfig(parameters=parameters)),
    )
    log_file = tmp_path / "logs" / "session.log"
    configure_session_logging(log_file)

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
        window._session_shutdown = True
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_queued_jobs_keep_their_snapshot_when_the_panel_changes(
    make_window, tmp_path: Path
) -> None:
    # Persisting the panel must not reach backwards into work already queued: each job
    # captures the panel at creation and holds that snapshot, so editing the panel
    # afterwards changes the next job, never the queued one.
    window = make_window()
    window.open_paths([_png(tmp_path, "a.png")])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    window.tile.setCurrentIndex(window.tile.findData(128))
    window._queue_selected_image()
    queued = window.jobs[0]
    assert queued.settings.tile == 128

    window.tile.setCurrentIndex(window.tile.findData(1024))
    window._flush_parameters_save()

    assert queued.settings.tile == 128
    assert window.config.parameters.tile == 1024
    window._queue_selected_image()
    assert window.jobs[1].settings.tile == 1024


def test_queued_jobs_keep_the_scale_they_were_enqueued_with(make_window, tmp_path: Path) -> None:
    # The enqueue snapshot, through the real window, for the field that just joined it.
    # Scale is now persisted and resettable, so it is exactly the field that could
    # start leaking backwards into queued work — a job must keep the scale the panel
    # had when the user pressed Queue, in its settings, its planned output name, and
    # the row the queue table shows.
    window = make_window()
    window.open_paths([_png(tmp_path, "a.png")])
    window.model_checks["realesr-general-x4v3"].setChecked(True)
    window.scale_buttons[2].click()
    window._queue_selected_image()
    queued = window.jobs[0]

    assert queued.settings.scale == 2
    assert queued.output_path.name == "a-realesr-general-x4v3-2x.png"
    assert window.queue_table.item(0, 2).text() == "2x"

    # The panel moves on, and the reset button is the most forceful way it can.
    _reset_button(window).click()

    assert queued.settings.scale == 2
    assert queued.output_path.name == "a-realesr-general-x4v3-2x.png"
    assert window.queue_table.item(0, 2).text() == "2x"
    assert window.config.parameters.scale == DEFAULT_SCALE

    # The next job takes the panel as it now stands.
    window._queue_selected_image()
    assert window.jobs[1].settings.scale == DEFAULT_SCALE
    assert window.queue_table.item(1, 2).text() == "4x"


def test_a_stray_persisted_tile_is_quarantined_before_the_panel_is_seeded(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The panel's combo seeds via findData, which answers -1 for a value it does not
    # hold — and setCurrentIndex(-1) blanks the box, making currentData() None. That
    # None would flow straight back into JobSettings.tile on the next enqueue or save.
    #
    # Nothing in the panel prevents it, so this walks the real loader (not an injected
    # AppConfig) with a tile that is not a choice and pins the whole recovery chain:
    # invalid store -> quarantine/reset -> a panel that shows a real default value.
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.json").write_text(
        json.dumps({"max_concurrent_jobs": 1, "auto_download": True,
                    "font_family": "", "parameters": {"tile": 9999}})
    )
    monkeypatch.setenv("PIXELUP_HOME", str(home))
    monkeypatch.setattr(JobRunner, "schedule", lambda self, max_concurrent_jobs: None)
    notices: list[str] = []
    monkeypatch.setattr(
        "pixelup.gui.warn_config_reset",
        lambda _parent, name: notices.append(name),
    )
    log_file = tmp_path / "logs" / "session.log"
    configure_session_logging(log_file)

    window = MainWindow(log_file=log_file)
    try:
        assert window._config_quarantined_to is not None
        assert window.tile.currentIndex() != -1, "the combo blanked on an unknown tile"
        assert window.tile.currentData() is not None
        assert window.tile.currentData() == JobSettings().tile
        # And the value the panel would hand a job is a real one, not None.
        assert window.current_job_settings().tile in TILE_VALUES
        qapp.processEvents()
        assert notices == [window._config_quarantined_to.name]
    finally:
        window._session_shutdown = True
        window.close()
        window.deleteLater()
        qapp.processEvents()

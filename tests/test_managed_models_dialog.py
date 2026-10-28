from __future__ import annotations

import threading
from pathlib import Path

import pytest
from event_wait import process_until
from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QApplication, QDialog, QLabel

from pixelup.dialog_shell import TABLE_WIDTH
from pixelup.errors import ErrorCode, PixelupError
from pixelup.i18n.localizer import english
from pixelup.i18n.message import Message
from pixelup.managed_models_dialog import ManagedModelsDialog
from pixelup.model_management import (
    MANAGED_ARTIFACT_NAMES,
    MANAGED_MODEL_BUNDLES,
    artifact_size_bytes,
)
from pixelup.model_manager import ModelManager
from pixelup.models import model_file


def _ready(manager: ModelManager, qapp: QApplication) -> None:
    process_until(lambda: manager.readiness_known, what="Model readiness")


def _finish_manager(manager: ModelManager, qapp: QApplication) -> None:
    process_until(manager.cleanup_for_quit, what="Installs settle")
    qapp.processEvents()
    assert manager.cleanup_for_quit()
    _ready(manager, qapp)


def test_manual_install_updates_application_owned_readiness(
    qapp: QApplication, tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[str, bool]] = []

    def install(models_dir: Path, name: str, **kwargs: object) -> dict[str, object]:
        calls.append((name, bool(kwargs["force"])))
        target = model_file(models_dir, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"model")
        callback = kwargs["on_download"]
        callback(name, 5, 5)  # type: ignore[operator]
        return {"status": "downloaded"}

    monkeypatch.setattr("pixelup.model_manager.download_model", install)
    manager = ModelManager(tmp_path)
    _ready(manager, qapp)
    dialog = ManagedModelsDialog(manager)
    try:
        assert dialog.row_action_buttons[0].text() == "Install"

        dialog._install_bundle(0)
        _finish_manager(manager, qapp)

        bundle = MANAGED_MODEL_BUNDLES[0]
        assert calls == [(bundle.artifact_names[0], False)]
        assert dialog.status_labels[0].text() == "Installed"
        assert dialog.row_action_buttons[0].text() == "Reinstall"
    finally:
        dialog.deleteLater()


def test_reinstall_forces_atomic_reacquisition(
    qapp: QApplication, tmp_path: Path, monkeypatch
) -> None:
    bundle = MANAGED_MODEL_BUNDLES[0]
    path = model_file(tmp_path, bundle.artifact_names[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"old")
    forces: list[bool] = []

    def install(_models_dir: Path, _name: str, **kwargs: object) -> dict[str, object]:
        forces.append(bool(kwargs["force"]))
        return {"status": "downloaded"}

    monkeypatch.setattr("pixelup.model_manager.download_model", install)
    manager = ModelManager(tmp_path)
    _ready(manager, qapp)
    dialog = ManagedModelsDialog(manager)
    try:
        assert dialog.row_action_buttons[0].text() == "Reinstall"
        dialog._install_bundle(0)
        _finish_manager(manager, qapp)
        assert forces == [True]
    finally:
        dialog.deleteLater()


def test_queue_install_only_changes_manager_state(
    qapp: QApplication, tmp_path: Path, monkeypatch
) -> None:
    required = ("RealESRGAN_x4plus", "RealESRNet_x4plus")

    def install(models_dir: Path, name: str, **_kwargs: object) -> dict[str, object]:
        target = model_file(models_dir, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"model")
        return {"status": "downloaded"}

    monkeypatch.setattr("pixelup.model_manager.download_model", install)
    manager = ModelManager(tmp_path)
    _ready(manager, qapp)
    dialog = ManagedModelsDialog(
        manager,
        required_artifacts=required,
        pending_job_count=12,
    )
    try:
        assert dialog.primary_button.text() == "Install and queue 12 jobs"
        assert all(not button.isEnabled() for button in dialog.row_action_buttons)
        assert "No jobs will be created" in dialog.summary_label.text()
        required_statuses = [
            label
            for bundle, label in zip(MANAGED_MODEL_BUNDLES, dialog.status_labels, strict=True)
            if set(required).intersection(bundle.artifact_names)
        ]
        assert required_statuses
        assert all("Required" not in label.text() for label in required_statuses)
        assert all(label.property("severity") == "warning" for label in required_statuses)
        assert all(label.styleSheet() == "" for label in required_statuses)
        dialog._install_all_or_cancel()
        _finish_manager(manager, qapp)

        assert dialog.result() == QDialog.DialogCode.Rejected
        assert manager.missing(required) == ()
        assert dialog.primary_button.isEnabled() is False
    finally:
        dialog.deleteLater()


def test_install_failure_remains_visible_and_retryable(
    qapp: QApplication, tmp_path: Path, monkeypatch
) -> None:
    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise PixelupError(
            ErrorCode.MODEL_DOWNLOAD_FAILED,
            Message.of("error.modelDownloadFailed", model="RealESRGAN_x4plus"),
        )

    monkeypatch.setattr("pixelup.model_manager.download_model", fail)
    manager = ModelManager(tmp_path)
    _ready(manager, qapp)
    dialog = ManagedModelsDialog(manager)
    try:
        dialog._install_bundle(0)
        _finish_manager(manager, qapp)
        assert manager.failed_operations[0].kind == "failed"
        assert dialog.result_view.isVisibleTo(dialog)
        failure = "Could not download model 'RealESRGAN_x4plus'."
        assert dialog.result_view.message_label.text() == failure
        assert dialog.result_view.accessibleName() == failure
        assert dialog.row_action_buttons[0].isEnabled()
    finally:
        dialog.deleteLater()


def test_unexpected_install_failure_uses_safe_copy(
    qapp: QApplication, tmp_path: Path, monkeypatch
) -> None:
    diagnostic_sentinel = "EACCES Error invoking worker /private/var/tmp/pixelup-model"

    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise OSError(diagnostic_sentinel)

    monkeypatch.setattr("pixelup.model_manager.download_model", fail)
    manager = ModelManager(tmp_path)
    _ready(manager, qapp)
    dialog = ManagedModelsDialog(manager)
    try:
        dialog._install_bundle(0)
        _finish_manager(manager, qapp)

        assert dialog.result_view.message_label.text() == (
            "The models could not be installed. Try again."
        )
        assert diagnostic_sentinel not in dialog.result_view.message_label.text()
    finally:
        dialog.deleteLater()


def test_failure_refreshes_partial_application_owned_readiness(
    qapp: QApplication, tmp_path: Path, monkeypatch
) -> None:
    required = ("RealESRGAN_x4plus", "RealESRNet_x4plus")

    def install(models_dir: Path, name: str, **_kwargs: object) -> dict[str, object]:
        if name == required[1]:
            raise PixelupError(
                ErrorCode.MODEL_DOWNLOAD_FAILED,
                Message.of("error.modelDownloadFailed", model=name),
            )
        target = model_file(models_dir, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"model")
        return {"status": "downloaded"}

    monkeypatch.setattr("pixelup.model_manager.download_model", install)
    manager = ModelManager(tmp_path)
    _ready(manager, qapp)
    assert manager.install(required, force=False)
    _finish_manager(manager, qapp)

    assert manager.failed_operations[0].kind == "failed"
    assert required[0] in manager.ready_names
    assert manager.missing(required) == (required[1],)


def test_closing_dialog_does_not_cancel_application_owned_install(
    qapp: QApplication, tmp_path: Path, monkeypatch
) -> None:
    started = threading.Event()
    release = threading.Event()

    def wait_then_install(models_dir: Path, name: str, **kwargs: object) -> dict[str, object]:
        started.set()
        assert release.wait()
        assert kwargs["should_cancel"]() is False  # type: ignore[operator]
        target = model_file(models_dir, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"model")
        return {"status": "downloaded"}

    monkeypatch.setattr("pixelup.model_manager.download_model", wait_then_install)
    manager = ModelManager(tmp_path)
    _ready(manager, qapp)
    dialog = ManagedModelsDialog(manager)
    dialog._install_bundle(0)
    assert started.wait()

    dialog.reject()
    assert dialog.result() == QDialog.DialogCode.Rejected
    assert manager.active_operations

    release.set()
    _finish_manager(manager, qapp)
    assert manager.active_operations == ()
    assert manager.ready_names
    dialog.deleteLater()


def test_explicit_cancel_stops_application_owned_install(
    qapp: QApplication, tmp_path: Path, monkeypatch
) -> None:
    started = threading.Event()

    def wait_for_cancel(_models_dir: Path, _name: str, **kwargs: object) -> dict[str, object]:
        started.set()
        should_cancel = kwargs["should_cancel"]
        # should_cancel is the install's cancel Event.is_set: wait on that Event.
        should_cancel.__self__.wait()  # type: ignore[attr-defined]
        raise PixelupError(ErrorCode.JOB_CANCELLED, "Model installation cancelled.")

    monkeypatch.setattr("pixelup.model_manager.download_model", wait_for_cancel)
    manager = ModelManager(tmp_path)
    _ready(manager, qapp)
    dialog = ManagedModelsDialog(manager)
    cancellation_observed: list[tuple[str, ...]] = []
    bundle_names = MANAGED_MODEL_BUNDLES[0].artifact_names
    manager.cancelled.connect(cancellation_observed.append)
    try:
        dialog._install_bundle(0)
        assert started.wait()
        dialog._install_bundle(0)
        _finish_manager(manager, qapp)
        assert bundle_names in cancellation_observed
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        dialog.deleteLater()


def test_manual_surface_has_independent_actions_and_truthful_columns(
    qapp: QApplication, tmp_path: Path
) -> None:
    manager = ModelManager(tmp_path)
    _ready(manager, qapp)
    dialog = ManagedModelsDialog(manager)
    try:
        labels = {label.text() for label in dialog.findChildren(QLabel)}

        assert dialog.windowTitle() == "Managed models"
        # Named by its title bar, not again inside the body.
        assert "Managed models" not in labels
        assert "Size" in labels
        assert len(dialog.row_action_buttons) == len(MANAGED_MODEL_BUNDLES)
        assert {button.text() for button in dialog.row_action_buttons} == {"Install"}
        assert dialog.primary_button.text() == "Install all"

        footer = dialog.layout().itemAt(dialog.layout().count() - 1).widget().layout()
        # Showing the folder and installing at the left end; Close at the right.
        assert footer.itemAt(0).widget() is dialog.reveal_button
        assert footer.itemAt(1).widget() is dialog.primary_button
        assert footer.itemAt(2).spacerItem() is not None
        assert footer.itemAt(3).widget() is dialog.dismiss_button
    finally:
        dialog.deleteLater()


def test_coming_to_the_front_finds_model_files_placed_by_hand(
    qapp: QApplication, tmp_path: Path, process_until, background
) -> None:
    manager = ModelManager(tmp_path)
    _ready(manager, qapp)
    dialog = ManagedModelsDialog(manager)
    try:
        dialog.show()
        qapp.processEvents()
        process_until(background.settled, what="The models scan")
        qapp.processEvents()
        assert dialog.status_labels[0].text() == "Not installed"
        row_button = dialog.row_action_buttons[0]
        row_button.setFocus()

        for name in MANAGED_MODEL_BUNDLES[0].artifact_names:
            path = model_file(tmp_path, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"model")
        QApplication.sendEvent(dialog, QEvent(QEvent.Type.WindowActivate))
        process_until(
            lambda: dialog.status_labels[0].text() == "Installed",
            what="The dialog's rescan",
        )

        assert row_button.text() == "Reinstall"
        # The rows are updated in place, so the focused one keeps the focus.
        assert dialog.focusWidget() is row_button
    finally:
        dialog.close()
        dialog.deleteLater()


def test_install_all_requests_every_missing_artifact(
    qapp: QApplication, tmp_path: Path, monkeypatch
) -> None:
    installed: list[str] = []

    def install(models_dir: Path, name: str, **_kwargs: object) -> dict[str, object]:
        installed.append(name)
        target = model_file(models_dir, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"model")
        return {"status": "downloaded"}

    monkeypatch.setattr("pixelup.model_manager.download_model", install)
    manager = ModelManager(tmp_path)
    _ready(manager, qapp)
    dialog = ManagedModelsDialog(manager)
    try:
        dialog._install_all_or_cancel()
        _finish_manager(manager, qapp)

        assert set(installed) == set(MANAGED_ARTIFACT_NAMES)
        assert len(installed) == len(MANAGED_ARTIFACT_NAMES)
        assert all(button.text() == "Reinstall" for button in dialog.row_action_buttons)
        assert dialog.primary_button.text() == "Install all"
        assert not dialog.primary_button.isEnabled()
    finally:
        dialog.deleteLater()


def test_install_all_runs_one_download_at_a_time(
    qapp: QApplication, tmp_path: Path, monkeypatch
) -> None:
    lock = threading.Lock()
    current = 0
    peak = 0
    reached_cap = threading.Event()
    release = threading.Event()

    def install(models_dir: Path, name: str, **_kwargs: object) -> dict[str, object]:
        nonlocal current, peak
        with lock:
            current += 1
            peak = max(peak, current)
            reached_cap.set()
        assert release.wait()
        with lock:
            current -= 1
        target = model_file(models_dir, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"model")
        return {"status": "downloaded"}

    monkeypatch.setattr("pixelup.model_manager.download_model", install)
    manager = ModelManager(tmp_path)
    _ready(manager, qapp)
    dialog = ManagedModelsDialog(manager)
    try:
        dialog._install_all_or_cancel()
        # Every missing artifact was requested at once: one downloads, the rest
        # wait their turn as "queued".
        assert reached_cap.wait()
        qapp.processEvents()
        assert len(manager.active_operations) == 1
        assert any(operation.kind == "queued" for operation in manager.operations)

        release.set()
        _finish_manager(manager, qapp)
        assert peak == 1
        assert all(model_file(tmp_path, name).exists() for name in MANAGED_ARTIFACT_NAMES)
    finally:
        release.set()
        dialog.deleteLater()


def test_a_second_row_install_waits_its_turn(
    qapp: QApplication, tmp_path: Path, monkeypatch
) -> None:
    started: list[str] = []
    first_started = threading.Event()
    release = threading.Event()

    def install(models_dir: Path, name: str, **_kwargs: object) -> dict[str, object]:
        started.append(name)
        first_started.set()
        release.wait()
        target = model_file(models_dir, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"model")
        return {"status": "downloaded"}

    monkeypatch.setattr("pixelup.model_manager.download_model", install)
    manager = ModelManager(tmp_path)
    _ready(manager, qapp)
    dialog = ManagedModelsDialog(manager)
    try:
        dialog._install_bundle(0)
        assert first_started.wait()
        qapp.processEvents()
        assert dialog.row_action_buttons[0].text() == "Cancel"
        assert dialog.row_action_buttons[1].isEnabled()

        dialog._install_bundle(1)
        qapp.processEvents()
        assert len(manager.active_operations) == 1
        assert manager.operations_for(MANAGED_MODEL_BUNDLES[1].artifact_names)[0].kind == "queued"

        release.set()
        _finish_manager(manager, qapp)
        assert started == [
            MANAGED_MODEL_BUNDLES[0].artifact_names[0],
            MANAGED_MODEL_BUNDLES[1].artifact_names[0],
        ]
    finally:
        release.set()
        dialog.deleteLater()


def test_active_progress_is_reported_in_the_row_status(
    qapp: QApplication, tmp_path: Path, monkeypatch
) -> None:
    bundle = MANAGED_MODEL_BUNDLES[0]
    total = artifact_size_bytes(bundle.artifact_names)
    progress_sent = threading.Event()
    release = threading.Event()

    def install(models_dir: Path, name: str, **kwargs: object) -> dict[str, object]:
        callback = kwargs["on_download"]
        callback(name, total // 2, total)  # type: ignore[operator]
        progress_sent.set()
        assert release.wait()
        target = model_file(models_dir, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"model")
        return {"status": "downloaded"}

    monkeypatch.setattr("pixelup.model_manager.download_model", install)
    manager = ModelManager(tmp_path)
    _ready(manager, qapp)
    dialog = ManagedModelsDialog(manager)
    try:
        dialog._install_bundle(0)
        assert progress_sent.wait()
        expected = (total // 2) * 100 // total
        while (
            f"{expected}%" not in dialog.status_labels[0].text()
        ):
            qapp.processEvents()
        assert dialog.status_labels[0].text() == f"Installing {expected}%"
    finally:
        release.set()
        _finish_manager(manager, qapp)
        dialog.deleteLater()


def test_columns_are_content_derived_inside_a_width_that_is_chosen(
    qapp: QApplication, tmp_path: Path
) -> None:
    # The columns still size to their own longest text, so a status never makes
    # them jump — but the dialog does not size to them. It takes the one chosen
    # width for a table, which has to be wide enough to hold them
    # (modal-dialog-conventions).
    manager = ModelManager(tmp_path)
    _ready(manager, qapp)
    dialog = ManagedModelsDialog(manager)
    try:
        longest_purpose = max(
            (english().of(bundle.purpose) for bundle in MANAGED_MODEL_BUNDLES),
            key=len,
        )
        assert dialog.column_minimum_widths[1] >= (
            dialog.fontMetrics().horizontalAdvance(longest_purpose) + 24
        )
        assert dialog.width() == TABLE_WIDTH
        assert dialog.width() >= sum(dialog.column_minimum_widths)
        assert dialog.windowType() == Qt.WindowType.Dialog
        assert dialog.windowModality() == Qt.WindowModality.ApplicationModal
    finally:
        dialog.deleteLater()


def test_dynamic_result_grows_dialog_without_compressing_rows_or_footer(
    qapp: QApplication, tmp_path: Path
) -> None:
    manager = ModelManager(tmp_path)
    _ready(manager, qapp)
    dialog = ManagedModelsDialog(manager)
    try:
        dialog.show()
        qapp.processEvents()
        initial_height = dialog.height()
        row_height = dialog.row_action_buttons[0].height()
        close_height = dialog.dismiss_button.height()

        dialog._show_error(Message("managedModels.revealFailed"))
        qapp.processEvents()

        assert dialog.height() >= initial_height
        assert dialog.result_view.isVisibleTo(dialog)
        assert dialog.row_action_buttons[0].height() >= row_height
        assert dialog.dismiss_button.height() >= close_height
        assert dialog.dismiss_button.isVisibleTo(dialog)
        assert dialog.primary_button.isVisibleTo(dialog)
    finally:
        dialog.deleteLater()


def test_the_models_panel_separates_its_headings_with_the_sheets_line(
    qapp: QApplication, tmp_path: Path
) -> None:
    from PySide6.QtWidgets import QFrame

    manager = ModelManager(tmp_path)
    _ready(manager, qapp)
    dialog = ManagedModelsDialog(manager)
    try:
        lines = [
            frame
            for frame in dialog.models_panel.findChildren(QFrame)
            if frame.frameShape() == QFrame.Shape.HLine
        ]
        assert len(lines) == 1
        assert lines[0].styleSheet() == ""
    finally:
        dialog.deleteLater()


@pytest.mark.parametrize("draft", ["", "~/draft models", "relative/models", "absolute"])
def test_folder_picker_uses_effective_draft_and_keeps_selection_until_apply(
    qapp, tmp_path, monkeypatch, draft
) -> None:
    from pixelup import managed_models_dialog
    from pixelup.model_manager import ModelManager

    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("MODEL_NAME", "expanded")
    models_dir = tmp_path / "models"
    default_models_dir = tmp_path / "default models"
    chosen = str(tmp_path / " literal $MODEL_NAME %MODEL_NAME% ")
    if draft == "absolute":
        draft = str(tmp_path / "absolute draft")
    expected = {
        "": default_models_dir,
        "~/draft models": home / "draft models",
        "relative/models": home / "relative/models",
    }.get(draft, Path(draft))
    created = []
    saved = []
    accepted = True

    class Picker:
        def __init__(self, parent, _title, directory):
            created.append((parent, directory))

        def setFileMode(self, _mode):
            pass

        def setOption(self, _option):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted if accepted else QDialog.DialogCode.Rejected

        def selectedFiles(self):
            return [chosen]

        def deleteLater(self):
            pass

    Picker.FileMode = managed_models_dialog.QFileDialog.FileMode
    Picker.Option = managed_models_dialog.QFileDialog.Option
    monkeypatch.setattr(managed_models_dialog, "QFileDialog", Picker)
    manager = ModelManager(models_dir)
    _ready(manager, qapp)
    dialog = managed_models_dialog.ManagedModelsDialog(
        manager,
        models_folder=str(models_dir),
        default_models_dir=default_models_dir,
        save_folder=lambda value, done: saved.append(value),
    )
    try:
        dialog.folder_edit.setText(draft)
        dialog._choose_folder()
        assert created == [(dialog, str(expected))]
        assert dialog.folder_edit.text() == chosen
        assert saved == []

        accepted = False
        dialog._choose_folder()
        assert created[-1] == (dialog, chosen)
        assert dialog.folder_edit.text() == chosen
        assert saved == []

        dialog.apply_folder_button.click()
        assert saved == [chosen]
    finally:
        dialog.deleteLater()


@pytest.mark.parametrize("queue_preflight", [False, True])
def test_failed_readiness_disables_install_and_recovers_after_successful_retry(
    qapp, tmp_path, monkeypatch, process_until, queue_preflight, background,
) -> None:
    from pixelup import model_manager

    present, absent = MANAGED_ARTIFACT_NAMES[:2]
    model_file(tmp_path, present).write_bytes(b"present")
    manager = ModelManager(tmp_path)
    _ready(manager, qapp)
    real_read = model_manager.read_ready_names
    sentinel = "EACCES /internal/models ORIGINAL-DIALOG-SCAN-SENTINEL"
    failure = PermissionError(13, sentinel)

    def unreadable(_folder):
        raise failure

    monkeypatch.setattr(model_manager, "read_ready_names", unreadable)
    manager.refresh_readiness()
    process_until(background.settled, what="Failed models folder scan")
    qapp.processEvents()
    dialog = ManagedModelsDialog(
        manager,
        required_artifacts=(absent,) if queue_preflight else (),
        pending_job_count=2 if queue_preflight else 0,
    )
    try:
        expected = english().of(Message("managedModels.readinessFailed"))
        assert manager.readiness_error is failure
        assert not manager.readiness_known
        assert manager.ready_names == frozenset({present})
        assert dialog.summary_label.text() == expected
        assert all(label.text() == expected for label in dialog.status_labels)
        assert dialog.result_view.message_label.text() == expected
        assert dialog.result_view.isVisibleTo(dialog)
        assert sentinel not in dialog.result_view.message_label.text()
        assert not dialog.primary_button.isEnabled()
        assert all(not button.isEnabled() for button in dialog.row_action_buttons)

        monkeypatch.setattr(model_manager, "read_ready_names", real_read)
        manager.refresh_readiness()
        process_until(
            lambda: manager.readiness_known,
            what="Successful models folder retry",
        )
        assert manager.readiness_error is None
        assert dialog.result_view.isHidden()
        assert dialog.primary_button.isEnabled()
        assert all(button.isEnabled() is not queue_preflight
                   for button in dialog.row_action_buttons)
        assert all(label.text() != expected for label in dialog.status_labels)
    finally:
        dialog.deleteLater()
        manager.begin_shutdown()
        process_until(manager.cleanup_for_quit, what="Model scan cleanup")

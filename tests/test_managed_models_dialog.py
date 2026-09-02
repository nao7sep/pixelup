from __future__ import annotations

import threading
import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QProgressBar, QRadioButton

from pixelup.errors import ErrorCode, PixelupError
from pixelup.managed_models_dialog import ManagedModelsDialog
from pixelup.model_management import (
    MANAGED_ARTIFACT_NAMES,
    MANAGED_MODEL_BUNDLES,
    artifact_size_bytes,
)
from pixelup.model_manager import ModelManager
from pixelup.models import model_file


def _finish_manager(manager: ModelManager, qapp: QApplication) -> None:
    deadline = time.monotonic() + 3
    while not manager.cleanup_for_quit() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.001)
    qapp.processEvents()
    assert manager.cleanup_for_quit()


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
    required = ("RealESRGAN_x4plus", "GFPGANv1.4")

    def install(models_dir: Path, name: str, **_kwargs: object) -> dict[str, object]:
        target = model_file(models_dir, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"model")
        return {"status": "downloaded"}

    monkeypatch.setattr("pixelup.model_manager.download_model", install)
    manager = ModelManager(tmp_path)
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
        assert all("color:" in label.styleSheet() for label in required_statuses)
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
        raise PixelupError(ErrorCode.MODEL_DOWNLOAD_FAILED, "Network unavailable.")

    monkeypatch.setattr("pixelup.model_manager.download_model", fail)
    manager = ModelManager(tmp_path)
    dialog = ManagedModelsDialog(manager)
    try:
        dialog._install_bundle(0)
        _finish_manager(manager, qapp)
        assert manager.failed_operations[0].kind == "failed"
        assert dialog.result_view.isVisibleTo(dialog)
        assert dialog.result_view.message_label.text() == "Network unavailable."
        assert dialog.result_view.accessibleName() == "Network unavailable."
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
    required = ("RealESRGAN_x4plus", "GFPGANv1.4")

    def install(models_dir: Path, name: str, **_kwargs: object) -> dict[str, object]:
        if name == required[1]:
            raise PixelupError(ErrorCode.MODEL_DOWNLOAD_FAILED, "Second download failed.")
        target = model_file(models_dir, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"model")
        return {"status": "downloaded"}

    monkeypatch.setattr("pixelup.model_manager.download_model", install)
    manager = ModelManager(tmp_path)
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
        assert release.wait(2)
        assert kwargs["should_cancel"]() is False  # type: ignore[operator]
        target = model_file(models_dir, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"model")
        return {"status": "downloaded"}

    monkeypatch.setattr("pixelup.model_manager.download_model", wait_then_install)
    manager = ModelManager(tmp_path)
    dialog = ManagedModelsDialog(manager)
    dialog._install_bundle(0)
    assert started.wait(1)

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
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if should_cancel():  # type: ignore[operator]
                raise PixelupError(ErrorCode.JOB_CANCELLED, "Model installation cancelled.")
            time.sleep(0.001)
        raise AssertionError("cancel was not delivered")

    monkeypatch.setattr("pixelup.model_manager.download_model", wait_for_cancel)
    manager = ModelManager(tmp_path)
    dialog = ManagedModelsDialog(manager)
    cancellation_observed: list[tuple[str, ...]] = []
    bundle_names = MANAGED_MODEL_BUNDLES[0].artifact_names
    manager.cancelled.connect(cancellation_observed.append)
    try:
        dialog._install_bundle(0)
        assert started.wait(1)
        dialog._install_bundle(0)
        _finish_manager(manager, qapp)
        assert bundle_names in cancellation_observed
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        dialog.deleteLater()


def test_manual_surface_has_independent_actions_and_truthful_columns(
    qapp: QApplication, tmp_path: Path
) -> None:
    dialog = ManagedModelsDialog(ModelManager(tmp_path))
    try:
        labels = {label.text() for label in dialog.findChildren(QLabel)}

        assert dialog.windowTitle() == "Managed models"
        assert "Managed models" in labels
        assert "Size" in labels
        assert "Download" not in labels
        assert dialog.findChildren(QRadioButton) == []
        assert dialog.findChildren(QProgressBar) == []
        assert len(dialog.row_action_buttons) == len(MANAGED_MODEL_BUNDLES)
        assert {button.text() for button in dialog.row_action_buttons} == {"Install"}
        assert dialog.primary_button.text() == "Install all"

        footer = dialog.layout().itemAt(dialog.layout().count() - 1).layout()
        assert footer.itemAt(1).widget() is dialog.dismiss_button
        assert footer.itemAt(2).widget() is dialog.reveal_button
        assert footer.itemAt(3).widget() is dialog.primary_button
    finally:
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


def test_independent_row_installs_run_concurrently(
    qapp: QApplication, tmp_path: Path, monkeypatch
) -> None:
    started: list[str] = []
    started_lock = threading.Lock()
    two_started = threading.Event()
    release = threading.Event()

    def install(models_dir: Path, name: str, **_kwargs: object) -> dict[str, object]:
        with started_lock:
            started.append(name)
            if len(started) == 2:
                two_started.set()
        assert release.wait(2)
        target = model_file(models_dir, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"model")
        return {"status": "downloaded"}

    monkeypatch.setattr("pixelup.model_manager.download_model", install)
    manager = ModelManager(tmp_path)
    dialog = ManagedModelsDialog(manager)
    try:
        dialog._install_bundle(0)
        qapp.processEvents()
        assert dialog.row_action_buttons[0].text() == "Cancel"
        assert dialog.row_action_buttons[1].isEnabled()
        assert manager.install(
            MANAGED_MODEL_BUNDLES[0].artifact_names,
            force=False,
        ) is None

        dialog._install_bundle(1)
        assert two_started.wait(1)
        assert len(manager.active_operations) == 2

        release.set()
        _finish_manager(manager, qapp)
        assert set(started) == {
            MANAGED_MODEL_BUNDLES[0].artifact_names[0],
            MANAGED_MODEL_BUNDLES[1].artifact_names[0],
        }
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
        assert release.wait(2)
        target = model_file(models_dir, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"model")
        return {"status": "downloaded"}

    monkeypatch.setattr("pixelup.model_manager.download_model", install)
    manager = ModelManager(tmp_path)
    dialog = ManagedModelsDialog(manager)
    try:
        dialog._install_bundle(0)
        assert progress_sent.wait(1)
        expected = (total // 2) * 100 // total
        deadline = time.monotonic() + 1
        while (
            f"{expected}%" not in dialog.status_labels[0].text()
            and time.monotonic() < deadline
        ):
            qapp.processEvents()
        assert dialog.status_labels[0].text() == f"Installing {expected}%"
    finally:
        release.set()
        _finish_manager(manager, qapp)
        dialog.deleteLater()


def test_dialog_columns_are_content_derived_and_window_is_native_titled(
    qapp: QApplication, tmp_path: Path
) -> None:
    dialog = ManagedModelsDialog(ModelManager(tmp_path))
    try:
        longest_purpose = max(
            (bundle.purpose for bundle in MANAGED_MODEL_BUNDLES),
            key=len,
        )
        assert dialog.column_minimum_widths[1] >= (
            dialog.fontMetrics().horizontalAdvance(longest_purpose) + 24
        )
        assert dialog.sizeHint().width() >= sum(dialog.column_minimum_widths)
        assert dialog.windowType() == Qt.WindowType.Dialog
        assert dialog.windowModality() == Qt.WindowModality.ApplicationModal
    finally:
        dialog.deleteLater()


def test_dynamic_result_grows_dialog_without_compressing_rows_or_footer(
    qapp: QApplication, tmp_path: Path
) -> None:
    dialog = ManagedModelsDialog(ModelManager(tmp_path))
    try:
        dialog.show()
        qapp.processEvents()
        initial_height = dialog.height()
        row_height = dialog.row_action_buttons[0].height()
        close_height = dialog.dismiss_button.height()

        dialog._show_error(
            "Could not open the models folder. Check its permissions and try again."
        )
        qapp.processEvents()

        assert dialog.height() > initial_height
        assert dialog.row_action_buttons[0].height() >= row_height
        assert dialog.dismiss_button.height() >= close_height
        assert dialog.dismiss_button.isVisibleTo(dialog)
        assert dialog.primary_button.isVisibleTo(dialog)
    finally:
        dialog.deleteLater()

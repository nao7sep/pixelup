from __future__ import annotations

import threading
import time
from pathlib import Path

from PySide6.QtWidgets import QApplication, QDialog

from pixelup.errors import ErrorCode, PixelupError
from pixelup.managed_models_dialog import ManagedModelsDialog
from pixelup.model_management import MANAGED_MODEL_BUNDLES
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
        dialog.bundle_buttons[0].setChecked(True)
        assert dialog.install_button.text() == "Install selected"

        dialog._install_or_cancel()
        _finish_manager(manager, qapp)

        bundle = MANAGED_MODEL_BUNDLES[0]
        assert calls == [(bundle.artifact_names[0], False)]
        assert dialog.status_labels[0].text() == "Ready"
        assert dialog.install_button.text() == "Reinstall selected"
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
        assert dialog.install_button.text() == "Reinstall selected"
        dialog._install_or_cancel()
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
        assert dialog.install_button.text() == "Install and queue 12 jobs"
        assert "No jobs will be created" in dialog.summary_label.text()
        dialog._install_or_cancel()
        _finish_manager(manager, qapp)

        assert dialog.result() == QDialog.DialogCode.Rejected
        assert manager.missing(required) == ()
        assert dialog.install_button.isEnabled() is False
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
        dialog._install_or_cancel()
        _finish_manager(manager, qapp)
        assert manager.operation.kind == "failed"
        assert dialog.result_frame.isVisibleTo(dialog)
        assert dialog.result_label.text() == "Network unavailable."
        assert dialog.install_button.isEnabled()
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

    assert manager.operation.kind == "failed"
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
    dialog._install_or_cancel()
    assert started.wait(1)

    dialog.reject()
    assert dialog.result() == QDialog.DialogCode.Rejected
    assert manager.operation.kind == "running"

    release.set()
    _finish_manager(manager, qapp)
    assert manager.operation.kind == "idle"
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
    try:
        dialog._install_or_cancel()
        assert started.wait(1)
        dialog._install_or_cancel()
        _finish_manager(manager, qapp)
        assert manager.operation.kind == "cancelled"
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        dialog.deleteLater()

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtWidgets import QApplication, QDialog

from pixelup.errors import ErrorCode, PixelupError
from pixelup.managed_models_dialog import ManagedModelsDialog
from pixelup.model_management import MANAGED_MODEL_BUNDLES
from pixelup.models import model_file


def _finish_thread(dialog: ManagedModelsDialog, qapp: QApplication) -> None:
    deadline = time.monotonic() + 3
    while dialog._thread is not None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.001)
    assert dialog._thread is None


def test_manual_install_updates_readiness(qapp: QApplication, tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []

    def install(models_dir: Path, name: str, **kwargs: object) -> dict[str, object]:
        calls.append((name, bool(kwargs["force"])))
        target = model_file(models_dir, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"model")
        callback = kwargs["on_download"]
        callback(name, 5, 5)  # type: ignore[operator]
        return {"status": "downloaded"}

    monkeypatch.setattr("pixelup.managed_models_dialog.download_model", install)
    dialog = ManagedModelsDialog(tmp_path)
    try:
        dialog.table.selectRow(0)
        assert dialog.install_button.text() == "Install selected"

        dialog._install()
        _finish_thread(dialog, qapp)

        bundle = MANAGED_MODEL_BUNDLES[0]
        assert calls == [(bundle.artifact_names[0], False)]
        assert dialog.table.item(0, 3).text() == "Ready"
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

    monkeypatch.setattr("pixelup.managed_models_dialog.download_model", install)
    dialog = ManagedModelsDialog(tmp_path)
    try:
        dialog.table.selectRow(0)
        assert dialog.install_button.text() == "Reinstall selected"
        dialog._install()
        _finish_thread(dialog, qapp)
        assert forces == [True]
    finally:
        dialog.deleteLater()


def test_queue_install_accepts_only_after_all_requirements_are_ready(
    qapp: QApplication, tmp_path: Path, monkeypatch
) -> None:
    required = ("RealESRGAN_x4plus", "GFPGANv1.4")

    def install(models_dir: Path, name: str, **_kwargs: object) -> dict[str, object]:
        target = model_file(models_dir, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"model")
        return {"status": "downloaded"}

    monkeypatch.setattr("pixelup.managed_models_dialog.download_model", install)
    dialog = ManagedModelsDialog(
        tmp_path,
        required_artifacts=required,
        pending_job_count=12,
    )
    try:
        assert dialog.install_button.text() == "Install and queue 12 jobs"
        assert "No jobs will be created" in dialog.summary_label.text()
        dialog._install()
        _finish_thread(dialog, qapp)
        assert dialog.result() == QDialog.DialogCode.Accepted
        assert all(model_file(tmp_path, name).is_file() for name in required)
    finally:
        dialog.deleteLater()


def test_install_failure_stays_local_and_retryable(
    qapp: QApplication, tmp_path: Path, monkeypatch
) -> None:
    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise PixelupError(ErrorCode.MODEL_DOWNLOAD_FAILED, "Network unavailable.")

    monkeypatch.setattr("pixelup.managed_models_dialog.download_model", fail)
    dialog = ManagedModelsDialog(tmp_path)
    try:
        dialog._install()
        _finish_thread(dialog, qapp)
        assert dialog.result() == QDialog.DialogCode.Rejected
        assert dialog.result_frame.isVisibleTo(dialog)
        assert dialog.result_label.text() == "Network unavailable."
        assert dialog.install_button.isEnabled()
    finally:
        dialog.deleteLater()


def test_closing_during_install_cancels_then_closes(
    qapp: QApplication, tmp_path: Path, monkeypatch
) -> None:
    def wait_for_cancel(_models_dir: Path, _name: str, **kwargs: object) -> dict[str, object]:
        should_cancel = kwargs["should_cancel"]
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if should_cancel():  # type: ignore[operator]
                raise PixelupError(ErrorCode.JOB_CANCELLED, "Model installation cancelled.")
            time.sleep(0.001)
        raise AssertionError("cancel was not delivered")

    monkeypatch.setattr("pixelup.managed_models_dialog.download_model", wait_for_cancel)
    dialog = ManagedModelsDialog(tmp_path)
    try:
        dialog._install()
        dialog.reject()
        _finish_thread(dialog, qapp)
        assert dialog.result() == QDialog.DialogCode.Rejected
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        dialog.deleteLater()

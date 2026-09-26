from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from pixelup.model_management import MANAGED_ARTIFACT_NAMES
from pixelup.model_manager import ModelManager
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
) -> None:
    manager = ModelManager(tmp_path)
    thread = _QuittableThread()
    manager._threads[7] = thread  # type: ignore[assignment]

    manager._worker_finished(7, True, False, None)

    assert manager._install_results[7] == (True, False, None)
    assert thread.quit_calls == 1


@pytest.mark.heavy
def test_every_managed_model_is_installed_and_matches_its_pin(
    qapp: QApplication,
    heavy_models_dir: Path,
) -> None:
    manager = ModelManager(heavy_models_dir)
    total = len(MANAGED_ARTIFACT_NAMES)
    assert manager.ready_count() == (total, total)
    for name in MANAGED_ARTIFACT_NAMES:
        # Raises unless the file's size and SHA-256 match the registry's pin.
        verify_model_file(model_file(heavy_models_dir, name), known_model(name))
    manager.deleteLater()

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication

from pixelup.model_manager import ModelManager


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

    manager._worker_finished(7, True, False, "")

    assert manager._install_results[7] == (True, False, "")
    assert thread.quit_calls == 1

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from PySide6.QtCore import QEventLoop
from PySide6.QtWidgets import QApplication, QStyleFactory

from pixelup.backup_store import close_backup_store
from pixelup.model_management import MANAGED_ARTIFACT_NAMES
from pixelup.model_manager import ModelManager
from pixelup.session_log import LOGGER_NAME

CORPUS = Path(__file__).resolve().parents[2] / "company" / "assets" / "test-fixtures"
MODEL_INSTALL_TIMEOUT_S = 60 * 60


@pytest.fixture
def file_symlink_capability(tmp_path: Path) -> None:
    """Skip only file-symlink contracts when this Windows token cannot create one."""
    target = tmp_path / "symlink-capability-target"
    link = tmp_path / "symlink-capability-link"
    target.write_bytes(b"probe")
    try:
        link.symlink_to(target)
    except OSError as exc:
        if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
            pytest.skip("file symlink creation requires Developer Mode or elevation on Windows")
        raise
    else:
        link.unlink()


@pytest.fixture
def make_directory_alias() -> Callable[[Path, Path], None]:
    """Create a real directory alias, using a privilege-free junction on Windows."""

    def create(alias: Path, target: Path) -> None:
        if os.name != "nt":
            alias.symlink_to(target, target_is_directory=True)
            return
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(alias), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            pytest.fail(
                "could not create Windows directory junction: "
                f"{completed.stdout}{completed.stderr}"
            )

    return create


@pytest.fixture(autouse=True)
def _reset_pixelup_logging():
    """Restore the shared logger and excepthook after every test.

    PixelUp's session logger is a process-wide singleton; tests call
    configure_session_logging() which mutates its handlers/level and installs a
    sys.excepthook. Without this teardown that state leaks across tests and makes
    the suite order-dependent.
    """
    saved_excepthook = sys.excepthook
    yield
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)
    sys.excepthook = saved_excepthook


@pytest.fixture(autouse=True)
def _reset_backup_store():
    """Close the write-through backup store after every test.

    The store is a process-wide singleton opened lazily against whatever
    PIXELUP_HOME resolves to at first use. Tests redirect PIXELUP_HOME to a
    throwaway root; without this teardown the singleton would stay bound to the
    first test's root (and hold that file handle open) for the whole session. This
    closes it and resets the singleton so the next test's first save re-opens the
    store under that test's own root (data-backup-conventions test migration).
    """
    yield
    close_backup_store()


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """A single offscreen QApplication shared by all GUI tests.

    Widget construction needs a running QApplication. The offscreen platform
    keeps the suite headless so it never opens real windows.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    # The style build_app installs. Control geometry and the palette both come
    # from it, so a test that measures what the app draws has to measure it under
    # the same style rather than the platform default.
    if "Fusion" in QStyleFactory.keys():
        app.setStyle("Fusion")
    return app


ProcessUntil = Callable[..., None]


@pytest.fixture(scope="session")
def process_until(qapp: QApplication) -> ProcessUntil:
    """Run the Qt event loop until ``done()`` holds, failing after ``timeout_s``."""

    def run(done: Callable[[], bool], *, timeout_s: float, what: str) -> None:
        deadline = time.monotonic() + timeout_s
        while not done():
            if time.monotonic() > deadline:
                pytest.fail(f"{what} did not finish within {timeout_s:.0f} s")
            qapp.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 100)
            time.sleep(0.02)

    return run


@pytest.fixture(scope="session")
def heavy_models_dir(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    process_until: ProcessUntil,
) -> Path:
    """Every managed Real-ESRGAN artifact, acquired through PixelUp's own ModelManager.

    The weights persist in pytest's cache between runs and follow the app's rule:
    install what is missing, and never download a present file again, since each
    pin names its own file. Only heavy tests request this fixture.
    """
    models_dir = request.config.cache.mkdir("pixelup-heavy-models")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("PIXELUP_HOME", str(tmp_path_factory.mktemp("model-install-home")))
        manager = ModelManager(models_dir)
        missing = manager.missing(MANAGED_ARTIFACT_NAMES)
        if missing:
            operation_id = manager.install(missing, force=False)
            assert operation_id is not None
            process_until(
                lambda: not manager.active_operations and manager.cleanup_for_quit(),
                timeout_s=MODEL_INSTALL_TIMEOUT_S,
                what="Installing the managed models",
            )
            failures = [operation.error for operation in manager.failed_operations]
            assert not failures, f"Installing the managed models failed: {failures}"
        manager.deleteLater()
    return models_dir


@pytest.fixture
def corpus_file() -> Callable[[str], Path]:
    """A file from the shared test-fixture corpus, checked out beside this repository."""

    def resolve(relative: str) -> Path:
        path = CORPUS / relative
        if not path.is_file():
            pytest.fail(
                f"{path} is missing. Heavy tests read the shared test-fixture corpus; "
                "check out the company repository beside this one."
            )
        return path

    return resolve


@pytest.fixture(autouse=True)
def _english_interface(monkeypatch: pytest.MonkeyPatch):
    """Run every test in English, whatever language the computer running it speaks.

    build_app settles the interface language from the computer's own list; a test
    asserts the English because it is the source language, so the computer's list
    is pinned to English here, AppKit is left alone, and a test that switched the
    language is put back.
    """
    from pixelup.i18n import bootstrap, localizer

    monkeypatch.setattr(bootstrap, "read_computer_languages", lambda: ("en",))
    monkeypatch.setattr(bootstrap, "align_appkit", lambda tag: None)
    yield
    localizer.use("en", ("en",))

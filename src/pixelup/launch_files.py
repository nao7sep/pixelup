"""The files PixelUp opens to start, checked before it writes anything
(store-recovery-conventions, What a refused file does).

Settings that a newer PixelUp wrote or that cannot be read stop the launch, and
so does a records or backup database a newer PixelUp wrote. A records or backup
database that cannot be read is switched off for the launch instead, and the
window says so. Every refused file is named at once and none is changed: the
check only reads. ``window.ini`` is Qt's disposable window state, which falls back
to the designed geometry on its own, and the fallback logs are plain text.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from pixelup.app_config import ConfigLoadResult, load_app_config_result
from pixelup.backup_store import STORE_FILE_NAME
from pixelup.config import resolve_state_dir
from pixelup.formats import (
    BACKUPS_FORMAT,
    RECORDS_FORMAT,
    NewerFormatError,
    RefusedFile,
    RefusedFilesError,
    probe_sqlite_store,
)
from pixelup.i18n.message import Message, join
from pixelup.session_log import RECORDS_FILE_NAME


@dataclass(frozen=True, slots=True)
class LaunchFiles:
    """What the launch may use: the settings, and which side stores stay off."""

    settings: ConfigLoadResult
    records_off: bool = False
    backups_off: bool = False


def check_launch_files() -> LaunchFiles:
    """Read every file the launch opens; raises ``RefusedFilesError`` naming each
    one that stops it."""
    root = resolve_state_dir()
    refused: list[RefusedFile] = []
    settings: ConfigLoadResult | None = None
    try:
        settings = load_app_config_result(root)
    except RefusedFilesError as error:
        refused.extend(error.files)
    off: set[str] = set()
    for name, expected in ((RECORDS_FILE_NAME, RECORDS_FORMAT), (STORE_FILE_NAME, BACKUPS_FORMAT)):
        try:
            probe_sqlite_store(root / name, expected)
        except NewerFormatError:
            refused.append(RefusedFile(root / name, newer=True))
        except (ValueError, sqlite3.DatabaseError):
            off.add(name)
    if refused or settings is None:
        raise RefusedFilesError(tuple(refused))
    return LaunchFiles(
        settings,
        records_off=RECORDS_FILE_NAME in off,
        backups_off=STORE_FILE_NAME in off,
    )


def refused_files_message(files: tuple[RefusedFile, ...]) -> Message:
    """The launch stop's text: each refused file and whether it is newer or unreadable."""
    lines = [
        Message.of(
            "startup.refusedNewer" if file.newer else "startup.refusedUnreadable",
            path=str(file.path),
        )
        for file in files
    ]
    return Message.of("startup.refusedFiles", files=join("startup.refusedList", lines))

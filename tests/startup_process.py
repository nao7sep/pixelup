"""Child-process probe of the production startup boundary; substitutes only UI glue."""

from __future__ import annotations

import json
import os
import runpy
import sys
import threading
from dataclasses import replace
from pathlib import Path

from PySide6.QtWidgets import QApplication

from pixelup import backup_store, gui
from pixelup.app_config import load_app_config_result, save_app_config
from pixelup.config import resolve_runtime_dirs, resolve_state_dir, write_managed_text
from pixelup.i18n import bootstrap, localizer
from pixelup.i18n.message import Message
from pixelup.session_log import flush_records


def emit(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}), flush=True)


def command() -> str:
    line = sys.stdin.readline()
    if not line:
        raise RuntimeError("startup probe lost its controlling process")
    return line.strip()


mode, entry = sys.argv[1:]
sys.argv = ["pixelup"]
bootstrap.read_computer_languages = lambda: ("en",)
bootstrap.align_appkit = lambda tag: None
pending_write, release_write = threading.Event(), threading.Event()
real_backup_write = backup_store._write


def backup_write(store, session, path, data) -> None:
    if mode == "settlement" and Path(path).name == "settlement.txt":
        pending_write.set()
        release_write.wait()
    real_backup_write(store, session, path, data)


backup_store._write = backup_write


class ProbeWindow:
    """Keep actual store writers, replacing window drawing/model/OS interaction."""

    _os_close_accepted = False

    def __init__(self, *, runtime_dirs=None) -> None:
        emit("writer_entered")
        self.runtime_dirs = runtime_dirs or resolve_runtime_dirs()
        loaded = load_app_config_result()
        save_app_config(replace(loaded.config, font_family="Menlo"), loaded.config, loaded.files)
        state = gui.window_state()
        state.save("admission-probe", os.getpid())
        if mode == "startup_failure":
            raise OSError("EIO /private/internal/startup-path hostile-startup-diagnostic")

    def show_prepared(self) -> None:
        pass

    def holds_work_after_os_close(self) -> bool:
        return False


gui.MainWindow = ProbeWindow


def run_loop(self) -> int:
    flush_records()
    gui.drain_backups()
    if mode == "settlement":
        # Admit a real optional backup write and hold its physical writer: main's
        # drain waits for it, so the release arrives on a thread of its own.
        write_managed_text(resolve_state_dir() / "settlement.txt", "settled\n")
        pending_write.wait()
    emit("ready")
    assert command() == "quit"
    if mode == "settlement":
        def settle() -> None:
            assert command() == "settle"
            release_write.set()

        threading.Thread(target=settle, daemon=True).start()
        emit("quitting")
    return 0


QApplication.exec = run_loop


def startup_failure(message: Message, hint: Message | None) -> None:
    emit("failure", key=message.key, text=localizer.of(message))
    assert command() == "return"


gui.show_startup_failure = startup_failure
emit("imported")
assert command() == "start"
if entry == "module":
    try:
        runpy.run_module("pixelup", run_name="__main__")
    except SystemExit as exc:
        result = exc.code
else:
    result = gui.main()
emit("returned", result=result)
assert command() == "exit"
raise SystemExit(result)

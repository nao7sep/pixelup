from __future__ import annotations

import json
import os
import queue
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from pixelup import gui, startup_admission
from pixelup.errors import ErrorCode, PixelupError
from pixelup.i18n.message import Message

PROBE = Path(__file__).with_name("startup_process.py")


class StartupProcess:
    def __init__(self, root: Path, mode: str, entry: str) -> None:
        env = os.environ.copy()
        env.update(PIXELUP_DATA_DIR=str(root), QT_QPA_PLATFORM="offscreen")
        for name in ("PIXELUP_SELFTEST", "PIXELUP_MODELS_DIR", "PIXELUP_TEMP_DIR", "PIXELUP_DEBUG"):
            env.pop(name, None)
        self.process = subprocess.Popen(
            [sys.executable, str(PROBE), mode, entry],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.events: queue.Queue[dict | None] = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            assert self.next()["event"] == "imported"
        except BaseException:
            self.close()
            raise

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.events.put(json.loads(line))
        self.events.put(None)

    def send(self, command: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def next(self) -> dict:
        event = self.events.get()
        if event is None:
            assert self.process.stderr is not None
            pytest.fail(f"startup child exited early: {self.process.stderr.read()}")
        return event

    def start(self) -> dict:
        self.send("start")
        event = self.next()
        if event["event"] == "writer_entered":
            return self.next()
        return event

    def return_from_main(self, *, failed: bool = False) -> None:
        self.send("return" if failed else "quit")
        assert self.next() == {"event": "returned", "result": 1 if failed else 0}

    def exit(self, *, code: int = 0) -> None:
        self.send("exit")
        assert self.process.wait() == code

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self.reader.join()
        assert not self.reader.is_alive()
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None:
                stream.close()


@pytest.fixture
def launch():
    children: list[StartupProcess] = []

    def create(root: Path, *, mode: str = "normal", entry: str = "gui") -> StartupProcess:
        child = StartupProcess(root, mode, entry)
        children.append(child)
        return child

    yield create
    for child in children:
        child.close()
    assert all(child.process.poll() is not None for child in children)


def store_snapshot(root: Path) -> dict:
    """Compare actual store contents rather than SQLite WAL bookkeeping bytes."""
    result = {name: (root / name).read_bytes() for name in ("config.json", "window.ini")}
    for name in ("records.sqlite3", "backups.sqlite3"):
        with sqlite3.connect(root / name) as connection:
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            result[name] = {
                table: connection.execute(f'SELECT * FROM "{table}"').fetchall()
                for (table,) in tables
            }
    result["logs"] = tuple(path.name for path in (root / "logs").iterdir())
    return result


def assert_contended(child: StartupProcess) -> None:
    assert child.start()["key"] == "startup.alreadyRunning"
    child.return_from_main(failed=True)
    child.exit(code=1)


def test_contender_cannot_open_mutable_stores_even_when_claim_file_is_old(tmp_path, launch):
    owner = launch(tmp_path)
    assert owner.start() == {"event": "ready"}
    before = store_snapshot(tmp_path)
    # A live long-running owner must never lose admission because a clock or
    # timestamp says the claim is old. No time-based expiry is consulted.
    os.utime(tmp_path / "writer.lock", (1, 1))
    assert_contended(launch(tmp_path, entry="module"))
    assert store_snapshot(tmp_path) == before
    owner.return_from_main()
    owner.exit()


def test_simultaneous_launches_admit_exactly_one_writer(tmp_path, launch):
    first, second = launch(tmp_path), launch(tmp_path, entry="module")
    first.send("start")
    second.send("start")
    outcomes = [first.next(), second.next()]
    assert sorted(event["event"] for event in outcomes) == ["failure", "writer_entered"]
    winner, loser = (first, second) if outcomes[0]["event"] == "writer_entered" else (second, first)
    assert winner.next() == {"event": "ready"}
    assert next(event for event in outcomes if event["event"] == "failure")["key"] == (
        "startup.alreadyRunning"
    )
    loser.return_from_main(failed=True)
    loser.exit(code=1)
    winner.return_from_main()
    winner.exit()


def test_root_alias_contends_but_independent_root_can_run(
    tmp_path, launch, make_directory_alias,
):
    root, alias, independent = tmp_path / "data", tmp_path / "alias", tmp_path / "independent"
    root.mkdir()
    make_directory_alias(alias, root)
    owner = launch(root)
    assert owner.start() == {"event": "ready"}
    assert_contended(launch(alias))
    other = launch(independent)
    assert other.start() == {"event": "ready"}
    other.return_from_main()
    other.exit()
    owner.return_from_main()
    owner.exit()


@pytest.mark.parametrize("forced", [False, True])
def test_admission_is_released_by_actual_owner_exit(tmp_path, launch, forced):
    owner = launch(tmp_path)
    assert owner.start() == {"event": "ready"}
    if forced:
        owner.process.kill()
        owner.process.wait()
    else:
        owner.return_from_main()
        # main() returning and logging.shutdown() finishing do not end ownership.
        assert_contended(launch(tmp_path))
        owner.exit()
    replacement = launch(tmp_path)
    assert replacement.start() == {"event": "ready"}
    replacement.return_from_main()
    replacement.exit()


def test_an_ordinary_quit_holds_admission_while_its_backups_land(tmp_path, launch):
    owner = launch(tmp_path, mode="settlement")
    assert owner.start() == {"event": "ready"}
    owner.send("quit")
    assert owner.next() == {"event": "quitting"}
    path = str(tmp_path / "settlement.txt")
    with sqlite3.connect(tmp_path / "backups.sqlite3") as connection:
        rows = connection.execute("SELECT content FROM backups WHERE path=?", (path,)).fetchall()
        assert rows == []
    assert_contended(launch(tmp_path))
    owner.send("settle")
    assert owner.next() == {"event": "returned", "result": 0}
    with sqlite3.connect(tmp_path / "backups.sqlite3") as connection:
        row = connection.execute("SELECT content FROM backups WHERE path=?", (path,)).fetchone()
        assert row == (b"settled\n",)
    owner.exit()


def test_failed_startup_keeps_admission_until_process_exit(tmp_path, launch):
    owner = launch(tmp_path, mode="startup_failure")
    failure = owner.start()
    assert failure["key"] == "startup.storageUnreadable"
    assert "hostile-startup-diagnostic" not in failure["text"]
    owner.return_from_main(failed=True)
    assert_contended(launch(tmp_path))
    owner.exit(code=1)
    replacement = launch(tmp_path)
    assert replacement.start() == {"event": "ready"}
    replacement.return_from_main()
    replacement.exit()


def test_admission_descriptor_is_noninheritable_and_second_bootstrap_is_refused(tmp_path):
    startup_admission.admit_data_root()
    descriptor = startup_admission._admitted_descriptors[-1]
    assert not os.get_inheritable(descriptor)
    with pytest.raises(PixelupError) as raised:
        startup_admission.admit_data_root()
    assert raised.value.code == ErrorCode.APP_ALREADY_RUNNING


def test_native_admission_failure_keeps_the_normal_shell_unmounted(
    tmp_path, monkeypatch, qapp,
):
    # Opening a directory as the claim file fails natively on macOS and Windows.
    # It must reach the safe existing startup surface, with no log/window writer.
    (tmp_path / "writer.lock").mkdir()
    monkeypatch.setattr(gui, "configure_session_logging", lambda: pytest.fail("opened logging"))
    monkeypatch.setattr(gui, "MainWindow", lambda **_: pytest.fail("opened normal window"))
    shown = []
    monkeypatch.setattr(gui, "show_startup_failure", lambda detail, hint: shown.append(detail))
    assert gui.main() == 1
    assert shown == [Message("startup.storageUnreadable")]


def test_native_contention_is_structured_and_keeps_its_cause(tmp_path, launch):
    owner = launch(tmp_path)
    assert owner.start() == {"event": "ready"}
    with pytest.raises(PixelupError) as raised:
        startup_admission.admit_data_root()
    assert raised.value.code == ErrorCode.APP_ALREADY_RUNNING
    assert raised.value.message == Message("startup.alreadyRunning")
    assert isinstance(raised.value.__cause__, OSError)
    owner.return_from_main()
    owner.exit()

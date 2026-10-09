from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from pixelup.formats import NewerFormatError, check_sqlite_format, open_sqlite_store

SCHEMA = "CREATE TABLE facts (value TEXT); CREATE INDEX facts_value ON facts(value);"


@pytest.mark.parametrize("version", [-1, 2])
def test_an_existing_store_without_a_usable_marker_is_left_untouched(
    tmp_path: Path, version: int
) -> None:
    path = tmp_path / "facts.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(f"PRAGMA user_version = {version}")
    before = path.read_bytes()

    with pytest.raises(NewerFormatError if version == 2 else ValueError):
        open_sqlite_store(path, 1, SCHEMA)

    assert path.read_bytes() == before


def test_an_unmarked_store_with_a_schema_is_refused_untouched(tmp_path: Path) -> None:
    path = tmp_path / "facts.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE other (value TEXT)")
    before = path.read_bytes()

    with pytest.raises(ValueError):
        open_sqlite_store(path, 1, SCHEMA)

    assert path.read_bytes() == before


@pytest.mark.parametrize("existing", [None, b""])
def test_a_new_or_empty_file_gets_its_schema_and_marker(
    tmp_path: Path, existing: bytes | None
) -> None:
    # An empty file is what an interrupted creation leaves; it holds nothing, so it
    # is created like a missing one.
    path = tmp_path / "facts.sqlite3"
    if existing is not None:
        path.write_bytes(existing)

    connection = open_sqlite_store(path, 1, SCHEMA)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    finally:
        connection.close()

    reopened = open_sqlite_store(path, 1, SCHEMA)
    reopened.close()
    assert sorted(item.name for item in tmp_path.iterdir() if "lock" in item.name) == []


def test_failed_schema_never_publishes_a_supported_marker(tmp_path: Path) -> None:
    path = tmp_path / "facts.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        open_sqlite_store(path, 1, "CREATE TABLE facts(value TEXT); NOT SQL;")
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute("SELECT name FROM sqlite_master").fetchall() == []
        with pytest.raises(ValueError):
            check_sqlite_format(connection, path, 1)


def test_initialization_keeps_primary_when_rollback_and_close_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = sqlite3.OperationalError("primary schema failure")
    closed: list[bool] = []
    real_connect = sqlite3.connect

    class FailingCleanup:
        def __init__(self, *args, **kwargs):
            self.connection = real_connect(*args, **kwargs)

        def execute(self, sql, *args):
            if sql.strip().startswith("CREATE"):
                raise primary
            return self.connection.execute(sql, *args)

        def rollback(self):
            self.connection.rollback()
            raise sqlite3.OperationalError("cleanup rollback failure")

        def close(self):
            self.connection.close()
            closed.append(True)
            raise sqlite3.OperationalError("cleanup close failure")

    monkeypatch.setattr(sqlite3, "connect", FailingCleanup)
    with pytest.raises(sqlite3.OperationalError) as raised:
        open_sqlite_store(tmp_path / "facts.sqlite3", 1, SCHEMA)
    assert raised.value is primary
    assert closed == [True]
    assert "cleanup rollback failure" in raised.value.__notes__[0]
    assert "cleanup close failure" in raised.value.__notes__[1]

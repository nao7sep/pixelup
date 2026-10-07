from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from pixelup.formats import NewerFormatError, check_sqlite_format, open_sqlite_store

SCHEMA = "CREATE TABLE facts (value TEXT); CREATE INDEX facts_value ON facts(value);"


@pytest.mark.parametrize("version", [0, -1, 2])
def test_existing_empty_schema_is_not_fresh_provenance(tmp_path: Path, version: int) -> None:
    path = tmp_path / "facts.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(f"PRAGMA user_version = {version}")
    before = path.read_bytes()

    with pytest.raises(NewerFormatError if version == 2 else ValueError):
        open_sqlite_store(path, 1, SCHEMA)

    assert path.read_bytes() == before


def test_concurrent_first_open_initializes_one_complete_format(tmp_path: Path) -> None:
    path = tmp_path / "facts.sqlite3"

    def opened() -> tuple[int, int]:
        connection = open_sqlite_store(path, 1, SCHEMA)
        try:
            return (
                connection.execute("PRAGMA user_version").fetchone()[0],
                connection.execute("SELECT count(*) FROM facts").fetchone()[0],
            )
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: opened(), range(2)))
    assert results == [(1, 0), (1, 0)]


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

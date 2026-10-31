from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from pixelup import __version__
from pixelup.formats import (
    Format,
    NewerFormatError,
    check_sqlite_format,
    json_content,
    json_text,
    open_sqlite_store,
)

SCHEMA = "CREATE TABLE facts (value TEXT); CREATE INDEX facts_value ON facts(value);"
FACTS = Format("facts", 1)


def _store(path: Path, version: int, rows: dict[str, str] | None) -> bytes:
    """An existing database with ``rows`` in its ``file_header`` (no table when None)."""
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE facts (value TEXT)")
        if rows is not None:
            connection.execute(
                "CREATE TABLE file_header (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.executemany("INSERT INTO file_header VALUES (?, ?)", rows.items())
        connection.execute(f"PRAGMA user_version = {version}")
    connection.close()
    return path.read_bytes()


OURS = {"format.owner": "pixelup", "format.kind": "facts"}


@pytest.mark.parametrize(
    ("version", "rows"),
    [
        (-1, OURS),
        (0, OURS),
        (1, None),
        (1, {}),
        (1, {"format.owner": "pixelup"}),
        (1, {"format.kind": "facts"}),
        (1, {"format.owner": "picpatch", "format.kind": "facts"}),
        (1, {"format.owner": "pixelup", "format.kind": "records"}),
        (2, {"format.owner": "pixelup", "format.kind": "records"}),
    ],
)
def test_a_store_without_its_own_header_is_unreadable_and_left_untouched(
    tmp_path: Path, version: int, rows: dict[str, str] | None
) -> None:
    path = tmp_path / "facts.sqlite3"
    before = _store(path, version, rows)

    with pytest.raises(ValueError):
        open_sqlite_store(path, FACTS, SCHEMA)

    assert path.read_bytes() == before


def test_a_newer_store_is_left_untouched(tmp_path: Path) -> None:
    path = tmp_path / "facts.sqlite3"
    before = _store(path, 2, OURS)

    with pytest.raises(NewerFormatError) as raised:
        open_sqlite_store(path, FACTS, SCHEMA)

    assert raised.value.version == 2
    assert path.read_bytes() == before


def test_an_admitted_store_gets_its_writer_rows_rewritten_and_never_reads_them(
    tmp_path: Path,
) -> None:
    path = tmp_path / "facts.sqlite3"
    _store(path, 1, {**OURS, "writer.name": "other", "writer.caller.name": "x"})

    open_sqlite_store(path, FACTS, SCHEMA).close()

    with sqlite3.connect(path) as connection:
        rows = dict(connection.execute("SELECT key, value FROM file_header").fetchall())
    connection.close()
    assert rows == {**OURS, "writer.name": "pixelup", "writer.version": __version__}


def test_an_unmarked_store_with_a_schema_is_refused_untouched(tmp_path: Path) -> None:
    path = tmp_path / "facts.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE other (value TEXT)")
    before = path.read_bytes()

    with pytest.raises(ValueError):
        open_sqlite_store(path, FACTS, SCHEMA)

    assert path.read_bytes() == before


@pytest.mark.parametrize("existing", [None, b""])
def test_a_new_or_empty_file_gets_its_schema_and_header(
    tmp_path: Path, existing: bytes | None
) -> None:
    # An empty file is what an interrupted creation leaves; it holds nothing, so it
    # is created like a missing one.
    path = tmp_path / "facts.sqlite3"
    if existing is not None:
        path.write_bytes(existing)

    connection = open_sqlite_store(path, FACTS, SCHEMA)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert dict(connection.execute("SELECT key, value FROM file_header").fetchall()) == {
            **OURS,
            "writer.name": "pixelup",
            "writer.version": __version__,
        }
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    finally:
        connection.close()

    reopened = open_sqlite_store(path, FACTS, SCHEMA)
    reopened.close()
    assert sorted(item.name for item in tmp_path.iterdir() if "lock" in item.name) == []


def test_failed_schema_never_publishes_a_header(tmp_path: Path) -> None:
    path = tmp_path / "facts.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        open_sqlite_store(path, FACTS, "CREATE TABLE facts(value TEXT); NOT SQL;")
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute("SELECT name FROM sqlite_master").fetchall() == []
        with pytest.raises(ValueError):
            check_sqlite_format(connection, path, FACTS)


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
        open_sqlite_store(tmp_path / "facts.sqlite3", FACTS, SCHEMA)
    assert raised.value is primary
    assert closed == [True]
    assert "cleanup rollback failure" in raised.value.__notes__[0]
    assert "cleanup close failure" in raised.value.__notes__[1]


def test_json_text_puts_the_header_first_and_sorts_the_content() -> None:
    text = json_text(FACTS, {"b": {"z": 1, "a": [{"y": 2, "x": 3}]}, "a": 1})
    data = json.loads(text)
    assert list(data) == ["format", "writer", "a", "b"]
    assert data["format"] == {"owner": "pixelup", "kind": "facts", "version": 1}
    assert data["writer"] == {"name": "pixelup", "version": __version__}
    assert list(data["b"]) == ["a", "z"] and list(data["b"]["a"][0]) == ["x", "y"]
    assert text.endswith("}\n")


@pytest.mark.parametrize(
    "data",
    [
        [],
        {"value": 1},
        {"format_version": 1, "value": 1},
        {"format": 1},
        {"format": "pixelup.facts"},
        {"format": {"owner": "pixelup", "kind": "facts"}},
        {"format": {"owner": "pixelup", "kind": "facts", "version": "1"}},
        {"format": {"owner": "pixelup", "kind": "facts", "version": 0}},
        {"format": {"owner": "pixelup", "kind": "facts", "version": True}},
        {"format": {"owner": "pixelup", "kind": 3, "version": 1}},
        {"format": {"owner": "picpatch", "kind": "facts", "version": 1}},
        {"format": {"owner": "pixelup", "kind": "sidecar", "version": 1}},
        {"format": {"owner": "pixelup", "kind": "sidecar", "version": 2}},
    ],
)
def test_json_without_its_own_header_is_unreadable(data: object) -> None:
    with pytest.raises(ValueError):
        json_content(data, FACTS, Path("facts.json"))


def test_newer_json_is_named_newer() -> None:
    data = {"format": {"owner": "pixelup", "kind": "facts", "version": 2}}
    with pytest.raises(NewerFormatError) as raised:
        json_content(data, FACTS, Path("facts.json"))
    assert raised.value.version == 2


@pytest.mark.parametrize("writer", [None, "x", 7, {"name": "picpatch", "version": "9"}])
def test_json_content_ignores_the_writer_and_drops_the_header(writer: object) -> None:
    data = {
        "format": {"owner": "pixelup", "kind": "facts", "version": 1, "later": True},
        "value": 1,
    }
    if writer is not None:
        data["writer"] = writer
    assert json_content(data, FACTS, Path("facts.json")) == {"value": 1}

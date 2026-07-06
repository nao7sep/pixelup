from __future__ import annotations

import json
import os
import re
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from pixelup.backup import (
    IndexRecord,
    latest_records,
    plan_changes,
    run_backup,
)


def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A PIXELUP_HOME the backup engine will resolve; standard subdirs created."""
    root = tmp_path / "home"
    monkeypatch.setenv("PIXELUP_HOME", str(root))
    from pixelup.config import resolve_state_dir

    return resolve_state_dir()


def _write_config(home: Path, text: str = '{"quality": 95}\n') -> Path:
    path = home / "config.json"
    path.write_text(text, encoding="utf-8")
    return path


def _read_index(home: Path) -> list[dict]:
    """The index's `entries` rows (the file is a JSON object wrapping them — the fleet shape)."""
    return json.loads((home / "backups" / "index.json").read_text(encoding="utf-8"))["entries"]


def _archives(home: Path) -> list[Path]:
    return sorted((home / "backups").glob("backup-*.zip"))


def test_first_run_captures_config_with_shared_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    _write_config(home)

    # Non-zero microseconds pin millisecond precision (not just zero-padding).
    report = run_backup(home, now=datetime(2026, 7, 1, 2, 22, 20, 456000, tzinfo=UTC))

    assert report.outcome == "files_archived"
    assert report.files_archived == 1
    assert report.index_was_reset is False

    archives = _archives(home)
    assert len(archives) == 1
    assert archives[0].name == "backup-20260701-022220-456-utc.zip"
    with zipfile.ZipFile(archives[0]) as zf:
        assert zf.namelist() == ["config.json"]

    raw = json.loads((home / "backups" / "index.json").read_text(encoding="utf-8"))
    assert set(raw.keys()) == {"entries"}  # object wrapper (fleet shape), not a bare array
    index = _read_index(home)
    assert len(index) == 1
    row = index[0]
    assert set(row) == {"archivedAt", "archivePath", "sizeBytes", "lastWriteUtc"}
    assert row["archivedAt"] == "20260701-022220-456-utc"
    assert row["archivePath"] == "config.json"
    assert isinstance(row["sizeBytes"], int)
    assert row["lastWriteUtc"].endswith("Z")


def test_nothing_changed_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    _write_config(home)

    run_backup(home, now=datetime(2026, 7, 1, 2, 22, 20, tzinfo=UTC))
    first_index = _read_index(home)

    report = run_backup(home, now=datetime(2026, 7, 1, 3, 0, 0, tzinfo=UTC))

    assert report.outcome == "nothing_changed"
    assert report.files_archived == 0
    # No new archive, index untouched.
    assert len(_archives(home)) == 1
    assert _read_index(home) == first_index


def test_one_file_changed_appends_a_new_row_and_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    config = _write_config(home)

    run_backup(home, now=datetime(2026, 7, 1, 2, 22, 20, tzinfo=UTC))

    # A genuine edit: larger content, mtime pushed well past the 2 s window.
    config.write_text('{"quality": 80, "tile": 512}\n', encoding="utf-8")
    future = (datetime(2026, 7, 1, 2, 22, 20, tzinfo=UTC) + timedelta(hours=1)).timestamp()
    os.utime(config, (future, future))

    report = run_backup(home, now=datetime(2026, 7, 1, 4, 0, 0, tzinfo=UTC))

    assert report.outcome == "files_archived"
    assert report.files_archived == 1
    assert len(_archives(home)) == 2

    index = _read_index(home)
    assert len(index) == 2  # append-only: one row per file archived per run
    assert [r["archivedAt"] for r in index] == [
        "20260701-022220-000-utc",
        "20260701-040000-000-utc",
    ]


def test_archive_and_index_temp_files_use_stem_nanoid_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every atomic write's temp name is <stem>-<nanoid>.tmp, same directory as
    # the target: capture the os.replace(src, dst) pairs to see the transient
    # temp names before they are renamed onto backup-<archivedAt>.zip / index.json.
    home = _home(tmp_path, monkeypatch)
    _write_config(home)
    replaced: dict[str, str] = {}
    original_replace = os.replace

    def capture_replace(src: object, dst: object) -> None:
        replaced[Path(dst).name] = Path(src).name
        original_replace(src, dst)

    monkeypatch.setattr(os, "replace", capture_replace)

    run_backup(home, now=datetime(2026, 7, 1, 2, 22, 20, 456000, tzinfo=UTC))

    zip_temp = replaced["backup-20260701-022220-456-utc.zip"]
    assert re.fullmatch(r"backup-20260701-022220-456-utc-[A-Za-z0-9_-]{21}\.tmp", zip_temp)
    index_temp = replaced["index.json"]
    assert re.fullmatch(r"index-[A-Za-z0-9_-]{21}\.tmp", index_temp)


def test_colliding_stamp_advances_to_next_free_millisecond(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    _write_config(home)

    # Pre-create an archive at the stamp this run would otherwise use (e.g. a
    # second instance of the app that started at the exact same millisecond).
    backups_dir = home / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    (backups_dir / "backup-20260701-022220-456-utc.zip").write_bytes(b"")

    report = run_backup(home, now=datetime(2026, 7, 1, 2, 22, 20, 456000, tzinfo=UTC))

    assert report.outcome == "files_archived"
    # The pre-existing archive is untouched; the new one lands one millisecond later.
    archives = _archives(home)
    assert [a.name for a in archives] == [
        "backup-20260701-022220-456-utc.zip",
        "backup-20260701-022220-457-utc.zip",
    ]
    assert report.archive_path is not None
    assert Path(report.archive_path).name == "backup-20260701-022220-457-utc.zip"

    index = _read_index(home)
    assert len(index) == 1
    assert index[0]["archivedAt"] == "20260701-022220-457-utc"


def test_broken_index_is_reset_and_full_backup_taken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    _write_config(home)
    (home / "backups").mkdir(parents=True, exist_ok=True)
    (home / "backups" / "index.json").write_text("{ this is not valid", encoding="utf-8")

    report = run_backup(home, now=datetime(2026, 7, 1, 2, 22, 20, tzinfo=UTC))

    assert report.index_was_reset is True
    assert report.outcome == "files_archived"
    assert report.files_archived == 1
    # The reset index now holds the fresh full-backup row.
    index = _read_index(home)
    assert len(index) == 1
    assert index[0]["archivePath"] == "config.json"
    # The corrupt index was quarantined, not silently discarded: an
    # index-<ms-utc>.invalid sibling preserves the original bytes, the same
    # quarantine-then-reset discipline config.json uses (storage-path conventions).
    quarantined = list((home / "backups").glob("index-*.invalid"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == "{ this is not valid"


def test_models_and_logs_and_temp_are_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    _write_config(home)
    # Large re-fetchable weights, job staging, logs, and lock files: never archived.
    (home / "models").mkdir(parents=True, exist_ok=True)
    (home / "models" / "RealESRGAN_x4plus.pth").write_bytes(b"x" * 4096)
    (home / "models" / ".locks").mkdir(parents=True, exist_ok=True)
    (home / "models" / ".locks" / "x.lock").write_text("lock", encoding="utf-8")
    (home / "temp").mkdir(parents=True, exist_ok=True)
    (home / "temp" / "job.tmp").write_text("staging", encoding="utf-8")
    (home / "logs").mkdir(parents=True, exist_ok=True)
    (home / "logs" / "session.log").write_text("{}", encoding="utf-8")

    report = run_backup(home, now=datetime(2026, 7, 1, 2, 22, 20, tzinfo=UTC))

    assert report.outcome == "files_archived"
    with zipfile.ZipFile(_archives(home)[0]) as zf:
        names = zf.namelist()
    assert names == ["config.json"]  # only the one durable user-owned file


def test_always_exclude_noise_files_are_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    _write_config(home)
    (home / ".DS_Store").write_text("junk", encoding="utf-8")
    (home / "Thumbs.db").write_text("junk", encoding="utf-8")
    (home / "desktop.ini").write_text("junk", encoding="utf-8")
    (home / "Desktop.ini").write_text("junk", encoding="utf-8")  # OS-noise floor, case-insensitive
    (home / "scratch.tmp").write_text("junk", encoding="utf-8")

    run_backup(home, now=datetime(2026, 7, 1, 2, 22, 20, tzinfo=UTC))

    with zipfile.ZipFile(_archives(home)[0]) as zf:
        assert zf.namelist() == ["config.json"]


def test_plan_changes_is_pure_and_uses_the_2s_window() -> None:
    from pixelup.backup import Candidate

    base = "2026-07-01T02:22:20Z"
    prior = {
        "config.json": IndexRecord("20260701-022220-utc", "config.json", 100, base),
    }
    # Same size, mtime within 2 s -> not changed.
    within = Candidate("config.json", 100, "2026-07-01T02:22:21Z", Path("/x"))
    assert plan_changes([within], prior) == []
    # Same size, mtime beyond 2 s -> changed.
    beyond = Candidate("config.json", 100, "2026-07-01T02:22:25Z", Path("/x"))
    assert plan_changes([beyond], prior) == [beyond]
    # Different size -> changed.
    resized = Candidate("config.json", 200, base, Path("/x"))
    assert plan_changes([resized], prior) == [resized]
    # No prior record -> changed.
    fresh = Candidate("new.json", 10, base, Path("/x"))
    assert plan_changes([fresh], {}) == [fresh]


def test_latest_records_keeps_max_archived_at() -> None:
    records = [
        IndexRecord("20260701-000000-utc", "config.json", 1, "2026-07-01T00:00:00Z"),
        IndexRecord("20260701-050000-utc", "config.json", 2, "2026-07-01T05:00:00Z"),
        IndexRecord("20260701-030000-utc", "config.json", 3, "2026-07-01T03:00:00Z"),
    ]
    latest = latest_records(records)
    assert latest["config.json"].archived_at == "20260701-050000-utc"
    assert latest["config.json"].size_bytes == 2

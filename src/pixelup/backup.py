from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from pixelup.config import resolve_state_dir
from pixelup.session_log import log

# The just-in-case startup backup (data-backup-conventions). A best-effort,
# silent, incremental snapshot of the app's home root (~/.pixelup/), taken at
# app start on a background thread. It never blocks startup, never surfaces an
# error, and never crashes the app: the whole pass is wrapped and swallowed.
#
# Scope is HOME-ROOT-ONLY. PixelUp keeps no externally-linked documents; the one
# durable user-owned file under the root is config.json. Everything else there is
# excluded (see EXCLUDED_DIR_NAMES / _is_excluded): re-fetchable model weights
# (models/), job staging (temp/), the session logs (logs/), and the backups/ dir
# itself, plus the always-exclude noise files.

BACKUPS_DIR_NAME = "backups"
INDEX_FILE_NAME = "index.json"
BACKUP_STAMP_FORMAT = "%Y%m%d-%H%M%S-utc"

# The 2-second window absorbs FAT/exFAT modification-time granularity: two
# lastWriteUtc values within 2 s count as equal (data-backup-conventions).
MTIME_EQUAL_WINDOW_MS = 2000

# Top-level directory names under the home root that are never archived. models/
# and temp/ are app-specific (large re-fetchable weights; job staging); logs/ and
# backups/ are always-exclude by the shared spec.
EXCLUDED_DIR_NAMES = frozenset({BACKUPS_DIR_NAME, "logs", "models", "temp"})

# Always-exclude OS/file-manager noise, matched anywhere in the tree by lowercased base name (the fleet
# floor): a file manager drops these into any directory the user opens.
EXCLUDED_FILE_NAMES = frozenset({".ds_store", "thumbs.db", "desktop.ini"})
EXCLUDED_SUFFIXES = (".tmp",)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One file considered for backup: its archive entry path (home-relative,
    forward-slash) plus the stat used for change detection."""

    archive_path: str
    size_bytes: int
    last_write_utc: str
    source: Path


@dataclass(frozen=True, slots=True)
class IndexRecord:
    """One row of backups/index.json, exactly the shared four-field schema."""

    archived_at: str
    archive_path: str
    size_bytes: int
    last_write_utc: str

    def to_json(self) -> dict[str, Any]:
        return {
            "archivedAt": self.archived_at,
            "archivePath": self.archive_path,
            "sizeBytes": self.size_bytes,
            "lastWriteUtc": self.last_write_utc,
        }


@dataclass(slots=True)
class BackupReport:
    """The outcome of one backup pass, logged and returned to callers/tests."""

    outcome: str = "nothing_changed"  # nothing_changed | files_archived | fatal
    files_archived: int = 0
    skips: list[dict[str, str]] = field(default_factory=list)
    index_was_reset: bool = False
    archive_path: str | None = None
    error: str | None = None


def _is_excluded(relative: PurePosixPath) -> bool:
    """Pure predicate: True iff a home-relative path is excluded from backup.

    Excludes any file whose top-level directory is in EXCLUDED_DIR_NAMES (which
    covers models/.locks/ transitively) and the always-exclude noise files.
    """
    parts = relative.parts
    if parts and parts[0] in EXCLUDED_DIR_NAMES:
        return True
    name = relative.name.lower()
    if name in EXCLUDED_FILE_NAMES:
        return True
    return any(name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES)


def _iso_whole_second(moment: datetime) -> str:
    """Whole-second ISO-8601 UTC with a trailing Z (e.g. 2026-07-01T02:22:20Z)."""
    return moment.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso_ms(value: str) -> int | None:
    """Parse a stored lastWriteUtc back to epoch milliseconds; None if unparseable."""
    try:
        text = value.replace("Z", "+00:00")
        moment = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return int(moment.timestamp() * 1000)


def collect_candidates(home_root: Path) -> tuple[list[Candidate], list[dict[str, str]]]:
    """Walk the home root, stat each non-excluded file, and fold to case-insensitive
    unique archive paths. Returns (candidates, skips). Unreadable files are logged
    skips, not errors; the walk continues.

    This is the collector: the only home-tree I/O in the module.
    """
    candidates: list[Candidate] = []
    skips: list[dict[str, str]] = []
    seen_folded: dict[str, str] = {}

    if not home_root.is_dir():
        return candidates, skips

    for path in sorted(home_root.rglob("*")):
        # Never follow a symlink: skip it (a link is not the app's own data, and following one risks
        # escaping the root). rglob does not recurse into symlinked directories, so a symlinked subtree
        # stays out too. Only real, regular files are archived.
        if path.is_symlink() or not path.is_file():
            continue
        relative = PurePosixPath(path.relative_to(home_root).as_posix())
        if _is_excluded(relative):
            continue
        archive_path = str(relative)
        folded = archive_path.lower()
        if folded in seen_folded:
            skips.append({"reason": "case_insensitive_collision", "archivePath": archive_path})
            continue
        try:
            stat = path.stat()
        except OSError as exc:
            skips.append({"reason": "unreadable", "archivePath": archive_path, "error": str(exc)})
            continue
        seen_folded[folded] = archive_path
        candidates.append(
            Candidate(
                archive_path=archive_path,
                size_bytes=stat.st_size,
                last_write_utc=_iso_whole_second(datetime.fromtimestamp(stat.st_mtime, UTC)),
                source=path,
            )
        )
    return candidates, skips


def latest_records(records: list[IndexRecord]) -> dict[str, IndexRecord]:
    """Pure: the last-known state per archivePath = the record with the maximum
    archivedAt (lexicographic on the yyyymmdd-hhmmss-utc stamp)."""
    latest: dict[str, IndexRecord] = {}
    for record in records:
        current = latest.get(record.archive_path)
        if current is None or record.archived_at > current.archived_at:
            latest[record.archive_path] = record
    return latest


def plan_changes(
    candidates: list[Candidate],
    prior: dict[str, IndexRecord],
) -> list[Candidate]:
    """Pure change-plan function (no I/O). A candidate is captured iff there is no
    prior record, its size differs, or its mtime differs by more than the 2 s
    window from the prior recorded time."""
    changed: list[Candidate] = []
    for candidate in candidates:
        record = prior.get(candidate.archive_path)
        if record is None:
            changed.append(candidate)
            continue
        if candidate.size_bytes != record.size_bytes:
            changed.append(candidate)
            continue
        now_ms = _parse_iso_ms(candidate.last_write_utc)
        prior_ms = _parse_iso_ms(record.last_write_utc)
        if now_ms is None or prior_ms is None or abs(now_ms - prior_ms) > MTIME_EQUAL_WINDOW_MS:
            changed.append(candidate)
    return changed


def load_index(index_path: Path) -> tuple[list[IndexRecord], bool]:
    """Load index.json. Returns (records, was_reset).

    Missing is the normal first run: empty, not reset. Corrupt/unparseable resets
    to empty and reports was_reset=True (the next run is then a full backup).
    """
    if not index_path.exists():
        return [], False
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], True
    if not isinstance(data, dict):
        return [], True
    rows = data.get("entries")
    if not isinstance(rows, list):
        return [], True
    records: list[IndexRecord] = []
    for row in rows:
        if not isinstance(row, dict):
            return [], True
        try:
            records.append(
                IndexRecord(
                    archived_at=str(row["archivedAt"]),
                    archive_path=str(row["archivePath"]),
                    size_bytes=int(row["sizeBytes"]),
                    last_write_utc=str(row["lastWriteUtc"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            return [], True
    return records, False


def _write_archive(backups_dir: Path, archived_at: str, changed: list[Candidate]) -> Path:
    """Write the changed files into a zip at a temp path, then atomically rename
    it to backup-<archived_at>.zip. Archive first, index second (load-bearing)."""
    target = backups_dir / f"backup-{archived_at}.zip"
    temp_path = backups_dir / f".backup-{archived_at}.{os.getpid()}.{uuid4().hex}.tmp"
    try:
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for candidate in changed:
                archive.write(candidate.source, arcname=candidate.archive_path)
        os.replace(temp_path, target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return target


def _write_index(index_path: Path, records: list[IndexRecord]) -> None:
    """Atomically write the whole index (temp then os.replace)."""
    payload = json.dumps({"entries": [record.to_json() for record in records]}, indent=2) + "\n"
    temp_path = index_path.parent / f".{INDEX_FILE_NAME}.{os.getpid()}.{uuid4().hex}.tmp"
    try:
        temp_path.write_text(payload, encoding="utf-8")
        os.replace(temp_path, index_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def run_backup(home_root: Path, *, now: datetime | None = None) -> BackupReport:
    """The engine: load index, collect, plan, write archive, write index.

    Orchestration only; the decisions live in the pure functions above. On the
    skip-empty path (nothing changed) it writes nothing at all.
    """
    report = BackupReport()
    moment = now.astimezone(UTC) if now is not None else datetime.now(UTC)
    archived_at = moment.strftime(BACKUP_STAMP_FORMAT)

    backups_dir = home_root / BACKUPS_DIR_NAME
    index_path = backups_dir / INDEX_FILE_NAME

    records, was_reset = load_index(index_path)
    report.index_was_reset = was_reset

    candidates, skips = collect_candidates(home_root)
    report.skips = skips

    changed = plan_changes(candidates, latest_records(records))
    if not changed:
        report.outcome = "nothing_changed"
        return report

    # Secrets are excluded from backups, so no permission hardening is needed;
    # the backups dir is created with default modes.
    backups_dir.mkdir(parents=True, exist_ok=True)

    target = _write_archive(backups_dir, archived_at, changed)

    new_records = records + [
        IndexRecord(
            archived_at=archived_at,
            archive_path=candidate.archive_path,
            size_bytes=candidate.size_bytes,
            last_write_utc=candidate.last_write_utc,
        )
        for candidate in changed
    ]
    _write_index(index_path, new_records)

    report.outcome = "files_archived"
    report.files_archived = len(changed)
    report.archive_path = str(target)
    return report


def run_startup_backup(home_root: Path | None = None) -> None:
    """Background entry point: best-effort, silent, logged. Never raises.

    Wired from MainWindow.__init__ on a daemon thread (see gui.py). Uses pure file
    I/O only — it must not touch Qt objects from the thread.
    """
    try:
        root = resolve_state_dir(home_root)
        report = run_backup(root)
        if report.outcome == "files_archived":
            log.info(
                "backup.files_archived",
                files=report.files_archived,
                archive=report.archive_path,
                skips=len(report.skips),
                index_was_reset=report.index_was_reset,
            )
        else:
            # The common outcome; at debug so a normal no-op run is silent in production.
            log.debug(
                "backup.nothing_changed",
                skips=len(report.skips),
                index_was_reset=report.index_was_reset,
            )
        for skip in report.skips:
            log.warning("backup.skip", **skip)
    except Exception as exc:  # noqa: BLE001 - best-effort: log and swallow, never crash.
        log.error("backup.fatal", exc_info=True, reason=str(exc))

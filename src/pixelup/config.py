from __future__ import annotations

import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pixelup.errors import ErrorCode, PixelupError
from pixelup.i18n.message import Message
from pixelup.timestamps import utc_now_stamp

APP_NAME = "pixelup"
HOME_ENV = "PIXELUP_DATA_DIR"
MODELS_ENV = "PIXELUP_MODELS_DIR"
TEMP_ENV = "PIXELUP_TEMP_DIR"

# An environment reference in a path override: ${VAR}, $VAR or %VAR%.
_ENV_REF = re.compile(r"\$\{(\w+)\}|\$(\w+)|%(\w+)%")


@dataclass(frozen=True, slots=True)
class RuntimeDirs:
    models_dir: Path
    temp_dir: Path


def resolve_runtime_dirs(
    *,
    models_dir: Path | None = None,
    temp_dir: Path | None = None,
    env: dict[str, str] | None = None,
) -> RuntimeDirs:
    source_env = env if env is not None else os.environ
    return RuntimeDirs(
        models_dir=resolve_models_dir(models_dir, source_env),
        temp_dir=resolve_temp_dir(temp_dir, source_env),
    )


def resolve_models_dir(override: Path | None, env: dict[str, str] | None = None) -> Path:
    return _resolve_dir(override, MODELS_ENV, "models", env)


def saved_models_dir(value: str, default: Path) -> Path:
    """Resolve a saved literal folder without touching a possibly unavailable disk.

    Empty follows the runtime default; relative authored paths are home-relative.
    Picker paths keep literal environment-looking characters. Symlinks need not be
    resolved to use the folder and resolving them here could stall the GUI thread.
    """
    if not value:
        return default
    path = Path(value)
    if path.parts and path.parts[0] == "~":
        path = Path.home().joinpath(*path.parts[1:])
    return Path(os.path.normpath(path if path.is_absolute() else Path.home() / path))


def resolve_temp_dir(override: Path | None, env: dict[str, str] | None = None) -> Path:
    return _resolve_dir(override, TEMP_ENV, "temp", env)


def resolve_state_dir(override: Path | None = None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    return _default_state_dir().resolve()


def window_settings_path() -> Path:
    """Return the native Qt settings file used for disposable window geometry."""
    return resolve_state_dir() / "window.ini"


def ensure_models_dir(path: Path) -> Path:
    return _ensure_dir(path, ErrorCode.MODEL_NOT_FOUND, Message("error.modelsDirCreateFailed"))


def ensure_temp_dir(path: Path) -> Path:
    return _ensure_dir(path, ErrorCode.OUTPUT_UNWRITABLE, Message("error.tempDirCreateFailed"))


def _resolve_dir(
    override: Path | None,
    env_name: str,
    leaf: str,
    env: dict[str, str] | None,
) -> Path:
    if override is not None:
        return _anchored_to_home(override.expanduser())
    source_env = env if env is not None else os.environ
    if env_value := source_env.get(env_name):
        return _anchored_to_home(_expand_override(env_value, env_name, source_env))
    return _default_state_dir(source_env).joinpath(leaf).resolve()


def _anchored_to_home(path: Path) -> Path:
    """``path`` made absolute against the home directory, never the working directory."""
    return (path if path.is_absolute() else Path.home() / path).resolve()


def _default_state_dir(env: dict[str, str] | None = None) -> Path:
    """Resolve the storage root: ``PIXELUP_DATA_DIR`` when set, else ``~/.pixelup``.

    Resolution is lazy on purpose — every call reads the environment afresh — so
    ``PIXELUP_DATA_DIR`` set before launch is honored and tests can vary it without a
    private setter. The override is expanded (leading ``~``, ``$VAR``/``%VAR%``)
    and made absolute against the *home* directory, never the working directory,
    so the override can never reintroduce a cwd dependence. There is exactly one
    root: an unusable home is a reported startup error, never a silent second
    root under the OS-native per-app directory.
    """
    source_env = env if env is not None else os.environ
    override = source_env.get(HOME_ENV, "")
    if override.strip():
        expanded = _expand_override(override.strip(), HOME_ENV, source_env)
        return _ensure_state_root(_anchored_to_home(expanded))
    return _ensure_state_root((Path.home() / f".{APP_NAME}").resolve())


def _expand_override(raw: str, variable: str, env: Mapping[str, str]) -> Path:
    """Expand ``~`` and environment references in the raw value of ``variable``.

    ``$FOO``, ``${FOO}`` and ``%FOO%`` are all expanded on every platform; the host's
    ``os.path.expandvars`` leaves ``%FOO%`` literal on macOS. A reference to a
    variable that is unset or empty does not resolve to the path the user meant, so
    per the storage-path-conventions it is a startup error, never a fallback.
    """
    unresolved = False

    def value(match: re.Match[str]) -> str:
        nonlocal unresolved
        name = next(group for group in match.groups() if group is not None)
        found = env.get(name, "")
        unresolved = unresolved or not found
        return found

    expanded = _ENV_REF.sub(value, os.path.expanduser(raw))
    if not expanded or unresolved:
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE,
            Message.of("error.homeUnusable", variable=variable),
            hint=Message.of("error.hintHomeVariables", variable=variable),
            details={"value": raw, "expanded": expanded},
        )
    return Path(expanded)


def _ensure_state_root(root: Path) -> Path:
    """Create the storage root and its standard subdirectories on first use.

    Raises a clear startup error and stops if the root cannot be created or is
    not a usable directory, rather than relocating data somewhere the user did
    not ask for.
    """
    try:
        # The root is created with the owner-only (0700) mode directly, rather than under the
        # default umask and relying solely on `_tighten_root_permissions` afterward — that
        # sequence would leave a brief window where a freshly created root is world-readable.
        # The mode has no group/other bits, so a stricter umask can only narrow it further,
        # never widen it. Subdirectories keep their existing default-mode creation;
        # `_tighten_root_permissions` below only ever touches the root itself.
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        for path in (root / "logs", root / "models", root / "temp"):
            path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE,
            Message("error.storageCreateFailed"),
            hint=Message.of("error.hintHomeWritableLocation", variable=HOME_ENV),
            details={"path": str(root), "reason": str(exc)},
        ) from exc
    if not root.is_dir():
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE,
            Message("error.storageNotDirectory"),
            hint=Message.of("error.hintHomeWritableDirectory", variable=HOME_ENV),
            details={"path": str(root)},
        )
    _tighten_root_permissions(root)
    return root


def _tighten_root_permissions(root: Path) -> None:
    """Enforce an owner-only (0700) storage root on POSIX.

    Per the storage-path-conventions, the root is created owner-only and
    tightened at each launch when an existing root is broader, because
    derived data and logs must never be readable by accounts that cannot
    read their sources. Windows uses its own permission model and is
    unaffected. A failure to tighten is logged and does not stop the app —
    only the root itself is touched, never its contents.
    """
    if sys.platform == "win32":
        return
    try:
        mode = stat.S_IMODE(root.stat().st_mode)
        if mode & 0o077:
            root.chmod(0o700)
    except OSError as exc:
        print(
            f"pixelup: could not tighten permissions on storage root {root}: {exc}",
            file=sys.stderr,
        )


def _ensure_dir(path: Path, code: ErrorCode, message: Message) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PixelupError(code, message, details={"path": str(path), "reason": str(exc)}) from exc
    if not path.is_dir():
        raise PixelupError(code, message, details={"path": str(path)})
    return path


def quarantine_corrupt_file(path: Path) -> Path:
    """Move a corrupt managed file aside to ``<stem>-<utc>.invalid`` and return the new path.

    The one place PixelUp quarantines an unreadable managed file. The storage-path
    conventions forbid silently discarding a corrupt managed file: the load path may
    only halt or *quarantine-then-reset*, and either way the original bytes are
    preserved. This is that quarantine step, used by the ``config.json`` load path,
    which then runs on the built-ins in memory and writes no replacement file.

    The quarantine name follows the derived-filename grammar
    ``<stem>-<discriminator>.<role-extension>``: the discriminator is a second UTC
    stamp because the moment of quarantine carries meaning, and the role extension
    is ``.invalid`` — the file *is* now an invalid cast-off, so its original
    extension is replaced rather than appended to, keeping the debris out of any
    ``*.json`` scan. The name is claimed with exclusive creation before the move, so
    an earlier quarantine is never replaced: a clash (two launches in one second,
    the unsupported second-instance case) fails like any other read failure and
    leaves the original in place. The move is an atomic same-directory rename onto
    that claimed name, so no interruption can leave a half-copied ``.invalid`` file
    next to a still-corrupt original.
    """
    target = path.with_name(f"{path.stem}-{utc_now_stamp()}.invalid")
    os.close(os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    # not recorded: this is a move-aside of an already-unreadable managed file, not a
    # managed-text write — no new content is produced here, and the corrupt bytes are
    # not a version to preserve in the history (data-backup-conventions).
    try:
        os.replace(path, target)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return target


def write_managed_text(path: Path, text: str) -> None:
    """The single managed-text atomic-write choke point for PixelUp.

    Every durable text file the app owns, including ``config.json``, is written
    through here. A managed-text write that bypasses this helper is a silent backup
    gap; there is deliberately no second atomic-write path for managed text in the app.

    Writes ``text`` (UTF-8) to a temp file the standard library creates exclusively
    beside ``path`` (``<stem>-<random>.tmp``, the storage-path conventions'
    derived-filename grammar), then atomically renames it over ``path``, so a crash
    mid-write cannot corrupt the target. Raises on failure; the caller logs it
    through the session log.

    **The data-backup record fires strictly AFTER the rename lands
    (data-backup-conventions).** Recording before the rename would risk a "backup of
    a save that never happened": if the rename then failed, the history would hold a
    version that never reached disk. So: rename lands, *then* record the exact bytes
    just written — the same ``data`` buffer already in hand, never a re-read of the
    file. The record is best-effort and silent; it never raises back into this
    write and never affects the save's success (see :mod:`pixelup.backup_store`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    descriptor, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.stem}-", suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(data)
        if sys.platform == "darwin":
            _keep_replaced_file_mode(path, temp_path)
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    # After the rename: the file is exactly where it belongs, so record the bytes we
    # just wrote. Imported lazily to avoid a config <-> backup_store import cycle
    # (backup_store resolves its store path through resolve_state_dir here). record()
    # catches, logs once, and swallows every failure, so a backup problem can never
    # break the save that already succeeded above.
    from pixelup.backup_store import record

    record(path, data)


def _keep_replaced_file_mode(path: Path, temp_path: Path) -> None:
    """Keep ordinary permission bits on a macOS replacement through the runtime."""
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        return
    os.chmod(temp_path, mode)

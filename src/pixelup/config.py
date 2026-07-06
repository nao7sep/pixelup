from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from pixelup.errors import ErrorCode, PixelupError
from pixelup.nanoid import nanoid
from pixelup.timestamps import utc_now_stamp_ms

APP_NAME = "pixelup"
HOME_ENV = "PIXELUP_HOME"
MODELS_ENV = "PIXELUP_MODELS_DIR"
TEMP_ENV = "PIXELUP_TEMP_DIR"

# An env reference left over after expansion — $VAR, ${VAR}, or %VAR% — means
# the referenced variable was unset (os.path.expandvars leaves an unset
# reference literal rather than raising).
_UNRESOLVED_ENV_REF = re.compile(r"\$\{\w+\}|\$\w+|%\w+%")


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


def resolve_temp_dir(override: Path | None, env: dict[str, str] | None = None) -> Path:
    return _resolve_dir(override, TEMP_ENV, "temp", env)


def resolve_state_dir(override: Path | None = None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    return _default_state_dir().resolve()


def ensure_models_dir(path: Path) -> Path:
    return _ensure_dir(path, ErrorCode.MODEL_NOT_FOUND, "Could not create the models directory.")


def ensure_temp_dir(path: Path) -> Path:
    return _ensure_dir(path, ErrorCode.OUTPUT_UNWRITABLE, "Could not create the temp directory.")


def _resolve_dir(
    override: Path | None,
    env_name: str,
    leaf: str,
    env: dict[str, str] | None,
) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    source_env = env if env is not None else os.environ
    if env_value := source_env.get(env_name):
        return Path(env_value).expanduser().resolve()
    return _default_state_dir(source_env).joinpath(leaf).resolve()


def _default_state_dir(env: dict[str, str] | None = None) -> Path:
    """Resolve the storage root: ``PIXELUP_HOME`` when set, else ``~/.pixelup``.

    Resolution is lazy on purpose — every call reads the environment afresh — so
    ``PIXELUP_HOME`` set before launch is honored and tests can vary it without a
    private setter. The override is expanded (leading ``~``, ``$VAR``/``%VAR%``)
    and made absolute against the *home* directory, never the working directory,
    so the override can never reintroduce a cwd dependence. There is exactly one
    root: an unusable home is a reported startup error, never a silent second
    root under the OS-native per-app directory.
    """
    source_env = env if env is not None else os.environ
    override = source_env.get(HOME_ENV, "")
    if override.strip():
        root = _expand_home_override(override.strip())
        if not root.is_absolute():
            root = Path.home() / root
        return _ensure_state_root(root.resolve())
    return _ensure_state_root((Path.home() / f".{APP_NAME}").resolve())


def _expand_home_override(raw: str) -> Path:
    """Expand ``~`` and environment references in a raw ``PIXELUP_HOME`` value.

    An unset variable referenced in the value (``$FOO``, ``${FOO}``, ``%FOO%``)
    is left literal by ``os.path.expandvars`` rather than raising, and a
    variable that is set but empty silently expands to nothing — which, once
    the caller falls back to anchoring a non-absolute result against the home
    directory, would otherwise collapse the storage root onto bare ``$HOME``.
    Per the storage-path-conventions, an override that does not resolve to a
    usable directory is a startup error, never a silent fallback, so both
    cases raise here instead of producing a path.
    """
    expanded = os.path.expandvars(os.path.expanduser(raw))
    if not expanded or _UNRESOLVED_ENV_REF.search(expanded):
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE,
            f"{HOME_ENV} does not expand to a usable path.",
            hint=(
                f"Check that every environment variable referenced in {HOME_ENV} "
                "is set to a non-empty value."
            ),
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
        for path in (root, root / "logs", root / "models", root / "temp"):
            path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE,
            "Could not create the PixelUp storage directory.",
            hint=f"Set {HOME_ENV} to a writable location, or make the path usable.",
            details={"path": str(root), "reason": str(exc)},
        ) from exc
    if not root.is_dir():
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE,
            "The PixelUp storage path is not a usable directory.",
            hint=f"Set {HOME_ENV} to a writable directory.",
            details={"path": str(root)},
        )
    return root


def _ensure_dir(path: Path, code: ErrorCode, message: str) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PixelupError(code, message, details={"path": str(path), "reason": str(exc)}) from exc
    if not path.is_dir():
        raise PixelupError(code, message, details={"path": str(path)})
    return path


def quarantine_corrupt_file(path: Path) -> Path:
    """Move a corrupt managed file aside to ``<stem>-<ms-utc>.invalid`` and return the new path.

    The one place PixelUp quarantines an unreadable managed file. The storage-path
    conventions forbid silently discarding a corrupt managed file: the load path may
    only halt or *quarantine-then-reset*, and either way the original bytes are
    preserved. This is that quarantine step, used by the ``config.json`` load path
    when it resets a corrupt file to defaults.

    The quarantine name follows the derived-filename grammar
    ``<stem>-<discriminator>.<role-extension>``: the discriminator is a millisecond
    UTC stamp (``yyyymmdd-hhmmss-fff-utc``) because the moment of quarantine carries
    meaning, and the role extension is ``.invalid`` — the file *is* now an invalid
    cast-off, so its original extension is replaced rather than appended to, keeping
    the debris out of any ``*.json`` scan. On the rare same-millisecond collision the
    stem gains a nanoid so a second quarantine never clobbers the first. The move is
    an atomic same-directory rename, so no interruption can leave a half-copied
    ``.invalid`` file next to a still-corrupt original.
    """
    target = path.with_name(f"{path.stem}-{utc_now_stamp_ms()}.invalid")
    if target.exists():
        target = path.with_name(f"{path.stem}-{utc_now_stamp_ms()}-{nanoid()}.invalid")
    # not recorded: this is a move-aside of an already-unreadable managed file, not a
    # managed-text write — no new content is produced here, and the corrupt bytes are
    # not a version to preserve in the history (the store never captured them, so
    # there is nothing to add). The subsequent fresh save through write_managed_text
    # is what records the recovered-to-defaults content (data-backup-conventions).
    os.replace(path, target)
    return target


def write_managed_text(path: Path, text: str) -> None:
    """The single managed-text atomic-write choke point for PixelUp.

    Every durable text file the app owns is written through here — today that is
    exactly ``config.json`` (:func:`pixelup.app_config.save_app_config`), the app's
    one managed text store. A managed-text write that bypasses this helper is a
    silent backup gap; there is deliberately no second atomic-write path for managed
    text in the app.

    Writes ``text`` (UTF-8) to a same-directory temp named ``<stem>-<nanoid>.tmp``
    (the storage-path conventions' derived-filename grammar — the nanoid guarantees
    two concurrent writers never share a temp), then atomically renames it over
    ``path``, so a crash mid-write cannot corrupt the target. Raises on failure; the
    caller logs it through the session log.

    **The data-backup record fires strictly AFTER the rename lands
    (data-backup-conventions).** Recording before the rename would risk a "backup of
    a save that never happened": if the rename then failed, the history would hold a
    version that never reached disk. So: rename lands, *then* record the exact bytes
    just written — the same ``data`` buffer already in hand, never a re-read of the
    file (which would risk capturing a concurrent writer's content, not what this
    call wrote). The record is best-effort and silent; it never raises back into this
    write and never affects the save's success (see :mod:`pixelup.backup_store`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    temp_path = path.with_name(f"{path.stem}-{nanoid()}.tmp")
    try:
        temp_path.write_bytes(data)
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

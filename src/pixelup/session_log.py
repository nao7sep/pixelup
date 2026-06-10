from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pixelup.config import resolve_state_dir
from pixelup.timestamps import to_utc_iso_ms

LOGGER_NAME = "pixelup"
DEBUG_ENV = "PIXELUP_DEBUG"

# Field names whose values are masked before a record is serialized. Matched by
# exact, case-insensitive name — never by substring — so the redactor can only
# ever blank a value, never corrupt surrounding content. PixelUp handles no
# secrets today; this is the standing backstop the logging convention requires.
_REDACTED_KEYS = frozenset({"apikey", "api_key", "authorization", "token", "password", "secret"})
_REDACTED_PLACEHOLDER = "[redacted]"

# Keys a caller's structured field must never overwrite: the three-part envelope
# plus `error`, which is reserved for the attached-exception payload.
_RESERVED_FIELDS = frozenset({"time", "level", "message", "error"})

# stdlib level -> the four convention level names (WARNING renders as "warn").
_LEVEL_NAMES = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    logging.WARNING: "warn",
    logging.ERROR: "error",
    logging.CRITICAL: "error",
}


def redact_log_fields(value: Any) -> Any:
    """Return log fields with sensitive values masked, structure untouched.

    Denied field names are matched by exact, case-insensitive name (never by
    substring), so ``token`` never matches ``tokenCount``. Only a matched value
    is replaced with a fixed marker; mappings and lists are recursed; every other
    value is returned as-is. Pure and total: it never raises and never drops
    fields, so it cannot corrupt a log line.
    """
    if isinstance(value, Mapping):
        return {
            key: (
                _REDACTED_PLACEHOLDER
                if isinstance(key, str) and key.lower() in _REDACTED_KEYS
                else redact_log_fields(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_log_fields(item) for item in value]
    return value


def _error_object(exc_info: Any) -> dict[str, Any]:
    """Full-fidelity error payload: type, message, and the formatted traceback.

    ``traceback.format_exception`` already walks the ``__cause__`` / ``__context__``
    chain, so the rendered traceback carries the cause chain the convention asks for.
    """
    if not isinstance(exc_info, tuple) or exc_info[1] is None:
        return {}
    exc_type, exc, tb = exc_info
    return {
        "type": (exc_type or type(exc)).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(exc_type, exc, tb)).rstrip(),
    }


def _dumps(entry: dict[str, Any]) -> str:
    """Serialize a log entry to one JSON line that can never fail.

    The happy path leans on ``default=str`` to coerce stray non-serializable
    values. If anything still defeats serialization — most often a nested dict
    with non-string keys — the entry is coerced field-by-field into a
    guaranteed-serializable form rather than letting the exception bubble up and
    drop the line. A log line must never be lost.
    """
    try:
        return json.dumps(entry, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return json.dumps(_json_safe(entry), ensure_ascii=False)


def _json_safe(value: Any) -> Any:
    """Total coercion to a JSON-serializable shape: objects become string-keyed,
    scalars pass through, everything else is stringified. Never raises."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return str(value)
    except Exception:
        return f"<unrenderable {type(value).__name__}>"


class JsonlFormatter(logging.Formatter):
    """Render each record as one compact JSON object: the convention envelope
    (``time``, ``level``, ``message``) plus the caller's structured fields, plus
    an ``error`` object carrying full exception fidelity when one is attached."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "time": to_utc_iso_ms(datetime.fromtimestamp(record.created, UTC)),
            "level": _LEVEL_NAMES.get(record.levelno, record.levelname.lower()),
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, Mapping):
            for key, value in redact_log_fields(fields).items():
                if key not in _RESERVED_FIELDS:
                    entry[key] = value
        if record.exc_info:
            error = _error_object(record.exc_info)
            if error:
                entry["error"] = error
        return _dumps(entry)


class SessionLog:
    """The app-wide structured logger. Callers pass a short, stable ``message``
    plus keyword fields; the handler serializes them as one JSON line. ``message``
    is the greppable event identity, fields carry the data — never build a
    pre-formatted string."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def debug(self, message: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, message, fields)

    def info(self, message: str, **fields: Any) -> None:
        self._emit(logging.INFO, message, fields)

    def warning(self, message: str, **fields: Any) -> None:
        self._emit(logging.WARNING, message, fields)

    def error(self, message: str, *, exc_info: Any = None, **fields: Any) -> None:
        self._emit(logging.ERROR, message, fields, exc_info=exc_info)

    def exception(self, message: str, **fields: Any) -> None:
        """Log at error level with the exception currently being handled; call
        from inside an ``except`` block."""
        self._emit(logging.ERROR, message, fields, exc_info=True)

    def _emit(
        self,
        level: int,
        message: str,
        fields: Mapping[str, Any],
        *,
        exc_info: Any = None,
    ) -> None:
        if not self._logger.isEnabledFor(level):
            return
        self._logger.log(level, message, extra={"fields": dict(fields)}, exc_info=exc_info)


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


# Process-wide singleton. Import this and call log.info(...) etc.; before
# configure_session_logging() runs it falls back to stdlib's last-resort handler.
log = SessionLog(get_logger())


def debug_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Whether developer ``debug`` logging is on. Off by default — debug is for a
    developer diagnosing the app and must never be emitted on end-user machines.
    Enable it only by setting ``PIXELUP_DEBUG`` to a truthy value."""
    source = env if env is not None else os.environ
    return source.get(DEBUG_ENV, "").strip().lower() not in {"", "0", "false", "no", "off"}


def session_log_path(*, state_dir: Path | None = None, now: datetime | None = None) -> Path:
    moment = now.astimezone(UTC) if now is not None else datetime.now(UTC)
    stamp = moment.strftime("%Y%m%d-%H%M%S-utc")
    root = resolve_state_dir(state_dir)
    return root / "logs" / f"{stamp}.log"


def configure_session_logging(log_path: Path | None = None) -> Path:
    """Open this launch's session log and route the app logger to it.

    Writes one JSON object per line. Debug is enabled only when PIXELUP_DEBUG is
    set. If the log file cannot be opened, logging degrades to stderr rather than
    failing the launch — best effort, surfaced, never silently swallowed.
    """
    path = log_path or session_log_path()

    logger = get_logger()
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    logger.setLevel(logging.DEBUG if debug_enabled() else logging.INFO)
    logger.propagate = False

    handler = _build_file_handler(path)
    degraded = handler is None
    if handler is None:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonlFormatter())
    logger.addHandler(handler)

    _install_excepthook()
    if degraded:
        log.warning("log.file_unavailable", log_path=str(path), fallback="stderr")
    else:
        log.info("log.session_started", log_path=str(path), debug=debug_enabled())
    return path


def _build_file_handler(path: Path) -> logging.FileHandler | None:
    # The file is a StreamHandler under the hood, so every record is flushed on
    # emit — the convention's "flush warn/error/debug immediately" holds for free.
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return logging.FileHandler(path, encoding="utf-8")
    except OSError:
        return None


# The interpreter's excepthook from before PixelUp first wrapped it. Captured
# once so repeated configure_session_logging() calls re-install our hook without
# chaining onto a previous copy of itself — which would log a single crash once
# per configuration.
_BASE_EXCEPTHOOK: Any = None


def _install_excepthook() -> None:
    global _BASE_EXCEPTHOOK
    if _BASE_EXCEPTHOOK is None:
        _BASE_EXCEPTHOOK = sys.excepthook

    def hook(exc_type: type[BaseException], exc: BaseException, tb: object) -> None:
        log.error("unhandled.exception", exc_info=(exc_type, exc, tb))
        _BASE_EXCEPTHOOK(exc_type, exc, tb)

    sys.excepthook = hook

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from pixelup.config import resolve_state_dir

LOGGER_NAME = "pixelup"


class UtcIsoFormatter(logging.Formatter):
    def formatTime(  # noqa: N802
        self,
        record: logging.LogRecord,
        datefmt: str | None = None,
    ) -> str:
        del datefmt
        timestamp = datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds")
        return timestamp.replace("+00:00", "Z")


def session_log_path(*, state_dir: Path | None = None, now: datetime | None = None) -> Path:
    moment = now.astimezone(UTC) if now is not None else datetime.now(UTC)
    stamp = moment.strftime("%Y%m%d-%H%M%S-utc")
    root = resolve_state_dir(state_dir)
    return root / "logs" / f"{stamp}.log"


def configure_session_logging(log_path: Path | None = None) -> Path:
    path = log_path or session_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = get_logger()
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    logger.setLevel(logging.INFO)
    logger.propagate = False

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(UtcIsoFormatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)

    _install_excepthook(logger)
    logger.info("Session started log_path=%s", path)
    return path


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def _install_excepthook(logger: logging.Logger) -> None:
    previous = sys.excepthook

    def hook(exc_type: type[BaseException], exc: BaseException, tb: object) -> None:
        logger.exception("Unhandled exception", exc_info=(exc_type, exc, tb))
        previous(exc_type, exc, tb)

    sys.excepthook = hook

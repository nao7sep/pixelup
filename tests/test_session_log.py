from datetime import UTC, datetime
from pathlib import Path

from pixelup.session_log import configure_session_logging, get_logger, session_log_path


def test_session_log_path_uses_utc_timestamp(tmp_path: Path) -> None:
    path = session_log_path(
        state_dir=tmp_path,
        now=datetime(2026, 5, 7, 5, 43, 21, tzinfo=UTC),
    )

    assert path == tmp_path / "logs" / "20260507-054321-utc.log"


def test_configure_session_logging_writes_log_file(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "session.log"

    configure_session_logging(log_path)
    logger = get_logger()
    logger.info("hello from test")
    for handler in logger.handlers:
        handler.flush()

    assert "hello from test" in log_path.read_text(encoding="utf-8")

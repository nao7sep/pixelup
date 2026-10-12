"""One admitted writer per actual data root, for the whole process lifetime."""

from __future__ import annotations

import errno
import os

from pixelup.config import resolve_state_dir
from pixelup.errors import ErrorCode, PixelupError
from pixelup.i18n.message import Message

# Keep raw, noninheritable descriptors rather than file/Qt objects: their
# destructors could release admission while a background writer still settles
# during shutdown. The OS closes these descriptors only when the process exits,
# including a crash or the quit watchdog's forced exit. There is no unlock path.
# The list documents deliberate lifetime ownership; integers have no destructor
# that closes their descriptors during interpreter finalization.
_admitted_descriptors: list[int] = []
_CLAIM_FILE = "writer.lock"


def admit_data_root() -> None:
    """Claim before opening any shared settings, records, backups or log writer.

    Called on the startup thread. The persistent claim file is never unlinked or
    replaced: an age, stale PID or a cleanup race cannot admit a second live
    writer. This uses native local-filesystem exclusion on macOS and Windows;
    it does not coordinate external editors or roots shared across machines.
    Unsupported locking fails startup rather than allowing an unguarded writer.
    """
    root = resolve_state_dir()
    # os.open returns a noninheritable descriptor. No truncate or payload write:
    # Windows permits a byte-range lock beyond the end of this empty file.
    descriptor = os.open(root / _CLAIM_FILE, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException as exc:
        os.close(descriptor)
        if isinstance(exc, OSError) and exc.errno in (errno.EACCES, errno.EAGAIN):
            raise PixelupError(
                ErrorCode.APP_ALREADY_RUNNING,
                Message("startup.alreadyRunning"),
                details={"data_root": str(root)},
            ) from exc
        raise
    _admitted_descriptors.append(descriptor)

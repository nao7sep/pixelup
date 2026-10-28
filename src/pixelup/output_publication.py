"""Publish a complete staged member under its checked final name."""

from __future__ import annotations

import errno
import os
from pathlib import Path


def publish_complete(stage: Path, target: Path) -> Path:
    try:
        os.link(stage, target, follow_symlinks=False)
    except OSError as exc:
        if exc.errno not in {
            errno.EACCES, errno.EINVAL, errno.ENOSYS, errno.ENOTSUP,
            errno.EOPNOTSUPP, errno.EPERM,
        }:
            raise
        # Without hard links the stage takes the name checked just before
        # publication by an ordinary rename.
        os.rename(stage, target)
    return target

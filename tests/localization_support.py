"""Temporary interface language changes for tests that inspect localized surfaces."""

from collections.abc import Iterator
from contextlib import contextmanager

from pixelup.i18n import localizer


@contextmanager
def speaking(tag: str) -> Iterator[None]:
    """Restore the previous language and preference even when a test raises."""
    previous_tag = localizer.language()
    previous_preference = localizer.preference()
    localizer._switch(tag)
    try:
        yield
    finally:
        localizer._preference = previous_preference
        if localizer.language() != previous_tag:
            localizer._switch(previous_tag)

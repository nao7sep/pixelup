from __future__ import annotations

import re
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
AREA_MAP = TESTS_DIR / "README.md"
BACKTICKED = re.compile(r"`([^`]+)`")


def _areas() -> dict[str, list[str]]:
    """The area map's table, as area name to the test paths standing for that area."""
    rows = [
        line
        for line in AREA_MAP.read_text(encoding="utf-8").splitlines()
        if line.startswith("|")
    ][2:]  # Past the header row and the row of dashes.
    return {
        cells[0].strip(): BACKTICKED.findall(cells[2])
        for cells in (row.split("|")[1:-1] for row in rows)
    }


def test_the_map_names_more_than_one_area() -> None:
    assert len(_areas()) > 1


def test_every_area_names_at_least_one_test() -> None:
    unrepresented = [area for area, paths in _areas().items() if not paths]

    assert not unrepresented, f"areas with no test standing for them: {unrepresented}"


def test_every_listed_test_exists() -> None:
    missing = sorted(
        path for paths in _areas().values() for path in paths if not (TESTS_DIR / path).exists()
    )

    assert not missing, f"tests/README.md names tests that no longer exist: {missing}"

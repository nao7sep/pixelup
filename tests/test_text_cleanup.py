import pytest

from pixelup.text_cleanup import single_line


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  Arial  ", "Arial"),
        ("   ", ""),
        ("　Arial　", "Arial"),
        ("Menlo,\n  Arial", "Menlo, Arial"),
        ("Menlo,\r\n\r\nArial", "Menlo, Arial"),
        ("Helvetica  Neue", "Helvetica  Neue"),
    ],
)
def test_single_line_flattens_breaks_and_trims_ends(text: str, expected: str) -> None:
    assert single_line(text) == expected

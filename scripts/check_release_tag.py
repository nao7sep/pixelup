from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

PROJECT_FILE = Path(__file__).resolve().parents[1] / "pyproject.toml"


def expected_release_tag(project_file: Path = PROJECT_FILE) -> str:
    with project_file.open("rb") as file:
        project = tomllib.load(file)["project"]
    return f"v{project['version']}"


def validate_release_tag(tag: str, project_file: Path = PROJECT_FILE) -> None:
    expected = expected_release_tag(project_file)
    if tag != expected:
        raise ValueError(f"Release tag {tag!r} does not match project version {expected!r}.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a release tag against pyproject.toml.")
    parser.add_argument("tag")
    args = parser.parse_args(argv)
    try:
        validate_release_tag(args.tag)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

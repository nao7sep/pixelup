#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

log_step() {
  printf '\n==> %s\n' "$1"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

pause_on_failure() {
  local status="$1"
  if [[ "$status" -ne 0 && "$status" -ne 130 ]]; then
    echo
    echo "pixelup update-packages failed with exit code $status."
    read -r -p "Press Enter to close..."
  fi
}

trap 'pause_on_failure $?' EXIT

require_command uv

cd "$REPO_DIR"

log_step "Updating locked packages within declared constraints"
uv lock --upgrade

log_step "Installing updated dependencies"
uv sync --extra dev

log_step "Running lint"
uv run --extra dev ruff check .

log_step "Running tests"
uv run --extra dev pytest -q

log_step "Checking lockfile"
uv lock --check

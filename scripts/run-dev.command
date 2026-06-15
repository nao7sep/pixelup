#!/usr/bin/env bash
set -euo pipefail

# run-dev: run the CLI from source (uv). A Python CLI has no separate production
# build to launch, so this is its only launcher — uv runs the source directly.

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
    echo "pixelup run-dev failed with exit code $status."
    read -r -p "Press Enter to close..."
  fi
}

trap 'pause_on_failure $?' EXIT

require_command uv

cd "$REPO_DIR"

log_step "Installing dependencies required for launch"
uv sync --extra dev

log_step "Starting PixelUp"
uv run pixelup "$@"

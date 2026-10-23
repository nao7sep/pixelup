#!/usr/bin/env bash
set -euo pipefail

# run-dev: run PixelUp (a PySide6 GUI app) from source via uv. This is the fast
# dev-loop launcher; rebuild and run-built cover the frozen PyInstaller build.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNTIME_TOKEN="run-dev-$$-$(date +%s)-$RANDOM"
source "$SCRIPT_DIR/launcher-runtime.sh"

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
  if [[ "$status" -ne 0 && ( "$status" -lt 128 || "$status" -gt 143 ) ]]; then
    echo
    echo "pixelup run-dev failed with exit code $status."
    read -r -p "Press Enter to close..."
  fi
}

cleanup() {
  local status="$?"
  trap - EXIT
  if is_launcher_runtime_owner "$RUNTIME_TOKEN" "$REPO_DIR"; then
    stop_owned_runtime PixelUp "$REPO_DIR" "$RUNTIME_TOKEN" "$RUNTIME_TOKEN" >/dev/null 2>&1 || true
    release_launcher_runtime "$RUNTIME_TOKEN" "$REPO_DIR"
  fi
  pause_on_failure "$status"
  exit "$status"
}

trap cleanup EXIT

require_command uv

cd "$REPO_DIR"

log_step "Replacing any existing PixelUp runtime"
claim_launcher_runtime "$RUNTIME_TOKEN" "$REPO_DIR"
stop_owned_runtime PixelUp "$REPO_DIR" "$RUNTIME_TOKEN"

log_step "Installing dependencies required for launch"
uv sync --extra dev

log_step "Starting PixelUp"
start_owned_runtime "$REPO_DIR" "$RUNTIME_TOKEN" "$REPO_DIR/.venv/bin/python" -m pixelup "$@" || {
  status=$?
  [[ "$status" -eq 2 ]] && exit 0
  exit "$status"
}
DEV_PID=$LAUNCHED_RUNTIME_PID
wait_for_captured_runtime PixelUp "$DEV_PID" "$LAUNCHED_RUNTIME_IDENTITY" 120
log_step "PixelUp is ready"
wait "$DEV_PID"

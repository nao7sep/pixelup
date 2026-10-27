#!/usr/bin/env bash
set -euo pipefail

# run-dev: run PixelUp (a PySide6 GUI app) from source via uv. This is the fast
# dev-loop launcher; rebuild and run-built cover the frozen PyInstaller build.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_PID=""
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
  # Closing this launcher closes the PixelUp it started, unless a later launcher has replaced it.
  if [[ -n "$APP_PID" ]]; then
    stop_recorded_runtime "$APP_PID" >/dev/null 2>&1 || true
  fi
  pause_on_failure "$status"
  exit "$status"
}

trap cleanup EXIT

require_command uv

cd "$REPO_DIR"

log_step "Replacing any existing PixelUp runtime"
stop_recorded_runtime

log_step "Installing dependencies required for launch"
uv sync --extra dev

log_step "Starting PixelUp"
# A framework Python re-executes itself as its app bundle's interpreter, so ask
# the interpreter which executable its process ends up running.
python_executable="$("$REPO_DIR/.venv/bin/python" -c 'import os, subprocess; print(subprocess.run(["ps", "-p", str(os.getpid()), "-o", "comm="], capture_output=True, text=True, check=True).stdout.strip())')"
start_recorded_runtime "$python_executable" "$REPO_DIR/.venv/bin/python" -m pixelup "$@"
APP_PID=$RUNTIME_PID
log_step "PixelUp is ready"
wait "$APP_PID"

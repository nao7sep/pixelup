#!/usr/bin/env bash
set -euo pipefail

# run-built: launch the EXISTING frozen PixelUp.app without rebuilding, so it
# starts instantly. This is the daily-use launcher and the one that surfaces
# frozen-only failures (bundled resource paths, missing hidden imports) that the
# from-source run-dev never sees. It never freezes — if you changed source, run
# rebuild first.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_BUNDLE="$REPO_DIR/dist/PixelUp.app"

log_step() {
  printf '\n==> %s\n' "$1"
}

pause_on_failure() {
  local status="$1"
  if [[ "$status" -ne 0 && "$status" -ne 130 ]]; then
    echo
    echo "pixelup run-built failed with exit code $status."
    read -r -p "Press Enter to close..."
  fi
}

trap 'pause_on_failure $?' EXIT

cd "$REPO_DIR"

# No freeze here: this launcher must start instantly. If there is no usable bundle
# yet, stop and point at rebuild rather than launching something stale or empty.
if [[ ! -d "$APP_BUNDLE/Contents/MacOS" ]]; then
  echo "No build found — run rebuild first."
  exit 1
fi

built_at="$(stat -f '%Sm' -t '%Y-%m-%d %H:%M:%S %Z' "$APP_BUNDLE/Contents/MacOS/PixelUp" 2>/dev/null || echo 'unknown')"
log_step "Launching the existing build (built: $built_at)"
echo "If you changed source since then, run rebuild instead."

open "$APP_BUNDLE"

#!/usr/bin/env bash
set -euo pipefail

# rebuild: freeze a fresh PixelUp.app with PyInstaller and launch it — the same
# frozen bundle the release pipeline builds (scripts/package.sh wraps this same
# output into the .dmg/.zip; rebuild stops at the launchable .app, no installer).
# Slow; run after changing source. run-built is the no-build fast path afterward.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_BUNDLE="$REPO_DIR/dist/PixelUp.app"

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
    echo "pixelup rebuild failed with exit code $status."
    read -r -p "Press Enter to close..."
  fi
}

trap 'pause_on_failure $?' EXIT

require_command uv

cd "$REPO_DIR"

# Clear stale output first so a build that fails to emit a file cannot be masked
# by a leftover artifact from a previous run.
log_step "Removing stale build output"
rm -rf "$REPO_DIR/dist" "$REPO_DIR/build-pyinstaller"

log_step "Freezing PixelUp.app (uv installs the build deps, then PyInstaller runs)"
uv run --extra build pyinstaller pixelup.spec --workpath build-pyinstaller --distpath dist --noconfirm

log_step "Finalizing the macOS bundle"
"$SCRIPT_DIR/finalize-macos-bundle.sh" "$APP_BUNDLE"

# Fail fast if freezing dropped a lazily-imported dependency (see gui._selftest),
# rather than launching a bundle that crashes only on first upscale.
log_step "Self-testing the frozen bundle"
PIXELUP_SELFTEST=1 "$APP_BUNDLE/Contents/MacOS/PixelUp"

log_step "Launching PixelUp"
open "$APP_BUNDLE"

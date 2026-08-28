#!/usr/bin/env bash
set -euo pipefail

# Complete a freshly frozen macOS bundle with the committed Liquid Glass catalog,
# then restore the ad-hoc signature invalidated by adding that resource.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_BUNDLE="${1:?usage: finalize-macos-bundle.sh <app-bundle>}"
CAR="$REPO_DIR/build/Assets.car"

[ -d "$APP_BUNDLE" ] || { echo "missing app bundle: $APP_BUNDLE" >&2; exit 1; }
[ -f "$CAR" ] || {
  echo "missing $CAR — regenerate via the liquid-glass-icon tool (apps/pixelup.mjs build + deploy)" >&2
  exit 1
}

cp "$CAR" "$APP_BUNDLE/Contents/Resources/Assets.car"
codesign --force --deep --sign - "$APP_BUNDLE"

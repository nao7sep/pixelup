#!/usr/bin/env bash
set -euo pipefail

# pixelup macOS packaging: freeze PixelUp.app with PyInstaller, then emit the two
# release artifacts into dist/ — pixelup-<version>.dmg (installer, with an
# /Applications drop target) and pixelup-<version>-mac.zip (the portable .app).
# Runnable locally and by CI (macos-latest); needs uv + macOS's hdiutil/ditto.
# Per the app-release-conventions the packaging lives here so CI just calls it.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

APP="PixelUp"
DIST="dist"
VERSION="$(uv run python -c "import tomllib, pathlib; print(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['version'])")"

rm -rf "$DIST" build-pyinstaller
# uv run --extra build auto-syncs the runtime + build (PyInstaller) deps first.
uv run --extra build pyinstaller pixelup.spec --workpath build-pyinstaller --distpath "$DIST" --noconfirm

# Fail fast if freezing dropped a lazily-imported dependency (see gui._selftest).
PIXELUP_SELFTEST=1 "$DIST/$APP.app/Contents/MacOS/$APP"

# Installer .dmg: stage the .app beside an /Applications symlink for drag-to-install,
# then compress (UDZO).
STAGE="$(mktemp -d)"
cp -R "$DIST/$APP.app" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
DMG="$DIST/pixelup-$VERSION.dmg"
rm -f "$DMG"
hdiutil create -volname "$APP" -srcfolder "$STAGE" -ov -format UDZO "$DMG"
rm -rf "$STAGE"

# Portable: the .app zipped as-is (ditto preserves the bundle and its symlinks).
ditto -c -k --keepParent "$DIST/$APP.app" "$DIST/pixelup-$VERSION-mac.zip"

ls -lh "$DIST"

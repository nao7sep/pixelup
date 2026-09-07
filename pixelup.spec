# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for PixelUp — the fleet's Python/PySide6 GUI freezer.

One spec drives both platforms: on macOS it emits ``PixelUp.app`` (a BUNDLE);
on Windows a ``PixelUp/`` onedir holding ``PixelUp.exe``. The wrapping into
release artifacts (.dmg + portable zip on mac; Inno ``setup.exe`` + portable zip
on win) is done by ``scripts/package.sh`` / ``scripts/package.ps1`` — this spec
only produces the frozen app, per the app-release-conventions (complexity in the
repo, not in CI).

The small Real-ESRGAN runtime lives inside PixelUp and uses ordinary imports.
PyInstaller's maintained hooks own torch, cv2, PySide6, and Pillow. A packaged
self-test imports the complete runtime so a missing component fails the build.
"""

import sys
import tomllib
from pathlib import Path

# The version is read from pyproject.toml — the single source of truth (app-release
# conventions) — at freeze time, never hardcoded here: a literal would silently drift
# from the real version on the next bump. The spec is a Python file that runs during
# the freeze, so it reads the SSOT directly rather than relying on package.sh/.ps1 to
# thread it in, keeping the version correct however the freeze is invoked (CI, either
# package script, or a bare ``pyinstaller pixelup.spec``). This feeds the macOS bundle's
# CFBundleShortVersionString / CFBundleVersion so a frozen .app reports its real version
# instead of 0.0.0. SPECPATH is the spec's own directory (injected by PyInstaller), so
# the read is independent of the working directory the freeze was launched from.
_VERSION = tomllib.loads(
    (Path(SPECPATH) / "pyproject.toml").read_text(encoding="utf-8")
)["project"]["version"]

# Windows shell surfaces use a runtime icon. macOS takes its Dock icon from the
# bundle's .icns/Assets.car pair and must not receive a Qt application icon.
datas = [
    ("src/pixelup/resources/icon-win.png", "pixelup/resources"),
]
hiddenimports = [
    # urllib asks the codec registry for this name only when an HTTPS hostname is
    # encoded. There is no import edge for PyInstaller to discover, so a frozen
    # downloader otherwise fails before connecting with "unknown encoding: idna".
    "encodings.idna",
]


def _is_shadowing_windows_icu(binary):
    """True for non-Qt ICU DLLs that must not enter the Windows app root."""
    if sys.platform != "win32":
        return False

    name = Path(binary[0]).name.casefold()
    source_parts = {part.casefold() for part in Path(binary[1]).parts}
    return (
        name.endswith(".dll")
        and name.startswith(("icudt", "icuin", "icuuc"))
        and "pyside6" not in source_parts
    )


def _is_editable_install_metadata(data):
    """True for local editable-install metadata that exposes the build checkout."""
    destination = Path(data[0])
    return destination.name == "direct_url.json" and any(
        part.endswith(".dist-info") for part in destination.parts
    )

is_mac = sys.platform == "darwin"
icon = "build/icon.icns" if is_mac else "build/icon.ico"

a = Analysis(
    ["src/pixelup/__main__.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Trim obvious dead weight; pytest/tkinter are never used by the shipped app.
    excludes=["tkinter", "pytest", "_pytest"],
    noarchive=False,
)
# An editable project install records its absolute checkout URL in
# ``pixelup-*.dist-info/direct_url.json``. It is useful to the development
# environment but is neither runtime input nor safe release metadata.
a.datas = [data for data in a.datas if not _is_editable_install_metadata(data)]
# Analysis can discover DLLs outside the explicit inference collections through
# another hook or the host PATH. Windows searches the app root before Qt's own
# directory, so reject every non-PySide6 ICU copy at this final owning boundary
# as well. A future Qt-owned ICU remains intact, and the frozen self-test proves
# that the resulting Qt runtime actually loads.
a.binaries = [
    binary for binary in a.binaries if not _is_shadowing_windows_icu(binary)
]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PixelUp",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # windowed GUI app — no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="PixelUp",
)

if is_mac:
    app = BUNDLE(
        coll,
        name="PixelUp.app",
        icon=icon,
        bundle_identifier="com.nao7sep.pixelup",
        info_plist={
            "CFBundleName": "PixelUp",
            "CFBundleDisplayName": "PixelUp",
            # Both version keys are derived from the pyproject.toml SSOT (_VERSION above),
            # so a frozen .app reports its real version instead of the 0.0.0 PyInstaller
            # defaults to when they are absent. CFBundleShortVersionString is the marketing
            # version shown in the Finder/About; CFBundleVersion is the build version macOS
            # requires be present — both track the one SSOT string.
            "CFBundleShortVersionString": _VERSION,
            "CFBundleVersion": _VERSION,
            "NSHighResolutionCapable": True,
            # Dual-key icon: CFBundleIconFile (the classic .icns, set by icon= above) is read by
            # macOS < 26; CFBundleIconName points macOS 26 (Tahoe) at the Liquid Glass Assets.car
            # that the shared macOS bundle finalizer copies into Contents/Resources/ after
            # the freeze (see the liquid-glass-icon-workflow). The catalog is generated from
            # company/assets by company/tools/liquid-glass-icon/apps/pixelup.mjs.
            "CFBundleIconName": "pixel-butterfly-paper",
            # No document types / URL schemes; PixelUp takes image paths as argv.
        },
    )

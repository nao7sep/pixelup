# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for PixelUp — the fleet's Python/PySide6 GUI freezer.

One spec drives both platforms: on macOS it emits ``PixelUp.app`` (a BUNDLE);
on Windows a ``PixelUp/`` onedir holding ``PixelUp.exe``. The wrapping into
release artifacts (.dmg + portable zip on mac; Inno ``setup.exe`` + portable zip
on win) is done by ``scripts/package.sh`` / ``scripts/package.ps1`` — this spec
only produces the frozen app, per the app-release-conventions (complexity in the
repo, not in CI).

Why the collect_all block: the inference stack (realesrgan, basicsr, gfpgan,
facexlib) imports its network architectures lazily and dynamically (deep inside
functions, via a registry that scans package dirs), so PyInstaller's static
import graph misses them. Each is pulled in whole. torch, torchvision, cv2,
PySide6 and Pillow have maintained PyInstaller hooks and are left to those.
``PIXELUP_SELFTEST=1`` (see ``gui._selftest``) imports every one of these against
the finished bundle, so a missing hidden import fails the build, not the user.
"""

import sys
import tomllib
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

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
binaries = []
# The two arch modules that are imported by string path at runtime; named
# explicitly so they survive even if a future collect_all misses them.
hiddenimports = [
    "basicsr.archs.rrdbnet_arch",
    # urllib asks the codec registry for this name only when an HTTPS hostname is
    # encoded. There is no import edge for PyInstaller to discover, so a frozen
    # downloader otherwise fails before connecting with "unknown encoding: idna".
    "encodings.idna",
    "realesrgan.archs.srvgg_arch",
]

for pkg in ("realesrgan", "basicsr", "gfpgan", "facexlib"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

is_mac = sys.platform == "darwin"
icon = "build/icon.icns" if is_mac else "build/icon.ico"

a = Analysis(
    ["src/pixelup/__main__.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Trim obvious dead weight; pytest/tkinter are never used by the shipped app.
    excludes=["tkinter", "pytest", "_pytest"],
    noarchive=False,
)
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

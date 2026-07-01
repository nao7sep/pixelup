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

from PyInstaller.utils.hooks import collect_all

# The app finds its window icon at runtime via
# ``Path(pixelup.gui.__file__).parent / "resources" / "icon.png"`` — inside a
# frozen bundle that resolves to ``pixelup/resources/icon.png``, so place it there.
datas = [("src/pixelup/resources/icon.png", "pixelup/resources")]
binaries = []
# The two arch modules that are imported by string path at runtime; named
# explicitly so they survive even if a future collect_all misses them.
hiddenimports = [
    "basicsr.archs.rrdbnet_arch",
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
            "NSHighResolutionCapable": True,
            # No document types / URL schemes; PixelUp takes image paths as argv.
        },
    )

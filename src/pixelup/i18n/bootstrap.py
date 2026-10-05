from __future__ import annotations

import ctypes
import ctypes.util
import json
import subprocess
import sys

from PySide6.QtCore import QLocale

from pixelup.formats import CONFIG_FORMAT_VERSION, format_version
from pixelup.i18n import languages, localizer

# Settles the interface language before anything draws.
#
# Two things have to happen this early. The first window's very first frame must
# already be in the reader's language, and on macOS the items AppKit contributes
# itself (Services, Emoji & Symbols, Start Dictation, AutoFill) are drawn in the
# one language it settles on when the application object is created, which Qt
# does inside the QApplication constructor. So the language is resolved and handed
# to AppKit before QApplication exists, from the saved preference read straight
# out of config.json.


def settle_language() -> None:
    """Resolve the language, tell the localizer, and point AppKit at it."""
    # The computer's list is read before AppKit is pointed anywhere: the volatile
    # AppleLanguages below is exactly what Qt would otherwise read back as "the
    # computer's languages".
    computer = read_computer_languages()
    localizer.use(saved_preference(), computer)
    align_appkit(localizer.language())


def read_computer_languages() -> tuple[str, ...]:
    """The languages the computer is set to, most preferred first.

    Read once, at launch, so System cannot mean one language now and another an
    hour later. On Windows, Qt answers from GetUserPreferredUILanguages, which is
    the ordered list that decides which interface language a reader gets. On
    macOS, Qt answers from LANG and LC_* instead of AppleLanguages when nothing
    has set up NSApplication's own locale machinery, which is exactly the case
    for a process started from a .command script or Terminal: LANG there is
    often unset, or stuck at "en_US.UTF-8" no matter what the reader actually
    prefers. So AppleLanguages, the ordered list AppKit itself uses, is read
    straight from the reader's defaults there instead.
    """
    if sys.platform == "darwin":
        apple_languages = read_apple_languages()
        if apple_languages:
            return apple_languages
    system = QLocale.system()
    return tuple(system.uiLanguages()) or (system.bcp47Name(),)


def read_apple_languages() -> tuple[str, ...]:
    """The reader's own AppleLanguages, most preferred first, or empty if unreadable."""
    try:
        output = subprocess.run(
            ["defaults", "read", "-g", "AppleLanguages"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout
    except Exception:  # noqa: BLE001 - any failure here just falls back to Qt.
        return ()
    tags = []
    for line in output.splitlines():
        text = line.strip().rstrip(",")
        if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
            tags.append(text[1:-1])
    return tuple(tags)


def saved_preference() -> str:
    """The language preference in config.json, or System when there is none to read.

    The read is deliberately its own, and forgiving: the real loader quarantines a
    file it cannot parse, and that decision belongs to the window's startup path,
    not to a language lookup. A file that cannot be read here means System, and
    a startup failure then speaks the computer's language, as the
    localization-conventions ask of it.
    """
    try:
        # Imported here: app_config pulls in the processing modules, which this
        # early, dependency-light step has no other reason to load.
        from pixelup.app_config import config_path

        path = config_path()
        if not path.is_file():
            return languages.SYSTEM
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - any failure here simply means System.
        return languages.SYSTEM
    if not isinstance(data, dict) or not _readable_config_version(data.get("format_version")):
        return languages.SYSTEM
    value = data.get("language")
    return value if languages.is_preference(value) else languages.SYSTEM


def _readable_config_version(value: object) -> bool:
    """Whether this build reads a config.json with this marker; the window's
    loader leaves a newer PixelUp's file on the built-ins, System among them."""
    try:
        return format_version(value) <= CONFIG_FORMAT_VERSION
    except ValueError:
        return False


def align_appkit(tag: str) -> None:
    """Point AppKit at ``tag`` through the volatile argument domain.

    ``AppleLanguages`` is set in NSArgumentDomain, which lives for this process
    only and writes nothing to the reader's defaults. A language saved
    mid-session reaches AppKit's own items at the next launch.
    """
    if sys.platform != "darwin":
        return
    try:
        _set_volatile_apple_languages(tag)
    except Exception:  # noqa: BLE001 - never a reason to fail to start.
        # Without this, the menu items macOS draws itself follow the computer
        # instead of the app: a mixed menu, not a broken one.
        pass


def _set_volatile_apple_languages(tag: str) -> None:
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc") or "libobjc.dylib")
    ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Foundation.framework/Foundation")
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]

    def send(receiver: int, selector: str, *args: object, argtypes: tuple = ()) -> int:
        # objc_msgSend is not variadic on arm64: it has to be called through a
        # prototype with the exact argument types of the method it reaches.
        prototype = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, *argtypes)
        function = prototype(("objc_msgSend", objc))
        return function(receiver, objc.sel_registerName(selector.encode()), *args)

    def cls(name: str) -> int:
        return objc.objc_getClass(name.encode())

    def ns_string(text: str) -> int:
        return send(
            cls("NSString"),
            "stringWithUTF8String:",
            text.encode("utf-8"),
            argtypes=(ctypes.c_char_p,),
        )

    pointer = (ctypes.c_void_p,)
    tags = send(cls("NSArray"), "arrayWithObject:", ns_string(tag), argtypes=pointer)
    domain = send(
        cls("NSDictionary"),
        "dictionaryWithObject:forKey:",
        tags,
        ns_string("AppleLanguages"),
        argtypes=pointer * 2,
    )
    defaults = send(cls("NSUserDefaults"), "standardUserDefaults")
    send(
        defaults,
        "setVolatileDomain:forName:",
        domain,
        ns_string("NSArgumentDomain"),
        argtypes=pointer * 2,
    )

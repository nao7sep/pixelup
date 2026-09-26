"""How PixelUp picks its language, counts, formats and hands the choice to Qt and AppKit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QLibraryInfo, QLocale

from pixelup.app_config import AppConfig, load_app_config, save_app_config
from pixelup.errors import ErrorCode, PixelupError
from pixelup.i18n import bootstrap, languages, localizer, plural
from pixelup.i18n.message import Message, join
from pixelup.i18n.translator import Translator


def test_names_are_each_language_in_its_own_words() -> None:
    assert languages.name_of("de") == "Deutsch"
    assert languages.name_of("zh-Hans") == "中文"
    assert languages.name_of("pt-BR") == "Português"


@pytest.mark.parametrize(
    ("saved", "expected"),
    [
        (None, "system"),
        ("", "system"),
        ("  ", "system"),
        (7, "system"),
        ("SYSTEM", "system"),
        ("klingon", "system"),
        ("ja", "ja"),
        ("PT-br", "pt-BR"),
        (" zh-hans ", "zh-Hans"),
    ],
)
def test_a_saved_preference_is_a_tag_or_system(saved: object, expected: str) -> None:
    assert languages.normalize_preference(saved) == expected


@pytest.mark.parametrize(
    ("computer", "expected"),
    [
        (["ja-JP", "en-US"], "ja"),
        (["zh-Hant-TW"], "zh-Hans"),
        (["zh_HK"], "zh-Hans"),
        (["pt-PT"], "pt-BR"),
        (["es-MX"], "es"),
        (["fr-CA"], "fr"),
        (["nl-NL", "de-CH"], "de"),
        (["nl-NL", "sv-SE"], "en"),
        (["C"], "en"),
        ([], "en"),
    ],
)
def test_system_takes_the_first_computer_language_in_the_set(
    computer: list[str], expected: str
) -> None:
    assert languages.resolve("system", computer) == expected
    assert languages.resolve("ko", computer) == "ko"


def test_the_formatting_locale_keeps_the_readers_own_region_in_their_language() -> None:
    # A British computer set to English keeps British dates and separators; an
    # interface language the computer does not use takes that language's own.
    assert languages.formatting_locale("en", QLocale("en-GB")).name() == "en_GB"
    assert languages.formatting_locale("de", QLocale("en-GB")).name() == "de_DE"


@pytest.mark.parametrize("tag", [tag for tag in languages.TAGS if tag != "en"])
def test_qt_ships_its_own_translation_for_every_language(tag: str) -> None:
    # Qt's own text (a dialog button box's Close, the macOS application menu) is
    # translated from these files; a missing one would leave those words English.
    name = languages.qt_translation_name(tag)
    directory = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath))
    assert (directory / f"qtbase_{name}.qm").is_file()


@pytest.mark.parametrize(
    ("tag", "count", "expected"),
    [
        ("en", 1, "one"),
        ("en", 0, "other"),
        ("de", 2, "other"),
        ("fr", 0, "one"),
        ("fr", 1_000_000, "many"),
        ("es", 0, "other"),
        ("pt-BR", 1, "one"),
        ("it", 2_000_000, "many"),
        ("ru", 1, "one"),
        ("ru", 21, "one"),
        ("ru", 11, "many"),
        ("ru", 3, "few"),
        ("ru", 14, "many"),
        ("ru", 22, "few"),
        ("ru", 5, "many"),
        ("ja", 1, "other"),
    ],
)
def test_counts_take_their_languages_plural_form(tag: str, count: int, expected: str) -> None:
    assert plural.category_for(tag, count) == expected
    assert expected in plural.categories_of(tag)


def test_the_translator_fills_plurals_numbers_lists_and_nested_messages() -> None:
    english = Translator("en", QLocale("en"))
    assert english.t("images.added", count=1) == "Added 1 image."
    assert english.t("images.added", count=1234) == "Added 1,234 images."
    assert english.of(Message.of("images.alreadyOpen", names=("a.png", "b.png", "c.png"))) == (
        "Already open: a.png, b.png, and c.png."
    )
    summary = join(
        "images.jobsJoin",
        [Message.of("images.jobsDone", count=2), Message.of("images.jobsFailed", count=1)],
    )
    assert english.of(summary) == "2 done, 1 failed"

    russian = Translator("ru", QLocale("ru"))
    assert russian.t("managedModels.jobs", count=2) == "2 задания"
    assert russian.t("managedModels.jobs", count=5) == "5 заданий"
    german = Translator("de", QLocale("de"))
    assert german.t("units.megabytes", value=4.66) == "4,7 MB"


def test_an_unknown_key_shows_the_key_so_the_rendered_key_gate_sees_it() -> None:
    assert Translator("ja", QLocale("ja")).t("no.suchKey") == "no.suchKey"
    assert str(Message("images.title")) == "images.title"


def test_a_failure_logs_english_and_shows_the_readers_language() -> None:
    error = PixelupError(
        ErrorCode.MODEL_NOT_FOUND,
        Message.of("error.modelMissing", model="x4plus"),
        hint=Message("error.hintInstallModel"),
    )
    with localizer.speaking("ja"):
        assert str(error).startswith("Model 'x4plus' is not present")
        assert localizer.of(error.message) == "モデル「x4plus」がモデルフォルダーにありません。"


def test_a_language_change_is_announced_and_restored(qapp) -> None:
    heard: list[str] = []
    localizer.changed.connect(lambda: heard.append(localizer.language()))
    try:
        with localizer.speaking("fr"):
            assert localizer.t("quit.quit") == "Quitter"
            assert QLocale().language() == QLocale.Language.French
        assert localizer.language() == "en"
        assert heard == ["fr", "en"]
    finally:
        localizer.changed.disconnect()


def test_system_is_resolved_once_from_the_list_read_at_launch() -> None:
    localizer.use("system", ["ko-KR"])
    assert localizer.language() == "ko"
    # A later change in Settings resolves System against the same list.
    localizer.use("de")
    localizer.use("system")
    assert localizer.language() == "ko"
    assert localizer.preference() == "system"


def test_the_saved_language_is_read_straight_from_config_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PIXELUP_HOME", str(tmp_path))
    assert bootstrap.saved_preference() == "system"
    (tmp_path / "config.json").write_text(json.dumps({"language": "it"}), encoding="utf-8")
    assert bootstrap.saved_preference() == "it"
    (tmp_path / "config.json").write_text("{not json", encoding="utf-8")
    assert bootstrap.saved_preference() == "system"


def test_settle_language_speaks_the_saved_choice_before_the_app_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PIXELUP_HOME", str(tmp_path))
    aligned: list[str] = []
    monkeypatch.setattr(bootstrap, "align_appkit", aligned.append)
    monkeypatch.setattr(bootstrap, "read_computer_languages", lambda: ("es-MX",))
    bootstrap.settle_language()
    assert localizer.language() == "es"
    (tmp_path / "config.json").write_text(json.dumps({"language": "ru"}), encoding="utf-8")
    bootstrap.settle_language()
    assert localizer.language() == "ru"
    assert aligned == ["es", "ru"]


def test_the_language_setting_round_trips_and_an_unknown_value_means_system(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    save_app_config(AppConfig(language="zh-Hans"), path)
    assert json.loads(path.read_text(encoding="utf-8"))["language"] == "zh-Hans"
    assert load_app_config(path).language == "zh-Hans"
    # Unknown to this build is not corruption: the rest of the settings survive.
    path.write_text(
        json.dumps({"language": "tlh", "max_concurrent_jobs": 3}), encoding="utf-8"
    )
    loaded = load_app_config(path)
    assert (loaded.language, loaded.max_concurrent_jobs) == ("system", 3)


@pytest.mark.skipif(sys.platform != "darwin", reason="AppKit is macOS only")
def test_appkit_is_pointed_at_the_language_for_this_process_only() -> None:
    # In a child process, so the test run's own defaults stay as they were.
    probe = (
        "import ctypes, ctypes.util\n"
        "from pixelup.i18n.bootstrap import align_appkit\n"
        "align_appkit('ja')\n"
        "objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library('objc'))\n"
        "objc.objc_getClass.restype = ctypes.c_void_p\n"
        "objc.objc_getClass.argtypes = [ctypes.c_char_p]\n"
        "objc.sel_registerName.restype = ctypes.c_void_p\n"
        "objc.sel_registerName.argtypes = [ctypes.c_char_p]\n"
        "P = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)\n"
        "P1 = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,"
        " ctypes.c_void_p)\n"
        "S = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,"
        " ctypes.c_char_p)\n"
        "U = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p)\n"
        "send = lambda proto, *a: proto(('objc_msgSend', objc))(*a)\n"
        "sel = lambda name: objc.sel_registerName(name.encode())\n"
        "key = send(S, objc.objc_getClass(b'NSString'), sel('stringWithUTF8String:'),"
        " b'AppleLanguages')\n"
        "defaults = send(P, objc.objc_getClass(b'NSUserDefaults'),"
        " sel('standardUserDefaults'))\n"
        "tags = send(P1, defaults, sel('objectForKey:'), key)\n"
        "first = send(P1, tags, sel('objectAtIndex:'), 0)\n"
        "print(send(U, first, sel('UTF8String')).decode())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, check=True
    )
    assert result.stdout.strip() == "ja"

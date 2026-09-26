"""PixelUp's interface languages (localization-conventions).

- ``languages``: the ten tags, their own names, preference normalization, System
  resolution and the formatting locale.
- ``plural``: the CLDR cardinal rules for whole numbers.
- ``catalogue``: the JSON catalogues in ``locales/``, one per language.
- ``message``: text as it is held, a key plus its values (Qt-free, so processing
  modules can raise with one).
- ``translator``: one immutable translator per language.
- ``localizer``: the current translator, Qt's own translation, and ``changed``.
- ``localized``: widget bindings that hold a key and follow a language change.
- ``bootstrap``: settles the language before the application object exists.
"""

from pixelup.i18n.message import Message, join

__all__ = ["Message", "join"]

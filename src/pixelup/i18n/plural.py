from __future__ import annotations

# Which plural form a count takes in each language, and which forms a catalogue
# entry must carry. Python's standard library has no plural rules, and Qt's
# QLocale formats numbers but never says which words go with them, so the ten
# languages' CLDR cardinal rules are written out here. Counts in the interface are
# whole numbers, so the rules are given for integers, which is what removes CLDR's
# compound conditions on the decimal operands.

ONE = "one"
FEW = "few"
MANY = "many"
OTHER = "other"


def categories_of(tag: str) -> tuple[str, ...]:
    """The forms a language's plural entry must have, and no others."""
    match tag:
        case "ja" | "ko" | "zh-Hans":
            return (OTHER,)
        case "en" | "de":
            return (ONE, OTHER)
        case "es" | "fr" | "it" | "pt-BR":
            return (ONE, MANY, OTHER)
        case "ru":
            return (ONE, FEW, MANY, OTHER)
    raise ValueError(f"not an interface language: {tag}")


def category_for(tag: str, count: int) -> str:
    """The form ``count`` takes in ``tag``."""
    n = abs(count)
    match tag:
        case "ja" | "ko" | "zh-Hans":
            return OTHER
        case "en" | "de":
            return ONE if n == 1 else OTHER
        case "es" | "it":
            return ONE if n == 1 else _romance(n)
        case "fr" | "pt-BR":
            # French and Brazilian Portuguese count zero with the singular.
            return ONE if n in (0, 1) else _romance(n)
        case "ru":
            if n % 10 == 1 and n % 100 != 11:
                return ONE
            if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
                return FEW
            return MANY
    return OTHER


def _romance(n: int) -> str:
    # The Romance "many" form is for round millions ("1 000 000 de fichiers").
    return MANY if n != 0 and n % 1_000_000 == 0 else OTHER

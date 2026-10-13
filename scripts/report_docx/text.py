"""Persian text helpers: digits, ordinals and splitting text into Persian and Latin runs.

Word lays out Persian correctly only in runs marked right-to-left (``w:rtl``),
which also select the complex-script font (B Nazanin). Latin words must sit in
their own, unmarked runs so they take the Latin font (Times New Roman) and keep
their left-to-right order. Numbers are written with Persian digits inside the
Persian runs; Word then reads «۲-۱-» chapter-first, as the template requires.
"""

from __future__ import annotations

import re

_FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
_ORDINALS = ("اول", "دوم", "سوم", "چهارم", "پنجم", "ششم", "هفتم", "هشتم", "نهم", "دهم")

# A Latin token starts with a letter and ends with a letter or digit, so a
# sentence-final period or a closing bracket stays with the Persian text.
_TOKEN = r"[A-Za-z](?:[A-Za-z0-9\-/.+_']*[A-Za-z0-9])?"
LATIN = re.compile(rf"{_TOKEN}(?:[ ]+{_TOKEN})*")


def fa_digits(s: str) -> str:
    """Latin digits to Persian digits, except inside Latin words such as «F1» or «S1C»."""
    return "".join(piece if not rtl else piece.translate(_FA) for piece, rtl in split_scripts(s))


def ordinal(n: int) -> str:
    """Persian ordinal word for a chapter number (1 -> «اول»)."""
    return _ORDINALS[n - 1]


def split_scripts(s: str) -> list[tuple[str, bool]]:
    """Split text into (piece, is_rtl) runs: Latin phrases unmarked, the rest right-to-left."""
    parts: list[tuple[str, bool]] = []
    pos = 0
    for m in LATIN.finditer(s):
        if m.start() > pos:
            parts.append((s[pos : m.start()], True))
        parts.append((m.group(0), False))
        pos = m.end()
    if pos < len(s):
        parts.append((s[pos:], True))
    return parts

"""Read what scripts/make_reports.py writes: number macros and booktabs tables.

The LaTeX files stay the single source of the numbers; this module turns their
small, fixed vocabulary (``\\LRE{}``, ``\\lr{}``, ``\\textbf{}``) into plain
Persian text for Word: digits inside ``\\LRE`` become Persian digits, ``\\lr``
is dropped (Latin is detected again when runs are made) and bold becomes
``**...**`` for the markup renderer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from scripts.report_docx.text import fa_digits

_MACRO = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$")


@dataclass(frozen=True)
class Table:
    header: list[str]
    rows: list[list[str]]
    caption: str
    short: str
    label: str


def _group(s: str, start: int) -> tuple[str, int]:
    """The contents of the brace group opening at ``s[start]`` and the index after it."""
    if s[start] != "{":
        raise ValueError(f"expected '{{' at {start} in {s!r}")
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1 : i], i + 1
    raise ValueError(f"unbalanced braces in {s!r}")


def to_plain(s: str) -> str:
    """One LaTeX cell or macro body as plain Persian text."""
    out: list[str] = []
    i = 0
    while i < len(s):
        for command in ("\\LRE", "\\lr", "\\textbf"):
            if s.startswith(command + "{", i):
                inner, i = _group(s, i + len(command))
                if command == "\\LRE":
                    out.append(fa_digits(to_plain(inner)))
                elif command == "\\lr":
                    out.append(to_plain(inner))
                else:
                    out.append(f"**{to_plain(inner)}**")
                break
        else:
            if s.startswith("\\%", i):
                out.append("٪")
                i += 2
            elif s[i] == "\\":
                raise ValueError(f"unexpected LaTeX command at {s[i : i + 20]!r}")
            else:
                out.append(s[i])
                i += 1
    return "".join(out).strip()


def parse_numbers(tex: str) -> dict[str, str]:
    """``\\newcommand{\\name}{...}`` lines as {name: Persian text}."""
    numbers = {}
    for line in tex.splitlines():
        m = _MACRO.match(line.strip())
        if m:
            numbers[m.group(1)] = to_plain(m.group(2))
    return numbers


def _caption(tex: str) -> tuple[str, str]:
    start = tex.index("\\caption")
    i = start + len("\\caption")
    short = None
    if tex[i] == "[":
        end = tex.index("]", i)
        short = tex[i + 1 : end]
        i = end + 1
    caption, _ = _group(tex, i)
    return to_plain(short if short is not None else caption), to_plain(caption)


def parse_table(tex: str) -> Table:
    """A table written by make_reports.latex_table()."""
    body = tex[tex.index("\\toprule") + len("\\toprule") : tex.index("\\bottomrule")]
    head, rows = body.split("\\midrule")

    def cells(line: str) -> list[str]:
        return [to_plain(c) for c in re.split(r"(?<!\\)&", line.strip().removesuffix("\\\\"))]

    header = cells(head)
    data = [cells(line) for line in rows.strip().splitlines() if line.strip()]
    short, caption = _caption(tex)
    label = re.search(r"\\label\{([^}]*)\}", tex).group(1)
    return Table(header, data, caption, short, label)

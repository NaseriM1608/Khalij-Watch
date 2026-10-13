"""The report's content markup: a small Markdown-like format for Persian chapters.

    # عنوان فصل            chapter («فصل اول: ...»); ``#*`` for an unnumbered one
    ## عنوان {#sec:x}       heading level 2 («۱-۱-»), optional label; ### and #### deeper
    متن بند ...             paragraph (consecutive lines are joined)
    - مورد                  bullet list; ``1. مورد`` numbered list
    :::figure id=.. src=.. width=.. short="..."
    شرح شکل
    :::                     block (figure, table, equation, pagebreak) with attributes and body

Inline markup inside text is handled by the renderer: ``**پررنگ**``,
``[@key; @key2]`` citations, ``{{macro}}`` numbers, ``[[fig:x]]`` references,
``^[...]`` footnotes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HEADING = re.compile(r"^(#{1,4})(\*?)\s+(.*?)\s*(?:\{#([\w:-]+)\})?\s*$")
_ATTR = re.compile(r'(\w+)=(?:"([^"]*)"|(\S+))')
_NUMBERED = re.compile(r"^\d+\.\s+(.*)$")


@dataclass
class Block:
    kind: str
    text: str = ""
    level: int = 0
    label: str | None = None
    numbered: bool = True
    items: list[str] = field(default_factory=list)
    attrs: dict[str, str] = field(default_factory=dict)
    body: str = ""


def _attrs(s: str) -> dict[str, str]:
    return {
        m.group(1): m.group(2) if m.group(2) is not None else m.group(3) for m in _ATTR.finditer(s)
    }


def parse(src: str) -> list[Block]:
    """Blocks of a content file, in order."""
    blocks: list[Block] = []
    lines = src.splitlines()
    i = 0
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(Block("paragraph", text=" ".join(s.strip() for s in paragraph)))
            paragraph.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("%%"):
            flush()
            i += 1
            continue
        if stripped.startswith(":::"):
            flush()
            head = stripped[3:].strip()
            kind, _, rest = head.partition(" ")
            body: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() != ":::":
                body.append(lines[i])
                i += 1
            if i == len(lines):
                raise ValueError(f"unclosed ':::{kind}' block")
            blocks.append(Block(kind, attrs=_attrs(rest), body="\n".join(body).strip()))
            i += 1
            continue
        m = _HEADING.match(stripped)
        if m:
            flush()
            hashes, star, title, label = m.groups()
            if len(hashes) == 1:
                blocks.append(Block("chapter", text=title, label=label, numbered=not star))
            else:
                blocks.append(Block("heading", text=title, level=len(hashes), label=label))
            i += 1
            continue
        if stripped.startswith("- ") or _NUMBERED.match(stripped):
            flush()
            kind = "bullets" if stripped.startswith("- ") else "numbered"
            items: list[str] = []
            while i < len(lines):
                s = lines[i].strip()
                if kind == "bullets" and s.startswith("- "):
                    items.append(s[2:].strip())
                elif kind == "numbered" and _NUMBERED.match(s):
                    items.append(_NUMBERED.match(s).group(1).strip())
                elif s and lines[i].startswith("  ") and items:
                    items[-1] += " " + s
                else:
                    break
                i += 1
            blocks.append(Block(kind, items=items))
            continue
        paragraph.append(line)
        i += 1
    flush()
    return blocks

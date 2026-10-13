"""Render content blocks to WordprocessingML, numbering everything the template's way.

Chapters read «فصل اول: ...»; sections, figures, tables and equations are
numbered chapter-first («۲-۱-»), captions as «شکل ۲-۱- ...» below figures and
«جدول ۲-۱- ...» above tables, equations «(۲-۱)» at the right margin. Labels are
numbered in a first pass, so a reference may come before its target.
Citations are numbered in order of first appearance ([۱]).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from scripts.report_docx import omml, ooxml
from scripts.report_docx.latex_inputs import Table
from scripts.report_docx.markup import Block
from scripts.report_docx.text import fa_digits, ordinal

_CITE = re.compile(r"\[@([^\]]+)\]")
_MACRO = re.compile(r"\{\{(\w+)\}\}")
_REF = re.compile(r"\[\[([\w:-]+)\]\]")
_FOOTNOTE = re.compile(r"\^\[([^\]]+)\]")
_CHAPTER_AFTER = 4800  # twips below a chapter title: the template starts chapter text mid-page


@dataclass
class Chapter:
    number: int | None
    title: str

    @property
    def label(self) -> str:
        return f"فصل {ordinal(self.number)}: {self.title}" if self.number else self.title


@dataclass
class Renderer:
    numbers: dict[str, str]
    tables: dict[str, Table]
    charts: dict[str, tuple[str, int, int]]  # name -> (relationship id, pixel width, pixel height)
    references: dict[str, str]
    section_end: Callable[[Chapter], str] | None = None
    cited: list[str] = field(default_factory=list)
    footnotes: list[tuple[int, str]] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    _picture_id: int = 100

    # --- numbering ----------------------------------------------------------

    def number_labels(self, blocks: list[Block]) -> None:
        chapter, sections = 0, [0, 0, 0]
        numbered = True
        counters = {"figure": 0, "table": 0, "equation": 0}
        for b in blocks:
            if b.kind == "chapter":
                numbered = b.numbered
                if numbered:
                    chapter += 1
                sections = [0, 0, 0]
                counters = dict.fromkeys(counters, 0)
                if b.label:
                    self.labels[b.label] = fa_digits(str(chapter)) if numbered else b.text
            elif b.kind == "heading":
                depth = b.level - 2
                sections[depth] += 1
                sections[depth + 1 :] = [0] * (len(sections) - depth - 1)
                if b.label:
                    self.labels[b.label] = (
                        self._section_number(chapter, sections[: depth + 1]) if numbered else b.text
                    )
            elif b.kind in counters:
                counters[b.kind] += 1
                if "id" in b.attrs:
                    prefix = {"figure": "fig", "table": "tab", "equation": "eq"}[b.kind]
                    self.labels[f"{prefix}:{b.attrs['id']}"] = fa_digits(
                        f"{chapter}-{counters[b.kind]}"
                    )

    @staticmethod
    def _section_number(chapter: int, parts: list[int]) -> str:
        return fa_digits("-".join(str(n) for n in [chapter, *parts]))

    # --- inline text ----------------------------------------------------------

    def _cite(self, m: re.Match) -> str:
        nums = []
        for key in (k.strip().lstrip("@") for k in m.group(1).split(";")):
            if key not in self.references:
                raise KeyError(f"unknown reference {key}")
            if key not in self.cited:
                self.cited.append(key)
            nums.append(self.cited.index(key) + 1)
        return "[" + "، ".join(fa_digits(str(n)) for n in sorted(set(nums))) + "]"

    def _expand(self, t: str) -> str:
        def macro(m: re.Match) -> str:
            if m.group(1) not in self.numbers:
                raise KeyError(f"unknown number macro {m.group(1)}")
            return self.numbers[m.group(1)]

        def ref(m: re.Match) -> str:
            if m.group(1) not in self.labels:
                raise KeyError(f"unknown label {m.group(1)}")
            return self.labels[m.group(1)]

        t = _MACRO.sub(macro, t)
        t = _REF.sub(ref, t)
        return _CITE.sub(self._cite, t)

    def inline(self, t: str, *, size: float | None = None, bold: bool = False) -> str:
        """Runs for a piece of content text with its inline markup."""
        t = self._expand(t)
        out = []
        for k, chunk in enumerate(t.split("**")):
            is_bold = bold or k % 2 == 1
            pos = 0
            for m in _FOOTNOTE.finditer(chunk):
                out.append(ooxml.text_runs(chunk[pos : m.start()], bold=is_bold, size=size))
                out.append(self._footnote(m.group(1)))
                pos = m.end()
            out.append(ooxml.text_runs(chunk[pos:], bold=is_bold, size=size))
        return "".join(out)

    def _footnote(self, note: str) -> str:
        fid = len(self.footnotes) + 1
        self.footnotes.append((fid, note))
        return (
            '<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr>'
            f'<w:footnoteReference w:id="{fid}"/></w:r>'
        )

    # --- blocks ------------------------------------------------------------------

    def render(self, blocks: list[Block]) -> str:
        self.number_labels(blocks)
        out: list[str] = []
        chapter, sections = 0, [0, 0, 0]
        for b in blocks:
            if b.kind == "chapter":
                if self.chapters and self.section_end is not None:
                    out.append(self.section_end(self.chapters[-1]))
                elif self.chapters:
                    out.append(ooxml.page_break())
                if b.numbered:
                    chapter += 1
                    sections = [0, 0, 0]
                current = Chapter(chapter if b.numbered else None, b.text)
                self.chapters.append(current)
                out.append(
                    ooxml.paragraph(
                        ooxml.text_runs(current.label),
                        style="Heading1",
                        after=_CHAPTER_AFTER,
                        keep_next=True,
                    )
                )
            elif b.kind == "heading":
                depth = b.level - 2
                sections[depth] += 1
                sections[depth + 1 :] = [0] * (len(sections) - depth - 1)
                prefix = (
                    self._section_number(chapter, sections[: depth + 1]) + "- "
                    if self.chapters[-1].number
                    else ""
                )
                out.append(
                    ooxml.paragraph(
                        ooxml.text_runs(prefix) + self.inline(b.text),
                        style=f"Heading{b.level}",
                        keep_next=True,
                    )
                )
            elif b.kind == "paragraph":
                out.append(ooxml.paragraph(self.inline(b.text)))
            elif b.kind in ("bullets", "numbered"):
                out.extend(self._list(b))
            elif b.kind == "figure":
                out.extend(self._figure(b))
            elif b.kind == "table":
                out.extend(self._table(b))
            elif b.kind == "equation":
                out.append(self._equation(b))
            elif b.kind == "pagebreak":
                out.append(ooxml.page_break())
            elif b.kind == "references":
                out.append(self._references())
            else:
                raise ValueError(f"unknown block {b.kind!r}")
        return "".join(out)

    def _list(self, b: Block) -> list[str]:
        out = []
        for k, item in enumerate(b.items, start=1):
            marker = "•" if b.kind == "bullets" else fa_digits(f"{k}.")
            out.append(
                ooxml.paragraph(
                    ooxml.run(marker, rtl=True) + "<w:r><w:tab/></w:r>" + self.inline(item),
                    indent='<w:ind w:left="567" w:hanging="283"/>',
                    tabs='<w:tab w:val="left" w:pos="567"/>',
                    after=60,
                )
            )
        return out

    def _caption(self, kind: str, label: str, caption: str, style: str, short: str) -> str:
        word = {"figure": "شکل", "table": "جدول"}[kind]
        title = ooxml.paragraph(
            ooxml.text_runs(f"{word} {label}- ") + self.inline(short, size=12),
            style=style,
            align="center",
            keep_next=kind == "table" or short != caption,
            before=60,
            after=60,
        )
        detail = (
            ooxml.paragraph(
                self.inline(caption, size=12),
                align="center",
                line=240,
                keep_next=kind == "table",
                after=200 if kind == "figure" else 60,
            )
            if short != caption
            else ""
        )
        return title + detail

    def _picture(self, key: str, width_cm: float) -> str:
        if key not in self.charts:
            raise KeyError(f"unknown figure source {key}")
        rid, px_w, px_h = self.charts[key]
        cx = round(width_cm * ooxml.EMU_PER_CM)
        cy = round(cx * px_h / px_w)
        self._picture_id += 1
        return ooxml.inline_image(rid, cx, cy, self._picture_id, key)

    def _figure(self, b: Block) -> list[str]:
        sources = [s.split(":", 1)[1] for s in b.attrs["src"].split()]
        width = float(b.attrs.get("width", "14"))
        label = self.labels[f"fig:{b.attrs['id']}"]
        if len(sources) == 1:
            pictures = ooxml.paragraph(
                self._picture(sources[0], width), align="center", keep_next=True, before=120
            )
        else:
            cols = int(b.attrs.get("cols", "2"))
            subs = b.attrs.get("sub", "").split("|")
            cells = [
                ooxml.paragraph(self._picture(s, width), align="center", keep_next=True)
                + ooxml.paragraph(
                    self.inline(subs[k] if k < len(subs) else "", size=12),
                    align="center",
                    after=120,
                )
                for k, s in enumerate(sources)
            ]
            rows = [cells[i : i + cols] for i in range(0, len(cells), cols)]
            pictures = ooxml.table(rows, header_rows=0, borders=False)
        return [
            pictures,
            self._caption("figure", label, b.body, "FigureCaption", b.attrs.get("short", b.body)),
        ]

    def _table(self, b: Block) -> list[str]:
        label = self.labels[f"tab:{b.attrs['id']}"]
        if "src" in b.attrs:
            key = b.attrs["src"].split(":", 1)[1]
            if key not in self.tables:
                raise KeyError(f"unknown table source {key}")
            data = self.tables[key]
            header, rows, caption = data.header, data.rows, b.body or data.caption
            short = b.attrs.get("short", data.short)
        else:
            lines = b.body.splitlines()
            caption = " ".join(s.strip() for s in lines if not s.strip().startswith("|"))
            grid = [
                [c.strip() for c in s.strip().strip("|").split("|")]
                for s in lines
                if s.strip().startswith("|") and not re.match(r"^\|[\s:|-]+\|$", s.strip())
            ]
            header, rows = grid[0], grid[1:]
            short = b.attrs.get("short", caption)

        def cell(t: str, bold: bool, first: bool, keep_next: bool) -> str:
            return ooxml.paragraph(
                self.inline(t, size=11, bold=bold),
                style="TableText",
                align="right" if first else "center",
                keep_next=keep_next,
            )

        # The report uses compact tables; keep their rows with the caption,
        # then release the final row so the following prose can flow normally.
        grid_xml = [[cell(t, True, k == 0, bool(rows)) for k, t in enumerate(header)]]
        grid_xml += [
            [cell(t, False, k == 0, i < len(rows) - 1) for k, t in enumerate(r)]
            for i, r in enumerate(rows)
        ]
        return [
            self._caption("table", label, caption, "TableCaption", short),
            ooxml.table(grid_xml),
            ooxml.paragraph("", after=120),
        ]

    def _references(self) -> str:
        """The cited works in order of first citation, «[۱]» at the right as in the template."""
        out = []
        for k, key in enumerate(self.cited, start=1):
            label = ooxml.run(f"[{fa_digits(str(k))}]", rtl=True, size=11)
            out.append(
                ooxml.paragraph(
                    label
                    + "<w:r><w:rPr><w:rtl/></w:rPr><w:tab/></w:r>"
                    + ooxml.run(self.references[key], rtl=False, size=11, font="Times New Roman"),
                    style="EndNoteBibliography",
                    after=100,
                    line=240,
                    align="both",
                    indent='<w:ind w:left="567" w:hanging="567"/>',
                    tabs='<w:tab w:val="left" w:pos="567"/>',
                )
            )
        return "".join(out)

    def _equation(self, b: Block) -> str:
        label = self.labels[f"eq:{b.attrs['id']}"]
        return ooxml.paragraph(
            ooxml.run(f"({label})", rtl=True) + "<w:r><w:tab/></w:r>" + omml.to_omml(b.body),
            tabs=f'<w:tab w:val="center" w:pos="{ooxml.TEXT_WIDTH_TWIPS // 2}"/>',
            before=120,
            after=120,
        )

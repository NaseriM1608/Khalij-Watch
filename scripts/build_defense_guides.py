"""Build the two Persian defense-study guides without editing the accepted report."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_report_docx import export_with_word  # noqa: E402
from scripts.report_docx import content, markup, ooxml  # noqa: E402
from scripts.report_docx.document import (  # noqa: E402
    FOOTER_LETTERS,
    HEADER_EMPTY,
    Assembler,
    section,
)
from scripts.report_docx.latex_inputs import parse_numbers  # noqa: E402
from scripts.report_docx.render import Renderer  # noqa: E402

GUIDES = ROOT / "reports" / "defense"


def format_study_body(xml: str) -> str:
    """Give the report renderer's content the compact study-guide layout."""
    xml = re.sub(r'w:val="Heading([1-4])"', r'w:val="StudyHeading\1"', xml)
    xml = xml.replace('w:after="4800"', 'w:after="240"')
    xml = re.sub(r"<w:pPr>(?!<w:pStyle)", '<w:pPr><w:pStyle w:val="StudyBody"/>', xml)

    def allow_long_table(m: re.Match) -> str:
        table = m.group(0)
        if table.count("<w:tr>") <= 10:
            return table
        first_end = table.index("</w:tr>") + len("</w:tr>")
        return table[:first_end] + table[first_end:].replace("<w:keepNext/>", "")

    return re.sub(r"<w:tbl>.*?</w:tbl>", allow_long_table, xml, flags=re.S)


def study_styles() -> str:
    """Explicit Persian/Latin typography; no university chapter-title whitespace."""
    out = []
    specs = [("Title", 24, 18, 0, 360, None), ("StudyBody", 14, 12, 0, 100, None)]
    specs += [
        (f"StudyHeading{i}", 20 if i == 1 else 16 if i == 2 else 14, 12, 200, 100, i - 1)
        for i in range(1, 5)
    ]
    for name, fa_size, en_size, before, after, level in specs:
        bold = "<w:b/><w:bCs/>" if name != "StudyBody" else ""
        keep = "<w:keepNext/>" if level is not None or name == "Title" else ""
        outline = f'<w:outlineLvl w:val="{level}"/>' if level is not None else ""
        align = "center" if name == "Title" else "both" if name == "StudyBody" else "right"
        out.append(
            f'<w:style w:type="paragraph" w:customStyle="1" w:styleId="{name}">'
            f'<w:name w:val="{name}"/><w:basedOn w:val="Normal"/><w:qFormat/>'
            f'<w:pPr>{keep}<w:bidi/><w:snapToGrid w:val="0"/>'
            f'<w:spacing w:before="{before}" w:after="{after}" w:line="300" w:lineRule="auto"/>'
            f'<w:jc w:val="{align}"/>{outline}</w:pPr><w:rPr>'
            '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="B Nazanin"/>'
            f'{bold}<w:color w:val="000000"/><w:sz w:val="{2 * en_size}"/>'
            f'<w:szCs w:val="{2 * fa_size}"/></w:rPr></w:style>'
        )
    return "".join(out)


def assemble_guide(name: str, destination: Path) -> dict:
    meta = json.loads((GUIDES / "meta.json").read_text(encoding="utf-8"))
    info = meta["guides"][name]
    blocks = markup.parse((GUIDES / f"{name}.md").read_text(encoding="utf-8"))
    assembler = Assembler(ROOT / "reports/word/template.docx")
    styles = assembler.parts["word/styles.xml"].decode("utf-8")
    styles = re.sub(r'<w:style\b[^>]*w:styleId="Title"[^>]*>.*?</w:style>', "", styles, flags=re.S)
    assembler.parts["word/styles.xml"] = styles.replace(
        "</w:styles>", study_styles() + "</w:styles>"
    ).encode("utf-8")

    images = {}
    for block in blocks:
        if block.kind != "figure":
            continue
        for source in block.attrs["src"].split():
            prefix, key = source.split(":", 1)
            directory = ROOT / (
                "reports/word/build/charts" if prefix == "chart" else "reports/figures"
            )
            images[key] = assembler.add_image(key, (directory / f"{key}.png").read_bytes())
    numbers = parse_numbers((ROOT / "reports/tables/numbers.tex").read_text(encoding="utf-8"))
    refs = content.parse_references(
        (ROOT / "reports/word/content/references.md").read_text(encoding="utf-8")
    )
    renderer = Renderer(numbers=numbers, tables={}, charts=images, references=refs)
    main = format_study_body(renderer.render(blocks))
    cover = (
        ooxml.paragraph(
            ooxml.latin_runs("khalij-watch", size=18), align="center", before=1400, after=240
        )
        + ooxml.paragraph(ooxml.text_runs(info["title"], bold=True), style="Title", align="center")
        + ooxml.paragraph(ooxml.text_runs(info["subtitle"], size=16), align="center", after=600)
        + ooxml.paragraph(
            ooxml.text_runs(info["description"]), style="StudyBody", align="right", after=600
        )
        + ooxml.paragraph(ooxml.text_runs(meta["author"], bold=True), align="center", after=240)
        + ooxml.paragraph(ooxml.text_runs(meta["date"]), align="center")
        + ooxml.page_break()
        + ooxml.paragraph(ooxml.text_runs(meta["contents_title"], bold=True, size=18), after=240)
        + ooxml.paragraph(ooxml.field('TOC \\h \\z \\t "StudyHeading2,1"'))
        + ooxml.page_break()
    )
    sect = section(header=HEADER_EMPTY, footer=FOOTER_LETTERS, fmt="decimal", start=1)
    sect = sect.replace("<w:docGrid", "<w:titlePg/><w:docGrid")
    assembler.write(destination, cover + main + sect, renderer.footnotes, renderer)
    return {
        "guide": name,
        "blocks": len(blocks),
        "cited_references": len(renderer.cited),
        "source_words": len((GUIDES / f"{name}.md").read_text(encoding="utf-8").split()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("phases", "technical"))
    args = parser.parse_args()
    for name in (args.only,) if args.only else ("phases", "technical"):
        target = GUIDES / "build" / f"{name}.docx"
        staging = target.with_stem(f"{name}-build-{uuid4().hex[:12]}")
        print(json.dumps(assemble_guide(name, staging)), flush=True)
        code = export_with_word(staging, target)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

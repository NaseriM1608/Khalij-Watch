"""Tests for the Word report builder (Persian text runs, LaTeX inputs, equations, numbering)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

from scripts.report_docx import latex_inputs, markup, omml, text
from scripts.report_docx.render import Renderer

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
NS = (
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"'
)


def parse(xml: str) -> ET.Element:
    return ET.fromstring(f"<root {NS}>{xml}</root>")


def runs_of(xml: str) -> list[tuple[str, bool]]:
    """(text, is_rtl) per run."""
    out = []
    for r in parse(xml).iter(W + "r"):
        t = "".join(x.text or "" for x in r.iter(W + "t"))
        rpr = r.find(W + "rPr")
        out.append((t, rpr is not None and rpr.find(W + "rtl") is not None))
    return out


# --- text -------------------------------------------------------------------


def test_latin_digits_become_persian_digits():
    assert text.fa_digits("F1 0.817 on 92 passes") == "F1 ۰.۸۱۷ on ۹۲ passes"


def test_persian_text_is_marked_right_to_left_and_latin_words_are_not():
    """Word lays out Persian only in runs marked rtl; Latin words go in their own runs."""
    parts = text.split_scripts("رمزگذار EfficientNet-B0 و Global Fishing Watch.")
    assert parts == [
        ("رمزگذار ", True),
        ("EfficientNet-B0", False),
        (" و ", True),
        ("Global Fishing Watch", False),
        (".", True),
    ]


def test_a_sentence_final_period_stays_with_the_persian_text():
    assert text.split_scripts("سامانه AIS.") == [("سامانه ", True), ("AIS", False), (".", True)]


def test_chapter_ordinals_are_persian_words():
    assert text.ordinal(1) == "اول"
    assert text.ordinal(5) == "پنجم"


# --- LaTeX inputs (tables and numbers made by make_reports.py) ---------------


def test_number_macros_become_persian_text():
    tex = (
        "% Generated\n"
        "\\newcommand{\\testFOneModel}{\\LRE{0٫817}}\n"
        "\\newcommand{\\testDiffCi}{\\LRE{+0٫002} تا \\LRE{+0٫016}}\n"
        "\\newcommand{\\darkVal}{\\LRE{4٫0}٪}\n"
    )
    numbers = latex_inputs.parse_numbers(tex)
    assert numbers == {
        "testFOneModel": "۰٫۸۱۷",
        "testDiffCi": "+۰٫۰۰۲ تا +۰٫۰۱۶",
        "darkVal": "۴٫۰٪",
    }


def test_a_booktabs_table_is_read_with_its_captions_and_label():
    tex = (
        "\\begin{table}[htbp]\n\\centering\n\\small\n"
        "\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}{lcc}\n\\toprule\n"
        "آشکارساز & \\lr{F1} & بازه ۹۵٪ \\\\\n\\midrule\n"
        "\\lr{CFAR} & \\LRE{0٫808} & \\LRE{+0٫002} تا \\LRE{+0٫016} \\\\\n"
        "\\textbf{مدل} & \\textbf{\\LRE{0٫817}} & — \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
        "\\caption[کوتاه]{شرح بلند.}\n\\label{tab:test92}\n\\end{table}\n"
    )
    table = latex_inputs.parse_table(tex)
    assert table.header == ["آشکارساز", "F1", "بازه ۹۵٪"]
    assert table.rows == [
        ["CFAR", "۰٫۸۰۸", "+۰٫۰۰۲ تا +۰٫۰۱۶"],
        ["**مدل**", "**۰٫۸۱۷**", "—"],
    ]
    assert (table.short, table.caption, table.label) == ("کوتاه", "شرح بلند.", "tab:test92")


# --- markup -----------------------------------------------------------------


def test_the_markup_gives_headings_paragraphs_lists_and_blocks():
    src = (
        "# مقدمه\n\n"
        "## بیان مسئله {#sec:problem}\n\n"
        "متن نخست\nادامه همان بند.\n\n"
        "- مورد یک\n- مورد دو\n\n"
        "1. گام یک\n2. گام دو\n\n"
        ":::figure id=map src=chart:map width=14 short=نقشه\n"
        "شرح شکل.\n"
        ":::\n"
    )
    blocks = markup.parse(src)
    kinds = [b.kind for b in blocks]
    assert kinds == ["chapter", "heading", "paragraph", "bullets", "numbered", "figure"]
    assert blocks[1].level == 2 and blocks[1].label == "sec:problem"
    assert blocks[2].text == "متن نخست ادامه همان بند."
    assert blocks[3].items == ["مورد یک", "مورد دو"]
    assert blocks[5].attrs == {"id": "map", "src": "chart:map", "width": "14", "short": "نقشه"}
    assert blocks[5].body == "شرح شکل."


def test_an_unnumbered_chapter_is_marked_with_a_star():
    blocks = markup.parse("#* نتیجه‌گیری و پیشنهادات\n")
    assert blocks[0].kind == "chapter" and blocks[0].numbered is False


def test_headings_in_unnumbered_conclusion_do_not_continue_chapter_numbers():
    r = make_renderer()
    xml = r.render(
        markup.parse("# نتایج\n\n## آزمون\n\n#* نتیجه‌گیری و پیشنهادات\n\n## محدودیت‌ها\n")
    )
    assert "۱-۱- آزمون" in body_text(xml)
    assert "۱-۲- محدودیت‌ها" not in body_text(xml)
    assert "محدودیت‌ها" in body_text(xml)


# --- equations --------------------------------------------------------------


def test_a_fraction_becomes_an_office_math_fraction():
    xml = omml.to_omml(r"F_1 = \frac{2PR}{P+R}")
    root = parse(xml)
    m = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
    assert root.find(f".//{m}f") is not None
    assert root.find(f".//{m}sSub") is not None
    num = "".join(t.text for t in root.find(f".//{m}num").iter(m + "t"))
    assert num == "2PR"


def test_cases_become_an_equation_array_inside_a_brace():
    xml = omml.to_omml(r"L = \begin{cases} a & y=1 \\ b & \text{otherwise} \end{cases}")
    m = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
    root = parse(xml)
    assert root.find(f".//{m}d") is not None
    assert len(root.findall(f".//{m}eqArr/{m}e")) == 2


def test_unknown_commands_fail_loudly():
    with pytest.raises(ValueError, match="unknown"):
        omml.to_omml(r"\frobnicate{x}")


# --- rendering: numbering, references, citations -----------------------------


def make_renderer() -> Renderer:
    return Renderer(
        numbers={"testFOneModel": "۰٫۸۱۷"},
        tables={},
        charts={},
        references={
            "torres2012": "R. Torres et al., GMES Sentinel-1 mission, 2012.",
            "crisp2004": "D. J. Crisp, The state-of-the-art in ship detection, 2004.",
        },
    )


def body_text(xml: str) -> str:
    return "".join(t.text or "" for t in parse(xml).iter(W + "t"))


def test_headings_get_chapter_first_numbers_and_chapters_get_ordinals():
    r = make_renderer()
    xml = r.render(markup.parse("# مقدمه\n\n## بیان مسئله\n\n### جزئیات\n\n# پیشینه\n\n## روش‌ها\n"))
    text_all = body_text(xml)
    assert "فصل اول: مقدمه" in text_all
    assert "۱-۱- بیان مسئله" in text_all
    assert "۱-۱-۱- جزئیات" in text_all
    assert "فصل دوم: پیشینه" in text_all
    assert "۲-۱- روش‌ها" in text_all


def test_citations_are_numbered_in_order_of_first_appearance():
    r = make_renderer()
    xml = r.render(markup.parse("# مقدمه\n\nنخست [@crisp2004] سپس [@torres2012; @crisp2004].\n"))
    assert "نخست [۱] سپس [۱، ۲]." in body_text(xml)
    assert r.cited == ["crisp2004", "torres2012"]


def test_an_unknown_citation_key_fails_loudly():
    r = make_renderer()
    with pytest.raises(KeyError, match="nosuch"):
        r.render(markup.parse("# مقدمه\n\nمتن [@nosuch].\n"))


def test_number_macros_and_cross_references_are_filled_in():
    r = make_renderer()
    src = (
        "# مقدمه\n\nچنان‌که در شکل [[fig:map]] و بخش [[sec:data]] آمده، F1 برابر {{testFOneModel}} بود.\n\n"
        "## داده‌ها {#sec:data}\n\n"
        ":::figure id=map src=chart:map width=10 short=نقشه\nشرح.\n:::\n"
    )
    r.charts = {"map": ("rIdTest", 1000, 500)}
    xml = r.render(markup.parse(src))
    assert "شکل ۱-۱ و بخش ۱-۱ آمده، F1 برابر ۰٫۸۱۷ بود." in body_text(xml)
    assert "شکل ۱-۱- نقشه" in body_text(xml)
    assert "شرح." in body_text(xml)


def test_persian_runs_are_rtl_and_latin_runs_are_not_in_rendered_paragraphs():
    r = make_renderer()
    xml = r.render(markup.parse("# مقدمه\n\nداده‌های Sentinel-1 در خلیج فارس.\n"))
    paragraph_runs = [rt for rt in runs_of(xml) if rt[0] in ("Sentinel-1", "داده‌های ")]
    assert ("داده‌های ", True) in paragraph_runs
    assert ("Sentinel-1", False) in paragraph_runs


def test_every_figure_and_table_needs_a_known_source():
    r = make_renderer()
    with pytest.raises(KeyError, match="nochart"):
        r.render(
            markup.parse(
                "# مقدمه\n\n:::figure id=x src=chart:nochart width=10 short=s\nشرح.\n:::\n"
            )
        )


def test_a_compact_table_stays_with_its_caption_and_releases_following_text():
    r = make_renderer()
    xml = r.render(
        markup.parse(
            "# روش\n\n:::table id=labels\nوضعیت‌های برچسب\n"
            "| وضعیت | معنی |\n| مثبت | مرکز شناور |\n| منفی | مرکز زمینه |\n:::\n"
            "\nمتن بعد از جدول.\n"
        )
    )
    root = parse(xml)
    rows = root.findall(f".//{W}tbl/{W}tr")
    assert len(rows) == 3
    for row in rows[:-1]:
        assert all(p.find(f"{W}pPr/{W}keepNext") is not None for p in row.iter(W + "p"))
    assert all(p.find(f"{W}pPr/{W}keepNext") is None for p in rows[-1].iter(W + "p"))
    caption = next(
        p
        for p in root.iter(W + "p")
        if "وضعیت‌های برچسب" in body_text(ET.tostring(p, encoding="unicode"))
    )
    assert caption.find(f"{W}pPr/{W}pStyle").get(W + "val") == "TableCaption"


def test_rendered_body_is_well_formed_xml():
    r = make_renderer()
    xml = r.render(markup.parse("# مقدمه\n\n**پررنگ** و عادی [@torres2012].\n\n- یک\n- دو\n"))
    parse(xml)  # raises on malformed XML
    assert re.search(r"<w:b/>", xml)


def test_assembled_document_declares_namespaces_for_inserted_pictures(tmp_path):
    from scripts.report_docx import ooxml
    from scripts.report_docx.document import Assembler

    template = Path(__file__).resolve().parents[1] / "reports/word/template.docx"
    assembler = Assembler(template)
    picture = ooxml.inline_image("rId8", 1000, 1000, 1, "logo")
    assembler.write(tmp_path / "report.docx", ooxml.paragraph(picture), [], make_renderer())
    with zipfile.ZipFile(tmp_path / "report.docx") as package:
        ET.fromstring(package.read("word/document.xml"))


def test_figure_index_uses_short_caption_without_shortening_visible_caption():
    r = make_renderer()
    r.charts = {"map": ("rIdTest", 1000, 500)}
    xml = r.render(
        markup.parse('# مقدمه\n\n:::figure id=map src=chart:map short="نقشه"\nشرح بلند شکل.\n:::\n')
    )
    captions = [
        p
        for p in parse(xml).iter(W + "p")
        if p.find(f"{W}pPr/{W}pStyle") is not None
        and p.find(f"{W}pPr/{W}pStyle").get(W + "val") == "FigureCaption"
    ]
    title = "".join(t.text or "" for p in captions for t in p.iter(W + "t"))
    assert "نقشه" in title
    assert "شرح بلند" not in title
    assert "شرح بلند شکل." in body_text(xml)


def test_template_fonts_with_multiple_spaces_are_replaced_in_report_styles(tmp_path):
    from scripts.report_docx.document import Assembler

    template = Path(__file__).resolve().parents[1] / "reports/word/template.docx"
    assembler = Assembler(template)
    assembler.write(tmp_path / "report.docx", "", [], make_renderer())
    with zipfile.ZipFile(tmp_path / "report.docx") as package:
        styles = package.read("word/styles.xml").decode()
    assert not re.search(r'w:cs="2\s+Lotus"', styles)


def test_contents_style_spacing_precedes_indentation_in_ooxml(tmp_path):
    from scripts.report_docx.document import Assembler

    template = Path(__file__).resolve().parents[1] / "reports/word/template.docx"
    assembler = Assembler(template)
    assembler.write(tmp_path / "report.docx", "", [], make_renderer())
    with zipfile.ZipFile(tmp_path / "report.docx") as package:
        styles = ET.fromstring(package.read("word/styles.xml"))
    for style in styles.findall(W + "style"):
        if style.get(W + "styleId") in ("TOC2", "TOC3", "TOC4"):
            children = [c.tag for c in style.find(W + "pPr")]
            assert children.index(W + "spacing") < children.index(W + "ind")

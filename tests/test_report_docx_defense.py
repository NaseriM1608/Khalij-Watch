"""Behavioral checks for the defense guides' compact, readable pagination."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from scripts.build_defense_guides import format_study_body
from scripts.report_docx import markup
from scripts.report_docx.render import Renderer

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def rendered(source: str) -> str:
    return Renderer(numbers={}, tables={}, charts={}, references={}).render(markup.parse(source))


def parsed(xml: str) -> ET.Element:
    return ET.fromstring(
        '<root xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        + xml
        + "</root>"
    )


def test_study_guide_text_starts_below_title_without_university_chapter_gap():
    body = format_study_body(rendered("# راهنمای مطالعه\n\n## مرحله نخست\n\nتوضیح مرحله.\n"))
    root = parsed(body)
    first = root.find(W + "p")
    assert first.find(f"{W}pPr/{W}pStyle").get(W + "val") == "StudyHeading1"
    assert int(first.find(f"{W}pPr/{W}spacing").get(W + "after")) <= 240
    assert root.findall(W + "p")[-1].find(f"{W}pPr/{W}pStyle").get(W + "val") == "StudyBody"


def test_long_parameter_table_can_continue_with_repeated_header():
    rows = "\n".join(f"| مورد {i} | توضیح |" for i in range(15))
    body = format_study_body(
        rendered(f"# روش\n\n:::table id=params\nپارامترها\n| نام | شرح |\n{rows}\n:::\n")
    )
    table = parsed(body).find(W + "tbl")
    assert table.find(f"{W}tr/{W}trPr/{W}tblHeader") is not None
    for row in table.findall(W + "tr")[1:]:
        assert all(p.find(f"{W}pPr/{W}keepNext") is None for p in row.iter(W + "p"))


def test_short_summary_table_stays_together():
    body = format_study_body(
        rendered("# روش\n\n:::table id=short\nخلاصه\n| نام | شرح |\n| الف | ب |\n| ج | د |\n:::\n")
    )
    table = parsed(body).find(W + "tbl")
    for row in table.findall(W + "tr")[:-1]:
        assert all(p.find(f"{W}pPr/{W}keepNext") is not None for p in row.iter(W + "p"))
